from __future__ import annotations

import pytest
import torch
import triton

from causallsso.ops.paired_wy import (
    paired_wy_backward,
    paired_wy_forward,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


def _rho(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.double() - expected.double()
    return (
        difference.square().mean().sqrt()
        / (expected.double().square().mean().sqrt() + 1e-8)
    ).item()


def _fixtures(length: int = 33, value_dim: int = 47):
    torch.manual_seed(20260824 + length + value_dim)
    batch, heads, rank, chunk = 1, 2, 128, 32
    chunks = triton.cdiv(length, chunk)
    panels = batch * heads * chunks
    lower = 0.01 * torch.randn(
        panels, chunk, chunk, device="cuda", dtype=torch.float32
    )
    system = torch.eye(chunk, device="cuda").expand(panels, -1, -1).clone()
    system += torch.tril(lower, diagonal=-1)
    valid_last = length - (chunks - 1) * chunk
    if valid_last < chunk:
        tail = system[-1]
        tail[valid_last:] = 0.0
        tail[:, valid_last:] = 0.0
        tail.diagonal()[valid_last:] = 1.0
    system = system.reshape(batch, heads, chunks, chunk, chunk)
    erase_dual = torch.randn(
        batch, length, heads, rank, device="cuda", dtype=torch.bfloat16
    )
    inclusive_decay = -0.03 * torch.rand(
        batch, length, heads, rank, device="cuda", dtype=torch.float32
    ).cumsum(dim=1)
    write = torch.randn(
        batch,
        length,
        heads,
        1,
        value_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(write)
    return system, erase_dual, inclusive_decay, write, value


def _panel_solve(
    system: torch.Tensor,
    rhs: torch.Tensor,
    *,
    upper: bool = False,
) -> torch.Tensor:
    batch, length, heads, width = rhs.shape
    chunks = triton.cdiv(length, 32)
    output = torch.empty_like(rhs, dtype=torch.float64)
    for batch_index in range(batch):
        for head in range(heads):
            for chunk in range(chunks):
                start = chunk * 32
                stop = min(start + 32, length)
                count = stop - start
                matrix = system[batch_index, head, chunk, :count, :count]
                output[batch_index, start:stop, head] = (
                    torch.linalg.solve_triangular(
                        matrix.double(),
                        rhs[batch_index, start:stop, head].double(),
                        upper=upper,
                        unitriangular=True,
                    )
                )
    return output


def test_paired_wy_c32_forward_and_matrix_reverse() -> None:
    system, erase_dual, inclusive_decay, write, value = _fixtures()
    actual = paired_wy_forward(
        system, erase_dual, inclusive_decay, write, value
    )
    edit_rhs = erase_dual.float() * inclusive_decay.exp()
    value_rhs = (write.float() * value.float()).squeeze(-2)
    expected_edit = _panel_solve(system, edit_rhs)
    expected_value = _panel_solve(system, value_rhs)
    assert _rho(actual.edit, expected_edit) <= 5e-3
    assert _rho(actual.value, expected_value) <= 5e-3

    torch.manual_seed(20260825)
    grad_edit = torch.randn_like(actual.edit, dtype=torch.float32)
    grad_value = torch.randn_like(actual.value, dtype=torch.float32)
    adjoint = paired_wy_backward(
        system,
        actual.edit,
        actual.value,
        write,
        value,
        grad_edit,
        grad_value,
    )
    expected_edit_bar = _panel_solve(
        system.transpose(-1, -2),
        grad_edit,
        upper=True,
    )
    expected_value_bar = _panel_solve(
        system.transpose(-1, -2),
        grad_value,
        upper=True,
    )
    assert _rho(adjoint.edit_rhs, expected_edit_bar) <= 5e-3
    expected_write = (
        expected_value_bar.float().unsqueeze(-2) * value.float()
    ).to(torch.bfloat16)
    expected_value_leaf = (
        expected_value_bar.float().unsqueeze(-2) * write.float()
    ).to(torch.bfloat16)
    assert _rho(adjoint.write, expected_write) <= 5e-3
    assert _rho(adjoint.value, expected_value_leaf) <= 5e-3

    batch, length, heads, _ = actual.edit.shape
    chunks = triton.cdiv(length, 32)
    expected_system = torch.zeros_like(system)
    for batch_index in range(batch):
        for head in range(heads):
            for chunk in range(chunks):
                start = chunk * 32
                stop = min(start + 32, length)
                count = stop - start
                rhs_bar = torch.cat(
                    (
                        expected_edit_bar[batch_index, start:stop, head],
                        expected_value_bar[batch_index, start:stop, head],
                    ),
                    dim=-1,
                )
                solution = torch.cat(
                    (
                        actual.edit[batch_index, start:stop, head].double(),
                        actual.value[batch_index, start:stop, head].double(),
                    ),
                    dim=-1,
                )
                expected_system[batch_index, head, chunk, :count, :count] = (
                    torch.tril(-(rhs_bar @ solution.T), diagonal=-1)
                ).float()
    assert _rho(adjoint.system, expected_system) <= 5e-3


@pytest.mark.parametrize("length", (1, 16, 17, 31, 32, 33))
def test_paired_wy_irregular_chunks_are_finite(length: int) -> None:
    fixtures = _fixtures(length=length, value_dim=13)
    output = paired_wy_forward(*fixtures)
    assert torch.isfinite(output.edit).all()
    assert torch.isfinite(output.value).all()
