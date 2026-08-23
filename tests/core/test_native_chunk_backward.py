from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from causallsso.ops.chunk_frame import chunk_frame
from causallsso.ops.native_chunk import native_chunk_frame


_INPUT_NAMES = (
    "u",
    "h",
    "geometry_log_decay",
    "key",
    "erase",
    "query",
    "geometry_strength",
    "boundary_m",
    "boundary_J",
    "boundary_D",
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)


def _inputs(
    *,
    length: int,
    strength: float = 0.7,
    seed: int = 20260824,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed + length)
    batch, heads, rank = 1, 1, 128
    chunks = (length + 31) // 32
    u = F.normalize(
        torch.randn(batch, length, heads, rank, device="cuda"), dim=-1
    )
    h = 0.2 * torch.randn_like(u)
    key = F.normalize(
        torch.randn(batch, length, heads, 1, rank, device="cuda"), dim=-1
    )
    raw_j = 0.04 * torch.randn(
        batch, heads, chunks, rank, rank, device="cuda"
    )
    return (
        u,
        h,
        -0.08 * torch.rand(batch, length, heads, device="cuda"),
        key,
        2.0 * torch.rand_like(key),
        F.normalize(torch.randn_like(u), dim=-1),
        torch.full((heads,), strength, device="cuda"),
        0.5 + torch.rand(batch, heads, chunks, device="cuda"),
        raw_j @ raw_j.transpose(-1, -2),
        0.04 * torch.randn(
            batch, heads, chunks, rank, rank, device="cuda"
        ),
    )


def _cancellation_inputs(kind: str) -> tuple[torch.Tensor, ...]:
    inputs = list(_inputs(length=32, seed=1947 + (kind == "D")))
    torch.manual_seed(1947 + (kind == "D"))
    left = F.normalize(
        torch.randn(128, device="cuda", dtype=torch.float64), dim=0
    )
    right = F.normalize(
        torch.randn(128, device="cuda", dtype=torch.float64), dim=0
    )
    inputs[7].fill_(1.0)
    inputs[8].zero_()
    inputs[9].zero_()
    inputs[2].zero_()
    if kind == "J":
        inputs[8][0, 0, 0] = (-4096.0 * torch.outer(left, left)).float()
        inputs[0][0, 0, 0] = (64.0 * left).float()
        inputs[0][0, 0, 0, 0] += 1e-4
    else:
        inputs[9][0, 0, 0] = (4096.0 * torch.outer(left, right)).float()
        inputs[0][0, 0, 0] = left.float()
        inputs[1][0, 0, 0] = ((-4096.0 + 0.01) * right).float()
    inputs[2][:, 3] = -110.0
    inputs[2][:, 7] = -1000.0
    return tuple(inputs)


def _zero_mass_boundary_inputs() -> tuple[torch.Tensor, ...]:
    inputs = list(_inputs(length=5, seed=2718))
    inputs[7].zero_()
    assert torch.count_nonzero(inputs[8]) > 0
    assert torch.count_nonzero(inputs[9]) > 0
    return tuple(inputs)


def _metrics(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> tuple[float, float]:
    expected64 = expected.double()
    actual64 = actual.double()
    difference = actual64 - expected64
    rho = difference.square().mean().sqrt() / (
        expected64.square().mean().sqrt() + 1e-8
    )
    return rho.item(), difference.abs().max().item()


def _assert_gradient(
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    assert torch.isfinite(actual).all(), name
    rho, absolute = _metrics(expected, actual)
    assert absolute <= 1e-6 or rho <= 1e-3, (
        f"{name}: rho={rho:.6e}, a_inf={absolute:.6e}"
    )


def _vjp(
    implementation: Callable[..., tuple[torch.Tensor, ...]],
    master_inputs: tuple[torch.Tensor, ...],
    output_cotangents: tuple[torch.Tensor, ...],
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, ...]:
    variables = tuple(
        tensor.detach().to(dtype).requires_grad_(True) for tensor in master_inputs
    )
    outputs = implementation(*variables)
    return torch.autograd.grad(
        outputs,
        variables,
        tuple(cotangent.to(dtype) for cotangent in output_cotangents),
        allow_unused=True,
    )


def _native(
    *inputs: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return native_chunk_frame(*inputs)


def _reference(
    *inputs: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return chunk_frame(*inputs, chunk_size=32)


def _cotangents(
    outputs: tuple[torch.Tensor, ...],
    *,
    route: str,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    values = [torch.zeros_like(output) for output in outputs]
    if route == "all":
        values = [torch.randn_like(output) for output in outputs]
    else:
        values[("d", "e", "chi").index(route)] = torch.randn_like(
            outputs[("d", "e", "chi").index(route)]
        )
    return tuple(values)


@pytest.mark.parametrize("route", ("d", "e", "chi", "all"))
def test_native_chunk_each_action_vjp_matches_fp64(
    route: str,
) -> None:
    inputs = _inputs(length=1, seed=3100)
    with torch.no_grad():
        outputs = _reference(
            *(tensor.double() for tensor in inputs),
        )
        cotangents = _cotangents(outputs, route=route, seed=3300 + len(route))
    expected = _vjp(
        _reference,
        inputs,
        cotangents,
        dtype=torch.float64,
    )
    actual = _vjp(
        _native,
        inputs,
        cotangents,
        dtype=torch.float32,
    )
    for name, expected_gradient, actual_gradient in zip(
        _INPUT_NAMES, expected, actual
    ):
        assert expected_gradient is not None, name
        assert actual_gradient is not None, name
        _assert_gradient(name, expected_gradient, actual_gradient)


@pytest.mark.parametrize(
    "case",
    ("tail", "identity", "zero_mass", "J", "D"),
)
def test_native_chunk_complete_vjp_envelope(
    case: str,
) -> None:
    if case == "tail":
        inputs = _inputs(length=33, seed=4100)
    elif case == "identity":
        inputs = _inputs(length=31, strength=0.0, seed=4200)
    elif case == "zero_mass":
        inputs = _zero_mass_boundary_inputs()
    else:
        inputs = _cancellation_inputs(case)

    with torch.no_grad():
        outputs = _reference(
            *(tensor.double() for tensor in inputs),
        )
        cotangents = _cotangents(outputs, route="all", seed=4300 + len(case))
    expected = _vjp(
        _reference,
        inputs,
        cotangents,
        dtype=torch.float64,
    )
    actual = _vjp(
        _native,
        inputs,
        cotangents,
        dtype=torch.float32,
    )
    for name, expected_gradient, actual_gradient in zip(
        _INPUT_NAMES, expected, actual
    ):
        assert expected_gradient is not None, name
        assert actual_gradient is not None, name
        _assert_gradient(f"{case}.{name}", expected_gradient, actual_gradient)


def test_native_chunk_backward_is_bitwise_repeatable() -> None:
    inputs = _inputs(length=33, seed=5100)
    with torch.no_grad():
        outputs = _reference(
            *(tensor.double() for tensor in inputs)
        )
        cotangents = _cotangents(outputs, route="all", seed=5200)
    first = _vjp(
        _native,
        inputs,
        cotangents,
        dtype=torch.float32,
    )
    second = _vjp(
        _native,
        inputs,
        cotangents,
        dtype=torch.float32,
    )
    for name, left, right in zip(_INPUT_NAMES, first, second):
        assert left is not None, name
        assert right is not None, name
        assert torch.equal(left, right), name
