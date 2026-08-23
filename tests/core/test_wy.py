from __future__ import annotations

import importlib.util

import pytest
import torch

from causallsso.ops.wy import wy_associative


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("fla") is None,
    reason="CUDA and flash-linear-attention are required",
)


def _rho(reference: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    reference = reference.double()
    actual = actual.double()
    return (actual - reference).square().mean().sqrt() / (
        reference.square().mean().sqrt() + 1e-8
    )


def _oracle(
    chi: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    z: torch.Tensor,
    log_decay: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state
    outputs = []
    for token in range(chi.shape[1]):
        state = torch.exp(log_decay[:, token])[..., None] * state
        for edit in range(d.shape[3]):
            prediction = torch.einsum(
                "bhrv,bhr->bhv", state, e[:, token, :, edit]
            )
            innovation = z[:, token, :, edit] - prediction
            state = state + torch.einsum(
                "bhr,bhv->bhrv", d[:, token, :, edit], innovation
            )
        outputs.append(
            torch.einsum("bhrv,bhr->bhv", state, chi[:, token])
        )
    return torch.stack(outputs, dim=1), state


def _runtime_inputs(edits: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(916 + edits)
    batch, length, heads, rank, value_dim = 1, 35, 2, 32, 24
    prefix = (batch, length, heads)

    def variable(value: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        return value.to(dtype).detach().requires_grad_(True)

    return (
        variable(torch.randn(*prefix, rank, device="cuda")),
        variable(0.04 * torch.randn(*prefix, edits, rank, device="cuda")),
        variable(0.04 * torch.randn(*prefix, edits, rank, device="cuda")),
        variable(0.08 * torch.randn(*prefix, edits, value_dim, device="cuda")),
        variable(-0.03 * torch.rand(*prefix, rank, device="cuda")),
        variable(
            0.03 * torch.randn(batch, heads, rank, value_dim, device="cuda"),
            torch.float32,
        ),
    )


@pytest.mark.parametrize("edits", [1, 2, 4])
def test_wy_forward_and_final_state_match_fp64_oracle(edits: int) -> None:
    runtime = _runtime_inputs(edits)
    output, final_state = wy_associative(
        *runtime[:5], initial_state=runtime[5], output_final_state=True
    )
    expected_output, expected_state = _oracle(
        *(value.detach().double() for value in runtime)
    )

    assert final_state is not None
    assert final_state.dtype == torch.float32
    assert _rho(expected_output, output) < 5e-3
    assert _rho(expected_state, final_state) < 5e-3


@pytest.mark.parametrize("edits", [1, 2, 4])
def test_wy_backward_matches_fp64_oracle(edits: int) -> None:
    runtime = _runtime_inputs(edits)
    reference = tuple(
        value.detach().double().requires_grad_(True) for value in runtime
    )
    torch.manual_seed(1200 + edits)
    upstream_output = torch.randn(
        runtime[0].shape[:3] + (runtime[3].shape[-1],), device="cuda"
    )
    upstream_state = torch.randn_like(runtime[5])

    output, final_state = wy_associative(
        *runtime[:5], initial_state=runtime[5], output_final_state=True
    )
    assert final_state is not None
    loss = (
        (output.float() * upstream_output).sum()
        + (final_state * upstream_state).sum()
    )
    actual_gradients = torch.autograd.grad(loss, runtime)

    expected_output, expected_state = _oracle(*reference)
    expected_loss = (
        (expected_output * upstream_output.double()).sum()
        + (expected_state * upstream_state.double()).sum()
    )
    expected_gradients = torch.autograd.grad(expected_loss, reference)

    for expected, actual in zip(expected_gradients, actual_gradients):
        assert torch.isfinite(actual).all()
        assert _rho(expected, actual) < 1e-2
