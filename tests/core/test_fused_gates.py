from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("fla")

from causallsso.ops.fused_gates import fused_native_solvedelta_gates


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _inputs(length: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260913 + length)
    batch, heads, rank = 2, 3, 128

    def leaf(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().requires_grad_(True)

    return (
        leaf(torch.randn(batch, length, heads, 1, rank, device="cuda", dtype=torch.bfloat16)),
        leaf(torch.randn(batch, length, heads, 1, 96, device="cuda", dtype=torch.bfloat16)),
        leaf(torch.randn(batch, length, heads, device="cuda", dtype=torch.bfloat16)),
        leaf(torch.randn(batch, length, heads, rank, device="cuda", dtype=torch.bfloat16)),
        leaf(torch.randn(heads, device="cuda")),
        leaf(torch.randn(heads, rank, device="cuda")),
        leaf(torch.randn(heads, device="cuda")),
        leaf(torch.randn(heads, rank, device="cuda")),
    )


def _reference(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    erase, write, geometry, associative, geometry_rate, associative_rate, geometry_bias, associative_bias = inputs
    heads = geometry.shape[-1]
    rank = associative.shape[-1]
    return (
        (2.0 * torch.sigmoid(erase.float())).to(torch.bfloat16),
        (2.0 * torch.sigmoid(write.float())).to(torch.bfloat16),
        -torch.exp(geometry_rate).view(1, 1, heads)
        * F.softplus(geometry.float() + geometry_bias.view(1, 1, heads)),
        -torch.exp(associative_rate).view(1, 1, heads, rank)
        * F.softplus(
            associative.float()
            + associative_bias.view(1, 1, heads, rank)
        ),
    )


@pytest.mark.parametrize("length", (1, 31, 33))
def test_fused_native_gates_match_declared_expression(length: int) -> None:
    inputs = _inputs(length)
    actual = fused_native_solvedelta_gates(*inputs)
    expected = _reference(inputs)
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[2], expected[2], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual[3], expected[3], rtol=2e-6, atol=2e-6)


def test_fused_native_gate_vjp_matches_declared_expression() -> None:
    actual_inputs = _inputs(33)
    reference_inputs = tuple(
        tensor.detach().requires_grad_(True) for tensor in actual_inputs
    )
    actual = fused_native_solvedelta_gates(*actual_inputs)
    expected = _reference(reference_inputs)
    torch.manual_seed(20260914)
    cotangents = tuple(torch.randn_like(tensor) for tensor in actual)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, cotangents)
    expected_gradients = torch.autograd.grad(
        expected, reference_inputs, cotangents
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients
    ):
        difference = (
            actual_gradient.float() - expected_gradient.float()
        ).norm()
        scale = expected_gradient.float().norm().clamp_min(1.0e-8)
        assert (difference / scale).item() <= 1.0e-4


def test_fused_native_gate_backward_does_not_materialize_unused_routes() -> None:
    inputs = _inputs(3)
    outputs = fused_native_solvedelta_gates(*inputs)
    gradients = torch.autograd.grad(
        outputs[2].sum(), inputs, allow_unused=True
    )
    assert gradients[2] is not None
    assert gradients[4] is not None
    assert gradients[6] is not None
    assert all(
        gradients[index] is None for index in (0, 1, 3, 5, 7)
    )
