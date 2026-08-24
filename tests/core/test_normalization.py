from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from causallsso.ops.normalization import normalize_solvedelta_inputs


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _runtime(length: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260831 + length)
    prefix = (2, length, 3)
    return (
        torch.randn(*prefix, 128, device="cuda", dtype=torch.bfloat16),
        torch.randn(*prefix, 128, device="cuda", dtype=torch.bfloat16),
        torch.randn(*prefix, 1, 128, device="cuda", dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("length", (1, 31, 33))
def test_fused_normalization_matches_declared_expression(length: int) -> None:
    inputs = _runtime(length)
    actual = normalize_solvedelta_inputs(*inputs)
    expected = tuple(
        F.normalize(tensor.float(), p=2, dim=-1).to(torch.float16)
        for tensor in inputs
    )
    for fused, reference in zip(actual, expected):
        assert fused.dtype == torch.float16
        difference = (fused.float() - reference.float()).norm()
        scale = reference.float().norm().clamp_min(1.0e-8)
        assert (difference / scale).item() <= 1.0e-4


def test_fused_normalization_vjp_matches_declared_expression() -> None:
    runtime = _runtime(33)
    actual_inputs = tuple(
        tensor.detach().requires_grad_(True) for tensor in runtime
    )
    reference_inputs = tuple(
        tensor.detach().requires_grad_(True) for tensor in runtime
    )
    actual = normalize_solvedelta_inputs(*actual_inputs)
    reference = tuple(
        F.normalize(tensor.float(), p=2, dim=-1).to(torch.float16)
        for tensor in reference_inputs
    )
    torch.manual_seed(20260901)
    cotangents = tuple(torch.randn_like(tensor) for tensor in actual)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, cotangents)
    reference_gradients = torch.autograd.grad(
        reference, reference_inputs, cotangents
    )
    for fused, expected in zip(actual_gradients, reference_gradients):
        difference = (fused.float() - expected.float()).norm()
        scale = expected.float().norm().clamp_min(1.0e-8)
        assert (difference / scale).item() <= 3.0e-3


def test_fused_normalization_zero_vector_uses_epsilon_branch() -> None:
    geometry, query, key = _runtime(1)
    geometry.zero_()
    query.zero_()
    key.zero_()
    inputs = tuple(tensor.requires_grad_(True) for tensor in (geometry, query, key))
    outputs = normalize_solvedelta_inputs(*inputs)
    assert all(torch.count_nonzero(output).item() == 0 for output in outputs)
    gradients = torch.autograd.grad(
        outputs,
        inputs,
        tuple(torch.ones_like(output) for output in outputs),
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
