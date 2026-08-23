import pytest
import torch

from causallsso import bounded_ldu_reference
from causallsso.ops import bounded_ldu_vjp128


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _errors(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    reference = reference.double()
    actual = actual.double()
    absolute = (actual - reference).abs().max().item()
    relative = (
        (actual - reference).square().mean().sqrt()
        / (reference.square().mean().sqrt() + 1e-8)
    ).item()
    return absolute, relative


@pytest.mark.parametrize("coordinate_scale", [0.02, 2.0])
def test_bounded_ldu_vjp128_matches_reference_autograd(
    coordinate_scale: float,
) -> None:
    torch.manual_seed(20260822)
    batch, heads, rank = 2, 3, 128
    H = coordinate_scale * torch.randn(
        batch, heads, rank, rank, device="cuda", dtype=torch.float32
    )
    R = coordinate_scale * torch.randn_like(H)
    strength = torch.sigmoid(torch.randn(heads, device="cuda", dtype=torch.float32))
    grad_lower = torch.randn_like(H)
    grad_diagonal = torch.randn(
        batch, heads, rank, device="cuda", dtype=torch.float32
    )
    grad_upper = torch.randn_like(H)
    grad_omega = torch.randn_like(H)

    strength_per_system = strength.broadcast_to(batch, heads).clone()
    oracle_inputs = tuple(
        tensor.detach().double().requires_grad_(True)
        for tensor in (H, R, strength_per_system)
    )
    factors = bounded_ldu_reference(*oracle_inputs)
    output_grads = (
        grad_lower.double(),
        grad_diagonal.double(),
        grad_upper.double(),
        grad_omega.double(),
    )
    expected = torch.autograd.grad(factors, oracle_inputs, output_grads)
    actual = bounded_ldu_vjp128(
        H,
        R,
        strength,
        grad_lower,
        grad_diagonal,
        grad_upper,
        grad_omega,
    )

    for expected_grad, actual_grad in zip(expected, actual):
        assert torch.isfinite(actual_grad).all()
        absolute, relative = _errors(expected_grad, actual_grad)
        assert absolute <= 1e-6 or relative < 1e-3


def test_bounded_ldu_vjp128_zero_strength_has_correct_partial() -> None:
    torch.manual_seed(20260823)
    systems, rank = 2, 128
    H = torch.randn(systems, rank, rank, device="cuda", dtype=torch.float32)
    R = torch.randn_like(H)
    strength = torch.zeros(systems, device="cuda", dtype=torch.float32)
    grad_lower = torch.randn_like(H)
    grad_diagonal = torch.randn(systems, rank, device="cuda", dtype=torch.float32)
    grad_upper = torch.randn_like(H)
    grad_omega = torch.randn_like(H)

    oracle_inputs = tuple(
        tensor.detach().double().requires_grad_(True)
        for tensor in (H, R, strength)
    )
    factors = bounded_ldu_reference(*oracle_inputs)
    expected = torch.autograd.grad(
        factors,
        oracle_inputs,
        (
            grad_lower.double(),
            grad_diagonal.double(),
            grad_upper.double(),
            grad_omega.double(),
        ),
    )
    actual = bounded_ldu_vjp128(
        H,
        R,
        strength,
        grad_lower,
        grad_diagonal,
        grad_upper,
        grad_omega,
    )

    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], torch.zeros_like(actual[1]), rtol=0, atol=0)
    absolute, relative = _errors(expected[2], actual[2])
    assert absolute <= 1e-6 or relative < 1e-3


@pytest.mark.parametrize("active_cotangent", ["lower", "diagonal", "upper", "omega"])
def test_bounded_ldu_vjp128_independent_cotangents(
    active_cotangent: str,
) -> None:
    torch.manual_seed(20260824)
    rank = 128
    H = 0.03 * torch.randn(1, rank, rank, device="cuda", dtype=torch.float32)
    R = 0.03 * torch.randn_like(H)
    strength = torch.tensor([0.4], device="cuda", dtype=torch.float32)
    cotangents = {
        "lower": torch.zeros_like(H),
        "diagonal": torch.zeros(1, rank, device="cuda", dtype=torch.float32),
        "upper": torch.zeros_like(H),
        "omega": torch.zeros_like(H),
    }
    cotangents[active_cotangent] = torch.randn_like(cotangents[active_cotangent])

    oracle_inputs = tuple(
        tensor.detach().double().requires_grad_(True)
        for tensor in (H, R, strength)
    )
    factors = bounded_ldu_reference(*oracle_inputs)
    output_grads = tuple(
        cotangents[name].double()
        for name in ("lower", "diagonal", "upper", "omega")
    )
    expected = torch.autograd.grad(factors, oracle_inputs, output_grads)
    actual = bounded_ldu_vjp128(
        H,
        R,
        strength,
        cotangents["lower"],
        cotangents["diagonal"],
        cotangents["upper"],
        cotangents["omega"],
    )

    repeated = bounded_ldu_vjp128(
        H,
        R,
        strength,
        cotangents["lower"],
        cotangents["diagonal"],
        cotangents["upper"],
        cotangents["omega"],
    )
    for expected_grad, actual_grad, repeated_grad in zip(expected, actual, repeated):
        absolute, relative = _errors(expected_grad, actual_grad)
        assert absolute <= 1e-6 or relative < 1e-3
        torch.testing.assert_close(actual_grad, repeated_grad, rtol=0, atol=0)
