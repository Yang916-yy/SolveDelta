import pytest
import torch

from causallsso.oracle import mathdx_available, mathdx_trsm128


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not mathdx_available(),
    reason="built MathDx CUDA extension required",
)


@pytest.mark.parametrize("upper", [False, True])
def test_mathdx_trsm128_forward_and_backward(upper: bool) -> None:
    torch.manual_seed(916)
    batch, rank, rhs_count = 8, 128, 2
    raw = 0.01 * torch.randn(batch, rank, rank, device="cuda", dtype=torch.float32)
    factor = (
        torch.triu(raw, diagonal=1) if upper else torch.tril(raw, diagonal=-1)
    ) + torch.eye(rank, device="cuda")
    factor.requires_grad_(True)
    rhs = torch.randn(batch, rank, rhs_count, device="cuda", dtype=torch.float32, requires_grad=True)
    native = mathdx_trsm128(factor, rhs, upper=upper)

    factor64 = factor.detach().double().requires_grad_()
    rhs64 = rhs.detach().double().requires_grad_()
    reference = torch.linalg.solve_triangular(
        factor64, rhs64, upper=upper, unitriangular=True
    )
    relative = (native.double() - reference).square().mean().sqrt() / (
        reference.square().mean().sqrt() + 1e-8
    )
    assert relative < 5e-5
    residual = factor64 @ native.double() - rhs64
    eta = torch.linalg.norm(residual) / (
        torch.linalg.matrix_norm(factor64, 2).max() * torch.linalg.norm(native.double())
        + torch.linalg.norm(rhs64)
        + 1e-12
    )
    assert eta < 2e-5

    upstream = torch.randn_like(native)
    (native * upstream).sum().backward()
    (reference * upstream.double()).sum().backward()
    for actual, expected in ((rhs.grad, rhs64.grad), (factor.grad, factor64.grad)):
        rho = (actual.double() - expected).square().mean().sqrt() / (
            expected.square().mean().sqrt() + 1e-8
        )
        assert rho < 2e-4
