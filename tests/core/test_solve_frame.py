import pytest
import torch

from causallsso import apply_dual_reference, apply_primal_reference
from causallsso.ops import mathdx_available, mathdx_solve_frame128


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not mathdx_available(),
    reason="CUDA and built MathDx extension required",
)


def _rho(reference: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    return (actual.double() - reference.double()).square().mean().sqrt() / (
        reference.double().square().mean().sqrt() + 1e-8
    )


@pytest.mark.parametrize(
    ("dual_dtype", "ceiling"),
    [(torch.float32, 5e-5), (torch.bfloat16, 5e-3), (torch.float16, 5e-3)],
)
@pytest.mark.parametrize("edits", [1, 2])
def test_mathdx_solve_frame_matches_fp64(dual_dtype, ceiling, edits) -> None:
    torch.manual_seed(916)
    batch, rank = 3, 128
    nl = 0.01 * torch.tril(torch.randn(batch, rank, rank, device="cuda"), diagonal=-1)
    nu = 0.01 * torch.triu(torch.randn(batch, rank, rank, device="cuda"), diagonal=1)
    eye = torch.eye(rank, device="cuda")
    lower = (eye + nl).requires_grad_()
    upper = (eye + nu).requires_grad_()
    diagonal = torch.exp(0.05 * torch.randn(batch, rank, device="cuda")).requires_grad_()
    keys = torch.randn(batch, edits, rank, device="cuda", requires_grad=True)
    erase = torch.randn_like(keys, requires_grad=True)
    query = torch.randn(batch, rank, device="cuda", requires_grad=True)

    d, e, chi = mathdx_solve_frame128(
        lower, diagonal, upper, keys, erase, query, dual_dtype=dual_dtype
    )
    d_ref = apply_primal_reference(
        lower.double(), diagonal.double(), upper.double(), keys.double().transpose(-1, -2)
    ).transpose(-1, -2)
    dual_rhs = torch.cat((erase, query.unsqueeze(-2)), dim=-2).double().transpose(-1, -2)
    dual_ref = apply_dual_reference(
        lower.double(), diagonal.double(), upper.double(), dual_rhs
    ).transpose(-1, -2)
    assert _rho(d_ref, d) < 5e-5
    assert _rho(dual_ref[..., :edits, :], e) < ceiling
    assert _rho(dual_ref[..., edits, :], chi) < ceiling
    pairing = torch.einsum("bkr,bkr->bk", d, e)
    pairing_ref = torch.einsum("bkr,bkr->bk", keys, erase)
    pairing_ceiling = 5e-5 if dual_dtype == torch.float32 else 5e-3
    pairing_drift = (pairing - pairing_ref).abs() / (
        e.norm(dim=-1) * d.norm(dim=-1)
        + erase.norm(dim=-1) * keys.norm(dim=-1)
        + 1e-12
    )
    assert pairing_drift.max() < pairing_ceiling

    loss = d.square().mean() + e.square().mean() + chi.square().mean()
    loss.backward()
    for tensor in (lower, upper, diagonal, keys, erase, query):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
