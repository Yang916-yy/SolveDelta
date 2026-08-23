import pytest
import torch
import torch.nn.functional as F

from causallsso import (
    apply_dual_reference,
    apply_primal_reference,
    bounded_ldu_reference,
)
from causallsso.ops import mathdx_available, panel_frame128


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not mathdx_available(),
    reason="CUDA and built native extension required",
)


def _rho(reference: torch.Tensor, actual: torch.Tensor) -> float:
    error = (actual.double() - reference.double()).square().mean().sqrt()
    scale = reference.double().square().mean().sqrt()
    return float(error / (scale + 1e-8))


def _inputs(length: int, *, heads: int = 2) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260824 + length)
    batch, rank = 1, 128
    chunks = (length + 31) // 32
    u = F.normalize(
        torch.randn(batch, length, heads, rank, device="cuda"), dim=-1
    )
    h = 0.2 * torch.randn_like(u)
    log_decay = -0.05 * torch.rand(batch, length, heads, device="cuda")
    keys = F.normalize(
        torch.randn(batch, length, heads, 1, rank, device="cuda"), dim=-1
    )
    erase = 2.0 * torch.rand_like(keys)
    query = F.normalize(torch.randn_like(u), dim=-1)
    boundary_m = 2.0 + torch.rand(batch, heads, chunks, device="cuda")
    boundary_j = 0.03 * torch.randn(
        batch, heads, chunks, rank, rank, device="cuda"
    )
    boundary_d = 0.03 * torch.randn_like(boundary_j)
    strength = torch.sigmoid(torch.randn(heads, device="cuda"))
    return (
        boundary_m,
        boundary_j,
        boundary_d,
        u,
        h,
        log_decay,
        keys,
        erase,
        query,
        strength,
    )


def _oracle(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    (
        boundary_m,
        boundary_j,
        boundary_d,
        u,
        h,
        log_decay,
        keys,
        erase,
        query,
        strength,
    ) = inputs
    _, length, _, _ = u.shape
    outputs_d, outputs_e, outputs_chi = [], [], []
    mass = moment_j = moment_d = None
    for token in range(length):
        chunk = token // 32
        if token % 32 == 0:
            mass = boundary_m[:, :, chunk]
            moment_j = boundary_j[:, :, chunk]
            moment_d = boundary_d[:, :, chunk]
        decay = torch.exp(log_decay[:, token])
        token_u = u[:, token]
        token_h = h[:, token]
        mass = decay * mass + 1.0
        moment_j = (
            decay[..., None, None] * moment_j
            + token_u[..., :, None] * token_u[..., None, :]
        )
        moment_d = (
            decay[..., None, None] * moment_d
            + token_u[..., :, None] * token_h[..., None, :]
        )
        lower, diagonal, upper, _ = bounded_ldu_reference(
            moment_j / mass[..., None, None],
            moment_d / mass[..., None, None],
            strength,
        )
        key = keys[:, token, :, 0]
        dual_rhs = erase[:, token, :, 0] * key
        outputs_d.append(
            apply_primal_reference(lower, diagonal, upper, key).unsqueeze(-2)
        )
        outputs_e.append(
            apply_dual_reference(lower, diagonal, upper, dual_rhs).unsqueeze(-2)
        )
        outputs_chi.append(
            apply_dual_reference(lower, diagonal, upper, query[:, token])
        )
    return (
        torch.stack(outputs_d, dim=1),
        torch.stack(outputs_e, dim=1),
        torch.stack(outputs_chi, dim=1),
    )


def _vjp(
    inputs: tuple[torch.Tensor, ...],
    output_grads: tuple[torch.Tensor, ...],
    *,
    fp64: bool,
) -> tuple[torch.Tensor, ...]:
    dtype = torch.float64 if fp64 else torch.float32
    leaves = tuple(
        value.detach().to(dtype).requires_grad_(True) for value in inputs
    )
    outputs = _oracle(*leaves) if fp64 else panel_frame128(*leaves)
    return torch.autograd.grad(
        outputs,
        leaves,
        tuple(value.to(dtype) for value in output_grads),
    )


def _assert_vjp_contract(
    expected: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...],
) -> None:
    for reference_grad, native_grad in zip(expected, actual):
        assert torch.isfinite(native_grad).all()
        max_error = (native_grad.double() - reference_grad).abs().max()
        assert max_error <= 1e-6 or _rho(reference_grad, native_grad) < 1e-3


def _cancellation_inputs(kind: str) -> tuple[torch.Tensor, ...]:
    inputs = list(_inputs(32, heads=1))
    torch.manual_seed(1947 + (kind == "D"))
    left = F.normalize(
        torch.randn(128, device="cuda", dtype=torch.float64), dim=0
    )
    right = F.normalize(
        torch.randn(128, device="cuda", dtype=torch.float64), dim=0
    )
    inputs[0].fill_(1.0)
    inputs[1].zero_()
    inputs[2].zero_()
    inputs[3].zero_()
    inputs[4].zero_()
    inputs[5].zero_()
    if kind == "J":
        inputs[1][0, 0, 0] = (-4096.0 * torch.outer(left, left)).float()
        inputs[3][0, 0, 0] = (64.0 * left).float()
        inputs[3][0, 0, 0, 0] += 1e-4
    else:
        inputs[2][0, 0, 0] = (4096.0 * torch.outer(left, right)).float()
        inputs[3][0, 0, 0] = left.float()
        inputs[4][0, 0, 0] = (-4096.0 + 0.01) * right.float()
    inputs[3][0, 1:, 0] = 0.05 * torch.randn(
        31, 128, device="cuda"
    )
    inputs[4][0, 1:, 0] = 0.05 * torch.randn(
        31, 128, device="cuda"
    )
    inputs[-1].fill_(0.7)
    return tuple(inputs)


@pytest.mark.parametrize("length", [7, 32, 33])
def test_panel_frame_forward_matches_fp64(length: int) -> None:
    inputs = _inputs(length)
    expected = _oracle(*(value.double() for value in inputs))
    actual = panel_frame128(*inputs)
    for reference, value in zip(expected, actual):
        assert torch.isfinite(value).all()
        assert _rho(reference, value) < 5e-4


@pytest.mark.parametrize("length", [7, 33])
def test_panel_frame_backward_matches_fp64(length: int) -> None:
    inputs = _inputs(length, heads=1)
    torch.manual_seed(303200 + length)
    output_grads = (
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[8]),
    )
    expected = _vjp(inputs, output_grads, fp64=True)
    actual = _vjp(inputs, output_grads, fp64=False)
    _assert_vjp_contract(expected, actual)


def test_panel_frame_identity_geometry_is_exact() -> None:
    inputs = list(_inputs(33))
    inputs[-1] = torch.zeros_like(inputs[-1])
    d, e, chi = panel_frame128(*inputs)
    torch.testing.assert_close(d, inputs[6], rtol=0.0, atol=0.0)
    torch.testing.assert_close(e, inputs[7] * inputs[6], rtol=0.0, atol=2e-7)
    torch.testing.assert_close(chi, inputs[8], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("log_decay", [-110.0, -1000.0])
def test_panel_frame_underflow_is_finite_and_repeatable(log_decay: float) -> None:
    inputs = list(_inputs(33))
    inputs[5] = torch.full_like(inputs[5], log_decay)
    first = panel_frame128(*inputs)
    second = panel_frame128(*inputs)
    for left, right in zip(first, second):
        assert torch.isfinite(left).all()
        assert torch.equal(left, right)


def test_panel_frame_identity_geometry_backward_contract() -> None:
    inputs = list(_inputs(33, heads=1))
    inputs[-1] = torch.zeros_like(inputs[-1])
    torch.manual_seed(404300)
    output_grads = (
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[8]),
    )
    expected = _vjp(tuple(inputs), output_grads, fp64=True)
    actual = _vjp(tuple(inputs), output_grads, fp64=False)
    _assert_vjp_contract(expected, actual)
    for gradient in actual[:6]:
        assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("log_decay", [-110.0, -1000.0])
def test_panel_frame_underflow_backward_contract(log_decay: float) -> None:
    inputs = list(_inputs(33, heads=1))
    inputs[5] = torch.full_like(inputs[5], log_decay)
    torch.manual_seed(505300 + int(-log_decay))
    output_grads = (
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[8]),
    )
    expected = _vjp(tuple(inputs), output_grads, fp64=True)
    actual = _vjp(tuple(inputs), output_grads, fp64=False)
    _assert_vjp_contract(expected, actual)


@pytest.mark.parametrize("kind", ["J", "D"])
def test_panel_frame_cancellation_backward_contract(kind: str) -> None:
    inputs = _cancellation_inputs(kind)
    torch.manual_seed(606300 + (kind == "D"))
    output_grads = (
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[8]),
    )
    expected = _vjp(inputs, output_grads, fp64=True)
    actual = _vjp(inputs, output_grads, fp64=False)
    _assert_vjp_contract(expected, actual)


def test_panel_frame_backward_is_repeatable() -> None:
    inputs = _inputs(33, heads=1)
    torch.manual_seed(707300)
    output_grads = (
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[8]),
    )
    first = _vjp(inputs, output_grads, fp64=False)
    second = _vjp(inputs, output_grads, fp64=False)
    for left, right in zip(first, second):
        assert torch.equal(left, right)
