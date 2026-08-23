import pytest
import torch
import torch.nn.functional as F

from causallsso import SolveDeltaState, solvedelta_reference
from causallsso.ops import triton_geometry_chunk_scan


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _rho(reference: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    reference = reference.double()
    actual = actual.double()
    return (actual - reference).square().mean().sqrt() / (
        reference.square().mean().sqrt() + 1e-8
    )


def _geometry_scan_recompute(
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    initial_m: torch.Tensor,
    initial_J: torch.Tensor,
    initial_D: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, ...]:
    current_m, current_J, current_D = initial_m, initial_J, initial_D
    boundary_m, boundary_J, boundary_D = [], [], []
    for start in range(0, u.shape[1], chunk_size):
        end = min(start + chunk_size, u.shape[1])
        boundary_m.append(current_m)
        boundary_J.append(current_J)
        boundary_D.append(current_D)
        logs = geometry_log_decay[:, start:end].permute(0, 2, 1)
        suffix = torch.flip(
            torch.cumsum(torch.flip(logs, dims=(-1,)), dim=-1), dims=(-1,)
        )
        weights = torch.exp(suffix - logs)
        local_u = u[:, start:end].permute(0, 2, 1, 3)
        local_h = h[:, start:end].permute(0, 2, 1, 3)
        weighted_u = local_u * weights[..., None]
        chunk_lambda = torch.exp(logs.sum(-1))
        current_m = chunk_lambda * current_m + weights.sum(-1)
        current_J = (
            chunk_lambda[..., None, None] * current_J
            + weighted_u.transpose(-1, -2) @ local_u
        )
        current_D = (
            chunk_lambda[..., None, None] * current_D
            + weighted_u.transpose(-1, -2) @ local_h
        )
    return (
        torch.stack(boundary_m, dim=2),
        torch.stack(boundary_J, dim=2),
        torch.stack(boundary_D, dim=2),
        current_m,
        current_J,
        current_D,
    )


@pytest.mark.parametrize(
    ("dtype", "precision", "ceiling"),
    [
        (torch.float32, "ieee", 2e-4),
        (torch.float32, "tf32", 2e-3),
        (torch.bfloat16, "ieee", 5e-3),
    ],
)
def test_triton_chunk_geometry_matches_fp64_reference(dtype, precision, ceiling) -> None:
    torch.manual_seed(916)
    batch, length, heads, rank, edits, value_dim = 1, 130, 2, 64, 1, 8
    u = F.normalize(torch.randn(batch, length, heads, rank, device="cuda"), dim=-1).to(dtype)
    h = (0.2 * torch.randn_like(u.float())).to(dtype)
    log_decay = (-0.05 * torch.rand(batch, length, heads, device="cuda")).to(dtype)
    boundary, final = triton_geometry_chunk_scan(
        u, h, log_decay, input_precision=precision
    )

    zeros_r = torch.zeros(batch, length, heads, rank, device="cuda", dtype=torch.float64)
    keys = torch.randn(batch, length, heads, edits, rank, device="cuda", dtype=torch.float64)
    values = torch.zeros(batch, length, heads, edits, value_dim, device="cuda", dtype=torch.float64)
    _, reference_final, history = solvedelta_reference(
        u.double(),
        h.double(),
        zeros_r + 1.0,
        keys,
        values,
        log_decay.double(),
        zeros_r,
        torch.ones_like(keys),
        torch.zeros_like(values),
        torch.zeros(batch, length, heads, edits, device="cuda", dtype=torch.float64),
        torch.zeros(heads, device="cuda", dtype=torch.float64),
        return_state_history=True,
    )
    reference_boundary_m = torch.stack(
        (
            torch.zeros_like(reference_final.m),
            history.m[:, 63],
            history.m[:, 127],
        ),
        dim=2,
    )
    reference_boundary_J = torch.stack(
        (
            torch.zeros_like(reference_final.J),
            history.J[:, 63],
            history.J[:, 127],
        ),
        dim=2,
    )
    reference_boundary_D = torch.stack(
        (
            torch.zeros_like(reference_final.D),
            history.D[:, 63],
            history.D[:, 127],
        ),
        dim=2,
    )
    assert _rho(reference_boundary_m, boundary.m) < ceiling
    assert _rho(reference_boundary_J, boundary.J) < ceiling
    assert _rho(reference_boundary_D, boundary.D) < ceiling
    assert _rho(reference_final.m, final.m) < ceiling
    assert _rho(reference_final.J, final.J) < ceiling
    assert _rho(reference_final.D, final.D) < ceiling


def test_triton_geometry_rejects_unsupported_chunk_size() -> None:
    u = torch.randn(1, 8, 1, 32, device="cuda")
    h = torch.randn_like(u)
    log_decay = torch.zeros(1, 8, 1, device="cuda")
    with pytest.raises(ValueError, match="one of 16, 32, 64"):
        triton_geometry_chunk_scan(u, h, log_decay, chunk_size=8)


def _geometry_vjp(
    inputs: tuple[torch.Tensor, ...],
    output_grads: tuple[torch.Tensor, ...],
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, ...]:
    u, h, log_decay, initial_m, initial_J, initial_D = (
        value.detach().requires_grad_(True) for value in inputs
    )
    initial = SolveDeltaState(
        initial_m,
        initial_J,
        initial_D,
        torch.empty(0, device=u.device, dtype=u.dtype),
    )
    boundary, final = triton_geometry_chunk_scan(
        u,
        h,
        log_decay,
        initial_state=initial,
        chunk_size=chunk_size,
        input_precision="ieee",
    )
    outputs = (
        boundary.m,
        boundary.J,
        boundary.D,
        final.m,
        final.J,
        final.D,
    )
    return torch.autograd.grad(outputs, (u, h, log_decay, initial_m, initial_J, initial_D), output_grads)


@pytest.mark.parametrize(
    ("length", "rank", "chunk_size", "underflow"),
    [
        (7, 128, 16, False),
        (16, 128, 16, False),
        (17, 128, 16, False),
        (65, 128, 16, False),
        (11, 128, 16, True),
        (130, 64, 64, False),
    ],
)
def test_triton_geometry_affine_adjoint_matches_fp64(
    length: int,
    rank: int,
    chunk_size: int,
    underflow: bool,
) -> None:
    torch.manual_seed(20260823 + length + rank)
    batch, heads = 1, 2
    u = F.normalize(
        torch.randn(batch, length, heads, rank, device="cuda"), dim=-1
    )
    h = 0.2 * torch.randn_like(u)
    log_decay = -0.2 * torch.rand(batch, length, heads, device="cuda")
    if underflow:
        log_decay[:, 3] = -110.0
        log_decay[:, 7] = -1000.0
    initial_m = 0.5 + torch.rand(batch, heads, device="cuda")
    # The optimized adjoint must not rely on canonical J symmetry.
    initial_J = 0.04 * torch.randn(batch, heads, rank, rank, device="cuda")
    initial_D = 0.04 * torch.randn_like(initial_J)
    inputs = (u, h, log_decay, initial_m, initial_J, initial_D)
    with torch.no_grad():
        reference_inputs = tuple(value.double() for value in inputs)
        reference_outputs = _geometry_scan_recompute(
            *reference_inputs, chunk_size
        )
    torch.manual_seed(20260901 + length + rank)
    output_grads = tuple(
        torch.randn_like(output, dtype=torch.float32)
        for output in reference_outputs
    )

    actual = _geometry_vjp(inputs, output_grads, chunk_size=chunk_size)
    reference_variables = tuple(
        value.double().detach().requires_grad_(True) for value in inputs
    )
    reference_outputs = _geometry_scan_recompute(
        *reference_variables, chunk_size
    )
    expected = torch.autograd.grad(
        reference_outputs,
        reference_variables,
        tuple(value.double() for value in output_grads),
    )
    ceiling = 1e-4
    for expected_grad, actual_grad in zip(expected, actual):
        assert torch.isfinite(actual_grad).all()
        assert _rho(expected_grad, actual_grad) < ceiling


def test_triton_geometry_affine_adjoint_is_bitwise_repeatable() -> None:
    torch.manual_seed(20260903)
    batch, length, heads, rank = 1, 17, 2, 128
    inputs = (
        F.normalize(
            torch.randn(batch, length, heads, rank, device="cuda"), dim=-1
        ),
        0.2 * torch.randn(batch, length, heads, rank, device="cuda"),
        -0.2 * torch.rand(batch, length, heads, device="cuda"),
        0.5 + torch.rand(batch, heads, device="cuda"),
        0.04 * torch.randn(batch, heads, rank, rank, device="cuda"),
        0.04 * torch.randn(batch, heads, rank, rank, device="cuda"),
    )
    with torch.no_grad():
        outputs = _geometry_scan_recompute(
            *(value.double() for value in inputs), 16
        )
    output_grads = tuple(torch.randn_like(value, dtype=torch.float32) for value in outputs)
    first = _geometry_vjp(inputs, output_grads, chunk_size=16)
    second = _geometry_vjp(inputs, output_grads, chunk_size=16)
    assert all(torch.equal(left, right) for left, right in zip(first, second))


def test_triton_geometry_affine_adjoint_driven_cancellation_contract() -> None:
    torch.manual_seed(1947)
    batch, length, heads, rank = 1, 16, 1, 128
    left = F.normalize(
        torch.randn(rank, device="cuda", dtype=torch.float64), dim=0
    )
    right = F.normalize(
        torch.randn(rank, device="cuda", dtype=torch.float64), dim=0
    )
    u = torch.zeros(batch, length, heads, rank, device="cuda")
    h = torch.zeros_like(u)
    u[:, 0, 0] = left.float()
    h[:, 0, 0] = (-4096.0 + 0.01) * right.float()
    u[:, 1:, 0] = F.normalize(
        torch.randn(batch, length - 1, rank, device="cuda"), dim=-1
    )
    h[:, 1:, 0] = 0.2 * torch.randn_like(h[:, 1:, 0])
    log_decay = torch.zeros(batch, length, heads, device="cuda")
    initial_m = torch.ones(batch, heads, device="cuda")
    initial_J = 0.04 * torch.randn(batch, heads, rank, rank, device="cuda")
    initial_D = (4096.0 * torch.outer(left, right)).float()[None, None]
    inputs = (u, h, log_decay, initial_m, initial_J, initial_D)
    with torch.no_grad():
        outputs = _geometry_scan_recompute(
            *(value.double() for value in inputs), 16
        )
    output_grads = tuple(torch.zeros_like(value, dtype=torch.float32) for value in outputs)
    output_grads = (
        *output_grads[:-1],
        torch.outer(left, right).float()[None, None],
    )
    actual = _geometry_vjp(inputs, output_grads, chunk_size=16)
    variables = tuple(value.double().detach().requires_grad_(True) for value in inputs)
    reference_outputs = _geometry_scan_recompute(*variables, 16)
    expected = torch.autograd.grad(
        reference_outputs,
        variables,
        tuple(value.double() for value in output_grads),
    )
    for expected_grad, actual_grad in zip(expected, actual):
        assert _rho(expected_grad, actual_grad) < 1e-4
    # The small tail follows a legal 2^12 cancellation. Lock its absolute
    # FP32 envelope separately so the large first coordinate cannot hide drift.
    torch.testing.assert_close(
        actual[2][:, 1:].double(), expected[2][:, 1:], rtol=3e-2, atol=5e-4
    )
