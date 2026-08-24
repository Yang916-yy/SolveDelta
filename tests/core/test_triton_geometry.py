import pytest
import torch
import torch.nn.functional as F

from causallsso import SolveDeltaState, solvedelta_reference
from causallsso.ops import triton_geometry_chunk_scan
from causallsso.ops.triton_geometry import (
    _triton_geometry_chunk_scan_backward,
    _triton_geometry_chunk_scan_forward,
)


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
    log_decay = -0.05 * torch.rand(batch, length, heads, device="cuda")
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


def test_triton_geometry_strided_inputs_match_contiguous_forward_and_vjp() -> None:
    torch.manual_seed(20260825)
    batch, length, heads, rank = 1, 33, 2, 64

    def padded_view(value: torch.Tensor) -> torch.Tensor:
        flat = value.reshape(batch, length, -1)
        storage = torch.cat((flat, torch.zeros_like(flat[..., :7])), dim=-1)
        return storage[..., : flat.shape[-1]].view_as(value)

    dense_u = F.normalize(
        torch.randn(batch, length, heads, rank, device="cuda"), dim=-1
    )
    dense_h = 0.2 * torch.randn_like(dense_u)
    dense_decay = -0.1 * torch.rand(batch, length, heads, device="cuda")
    strided_inputs = tuple(
        value.detach().requires_grad_(True)
        for value in (
            padded_view(dense_u),
            padded_view(dense_h),
            padded_view(dense_decay),
        )
    )
    assert all(not value.is_contiguous() for value in strided_inputs)

    with torch.no_grad():
        direct_boundary, direct_final = triton_geometry_chunk_scan(
            *(value.detach() for value in strided_inputs),
            chunk_size=32,
            input_precision="ieee",
        )
        dense_boundary, dense_final = triton_geometry_chunk_scan(
            dense_u,
            dense_h,
            dense_decay,
            chunk_size=32,
            input_precision="ieee",
        )
    for actual, expected in zip(
        (*direct_boundary[:3], *direct_final[:3]),
        (*dense_boundary[:3], *dense_final[:3]),
    ):
        assert torch.equal(actual, expected)

    boundary, final = triton_geometry_chunk_scan(
        *strided_inputs,
        chunk_size=32,
        input_precision="ieee",
    )
    expected_boundary, expected_final = triton_geometry_chunk_scan(
        dense_u,
        dense_h,
        dense_decay,
        chunk_size=32,
        input_precision="ieee",
    )
    actual_outputs = (*boundary[:3], *final[:3])
    expected_outputs = (*expected_boundary[:3], *expected_final[:3])
    for actual, expected in zip(actual_outputs, expected_outputs):
        assert torch.equal(actual, expected)

    output_grads = tuple(torch.randn_like(value) for value in actual_outputs)
    actual_grads = torch.autograd.grad(
        actual_outputs,
        strided_inputs,
        output_grads,
    )
    contiguous_inputs = tuple(
        value.detach().contiguous().requires_grad_(True)
        for value in strided_inputs
    )
    contiguous_boundary, contiguous_final = triton_geometry_chunk_scan(
        *contiguous_inputs,
        chunk_size=32,
        input_precision="ieee",
    )
    expected_grads = torch.autograd.grad(
        (*contiguous_boundary[:3], *contiguous_final[:3]),
        contiguous_inputs,
        output_grads,
    )
    for actual, expected in zip(actual_grads, expected_grads):
        assert torch.equal(actual, expected)


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
        torch.empty(0, device=u.device, dtype=torch.float32),
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
    ("batch", "length", "heads", "rank", "chunk_size", "underflow"),
    [
        (1, 7, 2, 128, 16, False),
        (1, 16, 2, 128, 16, False),
        (1, 17, 2, 128, 16, False),
        (1, 65, 2, 128, 16, False),
        (1, 11, 2, 128, 16, True),
        (1, 130, 2, 64, 64, False),
        (2, 35, 2, 128, 32, False),
    ],
)
def test_triton_geometry_affine_adjoint_matches_fp64(
    batch: int,
    length: int,
    heads: int,
    rank: int,
    chunk_size: int,
    underflow: bool,
) -> None:
    torch.manual_seed(20260823 + length + rank)
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


def test_triton_geometry_bf16_affine_adjoint_matches_quantized_fp64() -> None:
    torch.manual_seed(20260906)
    batch, length, heads, rank, chunk_size = 1, 65, 2, 128, 16
    inputs = (
        F.normalize(
            torch.randn(batch, length, heads, rank, device="cuda"), dim=-1
        ).to(torch.bfloat16),
        (0.2 * torch.randn(batch, length, heads, rank, device="cuda")).to(
            torch.bfloat16
        ),
        -0.2 * torch.rand(batch, length, heads, device="cuda"),
        0.5 + torch.rand(batch, heads, device="cuda"),
        0.04 * torch.randn(batch, heads, rank, rank, device="cuda"),
        0.04 * torch.randn(batch, heads, rank, rank, device="cuda"),
    )
    reference_inputs = tuple(value.double() for value in inputs)
    reference_outputs = _geometry_scan_recompute(
        *reference_inputs, chunk_size
    )
    torch.manual_seed(20260907)
    output_grads = tuple(
        torch.randn_like(value, dtype=torch.float32)
        for value in reference_outputs
    )

    actual = _geometry_vjp(inputs, output_grads, chunk_size=chunk_size)
    reference_variables = tuple(
        value.detach().double().requires_grad_(True) for value in inputs
    )
    expected_outputs = _geometry_scan_recompute(
        *reference_variables, chunk_size
    )
    expected = torch.autograd.grad(
        expected_outputs,
        reference_variables,
        tuple(value.double() for value in output_grads),
    )
    for index, (expected_grad, actual_grad) in enumerate(zip(expected, actual)):
        assert torch.isfinite(actual_grad).all()
        ceiling = 5e-3 if index < 3 else 5e-4
        assert _rho(expected_grad, actual_grad) < ceiling


def test_lower_level_bf16_scan_backward_keeps_fp32_partials_for_composition() -> None:
    torch.manual_seed(20260910)
    batch, length, heads, rank, chunk_size = 1, 17, 1, 128, 16
    u = F.normalize(
        torch.randn(batch, length, heads, rank, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    h = (0.2 * torch.randn_like(u.float())).to(torch.bfloat16)
    log_decay = -0.2 * torch.rand(batch, length, heads, device="cuda")
    initial = SolveDeltaState(
        0.5 + torch.rand(batch, heads, device="cuda"),
        0.04 * torch.randn(batch, heads, rank, rank, device="cuda"),
        0.04 * torch.randn(batch, heads, rank, rank, device="cuda"),
        torch.empty(0, device="cuda", dtype=torch.float32),
    )
    boundary, final = _triton_geometry_chunk_scan_forward(
        u,
        h,
        log_decay,
        initial_state=initial,
        chunk_size=chunk_size,
        input_precision="ieee",
    )
    outputs = (*boundary[:3], *final[:3])
    output_grads = tuple(
        torch.randn_like(output, dtype=torch.float32) for output in outputs
    )

    partials = _triton_geometry_chunk_scan_backward(
        u,
        h,
        log_decay,
        boundary.m,
        boundary.J,
        boundary.D,
        *output_grads,
        chunk_size,
    )
    assert all(partial.dtype == torch.float32 for partial in partials)
    for partial in partials[:2]:
        assert torch.count_nonzero(partial - partial.to(torch.bfloat16).float())

    public = _geometry_vjp(
        (u, h, log_decay, initial.m, initial.J, initial.D),
        output_grads,
        chunk_size=chunk_size,
    )
    assert public[0].dtype == public[1].dtype == torch.bfloat16
    assert public[2].dtype == torch.float32
    assert torch.equal(public[0], partials[0].to(torch.bfloat16))
    assert torch.equal(public[1], partials[1].to(torch.bfloat16))

    local_u = torch.randn_like(u, dtype=torch.float32)
    local_h = torch.randn_like(h, dtype=torch.float32)
    local_decay = torch.randn_like(log_decay)
    fused = _triton_geometry_chunk_scan_backward(
        u,
        h,
        log_decay,
        boundary.m,
        boundary.J,
        boundary.D,
        *output_grads,
        chunk_size,
        local_grad_u=local_u,
        local_grad_h=local_h,
        local_grad_log_decay=local_decay,
    )
    assert all(partial.dtype == torch.float32 for partial in fused)
    torch.testing.assert_close(fused[0], partials[0] + local_u)
    torch.testing.assert_close(fused[1], partials[1] + local_h)
    torch.testing.assert_close(fused[2], partials[2] + local_decay)
    for expected, actual in zip(partials[3:], fused[3:]):
        assert torch.equal(expected, actual)


def test_triton_geometry_bf16_deep_cancellation_uses_fp32_state() -> None:
    torch.manual_seed(20260908)
    batch, length, heads, rank, chunk_size = 1, 16, 1, 128, 16
    u = F.normalize(
        torch.randn(batch, length, heads, rank, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    h = (0.2 * torch.randn_like(u.float())).to(torch.bfloat16)
    u[0, 0, 0].zero_()
    u[0, 0, 0, :2] = torch.tensor(
        [0.6953125, 0.71875], device="cuda", dtype=torch.bfloat16
    )
    h[0, 0, 0].zero_()
    h[0, 0, 0, 0] = 0.6953125
    log_decay = torch.zeros(batch, length, heads, device="cuda")
    initial_m = torch.ones(batch, heads, device="cuda")
    initial_J = 0.04 * torch.randn(
        batch, heads, rank, rank, device="cuda"
    )
    initial_D = torch.zeros_like(initial_J)
    initial_D[0, 0, 1, 0] = -0.5
    inputs = (u, h, log_decay, initial_m, initial_J, initial_D)

    boundary = initial_D[0, 0, 1, 0].double()
    local = u[0, 0, 0, 1].double() * h[0, 0, 0, 0].double()
    kappa = (boundary.abs() + local.abs()) / (boundary + local).abs()
    assert kappa == 4095.0

    reference_variables = tuple(
        value.detach().double().requires_grad_(True) for value in inputs
    )
    expected_outputs = _geometry_scan_recompute(
        *reference_variables, chunk_size
    )
    torch.manual_seed(20260909)
    output_grads = tuple(
        torch.randn_like(value, dtype=torch.float32)
        for value in expected_outputs
    )
    actual = _geometry_vjp(inputs, output_grads, chunk_size=chunk_size)
    expected = torch.autograd.grad(
        expected_outputs,
        reference_variables,
        tuple(value.double() for value in output_grads),
    )
    for index, (expected_grad, actual_grad) in enumerate(zip(expected, actual)):
        assert torch.isfinite(actual_grad).all()
        ceiling = 5e-3 if index < 3 else 5e-4
        assert _rho(expected_grad, actual_grad) < ceiling


def test_triton_geometry_requires_fp32_decay_and_state() -> None:
    u = torch.zeros(1, 1, 1, 32, device="cuda", dtype=torch.bfloat16)
    h = torch.zeros_like(u)
    low_precision_decay = torch.zeros(
        1, 1, 1, device="cuda", dtype=torch.bfloat16
    )
    with pytest.raises(TypeError, match="geometry_log_decay must be FP32"):
        triton_geometry_chunk_scan(u, h, low_precision_decay)

    initial = SolveDeltaState(
        torch.zeros(1, 1, device="cuda", dtype=torch.bfloat16),
        torch.zeros(1, 1, 32, 32, device="cuda", dtype=torch.float32),
        torch.zeros(1, 1, 32, 32, device="cuda", dtype=torch.float32),
        torch.empty(0, device="cuda", dtype=torch.float32),
    )
    with pytest.raises(TypeError, match="initial_state.m must be FP32"):
        triton_geometry_chunk_scan(
            u,
            h,
            torch.zeros(1, 1, 1, device="cuda"),
            initial_state=initial,
        )


def test_triton_geometry_affine_adjoint_is_bitwise_repeatable() -> None:
    torch.manual_seed(20260903)
    batch, length, heads, rank, chunk_size = 2, 35, 2, 128, 32
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
            *(value.double() for value in inputs), chunk_size
        )
    output_grads = tuple(torch.randn_like(value, dtype=torch.float32) for value in outputs)
    first = _geometry_vjp(inputs, output_grads, chunk_size=chunk_size)
    second = _geometry_vjp(inputs, output_grads, chunk_size=chunk_size)
    assert all(torch.equal(left, right) for left, right in zip(first, second))


def test_triton_geometry_fp32_affine_adjoint_cancellation_diagnostic() -> None:
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
    # This is the retained FP32 diagnostic. The BF16 production fixture is
    # quantized before its cancellation ratio is classified.
    torch.testing.assert_close(
        actual[2][:, 1:].double(), expected[2][:, 1:], rtol=3e-2, atol=5e-4
    )
