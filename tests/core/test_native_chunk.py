import pytest
import torch
import torch.nn.functional as F

from causallsso.ops.chunk_frame import chunk_frame
from causallsso.ops.native_chunk import native_chunk_frame
from causallsso.reference import bounded_ldu_reference


def _cpu_inputs(*, length: int = 33) -> tuple[torch.Tensor, ...]:
    batch, heads, rank = 1, 2, 128
    chunks = (length + 31) // 32
    return (
        torch.empty(batch, length, heads, rank),
        torch.empty(batch, length, heads, rank),
        torch.empty(batch, length, heads),
        torch.empty(batch, length, heads, 1, rank),
        torch.empty(batch, length, heads, 1, rank),
        torch.empty(batch, length, heads, rank),
        torch.empty(heads),
        torch.empty(batch, heads, chunks),
        torch.empty(batch, heads, chunks, rank, rank),
        torch.empty(batch, heads, chunks, rank, rank),
    )


def test_native_chunk_rejects_cpu_before_loading_library() -> None:
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        native_chunk_frame(*_cpu_inputs())


def test_native_chunk_requires_one_edit() -> None:
    inputs = list(_cpu_inputs())
    inputs[3] = torch.empty(1, 33, 2, 2, 128)
    with pytest.raises(ValueError, match="key must have shape"):
        native_chunk_frame(*inputs)


def test_native_chunk_requires_exact_c32_boundaries() -> None:
    inputs = list(_cpu_inputs())
    inputs[7] = torch.empty(1, 2, 1)
    with pytest.raises(ValueError, match="boundary_m must have shape"):
        native_chunk_frame(*inputs)


def test_native_chunk_requires_fp32() -> None:
    inputs = list(_cpu_inputs())
    inputs[1] = inputs[1].double()
    with pytest.raises(TypeError, match="requires FP32 inputs"):
        native_chunk_frame(*inputs)


def _cuda_inputs(
    *,
    length: int,
    strength: float = 0.7,
    batch: int = 1,
    heads: int = 1,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260824 + length)
    rank = 128
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


def _rho(reference: torch.Tensor, actual: torch.Tensor) -> float:
    reference = reference.double()
    actual = actual.double()
    return float(
        (actual - reference).square().mean().sqrt()
        / (reference.square().mean().sqrt() + 1e-8)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("length", (1, 32, 33))
def test_native_chunk_forward_matches_fp64_chunk_oracle(
    length: int,
) -> None:
    inputs = list(_cuda_inputs(length=length))
    with torch.no_grad():
        actual = native_chunk_frame(*inputs)
        expected = chunk_frame(
            *(tensor.double() for tensor in inputs), chunk_size=32
        )
    for expected_tensor, actual_tensor in zip(expected, actual):
        assert torch.isfinite(actual_tensor).all()
        assert _rho(expected_tensor, actual_tensor) < 5e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_chunk_forward_supports_multiple_batches_and_heads() -> None:
    inputs = _cuda_inputs(length=3, batch=2, heads=3)
    with torch.no_grad():
        actual = native_chunk_frame(*inputs)
        expected = chunk_frame(
            *(tensor.double() for tensor in inputs), chunk_size=32
        )
    for expected_tensor, actual_tensor in zip(expected, actual):
        assert _rho(expected_tensor, actual_tensor) < 5e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_chunk_saved_forward_quantities_match_fp64() -> None:
    inputs = _cuda_inputs(length=3)
    contiguous = tuple(tensor.contiguous() for tensor in inputs)
    native_chunk_frame(*contiguous)
    raw = torch.ops.causallsso.c32_frame_forward(*contiguous)
    _, _, _, lower_primal, lower_dual_scaled = raw

    u, h, log_decay, key, erase, query, strength, bm, bJ, bD = (
        tensor.double() for tensor in inputs
    )
    mass = bm[:, :, 0]
    moment_j = bJ[:, :, 0]
    moment_d = bD[:, :, 0]
    expected_lower = []
    expected_dual = []
    for token in range(3):
        decay = torch.exp(log_decay[:, token])
        mass = decay * mass + 1
        moment_j = (
            decay[..., None, None] * moment_j
            + u[:, token, :, :, None] * u[:, token, :, None, :]
        )
        moment_d = (
            decay[..., None, None] * moment_d
            + u[:, token, :, :, None] * h[:, token, :, None, :]
        )
        lower, diagonal, _ = bounded_ldu_reference(
            moment_j / mass[..., None, None],
            moment_d / mass[..., None, None],
            strength,
        )
        key_rhs = key[:, token, :, 0].unsqueeze(-1)
        lower_key = torch.linalg.solve_triangular(
            lower, key_rhs, upper=False, unitriangular=True
        ).squeeze(-1)
        dual_rhs = torch.stack(
            (erase[:, token, :, 0] * key[:, token, :, 0], query[:, token]),
            dim=-1,
        )
        dual_scaled = diagonal.unsqueeze(-1) * (
            lower.transpose(-1, -2) @ dual_rhs
        )
        expected_lower.append(lower_key.unsqueeze(-2))
        expected_dual.append(dual_scaled.transpose(-1, -2))

    expected_lower = torch.stack(expected_lower, dim=1)
    expected_dual = torch.stack(expected_dual, dim=1)
    assert _rho(expected_lower, lower_primal) < 5e-4
    assert _rho(expected_dual, lower_dual_scaled) < 5e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_chunk_forward_is_repeatable_and_preserves_pairing() -> None:
    inputs = _cuda_inputs(length=33)
    contiguous = tuple(tensor.contiguous() for tensor in inputs)
    native_chunk_frame(*contiguous)
    first_raw = torch.ops.causallsso.c32_frame_forward(*contiguous)
    second_raw = torch.ops.causallsso.c32_frame_forward(*contiguous)
    for first, second in zip(first_raw, second_raw):
        assert torch.equal(first, second)
    d, e = first_raw[:2]
    expected_pairing = (
        inputs[3] * (inputs[4] * inputs[3])
    ).sum(dim=-1)
    actual_pairing = (d * e).sum(dim=-1)
    torch.testing.assert_close(
        actual_pairing, expected_pairing, rtol=5e-5, atol=5e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_chunk_identity_geometry_is_exact() -> None:
    inputs = list(_cuda_inputs(length=33, strength=0.0))
    d, e, chi = native_chunk_frame(*inputs)
    assert torch.equal(d, inputs[3])
    assert torch.equal(e, inputs[4] * inputs[3])
    assert torch.equal(chi, inputs[5])


def _cancellation_inputs(kind: str) -> tuple[torch.Tensor, ...]:
    inputs = list(_cuda_inputs(length=32))
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
    return tuple(inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ("J", "D"))
def test_native_chunk_cancellation_and_underflow_contract(kind: str) -> None:
    inputs = list(_cancellation_inputs(kind))
    inputs[2][:, 3] = -110.0
    inputs[2][:, 7] = -1000.0
    with torch.no_grad():
        actual = native_chunk_frame(*inputs)
        expected = chunk_frame(
            *(tensor.double() for tensor in inputs), chunk_size=32
        )
    for expected_tensor, actual_tensor in zip(expected, actual):
        assert torch.isfinite(actual_tensor).all()
        assert _rho(expected_tensor, actual_tensor) < 5e-4
