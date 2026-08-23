import pytest
import torch
import torch.nn.functional as F

from causallsso.ops.native_chunk import _load_chunk_library, native_geometry_frame
from causallsso.reference import SolveDeltaState, bounded_ldu_reference
from frame_oracle import frame_oracle as chunk_frame


def _cpu_inputs(*, length: int = 33) -> tuple[torch.Tensor, ...]:
    batch, heads, rank = 1, 2, 128
    chunks = (length + 31) // 32
    return (
        torch.empty(batch, length, heads, rank, dtype=torch.bfloat16),
        torch.empty(batch, length, heads, rank, dtype=torch.bfloat16),
        torch.empty(batch, length, heads),
        torch.empty(
            batch, length, heads, 1, rank, dtype=torch.bfloat16
        ),
        torch.empty(
            batch, length, heads, 1, rank, dtype=torch.bfloat16
        ),
        torch.empty(batch, length, heads, rank, dtype=torch.bfloat16),
        torch.empty(heads),
        torch.empty(batch, heads, chunks),
        torch.empty(batch, heads, chunks, rank, rank),
        torch.empty(batch, heads, chunks, rank, rank),
    )


def _resident_frame(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    _load_chunk_library()
    return tuple(
        torch.ops.causallsso.c32_frame_resident_forward(
            *(tensor.contiguous() for tensor in inputs)
        )
    )[:3]


def _public_geometry_call(inputs: tuple[torch.Tensor, ...]):
    initial = SolveDeltaState(
        inputs[7][:, :, 0],
        inputs[8][:, :, 0],
        inputs[9][:, :, 0],
        torch.empty(0, device=inputs[0].device, dtype=torch.float32),
    )
    return native_geometry_frame(*inputs[:7], initial_state=initial)


def test_native_geometry_rejects_cpu_before_loading_library() -> None:
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        _public_geometry_call(_cpu_inputs())


def test_native_geometry_requires_one_edit() -> None:
    inputs = list(_cpu_inputs())
    inputs[3] = torch.empty(1, 33, 2, 2, 128)
    with pytest.raises(ValueError, match="key must have shape"):
        _public_geometry_call(tuple(inputs))


def test_native_geometry_requires_bf16_vectors() -> None:
    inputs = list(_cpu_inputs())
    inputs[1] = inputs[1].float()
    with pytest.raises(TypeError, match="h must be BF16"):
        _public_geometry_call(tuple(inputs))


def test_native_geometry_requires_fp32_geometry() -> None:
    inputs = list(_cpu_inputs())
    inputs[2] = inputs[2].bfloat16()
    with pytest.raises(TypeError, match="geometry_log_decay must be FP32"):
        _public_geometry_call(tuple(inputs))


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
        torch.randn(batch, length, heads, rank, device="cuda").bfloat16().float(),
        dim=-1,
    ).bfloat16()
    h = (0.2 * torch.randn_like(u.float())).bfloat16()
    key = F.normalize(
        torch.randn(
            batch, length, heads, 1, rank, device="cuda"
        ).bfloat16().float(),
        dim=-1,
    ).bfloat16()
    raw_j = 0.04 * torch.randn(
        batch, heads, chunks, rank, rank, device="cuda"
    )
    return (
        u,
        h,
        -0.08 * torch.rand(batch, length, heads, device="cuda"),
        key,
        (2.0 * torch.rand_like(key.float())).bfloat16(),
        F.normalize(torch.randn_like(u).float(), dim=-1).bfloat16(),
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


def _assert_budget(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    rho_ceiling: float,
    name: str,
) -> None:
    assert torch.isfinite(actual).all(), name
    difference = actual.double() - reference.double()
    absolute = float(difference.abs().max())
    absolute_ceiling = 2e-4 if actual.dtype == torch.bfloat16 else 1e-6
    rho = _rho(reference, actual)
    assert absolute <= absolute_ceiling or rho <= rho_ceiling, (
        f"{name}: rho={rho:.6e}, a_inf={absolute:.6e}, "
        f"ceiling={rho_ceiling:.6e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("length", (3, 31, 32, 33))
def test_native_chunk_forward_matches_fp64_chunk_oracle(
    length: int,
) -> None:
    inputs = list(_cuda_inputs(length=length))
    with torch.no_grad():
        actual = _resident_frame(*inputs)
        expected = chunk_frame(
            *(tensor.double() for tensor in inputs), chunk_size=32
        )
    for name, expected_tensor, actual_tensor in zip(
        ("d", "e", "chi"), expected, actual
    ):
        assert actual_tensor.dtype == torch.bfloat16
        _assert_budget(
            expected_tensor,
            actual_tensor,
            rho_ceiling=6e-3,
            name=f"T={length}.{name}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_chunk_forward_supports_multiple_batches_and_heads() -> None:
    inputs = _cuda_inputs(length=3, batch=2, heads=3)
    with torch.no_grad():
        actual = _resident_frame(*inputs)
        expected = chunk_frame(
            *(tensor.double() for tensor in inputs), chunk_size=32
        )
    for name, expected_tensor, actual_tensor in zip(
        ("d", "e", "chi"), expected, actual
    ):
        _assert_budget(
            expected_tensor,
            actual_tensor,
            rho_ceiling=6e-3,
            name=f"multi_panel.{name}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_chunk_saved_forward_quantities_match_fp64() -> None:
    inputs = _cuda_inputs(length=3)
    contiguous = tuple(tensor.contiguous() for tensor in inputs)
    _resident_frame(*contiguous)
    raw = torch.ops.causallsso.c32_frame_resident_forward(*contiguous)
    (
        d,
        _,
        _,
        lower_primal,
        lower_dual_scaled,
        write_fp32,
        inverse_mass,
        _radial_scale,
        _radial_q2,
        diagonal,
        alpha0,
    ) = raw

    u, h, log_decay, key, erase, query, strength, bm, bJ, bD = (
        tensor.double() for tensor in inputs
    )
    mass = bm[:, :, 0]
    moment_j = bJ[:, :, 0]
    moment_d = bD[:, :, 0]
    expected_lower = []
    expected_dual = []
    expected_write = []
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
        lower, expected_diagonal, upper = bounded_ldu_reference(
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
        dual_scaled = expected_diagonal.unsqueeze(-1) * (
            lower.transpose(-1, -2) @ dual_rhs
        )
        write = torch.linalg.solve_triangular(
            upper,
            (lower_key / expected_diagonal).unsqueeze(-1),
            upper=True,
            unitriangular=True,
        ).squeeze(-1)
        expected_lower.append(lower_key)
        expected_dual.append(dual_scaled.transpose(-1, -2))
        expected_write.append(write)

    expected_lower = torch.stack(expected_lower, dim=1)
    expected_dual = torch.stack(expected_dual, dim=1)
    expected_write = torch.stack(expected_write, dim=1)
    _assert_budget(
        expected_lower,
        lower_primal,
        rho_ceiling=6e-3,
        name="lower_primal",
    )
    _assert_budget(
        expected_dual,
        lower_dual_scaled,
        rho_ceiling=6e-3,
        name="lower_dual_scaled",
    )
    _assert_budget(
        expected_write,
        write_fp32,
        rho_ceiling=6e-3,
        name="write_fp32",
    )
    assert torch.equal(write_fp32[:, :, :, None].bfloat16(), d)
    assert torch.count_nonzero(diagonal[:1, :3] != 1.0) > 0

    first_decay = torch.exp(log_decay[:, 0])
    first_mass = first_decay * bm[:, :, 0] + 1.0
    expected_inverse = (1.0 / first_mass).reshape(-1)
    expected_alpha0 = (first_decay / first_mass).reshape(-1)
    torch.testing.assert_close(
        inverse_mass[:, :, 0, 0].reshape(-1).double(),
        expected_inverse,
        rtol=2e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        alpha0.double(), expected_alpha0, rtol=2e-6, atol=1e-7
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_chunk_forward_is_repeatable_and_preserves_pairing() -> None:
    inputs = _cuda_inputs(length=33)
    contiguous = tuple(tensor.contiguous() for tensor in inputs)
    _resident_frame(*contiguous)
    first_raw = torch.ops.causallsso.c32_frame_resident_forward(*contiguous)
    second_raw = torch.ops.causallsso.c32_frame_resident_forward(*contiguous)
    for first, second in zip(first_raw, second_raw):
        assert torch.equal(first, second)
    d, e = first_raw[:2]
    expected_pairing = (
        inputs[3].float()
        * (inputs[4].float() * inputs[3].float())
    ).sum(dim=-1)
    actual_pairing = (d.float() * e.float()).sum(dim=-1)
    denominator = (
        d.float().norm(dim=-1) * e.float().norm(dim=-1)
        + inputs[3].float().norm(dim=-1)
        * (inputs[4].float() * inputs[3].float()).norm(dim=-1)
        + 1e-12
    )
    assert ((actual_pairing - expected_pairing).abs() / denominator).max() <= 8e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_chunk_identity_geometry_is_exact() -> None:
    inputs = list(_cuda_inputs(length=33, strength=0.0))
    d, e, chi = _resident_frame(*inputs)
    expected_e = (inputs[4].float() * inputs[3].float()).bfloat16()
    assert torch.equal(d, inputs[3])
    assert torch.equal(e, expected_e)
    assert torch.equal(chi, inputs[5])


def _cancellation_inputs(kind: str) -> tuple[torch.Tensor, ...]:
    inputs = list(_cuda_inputs(length=32))

    direction_master = torch.tensor(
        [0.6953125, 0.71875], device="cuda", dtype=torch.float64
    )
    direction = F.normalize(direction_master, dim=0).to(torch.bfloat16)
    assert torch.equal(
        direction,
        torch.tensor(
            [0.6953125, 0.71875], device="cuda", dtype=torch.bfloat16
        ),
    )
    inputs[7].fill_(1.0)
    inputs[8].zero_()
    inputs[9].zero_()
    inputs[2].zero_()
    inputs[0][0, 0, 0].zero_()
    inputs[0][0, 0, 0, :2] = direction
    inputs[1][0, 0, 0].zero_()
    if kind == "J":
        inputs[8][0, 0, 0, 0, 0] = 0.484375
        inputs[8][0, 0, 0, 1, 1] = 0.51953125
        inputs[8][0, 0, 0, 0, 1] = -0.5
        inputs[8][0, 0, 0, 1, 0] = -0.5
    else:
        inputs[1][0, 0, 0, 0] = 0.6953125
        inputs[9][0, 0, 0, 1, 0] = -0.5
    return tuple(inputs)


def _cancellation_ratio(kind: str, inputs: tuple[torch.Tensor, ...]) -> float:
    if kind == "J":
        boundary = inputs[8][0, 0, 0, 0, 1].double()
        local = (
            inputs[0][0, 0, 0, 0].double()
            * inputs[0][0, 0, 0, 1].double()
        )
    else:
        boundary = inputs[9][0, 0, 0, 1, 0].double()
        local = (
            inputs[0][0, 0, 0, 1].double()
            * inputs[1][0, 0, 0, 0].double()
        )
    return float((boundary.abs() + local.abs()) / (boundary + local).abs())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ("J", "D"))
def test_native_chunk_cancellation_and_underflow_contract(kind: str) -> None:
    inputs = list(_cancellation_inputs(kind))
    assert _cancellation_ratio(kind, tuple(inputs)) == 4095.0
    for index in (0, 1, 3, 4, 5):
        assert inputs[index].dtype == torch.bfloat16
    if kind == "J":
        assert torch.linalg.eigvalsh(inputs[8][0, 0, 0].double()).min() >= 0
    inputs[2][:, 3] = -110.0
    inputs[2][:, 7] = -1000.0
    with torch.no_grad():
        actual = _resident_frame(*inputs)
        expected = chunk_frame(
            *(tensor.double() for tensor in inputs), chunk_size=32
        )
    for name, expected_tensor, actual_tensor in zip(
        ("d", "e", "chi"), expected, actual
    ):
        _assert_budget(
            expected_tensor,
            actual_tensor,
            rho_ceiling=6e-3,
            name=f"cancellation.{kind}.{name}",
        )
