import pytest
import torch
import torch.nn.functional as F

from causallsso import (
    SolveDeltaState,
    apply_dual_reference,
    apply_primal_reference,
    bounded_ldu_reference,
    solvedelta_reference,
)
from causallsso.ops import (
    mathdx_available,
    cuda_chunk_solve_frame128,
    solvedelta_fused,
)
from causallsso.ops.mathdx import (
    _chunk_frame_recompute_one,
    _chunk_frame_vjp_one,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not mathdx_available(),
    reason="CUDA and built MathDx extension required",
)


def _rho(reference, actual):
    return (actual.double() - reference.double()).square().mean().sqrt() / (
        reference.double().square().mean().sqrt() + 1e-8
    )


def _oracle(u, h, log_decay, keys, erase, query, skew, strength):
    batch, length, heads, rank = u.shape
    mass = torch.zeros(batch, heads, dtype=torch.float64, device="cuda")
    J = torch.zeros(batch, heads, rank, rank, dtype=torch.float64, device="cuda")
    D = torch.zeros_like(J)
    ds, es, chis = [], [], []
    for t in range(length):
        decay = torch.exp(log_decay[:, t].double())
        mass = decay * mass + 1
        ut, ht = u[:, t].double(), h[:, t].double()
        J = decay[..., None, None] * J + ut[..., :, None] * ut[..., None, :]
        D = decay[..., None, None] * D + ut[..., :, None] * ht[..., None, :]
        lower, diagonal, upper, omega = bounded_ldu_reference(
            J / mass[..., None, None], D / mass[..., None, None], strength.double()
        )
        token_d, token_e = [], []
        for edit in range(keys.shape[-2]):
            a = keys[:, t, :, edit].double()
            b0 = erase[:, t, :, edit].double() * a
            tau = (b0 * a).sum(-1)
            direction = (omega @ a.unsqueeze(-1)).squeeze(-1)
            direction = direction / torch.sqrt(1 + direction.square().sum(-1, keepdim=True))
            b = b0 + (tau * (2 - tau) * skew[:, t, :, edit].double())[..., None] * direction
            token_d.append(apply_primal_reference(lower, diagonal, upper, a))
            token_e.append(apply_dual_reference(lower, diagonal, upper, b))
        ds.append(torch.stack(token_d, dim=2))
        es.append(torch.stack(token_e, dim=2))
        chis.append(apply_dual_reference(lower, diagonal, upper, query[:, t].double()))
    return torch.stack(ds, 1), torch.stack(es, 1), torch.stack(chis, 1)


def test_explicit_subchunk_vjp_matches_autograd_oracle() -> None:
    torch.manual_seed(20260822)
    batch, length, heads, rank = 1, 3, 1, 128
    initial_j = 0.01 * torch.randn(
        batch, heads, rank, rank, device="cuda"
    )
    initial_j = 0.5 * (initial_j + initial_j.transpose(-1, -2))
    inputs = (
        2.0 + torch.rand(batch, heads, device="cuda"),
        initial_j,
        0.01 * torch.randn(batch, heads, rank, rank, device="cuda"),
        F.normalize(torch.randn(batch, length, heads, rank, device="cuda"), dim=-1),
        0.2 * torch.randn(batch, length, heads, rank, device="cuda"),
        -0.05 * torch.rand(batch, length, heads, device="cuda"),
        F.normalize(
            torch.randn(batch, length, heads, 1, rank, device="cuda"), dim=-1
        ),
        2.0 * torch.rand(batch, length, heads, 1, rank, device="cuda"),
        F.normalize(torch.randn(batch, length, heads, rank, device="cuda"), dim=-1),
        2.0 * torch.rand(batch, length, heads, 1, device="cuda") - 1.0,
        torch.sigmoid(torch.randn(heads, device="cuda")),
    )
    oracle_inputs = tuple(x.detach().requires_grad_(True) for x in inputs)
    outputs = _chunk_frame_recompute_one(*oracle_inputs)
    output_grads = tuple(torch.randn_like(x) for x in outputs)
    expected = torch.autograd.grad(outputs, oracle_inputs, output_grads)
    actual = _chunk_frame_vjp_one(*inputs, *output_grads)
    for expected_grad, actual_grad in zip(expected, actual):
        assert torch.isfinite(actual_grad).all()
        assert _rho(expected_grad, actual_grad) < 1e-3


def test_fused_chunk_frame_matches_fp64() -> None:
    torch.manual_seed(916)
    batch, length, heads, rank = 1, 7, 2, 128
    u = F.normalize(torch.randn(batch, length, heads, rank, device="cuda"), dim=-1)
    h = 0.2 * torch.randn_like(u)
    keys = F.normalize(torch.randn(batch, length, heads, 1, rank, device="cuda"), dim=-1)
    erase = 2 * torch.rand_like(keys)
    query = F.normalize(torch.randn_like(u), dim=-1)
    skew = 2 * torch.rand(batch, length, heads, 1, device="cuda") - 1
    log_decay = -0.05 * torch.rand(batch, length, heads, device="cuda")
    strength = torch.sigmoid(torch.randn(heads, device="cuda"))
    boundary_m = torch.zeros(batch, heads, 1, device="cuda")
    boundary_J = torch.zeros(batch, heads, 1, rank, rank, device="cuda")
    boundary_D = torch.zeros_like(boundary_J)
    actual = cuda_chunk_solve_frame128(
        boundary_m, boundary_J, boundary_D, u, h, log_decay,
        keys, erase, query, skew, strength, chunk_size=64,
    )
    expected = _oracle(u, h, log_decay, keys, erase, query, skew, strength)
    for reference, native in zip(expected, actual):
        assert _rho(reference, native) < 5e-4


@pytest.mark.parametrize("length", [17, 65])
@pytest.mark.parametrize(
    ("outer_dtype", "output_ceiling"),
    [(torch.float16, 5e-3), (torch.bfloat16, 6e-3)],
    ids=("fp16", "bf16"),
)
def test_fused_forward_matches_fp64_contract(
    length: int,
    outer_dtype: torch.dtype,
    output_ceiling: float,
) -> None:
    torch.manual_seed(20260817 + length)
    batch, heads, rank, value_dim = 1, 2, 128, 32
    u = torch.randn(batch, length, heads, rank, device="cuda")
    h = 0.2 * torch.randn_like(u)
    query = torch.randn_like(u)
    keys = torch.randn(batch, length, heads, 1, rank, device="cuda")
    values = 0.1 * torch.randn(batch, length, heads, 1, value_dim, device="cuda")
    geometry_decay = -0.05 * torch.rand(batch, length, heads, device="cuda")
    associative_decay = -0.03 * torch.rand(batch, length, heads, rank, device="cuda")
    erase = 2 * torch.rand_like(keys)
    write = 2 * torch.rand_like(values)
    skew = 2 * torch.rand(batch, length, heads, 1, device="cuda") - 1
    strength = torch.sigmoid(torch.randn(heads, device="cuda"))
    with torch.inference_mode():
        actual, actual_state = solvedelta_fused(
            u, h, query, keys, values,
            geometry_decay, associative_decay, erase, write, skew, strength,
            output_final_state=True,
            outer_dtype=outer_dtype,
        )
    expected, expected_state = solvedelta_reference(
        u.double(), h.double(), query.double(), keys.double(), values.double(),
        geometry_decay.double(), associative_decay.double(), erase.double(),
        write.double(), skew.double(), strength.double(),
    )
    assert _rho(expected, actual) < output_ceiling
    assert _rho(expected_state.S, actual_state.S) < output_ceiling
    assert _rho(expected_state.J, actual_state.J) < 2e-4
    assert _rho(expected_state.D, actual_state.D) < 2e-4


@pytest.mark.parametrize("length", [3, 65])
@pytest.mark.parametrize(
    ("outer_dtype", "main_ceiling", "geometry_ceiling"),
    [(torch.float16, 1e-2, 2e-2), (torch.bfloat16, 1.5e-2, 2.5e-2)],
    ids=("fp16", "bf16"),
)
def test_fused_backward_matches_fp64_contract(
    length: int,
    outer_dtype: torch.dtype,
    main_ceiling: float,
    geometry_ceiling: float,
) -> None:
    torch.manual_seed(20260819 + length)
    batch, heads, rank, value_dim = 1, 1, 128, 16
    master = (
        torch.randn(batch, length, heads, rank, device="cuda"),
        0.2 * torch.randn(batch, length, heads, rank, device="cuda"),
        torch.randn(batch, length, heads, rank, device="cuda"),
        torch.randn(batch, length, heads, 1, rank, device="cuda"),
        0.1 * torch.randn(batch, length, heads, 1, value_dim, device="cuda"),
        -0.05 * torch.rand(batch, length, heads, device="cuda"),
        -0.03 * torch.rand(batch, length, heads, rank, device="cuda"),
        2 * torch.rand(batch, length, heads, 1, rank, device="cuda"),
        2 * torch.rand(batch, length, heads, 1, value_dim, device="cuda"),
        2 * torch.rand(batch, length, heads, 1, device="cuda") - 1,
        torch.sigmoid(torch.randn(heads, device="cuda")),
    )
    native_inputs = tuple(x.detach().requires_grad_(True) for x in master)
    reference_inputs = tuple(x.detach().double().requires_grad_(True) for x in master)
    initial_master = SolveDeltaState(
        5.0 + torch.rand(batch, heads, device="cuda"),
        0.02 * torch.eye(rank, device="cuda").expand(batch, heads, rank, rank).clone(),
        0.01 * torch.randn(batch, heads, rank, rank, device="cuda"),
        0.03 * torch.randn(batch, heads, rank, value_dim, device="cuda"),
    )
    native_initial = SolveDeltaState(
        *(x.detach().requires_grad_(True) for x in initial_master)
    )
    reference_initial = SolveDeltaState(
        *(x.detach().double().requires_grad_(True) for x in initial_master)
    )
    do = torch.randn(batch, length, heads, value_dim, device="cuda")
    dS = torch.randn(batch, heads, rank, value_dim, device="cuda")
    dm = 1e-3 * torch.randn(batch, heads, device="cuda")
    dJ = 1e-3 * torch.randn(batch, heads, rank, rank, device="cuda")
    dD = 1e-3 * torch.randn_like(dJ)

    actual, actual_state = solvedelta_fused(
        *native_inputs,
        initial_state=native_initial,
        output_final_state=True,
        outer_dtype=outer_dtype,
    )
    actual_loss = (actual.float() * do).sum()
    actual_loss = actual_loss + (actual_state.S.float() * dS).sum()
    actual_loss = actual_loss + (actual_state.m * dm).sum()
    actual_loss = actual_loss + (actual_state.J * dJ).sum()
    actual_loss = actual_loss + (actual_state.D * dD).sum()
    actual_loss.backward()

    expected, expected_state = solvedelta_reference(
        *reference_inputs, initial_state=reference_initial
    )
    expected_loss = (expected * do.double()).sum()
    expected_loss = expected_loss + (expected_state.S * dS.double()).sum()
    expected_loss = expected_loss + (expected_state.m * dm.double()).sum()
    expected_loss = expected_loss + (expected_state.J * dJ.double()).sum()
    expected_loss = expected_loss + (expected_state.D * dD.double()).sum()
    expected_loss.backward()

    for index, (reference, native) in enumerate(zip(reference_inputs, native_inputs)):
        assert torch.isfinite(native.grad).all()
        ceiling = main_ceiling if index in (2, 3, 4, 6) else geometry_ceiling
        assert _rho(reference.grad, native.grad) < ceiling
    for reference, native in zip(reference_initial, native_initial):
        assert native.grad is not None
        assert torch.isfinite(native.grad).all()
        assert _rho(reference.grad, native.grad) < geometry_ceiling


def test_skewless_c32_fused_state_vjp_matches_fp64_contract() -> None:
    torch.manual_seed(20260823)
    batch, length, heads, rank, value_dim = 1, 33, 1, 128, 16
    master = (
        torch.randn(batch, length, heads, rank, device="cuda"),
        0.2 * torch.randn(batch, length, heads, rank, device="cuda"),
        torch.randn(batch, length, heads, rank, device="cuda"),
        torch.randn(batch, length, heads, 1, rank, device="cuda"),
        0.1 * torch.randn(batch, length, heads, 1, value_dim, device="cuda"),
        -0.05 * torch.rand(batch, length, heads, device="cuda"),
        -0.03 * torch.rand(batch, length, heads, rank, device="cuda"),
        2.0 * torch.rand(batch, length, heads, 1, rank, device="cuda"),
        2.0 * torch.rand(batch, length, heads, 1, value_dim, device="cuda"),
        2.0 * torch.rand(batch, length, heads, 1, device="cuda") - 1.0,
        torch.sigmoid(torch.randn(heads, device="cuda")),
    )
    native_inputs = tuple(x.detach().requires_grad_(True) for x in master)
    reference_inputs = tuple(
        x.detach().double().requires_grad_(True) for x in master
    )
    initial_master = SolveDeltaState(
        4.0 + torch.rand(batch, heads, device="cuda"),
        0.02 * torch.randn(batch, heads, rank, rank, device="cuda"),
        0.02 * torch.randn(batch, heads, rank, rank, device="cuda"),
        0.03 * torch.randn(batch, heads, rank, value_dim, device="cuda"),
    )
    native_initial = SolveDeltaState(
        *(value.detach().requires_grad_(True) for value in initial_master)
    )
    reference_initial = SolveDeltaState(
        *(value.detach().double().requires_grad_(True) for value in initial_master)
    )
    output_cotangent = torch.randn(
        batch, length, heads, value_dim, device="cuda"
    )
    state_cotangents = (
        1e-2 * torch.randn(batch, heads, device="cuda"),
        1e-3 * torch.randn(batch, heads, rank, rank, device="cuda"),
        1e-3 * torch.randn(batch, heads, rank, rank, device="cuda"),
        torch.randn(batch, heads, rank, value_dim, device="cuda"),
    )

    actual, actual_state = solvedelta_fused(
        *native_inputs,
        initial_state=native_initial,
        output_final_state=True,
        outer_dtype=torch.bfloat16,
        skew_enabled=False,
    )
    actual_loss = (actual.float() * output_cotangent).sum()
    actual_loss = actual_loss + sum(
        (value.float() * cotangent).sum()
        for value, cotangent in zip(actual_state, state_cotangents)
    )
    actual_loss.backward()

    reference_arguments = list(reference_inputs)
    reference_arguments[9] = torch.zeros_like(reference_arguments[9])
    expected, expected_state = solvedelta_reference(
        *reference_arguments, initial_state=reference_initial
    )
    expected_loss = (expected * output_cotangent.double()).sum()
    expected_loss = expected_loss + sum(
        (value * cotangent.double()).sum()
        for value, cotangent in zip(expected_state, state_cotangents)
    )
    expected_loss.backward()

    assert _rho(expected, actual) < 6e-3
    for index, (reference_state, native_state) in enumerate(
        zip(expected_state, actual_state)
    ):
        ceiling = 6e-3 if index == 3 else 2e-4
        assert _rho(reference_state, native_state) < ceiling
    for index, (reference, native) in enumerate(
        zip(reference_inputs, native_inputs)
    ):
        if index == 9:
            assert native.grad is None
            continue
        assert native.grad is not None and torch.isfinite(native.grad).all()
        ceiling = 1.5e-2 if index in (2, 3, 4, 6) else 2.5e-2
        assert _rho(reference.grad, native.grad) < ceiling
    for reference, native in zip(reference_initial, native_initial):
        assert native.grad is not None and torch.isfinite(native.grad).all()
        assert _rho(reference.grad, native.grad) < 2.5e-2


def test_fused_backward_is_repeatable() -> None:
    """Guard the chunk/WY and shared-memory boundaries against CUDA races."""
    torch.manual_seed(20260820)
    batch, length, heads, rank, value_dim = 1, 3, 1, 128, 8
    master = (
        torch.randn(batch, length, heads, rank, device="cuda"),
        0.2 * torch.randn(batch, length, heads, rank, device="cuda"),
        torch.randn(batch, length, heads, rank, device="cuda"),
        torch.randn(batch, length, heads, 1, rank, device="cuda"),
        0.1 * torch.randn(batch, length, heads, 1, value_dim, device="cuda"),
        -0.05 * torch.rand(batch, length, heads, device="cuda"),
        -0.03 * torch.rand(batch, length, heads, rank, device="cuda"),
        2 * torch.rand(batch, length, heads, 1, rank, device="cuda"),
        2 * torch.rand(batch, length, heads, 1, value_dim, device="cuda"),
        2 * torch.rand(batch, length, heads, 1, device="cuda") - 1,
        torch.sigmoid(torch.randn(heads, device="cuda")),
    )

    def run_once() -> tuple[torch.Tensor, ...]:
        inputs = tuple(x.detach().requires_grad_(True) for x in master)
        output, _ = solvedelta_fused(*inputs)
        output.float().square().sum().backward()
        return tuple(x.grad.detach().clone() for x in inputs)

    first = run_once()
    second = run_once()
    for left, right in zip(first, second):
        assert torch.equal(left, right)
