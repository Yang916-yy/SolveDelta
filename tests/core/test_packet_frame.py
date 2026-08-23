import pytest
import torch
import torch.nn.functional as F

import causallsso.ops.packet_frame as packet_ops
from causallsso import (
    apply_dual_reference,
    apply_primal_reference,
    bounded_ldu_reference,
)
from causallsso.ops import (
    cuda_chunk_solve_frame128,
    mathdx_available,
    packet_frame128,
    triton_geometry_chunk_scan,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not mathdx_available(),
    reason="CUDA and built native extension required",
)


def _rho(reference: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    error = (actual.double() - reference.double()).square().mean().sqrt()
    scale = reference.double().square().mean().sqrt()
    return error / (scale + 1e-8)


def _random_inputs(length: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260822 + length)
    batch, heads, rank = 1, 2, 128
    u = F.normalize(torch.randn(batch, length, heads, rank, device="cuda"), dim=-1)
    h = 0.2 * torch.randn_like(u)
    log_decay = -0.05 * torch.rand(batch, length, heads, device="cuda")
    keys = F.normalize(
        torch.randn(batch, length, heads, 1, rank, device="cuda"), dim=-1
    )
    erase = 2.0 * torch.rand_like(keys)
    query = F.normalize(torch.randn_like(u), dim=-1)
    skew = 2.0 * torch.rand(batch, length, heads, 1, device="cuda") - 1.0
    strength = torch.sigmoid(torch.randn(heads, device="cuda"))
    return u, h, log_decay, keys, erase, query, skew, strength


def _cancellation_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1947)
    rank, length = 128, 16
    left = F.normalize(torch.randn(rank, device="cuda", dtype=torch.float64), dim=0)
    right = F.normalize(torch.randn(rank, device="cuda", dtype=torch.float64), dim=0)
    boundary_m = torch.ones(1, 1, 1, device="cuda")
    boundary_J = torch.zeros(1, 1, 1, rank, rank, device="cuda")
    boundary_D = (4096.0 * torch.outer(left, right)).float()[None, None, None]
    u = torch.zeros(1, length, 1, rank, device="cuda")
    h = torch.zeros_like(u)
    u[:, 0, 0] = left.float()
    h[:, 0, 0] = (-4096.0 + 0.01) * right.float()
    u[:, 1:, 0] = F.normalize(
        torch.randn(1, length - 1, rank, device="cuda"), dim=-1
    )
    log_decay = torch.zeros(1, length, 1, device="cuda")
    keys = F.normalize(torch.randn(1, length, 1, 1, rank, device="cuda"), dim=-1)
    erase = 0.2 + 1.6 * torch.rand_like(keys)
    query = F.normalize(torch.randn_like(u), dim=-1)
    skew = -0.8 + 1.6 * torch.rand(1, length, 1, 1, device="cuda")
    strength = torch.ones(1, device="cuda")
    return (
        boundary_m,
        boundary_J,
        boundary_D,
        u,
        h,
        log_decay,
        keys,
        erase,
        query,
        skew,
        strength,
    )


def _packet_frame_fp64(
    boundary_m: torch.Tensor,
    boundary_J: torch.Tensor,
    boundary_D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    log_decay: torch.Tensor,
    keys: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    skew: torch.Tensor,
    strength: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, length, _, rank = u.shape
    outputs_d, outputs_e, outputs_chi = [], [], []
    mass = moment_J = moment_D = None
    for token in range(length):
        chunk = token // 16
        if token % 16 == 0:
            mass = boundary_m[:, :, chunk]
            moment_J = boundary_J[:, :, chunk]
            moment_D = boundary_D[:, :, chunk]
        decay = torch.exp(log_decay[:, token])
        token_u = u[:, token]
        token_h = h[:, token]
        mass = decay * mass + 1.0
        moment_J = (
            decay[..., None, None] * moment_J
            + token_u[..., :, None] * token_u[..., None, :]
        )
        moment_D = (
            decay[..., None, None] * moment_D
            + token_u[..., :, None] * token_h[..., None, :]
        )
        lower, diagonal, upper, omega = bounded_ldu_reference(
            moment_J / mass[..., None, None],
            moment_D / mass[..., None, None],
            strength,
        )
        key = keys[:, token, :, 0]
        base_erase = erase[:, token, :, 0] * key
        tau = (base_erase * key).sum(-1)
        omega_key = (omega @ key.unsqueeze(-1)).squeeze(-1)
        direction = omega_key / torch.sqrt(
            1.0 + omega_key.square().sum(-1, keepdim=True)
        )
        dual_rhs = base_erase + (
            tau * (2.0 - tau) * skew[:, token, :, 0]
        )[..., None] * direction
        outputs_d.append(
            apply_primal_reference(lower, diagonal, upper, key).unsqueeze(-2)
        )
        outputs_e.append(
            apply_dual_reference(lower, diagonal, upper, dual_rhs).unsqueeze(-2)
        )
        outputs_chi.append(
            apply_dual_reference(lower, diagonal, upper, query[:, token])
        )
    assert rank == 128
    return (
        torch.stack(outputs_d, dim=1),
        torch.stack(outputs_e, dim=1),
        torch.stack(outputs_chi, dim=1),
    )


def _packet_vjp(
    inputs: tuple[torch.Tensor, ...],
    output_grads: tuple[torch.Tensor, ...],
    *,
    fp64: bool,
) -> tuple[torch.Tensor, ...]:
    dtype = torch.float64 if fp64 else torch.float32
    leaves = tuple(
        value.detach().to(dtype).clone().requires_grad_(True) for value in inputs
    )
    outputs = _packet_frame_fp64(*leaves) if fp64 else packet_frame128(*leaves)
    return torch.autograd.grad(
        outputs,
        leaves,
        tuple(value.to(dtype) for value in output_grads),
    )


def _assert_vjp_contract(
    expected: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...],
    *,
    ceiling: float = 1e-3,
) -> None:
    for reference, value in zip(expected, actual):
        assert torch.isfinite(value).all()
        max_error = (reference - value.double()).abs().max()
        assert max_error <= 1e-6 or _rho(reference, value) < ceiling


def _general_boundary_inputs(length: int) -> tuple[torch.Tensor, ...]:
    u, h, log_decay, keys, erase, query, skew, strength = _random_inputs(length)
    batch, _, heads, rank = u.shape
    chunks = (length + 15) // 16
    torch.manual_seed(303100 + length)
    boundary_m = 0.5 + torch.rand(batch, heads, chunks, device="cuda")
    boundary_J = 0.1 * torch.randn(
        batch, heads, chunks, rank, rank, device="cuda"
    )
    boundary_D = 0.1 * torch.randn_like(boundary_J)
    return (
        boundary_m,
        boundary_J,
        boundary_D,
        u,
        h,
        log_decay,
        keys,
        erase,
        query,
        skew,
        strength,
    )


def _chart_parameter_oracle(
    boundary_m: torch.Tensor,
    boundary_J: torch.Tensor,
    boundary_D: torch.Tensor,
    packed_u: torch.Tensor,
    packed_h: torch.Tensor,
    packed_log_decay: torch.Tensor,
    strength: torch.Tensor,
    *,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    programs, chunk_size, rank = packed_u.shape
    heads = strength.numel()
    chunks = boundary_m.shape[2]
    coefficient = torch.zeros(
        programs, chunk_size, 4, device="cuda", dtype=torch.float64
    )
    diagonal = torch.ones(
        programs, chunk_size, rank, device="cuda", dtype=torch.float64
    )
    eye = torch.eye(rank, device="cuda", dtype=torch.float64)
    radius = 1.0 / 8.0
    flat_m = boundary_m.reshape(programs).double()
    flat_J = boundary_J.reshape(programs, rank, rank).double()
    flat_D = boundary_D.reshape(programs, rank, rank).double()
    for system in range(programs):
        chunk = system % chunks
        head = (system // chunks) % heads
        mass = flat_m[system]
        moment_J = flat_J[system]
        moment_D = flat_D[system]
        geometry_strength = strength[head].double()
        for local_token in range(chunk_size):
            if chunk * chunk_size + local_token >= length:
                continue
            decay = torch.exp(packed_log_decay[system, local_token].double())
            token_u = packed_u[system, local_token].double()
            token_h = packed_h[system, local_token].double()
            mass = decay * mass + 1.0
            moment_J = decay * moment_J + torch.outer(token_u, token_u)
            moment_D = decay * moment_D + torch.outer(token_u, token_h)
            x_h = geometry_strength * (moment_J / mass - eye / rank)
            x_r = geometry_strength * moment_D / mass
            norm_sq = torch.stack(
                (
                    torch.tril(x_h, diagonal=-1).square().sum(),
                    torch.tril(x_r, diagonal=-1).square().sum(),
                    torch.triu(x_h, diagonal=1).square().sum(),
                    torch.triu(x_r, diagonal=1).square().sum(),
                )
            )
            coefficient[system, local_token] = (
                geometry_strength
                * radius
                * torch.rsqrt(radius * radius + norm_sq)
            )
            diagonal[system, local_token] = torch.exp(
                radius * torch.tanh(torch.diagonal(x_h) / radius)
                + radius * torch.tanh(torch.diagonal(x_r) / radius)
            )
    return coefficient, diagonal


def _affine_radial_oracle(
    boundary_J: torch.Tensor,
    boundary_D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    alpha: torch.Tensor,
    weights: torch.Tensor,
    strength: torch.Tensor,
    *,
    length: int,
) -> tuple[torch.Tensor, ...]:
    panels, chunk_size, rank = u.shape
    radius = 1.0 / 8.0
    coefficient = torch.zeros(
        panels, chunk_size, 4, device="cuda", dtype=torch.float64
    )
    diagonal = torch.ones(
        panels, chunk_size, rank, device="cuda", dtype=torch.float64
    )
    norm_sq = torch.zeros_like(coefficient)
    diagonal_h = torch.zeros_like(diagonal)
    diagonal_r = torch.zeros_like(diagonal)
    boundary_J, boundary_D, u, h, alpha, weights, strength = (
        value.double()
        for value in (boundary_J, boundary_D, u, h, alpha, weights, strength)
    )
    for target in range(length):
        state_J = alpha[:, target, None, None] * boundary_J
        state_J = state_J + torch.einsum(
            "ps,psi,psj->pij", weights[:, :, target], u, u
        )
        state_D = alpha[:, target, None, None] * boundary_D
        state_D = state_D + torch.einsum(
            "ps,psi,psj->pij", weights[:, :, target], u, h
        )
        g = strength[:, None, None]
        x_h = g * state_J
        x_r = g * state_D
        norm_sq[:, target] = torch.stack(
            (
                torch.tril(x_h, -1).square().sum((-2, -1)),
                torch.tril(x_r, -1).square().sum((-2, -1)),
                torch.triu(x_h, 1).square().sum((-2, -1)),
                torch.triu(x_r, 1).square().sum((-2, -1)),
            ),
            dim=-1,
        )
        coefficient[:, target] = (
            strength[:, None]
            * radius
            * torch.rsqrt(radius * radius + norm_sq[:, target])
        )
        diagonal_h[:, target] = torch.diagonal(state_J, dim1=-2, dim2=-1)
        diagonal_h[:, target] -= 1.0 / rank
        diagonal_r[:, target] = torch.diagonal(state_D, dim1=-2, dim2=-1)
        diagonal[:, target] = torch.exp(
            radius * torch.tanh(strength[:, None] * diagonal_h[:, target] / radius)
            + radius * torch.tanh(strength[:, None] * diagonal_r[:, target] / radius)
        )
    return coefficient, diagonal, norm_sq, diagonal_h, diagonal_r


def _raw_cancellation_parameters(moment: str, *, length: int = 7):
    torch.manual_seed(616100 + (moment == "D"))
    panels, chunk_size, rank = 1, 16, 128
    u = F.normalize(torch.randn(panels, chunk_size, rank, device="cuda"), dim=-1)
    h = 0.2 * torch.randn_like(u)
    u[:, length:] = 0.0
    h[:, length:] = 0.0
    scale = float(2**12)
    boundary_J = 0.02 * torch.randn(panels, rank, rank, device="cuda")
    boundary_D = 0.02 * torch.randn_like(boundary_J)
    alpha = torch.zeros(panels, chunk_size, device="cuda")
    alpha[:, :length] = 1.0
    weights = torch.zeros(panels, chunk_size, chunk_size, device="cuda")
    if moment == "J":
        boundary_J -= scale * u[:, 0, :, None] * u[:, 0, None, :]
        h[:, 0] = 0.0
        weights[:, 0, :length] = scale
    else:
        right = F.normalize(torch.randn(panels, rank, device="cuda"), dim=-1)
        boundary_D -= scale * u[:, 0, :, None] * right[:, None, :]
        h[:, 0] = scale * right
        weights[:, 0, :length] = 1.0
    causal = (
        torch.arange(chunk_size, device="cuda")[:, None]
        <= torch.arange(chunk_size, device="cuda")[None, :]
    )
    weights[:, 1:length, :length] += (
        0.05
        * torch.rand_like(weights[:, 1:length, :length])
        * causal[1:length, :length]
    )
    strength = torch.tensor([0.7], device="cuda")
    return tuple(
        value.contiguous()
        for value in (
            boundary_J,
            boundary_D,
            u,
            h,
            alpha,
            weights,
            strength,
        )
    )


@pytest.mark.parametrize("length", [7, 16, 17, 65])
def test_packet_frame_matches_independent_native_path(length: int) -> None:
    inputs = _random_inputs(length)
    u, h, log_decay, keys, erase, query, skew, strength = inputs
    boundaries, _ = triton_geometry_chunk_scan(
        u, h, log_decay, chunk_size=16, input_precision="ieee"
    )
    actual = packet_frame128(
        boundaries.m,
        boundaries.J,
        boundaries.D,
        u,
        h,
        log_decay,
        keys,
        erase,
        query,
        skew,
        strength,
    )
    expected = cuda_chunk_solve_frame128(
        boundaries.m,
        boundaries.J,
        boundaries.D,
        u,
        h,
        log_decay,
        keys,
        erase,
        query,
        skew,
        strength,
        chunk_size=16,
    )
    for reference, value in zip(expected, actual):
        assert torch.isfinite(value).all()
        assert _rho(reference, value) < 5e-4


@pytest.mark.parametrize("case", ["random", "cancellation", "underflow"])
def test_packet_chart_parameter_split_matches_fp64_oracle(case: str) -> None:
    if case in ("random", "underflow"):
        u, h, log_decay, keys, erase, query, skew, strength = _random_inputs(17)
        if case == "underflow":
            log_decay.fill_(-1000.0)
        boundaries, _ = triton_geometry_chunk_scan(
            u, h, log_decay, chunk_size=16, input_precision="ieee"
        )
        boundary_m, boundary_J, boundary_D = (
            boundaries.m,
            boundaries.J,
            boundaries.D,
        )
    else:
        (
            boundary_m,
            boundary_J,
            boundary_D,
            u,
            h,
            log_decay,
            keys,
            erase,
            query,
            skew,
            strength,
        ) = _cancellation_inputs()

    packed = packet_ops._pack_frame_inputs(
        u,
        h,
        keys.squeeze(-2),
        erase.squeeze(-2),
        query,
        log_decay,
        skew.squeeze(-1),
    )
    packed_u, packed_h, _, _, _, packed_log_decay, _ = packed
    alpha, _, coefficient, diagonal = packet_ops._packet_parameters(
        boundary_m,
        boundary_J,
        boundary_D,
        packed_u,
        packed_h,
        packed_log_decay,
        strength,
        length=u.shape[1],
        heads=u.shape[2],
    )
    reference_coefficient, reference_diagonal = _chart_parameter_oracle(
        boundary_m,
        boundary_J,
        boundary_D,
        packed_u,
        packed_h,
        packed_log_decay,
        strength,
        length=u.shape[1],
    )
    torch.testing.assert_close(
        coefficient.double(), reference_coefficient, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        diagonal.double(), reference_diagonal, rtol=1e-5, atol=5e-6
    )
    if case == "underflow":
        chunks = boundary_m.shape[2]
        panel_chunks = torch.arange(alpha.shape[0], device="cuda") % chunks
        valid = (
            panel_chunks[:, None] * alpha.shape[1]
            + torch.arange(alpha.shape[1], device="cuda")[None]
            < u.shape[1]
        )
        assert torch.count_nonzero(alpha.masked_select(valid)) == 0
        assert torch.count_nonzero(coefficient.masked_select(valid[..., None])) > 0


@pytest.mark.parametrize("moment", ["J", "D"])
def test_packet_radial_forward_handles_general_moment_cancellation(moment: str) -> None:
    length = 7
    inputs = _raw_cancellation_parameters(moment, length=length)
    actual = torch.ops.causallsso.packet_frame_radial_forward128(
        *inputs, 1, 1, length
    )
    repeated = torch.ops.causallsso.packet_frame_radial_forward128(
        *inputs, 1, 1, length
    )
    expected = _affine_radial_oracle(*inputs, length=length)
    for reference, value, repeated_value in zip(expected, actual, repeated):
        assert torch.isfinite(value).all()
        assert _rho(reference, value) < 2e-5
        torch.testing.assert_close(value, repeated_value, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual[0][:, length:], torch.zeros_like(actual[0][:, length:])
    )
    torch.testing.assert_close(
        actual[2][:, length:], torch.zeros_like(actual[2][:, length:])
    )
    torch.testing.assert_close(
        actual[3][:, length:], torch.zeros_like(actual[3][:, length:])
    )
    torch.testing.assert_close(
        actual[4][:, length:], torch.zeros_like(actual[4][:, length:])
    )
    torch.testing.assert_close(
        actual[1][:, length:], torch.ones_like(actual[1][:, length:])
    )


def test_packet_radial_forward_identity_keeps_base_diagonal_auxiliaries() -> None:
    inputs = list(_raw_cancellation_parameters("D"))
    inputs[-1] = torch.zeros_like(inputs[-1])
    coefficient, diagonal, norm_sq, diagonal_h, diagonal_r = (
        torch.ops.causallsso.packet_frame_radial_forward128(*inputs, 1, 1, 7)
    )
    assert torch.count_nonzero(coefficient) == 0
    assert torch.count_nonzero(norm_sq) == 0
    torch.testing.assert_close(diagonal, torch.ones_like(diagonal), rtol=0.0, atol=0.0)
    assert torch.count_nonzero(diagonal_h[:, :7]) > 0
    assert torch.count_nonzero(diagonal_r[:, :7]) > 0


def test_packet_frame_identity_geometry_is_exact_reduction() -> None:
    inputs = _random_inputs(17)
    u, h, log_decay, keys, erase, query, skew, strength = inputs
    boundaries, _ = triton_geometry_chunk_scan(
        u, h, log_decay, chunk_size=16, input_precision="ieee"
    )
    d, e, chi = packet_frame128(
        boundaries.m,
        boundaries.J,
        boundaries.D,
        u,
        h,
        log_decay,
        keys,
        erase,
        query,
        skew,
        torch.zeros_like(strength),
    )
    torch.testing.assert_close(d, keys, rtol=0.0, atol=0.0)
    torch.testing.assert_close(e, erase * keys, rtol=0.0, atol=2e-7)
    torch.testing.assert_close(chi, query, rtol=0.0, atol=0.0)


def test_packet_frame_legal_cancellation_contract() -> None:
    (
        boundary_m,
        boundary_J,
        boundary_D,
        u,
        h,
        log_decay,
        keys,
        erase,
        query,
        skew,
        strength,
    ) = _cancellation_inputs()
    rank, length = 128, 16

    actual = packet_frame128(
        boundary_m,
        boundary_J,
        boundary_D,
        u,
        h,
        log_decay,
        keys,
        erase,
        query,
        skew,
        strength,
    )

    mass = boundary_m[:, :, 0].double()
    moment_J = boundary_J[:, :, 0].double()
    moment_D = boundary_D[:, :, 0].double()
    expected_d, expected_e, expected_chi = [], [], []
    residuals, pairings = [], []
    for token in range(length):
        decay = torch.exp(log_decay[:, token].double())
        mass = decay * mass + 1.0
        token_u = u[:, token].double()
        token_h = h[:, token].double()
        moment_J = decay[..., None, None] * moment_J
        moment_J = moment_J + token_u[..., :, None] * token_u[..., None, :]
        moment_D = decay[..., None, None] * moment_D
        moment_D = moment_D + token_u[..., :, None] * token_h[..., None, :]
        lower, diagonal, upper, omega = bounded_ldu_reference(
            moment_J / mass[..., None, None],
            moment_D / mass[..., None, None],
            strength.double(),
        )
        key = keys[:, token, :, 0].double()
        base_erase = erase[:, token, :, 0].double() * key
        tau = (base_erase * key).sum(-1)
        omega_key = (omega @ key.unsqueeze(-1)).squeeze(-1)
        direction = omega_key / torch.sqrt(
            1.0 + omega_key.square().sum(-1, keepdim=True)
        )
        dual_rhs = base_erase + (
            tau * (2.0 - tau) * skew[:, token, :, 0].double()
        )[..., None] * direction
        token_d = apply_primal_reference(lower, diagonal, upper, key)
        token_e = apply_dual_reference(lower, diagonal, upper, dual_rhs)
        token_chi = apply_dual_reference(
            lower, diagonal, upper, query[:, token].double()
        )
        expected_d.append(token_d[:, :, None])
        expected_e.append(token_e[:, :, None])
        expected_chi.append(token_chi)

        matrix = lower @ (diagonal[..., :, None] * upper)
        actual_d = actual[0][:, token, :, 0].double()
        solve_error = (matrix @ actual_d.unsqueeze(-1)).squeeze(-1) - key
        residuals.append(
            solve_error.norm(dim=-1)
            / (
                torch.linalg.matrix_norm(matrix, ord=2) * actual_d.norm(dim=-1)
                + key.norm(dim=-1)
                + 1e-12
            )
        )
        actual_e = actual[1][:, token, :, 0].double()
        pairing_error = (actual_e * actual_d).sum(-1) - (dual_rhs * key).sum(-1)
        pairings.append(
            pairing_error.abs()
            / (
                actual_e.norm(dim=-1) * actual_d.norm(dim=-1)
                + dual_rhs.norm(dim=-1) * key.norm(dim=-1)
                + 1e-12
            )
        )

    expected = (
        torch.stack(expected_d, dim=1),
        torch.stack(expected_e, dim=1),
        torch.stack(expected_chi, dim=1),
    )
    for reference, value in zip(expected, actual):
        assert _rho(reference, value) < 5e-4
    assert torch.stack(residuals).max() < 2e-5
    assert torch.stack(pairings).max() < 5e-5


@pytest.mark.parametrize("length", [7, 17, 65])
def test_packet_frame_backward_matches_fp64_for_general_boundaries(length: int) -> None:
    inputs = _general_boundary_inputs(length)
    torch.manual_seed(404100 + length)
    output_grads = (
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[8]),
    )
    expected = _packet_vjp(inputs, output_grads, fp64=True)
    actual = _packet_vjp(inputs, output_grads, fp64=False)
    _assert_vjp_contract(expected, actual)


def test_packet_frame_backward_identity_geometry_contract() -> None:
    inputs = list(_general_boundary_inputs(17))
    inputs[-1] = torch.zeros_like(inputs[-1])
    torch.manual_seed(505100)
    output_grads = (
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[8]),
    )
    expected = _packet_vjp(tuple(inputs), output_grads, fp64=True)
    actual = _packet_vjp(tuple(inputs), output_grads, fp64=False)
    _assert_vjp_contract(expected, actual)
    for gradient in actual[:6]:
        assert torch.count_nonzero(gradient) == 0


def test_packet_frame_backward_reduces_shared_strength_across_batch_and_chunks() -> None:
    single = _general_boundary_inputs(17)
    batched = tuple(
        value if index == 10 else value.expand(2, *value.shape[1:]).clone()
        for index, value in enumerate(single)
    )
    torch.manual_seed(515100)
    output_grads = (
        torch.randn_like(batched[6]),
        torch.randn_like(batched[6]),
        torch.randn_like(batched[8]),
    )
    expected = _packet_vjp(batched, output_grads, fp64=True)
    actual = _packet_vjp(batched, output_grads, fp64=False)
    _assert_vjp_contract(expected, actual)


def test_packet_frame_backward_padded_tail_is_an_exact_noop() -> None:
    length = 7
    short_inputs = _general_boundary_inputs(length)
    padded_inputs = list(short_inputs)
    for index in range(3, 10):
        value = short_inputs[index]
        padded = value.new_zeros(value.shape[0], 16, *value.shape[2:])
        padded[:, :length] = value
        padded_inputs[index] = padded
    torch.manual_seed(525100)
    short_output_grads = (
        torch.randn_like(short_inputs[6]),
        torch.randn_like(short_inputs[6]),
        torch.randn_like(short_inputs[8]),
    )
    padded_output_grads = []
    for value in short_output_grads:
        padded = value.new_zeros(value.shape[0], 16, *value.shape[2:])
        padded[:, :length] = value
        padded_output_grads.append(padded)
    short = _packet_vjp(short_inputs, short_output_grads, fp64=False)
    padded = _packet_vjp(
        tuple(padded_inputs), tuple(padded_output_grads), fp64=False
    )
    for index in (0, 1, 2, 10):
        torch.testing.assert_close(short[index], padded[index], rtol=0.0, atol=0.0)
    for index in range(3, 10):
        torch.testing.assert_close(
            short[index], padded[index][:, :length], rtol=0.0, atol=0.0
        )
        assert torch.count_nonzero(padded[index][:, length:]) == 0


@pytest.mark.parametrize("log_decay_value", [-110.0, -1000.0])
def test_packet_frame_backward_handles_decay_underflow(log_decay_value: float) -> None:
    inputs = list(_general_boundary_inputs(17))
    inputs[5] = torch.full_like(inputs[5], log_decay_value)
    torch.manual_seed(535100)
    output_grads = (
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[8]),
    )
    expected = _packet_vjp(tuple(inputs), output_grads, fp64=True)
    actual = _packet_vjp(tuple(inputs), output_grads, fp64=False)
    _assert_vjp_contract(expected, actual)


def test_packet_frame_backward_legal_driven_cancellation_contract() -> None:
    inputs = _cancellation_inputs()
    torch.manual_seed(606100)
    output_grads = (
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[8]),
    )
    expected = _packet_vjp(inputs, output_grads, fp64=True)
    actual = _packet_vjp(inputs, output_grads, fp64=False)
    _assert_vjp_contract(expected, actual)


def test_packet_frame_backward_is_repeatable() -> None:
    inputs = _general_boundary_inputs(17)
    torch.manual_seed(707100)
    output_grads = (
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[6]),
        torch.randn_like(inputs[8]),
    )
    first = _packet_vjp(inputs, output_grads, fp64=False)
    second = _packet_vjp(inputs, output_grads, fp64=False)
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
