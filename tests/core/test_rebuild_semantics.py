from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from causallsso import SolveDelta, SolveDeltaConfig
from causallsso.ops.operator import solvedelta_native
from causallsso.ops.packing import build_packed_segments
from causallsso.ops.radial import strict_gram
from causallsso.ops.resident_frame import (
    resident_factor_direct,
    resident_factor_transpose,
    resident_primal,
)
from causallsso.reference import SolveDeltaState, solvedelta_reference


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native SolveDelta requires CUDA"
)


def _leaf(shape, *, scale=1.0, dtype=torch.bfloat16):
    return (scale * torch.randn(*shape, device="cuda")).to(dtype).requires_grad_()


def _inputs(*, length=17, edits=1, width=32, value_width=16):
    batch, heads = 1, 1
    return (
        _leaf((batch, length, heads, width)),
        _leaf((batch, length, heads, width), scale=0.2),
        _leaf((batch, length, heads, width)),
        _leaf((batch, length, heads, edits, width)),
        _leaf((batch, length, heads, edits, value_width), scale=0.2),
        (-0.02 * torch.rand(batch, length, heads, device="cuda")).requires_grad_(),
        (-0.02 * torch.rand(batch, length, heads, width, device="cuda")).requires_grad_(),
        _leaf((batch, length, heads, edits, width), scale=0.5),
        _leaf((batch, length, heads, edits, value_width), scale=0.5),
        torch.full((heads,), 0.1, device="cuda", requires_grad=True),
    )


def _reference_inputs_from_raw(args, *, requires_grad: bool):
    leaves = [value.detach().double().requires_grad_(requires_grad) for value in args]
    operator_args = list(leaves)
    operator_args[7] = 2.0 * torch.sigmoid(leaves[7])
    operator_args[8] = 2.0 * torch.sigmoid(leaves[8])
    return operator_args, leaves


def _state(width=32, value_width=16):
    m = torch.full((1, 1), 0.5, device="cuda", requires_grad=True)
    raw = 0.01 * torch.randn(1, 1, width, width, device="cuda")
    j = (0.5 * (raw + raw.transpose(-1, -2))).detach().requires_grad_()
    d = (0.01 * torch.randn(1, 1, width, width, device="cuda")).requires_grad_()
    s = (0.01 * torch.randn(1, 1, width, value_width, device="cuda")).requires_grad_()
    return SolveDeltaState(m, j, d, s)


@pytest.mark.parametrize("edits", [1, 2])
def test_composed_native_matches_fp64_token_oracle(edits):
    torch.manual_seed(100 + edits)
    args = _inputs(edits=edits)
    state0 = _state()
    output, state = solvedelta_native(
        *args,
        initial_state=state0,
        return_final_state=True,
        chunk_size=16,
    )

    reference_args, reference_leaves = _reference_inputs_from_raw(
        args, requires_grad=True
    )
    reference_state0 = SolveDeltaState(
        *(value.detach().double().requires_grad_() for value in state0)
    )
    expected_output, expected_state = solvedelta_reference(
        *reference_args, initial_state=reference_state0
    )

    assert (output.float() - expected_output.float()).abs().max() < 5e-3
    state_limits = (1e-5, 1e-3, 5e-3, 5e-3)
    for got, expected, limit in zip(state, expected_state, state_limits):
        assert (got - expected.float()).abs().max() < limit
    assert torch.equal(state.J, state.J.transpose(-1, -2))

    output_cotangent = torch.randn_like(output.float())
    state_cotangents = [torch.randn_like(value) for value in state]
    loss = (output.float() * output_cotangent).sum()
    expected_loss = (expected_output * output_cotangent.double()).sum()
    for got, expected, cotangent in zip(state, expected_state, state_cotangents):
        loss = loss + (got * cotangent).sum()
        expected_loss = expected_loss + (expected * cotangent.double()).sum()
    gradients = torch.autograd.grad(loss, (*args, *state0))
    expected_gradients = torch.autograd.grad(
        expected_loss, (*reference_leaves, *reference_state0)
    )
    for got, expected in zip(gradients, expected_gradients):
        assert torch.isfinite(got).all()
        assert (got.float() - expected.float()).abs().max() < 3e-2

def test_masks_resets_and_aligned_recurrent_split():
    torch.manual_seed(203)
    args = _inputs(length=32, edits=2)
    state0 = _state()
    valid = torch.ones(1, 32, dtype=torch.bool, device="cuda")
    valid[:, [3, 19]] = False
    reset = torch.zeros_like(valid)
    reset[:, 9] = True
    output, state = solvedelta_native(
        *args,
        initial_state=state0,
        valid_mask=valid,
        reset_mask=reset,
        return_final_state=True,
        chunk_size=16,
    )
    reference_args, _ = _reference_inputs_from_raw(args, requires_grad=False)
    reference_state = SolveDeltaState(*(value.detach().double() for value in state0))
    expected_output, expected_state = solvedelta_reference(
        *reference_args,
        initial_state=reference_state,
        valid_mask=valid,
        reset_mask=reset,
    )
    assert torch.equal(output[:, ~valid[0]], torch.zeros_like(output[:, ~valid[0]]))
    assert (output.float() - expected_output.float()).abs().max() < 5e-3
    for got, expected, limit in zip(state, expected_state, (1e-5, 1e-3, 5e-3, 5e-3)):
        assert (got - expected.float()).abs().max() < limit

    whole_output, whole_state = solvedelta_native(
        *args, initial_state=state0, return_final_state=True, chunk_size=16
    )
    first = [value[:, :16] if value.ndim > 1 and value.shape[1] == 32 else value for value in args]
    second = [value[:, 16:] if value.ndim > 1 and value.shape[1] == 32 else value for value in args]
    first_output, first_state = solvedelta_native(
        *first, initial_state=state0, return_final_state=True, chunk_size=16
    )
    second_output, second_state = solvedelta_native(
        *second, initial_state=first_state, return_final_state=True, chunk_size=16
    )
    assert torch.equal(whole_output, torch.cat((first_output, second_output), dim=1))
    for whole, split in zip(whole_state, second_state):
        assert torch.equal(whole, split)


def test_packed_multibatch_masks_resets_and_state_vjp():
    torch.manual_seed(229)
    batch, length, heads, edits, width, value_width = 3, 23, 2, 2, 32, 16

    def leaf(shape, *, scale=1.0, dtype=torch.bfloat16):
        return (scale * torch.randn(*shape, device="cuda")).to(dtype).requires_grad_()

    args = (
        leaf((batch, length, heads, width)),
        leaf((batch, length, heads, width), scale=0.2),
        leaf((batch, length, heads, width)),
        leaf((batch, length, heads, edits, width)),
        leaf((batch, length, heads, edits, value_width), scale=0.2),
        (-0.02 * torch.rand(batch, length, heads, device="cuda")).requires_grad_(),
        (-0.02 * torch.rand(batch, length, heads, width, device="cuda")).requires_grad_(),
        leaf((batch, length, heads, edits, width), scale=0.5),
        leaf((batch, length, heads, edits, value_width), scale=0.5),
        torch.full((heads,), 0.1, device="cuda", requires_grad=True),
    )
    m = torch.rand(batch, heads, device="cuda", requires_grad=True)
    raw_j = 0.01 * torch.randn(batch, heads, width, width, device="cuda")
    j = (0.5 * (raw_j + raw_j.transpose(-1, -2))).detach().requires_grad_()
    d = (0.01 * torch.randn(batch, heads, width, width, device="cuda")).requires_grad_()
    s = (
        0.01 * torch.randn(batch, heads, width, value_width, device="cuda")
    ).requires_grad_()
    state0 = SolveDeltaState(m, j, d, s)

    valid = torch.zeros(batch, length, dtype=torch.bool, device="cuda")
    valid[1] = True
    valid[1, [4, 11, 19]] = False
    valid[2, [0, 1, 2, 5, 6, 7, 9, 12, 13, 14, 18, 22]] = True
    reset = torch.zeros_like(valid)
    reset[0, 5] = True  # Invalid resets are semantic no-ops.
    reset[1, 0] = True
    reset[1, 4] = True
    reset[1, 13] = True
    reset[2, 2] = True
    reset[2, 7] = True
    reset[2, 18] = True

    output, state = solvedelta_native(
        *args,
        initial_state=state0,
        valid_mask=valid,
        reset_mask=reset,
        return_final_state=True,
        chunk_size=16,
    )
    reference_args, reference_leaves = _reference_inputs_from_raw(
        args, requires_grad=True
    )
    reference_state = SolveDeltaState(
        *(value.detach().double().requires_grad_() for value in state0)
    )
    expected_output, expected_state = solvedelta_reference(
        *reference_args,
        initial_state=reference_state,
        valid_mask=valid,
        reset_mask=reset,
    )
    assert torch.equal(output[~valid], torch.zeros_like(output[~valid]))
    assert (output.float() - expected_output.float()).abs().max() < 5e-3
    for got, expected, limit in zip(
        state, expected_state, (1e-5, 1e-3, 5e-3, 5e-3)
    ):
        assert (got - expected.float()).abs().max() < limit
    for got, initial in zip(state, state0):
        assert torch.equal(got[0], initial[0])

    output_cotangent = torch.randn_like(output.float())
    state_cotangents = [torch.randn_like(value) for value in state]
    loss = (output.float() * output_cotangent).sum()
    expected_loss = (expected_output * output_cotangent.double()).sum()
    for got, expected, cotangent in zip(state, expected_state, state_cotangents):
        loss = loss + (got * cotangent).sum()
        expected_loss = expected_loss + (expected * cotangent.double()).sum()
    gradients = torch.autograd.grad(loss, (*args, *state0))
    expected_gradients = torch.autograd.grad(
        expected_loss, (*reference_leaves, *reference_state)
    )
    for got, expected in zip(gradients, expected_gradients):
        assert torch.isfinite(got).all()
        assert (got.float() - expected.float()).abs().max() < 3e-2

    zero_output, zero_state = solvedelta_native(
        *args,
        valid_mask=valid,
        reset_mask=reset,
        return_final_state=True,
        chunk_size=16,
    )
    expected_zero_output, expected_zero_state = solvedelta_reference(
        *reference_args,
        valid_mask=valid,
        reset_mask=reset,
    )
    assert (zero_output.float() - expected_zero_output.float()).abs().max() < 5e-3
    for got, expected, limit in zip(
        zero_state, expected_zero_state, (1e-5, 1e-3, 5e-3, 5e-3)
    ):
        assert (got - expected.float()).abs().max() < limit


def test_packed_conv4_matches_token_recurrence_and_composed_vjp():
    torch.manual_seed(271)
    batch, length = 3, 11
    layer = SolveDelta(
        SolveDeltaConfig(
            hidden_size=32,
            num_heads=1,
            head_k_dim=32,
            head_v_dim=32,
        )
    ).cuda()
    layer.conv_weight.data = layer.conv_weight.data.to(torch.bfloat16)
    channels = layer.conv_weight.shape[0]
    x = torch.randn(
        batch, length, channels, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    initial = torch.randn(
        batch, channels, 3, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    valid = torch.tensor(
        [
            [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        ],
        dtype=torch.bool,
        device="cuda",
    )
    reset = torch.zeros_like(valid)
    reset[0, 3] = True
    reset[1, 4] = True  # Invalid reset is a no-op.
    reset[2, 6] = True
    plan = build_packed_segments(valid, reset)
    output, final = layer._packed_conv(
        x, initial, valid, reset, True, plan
    )

    ref_x = x.detach().float().requires_grad_()
    ref_initial = initial.detach().float().requires_grad_()
    ref_weight = layer.conv_weight.detach().float().requires_grad_()
    state = ref_initial
    expected_rows = []
    for token in range(length):
        is_valid = valid[:, token]
        is_reset = reset[:, token] & is_valid
        state = torch.where(
            is_reset[:, None, None], torch.zeros_like(state), state
        )
        window = torch.cat((state, ref_x[:, token, :, None]), dim=-1)
        candidate = F.silu((window * ref_weight[None]).sum(dim=-1))
        expected_rows.append(
            torch.where(is_valid[:, None], candidate, torch.zeros_like(candidate))
        )
        state = torch.where(
            is_valid[:, None, None], window[..., 1:], state
        )
    expected = torch.stack(expected_rows, dim=1)
    assert (output.float() - expected).abs().max() < 5e-3
    assert torch.equal(final, state.to(torch.bfloat16))

    output_cotangent = torch.randn_like(output.float())
    state_cotangent = torch.randn_like(final.float())
    gradients = torch.autograd.grad(
        (output.float() * output_cotangent).sum()
        + (final.float() * state_cotangent).sum(),
        (x, initial, layer.conv_weight),
    )
    expected_gradients = torch.autograd.grad(
        (expected * output_cotangent).sum()
        + (state * state_cotangent).sum(),
        (ref_x, ref_initial, ref_weight),
    )
    for got, expected_gradient in zip(gradients, expected_gradients):
        relative = (got.float() - expected_gradient).norm()
        relative = relative / expected_gradient.norm().clamp_min(1e-8)
        assert relative < 5e-3


def test_specialized_fused_decay_gate_matches_formula_and_composed_vjp():
    from causallsso.ops.gates import fused_decay_gate

    torch.manual_seed(277)
    storage = torch.randn(2, 17, 111, device="cuda", dtype=torch.bfloat16)
    raw = storage[..., 7:103].detach().requires_grad_()
    log_rate = torch.randn(96, device="cuda", requires_grad=True)
    bias = torch.randn(96, device="cuda", requires_grad=True)
    output = fused_decay_gate(raw, log_rate, bias)
    ref_raw = raw.detach().float().requires_grad_()
    ref_log_rate = log_rate.detach().requires_grad_()
    ref_bias = bias.detach().requires_grad_()
    expected = -ref_log_rate.exp() * F.softplus(ref_raw + ref_bias)
    assert (output - expected).abs().max() < 1e-5

    cotangent = torch.randn_like(output)
    gradients = torch.autograd.grad(
        (output * cotangent).sum(), (raw, log_rate, bias)
    )
    expected_gradients = torch.autograd.grad(
        (expected * cotangent).sum(),
        (ref_raw, ref_log_rate, ref_bias),
    )
    for index, (got, expected_gradient) in enumerate(zip(gradients, expected_gradients)):
        if index == 0:
            expected_gradient = expected_gradient.to(torch.bfloat16)
        relative = (got.float() - expected_gradient.float()).norm()
        relative = relative / expected_gradient.norm().clamp_min(1e-8)
        assert relative < 1e-5


def _delta_product_reference(args):
    u, h, q, keys, values, geometry_decay, associative_decay, erase, write, _ = args
    del u, h, geometry_decay
    keys = F.normalize(keys, dim=-1)
    q = F.normalize(q, dim=-1)
    batch, length, heads, _, width = keys.shape
    value_width = values.shape[-1]
    state = torch.zeros(
        batch, heads, width, value_width,
        dtype=keys.dtype, device=keys.device,
    )
    output = []
    for token in range(length):
        state = torch.exp(associative_decay[:, token])[..., None] * state
        for edit in range(keys.shape[3]):
            key = keys[:, token, :, edit]
            dual = erase[:, token, :, edit] * key
            target = write[:, token, :, edit] * values[:, token, :, edit]
            prediction = torch.einsum("bhrv,bhr->bhv", state, dual)
            state = state + key[..., None] * (target - prediction)[..., None, :]
        output.append(torch.einsum("bhrv,bhr->bhv", state, q[:, token]))
    return torch.stack(output, dim=1), state


@pytest.mark.parametrize("edits", [1, 2])
def test_identity_geometry_reduces_to_gdn2_and_delta_product(edits):
    torch.manual_seed(300 + edits)
    args = list(_inputs(length=7, edits=edits))
    args[-1] = torch.zeros_like(args[-1])
    reference_args, _ = _reference_inputs_from_raw(args, requires_grad=False)
    output, state = solvedelta_reference(*reference_args)
    expected_output, expected_s = _delta_product_reference(
        tuple(reference_args)
    )
    assert torch.equal(output, expected_output)
    assert torch.equal(state.S, expected_s)


def test_r_strict_gram_block_prefix_matches_same_packed_vjp():
    torch.manual_seed(401)
    panels, chunk_size, width = 2, 16, 64
    u = F.normalize(
        torch.randn(panels, chunk_size, width, device="cuda"), dim=-1
    ).to(torch.float16).requires_grad_()
    h = (0.2 * torch.randn(panels, chunk_size, width, device="cuda"))
    h = h.to(torch.bfloat16).requires_grad_()
    _, gram_lower, gram_upper = strict_gram(u, h)
    cotangents = (torch.randn_like(gram_lower), torch.randn_like(gram_upper))
    gradients = torch.autograd.grad(
        (gram_lower, gram_upper), (u, h), cotangents
    )

    ref_u = u.detach().to(torch.bfloat16).float().requires_grad_()
    ref_h = h.detach().float().requires_grad_()
    outer = ref_u[..., :, None] * ref_h[..., None, :]
    lower = torch.tril(outer, diagonal=-1).flatten(-2)
    upper = torch.triu(outer, diagonal=1).flatten(-2)
    expected_lower = lower @ lower.transpose(-1, -2)
    expected_upper = upper @ upper.transpose(-1, -2)
    expected_gradients = torch.autograd.grad(
        (expected_lower, expected_upper), (ref_u, ref_h), cotangents
    )

    for got, expected in zip(
        (gram_lower, gram_upper, *gradients),
        (expected_lower, expected_upper, *expected_gradients),
    ):
        assert torch.isfinite(got).all()
        relative = (got.float() - expected.float()).norm()
        relative = relative / expected.float().norm().clamp_min(1e-8)
        assert relative < 5e-3


def test_local_factor_is_one_generalized_delta_generator_with_exact_vjp():
    torch.manual_seed(405)
    chunk_size, width = 7, 11
    u = torch.randn(chunk_size, width, device="cuda", dtype=torch.float64)
    h = torch.randn_like(u)
    omega_h = torch.randn(chunk_size, device="cuda", dtype=torch.float64)
    omega_r = torch.randn_like(omega_h)
    cotangent = torch.randn(width, width, device="cuda", dtype=torch.float64)

    split_leaves = tuple(
        value.detach().requires_grad_() for value in (u, h, omega_h, omega_r)
    )
    su, sh, swh, swr = split_leaves
    split = su.T @ torch.diag(swh) @ su + su.T @ torch.diag(swr) @ sh
    split_grads = torch.autograd.grad((split * cotangent).sum(), split_leaves)

    fused_leaves = tuple(
        value.detach().requires_grad_() for value in (u, h, omega_h, omega_r)
    )
    fu, fh, fwh, fwr = fused_leaves
    generator = fwh[:, None] * fu + fwr[:, None] * fh
    fused = fu.T @ generator
    fused_grads = torch.autograd.grad((fused * cotangent).sum(), fused_leaves)

    assert torch.allclose(fused, split, atol=1e-12, rtol=1e-12)
    for got, expected in zip(fused_grads, split_grads):
        assert torch.allclose(got, expected, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("width", [32, 128])
def test_exact_coordinate_factor_actions_match_dense_triangular_reference(width):
    torch.manual_seed(407)
    panels, chunk_size = 2, 16
    rhs = torch.randn(
        panels, chunk_size, width, device="cuda", dtype=torch.bfloat16
    )
    u = F.normalize(torch.randn(panels, chunk_size, width, device="cuda"), dim=-1).half()
    h = (0.1 * torch.randn(panels, chunk_size, width, device="cuda")).bfloat16()
    j = 0.05 * torch.randn(panels, width, width, device="cuda")
    j = 0.5 * (j + j.transpose(-1, -2))
    d = 0.05 * torch.randn(panels, width, width, device="cuda")
    cumulative = -0.01 * torch.rand(
        panels, chunk_size, device="cuda"
    ).cumsum(dim=-1)
    indices = torch.arange(chunk_size, device="cuda")
    decay = torch.exp(cumulative[:, :, None] - cumulative[:, None, :])
    decay = torch.where(indices[:, None] >= indices[None, :], decay, 0.0)
    kappa_h = 0.01 * torch.randn(panels, chunk_size, device="cuda")
    kappa_r = 0.01 * torch.randn_like(kappa_h)
    boundary_h = 0.1 * torch.randn(panels, chunk_size, device="cuda")
    boundary_r = 0.1 * torch.randn_like(boundary_h)
    sigma = torch.exp(0.02 * torch.randn(panels, chunk_size, width, device="cuda")).half()

    def factor(panel, token, lower):
        matrix = boundary_h[panel, token] * j[panel]
        matrix = matrix + boundary_r[panel, token] * d[panel]
        omega_h = kappa_h[panel, token] * decay[panel, token]
        omega_r = kappa_r[panel, token] * decay[panel, token]
        matrix = matrix + (u[panel].float().T * omega_h) @ u[panel].float()
        matrix = matrix + (u[panel].float().T * omega_r) @ h[panel].float()
        strict = torch.tril(matrix, -1) if lower else torch.triu(matrix, 1)
        return torch.eye(width, device="cuda") + strict

    output, lower_output, _ = resident_primal(
        rhs, j, d, u, h, decay, kappa_h, kappa_r, kappa_r,
        boundary_h, boundary_r, boundary_r, sigma,
        num_warps=4,
    )
    expected_lower = torch.empty_like(lower_output.float())
    expected_output = torch.empty_like(output.float())
    for panel in range(panels):
        for token in range(chunk_size):
            lower = factor(panel, token, True)
            upper = factor(panel, token, False)
            solved = torch.linalg.solve_triangular(
                lower, rhs[panel, token].float()[:, None],
                upper=False, unitriangular=True,
            )[:, 0]
            expected_lower[panel, token] = solved
            expected_output[panel, token] = torch.linalg.solve_triangular(
                upper, (solved / sigma[panel, token].float())[:, None],
                upper=True, unitriangular=True,
            )[:, 0]
    assert (lower_output.float() - expected_lower).abs().max() < 3e-3
    assert (output.float() - expected_output).abs().max() < 2e-2

    cotangent = torch.randn_like(output.float())
    for lower in (True, False):
        got = resident_factor_transpose(
            cotangent, j, d, u, h, decay, kappa_h, kappa_r,
            boundary_h, boundary_r, lower=lower, num_warps=4,
        )
        direct = resident_factor_direct(
            rhs.float(), j, d, u, h, decay, kappa_h, kappa_r,
            boundary_h, boundary_r,
            lower=lower, transpose=False, num_warps=4,
        )
        expected_transpose = torch.empty_like(got)
        expected_direct = torch.empty_like(direct)
        for panel in range(panels):
            for token in range(chunk_size):
                matrix = factor(panel, token, lower)
                expected_transpose[panel, token] = torch.linalg.solve_triangular(
                    matrix.T, cotangent[panel, token, :, None],
                    upper=lower, unitriangular=True,
                )[:, 0]
                expected_direct[panel, token] = matrix @ rhs[panel, token].float()
        assert (got - expected_transpose).abs().max() < 2e-3
        assert (direct - expected_direct).abs().max() < 2e-3
