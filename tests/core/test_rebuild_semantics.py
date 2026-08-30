from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from causallsso import (
    SolveDelta,
    SolveDeltaConfig,
    SolveDeltaState,
    solvedelta_reference,
    solvedelta_zero_state,
)
from causallsso.ops.operator import solvedelta_native
from causallsso.ops.residual_frame.exterior import direct_e_residual
from causallsso.ops.residual_frame.predictor import oja_residual
from causallsso.ops.radial_norm_gate import (
    RadialRMSNormGated,
    radial_rms_norm_gated_reference,
)
from causallsso.reference import RELATIVE_FRAME_RADIUS


CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native SolveDelta requires CUDA"
)


def _reference_inputs(*, length: int = 6, rank: int = 8, value_dim: int = 5):
    torch.manual_seed(7)
    batch, heads = 2, 2
    u = torch.randn(batch, length, heads, rank, dtype=torch.float64)
    h = torch.randn_like(u)
    q = torch.randn_like(u)
    keys = torch.randn(batch, length, heads, 1, rank, dtype=torch.float64)
    values = torch.randn(
        batch, length, heads, 1, value_dim, dtype=torch.float64
    )
    associative_decay = -0.02 * torch.rand(
        batch, length, heads, rank, dtype=torch.float64
    )
    erase = torch.rand_like(keys)
    write = torch.rand_like(values)
    geometry_write = torch.full((heads,), 0.25, dtype=torch.float64)
    return (
        u,
        h,
        q,
        keys,
        values,
        associative_decay,
        erase,
        write,
        geometry_write,
    )


def test_fp64_residual_predictor_and_local_similarity():
    args = _reference_inputs()
    output, state = solvedelta_reference(*args)
    u, h, q, keys, values, log_decay, erase, write, geometry_write = args
    batch, length, heads, rank = u.shape
    expected = solvedelta_zero_state(
        batch,
        heads,
        rank,
        state.S.shape[-1],
        dtype=torch.float64,
        device=u.device,
    )
    normalized_u = F.normalize(u, dim=-1)
    normalized_q = F.normalize(q, dim=-1)
    normalized_keys = F.normalize(keys, dim=-1)
    predictor = expected.predictor
    memory = expected.S
    expected_outputs = []
    identity = torch.eye(rank, dtype=torch.float64).view(1, 1, rank, rank)
    for token in range(length):
        direction = normalized_u[:, token]
        residual = h[:, token] - (
            predictor @ direction.unsqueeze(-1)
        ).squeeze(-1)
        predictor_update = geometry_write.view(1, heads, 1) * residual
        predictor = (
            predictor
            + predictor_update[..., :, None] * direction[..., None, :]
        )
        residual_after = h[:, token] - (
            predictor @ direction.unsqueeze(-1)
        ).squeeze(-1)
        torch.testing.assert_close(
            residual_after,
            (1.0 - geometry_write).view(1, heads, 1) * residual,
            rtol=2e-12,
            atol=2e-12,
        )

        frame_scale = RELATIVE_FRAME_RADIUS / torch.sqrt(
            RELATIVE_FRAME_RADIUS**2
            + direction.square().sum(dim=-1, keepdim=True)
            * predictor_update.square().sum(dim=-1, keepdim=True)
        )
        frame_covector = frame_scale * predictor_update
        frame = (
            identity
            + direction[..., :, None] * frame_covector[..., None, :]
        )
        denominator = 1.0 + (direction * frame_covector).sum(dim=-1)
        assert torch.all(denominator > 1.0 - RELATIVE_FRAME_RADIUS)
        assert torch.all(denominator < 1.0 + RELATIVE_FRAME_RADIUS)
        key = normalized_keys[:, token, :, 0]
        erase_key = erase[:, token, :, 0] * key
        direct = key + direction * (
            frame_covector * key
        ).sum(dim=-1, keepdim=True)
        dual = erase_key - frame_covector * (
            (direction * erase_key).sum(dim=-1, keepdim=True)
            / denominator[..., None]
        )
        query = normalized_q[:, token] - frame_covector * (
            (direction * normalized_q[:, token]).sum(dim=-1, keepdim=True)
            / denominator[..., None]
        )
        torch.testing.assert_close(
            (dual * direct).sum(dim=-1),
            (erase_key * key).sum(dim=-1),
            rtol=2e-12,
            atol=2e-12,
        )
        expected_query = torch.linalg.solve(
            frame.transpose(-1, -2), normalized_q[:, token].unsqueeze(-1)
        ).squeeze(-1)
        torch.testing.assert_close(query, expected_query, rtol=2e-12, atol=2e-12)
        memory = log_decay[:, token].exp()[..., None] * memory
        value = write[:, token, :, 0] * values[:, token, :, 0]
        prediction = torch.einsum("bhrv,bhr->bhv", memory, dual)
        memory = memory + direct[..., :, None] * (
            value - prediction
        )[..., None, :]
        expected_outputs.append(torch.einsum("bhrv,bhr->bhv", memory, query))
    torch.testing.assert_close(
        output, torch.stack(expected_outputs, dim=1), rtol=2e-12, atol=2e-12
    )
    torch.testing.assert_close(
        state.predictor, predictor, rtol=2e-12, atol=2e-12
    )
    torch.testing.assert_close(state.S, memory, rtol=2e-12, atol=2e-12)


def test_frame_radius_bounds_adversarial_residual():
    radius = RELATIVE_FRAME_RADIUS
    direction = F.normalize(
        torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64), dim=0
    )
    for sign in (-1.0, 1.0):
        predictor_update = sign * 1.0e6 * direction
        scale = radius / torch.sqrt(
            radius**2
            + direction.square().sum() * predictor_update.square().sum()
        )
        frame_covector = scale * predictor_update
        denominator = 1.0 + frame_covector.dot(direction)
        if sign < 0:
            assert denominator > 1.0 - radius
            assert denominator < 1.0 - radius + 1.0e-6
        else:
            assert denominator < 1.0 + radius
            assert denominator > 1.0 + radius - 1.0e-6


def test_radial_output_gate_matches_explicit_formula_and_bounds():
    layer = SolveDelta(
        SolveDeltaConfig(
            hidden_size=24,
            num_heads=2,
            head_k_dim=12,
            head_v_dim=12,
            use_short_conv=False,
        )
    ).double()
    torch.manual_seed(13)
    output = torch.randn(2, 5, 2, 12, dtype=torch.float64)
    gate = torch.randn_like(output)
    with torch.no_grad():
        layer.output_norm.weight.copy_(torch.linspace(0.5, 1.5, 12))
        layer.output_norm.radial_strength.copy_(torch.tensor([0.8, 1.2]))
    rstd = torch.rsqrt(
        output.square().mean(dim=-1, keepdim=True) + layer.output_norm.eps
    )
    reference_rms = 1.0 / 12**0.5
    radial_coordinate = (1.0 - reference_rms * rstd) / (
        1.0 + reference_rms * rstd
    )
    alpha = torch.sigmoid(2.0 * layer.output_norm.radial_strength) - 0.5
    scale = 1.0 + alpha.view(1, 1, 2, 1) * radial_coordinate
    expected = (
        scale
        * output
        * rstd
        * layer.output_norm.weight
        * torch.sigmoid(gate)
    )
    actual = layer._gate_output(output, gate)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    assert torch.all(scale > 0.5)
    assert torch.all(scale < 1.5)


def test_radial_output_gate_has_strict_composed_vjp():
    torch.manual_seed(17)
    module = RadialRMSNormGated(5, 2, eps=1e-6).double()
    x = torch.randn(2, 3, 2, 5, dtype=torch.float64, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x_, gate_, weight_, strength_: radial_rms_norm_gated_reference(
            x_, gate_, weight_, strength_, module.eps
        ),
        (x, gate, module.weight, module.radial_strength),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )


def test_zero_geometry_write_reduces_to_gdn2_edit():
    args = list(_reference_inputs())
    args[-1] = torch.zeros_like(args[-1])
    output, state = solvedelta_reference(*args)
    _, _, q, keys, values, associative_decay, erase, write, _ = args
    q = F.normalize(q, dim=-1)
    keys = F.normalize(keys, dim=-1)
    memory = torch.zeros_like(state.S)
    expected = []
    for token in range(q.shape[1]):
        memory = associative_decay[:, token].exp()[..., None] * memory
        key = keys[:, token, :, 0]
        erase_key = erase[:, token, :, 0] * key
        value = write[:, token, :, 0] * values[:, token, :, 0]
        prediction = torch.einsum("bhrv,bhr->bhv", memory, erase_key)
        memory = memory + key[..., :, None] * (value - prediction)[..., None, :]
        expected.append(torch.einsum("bhrv,bhr->bhv", memory, q[:, token]))
    torch.testing.assert_close(
        output, torch.stack(expected, dim=1), rtol=2e-12, atol=2e-12
    )
    torch.testing.assert_close(state.S, memory, rtol=2e-12, atol=2e-12)


def test_reference_masks_resets_recurrent_split_and_vjp():
    args = tuple(value.requires_grad_() for value in _reference_inputs(length=8))
    valid = torch.tensor(
        [[1, 1, 0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 0, 1, 1]],
        dtype=torch.bool,
    )
    reset = torch.zeros_like(valid)
    reset[0, 4] = True
    reset[1, 2] = True
    output, state = solvedelta_reference(
        *args, valid_mask=valid, reset_mask=reset
    )
    assert torch.equal(output[~valid], torch.zeros_like(output[~valid]))

    split = 4
    left_args = tuple(value[:, :split] for value in args[:-1]) + (args[-1],)
    right_args = tuple(value[:, split:] for value in args[:-1]) + (args[-1],)
    left_output, left_state = solvedelta_reference(
        *left_args,
        valid_mask=valid[:, :split],
        reset_mask=reset[:, :split],
    )
    right_output, right_state = solvedelta_reference(
        *right_args,
        initial_state=left_state,
        valid_mask=valid[:, split:],
        reset_mask=reset[:, split:],
    )
    torch.testing.assert_close(
        output, torch.cat((left_output, right_output), dim=1), rtol=2e-12, atol=2e-12
    )
    for actual, expected in zip(right_state, state):
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)

    output_cotangent = torch.randn_like(output)
    state_cotangents = [torch.randn_like(value) for value in state]
    loss = (output * output_cotangent).sum() + sum(
        (value * cotangent).sum()
        for value, cotangent in zip(state, state_cotangents)
    )
    gradients = torch.autograd.grad(loss, args)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_current_contract_rejects_multiple_edits():
    with pytest.raises(ValueError, match="num_edits=1"):
        SolveDeltaConfig(hidden_size=32, num_heads=1, num_edits=2)
    args = list(_reference_inputs())
    args[3] = args[3].expand(-1, -1, -1, 2, -1).clone()
    args[4] = args[4].expand(-1, -1, -1, 2, -1).clone()
    args[6] = args[6].expand(-1, -1, -1, 2, -1).clone()
    args[7] = args[7].expand(-1, -1, -1, 2, -1).clone()
    with pytest.raises(ValueError, match="num_edits=1"):
        solvedelta_reference(*args)


def _native_inputs(*, length: int = 16, rank: int = 16, value_dim: int = 16):
    torch.manual_seed(23)
    batch, heads = 1, 2

    def leaf(shape, scale=1.0):
        return (scale * torch.randn(*shape, device="cuda")).to(
            torch.bfloat16
        ).requires_grad_()

    return (
        leaf((batch, length, heads, rank)),
        leaf((batch, length, heads, rank)),
        leaf((batch, length, heads, rank)),
        leaf((batch, length, heads, 1, rank)),
        leaf((batch, length, heads, 1, value_dim)),
        (-0.02 * torch.rand(batch, length, heads, rank, device="cuda"))
        .detach()
        .requires_grad_(),
        leaf((batch, length, heads, 1, rank), 0.5),
        leaf((batch, length, heads, 1, value_dim), 0.5),
        torch.full(
            (batch, length, heads), 0.2, device="cuda", requires_grad=True
        ),
    )


def _native_state(*, rank: int = 16, value_dim: int = 16):
    torch.manual_seed(31)
    return SolveDeltaState(
        (0.02 * torch.randn(1, 2, rank, rank, device="cuda")).requires_grad_(),
        (0.02 * torch.randn(1, 2, rank, value_dim, device="cuda"))
        .requires_grad_(),
    )


@CUDA_ONLY
@pytest.mark.parametrize("decay_scale", (0.4, 16.0))
def test_unbounded_direct_e_owner_matches_exact_fla_dplr(decay_scale):
    from fla.ops.generalized_delta_rule.dplr import chunk as dplr_chunk

    torch.manual_seed(127)
    batch, length, heads, rank, value_dim, chunk = 1, 16, 1, 16, 16, 16
    panels = batch * heads * (length // chunk)

    def leaf(shape, scale=0.1, dtype=torch.bfloat16):
        return (scale * torch.randn(*shape, device="cuda")).to(dtype).requires_grad_()

    direct = leaf((panels, 1, chunk, rank))
    paired = leaf((panels, 2, chunk, rank))
    value = leaf((batch, length, heads, value_dim))
    log_decay = (
        -decay_scale
        * (0.75 + 0.25 * torch.rand(batch, length, heads, rank, device="cuda"))
    ).requires_grad_()
    initial = leaf((batch, heads, rank, value_dim), dtype=torch.float32)
    inputs = (direct, paired, value, log_decay, initial)

    output, final = direct_e_residual(
        *inputs, chunk_size=chunk, output_final_state=True
    )

    def unpack(panel, route):
        return (
            panel[:, route]
            .view(batch, heads, length // chunk, chunk, rank)
            .reshape(batch, heads, length, rank)
            .transpose(1, 2)
            .contiguous()
        )

    d = unpack(direct, 0)
    e = unpack(paired, 0)
    q = unpack(paired, 1)
    generic = dplr_chunk.chunk_dplr_delta_rule
    while hasattr(generic, "__wrapped__"):
        generic = generic.__wrapped__
    # FLA's low-rank term consumes the pre-decay state. Multiplying a by the
    # current decay maps SolveDelta's erase action on S_decay to that ordering.
    a = (-e.float() * log_decay.exp()).to(e.dtype)
    expected_output, expected_final = generic(
        q=q,
        k=d,
        v=value,
        a=a,
        b=d,
        gk=log_decay,
        scale=1.0,
        initial_state=initial,
        output_final_state=True,
        safe_gate=False,
        chunk_size=chunk,
    )

    output_cotangent = torch.randn_like(output)
    state_cotangent = torch.randn_like(final)
    gradients = torch.autograd.grad(
        (output, final), inputs, (output_cotangent, state_cotangent), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        (expected_output, expected_final),
        inputs,
        (output_cotangent, state_cotangent),
    )
    torch.testing.assert_close(output, expected_output, rtol=8e-3, atol=1e-3)
    torch.testing.assert_close(final, expected_final, rtol=5e-3, atol=1e-4)
    for actual, expected in zip(gradients, expected_gradients):
        assert torch.isfinite(actual).all()
        relative = (actual.float() - expected.float()).norm()
        relative = relative / expected.float().norm().clamp_min(1e-8)
        assert relative < 1e-2


@CUDA_ONLY
def test_native_strided_sources_and_fused_raw_gates_match_packed_path():
    packed = tuple(value.detach().clone().requires_grad_() for value in _native_inputs())
    vector_indices = {0, 1, 2, 3, 4, 6, 7}
    strided = []
    for index, value in enumerate(packed):
        if index not in vector_indices:
            strided.append(value.detach().clone().requires_grad_())
            continue
        storage = torch.empty(
            *value.shape[:-1], value.shape[-1] + 7,
            dtype=value.dtype,
            device=value.device,
        )
        view = storage[..., : value.shape[-1]]
        view.copy_(value.detach())
        strided.append(view.detach().requires_grad_())
        assert not strided[-1].is_contiguous()
    strided = tuple(strided)

    packed_output, packed_state = solvedelta_native(
        *packed, return_final_state=True
    )
    output_without_state, absent_state = solvedelta_native(
        *packed, return_final_state=False
    )
    assert absent_state is None
    assert torch.equal(output_without_state, packed_output)
    strided_output, strided_state = solvedelta_native(
        *strided, return_final_state=True
    )
    assert torch.equal(strided_output, packed_output)
    for actual, expected in zip(strided_state, packed_state):
        assert torch.equal(actual, expected)

    torch.manual_seed(17)
    output_cotangent = torch.randn_like(packed_output)
    state_cotangents = [torch.randn_like(value) for value in packed_state]

    def loss(output, state):
        return (output * output_cotangent).sum() + sum(
            (value * cotangent).sum()
            for value, cotangent in zip(state, state_cotangents)
        )

    packed_gradients = torch.autograd.grad(loss(packed_output, packed_state), packed)
    strided_gradients = torch.autograd.grad(
        loss(strided_output, strided_state), strided
    )
    for actual, expected in zip(strided_gradients, packed_gradients):
        assert torch.equal(actual, expected)


@CUDA_ONLY
def test_native_forward_and_composed_state_vjp_match_fp64_oracle():
    args = _native_inputs()
    initial_state = _native_state()
    output, state = solvedelta_native(
        *args, initial_state=initial_state, return_final_state=True
    )
    reference_leaves = tuple(
        value.detach().double().requires_grad_() for value in args
    )
    reference_state = SolveDeltaState(
        *(value.detach().double().requires_grad_() for value in initial_state)
    )
    reference_output, expected_state = solvedelta_reference(
        *reference_leaves[:6],
        torch.sigmoid(reference_leaves[6]),
        torch.sigmoid(reference_leaves[7]),
        reference_leaves[8],
        initial_state=reference_state,
    )
    torch.testing.assert_close(
        output.float(), reference_output.float(), rtol=2e-2, atol=2e-2
    )
    for actual, expected, rtol, atol in zip(
        state,
        expected_state,
        (8e-3, 1e-2),
        (1.5e-2, 3e-2),
    ):
        torch.testing.assert_close(actual, expected.float(), rtol=rtol, atol=atol)

    torch.manual_seed(29)
    output_cotangent = torch.randn_like(output)
    state_cotangents = [torch.randn_like(value) for value in state]
    loss = (output * output_cotangent).sum() + sum(
        (value * cotangent).sum()
        for value, cotangent in zip(state, state_cotangents)
    )
    gradients = torch.autograd.grad(loss, (*args, *initial_state))
    reference_loss = (reference_output * output_cotangent.double()).sum() + sum(
        (value * cotangent.double()).sum()
        for value, cotangent in zip(expected_state, state_cotangents)
    )
    expected_gradients = list(
        torch.autograd.grad(reference_loss, (*reference_leaves, *reference_state))
    )
    for actual, expected in zip(gradients, expected_gradients):
        assert torch.isfinite(actual).all()
        relative = (actual.float() - expected.float()).norm()
        relative = relative / expected.float().norm().clamp_min(1e-8)
        assert relative < 3e-2


@CUDA_ONLY
def test_native_predictor_multitile_source_transpose_matches_fp64():
    torch.manual_seed(131)
    batch, length, heads, rank = 1, 32, 1, 128

    def bf16_leaf(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16).requires_grad_()

    target = bf16_leaf(batch, length, heads, rank)
    source = F.normalize(
        bf16_leaf(batch, length, heads, rank), dim=-1
    ).detach().requires_grad_()
    beta = torch.sigmoid(
        torch.randn(batch, length, heads, device="cuda") - 2.0
    ).requires_grad_()
    initial = (
        0.02 * torch.randn(batch, heads, rank, rank, device="cuda")
    ).requires_grad_()
    update, final = oja_residual(
        target,
        source,
        beta,
        initial,
        chunk_size=32,
        output_final_state=True,
    )

    reference_inputs = tuple(
        value.detach().double().requires_grad_()
        for value in (target, source, beta, initial)
    )
    target_ref, source_ref, beta_ref, predictor = reference_inputs
    updates = []
    for token in range(length):
        residual = target_ref[:, token] - (
            predictor @ source_ref[:, token].unsqueeze(-1)
        ).squeeze(-1)
        token_update = beta_ref[:, token, :, None] * residual
        predictor = (
            predictor
            + token_update[..., :, None] * source_ref[:, token, :, None, :]
        )
        updates.append(token_update)
    update_ref = torch.stack(updates, dim=1)

    update_cotangent = torch.randn_like(update)
    state_cotangent = torch.randn_like(final)
    gradients = torch.autograd.grad(
        (update, final),
        (target, source, beta, initial),
        (update_cotangent, state_cotangent),
    )
    reference_gradients = torch.autograd.grad(
        (update_ref, predictor),
        reference_inputs,
        (update_cotangent.double(), state_cotangent.double()),
    )
    for actual, expected in zip(gradients, reference_gradients):
        relative = (actual.float() - expected.float()).norm()
        relative = relative / expected.float().norm().clamp_min(1e-8)
        assert relative < 8e-3


@CUDA_ONLY
def test_radial_output_gate_cuda_bf16_forward_and_vjp():
    torch.manual_seed(19)
    heads, width = 2, 16
    module = RadialRMSNormGated(width, heads, eps=1e-6).cuda()
    with torch.no_grad():
        module.weight.copy_(torch.linspace(0.5, 1.5, width, device="cuda"))
        module.radial_strength.copy_(torch.tensor([0.8, 1.1], device="cuda"))
    x = torch.randn(
        2,
        5,
        heads,
        width,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    gate = torch.randn_like(x, requires_grad=True)
    output = module(x, gate)
    upstream = torch.randn_like(output.float())
    actual_gradients = torch.autograd.grad(
        (output.float() * upstream).sum(),
        (x, gate, module.weight, module.radial_strength),
    )

    reference_x = x.detach().float().requires_grad_()
    reference_gate = gate.detach().float().requires_grad_()
    reference_weight = module.weight.detach().clone().requires_grad_()
    reference_strength = (
        module.radial_strength.detach().clone().requires_grad_()
    )
    reference_output = radial_rms_norm_gated_reference(
        reference_x,
        reference_gate,
        reference_weight,
        reference_strength,
        module.eps,
    )
    reference_gradients = torch.autograd.grad(
        (reference_output * upstream).sum(),
        (
            reference_x,
            reference_gate,
            reference_weight,
            reference_strength,
        ),
    )
    torch.testing.assert_close(
        output.float(), reference_output, rtol=8e-3, atol=1e-2
    )
    for actual, expected in zip(actual_gradients, reference_gradients):
        relative = (actual.float() - expected.float()).norm()
        relative = relative / expected.float().norm().clamp_min(1e-8)
        assert relative < 5e-3


@CUDA_ONLY
def test_radial_output_gate_linear_lifetime_matches_separate_path():
    torch.manual_seed(23)
    batch, length, heads, width = 2, 5, 2, 16
    module = RadialRMSNormGated(width, heads, eps=1e-6).cuda()
    projection = torch.nn.Linear(heads * width, 24, bias=True).cuda()
    x_separate = torch.randn(
        batch,
        length,
        heads,
        width,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    gate_separate = torch.randn_like(x_separate, requires_grad=True)
    x_combined = x_separate.detach().clone().requires_grad_()
    gate_combined = gate_separate.detach().clone().requires_grad_()
    parameters = (
        module.weight,
        module.radial_strength,
        projection.weight,
        projection.bias,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        separate = projection(
            module(x_separate, gate_separate).reshape(batch, length, -1)
        )
        combined = module.forward_linear(x_combined, gate_combined, projection)
    upstream = torch.randn_like(separate)
    separate_gradients = torch.autograd.grad(
        (separate * upstream).sum(),
        (x_separate, gate_separate, *parameters),
    )
    combined_gradients = torch.autograd.grad(
        (combined * upstream).sum(),
        (x_combined, gate_combined, *parameters),
    )
    torch.testing.assert_close(combined, separate, rtol=0.0, atol=0.0)
    for actual, expected in zip(combined_gradients, separate_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


@CUDA_ONLY
def test_native_aligned_recurrent_split():
    args = _native_inputs(length=32)
    whole_output, whole_state = solvedelta_native(*args, return_final_state=True)
    first = tuple(value[:, :16] for value in args)
    second = tuple(value[:, 16:] for value in args)
    left_output, left_state = solvedelta_native(*first, return_final_state=True)
    right_output, right_state = solvedelta_native(
        *second, initial_state=left_state, return_final_state=True
    )
    torch.testing.assert_close(
        whole_output,
        torch.cat((left_output, right_output), dim=1),
        rtol=2e-2,
        atol=2e-2,
    )
    for actual, expected in zip(right_state, whole_state):
        torch.testing.assert_close(actual, expected, rtol=8e-3, atol=2e-2)


@CUDA_ONLY
def test_model_dense_and_masked_paths():
    torch.manual_seed(41)
    config = SolveDeltaConfig(
        hidden_size=16,
        num_heads=1,
        head_k_dim=16,
        head_v_dim=16,
        use_short_conv=False,
    )
    layer = SolveDelta(config).cuda().to(torch.bfloat16)
    hidden = torch.randn(
        1, 16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    output, state = layer(hidden, return_final_state=True)
    assert output.shape == hidden.shape
    assert output.dtype == torch.bfloat16
    assert all(value.dtype == torch.float32 for value in state.operator)
    torch.autograd.grad(
        output.float().square().mean()
        + sum(value.square().mean() for value in state.operator),
        (hidden, *tuple(layer.parameters())),
        allow_unused=True,
    )

    valid = torch.ones(1, 16, dtype=torch.bool, device="cuda")
    valid[:, [3, 11]] = False
    reset = torch.zeros_like(valid)
    reset[:, 8] = True
    masked, masked_state = layer(
        hidden.detach(),
        valid_mask=valid,
        reset_mask=reset,
        return_final_state=True,
    )
    assert torch.equal(masked[:, ~valid[0]], torch.zeros_like(masked[:, ~valid[0]]))
    assert torch.isfinite(masked).all()
    assert all(torch.isfinite(value).all() for value in masked_state.operator)


def test_model_reference_supports_non_native_width():
    torch.manual_seed(47)
    layer = SolveDelta(
        SolveDeltaConfig(
            hidden_size=24,
            num_heads=2,
            head_k_dim=12,
            head_v_dim=12,
            use_short_conv=False,
        )
    ).double()
    hidden = torch.randn(2, 5, 24, dtype=torch.float64, requires_grad=True)
    output, state = layer(hidden, return_final_state=True)
    assert output.shape == hidden.shape
    assert all(value.dtype == torch.float64 for value in state.operator)
    output.square().mean().backward()
    assert torch.isfinite(hidden.grad).all()
