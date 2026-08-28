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
    geometry_decay = torch.full(
        (batch, length, heads),
        torch.log(torch.tensor(0.99, dtype=torch.float64)),
    )
    associative_decay = -0.02 * torch.rand(
        batch, length, heads, rank, dtype=torch.float64
    )
    erase = 2.0 * torch.rand_like(keys)
    write = 2.0 * torch.rand_like(values)
    strength = torch.full((heads,), 0.25, dtype=torch.float64)
    return (
        u,
        h,
        q,
        keys,
        values,
        geometry_decay,
        associative_decay,
        erase,
        write,
        strength,
    )


def test_fp64_geometry_recurrence_and_cross_map():
    args = _reference_inputs()
    _, state = solvedelta_reference(*args, prior_mass=2.0)
    u, h, _, _, _, geometry_decay, _, _, _, _ = args
    batch, length, heads, rank = u.shape
    expected = solvedelta_zero_state(
        batch,
        heads,
        rank,
        state.S.shape[-1],
        prior_mass=2.0,
        dtype=torch.float64,
        device=u.device,
    )
    normalized_u = F.normalize(u, dim=-1)
    mass, J, D = expected.m, expected.J, expected.D
    for token in range(length):
        decay = geometry_decay[:, token].exp()
        direction = normalized_u[:, token]
        mass = decay * mass + 1.0
        J = (
            decay[..., None, None] * J
            + direction[..., :, None] * direction[..., None, :]
        )
        D = (
            decay[..., None, None] * D
            + direction[..., :, None] * h[:, token, ..., None, :]
        )
    torch.testing.assert_close(state.m, mass, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state.J, J, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state.D, D, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(
        torch.linalg.solve(state.J, state.D),
        torch.linalg.solve(J, D),
        rtol=2e-11,
        atol=2e-11,
    )


def test_zero_geometry_strength_reduces_to_gdn2_edit():
    args = list(_reference_inputs())
    args[-1] = torch.zeros_like(args[-1])
    output, state = solvedelta_reference(*args, prior_mass=2.0)
    _, _, q, keys, values, _, associative_decay, erase, write, _ = args
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
        *args, prior_mass=2.0, valid_mask=valid, reset_mask=reset
    )
    assert torch.equal(output[~valid], torch.zeros_like(output[~valid]))

    split = 4
    left_args = tuple(value[:, :split] for value in args[:-1]) + (args[-1],)
    right_args = tuple(value[:, split:] for value in args[:-1]) + (args[-1],)
    left_output, left_state = solvedelta_reference(
        *left_args,
        prior_mass=2.0,
        valid_mask=valid[:, :split],
        reset_mask=reset[:, :split],
    )
    right_output, right_state = solvedelta_reference(
        *right_args,
        prior_mass=2.0,
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
    args[7] = args[7].expand(-1, -1, -1, 2, -1).clone()
    args[8] = args[8].expand(-1, -1, -1, 2, -1).clone()
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
        torch.full(
            (batch, length, heads),
            torch.log(torch.tensor(0.99)),
            device="cuda",
            requires_grad=True,
        ),
        (-0.02 * torch.rand(batch, length, heads, rank, device="cuda"))
        .detach()
        .requires_grad_(),
        leaf((batch, length, heads, 1, rank), 0.5),
        leaf((batch, length, heads, 1, value_dim), 0.5),
        torch.full((heads,), 0.2, device="cuda", requires_grad=True),
    )


def _native_state(*, rank: int = 16, value_dim: int = 16):
    torch.manual_seed(31)
    raw = 0.05 * torch.randn(1, 2, rank, rank, device="cuda")
    J = raw @ raw.transpose(-1, -2)
    J = 0.5 * (J + J.transpose(-1, -2))
    J = J + 2.0 * torch.eye(rank, device="cuda")
    return SolveDeltaState(
        torch.full((1, 2), 2.0, device="cuda", requires_grad=True),
        J.detach().requires_grad_(),
        (0.02 * torch.randn(1, 2, rank, rank, device="cuda")).requires_grad_(),
        (0.02 * torch.randn(1, 2, rank, value_dim, device="cuda"))
        .requires_grad_(),
    )


@CUDA_ONLY
def test_native_strided_sources_and_fused_raw_gates_match_packed_path():
    packed = tuple(value.detach().clone().requires_grad_() for value in _native_inputs())
    vector_indices = {0, 1, 2, 3, 4, 7, 8}
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
        *reference_leaves[:7],
        2.0 * torch.sigmoid(reference_leaves[7]),
        2.0 * torch.sigmoid(reference_leaves[8]),
        reference_leaves[9],
        initial_state=reference_state,
    )
    torch.testing.assert_close(
        output.float(), reference_output.float(), rtol=2e-2, atol=2e-2
    )
    for actual, expected, rtol, atol in zip(
        state,
        expected_state,
        (2e-6, 4e-3, 8e-3, 1e-2),
        (2e-5, 4e-3, 1.5e-2, 3e-2),
    ):
        torch.testing.assert_close(actual, expected.float(), rtol=rtol, atol=atol)
    assert torch.equal(state.J, state.J.transpose(-1, -2))

    torch.manual_seed(29)
    output_cotangent = torch.randn_like(output)
    state_cotangents = [torch.randn_like(value) for value in state]
    state_cotangents[1] = 0.5 * (
        state_cotangents[1] + state_cotangents[1].transpose(-1, -2)
    )
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
    expected_gradients[-3] = 0.5 * (
        expected_gradients[-3] + expected_gradients[-3].transpose(-1, -2)
    )
    for actual, expected in zip(gradients, expected_gradients):
        assert torch.isfinite(actual).all()
        relative = (actual.float() - expected.float()).norm()
        relative = relative / expected.float().norm().clamp_min(1e-8)
        assert relative < 3e-2
    assert torch.equal(gradients[-3], gradients[-3].transpose(-1, -2))


@CUDA_ONLY
def test_native_aligned_recurrent_split():
    args = _native_inputs(length=32)
    whole_output, whole_state = solvedelta_native(*args, return_final_state=True)
    first = tuple(value[:, :16] for value in args[:-1]) + (args[-1],)
    second = tuple(value[:, 16:] for value in args[:-1]) + (args[-1],)
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
