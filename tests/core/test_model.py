import torch
import pytest

from causallsso import SolveDelta, SolveDeltaConfig
from causallsso.model import _CausalShortConvolution


def test_model_owns_projections_and_supports_non_128_reference_width() -> None:
    config = SolveDeltaConfig(
        hidden_size=24,
        num_heads=3,
        head_k_dim=5,
        head_v_dim=4,
        num_edits=2,
        bias=True,
    )
    model = SolveDelta(config).double()
    hidden = torch.randn(2, 4, 24, dtype=torch.float64, requires_grad=True)
    output, state = model(hidden, return_final_state=True)
    assert output.shape == hidden.shape
    assert state.operator.m.shape == (2, 3)
    assert state.operator.J.shape == state.operator.D.shape == (2, 3, 5, 5)
    assert state.operator.S.shape == (2, 3, 5, 4)
    assert state.conv_q.shape == (2, 15, 4)
    assert state.conv_k.shape == (2, 30, 4)
    assert state.conv_v.shape == (2, 24, 4)
    (output.square().mean() + state.operator.S.square().mean()).backward()
    assert torch.isfinite(hidden.grad).all()


def test_geometry_and_skew_structural_switches_are_exact() -> None:
    config = SolveDeltaConfig(16, 2, head_k_dim=4, head_v_dim=4, num_edits=1)
    model = SolveDelta(config).double()
    hidden = torch.randn(1, 3, 16, dtype=torch.float64)
    _, state = model(
        hidden,
        geometry_enabled=False,
        skew_enabled=False,
        return_final_state=True,
    )
    assert torch.isfinite(state.operator.S).all()


def test_short_conv_recurrent_split_matches_whole_sequence() -> None:
    torch.manual_seed(19)
    config = SolveDeltaConfig(16, 2, head_k_dim=4, head_v_dim=3)
    model = SolveDelta(config).double()
    hidden = torch.randn(2, 7, 16, dtype=torch.float64)

    whole, whole_state = model(hidden, return_final_state=True)
    left, left_state = model(hidden[:, :3], return_final_state=True)
    right, split_state = model(
        hidden[:, 3:], initial_state=left_state, return_final_state=True
    )

    torch.testing.assert_close(torch.cat((left, right), dim=1), whole)
    for expected, actual in zip(whole_state.operator, split_state.operator):
        torch.testing.assert_close(actual, expected)
    for expected, actual in zip(whole_state[1:], split_state[1:]):
        torch.testing.assert_close(actual, expected)


def test_short_conv_structural_switch_uses_silu_without_cache() -> None:
    torch.manual_seed(22)
    config = SolveDeltaConfig(
        16,
        2,
        head_k_dim=4,
        head_v_dim=3,
        use_short_conv=False,
    )
    model = SolveDelta(config).double()
    hidden = torch.randn(1, 4, 16, dtype=torch.float64, requires_grad=True)

    output, state = model(hidden, return_final_state=True)
    assert not hasattr(model, "q_conv1d")
    assert state.conv_q is state.conv_k is state.conv_v is None
    (output.square().mean() + state.operator.S.square().mean()).backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_complete_layer_initial_short_conv_caches_have_gradients() -> None:
    torch.manual_seed(23)
    config = SolveDeltaConfig(16, 2, head_k_dim=4, head_v_dim=3)
    model = SolveDelta(config).double()
    prefix = torch.randn(1, 3, 16, dtype=torch.float64)
    suffix = torch.randn(1, 2, 16, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        _, prefix_state = model(prefix, return_final_state=True)

    initial_state = type(prefix_state)(
        type(prefix_state.operator)(
            *(tensor.detach().requires_grad_(True) for tensor in prefix_state.operator)
        ),
        prefix_state.conv_q.detach().requires_grad_(True),
        prefix_state.conv_k.detach().requires_grad_(True),
        prefix_state.conv_v.detach().requires_grad_(True),
    )
    output, final_state = model(
        suffix,
        initial_state=initial_state,
        return_final_state=True,
    )
    loss = output.square().mean()
    loss = loss + sum(tensor.square().mean() for tensor in final_state.operator)
    loss = loss + sum(tensor.square().mean() for tensor in final_state[1:])
    loss.backward()

    for tensor in (*initial_state.operator, *initial_state[1:]):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_short_conv_matches_explicit_causal_formula() -> None:
    convolution = _CausalShortConvolution(2).double()
    with torch.no_grad():
        convolution.weight.copy_(torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0]], [[-1.0, 0.5, 0.0, 2.0]]],
            dtype=torch.float64,
        ))
    x = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]],
        dtype=torch.float64,
    )
    actual, state = convolution(x, output_final_state=True)
    padded = torch.nn.functional.pad(x.transpose(1, 2), (3, 0))
    expected = torch.nn.functional.conv1d(
        padded, convolution.weight, groups=2
    ).transpose(1, 2)
    expected = torch.nn.functional.silu(expected)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, x[:, -4:].transpose(1, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_short_conv_matches_explicit_recurrence_and_gradients() -> None:
    torch.manual_seed(21)
    batch, length, width = 2, 7, 5
    convolution = _CausalShortConvolution(width).cuda().float()
    x = torch.randn(
        batch, length, width, device="cuda", dtype=torch.float32,
        requires_grad=True,
    )
    cache = torch.randn(
        batch, width, 4, device="cuda", dtype=torch.float32,
        requires_grad=True,
    )
    output_grad = torch.randn_like(x)
    state_grad = torch.randn_like(cache)

    actual, actual_state = convolution(
        x, cache=cache, output_final_state=True
    )
    actual_loss = (actual * output_grad).sum() + (actual_state * state_grad).sum()
    actual_grads = torch.autograd.grad(
        actual_loss, (x, cache, convolution.weight)
    )

    reference_x = x.detach().requires_grad_(True)
    reference_cache = cache.detach().requires_grad_(True)
    reference_weight = convolution.weight.detach().requires_grad_(True)
    state = reference_cache
    outputs = []
    for token in range(length):
        state = torch.cat(
            (state[..., 1:], reference_x[:, token].unsqueeze(-1)), dim=-1
        )
        preactivation = torch.einsum(
            "bdw,dw->bd", state, reference_weight[:, 0, :]
        )
        outputs.append(torch.nn.functional.silu(preactivation))
    expected = torch.stack(outputs, dim=1)
    expected_loss = (expected * output_grad).sum() + (state * state_grad).sum()
    expected_grads = torch.autograd.grad(
        expected_loss, (reference_x, reference_cache, reference_weight)
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_state, state)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-4, atol=2e-4)


def test_masks_and_resets_apply_to_complete_layer_state() -> None:
    torch.manual_seed(20)
    config = SolveDeltaConfig(12, 2, head_k_dim=3, head_v_dim=2)
    model = SolveDelta(config).double()
    hidden = torch.randn(1, 7, 12, dtype=torch.float64)
    valid = torch.tensor([[True, True, False, True, True, True, True]])
    masked, masked_state = model(
        hidden, valid_mask=valid, return_final_state=True
    )
    compressed, compressed_state = model(
        hidden[:, valid[0]], return_final_state=True
    )
    assert torch.count_nonzero(masked[:, 2]) == 0
    torch.testing.assert_close(masked[:, valid[0]], compressed)
    for expected, actual in zip(compressed_state.operator, masked_state.operator):
        torch.testing.assert_close(actual, expected)
    for expected, actual in zip(compressed_state[1:], masked_state[1:]):
        torch.testing.assert_close(actual, expected)

    reset = torch.zeros_like(valid)
    reset[:, 4] = True
    reset_output, reset_state = model(
        hidden, valid_mask=valid, reset_mask=reset, return_final_state=True
    )
    suffix, suffix_state = model(hidden[:, 4:], return_final_state=True)
    torch.testing.assert_close(reset_output[:, 4:], suffix)
    for expected, actual in zip(suffix_state.operator, reset_state.operator):
        torch.testing.assert_close(actual, expected)
    for expected, actual in zip(suffix_state[1:], reset_state[1:]):
        torch.testing.assert_close(actual, expected)
