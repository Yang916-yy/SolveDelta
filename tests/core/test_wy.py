from __future__ import annotations

import importlib.util

import pytest
import torch

from causallsso.ops.wy import wy_associative


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("fla") is None,
    reason="CUDA and flash-linear-attention are required",
)


def _rho(reference: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    reference = reference.double()
    actual = actual.double()
    return (actual - reference).square().mean().sqrt() / (
        reference.square().mean().sqrt() + 1e-8
    )


def _oracle(
    chi: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    z: torch.Tensor,
    log_decay: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state
    outputs = []
    for token in range(chi.shape[1]):
        state = torch.exp(log_decay[:, token])[..., None] * state
        for edit in range(d.shape[3]):
            prediction = torch.einsum(
                "bhrv,bhr->bhv", state, e[:, token, :, edit]
            )
            innovation = z[:, token, :, edit] - prediction
            state = state + torch.einsum(
                "bhr,bhv->bhrv", d[:, token, :, edit], innovation
            )
        outputs.append(
            torch.einsum("bhrv,bhr->bhv", state, chi[:, token])
        )
    return torch.stack(outputs, dim=1), state


def _runtime_inputs(edits: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(916 + edits)
    batch, length, heads, rank, value_dim = 1, 35, 2, 32, 24
    prefix = (batch, length, heads)

    def variable(value: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return value.to(dtype).detach().requires_grad_(True)

    return (
        variable(torch.randn(*prefix, rank, device="cuda")),
        variable(0.04 * torch.randn(*prefix, edits, rank, device="cuda")),
        variable(0.04 * torch.randn(*prefix, edits, rank, device="cuda")),
        variable(0.08 * torch.randn(*prefix, edits, value_dim, device="cuda")),
        variable(
            -0.03 * torch.rand(*prefix, rank, device="cuda"),
            torch.float32,
        ),
        variable(
            0.03 * torch.randn(batch, heads, rank, value_dim, device="cuda"),
            torch.float32,
        ),
    )


def _legacy_wy(
    chi: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    z: torch.Tensor,
    log_decay: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule

    batch, length, heads, edits, rank = d.shape
    value_dim = z.shape[-1]
    if edits == 1:
        q = chi
        k = d.squeeze(3)
        v = z.squeeze(3)
        packed_e = e.squeeze(3)
        g = log_decay
    else:
        slot = torch.arange(edits, device=d.device).view(1, 1, edits, 1, 1)
        q = (chi.unsqueeze(2) * (slot == edits - 1)).reshape(
            batch, length * edits, heads, rank
        )
        k = d.transpose(2, 3).reshape(batch, length * edits, heads, rank)
        v = z.transpose(2, 3).reshape(batch, length * edits, heads, value_dim)
        packed_e = e.transpose(2, 3).reshape(batch, length * edits, heads, rank)
        g = (log_decay.unsqueeze(2) * (slot == 0)).reshape(
            batch, length * edits, heads, rank
        )
    a = -(packed_e.float() * torch.exp(g)).to(packed_e.dtype)
    expanded_output, final_state = chunk_dplr_delta_rule(
        q=q,
        k=k,
        v=v,
        a=a,
        b=k,
        gk=g,
        scale=1.0,
        initial_state=initial_state,
        output_final_state=True,
        safe_gate=True,
        chunk_size=32,
        disable_recompute=False,
    )
    assert final_state is not None
    if edits == 1:
        return expanded_output, final_state
    output = expanded_output.reshape(
        batch, length, edits, heads, value_dim
    )[:, :, -1]
    return output, final_state


@pytest.mark.parametrize("edits", [1, 2, 4])
def test_wy_forward_and_final_state_match_fp64_oracle(edits: int) -> None:
    runtime = _runtime_inputs(edits)
    output, final_state = wy_associative(
        *runtime[:5], initial_state=runtime[5], output_final_state=True
    )
    expected_output, expected_state = _oracle(
        *(value.detach().double() for value in runtime)
    )

    assert final_state is not None
    assert final_state.dtype == torch.float32
    assert _rho(expected_output, output) < 6e-3
    assert _rho(expected_state, final_state) < 6e-3


@pytest.mark.parametrize("edits", [1, 2, 4])
def test_wy_backward_matches_fp64_oracle(edits: int) -> None:
    runtime = _runtime_inputs(edits)
    reference = tuple(
        value.detach().double().requires_grad_(True) for value in runtime
    )
    torch.manual_seed(1200 + edits)
    upstream_output = torch.randn(
        runtime[0].shape[:3] + (runtime[3].shape[-1],), device="cuda"
    )
    upstream_state = torch.randn_like(runtime[5])

    output, final_state = wy_associative(
        *runtime[:5], initial_state=runtime[5], output_final_state=True
    )
    assert final_state is not None
    loss = (
        (output.float() * upstream_output).sum()
        + (final_state * upstream_state).sum()
    )
    actual_gradients = torch.autograd.grad(loss, runtime)

    expected_output, expected_state = _oracle(*reference)
    expected_loss = (
        (expected_output * upstream_output.double()).sum()
        + (expected_state * upstream_state.double()).sum()
    )
    expected_gradients = torch.autograd.grad(expected_loss, reference)

    for expected, actual in zip(expected_gradients, actual_gradients):
        assert torch.isfinite(actual).all()
        assert _rho(expected, actual) < 1.5e-2


@pytest.mark.parametrize("edits", [1, 2])
def test_direct_e_specialization_matches_materialized_fla(
    edits: int,
) -> None:
    runtime = _runtime_inputs(edits)
    torch.manual_seed(1901 + edits)
    upstream_output = torch.randn(
        runtime[0].shape[:3] + (runtime[3].shape[-1],), device="cuda"
    )
    upstream_state = torch.randn_like(runtime[5])

    output, final_state = wy_associative(
        *runtime[:5], initial_state=runtime[5], output_final_state=True
    )
    legacy_output, legacy_state = _legacy_wy(*runtime)
    assert final_state is not None
    assert _rho(legacy_output, output) < 4e-3
    assert _rho(legacy_state, final_state) < 4e-3

    loss = (
        (output.float() * upstream_output).sum()
        + (final_state * upstream_state).sum()
    )
    legacy_loss = (
        (legacy_output.float() * upstream_output).sum()
        + (legacy_state * upstream_state).sum()
    )
    actual_gradients = torch.autograd.grad(loss, runtime, retain_graph=True)
    legacy_gradients = torch.autograd.grad(legacy_loss, runtime)
    for actual, legacy in zip(actual_gradients, legacy_gradients):
        assert _rho(legacy, actual) < 8e-3


@pytest.mark.parametrize("edits", [1, 2])
def test_wy_passes_e_directly_and_aliases_b_to_k(
    edits: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import causallsso.ops.wy as wy_module

    captured: dict[str, object] = {}

    def fake_direct_e_delta_rule(**kwargs: object) -> tuple[torch.Tensor, None]:
        captured.update(kwargs)
        q = kwargs["q"]
        v = kwargs["v"]
        assert isinstance(q, torch.Tensor)
        assert isinstance(v, torch.Tensor)
        return q.new_zeros(q.shape[:-1] + (v.shape[-1],)), None

    monkeypatch.setattr(
        wy_module,
        "_direct_e_dplr_delta_rule",
        fake_direct_e_delta_rule,
    )
    runtime = _runtime_inputs(edits)
    wy_associative(*runtime[:5], initial_state=runtime[5])

    assert captured["b"] is captured["k"]
    assert "a" not in captured
    packed_e = captured["e"]
    packed_g = captured["g"]
    assert isinstance(packed_e, torch.Tensor)
    assert isinstance(packed_g, torch.Tensor)
    assert packed_e.dtype == torch.bfloat16
    assert packed_g.dtype == torch.float32
    assert captured["chunk_size"] == 32
    if edits == 1:
        expected_e = runtime[2].squeeze(3)
    else:
        batch, length, heads, _, rank = runtime[1].shape
        expected_e = runtime[2].transpose(2, 3).reshape(
            batch, length * edits, heads, rank
        )
    assert torch.equal(packed_e, expected_e)


def test_direct_e_autograd_saves_no_activation_tensor() -> None:
    runtime = _runtime_inputs(1)
    output, _ = wy_associative(
        *runtime[:5], initial_state=runtime[5], output_final_state=True
    )
    saved = output.grad_fn.saved_tensors
    input_storage = {tensor.data_ptr() for tensor in runtime}

    assert len(saved) == 7
    assert {tensor.data_ptr() for tensor in saved} <= input_storage


def test_direct_e_c32_r128_forward_and_backward_are_finite() -> None:
    torch.manual_seed(2718)
    batch, length, heads, rank, value_dim = 1, 33, 1, 128, 64
    prefix = (batch, length, heads)

    def variable(
        value: torch.Tensor, dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        return value.to(dtype).detach().requires_grad_(True)

    runtime = (
        variable(torch.randn(*prefix, rank, device="cuda")),
        variable(0.04 * torch.randn(*prefix, 1, rank, device="cuda")),
        variable(0.04 * torch.randn(*prefix, 1, rank, device="cuda")),
        variable(0.08 * torch.randn(*prefix, 1, value_dim, device="cuda")),
        variable(-0.02 * torch.rand(*prefix, rank, device="cuda"), torch.float32),
        variable(
            0.03 * torch.randn(batch, heads, rank, value_dim, device="cuda"),
            torch.float32,
        ),
    )
    output, final_state = wy_associative(
        *runtime[:5], initial_state=runtime[5], output_final_state=True
    )
    assert final_state is not None
    gradients = torch.autograd.grad(
        output.float().square().mean() + final_state.square().mean(), runtime
    )

    assert output.dtype == torch.bfloat16
    assert final_state.dtype == torch.float32
    assert all(torch.isfinite(tensor).all() for tensor in (output, final_state))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def _strided_leaf(tensor: torch.Tensor) -> torch.Tensor:
    storage = torch.stack((tensor.detach(), tensor.detach()), dim=-1)
    view = storage[..., 0]
    assert not view.is_contiguous()
    return view.requires_grad_(True)


def test_direct_e_canonicalizes_strided_inputs_and_state() -> None:
    runtime = tuple(tensor.detach() for tensor in _runtime_inputs(1))
    contiguous = tuple(tensor.clone().requires_grad_(True) for tensor in runtime)
    strided = tuple(_strided_leaf(tensor) for tensor in runtime)
    torch.manual_seed(31415)
    output_cotangent = torch.randn(
        runtime[0].shape[:3] + (runtime[3].shape[-1],),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state_cotangent = torch.randn_like(runtime[5])

    def evaluate(
        inputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        output, final_state = wy_associative(
            *inputs[:5], initial_state=inputs[5], output_final_state=True
        )
        assert final_state is not None
        loss = (output * output_cotangent).float().sum()
        loss = loss + (final_state * state_cotangent).sum()
        gradients = torch.autograd.grad(loss, inputs)
        return output, final_state, gradients

    expected_output, expected_state, expected_gradients = evaluate(contiguous)
    actual_output, actual_state, actual_gradients = evaluate(strided)
    assert torch.equal(actual_output, expected_output)
    assert torch.equal(actual_state, expected_state)
    for actual, expected in zip(actual_gradients, expected_gradients):
        assert torch.equal(actual, expected)


def test_direct_e_rejects_non_c32_chunk_size() -> None:
    runtime = _runtime_inputs(1)
    with pytest.raises(ValueError, match="requires chunk_size=32"):
        wy_associative(
            *runtime[:5],
            initial_state=runtime[5],
            chunk_size=16,
        )
