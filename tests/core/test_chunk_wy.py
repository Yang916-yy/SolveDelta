from __future__ import annotations

import importlib.util

import pytest
import torch

from causallsso.ops.chunk_wy import chunk_wy_solvedelta
from causallsso.reference import SolveDeltaState, solvedelta_reference


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("fla") is None,
    reason="CUDA and flash-linear-attention are required",
)


_INPUT_NAMES = (
    "u",
    "h",
    "q",
    "keys",
    "values",
    "geometry_log_decay",
    "associative_log_decay",
    "erase",
    "write",
    "geometry_strength",
)
_STATE_NAMES = ("m0", "J0", "D0", "S0")


def _metrics(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> tuple[float, float]:
    expected64 = expected.double()
    actual64 = actual.double()
    difference = actual64 - expected64
    rho = difference.square().mean().sqrt() / (
        expected64.square().mean().sqrt() + 1e-8
    )
    return rho.item(), difference.abs().max().item()


def _assert_with_budget(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    rho_ceiling: float,
    name: str,
) -> None:
    assert torch.isfinite(actual).all(), name
    rho, absolute = _metrics(expected, actual)
    assert absolute <= 1e-6 or rho <= rho_ceiling, (
        f"{name}: rho={rho:.6e}, a_inf={absolute:.6e}, "
        f"ceiling={rho_ceiling:.6e}"
    )


def _inputs(
    edits: int,
    *,
    length: int = 5,
    rank: int = 32,
    value_dim: int = 16,
    batch: int = 1,
    heads: int = 1,
) -> tuple[tuple[torch.Tensor, ...], SolveDeltaState]:
    torch.manual_seed(916 + edits)
    device = torch.device("cuda")
    prefix = (batch, length, heads)
    inputs = (
        torch.randn(*prefix, rank, device=device),
        0.1 * torch.randn(*prefix, rank, device=device),
        torch.randn(*prefix, rank, device=device),
        torch.randn(*prefix, edits, rank, device=device),
        0.2 * torch.randn(*prefix, edits, value_dim, device=device),
        -0.03 * torch.rand(*prefix, device=device),
        -0.02 * torch.rand(*prefix, rank, device=device),
        0.3 + 0.4 * torch.rand(*prefix, edits, rank, device=device),
        0.3 + 0.4 * torch.rand(*prefix, edits, value_dim, device=device),
        0.1 * torch.rand(heads, device=device),
    )
    initial_state = SolveDeltaState(
        0.2 * torch.rand(batch, heads, device=device),
        0.02 * torch.randn(batch, heads, rank, rank, device=device),
        0.02 * torch.randn(batch, heads, rank, rank, device=device),
        0.03 * torch.randn(batch, heads, rank, value_dim, device=device),
    )
    return inputs, initial_state


def _fp64_oracle(
    inputs: tuple[torch.Tensor, ...],
    initial_state: SolveDeltaState | None,
) -> tuple[torch.Tensor, SolveDeltaState]:
    promoted_initial = None
    if initial_state is not None:
        promoted_initial = SolveDeltaState(
            *(tensor.double() for tensor in initial_state)
        )
    return solvedelta_reference(
        *(tensor.double() for tensor in inputs),
        initial_state=promoted_initial,
    )


@pytest.mark.parametrize("edits", [1, 2, 4])
@pytest.mark.parametrize(
    ("wy_dtype", "outer_ceiling"),
    [(torch.float16, 5e-3), (torch.bfloat16, 6e-3)],
)
def test_chunk_wy_forward_and_final_state_match_fp64_oracle(
    edits: int,
    wy_dtype: torch.dtype,
    outer_ceiling: float,
) -> None:
    inputs, initial_state = _inputs(edits)
    actual_output, actual_state = chunk_wy_solvedelta(
        *inputs,
        initial_state=initial_state,
        backend="reference",
        wy_dtype=wy_dtype,
    )
    expected_output, expected_state = _fp64_oracle(inputs, initial_state)

    assert actual_output.dtype == wy_dtype
    assert all(tensor.dtype == torch.float32 for tensor in actual_state)
    _assert_with_budget(
        expected_output,
        actual_output,
        rho_ceiling=outer_ceiling,
        name="output",
    )
    for name, expected, actual in zip(
        SolveDeltaState._fields, expected_state, actual_state
    ):
        ceiling = outer_ceiling if name == "S" else 2e-4
        _assert_with_budget(
            expected,
            actual,
            rho_ceiling=ceiling,
            name=f"final_state.{name}",
        )


def test_chunk_wy_crosses_c32_boundary_with_selected_default_dtype() -> None:
    inputs, initial_state = _inputs(edits=1, length=33)
    actual_output, actual_state = chunk_wy_solvedelta(
        *inputs,
        initial_state=initial_state,
        backend="reference",
    )
    expected_output, expected_state = _fp64_oracle(inputs, initial_state)

    assert actual_output.dtype == torch.float16
    _assert_with_budget(
        expected_output,
        actual_output,
        rho_ceiling=5e-3,
        name="cross_chunk.output",
    )
    for name, expected, actual in zip(
        SolveDeltaState._fields, expected_state, actual_state
    ):
        ceiling = 5e-3 if name == "S" else 2e-4
        _assert_with_budget(
            expected,
            actual,
            rho_ceiling=ceiling,
            name=f"cross_chunk.final_state.{name}",
        )


@pytest.mark.parametrize(
    ("edits", "length"),
    [(1, 33), (2, 5), (4, 5)],
)
def test_chunk_wy_reference_staging_vjp_matches_fp64_oracle(
    edits: int,
    length: int,
) -> None:
    master_inputs, master_state = _inputs(edits, length=length)
    actual_inputs = tuple(
        tensor.detach().requires_grad_(True) for tensor in master_inputs
    )
    actual_initial = SolveDeltaState(
        *(tensor.detach().requires_grad_(True) for tensor in master_state)
    )
    actual_output, actual_state = chunk_wy_solvedelta(
        *actual_inputs,
        initial_state=actual_initial,
        backend="reference",
        wy_dtype=torch.float16,
    )

    torch.manual_seed(1200 + edits)
    output_cotangent = torch.randn_like(actual_output, dtype=torch.float32)
    state_cotangent = SolveDeltaState(
        *(torch.randn_like(tensor) for tensor in actual_state)
    )
    actual_loss = (actual_output.float() * output_cotangent).sum()
    actual_loss = actual_loss + sum(
        (tensor * cotangent).sum()
        for tensor, cotangent in zip(actual_state, state_cotangent)
    )
    actual_gradients = torch.autograd.grad(
        actual_loss,
        actual_inputs + tuple(actual_initial),
    )

    expected_inputs = tuple(
        tensor.detach().double().requires_grad_(True)
        for tensor in master_inputs
    )
    expected_initial = SolveDeltaState(
        *(
            tensor.detach().double().requires_grad_(True)
            for tensor in master_state
        )
    )
    expected_output, expected_state = solvedelta_reference(
        *expected_inputs,
        initial_state=expected_initial,
    )
    expected_loss = (expected_output * output_cotangent.double()).sum()
    expected_loss = expected_loss + sum(
        (tensor * cotangent.double()).sum()
        for tensor, cotangent in zip(expected_state, state_cotangent)
    )
    expected_gradients = torch.autograd.grad(
        expected_loss,
        expected_inputs + tuple(expected_initial),
    )

    vector_gradients = {"q", "keys", "values", "S0"}
    names = _INPUT_NAMES + _STATE_NAMES
    for name, expected, actual in zip(
        names, expected_gradients, actual_gradients
    ):
        ceiling = 1e-2 if name in vector_gradients else 2e-2
        _assert_with_budget(
            expected,
            actual,
            rho_ceiling=ceiling,
            name=f"grad_{name}",
        )


def test_chunk_wy_native_backend_does_not_fall_back() -> None:
    inputs, initial_state = _inputs(edits=1)
    with pytest.raises(ValueError, match="requires r=128 and K=1"):
        chunk_wy_solvedelta(
            *inputs,
            initial_state=initial_state,
            backend="native",
            wy_dtype=torch.float16,
        )


def test_chunk_wy_native_composition_matches_fp64_forward_state_and_vjp() -> None:
    master_inputs, master_state = _inputs(
        edits=1,
        length=33,
        rank=128,
        value_dim=32,
    )
    actual_inputs = tuple(
        tensor.detach().requires_grad_(True) for tensor in master_inputs
    )
    actual_initial = SolveDeltaState(
        *(tensor.detach().requires_grad_(True) for tensor in master_state)
    )
    actual_output, actual_state = chunk_wy_solvedelta(
        *actual_inputs,
        initial_state=actual_initial,
        backend="native",
        wy_dtype=torch.float16,
    )

    expected_inputs = tuple(
        tensor.detach().double().requires_grad_(True)
        for tensor in master_inputs
    )
    expected_initial = SolveDeltaState(
        *(
            tensor.detach().double().requires_grad_(True)
            for tensor in master_state
        )
    )
    expected_output, expected_state = solvedelta_reference(
        *expected_inputs,
        initial_state=expected_initial,
    )

    _assert_with_budget(
        expected_output,
        actual_output,
        rho_ceiling=5e-3,
        name="native.output",
    )
    for name, expected, actual in zip(
        SolveDeltaState._fields, expected_state, actual_state
    ):
        ceiling = 5e-3 if name == "S" else 2e-4
        _assert_with_budget(
            expected,
            actual,
            rho_ceiling=ceiling,
            name=f"native.final_state.{name}",
        )

    torch.manual_seed(20260825)
    output_cotangent = torch.randn_like(actual_output, dtype=torch.float32)
    state_cotangents = SolveDeltaState(
        *(torch.randn_like(tensor) for tensor in actual_state)
    )
    actual_loss = (actual_output.float() * output_cotangent).sum()
    actual_loss = actual_loss + sum(
        (tensor * cotangent).sum()
        for tensor, cotangent in zip(actual_state, state_cotangents)
    )
    expected_loss = (expected_output * output_cotangent.double()).sum()
    expected_loss = expected_loss + sum(
        (tensor * cotangent.double()).sum()
        for tensor, cotangent in zip(expected_state, state_cotangents)
    )
    actual_gradients = torch.autograd.grad(
        actual_loss,
        actual_inputs + tuple(actual_initial),
    )
    expected_gradients = torch.autograd.grad(
        expected_loss,
        expected_inputs + tuple(expected_initial),
    )

    vector_gradients = {"q", "keys", "values", "S0"}
    for name, expected, actual in zip(
        _INPUT_NAMES + _STATE_NAMES,
        expected_gradients,
        actual_gradients,
    ):
        ceiling = 1e-2 if name in vector_gradients else 2e-2
        _assert_with_budget(
            expected,
            actual,
            rho_ceiling=ceiling,
            name=f"native.grad_{name}",
        )


def test_chunk_wy_native_zero_initial_forward_and_vjp_smoke() -> None:
    master_inputs, _ = _inputs(
        edits=1,
        length=3,
        rank=128,
        value_dim=32,
    )
    actual_inputs = tuple(
        tensor.detach().requires_grad_(True) for tensor in master_inputs
    )
    actual_output, actual_state = chunk_wy_solvedelta(
        *actual_inputs,
        initial_state=None,
        backend="native",
        wy_dtype=torch.float16,
    )

    expected_inputs = tuple(
        tensor.detach().double().requires_grad_(True)
        for tensor in master_inputs
    )
    expected_output, expected_state = solvedelta_reference(
        *expected_inputs,
        initial_state=None,
    )
    _assert_with_budget(
        expected_output,
        actual_output,
        rho_ceiling=5e-3,
        name="native.zero_initial.output",
    )
    for name, expected, actual in zip(
        SolveDeltaState._fields, expected_state, actual_state
    ):
        ceiling = 5e-3 if name == "S" else 2e-4
        _assert_with_budget(
            expected,
            actual,
            rho_ceiling=ceiling,
            name=f"native.zero_initial.final_state.{name}",
        )

    torch.manual_seed(20260826)
    output_cotangent = torch.randn_like(actual_output, dtype=torch.float32)
    state_cotangents = SolveDeltaState(
        *(torch.randn_like(tensor) for tensor in actual_state)
    )
    actual_loss = (actual_output.float() * output_cotangent).sum()
    actual_loss = actual_loss + sum(
        (tensor * cotangent).sum()
        for tensor, cotangent in zip(actual_state, state_cotangents)
    )
    expected_loss = (expected_output * output_cotangent.double()).sum()
    expected_loss = expected_loss + sum(
        (tensor * cotangent.double()).sum()
        for tensor, cotangent in zip(expected_state, state_cotangents)
    )
    actual_gradients = torch.autograd.grad(actual_loss, actual_inputs)
    expected_gradients = torch.autograd.grad(expected_loss, expected_inputs)

    vector_gradients = {"q", "keys", "values"}
    for name, expected, actual in zip(
        _INPUT_NAMES,
        expected_gradients,
        actual_gradients,
    ):
        ceiling = 1e-2 if name in vector_gradients else 2e-2
        _assert_with_budget(
            expected,
            actual,
            rho_ceiling=ceiling,
            name=f"native.zero_initial.grad_{name}",
        )


def test_chunk_wy_native_multi_panel_composition_is_bitwise_repeatable() -> None:
    master_inputs, master_state = _inputs(
        edits=1,
        length=35,
        rank=128,
        value_dim=32,
        batch=2,
        heads=2,
    )
    torch.manual_seed(20260827)
    output_cotangent = torch.randn(
        2,
        35,
        2,
        32,
        device="cuda",
        dtype=torch.float32,
    )
    state_cotangents = SolveDeltaState(
        *(torch.randn_like(tensor) for tensor in master_state)
    )

    def evaluate() -> tuple[
        torch.Tensor,
        SolveDeltaState,
        tuple[torch.Tensor, ...],
    ]:
        inputs = tuple(
            tensor.detach().requires_grad_(True) for tensor in master_inputs
        )
        initial = SolveDeltaState(
            *(tensor.detach().requires_grad_(True) for tensor in master_state)
        )
        output, final_state = chunk_wy_solvedelta(
            *inputs,
            initial_state=initial,
            backend="native",
            wy_dtype=torch.float16,
        )
        loss = (output.float() * output_cotangent).sum()
        loss = loss + sum(
            (tensor * cotangent).sum()
            for tensor, cotangent in zip(final_state, state_cotangents)
        )
        gradients = torch.autograd.grad(
            loss,
            inputs + tuple(initial),
        )
        return output, final_state, gradients

    first_output, first_state, first_gradients = evaluate()
    second_output, second_state, second_gradients = evaluate()
    assert torch.equal(first_output, second_output)
    for name, first, second in zip(
        SolveDeltaState._fields, first_state, second_state
    ):
        assert torch.equal(first, second), name
    for name, first, second in zip(
        _INPUT_NAMES + _STATE_NAMES,
        first_gradients,
        second_gradients,
    ):
        assert torch.isfinite(first).all(), name
        assert torch.equal(first, second), name


def test_chunk_wy_validates_complete_initial_state() -> None:
    inputs, initial_state = _inputs(edits=1)
    malformed = initial_state._replace(S=initial_state.S[..., :-1])
    with pytest.raises(ValueError, match=r"initial_state.S must have shape"):
        chunk_wy_solvedelta(
            *inputs,
            initial_state=malformed,
            backend="reference",
            wy_dtype=torch.float16,
        )
