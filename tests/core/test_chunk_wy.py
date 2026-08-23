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
    difference = actual.double() - expected.double()
    rho = difference.square().mean().sqrt() / (
        expected.double().square().mean().sqrt() + 1e-8
    )
    return rho.item(), difference.abs().max().item()


def _assert_budget(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    rho_ceiling: float,
    name: str,
) -> None:
    assert torch.isfinite(actual).all(), name
    rho, absolute = _metrics(expected, actual)
    absolute_ceiling = 2e-4 if actual.dtype == torch.bfloat16 else 1e-6
    assert absolute <= absolute_ceiling or rho <= rho_ceiling, (
        f"{name}: rho={rho:.6e}, a_inf={absolute:.6e}, "
        f"ceiling={rho_ceiling:.6e}"
    )


def _inputs(
    *,
    length: int,
    value_dim: int = 32,
    batch: int = 1,
    heads: int = 1,
    seed: int = 916,
) -> tuple[tuple[torch.Tensor, ...], SolveDeltaState]:
    torch.manual_seed(seed + length)
    rank = 128
    prefix = (batch, length, heads)

    def bf16(value: torch.Tensor) -> torch.Tensor:
        return value.to(torch.bfloat16)

    inputs = (
        bf16(torch.randn(*prefix, rank, device="cuda")),
        bf16(0.1 * torch.randn(*prefix, rank, device="cuda")),
        bf16(torch.randn(*prefix, rank, device="cuda")),
        bf16(torch.randn(*prefix, 1, rank, device="cuda")),
        bf16(0.2 * torch.randn(*prefix, 1, value_dim, device="cuda")),
        -0.03 * torch.rand(*prefix, device="cuda"),
        -0.02 * torch.rand(*prefix, rank, device="cuda"),
        bf16(0.3 + 0.4 * torch.rand(*prefix, 1, rank, device="cuda")),
        bf16(
            0.3
            + 0.4 * torch.rand(*prefix, 1, value_dim, device="cuda")
        ),
        0.1 * torch.rand(heads, device="cuda"),
    )
    initial = SolveDeltaState(
        0.2 * torch.rand(batch, heads, device="cuda"),
        0.02 * torch.randn(batch, heads, rank, rank, device="cuda"),
        0.02 * torch.randn(batch, heads, rank, rank, device="cuda"),
        0.03
        * torch.randn(batch, heads, rank, value_dim, device="cuda"),
    )
    return inputs, initial


def _oracle(
    inputs: tuple[torch.Tensor, ...],
    initial: SolveDeltaState | None,
) -> tuple[torch.Tensor, SolveDeltaState]:
    promoted = None
    if initial is not None:
        promoted = SolveDeltaState(*(tensor.double() for tensor in initial))
    return solvedelta_reference(
        *(tensor.double() for tensor in inputs), initial_state=promoted
    )


def test_chunk_wy_cross_chunk_forward_state_and_vjp_match_fp64() -> None:
    master_inputs, master_state = _inputs(length=33)
    actual_inputs = tuple(
        tensor.detach().requires_grad_(True) for tensor in master_inputs
    )
    actual_initial = SolveDeltaState(
        *(tensor.detach().requires_grad_(True) for tensor in master_state)
    )
    actual_output, actual_state = chunk_wy_solvedelta(
        *actual_inputs, initial_state=actual_initial
    )

    reference_inputs = tuple(
        tensor.detach().double().requires_grad_(True)
        for tensor in master_inputs
    )
    reference_initial = SolveDeltaState(
        *(
            tensor.detach().double().requires_grad_(True)
            for tensor in master_state
        )
    )
    reference_output, reference_state = solvedelta_reference(
        *reference_inputs, initial_state=reference_initial
    )

    assert actual_output.dtype == torch.bfloat16
    assert all(tensor.dtype == torch.float32 for tensor in actual_state)
    _assert_budget(
        reference_output,
        actual_output,
        rho_ceiling=6e-3,
        name="output",
    )
    for name, expected, actual in zip(
        SolveDeltaState._fields, reference_state, actual_state
    ):
        _assert_budget(
            expected,
            actual,
            rho_ceiling=6e-3 if name == "S" else 5e-3,
            name=f"final_state.{name}",
        )

    torch.manual_seed(20260825)
    output_cotangent = torch.randn_like(actual_output)
    state_cotangents = SolveDeltaState(
        *(torch.randn_like(tensor) for tensor in actual_state)
    )
    actual_loss = (actual_output.float() * output_cotangent.float()).sum()
    actual_loss = actual_loss + sum(
        (tensor * cotangent).sum()
        for tensor, cotangent in zip(actual_state, state_cotangents)
    )
    reference_loss = (
        reference_output * output_cotangent.double()
    ).sum()
    reference_loss = reference_loss + sum(
        (tensor * cotangent.double()).sum()
        for tensor, cotangent in zip(reference_state, state_cotangents)
    )
    actual_gradients = torch.autograd.grad(
        actual_loss, actual_inputs + tuple(actual_initial)
    )
    reference_gradients = torch.autograd.grad(
        reference_loss, reference_inputs + tuple(reference_initial)
    )
    for name, expected, actual in zip(
        _INPUT_NAMES + _STATE_NAMES,
        reference_gradients,
        actual_gradients,
    ):
        if name in {"J0", "D0", "m0"}:
            ceiling = 5e-4
        elif name in {"q", "keys", "values", "S0"}:
            ceiling = 1.5e-2
        else:
            ceiling = 2.5e-2
        _assert_budget(
            expected,
            actual,
            rho_ceiling=ceiling,
            name=f"grad_{name}",
        )


@pytest.mark.parametrize("length", (3, 31, 32, 33))
def test_chunk_wy_zero_initial_tail_forward_backward_is_finite(
    length: int,
) -> None:
    inputs, _ = _inputs(length=length, batch=2, seed=1700)
    variables = tuple(
        tensor.detach().requires_grad_(True) for tensor in inputs
    )
    output, final = chunk_wy_solvedelta(*variables)
    assert output.shape == (2, length, 1, 32)
    assert output.dtype == torch.bfloat16
    assert all(tensor.dtype == torch.float32 for tensor in final)
    cotangents = (torch.randn_like(output),) + tuple(
        torch.randn_like(tensor) for tensor in final
    )
    gradients = torch.autograd.grad(
        (output, *final), variables, cotangents
    )
    for name, gradient, runtime in zip(_INPUT_NAMES, gradients, inputs):
        assert gradient.shape == runtime.shape, name
        assert gradient.dtype == runtime.dtype, name
        assert torch.isfinite(gradient).all(), name


def test_chunk_wy_multi_panel_is_bitwise_repeatable() -> None:
    master_inputs, master_state = _inputs(
        length=3, batch=2, heads=2, seed=2100
    )
    torch.manual_seed(20260827)
    output_cotangent = torch.randn(
        2, 3, 2, 32, device="cuda", dtype=torch.bfloat16
    )

    def evaluate() -> tuple[torch.Tensor, SolveDeltaState, tuple[torch.Tensor, ...]]:
        inputs = tuple(
            tensor.detach().requires_grad_(True) for tensor in master_inputs
        )
        initial = SolveDeltaState(
            *(tensor.detach().requires_grad_(True) for tensor in master_state)
        )
        output, final = chunk_wy_solvedelta(*inputs, initial_state=initial)
        state_cotangents = SolveDeltaState(
            *(torch.ones_like(tensor) for tensor in final)
        )
        loss = (output * output_cotangent).float().sum()
        loss = loss + sum(
            (tensor * cotangent).sum()
            for tensor, cotangent in zip(final, state_cotangents)
        )
        gradients = torch.autograd.grad(loss, inputs + tuple(initial))
        return output, final, gradients

    first_output, first_state, first_gradients = evaluate()
    second_output, second_state, second_gradients = evaluate()
    assert torch.equal(first_output, second_output)
    for left, right in zip(first_state, second_state):
        assert torch.equal(left, right)
    for name, left, right in zip(
        _INPUT_NAMES + _STATE_NAMES, first_gradients, second_gradients
    ):
        assert torch.isfinite(left).all(), name
        assert torch.equal(left, right), name


def test_chunk_wy_rejects_non_native_contract() -> None:
    inputs, initial = _inputs(length=3)
    with pytest.raises(TypeError, match="u must be BF16"):
        chunk_wy_solvedelta(inputs[0].float(), *inputs[1:])

    malformed = initial._replace(S=initial.S[..., :-1])
    with pytest.raises(ValueError, match=r"initial_state.S must have shape"):
        chunk_wy_solvedelta(*inputs, initial_state=malformed)

    keys = torch.cat((inputs[3], inputs[3]), dim=3)
    with pytest.raises(ValueError, match=r"keys must have shape \[B,T,H,1,128\]"):
        chunk_wy_solvedelta(
            inputs[0],
            inputs[1],
            inputs[2],
            keys,
            *inputs[4:],
        )


def test_chunk_wy_rejects_unsupported_cuda_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _ = _inputs(length=3)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    with pytest.raises(NotImplementedError, match="requires an SM120"):
        chunk_wy_solvedelta(*inputs)
