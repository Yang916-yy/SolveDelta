from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from causallsso.ops.native_chunk import _load_chunk_library
from causallsso.ops.resident_frame import resident_c32_frame_backward
from frame_oracle import (
    _frame_oracle_untied_strength,
    frame_oracle as chunk_frame,
)


_INPUT_NAMES = (
    "u",
    "h",
    "geometry_log_decay",
    "key",
    "erase",
    "query",
    "geometry_strength",
    "boundary_m",
    "boundary_J",
    "boundary_D",
)

_GEOMETRY_RHO_CEILING = 2.5e-2
_FP32_ABSOLUTE_CEILING = 1e-6
_STRENGTH_CHANNELS = 6


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)


def _inputs(
    *,
    length: int,
    batch: int = 1,
    heads: int = 1,
    strength: float = 0.7,
    seed: int = 20260824,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed + length)
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


def _cancellation_inputs(kind: str) -> tuple[torch.Tensor, ...]:
    inputs = list(_inputs(length=32, seed=1947 + (kind == "D")))

    direction_master = torch.tensor(
        [0.6953125, 0.71875], device="cuda", dtype=torch.float64
    )
    direction = F.normalize(direction_master, dim=0).to(torch.bfloat16)
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
    inputs[2][:, 3] = -110.0
    inputs[2][:, 7] = -1000.0
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


def _zero_mass_boundary_inputs() -> tuple[torch.Tensor, ...]:
    inputs = list(_inputs(length=5, seed=2718))
    inputs[7].zero_()
    assert torch.count_nonzero(inputs[8]) > 0
    assert torch.count_nonzero(inputs[9]) > 0
    return tuple(inputs)


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


def _assert_gradient(
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    assert torch.isfinite(actual).all(), name
    rho, absolute = _metrics(expected, actual)
    rho_ceiling = 1e-2 if name.rsplit(".", 1)[-1] in {
        "key",
        "erase",
        "query",
    } else _GEOMETRY_RHO_CEILING
    absolute_ceiling = (
        2e-4 if actual.dtype == torch.bfloat16 else _FP32_ABSOLUTE_CEILING
    )
    assert absolute <= absolute_ceiling or rho <= rho_ceiling, (
        f"{name}: rho={rho:.6e}, a_inf={absolute:.6e}, "
        f"ceiling={rho_ceiling:.6e}"
    )


def _untied_strength_vjp(
    master_inputs: tuple[torch.Tensor, ...],
    output_cotangents: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    inputs = tuple(tensor.detach().double() for tensor in master_inputs)
    tied_strength = inputs[6]
    untied_strength = (
        tied_strength.unsqueeze(0)
        .expand(_STRENGTH_CHANNELS, *tied_strength.shape)
        .clone()
        .requires_grad_(True)
    )
    diagnostic_inputs = (*inputs[:6], untied_strength, *inputs[7:])
    outputs = _frame_oracle_untied_strength(*diagnostic_inputs)
    (gradient,) = torch.autograd.grad(
        outputs,
        untied_strength,
        tuple(cotangent.double() for cotangent in output_cotangents),
    )
    return outputs, gradient


def _assert_tied_strength_gradient(
    name: str,
    expected_shared: torch.Tensor,
    actual_shared: torch.Tensor,
    expected_untied: torch.Tensor,
) -> None:
    expected_shared = expected_shared.double()
    actual_shared = actual_shared.double()
    expected_sum = expected_untied.sum(dim=0)
    torch.testing.assert_close(
        expected_sum,
        expected_shared,
        rtol=1e-12,
        atol=1e-12,
    )

    absolute = (actual_shared - expected_sum).abs()
    scale = math.sqrt(_STRENGTH_CHANNELS) * expected_untied.norm(dim=0)
    rho_tie = absolute / (scale + 1e-8)
    accepted = (absolute <= _FP32_ABSOLUTE_CEILING) | (
        rho_tie <= _GEOMETRY_RHO_CEILING
    )
    assert accepted.all(), (
        f"{name}: rho_tie={float(rho_tie.max()):.6e}, "
        f"a_inf={float(absolute.max()):.6e}, "
        f"ceiling={_GEOMETRY_RHO_CEILING:.6e}"
    )


def _vjp(
    implementation: Callable[..., tuple[torch.Tensor, ...]],
    master_inputs: tuple[torch.Tensor, ...],
    output_cotangents: tuple[torch.Tensor, ...],
    *,
    reference: bool,
) -> tuple[torch.Tensor | None, ...]:
    variables = tuple(
        (
            tensor.detach().double()
            if reference
            else tensor.detach()
        ).requires_grad_(True)
        for tensor in master_inputs
    )
    outputs = implementation(*variables)
    return torch.autograd.grad(
        outputs,
        variables,
        tuple(
            cotangent.double() if reference else cotangent
            for cotangent in output_cotangents
        ),
        allow_unused=True,
    )


class _TestResidentFrame(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        inputs = tuple(tensor.contiguous() for tensor in inputs)
        _load_chunk_library()
        outputs = tuple(
            torch.ops.causallsso.c32_frame_resident_forward(*inputs)
        )
        ctx.save_for_backward(*inputs, *outputs[3:])
        return outputs[:3]

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        saved = ctx.saved_tensors
        inputs = saved[: len(_INPUT_NAMES)]
        auxiliaries = saved[len(_INPUT_NAMES) :]
        return resident_c32_frame_backward(
            *inputs,
            *auxiliaries,
            *(gradient.contiguous() for gradient in grad_outputs),
        )


def _native(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return _TestResidentFrame.apply(*inputs)


def _reference(
    *inputs: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return chunk_frame(*inputs, chunk_size=32)


def _cotangents(
    outputs: tuple[torch.Tensor, ...],
    *,
    route: str,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    values = [
        torch.zeros(output.shape, device=output.device, dtype=torch.bfloat16)
        for output in outputs
    ]
    if route == "all":
        values = [
            torch.randn(output.shape, device=output.device).bfloat16()
            for output in outputs
        ]
    elif route == "joint":
        values[1] = torch.randn(outputs[1].shape, device=outputs[1].device).bfloat16()
        values[2] = torch.randn(outputs[2].shape, device=outputs[2].device).bfloat16()
    else:
        index = {"write": 0, "erase": 1, "read": 2}[route]
        values[index] = torch.randn(
            outputs[index].shape, device=outputs[index].device
        ).bfloat16()
    return tuple(values)


def test_untied_strength_diagnostic_ties_to_shared_oracle() -> None:
    inputs = _inputs(length=3, heads=2, seed=2900)
    reference_inputs = tuple(tensor.detach().double() for tensor in inputs)
    shared_strength = reference_inputs[6].detach().requires_grad_(True)
    shared_inputs = (*reference_inputs[:6], shared_strength, *reference_inputs[7:])
    shared_outputs = _reference(*shared_inputs)
    cotangents = _cotangents(shared_outputs, route="all", seed=3000)
    (shared_gradient,) = torch.autograd.grad(
        shared_outputs,
        shared_strength,
        tuple(cotangent.double() for cotangent in cotangents),
    )
    untied_outputs, untied_gradient = _untied_strength_vjp(inputs, cotangents)

    for shared_output, untied_output in zip(shared_outputs, untied_outputs):
        torch.testing.assert_close(
            untied_output,
            shared_output,
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(
        untied_gradient.sum(dim=0),
        shared_gradient,
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.count_nonzero(untied_gradient) == untied_gradient.numel()


@pytest.mark.parametrize("route", ("erase", "read", "joint"))
def test_native_chunk_dual_routes_match_quantized_fp64(
    route: str,
) -> None:
    inputs = _inputs(length=3, seed=3100)
    with torch.no_grad():
        outputs = _reference(
            *(tensor.double() for tensor in inputs),
        )
        cotangents = _cotangents(outputs, route=route, seed=3300 + len(route))
    expected = _vjp(
        _reference,
        inputs,
        cotangents,
        reference=True,
    )
    actual = _vjp(
        _native,
        inputs,
        cotangents,
        reference=False,
    )
    for name, expected_gradient, actual_gradient in zip(
        _INPUT_NAMES, expected, actual
    ):
        assert expected_gradient is not None, name
        assert actual_gradient is not None, name
        _assert_gradient(name, expected_gradient, actual_gradient)
    if route == "erase":
        assert torch.count_nonzero(actual[5]) == 0
    elif route == "read":
        assert torch.count_nonzero(actual[3]) == 0
        assert torch.count_nonzero(actual[4]) == 0


def test_native_chunk_t3_complete_vjp_matches_quantized_fp64() -> None:
    inputs = _inputs(length=3, seed=3700)
    with torch.no_grad():
        outputs = _reference(*(tensor.double() for tensor in inputs))
        cotangents = _cotangents(outputs, route="all", seed=3800)
    expected = _vjp(
        _reference, inputs, cotangents, reference=True
    )
    actual = _vjp(
        _native, inputs, cotangents, reference=False
    )
    for name, expected_gradient, actual_gradient, runtime in zip(
        _INPUT_NAMES, expected, actual, inputs
    ):
        assert expected_gradient is not None, name
        assert actual_gradient is not None, name
        assert actual_gradient.dtype == runtime.dtype, name
        _assert_gradient(f"T3.{name}", expected_gradient, actual_gradient)


@pytest.mark.parametrize(
    "case",
    ("tail", "identity", "zero_mass", "J", "D"),
)
def test_native_chunk_complete_vjp_envelope(
    case: str,
) -> None:
    if case == "tail":
        inputs = _inputs(length=33, seed=4100)
    elif case == "identity":
        inputs = _inputs(length=31, strength=0.0, seed=4200)
    elif case == "zero_mass":
        inputs = _zero_mass_boundary_inputs()
    else:
        inputs = _cancellation_inputs(case)
        assert _cancellation_ratio(case, inputs) == 4095.0

    with torch.no_grad():
        outputs = _reference(
            *(tensor.double() for tensor in inputs),
        )
        cotangents = _cotangents(outputs, route="all", seed=4300 + len(case))
    expected = _vjp(
        _reference,
        inputs,
        cotangents,
        reference=True,
    )
    actual = _vjp(
        _native,
        inputs,
        cotangents,
        reference=False,
    )
    expected_untied_strength = None
    if case in {"J", "D"}:
        _, expected_untied_strength = _untied_strength_vjp(inputs, cotangents)
    for name, expected_gradient, actual_gradient in zip(
        _INPUT_NAMES, expected, actual
    ):
        assert expected_gradient is not None, name
        assert actual_gradient is not None, name
        if name == "geometry_strength" and expected_untied_strength is not None:
            _assert_tied_strength_gradient(
                f"{case}.{name}",
                expected_gradient,
                actual_gradient,
                expected_untied_strength,
            )
        else:
            _assert_gradient(f"{case}.{name}", expected_gradient, actual_gradient)


def test_native_chunk_backward_is_bitwise_repeatable() -> None:
    inputs = _inputs(length=33, seed=5100)
    with torch.no_grad():
        outputs = _reference(
            *(tensor.double() for tensor in inputs)
        )
        cotangents = _cotangents(outputs, route="all", seed=5200)
    first = _vjp(
        _native,
        inputs,
        cotangents,
        reference=False,
    )
    second = _vjp(
        _native,
        inputs,
        cotangents,
        reference=False,
    )
    for name, left, right in zip(_INPUT_NAMES, first, second):
        assert left is not None, name
        assert right is not None, name
        assert torch.equal(left, right), name


def test_native_chunk_multi_panel_vjp_matches_fp64_and_is_repeatable() -> None:
    inputs = _inputs(length=3, batch=2, heads=2, seed=6100)
    with torch.no_grad():
        outputs = _reference(
            *(tensor.double() for tensor in inputs)
        )
        cotangents = _cotangents(outputs, route="all", seed=6200)
    expected = _vjp(
        _reference,
        inputs,
        cotangents,
        reference=True,
    )
    first = _vjp(
        _native,
        inputs,
        cotangents,
        reference=False,
    )
    second = _vjp(
        _native,
        inputs,
        cotangents,
        reference=False,
    )
    for name, expected_gradient, left, right in zip(
        _INPUT_NAMES, expected, first, second
    ):
        assert expected_gradient is not None, name
        assert left is not None, name
        assert right is not None, name
        _assert_gradient(f"multi_panel.{name}", expected_gradient, left)
        assert torch.equal(left, right), name


@pytest.mark.parametrize("length", (31, 32, 33))
def test_native_chunk_tail_forward_and_backward_are_finite(
    length: int,
) -> None:
    inputs = _inputs(length=length, batch=2, seed=7100)
    variables = tuple(tensor.detach().requires_grad_(True) for tensor in inputs)
    outputs = _native(*variables)
    cotangents = _cotangents(outputs, route="all", seed=7200 + length)
    gradients = torch.autograd.grad(outputs, variables, cotangents)
    for output in outputs:
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()
    for name, gradient, runtime in zip(_INPUT_NAMES, gradients, inputs):
        assert gradient.shape == runtime.shape, name
        assert gradient.dtype == runtime.dtype, name
        assert torch.isfinite(gradient).all(), name
