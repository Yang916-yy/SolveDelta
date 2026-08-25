from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F
import triton

from causallsso.ops.radial_compact import (
    _radial_compact_reverse_accumulate_trusted,
    radial_compact_forward,
    radial_compact_reverse,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_CHUNK = 32
_RANK = 128
_RADIUS = 1.0 / 8.0
_PRIVATE_SCALAR_RHO_CEILING = 1.0e-3
_PRIVATE_RADIAL_RHO_CEILING = 1.0e-2
_REPEATABILITY_RHO_CEILING = 1.0e-6


def _rho(reference: torch.Tensor, actual: torch.Tensor) -> float:
    reference = reference.double()
    actual = actual.double()
    return float(
        (actual - reference).square().mean().sqrt()
        / (reference.square().mean().sqrt() + 1e-8)
    )


def _inputs(panels: int = 2) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260824)
    u = F.normalize(
        torch.randn(panels, _CHUNK, _RANK, device="cuda").bfloat16().float(),
        dim=-1,
    ).half()
    h = (0.2 * torch.randn_like(u.float())).bfloat16()
    log_decay = -0.08 * torch.rand(panels, _CHUNK, device="cuda")
    strength = torch.linspace(0.35, 0.8, panels, device="cuda")
    boundary_m = torch.linspace(0.0, 2.0, panels, device="cuda")
    raw_j = 0.025 * torch.randn(panels, _RANK, _RANK, device="cuda")
    boundary_j = raw_j @ raw_j.transpose(-1, -2)
    boundary_d = 0.03 * torch.randn_like(boundary_j)
    return tuple(
        value.contiguous()
        for value in (
            u,
            h,
            log_decay,
            strength,
            boundary_m,
            boundary_j,
            boundary_d,
        )
    )


def _explicit_fp64_oracle(
    u: torch.Tensor,
    h: torch.Tensor,
    log_decay: torch.Tensor,
    strength: torch.Tensor,
    boundary_m: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    valid_count: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Tokenwise oracle after promoting the exact declared operand bits."""

    panels = u.shape[0]
    local_u = u.double()
    local_h = h.double()
    decay = torch.exp(log_decay.double())
    panel_strength = strength.double()
    inverse_mass = torch.zeros(panels, _CHUNK, device="cuda", dtype=torch.float64)
    theta = torch.zeros_like(inverse_mass)
    weights = torch.zeros(
        panels, _CHUNK, _CHUNK, device="cuda", dtype=torch.float64
    )
    radial_scale = torch.zeros(
        panels, _CHUNK, 4, device="cuda", dtype=torch.float64
    )
    radial_q2 = torch.full_like(radial_scale, _RADIUS * _RADIUS)
    diagonal = torch.ones(
        panels, _CHUNK, _RANK, device="cuda", dtype=torch.float64
    )

    for panel in range(panels):
        count = int(valid_count[panel])
        mass = boundary_m[panel].double()
        current_j = boundary_j[panel].double()
        current_d = boundary_d[panel].double()
        boundary_coefficient = torch.ones((), device="cuda", dtype=torch.float64)
        local_coefficients = torch.zeros(
            _CHUNK, device="cuda", dtype=torch.float64
        )
        for target in range(count):
            rho = decay[panel, target]
            mass = rho * mass + 1.0
            boundary_coefficient = rho * boundary_coefficient
            local_coefficients = rho * local_coefficients
            local_coefficients[target] = 1.0
            inverse_mass[panel, target] = 1.0 / mass
            theta[panel, target] = boundary_coefficient / mass
            weights[panel, target] = local_coefficients / mass

            u_t = local_u[panel, target]
            h_t = local_h[panel, target]
            current_j = rho * current_j + u_t[:, None] * u_t[None, :]
            current_d = rho * current_d + u_t[:, None] * h_t[None, :]
            normalized_j = current_j / mass
            normalized_d = current_d / mass
            route_norms = torch.stack(
                (
                    torch.tril(normalized_j, diagonal=-1).square().sum(),
                    torch.tril(normalized_d, diagonal=-1).square().sum(),
                    torch.triu(normalized_j, diagonal=1).square().sum(),
                    torch.triu(normalized_d, diagonal=1).square().sum(),
                )
            )
            q2 = (
                _RADIUS * _RADIUS
                + panel_strength[panel].square() * route_norms
            )
            radial_q2[panel, target] = q2
            radial_scale[panel, target] = (
                panel_strength[panel] * _RADIUS / torch.sqrt(q2)
            )

            diagonal_j = torch.diagonal(normalized_j) - 1.0 / _RANK
            diagonal_d = torch.diagonal(normalized_d)
            diagonal[panel, target] = torch.exp(
                _RADIUS
                * torch.tanh(panel_strength[panel] * diagonal_j / _RADIUS)
                + _RADIUS
                * torch.tanh(panel_strength[panel] * diagonal_d / _RADIUS)
            )

    return inverse_mass, theta, weights, radial_scale, radial_q2, diagonal


def _reverse_fp64_oracle(
    inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    grad_radial_scale: torch.Tensor,
    grad_log_diagonal: torch.Tensor,
    valid_count: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Independent dense token recurrence used only as a gradient oracle."""

    leaves = tuple(
        value.double().detach().requires_grad_() for value in inputs
    )
    u, h, log_decay, strength, boundary_m, boundary_j, boundary_d = leaves
    loss = torch.zeros((), device="cuda", dtype=torch.float64)
    for panel in range(u.shape[0]):
        count = int(valid_count[panel])
        mass = boundary_m[panel]
        current_j = boundary_j[panel]
        current_d = boundary_d[panel]
        for target in range(count):
            rho = torch.exp(log_decay[panel, target])
            local_u = u[panel, target]
            local_h = h[panel, target]
            mass = rho * mass + 1.0
            current_j = (
                rho * current_j + local_u[:, None] * local_u[None, :]
            )
            current_d = (
                rho * current_d + local_u[:, None] * local_h[None, :]
            )
            normalized_j = current_j / mass
            normalized_d = current_d / mass
            route_norms = torch.stack(
                (
                    torch.tril(normalized_j, diagonal=-1).square().sum(),
                    torch.tril(normalized_d, diagonal=-1).square().sum(),
                    torch.triu(normalized_j, diagonal=1).square().sum(),
                    torch.triu(normalized_d, diagonal=1).square().sum(),
                )
            )
            q2 = _RADIUS * _RADIUS + strength[panel].square() * route_norms
            radial_scale = strength[panel] * _RADIUS / torch.sqrt(q2)
            centered_j = torch.diagonal(normalized_j) - 1.0 / _RANK
            diagonal_d = torch.diagonal(normalized_d)
            log_diagonal = (
                _RADIUS
                * torch.tanh(strength[panel] * centered_j / _RADIUS)
                + _RADIUS
                * torch.tanh(strength[panel] * diagonal_d / _RADIUS)
            )
            loss = loss + (
                radial_scale * grad_radial_scale[panel, target].double()
            ).sum()
            loss = loss + (
                log_diagonal * grad_log_diagonal[panel, target].double()
            ).sum()
    return torch.autograd.grad(loss, leaves)


def _reverse(
    inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    grad_radial_scale: torch.Tensor,
    grad_log_diagonal: torch.Tensor,
    valid_count: torch.Tensor,
):
    output, saved = radial_compact_forward(
        *inputs,
        valid_count=valid_count,
        return_saved=True,
    )
    return radial_compact_reverse(
        *inputs,
        output,
        saved,
        grad_radial_scale,
        grad_log_diagonal,
        valid_count=valid_count,
    )


def _assert_reverse_matches(
    reference: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...],
    *,
    ceiling: float,
    strength_ceiling: float | None = None,
) -> None:
    for index, (expected, observed) in enumerate(
        zip(reference, actual, strict=True)
    ):
        local_ceiling = (
            strength_ceiling
            if index == 3 and strength_ceiling is not None
            else ceiling
        )
        assert torch.isfinite(observed).all()
        error = _rho(expected, observed)
        assert error < local_ceiling, (index, error, local_ceiling)


def test_radial_compact_stays_within_private_safety_envelope() -> None:
    inputs = _inputs()
    valid_count = torch.full((2,), _CHUNK, device="cuda", dtype=torch.int32)
    actual = radial_compact_forward(*inputs, valid_count=valid_count)
    expected = _explicit_fp64_oracle(*inputs, valid_count)

    assert _rho(expected[0], actual.inverse_mass) < _PRIVATE_SCALAR_RHO_CEILING
    assert _rho(expected[1], actual.theta) < _PRIVATE_SCALAR_RHO_CEILING
    assert _rho(expected[2], actual.weights) < _PRIVATE_SCALAR_RHO_CEILING
    for route in range(4):
        assert (
            _rho(expected[3][..., route], actual.radial_scale[..., route])
            < _PRIVATE_RADIAL_RHO_CEILING
        )
        assert (
            _rho(expected[4][..., route], actual.radial_q2[..., route])
            < _PRIVATE_RADIAL_RHO_CEILING
        )
    assert _rho(expected[5], actual.diagonal) < _PRIVATE_RADIAL_RHO_CEILING

    # The asymmetric D fixture must exercise distinct lower/upper paths.
    assert not torch.equal(actual.radial_q2[..., 1], actual.radial_q2[..., 3])


def test_radial_compact_nonzero_tail_is_structurally_invalid() -> None:
    inputs = list(_inputs())
    valid_count = torch.tensor([5, 19], device="cuda", dtype=torch.int32)
    clean = [value.clone() for value in inputs]
    dirty = [value.clone() for value in inputs]
    for panel, count in enumerate((5, 19)):
        clean[0][panel, count:] = 0
        clean[1][panel, count:] = 0
        clean[2][panel, count:] = 0
        dirty[0][panel, count:] = torch.randn_like(dirty[0][panel, count:])
        dirty[1][panel, count:] = torch.randn_like(dirty[1][panel, count:])
        dirty[2][panel, count:] = 3.0

    clean_output = radial_compact_forward(
        *(value.contiguous() for value in clean), valid_count=valid_count
    )
    dirty_output = radial_compact_forward(
        *(value.contiguous() for value in dirty), valid_count=valid_count
    )
    for clean_tensor, dirty_tensor in zip(clean_output, dirty_output, strict=True):
        torch.testing.assert_close(clean_tensor, dirty_tensor, rtol=0, atol=0)

    for panel, count in enumerate((5, 19)):
        assert torch.count_nonzero(dirty_output.inverse_mass[panel, count:]) == 0
        assert torch.count_nonzero(dirty_output.theta[panel, count:]) == 0
        assert torch.count_nonzero(dirty_output.weights[panel, count:]) == 0
        assert torch.count_nonzero(dirty_output.radial_scale[panel, count:]) == 0
        torch.testing.assert_close(
            dirty_output.radial_q2[panel, count:],
            torch.full_like(
                dirty_output.radial_q2[panel, count:], _RADIUS * _RADIUS
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            dirty_output.diagonal[panel, count:],
            torch.ones_like(dirty_output.diagonal[panel, count:]),
            rtol=0,
            atol=0,
        )


def test_radial_compact_j_d_lower_upper_2pow12_observable_contract() -> None:
    # Both BF16 values are exact and their product is 0.5 - 2^-12.
    first, second = 0.6953125, 0.71875
    u = torch.zeros(1, _CHUNK, _RANK, device="cuda", dtype=torch.float16)
    h = torch.zeros_like(u, dtype=torch.bfloat16)
    u[0, 0, 0], u[0, 0, 1] = first, second
    h[0, 0, 0], h[0, 0, 1] = first, second
    log_decay = torch.zeros(1, _CHUNK, device="cuda")
    strength = torch.ones(1, device="cuda")
    boundary_m = torch.ones(1, device="cuda")
    boundary_j = torch.zeros(1, _RANK, _RANK, device="cuda")
    boundary_d = torch.zeros_like(boundary_j)
    # This 2x2 J block is PSD. Its two strict entries test both sides.
    boundary_j[0, 0, 0] = 0.5
    boundary_j[0, 1, 1] = 0.5
    boundary_j[0, 0, 1] = -0.5
    boundary_j[0, 1, 0] = -0.5
    boundary_d[0, 0, 1] = -0.5
    boundary_d[0, 1, 0] = -0.5
    valid_count = torch.tensor([1], device="cuda", dtype=torch.int32)
    inputs = tuple(
        value.contiguous()
        for value in (
            u,
            h,
            log_decay,
            strength,
            boundary_m,
            boundary_j,
            boundary_d,
        )
    )
    actual = radial_compact_forward(*inputs, valid_count=valid_count)
    expected = _explicit_fp64_oracle(*inputs, valid_count)

    residual = 0.5 - first * second
    assert (0.5 + first * second) / residual == 4095.0
    normalized_residual = residual / 2.0
    assert normalized_residual == 2.0**-13
    assert normalized_residual / _RADIUS == 2.0**-10
    assert torch.isfinite(actual.radial_scale[0, 0]).all()
    assert torch.isfinite(actual.radial_q2[0, 0]).all()
    assert torch.all(actual.radial_q2[0, 0] > 0)

    # The private q2 and scale are not standalone observables. The chart stores
    # scale * Z, where Z is the realized zero-centered strict residual.
    expected_coordinate = expected[3][0, 0] * normalized_residual
    actual_coordinate = actual.radial_scale[0, 0].double() * normalized_residual
    assert (
        _rho(expected_coordinate, actual_coordinate)
        < _PRIVATE_RADIAL_RHO_CEILING
    )


def test_radial_compact_identity_strength_is_structurally_exact() -> None:
    inputs = list(_inputs(panels=4))
    inputs[3].zero_()
    first = radial_compact_forward(*inputs)
    for first_tensor in first:
        assert torch.isfinite(first_tensor).all()
    assert torch.count_nonzero(first.radial_scale) == 0
    torch.testing.assert_close(
        first.radial_q2,
        torch.full_like(first.radial_q2, _RADIUS * _RADIUS),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first.diagonal,
        torch.ones_like(first.diagonal),
        rtol=0,
        atol=0,
    )


def test_radial_compact_reverse_matches_fp64_oracle() -> None:
    torch.manual_seed(20260825)
    inputs = _inputs(panels=1)
    valid_count = torch.full((1,), _CHUNK, device="cuda", dtype=torch.int32)
    grad_radial_scale = (
        0.1 * torch.randn(1, _CHUNK, 4, device="cuda")
    ).contiguous()
    grad_log_diagonal = (
        0.01 * torch.randn(1, _CHUNK, _RANK, device="cuda")
    ).contiguous()
    actual = _reverse(
        inputs, grad_radial_scale, grad_log_diagonal, valid_count
    )
    expected = _reverse_fp64_oracle(
        inputs, grad_radial_scale, grad_log_diagonal, valid_count
    )
    _assert_reverse_matches(
        expected, actual, ceiling=_PRIVATE_RADIAL_RHO_CEILING
    )
    directions = tuple(torch.randn_like(value.double()) for value in inputs)
    observed_inner = sum(
        (gradient.double() * direction).sum()
        for gradient, direction in zip(actual, directions, strict=True)
    )
    expected_inner = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(expected, directions, strict=True)
    )
    torch.testing.assert_close(
        observed_inner,
        expected_inner,
        rtol=_PRIVATE_RADIAL_RHO_CEILING,
        atol=1e-7,
    )


def test_radial_compact_reverse_ignores_nonzero_tail() -> None:
    torch.manual_seed(20260826)
    inputs = list(_inputs(panels=2))
    valid_count = torch.tensor([5, 19], device="cuda", dtype=torch.int32)
    for panel, count in enumerate((5, 19)):
        inputs[0][panel, count:] = torch.randn_like(inputs[0][panel, count:])
        inputs[1][panel, count:] = torch.randn_like(inputs[1][panel, count:])
        inputs[2][panel, count:] = 2.0
    inputs = tuple(value.contiguous() for value in inputs)
    grad_radial_scale = torch.randn(
        2, _CHUNK, 4, device="cuda"
    ).contiguous()
    grad_log_diagonal = torch.randn(
        2, _CHUNK, _RANK, device="cuda"
    ).contiguous()
    actual = _reverse(
        inputs, grad_radial_scale, grad_log_diagonal, valid_count
    )
    expected = _reverse_fp64_oracle(
        inputs, grad_radial_scale, grad_log_diagonal, valid_count
    )
    _assert_reverse_matches(
        expected, actual, ceiling=_PRIVATE_RADIAL_RHO_CEILING
    )
    for panel, count in enumerate((5, 19)):
        assert torch.count_nonzero(actual.grad_u[panel, count:]) == 0
        assert torch.count_nonzero(actual.grad_h[panel, count:]) == 0
        assert torch.count_nonzero(actual.grad_log_decay[panel, count:]) == 0


def test_radial_compact_reverse_2pow12_cancellation() -> None:
    first, second = 0.6953125, 0.71875
    u = torch.zeros(1, _CHUNK, _RANK, device="cuda", dtype=torch.float16)
    h = torch.zeros_like(u, dtype=torch.bfloat16)
    u[0, 0, 0], u[0, 0, 1] = first, second
    h[0, 0, 0], h[0, 0, 1] = first, second
    log_decay = torch.zeros(1, _CHUNK, device="cuda")
    strength = torch.ones(1, device="cuda")
    boundary_m = torch.ones(1, device="cuda")
    boundary_j = torch.zeros(1, _RANK, _RANK, device="cuda")
    boundary_d = torch.zeros_like(boundary_j)
    boundary_j[0, 0, 0] = 0.5
    boundary_j[0, 1, 1] = 0.5
    boundary_j[0, 0, 1] = -0.5
    boundary_j[0, 1, 0] = -0.5
    boundary_d[0, 0, 1] = -0.5
    boundary_d[0, 1, 0] = -0.5
    inputs = tuple(
        value.contiguous()
        for value in (
            u,
            h,
            log_decay,
            strength,
            boundary_m,
            boundary_j,
            boundary_d,
        )
    )
    valid_count = torch.tensor([1], device="cuda", dtype=torch.int32)
    residual = 0.5 - first * second
    grad_radial_scale = torch.zeros(1, _CHUNK, 4, device="cuda")
    # For A=aZ, bar_a=<bar_A,Z>. This is the reachable cotangent produced by
    # the strict-chart 4095:1 fixture, rather than an impossible O(1) bar_a.
    grad_radial_scale[0, 0] = -0.5 * residual
    grad_log_diagonal = torch.zeros(
        1, _CHUNK, _RANK, device="cuda"
    )
    actual = _reverse(
        inputs, grad_radial_scale, grad_log_diagonal, valid_count
    )
    expected = _reverse_fp64_oracle(
        inputs, grad_radial_scale, grad_log_diagonal, valid_count
    )
    for index, (reference, observed) in enumerate(
        zip(expected, actual, strict=True)
    ):
        assert torch.isfinite(observed).all()
        absolute_error = (reference - observed.double()).abs().max().item()
        relative_error = _rho(reference, observed)
        if index in (0, 1):
            assert absolute_error <= 2e-4 or relative_error <= 2.5e-2
        elif index == 3:
            assert absolute_error <= 1e-6 or relative_error <= 2.5e-2
        else:
            assert (
                absolute_error <= 1e-6
                or relative_error <= _PRIVATE_RADIAL_RHO_CEILING
            )


def test_radial_compact_nonstructural_zero_uses_observable_contract() -> None:
    """Algebraic cancellation is not a structural bitwise identity gate."""

    torch.manual_seed(20260828)
    panels = 32
    u = torch.zeros(
        panels, _CHUNK, _RANK, device="cuda", dtype=torch.float16
    )
    h = torch.zeros_like(u, dtype=torch.bfloat16)
    u[:, 0] = F.normalize(
        torch.randn(panels, _RANK, device="cuda").bfloat16().float(), dim=-1
    ).half()
    h[:, 0] = (0.25 * torch.randn(panels, _RANK, device="cuda")).bfloat16()
    local_j = u[:, 0].float().unsqueeze(2) * u[:, 0].float().unsqueeze(1)
    local_d = u[:, 0].float().unsqueeze(2) * h[:, 0].float().unsqueeze(1)
    identity = torch.eye(_RANK, device="cuda").expand(panels, -1, -1)
    boundary_j = (2.0 * identity - local_j).contiguous()
    boundary_d = (-local_d).contiguous()
    assert torch.linalg.cholesky_ex(boundary_j).info.count_nonzero() == 0

    inputs = (
        u.contiguous(),
        h.contiguous(),
        torch.zeros(panels, _CHUNK, device="cuda"),
        torch.linspace(0.25, 0.9, panels, device="cuda"),
        torch.ones(panels, device="cuda"),
        boundary_j,
        boundary_d,
    )
    valid_count = torch.ones(panels, device="cuda", dtype=torch.int32)
    output, saved = radial_compact_forward(
        *inputs, valid_count=valid_count, return_saved=True
    )

    assert torch.isfinite(saved.radial_norm).all()
    assert torch.isfinite(output.radial_scale[:, 0]).all()
    assert torch.isfinite(output.radial_q2[:, 0]).all()
    assert torch.all(output.radial_q2[:, 0] > 0)

    moment_j = 0.5 * boundary_j + 0.5 * local_j
    moment_d = 0.5 * boundary_d + 0.5 * local_d
    strict_mask = ~torch.eye(_RANK, device="cuda", dtype=torch.bool)
    assert torch.count_nonzero(moment_j[:, strict_mask]) == 0
    assert torch.count_nonzero(moment_d[:, strict_mask]) == 0
    for route, moment in enumerate((moment_j, moment_d, moment_j, moment_d)):
        strict = torch.triu(moment, diagonal=1) if route >= 2 else torch.tril(
            moment, diagonal=-1
        )
        chart_coordinate = output.radial_scale[:, 0, route, None, None] * strict
        assert torch.count_nonzero(chart_coordinate) == 0

    grad_radial_scale = torch.zeros_like(output.radial_scale)
    gradients = radial_compact_reverse(
        *inputs,
        output,
        saved,
        grad_radial_scale,
        torch.zeros_like(output.diagonal),
        valid_count=valid_count,
    )
    for gradient in gradients:
        assert torch.count_nonzero(gradient) == 0


def test_radial_compact_nan_tail_is_unobservable_in_forward_and_reverse() -> None:
    torch.manual_seed(20260829)
    valid_count = torch.tensor([0, 1, 17], device="cuda", dtype=torch.int32)
    clean = list(_inputs(panels=3))
    dirty = [value.clone() for value in clean]
    for panel, count in enumerate(valid_count.tolist()):
        clean[0][panel, count:] = 0
        clean[1][panel, count:] = 0
        clean[2][panel, count:] = 0
        dirty[0][panel, count:] = torch.nan
        dirty[1][panel, count:] = torch.nan
        dirty[2][panel, count:] = torch.nan
    clean = tuple(value.contiguous() for value in clean)
    dirty = tuple(value.contiguous() for value in dirty)

    clean_output, clean_saved = radial_compact_forward(
        *clean, valid_count=valid_count, return_saved=True
    )
    dirty_output, dirty_saved = radial_compact_forward(
        *dirty, valid_count=valid_count, return_saved=True
    )
    for expected, actual in zip(clean_output, dirty_output, strict=True):
        assert torch.equal(expected, actual)
        assert torch.isfinite(actual).all()
    for expected, actual in zip(clean_saved, dirty_saved, strict=True):
        assert torch.equal(expected, actual)
        assert torch.isfinite(actual).all()

    clean_grad_scale = torch.randn_like(clean_output.radial_scale)
    clean_grad_diagonal = torch.randn_like(clean_output.diagonal)
    dirty_grad_scale = clean_grad_scale.clone()
    dirty_grad_diagonal = clean_grad_diagonal.clone()
    for panel, count in enumerate(valid_count.tolist()):
        clean_grad_scale[panel, count:] = 0
        clean_grad_diagonal[panel, count:] = 0
        dirty_grad_scale[panel, count:] = torch.nan
        dirty_grad_diagonal[panel, count:] = torch.nan

    clean_gradients = radial_compact_reverse(
        *clean,
        clean_output,
        clean_saved,
        clean_grad_scale,
        clean_grad_diagonal,
        valid_count=valid_count,
    )
    dirty_gradients = radial_compact_reverse(
        *dirty,
        dirty_output,
        dirty_saved,
        dirty_grad_scale,
        dirty_grad_diagonal,
        valid_count=valid_count,
    )
    for expected, actual in zip(clean_gradients, dirty_gradients, strict=True):
        assert torch.equal(expected, actual)
        assert torch.isfinite(actual).all()
    for panel, count in enumerate(valid_count.tolist()):
        assert torch.count_nonzero(dirty_gradients.grad_u[panel, count:]) == 0
        assert torch.count_nonzero(dirty_gradients.grad_h[panel, count:]) == 0
        assert (
            torch.count_nonzero(dirty_gradients.grad_log_decay[panel, count:])
            == 0
        )


def test_radial_reverse_composes_with_existing_frame_partials() -> None:
    inputs = _inputs(panels=2)
    valid_count = torch.tensor([32, 19], device="cuda", dtype=torch.int32)
    output, saved = radial_compact_forward(
        *inputs, valid_count=valid_count, return_saved=True
    )
    grad_scale = torch.randn_like(output.radial_scale)
    grad_diagonal = torch.randn_like(output.diagonal)
    radial = radial_compact_reverse(
        *inputs,
        output,
        saved,
        grad_scale,
        grad_diagonal,
        valid_count=valid_count,
    )
    base = (
        torch.randn_like(radial.grad_u),
        torch.randn_like(radial.grad_h),
        torch.randn_like(radial.grad_boundary_j),
        torch.randn_like(radial.grad_boundary_d),
    )
    for panel, count in enumerate(valid_count.tolist()):
        base[0][panel, count:] = 0
        base[1][panel, count:] = 0
    accumulated_storage = tuple(value.clone() for value in base)
    accumulated = _radial_compact_reverse_accumulate_trusted(
        *inputs,
        output,
        saved,
        grad_scale,
        grad_diagonal,
        valid_count,
        torch.zeros_like(output.theta),
        torch.zeros_like(output.weights),
        *accumulated_storage,
    )

    for expected_base, expected_radial, actual in zip(
        base,
        (
            radial.grad_u,
            radial.grad_h,
            radial.grad_boundary_j,
            radial.grad_boundary_d,
        ),
        (
            accumulated.grad_u,
            accumulated.grad_h,
            accumulated.grad_boundary_j,
            accumulated.grad_boundary_d,
        ),
        strict=True,
    ):
        torch.testing.assert_close(
            actual, expected_base + expected_radial, rtol=2e-5, atol=2e-5
        )
    for expected, actual in zip(
        (
            radial.grad_log_decay,
            radial.grad_strength,
            radial.grad_boundary_m,
        ),
        (
            accumulated.grad_log_decay,
            accumulated.grad_strength,
            accumulated.grad_boundary_m,
        ),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_radial_compact_requires_one_cuda_index_and_guards_launches() -> None:
    inputs = list(_inputs(panels=1))
    with torch.cuda.device(1):
        with pytest.raises(ValueError, match="same CUDA device"):
            radial_compact_forward(inputs[0], inputs[1].to("cuda:1"), *inputs[2:])
        output = radial_compact_forward(*inputs)
    assert all(value.device == inputs[0].device for value in output)


def test_radial_compact_reverse_repeatability_is_bounded() -> None:
    torch.manual_seed(20260827)
    inputs = _inputs(panels=4)
    valid_count = torch.tensor([32, 7, 31, 1], device="cuda", dtype=torch.int32)
    grad_radial_scale = torch.randn(
        4, _CHUNK, 4, device="cuda"
    ).contiguous()
    grad_log_diagonal = torch.randn(
        4, _CHUNK, _RANK, device="cuda"
    ).contiguous()
    first = _reverse(
        inputs, grad_radial_scale, grad_log_diagonal, valid_count
    )
    second = _reverse(
        inputs, grad_radial_scale, grad_log_diagonal, valid_count
    )
    for first_tensor, second_tensor in zip(first, second, strict=True):
        assert torch.isfinite(first_tensor).all()
        assert torch.isfinite(second_tensor).all()
        assert (
            _rho(first_tensor, second_tensor)
            <= _REPEATABILITY_RHO_CEILING
        )


@pytest.mark.skipif(
    os.environ.get("CAUSALLSSO_RUN_BENCHMARKS") != "1",
    reason="set CAUSALLSSO_RUN_BENCHMARKS=1 to run target benchmark",
)
def test_radial_compact_target_benchmark() -> None:
    # B=1, T=1024, H=8, C=32 gives 256 independent panels.
    inputs = _inputs(panels=256)
    radial_compact_forward(*inputs)
    milliseconds = triton.testing.do_bench(lambda: radial_compact_forward(*inputs))
    print(f"radial_compact P256 C32 r128: {milliseconds:.3f} ms")
    assert milliseconds < 10.0


@pytest.mark.skipif(
    os.environ.get("CAUSALLSSO_RUN_BENCHMARKS") != "1",
    reason="set CAUSALLSSO_RUN_BENCHMARKS=1 to run target benchmark",
)
def test_radial_compact_reverse_target_benchmark() -> None:
    inputs = _inputs(panels=256)
    valid_count = torch.full(
        (256,), _CHUNK, device="cuda", dtype=torch.int32
    )
    output, saved = radial_compact_forward(
        *inputs, valid_count=valid_count, return_saved=True
    )
    grad_radial_scale = torch.randn_like(output.radial_scale)
    grad_log_diagonal = torch.randn_like(output.diagonal)

    def reverse():
        return radial_compact_reverse(
            *inputs,
            output,
            saved,
            grad_radial_scale,
            grad_log_diagonal,
            valid_count=valid_count,
        )

    reverse()
    milliseconds = triton.testing.do_bench(reverse)
    print(f"radial_compact reverse P256 C32 r128: {milliseconds:.3f} ms")
    assert milliseconds < 20.0
