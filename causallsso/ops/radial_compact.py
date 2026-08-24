from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


_CHUNK = tl.constexpr(32)
_RANK = tl.constexpr(128)
_DIAGONAL_TARGET = tl.constexpr(8)
_DIAGONAL_R = tl.constexpr(16)
_MATRIX_TILE = tl.constexpr(16)
_MATRIX_TILES = tl.constexpr(64)

_HOST_CHUNK = 32
_HOST_RANK = 128
_HOST_MATRIX_TILES = 64
_RADIUS = 1.0 / 8.0


class RadialCompactOutput(NamedTuple):
    """Chunk-local scalar chart data in panel-major order."""

    inverse_mass: torch.Tensor
    theta: torch.Tensor
    weights: torch.Tensor
    radial_scale: torch.Tensor
    radial_q2: torch.Tensor
    diagonal: torch.Tensor


class RadialCompactSaved(NamedTuple):
    """The only forward statistic needed by the dedicated reverse."""

    radial_norm: torch.Tensor


class RadialCompactGradients(NamedTuple):
    grad_u: torch.Tensor
    grad_h: torch.Tensor
    grad_log_decay: torch.Tensor
    grad_strength: torch.Tensor
    grad_boundary_m: torch.Tensor
    grad_boundary_j: torch.Tensor
    grad_boundary_d: torch.Tensor


@triton.jit
def _temporal_coefficients_kernel(
    log_decay,
    boundary_mass,
    valid_count,
    inverse_mass,
    theta,
    weights,
):
    panel = tl.program_id(0)
    sources = tl.arange(0, _CHUNK)
    count = tl.load(valid_count + panel)
    mass = tl.load(boundary_mass + panel).to(tl.float32)
    boundary_numerator = 1.0
    previous = tl.zeros((_CHUNK,), tl.float32)

    for target in tl.static_range(0, _CHUNK):
        active = target < count
        log_rho = tl.load(
            log_decay + panel * _CHUNK + target,
            mask=active,
            other=0.0,
        )
        rho = tl.exp(log_rho.to(tl.float32))
        next_mass = rho * mass + 1.0
        inverse = 1.0 / next_mass
        retain = 1.0 - inverse
        next_boundary_numerator = rho * boundary_numerator
        local_theta = next_boundary_numerator * inverse
        row = retain * previous + tl.where(sources == target, inverse, 0.0)

        inverse = tl.where(active, inverse, 0.0)
        local_theta = tl.where(active, local_theta, 0.0)
        row = tl.where(active, row, 0.0)
        tl.store(inverse_mass + panel * _CHUNK + target, inverse)
        tl.store(theta + panel * _CHUNK + target, local_theta)
        tl.store(
            weights + (panel * _CHUNK + target) * _CHUNK + sources,
            row,
        )

        mass = tl.where(active, next_mass, mass)
        boundary_numerator = tl.where(
            active, next_boundary_numerator, boundary_numerator
        )
        previous = tl.where(active, row, previous)


@triton.jit
def _radial_residual_partial_kernel(
    u,
    h,
    inverse_mass,
    theta,
    boundary_j,
    boundary_d,
    valid_count,
    norm_partial,
):
    """Square the realized FP32 residual, never a reassociated quadratic."""

    panel = tl.program_id(0)
    row_block = tl.program_id(1)
    column_block = tl.program_id(2)
    tile = row_block * 8 + column_block
    rows = (
        row_block * _MATRIX_TILE + tl.arange(0, _MATRIX_TILE)
    )[:, None]
    columns = (
        column_block * _MATRIX_TILE + tl.arange(0, _MATRIX_TILE)
    )[None, :]
    count = tl.load(valid_count + panel)
    boundary_offset = (
        panel * _RANK * _RANK + rows * _RANK + columns
    )
    local_boundary_j = tl.load(boundary_j + boundary_offset).to(tl.float32)
    local_boundary_d = tl.load(boundary_d + boundary_offset).to(tl.float32)
    current_j = tl.zeros((_MATRIX_TILE, _MATRIX_TILE), tl.float32)
    current_d = tl.zeros((_MATRIX_TILE, _MATRIX_TILE), tl.float32)
    lower = columns < rows
    upper = columns > rows

    for target in tl.static_range(0, _CHUNK):
        active = target < count
        inverse = tl.load(
            inverse_mass + panel * _CHUNK + target,
            mask=active,
            other=0.0,
        )
        local_u_rows = tl.load(
            u + (panel * _CHUNK + target) * _RANK + rows,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        local_u_columns = tl.load(
            u + (panel * _CHUNK + target) * _RANK + columns,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        local_h_columns = tl.load(
            h + (panel * _CHUNK + target) * _RANK + columns,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        phi_j = local_u_rows * local_u_columns
        phi_d = local_u_rows * local_h_columns
        if target == 0:
            local_theta = tl.load(
                theta + panel * _CHUNK,
                mask=active,
                other=0.0,
            )
            next_j = local_theta * local_boundary_j + inverse * phi_j
            next_d = local_theta * local_boundary_d + inverse * phi_d
        else:
            retain = 1.0 - inverse
            next_j = retain * current_j + inverse * phi_j
            next_d = retain * current_d + inverse * phi_d
        current_j = tl.where(active, next_j, current_j)
        current_d = tl.where(active, next_d, current_d)

        lower_j = tl.sum(tl.sum(tl.where(lower, current_j * current_j, 0.0), axis=1), axis=0)
        lower_d = tl.sum(tl.sum(tl.where(lower, current_d * current_d, 0.0), axis=1), axis=0)
        upper_j = tl.sum(tl.sum(tl.where(upper, current_j * current_j, 0.0), axis=1), axis=0)
        upper_d = tl.sum(tl.sum(tl.where(upper, current_d * current_d, 0.0), axis=1), axis=0)
        base = ((panel * _CHUNK + target) * 4) * _MATRIX_TILES + tile
        tl.store(norm_partial + base, tl.where(active, lower_j, 0.0))
        tl.store(
            norm_partial + base + _MATRIX_TILES,
            tl.where(active, lower_d, 0.0),
        )
        tl.store(
            norm_partial + base + 2 * _MATRIX_TILES,
            tl.where(active, upper_j, 0.0),
        )
        tl.store(
            norm_partial + base + 3 * _MATRIX_TILES,
            tl.where(active, upper_d, 0.0),
        )


@triton.jit
def _radial_output_kernel(
    norm_partial,
    strength,
    valid_count,
    radial_scale,
    radial_q2,
    radial_norm,
    SAVE_NORM: tl.constexpr,
):
    panel = tl.program_id(0)
    target = tl.program_id(1)
    route_vector = tl.arange(0, 4)
    routes = route_vector[:, None]
    tiles = tl.arange(0, _MATRIX_TILES)[None, :]
    active = target < tl.load(valid_count + panel)
    partial = tl.load(
        norm_partial
        + ((panel * _CHUNK + target) * 4 + routes) * _MATRIX_TILES
        + tiles
    )
    norm = tl.sum(partial, axis=1)
    panel_strength = tl.load(strength + panel).to(tl.float32)
    radius = 0.125
    q2 = radius * radius + panel_strength * panel_strength * norm
    scale = panel_strength * radius * tl.rsqrt(q2)
    output = (panel * _CHUNK + target) * 4 + route_vector
    tl.store(
        radial_q2 + output,
        tl.where(active, q2, radius * radius),
    )
    tl.store(radial_scale + output, tl.where(active, scale, 0.0))
    if SAVE_NORM:
        tl.store(radial_norm + output, tl.where(active, norm, 0.0))


@triton.jit
def _radial_scalar_reverse_kernel(
    radial_norm,
    radial_scale,
    radial_q2,
    strength,
    valid_count,
    grad_radial_scale,
    grad_norm,
    grad_strength,
):
    panel = tl.program_id(0)
    targets = tl.arange(0, _CHUNK)[:, None]
    routes = tl.arange(0, 4)[None, :]
    active = targets < tl.load(valid_count + panel)
    offsets = (panel * _CHUNK + targets) * 4 + routes
    norm = tl.load(radial_norm + offsets, mask=active, other=0.0)
    q2 = tl.load(radial_q2 + offsets, mask=active, other=1.0)
    scale = tl.load(radial_scale + offsets, mask=active, other=0.0)
    grad_scale = tl.load(
        grad_radial_scale + offsets,
        mask=active,
        other=0.0,
    )
    panel_strength = tl.load(strength + panel).to(tl.float32)
    radius = 0.125
    grad_q2 = -0.5 * grad_scale * scale / q2
    # The forward norm is the square of the realized FP32 residual. At an
    # exactly zero residual its VJP is exactly zero; do not let a reassociated
    # reverse contraction manufacture a direction that forward never saw.
    local_grad_norm = tl.where(
        norm == 0.0,
        0.0,
        grad_q2 * panel_strength * panel_strength,
    )
    local_grad_strength = (
        grad_scale * radius * tl.rsqrt(q2)
        + grad_q2 * 2.0 * panel_strength * norm
    )
    tl.store(grad_norm + offsets, tl.where(active, local_grad_norm, 0.0))
    tl.store(
        grad_strength + panel,
        tl.sum(tl.sum(tl.where(active, local_grad_strength, 0.0), axis=1), axis=0),
    )


@triton.jit
def _radial_algebra_boundary_action_kernel(
    u,
    h,
    boundary_j,
    boundary_d,
    theta,
    weights,
    grad_norm,
    valid_count,
    grad_u,
    grad_h,
    correlation_partial,
    norm_partial,
    ADD_TO_GRAD: tl.constexpr,
):
    """Apply all radial boundary cotangents as broad C32 products."""

    panel = tl.program_id(0)
    row_block = tl.program_id(1)
    rows = row_block * _MATRIX_TILE + tl.arange(0, _MATRIX_TILE)
    sources = tl.arange(0, _CHUNK)
    targets = tl.arange(0, _CHUNK)[:, None]
    count = tl.load(valid_count + panel)
    active_source = sources < count
    active_target = targets < count
    local_theta = tl.load(
        theta + panel * _CHUNK + targets,
        mask=active_target,
        other=0.0,
    )
    temporal = tl.load(
        weights
        + (panel * _CHUNK + targets) * _CHUNK
        + sources[None, :],
        mask=active_target & active_source[None, :],
        other=0.0,
    )
    beta_base = (panel * _CHUNK + targets) * 4
    beta_lj = 2.0 * tl.load(grad_norm + beta_base, mask=active_target, other=0.0)
    beta_ld = 2.0 * tl.load(grad_norm + beta_base + 1, mask=active_target, other=0.0)
    beta_uj = 2.0 * tl.load(grad_norm + beta_base + 2, mask=active_target, other=0.0)
    beta_ud = 2.0 * tl.load(grad_norm + beta_base + 3, mask=active_target, other=0.0)
    coefficient_lj = tl.sum(temporal * (beta_lj * local_theta), axis=0)
    coefficient_ld = tl.sum(temporal * (beta_ld * local_theta), axis=0)
    coefficient_uj = tl.sum(temporal * (beta_uj * local_theta), axis=0)
    coefficient_ud = tl.sum(temporal * (beta_ud * local_theta), axis=0)

    panel_vector = panel * _CHUNK * _RANK
    panel_matrix = panel * _RANK * _RANK
    u_rows = tl.load(
        u + panel_vector + sources[:, None] * _RANK + rows[None, :],
        mask=active_source[:, None],
        other=0.0,
    ).to(tl.float32)
    grad_u_rows = tl.zeros((_MATRIX_TILE, _CHUNK), tl.float32)
    grad_h_rows = tl.zeros_like(grad_u_rows)
    corr_lj = tl.zeros((_CHUNK,), tl.float32)
    corr_ld = tl.zeros_like(corr_lj)
    corr_uj = tl.zeros_like(corr_lj)
    corr_ud = tl.zeros_like(corr_lj)
    norm_lj = 0.0
    norm_ld = 0.0
    norm_uj = 0.0
    norm_ud = 0.0

    for start in tl.static_range(0, _RANK, 32):
        columns = start + tl.arange(0, 32)
        u_columns = tl.load(
            u + panel_vector + sources[:, None] * _RANK + columns[None, :],
            mask=active_source[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        h_columns = tl.load(
            h + panel_vector + sources[:, None] * _RANK + columns[None, :],
            mask=active_source[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        direct_offset = panel_matrix + rows[:, None] * _RANK + columns[None, :]
        transpose_offset = panel_matrix + columns[:, None] * _RANK + rows[None, :]
        lower = columns[None, :] < rows[:, None]
        upper = columns[None, :] > rows[:, None]
        lower_t = columns[:, None] > rows[None, :]
        upper_t = columns[:, None] < rows[None, :]
        j_direct = tl.load(boundary_j + direct_offset).to(tl.float32)
        d_direct = tl.load(boundary_d + direct_offset).to(tl.float32)
        j_transpose = tl.trans(
            tl.load(boundary_j + transpose_offset).to(tl.float32)
        )
        d_transpose = tl.trans(
            tl.load(boundary_d + transpose_offset).to(tl.float32)
        )
        jl = tl.where(lower, j_direct, 0.0)
        ju = tl.where(upper, j_direct, 0.0)
        dl = tl.where(lower, d_direct, 0.0)
        du = tl.where(upper, d_direct, 0.0)
        jlt = tl.where(tl.trans(lower_t), j_transpose, 0.0)
        jut = tl.where(tl.trans(upper_t), j_transpose, 0.0)
        dlt = tl.where(tl.trans(lower_t), d_transpose, 0.0)
        dut = tl.where(tl.trans(upper_t), d_transpose, 0.0)

        jl_u = tl.dot(jl.to(tl.bfloat16), tl.trans(u_columns))
        ju_u = tl.dot(ju.to(tl.bfloat16), tl.trans(u_columns))
        jlt_u = tl.dot(jlt.to(tl.bfloat16), tl.trans(u_columns))
        jut_u = tl.dot(jut.to(tl.bfloat16), tl.trans(u_columns))
        dl_h = tl.dot(dl.to(tl.bfloat16), tl.trans(h_columns))
        du_h = tl.dot(du.to(tl.bfloat16), tl.trans(h_columns))
        dlt_u = tl.dot(dlt.to(tl.bfloat16), tl.trans(u_columns))
        dut_u = tl.dot(dut.to(tl.bfloat16), tl.trans(u_columns))

        grad_u_rows += (
            (jl_u + jlt_u) * coefficient_lj[None, :]
            + (ju_u + jut_u) * coefficient_uj[None, :]
            + dl_h * coefficient_ld[None, :]
            + du_h * coefficient_ud[None, :]
        )
        grad_h_rows += (
            dlt_u * coefficient_ld[None, :]
            + dut_u * coefficient_ud[None, :]
        )
        local_u_rows = tl.trans(u_rows)
        corr_lj += tl.sum(local_u_rows * jl_u, axis=0)
        corr_uj += tl.sum(local_u_rows * ju_u, axis=0)
        corr_ld += tl.sum(local_u_rows * dl_h, axis=0)
        corr_ud += tl.sum(local_u_rows * du_h, axis=0)
        norm_lj += tl.sum(jl * jl)
        norm_uj += tl.sum(ju * ju)
        norm_ld += tl.sum(dl * dl)
        norm_ud += tl.sum(du * du)

    output = panel_vector + sources[:, None] * _RANK + rows[None, :]
    result_u = tl.trans(grad_u_rows)
    result_h = tl.trans(grad_h_rows)
    if ADD_TO_GRAD:
        result_u += tl.load(
            grad_u + output,
            mask=active_source[:, None],
            other=0.0,
        ).to(tl.float32)
        result_h += tl.load(
            grad_h + output,
            mask=active_source[:, None],
            other=0.0,
        ).to(tl.float32)
    tl.store(grad_u + output, tl.where(active_source[:, None], result_u, 0.0))
    tl.store(grad_h + output, tl.where(active_source[:, None], result_h, 0.0))
    partial_base = ((panel * 4) * 8 + row_block) * _CHUNK + sources
    tl.store(
        correlation_partial + partial_base,
        tl.where(active_source, corr_lj, 0.0),
    )
    tl.store(
        correlation_partial + partial_base + 8 * _CHUNK,
        tl.where(active_source, corr_ld, 0.0),
    )
    tl.store(
        correlation_partial + partial_base + 16 * _CHUNK,
        tl.where(active_source, corr_uj, 0.0),
    )
    tl.store(
        correlation_partial + partial_base + 24 * _CHUNK,
        tl.where(active_source, corr_ud, 0.0),
    )
    norm_base = (panel * 4) * 8 + row_block
    tl.store(norm_partial + norm_base, norm_lj)
    tl.store(norm_partial + norm_base + 8, norm_ld)
    tl.store(norm_partial + norm_base + 16, norm_uj)
    tl.store(norm_partial + norm_base + 24, norm_ud)


@triton.jit
def _radial_residual_boundary_gradient_kernel(
    u,
    h,
    boundary_j,
    boundary_d,
    inverse_mass,
    theta,
    grad_norm,
    valid_count,
    grad_boundary_j,
    grad_boundary_d,
    ADD_TO_GRAD: tl.constexpr,
):
    panel = tl.program_id(0)
    row_block = tl.program_id(1)
    column_block = tl.program_id(2)
    rows = row_block * _MATRIX_TILE + tl.arange(0, _MATRIX_TILE)
    columns = column_block * _MATRIX_TILE + tl.arange(0, _MATRIX_TILE)
    count = tl.load(valid_count + panel)
    panel_vector = panel * _CHUNK * _RANK
    matrix_offset = (
        panel * _RANK * _RANK
        + rows[:, None] * _RANK
        + columns[None, :]
    )
    boundary_j_value = tl.load(boundary_j + matrix_offset).to(tl.float32)
    boundary_d_value = tl.load(boundary_d + matrix_offset).to(tl.float32)
    lower = columns[None, :] < rows[:, None]
    upper = columns[None, :] > rows[:, None]
    current_j = tl.zeros((_MATRIX_TILE, _MATRIX_TILE), tl.float32)
    current_d = tl.zeros_like(current_j)
    result_j = tl.zeros_like(current_j)
    result_d = tl.zeros_like(current_j)

    # Rebuild each realized FP32 residual in the same operation order as the
    # forward norm kernel. This is one O(C r^2) boundary transpose, not the
    # former three coordinate-tile VJP replays.
    for target in tl.static_range(0, _CHUNK):
        active = target < count
        inverse = tl.load(
            inverse_mass + panel * _CHUNK + target,
            mask=active,
            other=0.0,
        )
        local_theta = tl.load(
            theta + panel * _CHUNK + target,
            mask=active,
            other=0.0,
        )
        local_u_rows = tl.load(
            u + panel_vector + target * _RANK + rows[:, None],
            mask=active,
            other=0.0,
        ).to(tl.float32)
        local_u_columns = tl.load(
            u + panel_vector + target * _RANK + columns[None, :],
            mask=active,
            other=0.0,
        ).to(tl.float32)
        local_h_columns = tl.load(
            h + panel_vector + target * _RANK + columns[None, :],
            mask=active,
            other=0.0,
        ).to(tl.float32)
        phi_j = local_u_rows * local_u_columns
        phi_d = local_u_rows * local_h_columns
        if target == 0:
            next_j = local_theta * boundary_j_value + inverse * phi_j
            next_d = local_theta * boundary_d_value + inverse * phi_d
        else:
            retain = 1.0 - inverse
            next_j = retain * current_j + inverse * phi_j
            next_d = retain * current_d + inverse * phi_d
        current_j = tl.where(active, next_j, current_j)
        current_d = tl.where(active, next_d, current_d)

        beta_base = (panel * _CHUNK + target) * 4
        beta_lj = 2.0 * tl.load(
            grad_norm + beta_base, mask=active, other=0.0
        )
        beta_ld = 2.0 * tl.load(
            grad_norm + beta_base + 1, mask=active, other=0.0
        )
        beta_uj = 2.0 * tl.load(
            grad_norm + beta_base + 2, mask=active, other=0.0
        )
        beta_ud = 2.0 * tl.load(
            grad_norm + beta_base + 3, mask=active, other=0.0
        )
        coefficient_j = local_theta * tl.where(
            lower, beta_lj, tl.where(upper, beta_uj, 0.0)
        )
        coefficient_d = local_theta * tl.where(
            lower, beta_ld, tl.where(upper, beta_ud, 0.0)
        )
        result_j += coefficient_j * current_j
        result_d += coefficient_d * current_d

    if ADD_TO_GRAD:
        result_j += tl.load(grad_boundary_j + matrix_offset).to(tl.float32)
        result_d += tl.load(grad_boundary_d + matrix_offset).to(tl.float32)
    tl.store(grad_boundary_j + matrix_offset, result_j)
    tl.store(grad_boundary_d + matrix_offset, result_d)


@triton.jit
def _radial_algebra_local_kernel(
    u,
    h,
    theta,
    weights,
    grad_norm,
    valid_count,
    correlation_partial,
    grad_u,
    grad_h,
    grad_weights,
):
    panel = tl.program_id(0)
    source = tl.program_id(1)
    coordinates = tl.arange(0, _RANK)[None, :]
    partners = tl.arange(0, _CHUNK)[:, None]
    targets = tl.arange(0, _CHUNK)[:, None]
    count = tl.load(valid_count + panel)
    source_active = source < count
    partner_active = partners < count
    target_active = targets < count
    panel_vector = panel * _CHUNK * _RANK
    source_u = tl.load(
        u + panel_vector + source * _RANK + coordinates,
        mask=source_active,
        other=0.0,
    ).to(tl.float32)
    source_h = tl.load(
        h + panel_vector + source * _RANK + coordinates,
        mask=source_active,
        other=0.0,
    ).to(tl.float32)
    partner_u = tl.load(
        u + panel_vector + partners * _RANK + coordinates,
        mask=partner_active,
        other=0.0,
    ).to(tl.float32)
    partner_h = tl.load(
        h + panel_vector + partners * _RANK + coordinates,
        mask=partner_active,
        other=0.0,
    ).to(tl.float32)
    product_u = partner_u * source_u
    product_h = partner_h * source_h
    prefix_u = tl.cumsum(product_u, axis=1) - product_u
    prefix_h = tl.cumsum(product_h, axis=1) - product_h
    suffix_u = tl.sum(product_u, axis=1)[:, None] - product_u - prefix_u
    suffix_h = tl.sum(product_h, axis=1)[:, None] - product_h - prefix_h
    corr_lj = tl.sum(product_u * prefix_u, axis=1)
    corr_uj = tl.sum(product_u * suffix_u, axis=1)
    corr_ld = tl.sum(product_u * prefix_h, axis=1)
    corr_ud = tl.sum(product_u * suffix_h, axis=1)

    target_vector = tl.arange(0, _CHUNK)
    target_ids = target_vector[:, None]
    partner_ids = tl.arange(0, _CHUNK)[None, :]
    temporal_source = tl.load(
        weights + (panel * _CHUNK + target_vector) * _CHUNK + source,
        mask=target_vector < count,
        other=0.0,
    )
    temporal_partner = tl.load(
        weights + (panel * _CHUNK + target_ids) * _CHUNK + partner_ids,
        mask=target_active & (partner_ids < count),
        other=0.0,
    )
    beta_base = (panel * _CHUNK + target_vector) * 4
    beta_lj = 2.0 * tl.load(grad_norm + beta_base, mask=target_vector < count, other=0.0)
    beta_ld = 2.0 * tl.load(grad_norm + beta_base + 1, mask=target_vector < count, other=0.0)
    beta_uj = 2.0 * tl.load(grad_norm + beta_base + 2, mask=target_vector < count, other=0.0)
    beta_ud = 2.0 * tl.load(grad_norm + beta_base + 3, mask=target_vector < count, other=0.0)
    coefficient_lj = tl.sum(
        temporal_partner * (temporal_source * beta_lj)[:, None], axis=0
    )
    coefficient_ld = tl.sum(
        temporal_partner * (temporal_source * beta_ld)[:, None], axis=0
    )
    coefficient_uj = tl.sum(
        temporal_partner * (temporal_source * beta_uj)[:, None], axis=0
    )
    coefficient_ud = tl.sum(
        temporal_partner * (temporal_source * beta_ud)[:, None], axis=0
    )
    local_grad_u = tl.sum(
        partner_u
        * (
            (coefficient_lj + coefficient_uj)[:, None]
            * (prefix_u + suffix_u)
            + coefficient_ld[:, None] * prefix_h
            + coefficient_ud[:, None] * suffix_h
        ),
        axis=0,
    )
    local_grad_h = tl.sum(
        partner_h
        * (
            coefficient_ld[:, None] * suffix_u
            + coefficient_ud[:, None] * prefix_u
        ),
        axis=0,
    )
    output = panel_vector + source * _RANK + tl.arange(0, _RANK)
    tl.store(
        grad_u + output,
        tl.load(grad_u + output, mask=source_active, other=0.0) + local_grad_u,
        mask=source_active,
    )
    tl.store(
        grad_h + output,
        tl.load(grad_h + output, mask=source_active, other=0.0) + local_grad_h,
        mask=source_active,
    )

    blocks = tl.arange(0, 8)[None, :]
    corr_base = ((panel * 4) * 8) * _CHUNK + blocks * _CHUNK + source
    boundary_lj = tl.sum(tl.load(correlation_partial + corr_base), axis=1)
    boundary_ld = tl.sum(
        tl.load(correlation_partial + corr_base + 8 * _CHUNK), axis=1
    )
    boundary_uj = tl.sum(
        tl.load(correlation_partial + corr_base + 16 * _CHUNK), axis=1
    )
    boundary_ud = tl.sum(
        tl.load(correlation_partial + corr_base + 24 * _CHUNK), axis=1
    )
    target_theta = tl.load(
        theta + panel * _CHUNK + target_vector,
        mask=target_vector < count,
        other=0.0,
    )
    dot_lj = tl.sum(temporal_partner * corr_lj[None, :], axis=1)
    dot_ld = tl.sum(temporal_partner * corr_ld[None, :], axis=1)
    dot_uj = tl.sum(temporal_partner * corr_uj[None, :], axis=1)
    dot_ud = tl.sum(temporal_partner * corr_ud[None, :], axis=1)
    local_grad_weight = (
        beta_lj * (target_theta * boundary_lj + dot_lj)
        + beta_ld * (target_theta * boundary_ld + dot_ld)
        + beta_uj * (target_theta * boundary_uj + dot_uj)
        + beta_ud * (target_theta * boundary_ud + dot_ud)
    )
    tl.store(
        grad_weights + (panel * _CHUNK + tl.arange(0, _CHUNK)) * _CHUNK + source,
        tl.where(
            (tl.arange(0, _CHUNK) < count) & source_active,
            local_grad_weight,
            0.0,
        ),
    )


@triton.jit
def _radial_algebra_theta_kernel(
    theta,
    weights,
    grad_norm,
    valid_count,
    correlation_partial,
    norm_partial,
    grad_theta,
):
    panel = tl.program_id(0)
    target_vector = tl.arange(0, _CHUNK)
    source_vector = tl.arange(0, _CHUNK)
    targets = target_vector[:, None]
    sources = source_vector[None, :]
    blocks = tl.arange(0, 8)[:, None]
    count = tl.load(valid_count + panel)
    active_target = targets < count
    active_source = sources < count
    corr_base = ((panel * 4) * 8) * _CHUNK + blocks * _CHUNK + source_vector[None, :]
    corr_lj = tl.sum(tl.load(correlation_partial + corr_base), axis=0)
    corr_ld = tl.sum(
        tl.load(correlation_partial + corr_base + 8 * _CHUNK), axis=0
    )
    corr_uj = tl.sum(
        tl.load(correlation_partial + corr_base + 16 * _CHUNK), axis=0
    )
    corr_ud = tl.sum(
        tl.load(correlation_partial + corr_base + 24 * _CHUNK), axis=0
    )
    norm_base = (panel * 4) * 8 + tl.arange(0, 8)
    norm_lj = tl.sum(tl.load(norm_partial + norm_base), axis=0)
    norm_ld = tl.sum(tl.load(norm_partial + norm_base + 8), axis=0)
    norm_uj = tl.sum(tl.load(norm_partial + norm_base + 16), axis=0)
    norm_ud = tl.sum(tl.load(norm_partial + norm_base + 24), axis=0)
    temporal = tl.load(
        weights + (panel * _CHUNK + targets) * _CHUNK + sources,
        mask=active_target & active_source,
        other=0.0,
    )
    local_theta = tl.load(
        theta + panel * _CHUNK + target_vector,
        mask=target_vector < count,
        other=0.0,
    )
    beta_base = (panel * _CHUNK + target_vector) * 4
    beta_lj = 2.0 * tl.load(grad_norm + beta_base, mask=target_vector < count, other=0.0)
    beta_ld = 2.0 * tl.load(grad_norm + beta_base + 1, mask=target_vector < count, other=0.0)
    beta_uj = 2.0 * tl.load(grad_norm + beta_base + 2, mask=target_vector < count, other=0.0)
    beta_ud = 2.0 * tl.load(grad_norm + beta_base + 3, mask=target_vector < count, other=0.0)
    result = (
        beta_lj * (local_theta * norm_lj + tl.sum(temporal * corr_lj[None, :], axis=1))
        + beta_ld * (local_theta * norm_ld + tl.sum(temporal * corr_ld[None, :], axis=1))
        + beta_uj * (local_theta * norm_uj + tl.sum(temporal * corr_uj[None, :], axis=1))
        + beta_ud * (local_theta * norm_ud + tl.sum(temporal * corr_ud[None, :], axis=1))
    )
    tl.store(
        grad_theta + panel * _CHUNK + tl.arange(0, _CHUNK),
        tl.where(target_vector < count, result, 0.0),
    )


@triton.jit
def _diagonal_kernel(
    u,
    h,
    theta,
    weights,
    boundary_j,
    boundary_d,
    strength,
    valid_count,
    diagonal,
):
    panel = tl.program_id(0)
    targets = (
        tl.program_id(1) * _DIAGONAL_TARGET
        + tl.arange(0, _DIAGONAL_TARGET)
    )[:, None]
    coordinates = (
        tl.program_id(2) * _DIAGONAL_R
        + tl.arange(0, _DIAGONAL_R)
    )[None, :]
    count = tl.load(valid_count + panel)
    local_theta = tl.load(theta + panel * _CHUNK + targets)
    boundary_offset = panel * _RANK * _RANK + coordinates * (_RANK + 1)
    moment_j = local_theta * tl.load(boundary_j + boundary_offset)
    moment_d = local_theta * tl.load(boundary_d + boundary_offset)

    for source in tl.static_range(0, _CHUNK):
        source_active = source < count
        local_weight = tl.load(
            weights
            + (panel * _CHUNK + targets) * _CHUNK
            + source
        )
        local_u = tl.load(
            u + (panel * _CHUNK + source) * _RANK + coordinates,
            mask=source_active,
            other=0.0,
        ).to(tl.float32)
        local_h = tl.load(
            h + (panel * _CHUNK + source) * _RANK + coordinates,
            mask=source_active,
            other=0.0,
        ).to(tl.float32)
        moment_j += local_weight * local_u * local_u
        moment_d += local_weight * local_u * local_h

    panel_strength = tl.load(strength + panel).to(tl.float32)
    radius = 0.125
    centered_j = moment_j - 1.0 / 128.0
    log_diagonal = (
        radius * libdevice.tanh(panel_strength * centered_j / radius)
        + radius * libdevice.tanh(panel_strength * moment_d / radius)
    )
    value = tl.exp(log_diagonal)
    value = tl.where(targets < count, value, 1.0)
    tl.store(
        diagonal + (panel * _CHUNK + targets) * _RANK + coordinates,
        value,
    )


@triton.jit
def _diagonal_moment_reverse_kernel(
    u,
    h,
    theta,
    weights,
    boundary_j,
    boundary_d,
    strength,
    valid_count,
    grad_log_diagonal,
    grad_moment_j,
    grad_moment_d,
    grad_strength_partial,
):
    panel = tl.program_id(0)
    target_block = tl.program_id(1)
    coordinate_block = tl.program_id(2)
    target_vector = (
        target_block * _DIAGONAL_TARGET
        + tl.arange(0, _DIAGONAL_TARGET)
    )
    targets = target_vector[:, None]
    coordinates = (
        coordinate_block * _DIAGONAL_R
        + tl.arange(0, _DIAGONAL_R)
    )[None, :]
    count = tl.load(valid_count + panel)
    target_active = targets < count
    local_theta = tl.load(
        theta + panel * _CHUNK + targets,
        mask=target_active,
        other=0.0,
    )
    boundary_offset = panel * _RANK * _RANK + coordinates * (_RANK + 1)
    moment_j = local_theta * tl.load(boundary_j + boundary_offset)
    moment_d = local_theta * tl.load(boundary_d + boundary_offset)
    for source in tl.static_range(0, _CHUNK):
        source_active = source < count
        local_weight = tl.load(
            weights
            + (panel * _CHUNK + targets) * _CHUNK
            + source
        )
        local_u = tl.load(
            u + (panel * _CHUNK + source) * _RANK + coordinates,
            mask=source_active,
            other=0.0,
        ).to(tl.float32)
        local_h = tl.load(
            h + (panel * _CHUNK + source) * _RANK + coordinates,
            mask=source_active,
            other=0.0,
        ).to(tl.float32)
        moment_j += local_weight * local_u * local_u
        moment_d += local_weight * local_u * local_h

    panel_strength = tl.load(strength + panel).to(tl.float32)
    radius = 0.125
    centered_j = moment_j - 1.0 / 128.0
    tanh_j = libdevice.tanh(panel_strength * centered_j / radius)
    tanh_d = libdevice.tanh(panel_strength * moment_d / radius)
    sech_j = 1.0 - tanh_j * tanh_j
    sech_d = 1.0 - tanh_d * tanh_d
    grad_log = tl.load(
        grad_log_diagonal
        + (panel * _CHUNK + targets) * _RANK
        + coordinates,
        mask=target_active,
        other=0.0,
    )
    local_grad_j = grad_log * panel_strength * sech_j
    local_grad_d = grad_log * panel_strength * sech_d
    tl.store(
        grad_moment_j
        + (panel * _CHUNK + targets) * _RANK
        + coordinates,
        local_grad_j,
    )
    tl.store(
        grad_moment_d
        + (panel * _CHUNK + targets) * _RANK
        + coordinates,
        local_grad_d,
    )
    tl.store(
        grad_strength_partial
        + ((panel * _CHUNK + target_vector) * 8 + coordinate_block),
        tl.sum(grad_log * (sech_j * centered_j + sech_d * moment_d), axis=1),
    )


@triton.jit
def _diagonal_coefficient_reverse_kernel(
    u,
    h,
    boundary_j,
    boundary_d,
    grad_moment_j,
    grad_moment_d,
    grad_theta,
    grad_weights,
    valid_count,
):
    panel = tl.program_id(0)
    target_block = tl.program_id(1)
    source_block = tl.program_id(2)
    targets_vector = target_block * 4 + tl.arange(0, 4)
    sources_vector = source_block * 8 + tl.arange(0, 8)
    targets = targets_vector[:, None, None]
    sources = sources_vector[None, :, None]
    count = tl.load(valid_count + panel)
    target_active = targets < count
    source_active = sources < count
    weight_accumulator = tl.zeros((4, 8), tl.float32)
    theta_accumulator = tl.zeros((4,), tl.float32)

    for start in tl.static_range(0, _RANK, 16):
        coordinate_vector = start + tl.arange(0, 16)
        grad_j_2d = tl.load(
            grad_moment_j
            + (panel * _CHUNK + targets_vector[:, None]) * _RANK
            + coordinate_vector[None, :]
        )
        grad_d_2d = tl.load(
            grad_moment_d
            + (panel * _CHUNK + targets_vector[:, None]) * _RANK
            + coordinate_vector[None, :]
        )
        grad_j = grad_j_2d[:, None, :]
        grad_d = grad_d_2d[:, None, :]
        coordinates = coordinate_vector[None, None, :]
        local_u = tl.load(
            u + (panel * _CHUNK + sources) * _RANK + coordinates,
            mask=source_active,
            other=0.0,
        ).to(tl.float32)
        local_h = tl.load(
            h + (panel * _CHUNK + sources) * _RANK + coordinates,
            mask=source_active,
            other=0.0,
        ).to(tl.float32)
        weight_accumulator += tl.sum(
            grad_j * local_u * local_u + grad_d * local_u * local_h,
            axis=2,
        )
        boundary_offset = (
            panel * _RANK * _RANK
            + coordinate_vector * (_RANK + 1)
        )
        theta_accumulator += tl.sum(
            grad_j_2d * tl.load(boundary_j + boundary_offset)[None, :]
            + grad_d_2d
            * tl.load(boundary_d + boundary_offset)[None, :],
            axis=1,
        )

    tl.store(
        grad_weights
        + (panel * _CHUNK + targets_vector[:, None]) * _CHUNK
        + sources_vector[None, :],
        tl.where(
            (targets_vector[:, None] < count)
            & (sources_vector[None, :] < count),
            weight_accumulator,
            0.0,
        ),
    )
    if source_block == 0:
        tl.store(
            grad_theta + panel * _CHUNK + targets_vector,
            tl.where(targets_vector < count, theta_accumulator, 0.0),
        )


@triton.jit
def _diagonal_leaf_reverse_kernel(
    u,
    h,
    weights,
    grad_moment_j,
    grad_moment_d,
    grad_u,
    grad_h,
    valid_count,
):
    panel = tl.program_id(0)
    source = tl.program_id(1)
    coordinates = tl.arange(0, _RANK)
    source_active = source < tl.load(valid_count + panel)
    sum_j = tl.zeros((_RANK,), tl.float32)
    sum_d = tl.zeros((_RANK,), tl.float32)
    for target in tl.static_range(0, _CHUNK):
        local_weight = tl.load(
            weights + (panel * _CHUNK + target) * _CHUNK + source
        )
        sum_j += local_weight * tl.load(
            grad_moment_j
            + (panel * _CHUNK + target) * _RANK
            + coordinates
        )
        sum_d += local_weight * tl.load(
            grad_moment_d
            + (panel * _CHUNK + target) * _RANK
            + coordinates
        )
    local_u = tl.load(
        u + (panel * _CHUNK + source) * _RANK + coordinates,
        mask=source_active,
        other=0.0,
    ).to(tl.float32)
    local_h = tl.load(
        h + (panel * _CHUNK + source) * _RANK + coordinates,
        mask=source_active,
        other=0.0,
    ).to(tl.float32)
    output = (panel * _CHUNK + source) * _RANK + coordinates
    previous_u = tl.load(grad_u + output, mask=source_active, other=0.0)
    previous_h = tl.load(grad_h + output, mask=source_active, other=0.0)
    tl.store(
        grad_u + output,
        tl.where(
            source_active,
            previous_u + 2.0 * local_u * sum_j + local_h * sum_d,
            0.0,
        ),
    )
    tl.store(
        grad_h + output,
        tl.where(source_active, previous_h + local_u * sum_d, 0.0),
    )


@triton.jit
def _diagonal_boundary_reverse_kernel(
    theta,
    grad_moment_j,
    grad_moment_d,
    grad_boundary_j,
    grad_boundary_d,
):
    panel = tl.program_id(0)
    coordinates = tl.arange(0, _RANK)
    accumulator_j = tl.zeros((_RANK,), tl.float32)
    accumulator_d = tl.zeros((_RANK,), tl.float32)
    for target in tl.static_range(0, _CHUNK):
        local_theta = tl.load(theta + panel * _CHUNK + target)
        accumulator_j += local_theta * tl.load(
            grad_moment_j
            + (panel * _CHUNK + target) * _RANK
            + coordinates
        )
        accumulator_d += local_theta * tl.load(
            grad_moment_d
            + (panel * _CHUNK + target) * _RANK
            + coordinates
        )
    diagonal_offset = (
        panel * _RANK * _RANK + coordinates * (_RANK + 1)
    )
    tl.store(
        grad_boundary_j + diagonal_offset,
        tl.load(grad_boundary_j + diagonal_offset) + accumulator_j,
    )
    tl.store(
        grad_boundary_d + diagonal_offset,
        tl.load(grad_boundary_d + diagonal_offset) + accumulator_d,
    )


@triton.jit
def _temporal_reverse_kernel(
    log_decay,
    boundary_mass,
    valid_count,
    inverse_mass,
    theta,
    weights,
    grad_inverse,
    grad_theta_radial,
    grad_weights_radial,
    grad_theta_diagonal,
    grad_weights_diagonal,
    grad_theta_action,
    grad_weights_action,
    grad_log_decay,
    grad_boundary_mass,
    HAS_ACTION: tl.constexpr,
):
    """Reverse the scalar affine normalization in strict token order."""

    panel = tl.program_id(0)
    sources = tl.arange(0, _CHUNK)
    count = tl.load(valid_count + panel)
    carry_mass = 0.0
    carry_boundary_numerator = 0.0
    carry_row = tl.zeros((_CHUNK,), tl.float32)

    for reverse_target in tl.static_range(0, _CHUNK):
        target = _CHUNK - 1 - reverse_target
        active = target < count
        inverse = tl.load(inverse_mass + panel * _CHUNK + target)
        safe_inverse = tl.where(active, inverse, 1.0)
        local_theta = tl.load(theta + panel * _CHUNK + target)
        next_boundary_numerator = local_theta / safe_inverse
        rho = tl.exp(
            tl.load(
                log_decay + panel * _CHUNK + target,
                mask=active,
                other=0.0,
            ).to(tl.float32)
        )
        if target == 0:
            previous_mass = tl.load(boundary_mass + panel).to(tl.float32)
            previous_boundary_numerator = 1.0
            previous_row = tl.zeros((_CHUNK,), tl.float32)
        else:
            previous_inverse = tl.load(
                inverse_mass + panel * _CHUNK + target - 1
            )
            safe_previous_inverse = tl.where(active, previous_inverse, 1.0)
            previous_mass = 1.0 / safe_previous_inverse
            previous_boundary_numerator = (
                tl.load(theta + panel * _CHUNK + target - 1)
                / safe_previous_inverse
            )
            previous_row = tl.load(
                weights
                + (panel * _CHUNK + target - 1) * _CHUNK
                + sources
            )

        grad_row = (
            carry_row
            + tl.load(
                grad_weights_radial
                + (panel * _CHUNK + target) * _CHUNK
                + sources,
                mask=active,
                other=0.0,
            )
            + tl.load(
                grad_weights_diagonal
                + (panel * _CHUNK + target) * _CHUNK
                + sources,
                mask=active,
                other=0.0,
            )
        )
        if HAS_ACTION:
            grad_row += tl.load(
                grad_weights_action
                + (panel * _CHUNK + target) * _CHUNK
                + sources,
                mask=active,
                other=0.0,
            )
        local_grad_theta = (
            tl.load(
                grad_theta_radial + panel * _CHUNK + target,
                mask=active,
                other=0.0,
            )
            + tl.load(
                grad_theta_diagonal + panel * _CHUNK + target,
                mask=active,
                other=0.0,
            )
        )
        if HAS_ACTION:
            local_grad_theta += tl.load(
                grad_theta_action + panel * _CHUNK + target,
                mask=active,
                other=0.0,
            )
        local_grad_inverse = tl.load(
            grad_inverse + panel * _CHUNK + target,
            mask=active,
            other=0.0,
        )
        grad_next_boundary_numerator = (
            carry_boundary_numerator + local_grad_theta * safe_inverse
        )
        local_grad_inverse += (
            local_grad_theta * next_boundary_numerator
        )
        grad_retain = tl.sum(grad_row * previous_row, axis=0)
        next_carry_row = (1.0 - safe_inverse) * grad_row
        local_grad_inverse += (
            tl.sum(grad_row * tl.where(sources == target, 1.0, 0.0), axis=0)
            - grad_retain
        )
        grad_next_mass = carry_mass - local_grad_inverse * safe_inverse * safe_inverse
        grad_rho = (
            grad_next_mass * previous_mass
            + grad_next_boundary_numerator * previous_boundary_numerator
        )
        next_carry_mass = grad_next_mass * rho
        next_carry_boundary_numerator = grad_next_boundary_numerator * rho
        tl.store(
            grad_log_decay + panel * _CHUNK + target,
            tl.where(active, grad_rho * rho, 0.0),
        )
        carry_mass = tl.where(active, next_carry_mass, carry_mass)
        carry_boundary_numerator = tl.where(
            active,
            next_carry_boundary_numerator,
            carry_boundary_numerator,
        )
        carry_row = tl.where(active, next_carry_row, carry_row)

    tl.store(grad_boundary_mass + panel, carry_mass)


@triton.jit
def _combine_strength_reverse_kernel(
    radial_strength,
    diagonal_strength_partial,
    grad_strength,
):
    panel = tl.program_id(0)
    offsets = tl.arange(0, 256)
    diagonal = tl.sum(
        tl.load(diagonal_strength_partial + panel * 256 + offsets), axis=0
    )
    tl.store(grad_strength + panel, tl.load(radial_strength + panel) + diagonal)


def _check_tensor(
    tensor: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_same_device(
    reference: torch.device,
    tensors: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    for name, tensor in tensors:
        if tensor.device != reference:
            raise ValueError(f"{name} must be on the same CUDA device as u")


def _radial_compact_forward_launch(
    u: torch.Tensor,
    h: torch.Tensor,
    log_decay: torch.Tensor,
    strength: torch.Tensor,
    boundary_m: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    valid_count: torch.Tensor,
    return_saved: bool,
) -> RadialCompactOutput | tuple[RadialCompactOutput, RadialCompactSaved]:
    panels = u.shape[0]
    inverse_mass = torch.empty(
        panels, _HOST_CHUNK, device=u.device, dtype=torch.float32
    )
    theta = torch.empty_like(inverse_mass)
    weights = torch.empty(
        panels,
        _HOST_CHUNK,
        _HOST_CHUNK,
        device=u.device,
        dtype=torch.float32,
    )
    _temporal_coefficients_kernel[(panels,)](
        log_decay,
        boundary_m,
        valid_count,
        inverse_mass,
        theta,
        weights,
        num_warps=1,
    )

    norm_partial = torch.empty(
        panels,
        _HOST_CHUNK,
        4,
        _HOST_MATRIX_TILES,
        device=u.device,
        dtype=torch.float32,
    )
    _radial_residual_partial_kernel[(panels, 8, 8)](
        u,
        h,
        inverse_mass,
        theta,
        boundary_j,
        boundary_d,
        valid_count,
        norm_partial,
        num_warps=4,
    )
    radial_scale = torch.empty(
        panels, _HOST_CHUNK, 4, device=u.device, dtype=torch.float32
    )
    radial_q2 = torch.empty_like(radial_scale)
    radial_norm = torch.empty_like(radial_scale) if return_saved else radial_q2
    _radial_output_kernel[(panels, _HOST_CHUNK)](
        norm_partial,
        strength,
        valid_count,
        radial_scale,
        radial_q2,
        radial_norm,
        SAVE_NORM=return_saved,
        num_warps=4,
    )

    diagonal = torch.empty(
        panels,
        _HOST_CHUNK,
        _HOST_RANK,
        device=u.device,
        dtype=torch.float32,
    )
    _diagonal_kernel[(panels, 4, 8)](
        u,
        h,
        theta,
        weights,
        boundary_j,
        boundary_d,
        strength,
        valid_count,
        diagonal,
        num_warps=4,
    )
    output = RadialCompactOutput(
        inverse_mass,
        theta,
        weights,
        radial_scale,
        radial_q2,
        diagonal,
    )
    if not return_saved:
        return output
    return output, RadialCompactSaved(radial_norm)


def radial_compact_forward(
    u: torch.Tensor,
    h: torch.Tensor,
    log_decay: torch.Tensor,
    strength: torch.Tensor,
    boundary_m: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    *,
    valid_count: torch.Tensor | None = None,
    return_saved: bool = False,
) -> RadialCompactOutput | tuple[RadialCompactOutput, RadialCompactSaved]:
    """Construct compact radial/diagonal chart data for C32/r128 panels.

    ``u`` and ``h`` are the once-quantized BF16 operands. Persistent boundary
    states, temporal scalars, reductions, and outputs remain FP32. This
    standalone forward intentionally has no autograd registration; its reverse
    is the transpose of the same statistic graph, not a chain of generic VJPs.
    ``valid_count`` contains one value in ``[0,32]`` per panel; omitting it
    declares every panel full.

    Each radial norm is the sum of squares of one realized FP32 strict-matrix
    residual. The reverse replays that block recurrence and applies its
    transpose directly; it does not materialize tokenwise dense matrices or a
    chain of statistic VJPs.
    """

    if u.ndim != 3:
        raise ValueError("u must have shape [panels,32,128]")
    panels = u.shape[0]
    _check_tensor(u, "u", (panels, _HOST_CHUNK, _HOST_RANK), torch.bfloat16)
    _check_tensor(h, "h", tuple(u.shape), torch.bfloat16)
    _check_tensor(
        log_decay,
        "log_decay",
        (panels, _HOST_CHUNK),
        torch.float32,
    )
    _check_tensor(strength, "strength", (panels,), torch.float32)
    _check_tensor(boundary_m, "boundary_m", (panels,), torch.float32)
    _check_tensor(
        boundary_j,
        "boundary_j",
        (panels, _HOST_RANK, _HOST_RANK),
        torch.float32,
    )
    _check_tensor(boundary_d, "boundary_d", tuple(boundary_j.shape), torch.float32)
    if valid_count is None:
        with torch.cuda.device(u.device):
            valid_count = torch.full(
                (panels,),
                _HOST_CHUNK,
                device=u.device,
                dtype=torch.int32,
            )
    else:
        _check_tensor(valid_count, "valid_count", (panels,), torch.int32)
    _check_same_device(
        u.device,
        (
            ("h", h),
            ("log_decay", log_decay),
            ("strength", strength),
            ("boundary_m", boundary_m),
            ("boundary_j", boundary_j),
            ("boundary_d", boundary_d),
            ("valid_count", valid_count),
        ),
    )
    with torch.cuda.device(u.device):
        return _radial_compact_forward_launch(
            u,
            h,
            log_decay,
            strength,
            boundary_m,
            boundary_j,
            boundary_d,
            valid_count,
            return_saved,
        )


def _radial_compact_reverse_launch(
    u: torch.Tensor,
    h: torch.Tensor,
    log_decay: torch.Tensor,
    strength: torch.Tensor,
    boundary_m: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    output: RadialCompactOutput,
    saved: RadialCompactSaved,
    grad_radial_scale: torch.Tensor,
    grad_log_diagonal: torch.Tensor,
    valid_count: torch.Tensor,
    grad_theta_action: torch.Tensor | None,
    grad_weights_action: torch.Tensor | None,
    accumulated_gradients: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
) -> RadialCompactGradients:
    panels = u.shape[0]
    grad_norm = torch.empty_like(saved.radial_norm)
    grad_strength_radial = torch.empty_like(strength)
    _radial_scalar_reverse_kernel[(panels,)](
        saved.radial_norm,
        output.radial_scale,
        output.radial_q2,
        strength,
        valid_count,
        grad_radial_scale,
        grad_norm,
        grad_strength_radial,
        num_warps=4,
    )

    correlation_partial = torch.empty(
        panels,
        4,
        8,
        _HOST_CHUNK,
        device=u.device,
        dtype=torch.float32,
    )
    boundary_norm_partial = torch.empty(
        panels, 4, 8, device=u.device, dtype=torch.float32
    )
    vector_shape = (panels, _HOST_CHUNK, _HOST_RANK)
    add_to_grad = accumulated_gradients is not None
    if accumulated_gradients is None:
        grad_u = torch.empty(vector_shape, device=u.device, dtype=torch.float32)
        grad_h = torch.empty_like(grad_u)
        grad_boundary_j = torch.empty_like(boundary_j)
        grad_boundary_d = torch.empty_like(boundary_d)
    else:
        grad_u, grad_h, grad_boundary_j, grad_boundary_d = accumulated_gradients
    _radial_algebra_boundary_action_kernel[(panels, 8)](
        u,
        h,
        boundary_j,
        boundary_d,
        output.theta,
        output.weights,
        grad_norm,
        valid_count,
        grad_u,
        grad_h,
        correlation_partial,
        boundary_norm_partial,
        ADD_TO_GRAD=add_to_grad,
        num_warps=4,
        num_stages=3,
    )
    _radial_residual_boundary_gradient_kernel[(panels, 8, 8)](
        u,
        h,
        boundary_j,
        boundary_d,
        output.inverse_mass,
        output.theta,
        grad_norm,
        valid_count,
        grad_boundary_j,
        grad_boundary_d,
        ADD_TO_GRAD=add_to_grad,
        num_warps=2,
        num_stages=3,
    )
    grad_weights_radial = torch.empty_like(output.weights)
    _radial_algebra_local_kernel[(panels, _HOST_CHUNK)](
        u,
        h,
        output.theta,
        output.weights,
        grad_norm,
        valid_count,
        correlation_partial,
        grad_u,
        grad_h,
        grad_weights_radial,
        num_warps=4,
    )
    grad_theta_radial = torch.empty_like(output.theta)
    _radial_algebra_theta_kernel[(panels,)](
        output.theta,
        output.weights,
        grad_norm,
        valid_count,
        correlation_partial,
        boundary_norm_partial,
        grad_theta_radial,
        num_warps=4,
    )
    grad_inverse = torch.zeros_like(output.inverse_mass)

    grad_moment_j = torch.empty(
        panels,
        _HOST_CHUNK,
        _HOST_RANK,
        device=u.device,
        dtype=torch.float32,
    )
    grad_moment_d = torch.empty_like(grad_moment_j)
    grad_strength_diagonal_partial = torch.empty(
        panels,
        _HOST_CHUNK,
        8,
        device=u.device,
        dtype=torch.float32,
    )
    _diagonal_moment_reverse_kernel[(panels, 4, 8)](
        u,
        h,
        output.theta,
        output.weights,
        boundary_j,
        boundary_d,
        strength,
        valid_count,
        grad_log_diagonal,
        grad_moment_j,
        grad_moment_d,
        grad_strength_diagonal_partial,
        num_warps=4,
    )
    grad_theta_diagonal = torch.empty_like(output.theta)
    grad_weights_diagonal = torch.empty_like(output.weights)
    _diagonal_coefficient_reverse_kernel[(panels, 8, 4)](
        u,
        h,
        boundary_j,
        boundary_d,
        grad_moment_j,
        grad_moment_d,
        grad_theta_diagonal,
        grad_weights_diagonal,
        valid_count,
        num_warps=2,
    )
    _diagonal_leaf_reverse_kernel[(panels, _HOST_CHUNK)](
        u,
        h,
        output.weights,
        grad_moment_j,
        grad_moment_d,
        grad_u,
        grad_h,
        valid_count,
        num_warps=4,
    )
    _diagonal_boundary_reverse_kernel[(panels,)](
        output.theta,
        grad_moment_j,
        grad_moment_d,
        grad_boundary_j,
        grad_boundary_d,
        num_warps=4,
    )

    grad_log_decay = torch.empty_like(log_decay)
    grad_boundary_m = torch.empty_like(boundary_m)
    _temporal_reverse_kernel[(panels,)](
        log_decay,
        boundary_m,
        valid_count,
        output.inverse_mass,
        output.theta,
        output.weights,
        grad_inverse,
        grad_theta_radial,
        grad_weights_radial,
        grad_theta_diagonal,
        grad_weights_diagonal,
        grad_theta_action if grad_theta_action is not None else output.theta,
        grad_weights_action if grad_weights_action is not None else output.weights,
        grad_log_decay,
        grad_boundary_m,
        HAS_ACTION=grad_theta_action is not None,
        num_warps=1,
    )

    grad_strength = torch.empty_like(strength)
    _combine_strength_reverse_kernel[(panels,)](
        grad_strength_radial,
        grad_strength_diagonal_partial,
        grad_strength,
        num_warps=4,
    )
    return RadialCompactGradients(
        grad_u,
        grad_h,
        grad_log_decay,
        grad_strength,
        grad_boundary_m,
        grad_boundary_j,
        grad_boundary_d,
    )


def radial_compact_reverse(
    u: torch.Tensor,
    h: torch.Tensor,
    log_decay: torch.Tensor,
    strength: torch.Tensor,
    boundary_m: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    output: RadialCompactOutput,
    saved: RadialCompactSaved,
    grad_radial_scale: torch.Tensor,
    grad_log_diagonal: torch.Tensor,
    *,
    valid_count: torch.Tensor | None = None,
    grad_theta_action: torch.Tensor | None = None,
    grad_weights_action: torch.Tensor | None = None,
) -> RadialCompactGradients:
    """Apply the dedicated transpose of :func:`radial_compact_forward`.

    ``grad_log_diagonal`` is the cotangent after the solve action has already
    changed variables from ``diagonal`` to ``log(diagonal)``. This is the chart
    reverse interface and avoids an otherwise redundant elementwise VJP.
    Every returned partial is FP32 so the geometry scan can combine it before
    casting activation leaves.
    """

    if u.ndim != 3:
        raise ValueError("u must have shape [panels,32,128]")
    panels = u.shape[0]
    _check_tensor(u, "u", (panels, _HOST_CHUNK, _HOST_RANK), torch.bfloat16)
    _check_tensor(h, "h", tuple(u.shape), torch.bfloat16)
    _check_tensor(
        log_decay,
        "log_decay",
        (panels, _HOST_CHUNK),
        torch.float32,
    )
    _check_tensor(strength, "strength", (panels,), torch.float32)
    _check_tensor(boundary_m, "boundary_m", (panels,), torch.float32)
    _check_tensor(
        boundary_j,
        "boundary_j",
        (panels, _HOST_RANK, _HOST_RANK),
        torch.float32,
    )
    _check_tensor(boundary_d, "boundary_d", tuple(boundary_j.shape), torch.float32)
    _check_tensor(
        grad_radial_scale,
        "grad_radial_scale",
        (panels, _HOST_CHUNK, 4),
        torch.float32,
    )
    _check_tensor(
        grad_log_diagonal,
        "grad_log_diagonal",
        (panels, _HOST_CHUNK, _HOST_RANK),
        torch.float32,
    )
    if (grad_theta_action is None) != (grad_weights_action is None):
        raise ValueError(
            "grad_theta_action and grad_weights_action must be provided together"
        )
    if grad_theta_action is not None:
        _check_tensor(
            grad_theta_action,
            "grad_theta_action",
            (panels, _HOST_CHUNK),
            torch.float32,
        )
        _check_tensor(
            grad_weights_action,
            "grad_weights_action",
            (panels, _HOST_CHUNK, _HOST_CHUNK),
            torch.float32,
        )
    if valid_count is None:
        with torch.cuda.device(u.device):
            valid_count = torch.full(
                (panels,),
                _HOST_CHUNK,
                device=u.device,
                dtype=torch.int32,
            )
    else:
        _check_tensor(valid_count, "valid_count", (panels,), torch.int32)

    _check_tensor(
        output.inverse_mass,
        "output.inverse_mass",
        (panels, _HOST_CHUNK),
        torch.float32,
    )
    _check_tensor(output.theta, "output.theta", (panels, _HOST_CHUNK), torch.float32)
    _check_tensor(
        output.weights,
        "output.weights",
        (panels, _HOST_CHUNK, _HOST_CHUNK),
        torch.float32,
    )
    _check_tensor(
        output.radial_scale,
        "output.radial_scale",
        (panels, _HOST_CHUNK, 4),
        torch.float32,
    )
    _check_tensor(
        output.radial_q2,
        "output.radial_q2",
        (panels, _HOST_CHUNK, 4),
        torch.float32,
    )
    _check_tensor(
        output.diagonal,
        "output.diagonal",
        (panels, _HOST_CHUNK, _HOST_RANK),
        torch.float32,
    )
    _check_tensor(
        saved.radial_norm,
        "saved.radial_norm",
        (panels, _HOST_CHUNK, 4),
        torch.float32,
    )

    _check_same_device(
        u.device,
        (
            ("h", h),
            ("log_decay", log_decay),
            ("strength", strength),
            ("boundary_m", boundary_m),
            ("boundary_j", boundary_j),
            ("boundary_d", boundary_d),
            ("valid_count", valid_count),
            ("output.inverse_mass", output.inverse_mass),
            ("output.theta", output.theta),
            ("output.weights", output.weights),
            ("output.radial_scale", output.radial_scale),
            ("output.radial_q2", output.radial_q2),
            ("output.diagonal", output.diagonal),
            ("saved.radial_norm", saved.radial_norm),
            ("grad_radial_scale", grad_radial_scale),
            ("grad_log_diagonal", grad_log_diagonal),
        ),
    )
    with torch.cuda.device(u.device):
        return _radial_compact_reverse_launch(
            u,
            h,
            log_decay,
            strength,
            boundary_m,
            boundary_j,
            boundary_d,
            output,
            saved,
            grad_radial_scale,
            grad_log_diagonal,
            valid_count,
            grad_theta_action,
            grad_weights_action,
        )


def _radial_compact_reverse_accumulate_trusted(
    u: torch.Tensor,
    h: torch.Tensor,
    log_decay: torch.Tensor,
    strength: torch.Tensor,
    boundary_m: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    output: RadialCompactOutput,
    saved: RadialCompactSaved,
    grad_radial_scale: torch.Tensor,
    grad_log_diagonal: torch.Tensor,
    valid_count: torch.Tensor,
    grad_theta_action: torch.Tensor,
    grad_weights_action: torch.Tensor,
    grad_u: torch.Tensor,
    grad_h: torch.Tensor,
    grad_boundary_j: torch.Tensor,
    grad_boundary_d: torch.Tensor,
) -> RadialCompactGradients:
    """Accumulate into frame-owned partials with no standalone ABI checks."""

    with torch.cuda.device(u.device):
        return _radial_compact_reverse_launch(
            u,
            h,
            log_decay,
            strength,
            boundary_m,
            boundary_j,
            boundary_d,
            output,
            saved,
            grad_radial_scale,
            grad_log_diagonal,
            valid_count,
            grad_theta_action,
            grad_weights_action,
            (grad_u, grad_h, grad_boundary_j, grad_boundary_d),
        )

__all__ = [
    "RadialCompactGradients",
    "RadialCompactOutput",
    "RadialCompactSaved",
    "radial_compact_forward",
    "radial_compact_reverse",
]
