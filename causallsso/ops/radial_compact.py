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

_HOST_CHUNK = 32
_HOST_RANK = 128
_RADIUS = 1.0 / 8.0


@triton.jit
def _panel_count(panel, length, heads: tl.constexpr, chunks: tl.constexpr):
    chunk = panel % chunks
    return tl.minimum(_CHUNK, length - chunk * _CHUNK)


@triton.jit
def _raw_vector_offsets(
    panel,
    local_tokens,
    coordinates,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
):
    panel = panel.to(tl.int64)
    chunk = panel % chunks
    head_batch = panel // chunks
    head = head_batch % heads
    batch = head_batch // heads
    tokens = chunk * _CHUNK + local_tokens
    return (
        (batch * length * heads + tokens * heads + head) * _RANK
        + coordinates
    )


@triton.jit
def _raw_scalar_offsets(
    panel,
    local_tokens,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
):
    panel = panel.to(tl.int64)
    chunk = panel % chunks
    head_batch = panel // chunks
    head = head_batch % heads
    batch = head_batch // heads
    tokens = chunk * _CHUNK + local_tokens
    return batch * length * heads + tokens * heads + head


@triton.jit
def _panel_head(panel, heads: tl.constexpr, chunks: tl.constexpr):
    return (panel // chunks) % heads


class RadialCompactOutput(NamedTuple):
    """Chunk-local scalar chart data in panel-major order."""

    inverse_mass: torch.Tensor
    theta: torch.Tensor
    weights: torch.Tensor
    radial_scale: torch.Tensor
    radial_q2: torch.Tensor
    diagonal: torch.Tensor


class RadialCompactSaved(NamedTuple):
    """Reduced forward statistics consumed directly by the dedicated reverse."""

    radial_norm: torch.Tensor
    gram: torch.Tensor
    boundary_pair: torch.Tensor
    boundary_norm: torch.Tensor


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
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
):
    panel = tl.program_id(0)
    sources = tl.arange(0, _CHUNK)
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    mass = tl.load(boundary_mass + panel).to(tl.float32)
    boundary_numerator = 1.0
    previous = tl.zeros((_CHUNK,), tl.float32)

    for target in tl.static_range(0, _CHUNK):
        active = target < count
        decay_offset = (
            _raw_scalar_offsets(panel, target, length, heads, chunks)
            if RAW_LAYOUT
            else panel * _CHUNK + target
        )
        log_rho = tl.load(log_decay + decay_offset, mask=active, other=0.0)
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
def _radial_pair_statistics_kernel(
    u,
    h,
    boundary_j,
    boundary_d,
    valid_count,
    gram_partial,
    boundary_pair_partial,
    boundary_norm_partial,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
    ROUTE: tl.constexpr,
):
    """MESA-style Gram/Hadamard statistics for one strict chart route."""

    panel = tl.program_id(0)
    row_block = tl.program_id(1)
    tokens = tl.arange(0, _CHUNK)
    local = tl.arange(0, _MATRIX_TILE)
    rows = row_block * _MATRIX_TILE + local
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    active = tokens < count
    panel_vector = panel * _CHUNK * _RANK
    panel_matrix = panel * _RANK * _RANK

    row_offsets = (
        _raw_vector_offsets(
            panel, tokens[:, None], rows[None, :], length, heads, chunks
        )
        if RAW_LAYOUT
        else panel_vector + tokens[:, None] * _RANK + rows[None, :]
    )
    row_u = tl.load(
        u + row_offsets,
        mask=active[:, None],
        other=0.0,
    )
    row_u_fp32 = row_u.to(tl.float32)
    gram = tl.zeros((_CHUNK, _CHUNK), tl.float32)
    boundary_pair = tl.zeros((_CHUNK,), tl.float32)
    boundary_norm = 0.0

    for column_block in tl.static_range(0, 8):
        include = column_block > row_block if ROUTE >= 2 else column_block < row_block
        columns = column_block * _MATRIX_TILE + local
        if ROUTE & 1:
            column_offsets = (
                _raw_vector_offsets(
                    panel,
                    tokens[:, None],
                    columns[None, :],
                    length,
                    heads,
                    chunks,
                )
                if RAW_LAYOUT
                else panel_vector
                + tokens[:, None] * _RANK
                + columns[None, :]
            )
            column_values = tl.load(
                h + column_offsets,
                mask=include & active[:, None],
                other=0.0,
            )
        else:
            column_offsets = (
                _raw_vector_offsets(
                    panel,
                    tokens[:, None],
                    columns[None, :],
                    length,
                    heads,
                    chunks,
                )
                if RAW_LAYOUT
                else panel_vector
                + tokens[:, None] * _RANK
                + columns[None, :]
            )
            column_values = tl.load(
                u + column_offsets,
                mask=include & active[:, None],
                other=0.0,
            )
        if ROUTE & 1:
            boundary = tl.load(
                boundary_d
                + panel_matrix
                + rows[:, None] * _RANK
                + columns[None, :],
                mask=include,
                other=0.0,
            ).to(tl.float32)
        else:
            boundary = tl.load(
                boundary_j
                + panel_matrix
                + rows[:, None] * _RANK
                + columns[None, :],
                mask=include,
                other=0.0,
            ).to(tl.float32)

        row_gram = tl.dot(row_u, tl.trans(row_u))
        column_gram = tl.dot(column_values, tl.trans(column_values))
        gram += row_gram * column_gram

        # The persistent FP32 boundary has no FP16 range certificate. Its
        # named broad action therefore uses the frozen direct-BF16 schedule,
        # while the surrounding products and reductions remain FP32.
        boundary_action = tl.dot(
            boundary.to(tl.bfloat16),
            tl.trans(column_values.to(tl.bfloat16)),
        )
        boundary_pair += tl.sum(
            row_u_fp32 * tl.trans(boundary_action), axis=1
        )
        boundary_norm += tl.sum(boundary * boundary)

    if ROUTE & 1:
        diagonal_values = tl.load(
            h + row_offsets,
            mask=active[:, None],
            other=0.0,
        ).to(tl.float32)
    else:
        diagonal_values = row_u_fp32
    prefix = tl.zeros((_CHUNK, _CHUNK), tl.float32)
    diagonal_gram = tl.zeros((_CHUNK, _CHUNK), tl.float32)
    if ROUTE >= 2:
        for step in tl.static_range(0, _MATRIX_TILE):
            coordinate = _MATRIX_TILE - 1 - step
            selected = local == coordinate
            left = tl.sum(row_u_fp32 * selected[None, :], axis=1)
            right = tl.sum(diagonal_values * selected[None, :], axis=1)
            left_pair = left[:, None] * left[None, :]
            diagonal_gram += left_pair * prefix
            prefix += right[:, None] * right[None, :]
    else:
        for coordinate in tl.static_range(0, _MATRIX_TILE):
            selected = local == coordinate
            left = tl.sum(row_u_fp32 * selected[None, :], axis=1)
            right = tl.sum(diagonal_values * selected[None, :], axis=1)
            left_pair = left[:, None] * left[None, :]
            diagonal_gram += left_pair * prefix
            prefix += right[:, None] * right[None, :]
    gram += diagonal_gram

    if ROUTE & 1:
        diagonal_boundary = tl.load(
            boundary_d
            + panel_matrix
            + rows[:, None] * _RANK
            + rows[None, :]
        ).to(tl.float32)
    else:
        diagonal_boundary = tl.load(
            boundary_j
            + panel_matrix
            + rows[:, None] * _RANK
            + rows[None, :]
        ).to(tl.float32)
    diagonal_mask = (
        local[None, :] > local[:, None]
        if ROUTE >= 2
        else local[None, :] < local[:, None]
    )
    diagonal_boundary = tl.where(diagonal_mask, diagonal_boundary, 0.0)
    diagonal_action = tl.dot(
        diagonal_boundary.to(tl.bfloat16),
        tl.trans(diagonal_values.to(tl.bfloat16)),
    )
    boundary_pair += tl.sum(
        row_u_fp32 * tl.trans(diagonal_action), axis=1
    )
    boundary_norm += tl.sum(diagonal_boundary * diagonal_boundary)

    gram_base = (
        ((panel * 4 + ROUTE) * 8 + row_block) * _CHUNK
        + tokens[:, None]
    ) * _CHUNK + tokens[None, :]
    tl.store(gram_partial + gram_base, gram)
    vector_base = ((panel * 4 + ROUTE) * 8 + row_block) * _CHUNK + tokens
    tl.store(
        boundary_pair_partial + vector_base,
        tl.where(active, boundary_pair, 0.0),
    )
    tl.store(
        boundary_norm_partial + (panel * 4 + ROUTE) * 8 + row_block,
        boundary_norm,
    )


@triton.jit
def _radial_pair_output_kernel(
    theta,
    weights,
    strength,
    valid_count,
    gram_partial,
    boundary_pair_partial,
    boundary_norm_partial,
    radial_scale,
    radial_q2,
    radial_norm,
    saved_gram,
    saved_boundary_pair,
    saved_boundary_norm,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
    SAVE_STATISTICS: tl.constexpr,
):
    panel = tl.program_id(0)
    route = tl.program_id(1)
    targets = tl.arange(0, _CHUNK)
    sources = tl.arange(0, _CHUNK)
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    active_target = targets < count
    active_source = sources < count
    temporal = tl.load(
        weights
        + (panel * _CHUNK + targets[:, None]) * _CHUNK
        + sources[None, :],
        mask=active_target[:, None] & active_source[None, :],
        other=0.0,
    ).to(tl.float32)
    gram = tl.zeros((_CHUNK, _CHUNK), tl.float32)
    boundary_pair = tl.zeros((_CHUNK,), tl.float32)
    boundary_norm = 0.0
    for block in tl.static_range(0, 8):
        gram_base = (
            ((panel * 4 + route) * 8 + block) * _CHUNK
            + targets[:, None]
        ) * _CHUNK + sources[None, :]
        gram += tl.load(gram_partial + gram_base)
        vector_base = ((panel * 4 + route) * 8 + block) * _CHUNK + sources
        boundary_pair += tl.load(
            boundary_pair_partial + vector_base,
            mask=active_source,
            other=0.0,
        )
        boundary_norm += tl.load(
            boundary_norm_partial + (panel * 4 + route) * 8 + block
        )

    # This is MESA's matrix-free ((P K^T) odot M)V pattern specialized to
    # the symmetric local generator Gram. The dot operands are BF16 and every
    # result is accumulated and completed in FP32.
    local_action = tl.dot(
        temporal.to(tl.bfloat16), gram.to(tl.bfloat16)
    )
    local_norm = tl.sum(local_action * temporal, axis=1)
    boundary_cross = tl.sum(
        temporal * boundary_pair[None, :], axis=1
    )
    local_theta = tl.load(
        theta + panel * _CHUNK + targets,
        mask=active_target,
        other=0.0,
    ).to(tl.float32)
    norm = (
        local_theta * local_theta * boundary_norm
        + 2.0 * local_theta * boundary_cross
        + local_norm
    )
    strength_index = (
        _panel_head(panel, heads, chunks) if RAW_LAYOUT else panel
    )
    panel_strength = tl.load(strength + strength_index).to(tl.float32)
    radius = 0.125
    q2 = radius * radius + panel_strength * panel_strength * norm
    scale = panel_strength * radius * tl.rsqrt(q2)
    output = (panel * _CHUNK + targets) * 4 + route
    tl.store(radial_q2 + output, tl.where(active_target, q2, radius * radius))
    tl.store(radial_scale + output, tl.where(active_target, scale, 0.0))
    if SAVE_STATISTICS:
        tl.store(radial_norm + output, tl.where(active_target, norm, 0.0))
        saved_gram_base = (
            ((panel * 4 + route) * _CHUNK + targets[:, None]) * _CHUNK
            + sources[None, :]
        )
        tl.store(saved_gram + saved_gram_base, gram)
        saved_pair_base = (panel * 4 + route) * _CHUNK + sources
        tl.store(saved_boundary_pair + saved_pair_base, boundary_pair)
        tl.store(saved_boundary_norm + panel * 4 + route, boundary_norm)


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
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
):
    panel = tl.program_id(0)
    targets = tl.arange(0, _CHUNK)[:, None]
    routes = tl.arange(0, 4)[None, :]
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    active = targets < count
    offsets = (panel * _CHUNK + targets) * 4 + routes
    norm = tl.load(radial_norm + offsets, mask=active, other=0.0)
    q2 = tl.load(radial_q2 + offsets, mask=active, other=1.0)
    scale = tl.load(radial_scale + offsets, mask=active, other=0.0)
    grad_scale = tl.load(
        grad_radial_scale + offsets,
        mask=active,
        other=0.0,
    )
    strength_index = (
        _panel_head(panel, heads, chunks) if RAW_LAYOUT else panel
    )
    panel_strength = tl.load(strength + strength_index).to(tl.float32)
    radius = 0.125
    grad_q2 = -0.5 * grad_scale * scale / q2
    # A zero from the Gram/Hadamard expansion may be a rounded cancellation,
    # not a structural zero of the underlying strict matrix. Its transpose
    # therefore keeps the scalar cotangent; structural zeros vanish naturally
    # in the subsequent linear contractions.
    local_grad_norm = grad_q2 * panel_strength * panel_strength
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
def _radial_pair_boundary_leaf_transpose_kernel(
    u,
    h,
    boundary_j,
    boundary_d,
    boundary_coefficient,
    valid_count,
    grad_u,
    grad_h,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
    ADD_TO_GRAD: tl.constexpr,
):
    """Apply the four boundary-pair cotangents as broad C32 products."""

    panel = tl.program_id(0)
    row_block = tl.program_id(1)
    rows = row_block * _MATRIX_TILE + tl.arange(0, _MATRIX_TILE)
    sources = tl.arange(0, _CHUNK)
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    active_source = sources < count
    coefficient_lj = tl.load(
        boundary_coefficient + panel * 4 * _CHUNK + sources,
        mask=active_source,
        other=0.0,
    )
    coefficient_ld = tl.load(
        boundary_coefficient + (panel * 4 + 1) * _CHUNK + sources,
        mask=active_source,
        other=0.0,
    )
    coefficient_uj = tl.load(
        boundary_coefficient + (panel * 4 + 2) * _CHUNK + sources,
        mask=active_source,
        other=0.0,
    )
    coefficient_ud = tl.load(
        boundary_coefficient + (panel * 4 + 3) * _CHUNK + sources,
        mask=active_source,
        other=0.0,
    )

    panel_vector = panel * _CHUNK * _RANK
    panel_matrix = panel * _RANK * _RANK
    grad_u_rows = tl.zeros((_MATRIX_TILE, _CHUNK), tl.float32)
    grad_h_rows = tl.zeros_like(grad_u_rows)

    for start in tl.static_range(0, _RANK, 32):
        columns = start + tl.arange(0, 32)
        source_offsets = (
            _raw_vector_offsets(
                panel,
                sources[:, None],
                columns[None, :],
                length,
                heads,
                chunks,
            )
            if RAW_LAYOUT
            else panel_vector
            + sources[:, None] * _RANK
            + columns[None, :]
        )
        u_columns = tl.load(
            u + source_offsets,
            mask=active_source[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        h_columns = tl.load(
            h + source_offsets,
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
    output = (
        _raw_vector_offsets(
            panel,
            sources[:, None],
            rows[None, :],
            length,
            heads,
            chunks,
        )
        if RAW_LAYOUT
        else panel_vector + sources[:, None] * _RANK + rows[None, :]
    )
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
    if RAW_LAYOUT:
        tl.store(
            grad_u + output, result_u, mask=active_source[:, None]
        )
        tl.store(
            grad_h + output, result_h, mask=active_source[:, None]
        )
    else:
        tl.store(
            grad_u + output,
            tl.where(active_source[:, None], result_u, 0.0),
        )
        tl.store(
            grad_h + output,
            tl.where(active_source[:, None], result_h, 0.0),
        )


@triton.jit
def _radial_pair_boundary_transpose_kernel(
    u,
    h,
    boundary_j,
    boundary_d,
    theta,
    grad_norm,
    boundary_coefficient,
    valid_count,
    grad_boundary_j,
    grad_boundary_d,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
    ADD_TO_GRAD: tl.constexpr,
):
    panel = tl.program_id(0)
    row_block = tl.program_id(1)
    column_block = tl.program_id(2)
    rows = row_block * _MATRIX_TILE + tl.arange(0, _MATRIX_TILE)
    columns = column_block * _MATRIX_TILE + tl.arange(0, _MATRIX_TILE)
    tokens = tl.arange(0, _CHUNK)
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    active = tokens < count
    panel_vector = panel * _CHUNK * _RANK
    row_offsets = (
        _raw_vector_offsets(
            panel, tokens[:, None], rows[None, :], length, heads, chunks
        )
        if RAW_LAYOUT
        else panel_vector + tokens[:, None] * _RANK + rows[None, :]
    )
    column_offsets = (
        _raw_vector_offsets(
            panel,
            tokens[:, None],
            columns[None, :],
            length,
            heads,
            chunks,
        )
        if RAW_LAYOUT
        else panel_vector + tokens[:, None] * _RANK + columns[None, :]
    )
    matrix_offset = (
        panel * _RANK * _RANK
        + rows[:, None] * _RANK
        + columns[None, :]
    )
    local_u_rows = tl.load(
        u + row_offsets,
        mask=active[:, None],
        other=0.0,
    )
    local_u_columns = tl.load(
        u + column_offsets,
        mask=active[:, None],
        other=0.0,
    )
    local_h_columns = tl.load(
        h + column_offsets,
        mask=active[:, None],
        other=0.0,
    )
    local_theta = tl.load(
        theta + panel * _CHUNK + tokens,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    lower = columns[None, :] < rows[:, None]
    upper = columns[None, :] > rows[:, None]
    boundary_j_value = tl.load(boundary_j + matrix_offset).to(tl.float32)
    boundary_d_value = tl.load(boundary_d + matrix_offset).to(tl.float32)
    result_j = tl.zeros((_MATRIX_TILE, _MATRIX_TILE), tl.float32)
    result_d = tl.zeros((_MATRIX_TILE, _MATRIX_TILE), tl.float32)

    for route in tl.static_range(0, 4):
        beta = 2.0 * tl.load(
            grad_norm + (panel * _CHUNK + tokens) * 4 + route,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        boundary_scale = tl.sum(beta * local_theta * local_theta, axis=0)
        coefficient = tl.load(
            boundary_coefficient + (panel * 4 + route) * _CHUNK + tokens,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        right = local_h_columns if route & 1 else local_u_columns
        local_term = tl.dot(
            tl.trans(local_u_rows.to(tl.bfloat16)),
            (coefficient[:, None] * right).to(tl.bfloat16),
        )
        mask = upper if route >= 2 else lower
        if route & 1:
            result_d += tl.where(
                mask, boundary_scale * boundary_d_value + local_term, 0.0
            )
        else:
            result_j += tl.where(
                mask, boundary_scale * boundary_j_value + local_term, 0.0
            )

    if ADD_TO_GRAD:
        result_j += tl.load(grad_boundary_j + matrix_offset).to(tl.float32)
        result_d += tl.load(grad_boundary_d + matrix_offset).to(tl.float32)
    tl.store(grad_boundary_j + matrix_offset, result_j)
    tl.store(grad_boundary_d + matrix_offset, result_d)


@triton.jit
def _radial_pair_scalar_reverse_kernel(
    theta,
    weights,
    grad_norm,
    valid_count,
    gram,
    boundary_pair,
    boundary_norm,
    grad_theta_partial,
    grad_weights_partial,
    boundary_coefficient,
    local_coefficient,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
):
    panel = tl.program_id(0)
    route = tl.program_id(1)
    targets = tl.arange(0, _CHUNK)
    sources = tl.arange(0, _CHUNK)
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    active_target = targets < count
    active_source = sources < count
    temporal = tl.load(
        weights
        + (panel * _CHUNK + targets[:, None]) * _CHUNK
        + sources[None, :],
        mask=active_target[:, None] & active_source[None, :],
        other=0.0,
    ).to(tl.float32)
    local_theta = tl.load(
        theta + panel * _CHUNK + targets,
        mask=active_target,
        other=0.0,
    ).to(tl.float32)
    beta = 2.0 * tl.load(
        grad_norm + (panel * _CHUNK + targets) * 4 + route,
        mask=active_target,
        other=0.0,
    ).to(tl.float32)
    gram_base = (
        ((panel * 4 + route) * _CHUNK + targets[:, None]) * _CHUNK
        + sources[None, :]
    )
    local_gram = tl.load(
        gram + gram_base,
        mask=active_target[:, None] & active_source[None, :],
        other=0.0,
    ).to(tl.float32)
    pair_base = (panel * 4 + route) * _CHUNK + sources
    local_boundary_pair = tl.load(
        boundary_pair + pair_base,
        mask=active_source,
        other=0.0,
    ).to(tl.float32)
    local_boundary_norm = tl.load(
        boundary_norm + panel * 4 + route
    ).to(tl.float32)

    local_action = tl.dot(
        temporal.to(tl.bfloat16), local_gram.to(tl.bfloat16)
    )
    boundary_cross = tl.sum(
        temporal * local_boundary_pair[None, :], axis=1
    )
    grad_theta_route = beta * (
        local_theta * local_boundary_norm + boundary_cross
    )
    grad_weights_route = beta[:, None] * (
        local_theta[:, None] * local_boundary_pair[None, :] + local_action
    )
    weighted_temporal = beta[:, None] * temporal
    boundary_route = tl.sum(
        weighted_temporal * local_theta[:, None], axis=0
    )
    local_route = tl.dot(
        tl.trans(temporal.to(tl.bfloat16)),
        weighted_temporal.to(tl.bfloat16),
    )

    theta_base = (panel * 4 + route) * _CHUNK + targets
    tl.store(
        grad_theta_partial + theta_base,
        tl.where(active_target, grad_theta_route, 0.0),
    )
    matrix_base = (
        ((panel * 4 + route) * _CHUNK + targets[:, None]) * _CHUNK
        + sources[None, :]
    )
    matrix_mask = active_target[:, None] & active_source[None, :]
    tl.store(
        grad_weights_partial + matrix_base,
        tl.where(matrix_mask, grad_weights_route, 0.0),
    )
    tl.store(
        boundary_coefficient + theta_base,
        tl.where(active_source, boundary_route, 0.0),
    )
    tl.store(
        local_coefficient + matrix_base,
        tl.where(matrix_mask, local_route, 0.0),
    )


@triton.jit
def _reduce_radial_pair_scalar_reverse_kernel(
    grad_theta_partial,
    grad_weights_partial,
    grad_theta,
    grad_weights,
):
    panel = tl.program_id(0)
    targets = tl.arange(0, _CHUNK)
    sources = tl.arange(0, _CHUNK)
    theta_result = tl.zeros((_CHUNK,), tl.float32)
    weight_result = tl.zeros((_CHUNK, _CHUNK), tl.float32)
    for route in tl.static_range(0, 4):
        theta_result += tl.load(
            grad_theta_partial + (panel * 4 + route) * _CHUNK + targets
        )
        weight_result += tl.load(
            grad_weights_partial
            + ((panel * 4 + route) * _CHUNK + targets[:, None]) * _CHUNK
            + sources[None, :]
        )
    tl.store(grad_theta + panel * _CHUNK + targets, theta_result)
    tl.store(
        grad_weights
        + (panel * _CHUNK + targets[:, None]) * _CHUNK
        + sources[None, :],
        weight_result,
    )


@triton.jit
def _radial_pair_offdiagonal_transpose_kernel(
    u,
    h,
    local_coefficient,
    valid_count,
    grad_u,
    grad_h,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
):
    panel = tl.program_id(0)
    output_block = tl.program_id(1)
    tokens = tl.arange(0, _CHUNK)
    local = tl.arange(0, _MATRIX_TILE)
    output_coordinates = output_block * _MATRIX_TILE + local
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    active = tokens < count
    panel_vector = panel * _CHUNK * _RANK
    output_offsets = (
        _raw_vector_offsets(
            panel,
            tokens[:, None],
            output_coordinates[None, :],
            length,
            heads,
            chunks,
        )
        if RAW_LAYOUT
        else panel_vector
        + tokens[:, None] * _RANK
        + output_coordinates[None, :]
    )
    output_u = tl.load(
        u + output_offsets,
        mask=active[:, None],
        other=0.0,
    )
    output_h = tl.load(
        h + output_offsets,
        mask=active[:, None],
        other=0.0,
    )
    result_u = tl.zeros((_CHUNK, _MATRIX_TILE), tl.float32)
    result_h = tl.zeros((_CHUNK, _MATRIX_TILE), tl.float32)
    token_mask = active[:, None] & active[None, :]

    for route in tl.static_range(0, 4):
        coefficient = tl.load(
            local_coefficient
            + ((panel * 4 + route) * _CHUNK + tokens[:, None]) * _CHUNK
            + tokens[None, :],
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        for other_block in tl.static_range(0, 8):
            other_coordinates = other_block * _MATRIX_TILE + local
            other_offsets = (
                _raw_vector_offsets(
                    panel,
                    tokens[:, None],
                    other_coordinates[None, :],
                    length,
                    heads,
                    chunks,
                )
                if RAW_LAYOUT
                else panel_vector
                + tokens[:, None] * _RANK
                + other_coordinates[None, :]
            )
            row_active = (
                other_block < output_block
                if route < 2
                else other_block > output_block
            )
            column_active = (
                other_block > output_block
                if route < 2
                else other_block < output_block
            )
            if route & 1:
                other_right = tl.load(
                    h + other_offsets,
                    mask=row_active & active[:, None],
                    other=0.0,
                )
            else:
                other_right = tl.load(
                    u + other_offsets,
                    mask=row_active & active[:, None],
                    other=0.0,
                )
            right_gram = tl.dot(other_right, tl.trans(other_right))
            row_action = tl.dot(
                (coefficient * right_gram).to(tl.bfloat16),
                output_u.to(tl.bfloat16),
            )
            result_u += tl.where(row_active, row_action, 0.0)

            other_left = tl.load(
                u + other_offsets,
                mask=column_active & active[:, None],
                other=0.0,
            )
            left_gram = tl.dot(other_left, tl.trans(other_left))
            column_operand = output_h if route & 1 else output_u
            column_action = tl.dot(
                (tl.trans(coefficient) * left_gram).to(tl.bfloat16),
                column_operand.to(tl.bfloat16),
            )
            if route & 1:
                result_h += tl.where(column_active, column_action, 0.0)
            else:
                result_u += tl.where(column_active, column_action, 0.0)

    result_u += tl.load(
        grad_u + output_offsets, mask=active[:, None], other=0.0
    )
    result_h += tl.load(
        grad_h + output_offsets, mask=active[:, None], other=0.0
    )
    if RAW_LAYOUT:
        tl.store(grad_u + output_offsets, result_u, mask=active[:, None])
        tl.store(grad_h + output_offsets, result_h, mask=active[:, None])
    else:
        tl.store(
            grad_u + output_offsets,
            tl.where(active[:, None], result_u, 0.0),
        )
        tl.store(
            grad_h + output_offsets,
            tl.where(active[:, None], result_h, 0.0),
        )


@triton.jit
def _radial_pair_diagonal_transpose_kernel(
    u,
    h,
    local_coefficient,
    valid_count,
    grad_u,
    grad_h,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
):
    panel = tl.program_id(0)
    source = tl.program_id(1)
    partners = tl.arange(0, _CHUNK)[:, None]
    local = tl.arange(0, _MATRIX_TILE)[None, :]
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    source_active = source < count
    partner_active = partners < count
    panel_vector = panel * _CHUNK * _RANK

    for block in tl.static_range(0, 8):
        coordinates = block * _MATRIX_TILE + local
        source_offsets = (
            _raw_vector_offsets(
                panel, source, coordinates, length, heads, chunks
            )
            if RAW_LAYOUT
            else panel_vector + source * _RANK + coordinates
        )
        partner_offsets = (
            _raw_vector_offsets(
                panel, partners, coordinates, length, heads, chunks
            )
            if RAW_LAYOUT
            else panel_vector + partners * _RANK + coordinates
        )
        source_u = tl.load(
            u + source_offsets,
            mask=source_active,
            other=0.0,
        ).to(tl.float32)
        source_h = tl.load(
            h + source_offsets,
            mask=source_active,
            other=0.0,
        ).to(tl.float32)
        partner_u = tl.load(
            u + partner_offsets,
            mask=partner_active,
            other=0.0,
        ).to(tl.float32)
        partner_h = tl.load(
            h + partner_offsets,
            mask=partner_active,
            other=0.0,
        ).to(tl.float32)
        product_u = partner_u * source_u
        product_h = partner_h * source_h
        prefix_u = tl.cumsum(product_u, axis=1) - product_u
        prefix_h = tl.cumsum(product_h, axis=1) - product_h
        suffix_u = tl.sum(product_u, axis=1)[:, None] - product_u - prefix_u
        suffix_h = tl.sum(product_h, axis=1)[:, None] - product_h - prefix_h
        partner_ids = tl.arange(0, _CHUNK)
        coefficient_lj = tl.load(
            local_coefficient
            + (panel * 4 * _CHUNK + source) * _CHUNK
            + partner_ids,
            mask=partner_ids < count,
            other=0.0,
        )[:, None]
        coefficient_ld = tl.load(
            local_coefficient
            + ((panel * 4 + 1) * _CHUNK + source) * _CHUNK
            + partner_ids,
            mask=partner_ids < count,
            other=0.0,
        )[:, None]
        coefficient_uj = tl.load(
            local_coefficient
            + ((panel * 4 + 2) * _CHUNK + source) * _CHUNK
            + partner_ids,
            mask=partner_ids < count,
            other=0.0,
        )[:, None]
        coefficient_ud = tl.load(
            local_coefficient
            + ((panel * 4 + 3) * _CHUNK + source) * _CHUNK
            + partner_ids,
            mask=partner_ids < count,
            other=0.0,
        )[:, None]
        local_grad_u = tl.sum(
            partner_u
            * (
                (coefficient_lj + coefficient_uj)
                * (prefix_u + suffix_u)
                + coefficient_ld * prefix_h
                + coefficient_ud * suffix_h
            ),
            axis=0,
        )
        local_grad_h = tl.sum(
            partner_h
            * (
                coefficient_ld * suffix_u
                + coefficient_ud * prefix_u
            ),
            axis=0,
        )
        output = source_offsets
        tl.store(
            grad_u + output,
            tl.load(grad_u + output, mask=source_active, other=0.0)
            + local_grad_u,
            mask=source_active,
        )
        tl.store(
            grad_h + output,
            tl.load(grad_h + output, mask=source_active, other=0.0)
            + local_grad_h,
            mask=source_active,
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
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
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
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
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
        vector_offset = (
            _raw_vector_offsets(
                panel, source, coordinates, length, heads, chunks
            )
            if RAW_LAYOUT
            else (panel * _CHUNK + source) * _RANK + coordinates
        )
        local_u = tl.load(
            u + vector_offset,
            mask=source_active,
            other=0.0,
        ).to(tl.float32)
        local_h = tl.load(
            h + vector_offset,
            mask=source_active,
            other=0.0,
        ).to(tl.float32)
        moment_j += local_weight * local_u * local_u
        moment_d += local_weight * local_u * local_h

    strength_index = (
        _panel_head(panel, heads, chunks) if RAW_LAYOUT else panel
    )
    panel_strength = tl.load(strength + strength_index).to(tl.float32)
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
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
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
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
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
        vector_offset = (
            _raw_vector_offsets(
                panel, source, coordinates, length, heads, chunks
            )
            if RAW_LAYOUT
            else (panel * _CHUNK + source) * _RANK + coordinates
        )
        local_u = tl.load(
            u + vector_offset,
            mask=source_active,
            other=0.0,
        ).to(tl.float32)
        local_h = tl.load(
            h + vector_offset,
            mask=source_active,
            other=0.0,
        ).to(tl.float32)
        moment_j += local_weight * local_u * local_u
        moment_d += local_weight * local_u * local_h

    strength_index = (
        _panel_head(panel, heads, chunks) if RAW_LAYOUT else panel
    )
    panel_strength = tl.load(strength + strength_index).to(tl.float32)
    radius = 0.125
    centered_j = moment_j - 1.0 / 128.0
    tanh_j = libdevice.tanh(panel_strength * centered_j / radius)
    tanh_d = libdevice.tanh(panel_strength * moment_d / radius)
    sech_j = 1.0 - tanh_j * tanh_j
    sech_d = 1.0 - tanh_d * tanh_d
    grad_log_offset = (
        _raw_vector_offsets(
            panel, targets, coordinates, length, heads, chunks
        )
        if RAW_LAYOUT
        else (panel * _CHUNK + targets) * _RANK + coordinates
    )
    grad_log = tl.load(
        grad_log_diagonal + grad_log_offset,
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
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
):
    panel = tl.program_id(0)
    target_block = tl.program_id(1)
    source_block = tl.program_id(2)
    targets_vector = target_block * 4 + tl.arange(0, 4)
    sources_vector = source_block * 8 + tl.arange(0, 8)
    targets = targets_vector[:, None, None]
    sources = sources_vector[None, :, None]
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
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
        vector_offset = (
            _raw_vector_offsets(
                panel, sources, coordinates, length, heads, chunks
            )
            if RAW_LAYOUT
            else (panel * _CHUNK + sources) * _RANK + coordinates
        )
        local_u = tl.load(
            u + vector_offset,
            mask=source_active,
            other=0.0,
        ).to(tl.float32)
        local_h = tl.load(
            h + vector_offset,
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
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
):
    panel = tl.program_id(0)
    source = tl.program_id(1)
    coordinates = tl.arange(0, _RANK)
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    source_active = source < count
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
    output = (
        _raw_vector_offsets(
            panel, source, coordinates, length, heads, chunks
        )
        if RAW_LAYOUT
        else (panel * _CHUNK + source) * _RANK + coordinates
    )
    local_u = tl.load(
        u + output,
        mask=source_active,
        other=0.0,
    ).to(tl.float32)
    local_h = tl.load(
        h + output,
        mask=source_active,
        other=0.0,
    ).to(tl.float32)
    previous_u = tl.load(grad_u + output, mask=source_active, other=0.0)
    previous_h = tl.load(grad_h + output, mask=source_active, other=0.0)
    result_u = previous_u + 2.0 * local_u * sum_j + local_h * sum_d
    result_h = previous_h + local_u * sum_d
    if RAW_LAYOUT:
        tl.store(grad_u + output, result_u, mask=source_active)
        tl.store(grad_h + output, result_h, mask=source_active)
    else:
        tl.store(
            grad_u + output,
            tl.where(source_active, result_u, 0.0),
        )
        tl.store(
            grad_h + output,
            tl.where(source_active, result_h, 0.0),
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
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
    HAS_ACTION: tl.constexpr,
):
    """Reverse the scalar affine normalization in strict token order."""

    panel = tl.program_id(0)
    sources = tl.arange(0, _CHUNK)
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
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
        decay_offset = (
            _raw_scalar_offsets(panel, target, length, heads, chunks)
            if RAW_LAYOUT
            else panel * _CHUNK + target
        )
        rho = tl.exp(
            tl.load(log_decay + decay_offset, mask=active, other=0.0).to(
                tl.float32
            )
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
        if RAW_LAYOUT:
            tl.store(
                grad_log_decay + decay_offset,
                grad_rho * rho,
                mask=active,
            )
        else:
            tl.store(
                grad_log_decay + decay_offset,
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
    valid_count: torch.Tensor | None,
    return_saved: bool,
    *,
    raw_layout: bool = False,
) -> RadialCompactOutput | tuple[RadialCompactOutput, RadialCompactSaved]:
    if raw_layout:
        batch, length, heads, _ = u.shape
        chunks = triton.cdiv(length, _HOST_CHUNK)
        panels = batch * heads * chunks
        metadata = boundary_m
    else:
        panels = u.shape[0]
        length, heads, chunks = _HOST_CHUNK, 1, 1
        metadata = valid_count
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
        metadata,
        inverse_mass,
        theta,
        weights,
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
        num_warps=1,
    )

    gram_partial = torch.empty(
        panels,
        4,
        8,
        _HOST_CHUNK,
        _HOST_CHUNK,
        device=u.device,
        dtype=torch.float32,
    )
    boundary_pair_partial = torch.empty(
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
    for route in range(4):
        _radial_pair_statistics_kernel[(panels, 8)](
            u,
            h,
            boundary_j,
            boundary_d,
            metadata,
            gram_partial,
            boundary_pair_partial,
            boundary_norm_partial,
            length,
            heads=heads,
            chunks=chunks,
            RAW_LAYOUT=raw_layout,
            ROUTE=route,
            num_warps=4,
            num_stages=2,
        )
    radial_scale = torch.empty(
        panels, _HOST_CHUNK, 4, device=u.device, dtype=torch.float32
    )
    radial_q2 = torch.empty_like(radial_scale)
    radial_norm = torch.empty_like(radial_scale) if return_saved else radial_q2
    saved_gram = (
        torch.empty(
            panels,
            4,
            _HOST_CHUNK,
            _HOST_CHUNK,
            device=u.device,
            dtype=torch.float32,
        )
        if return_saved
        else gram_partial
    )
    saved_boundary_pair = (
        torch.empty(
            panels,
            4,
            _HOST_CHUNK,
            device=u.device,
            dtype=torch.float32,
        )
        if return_saved
        else boundary_pair_partial
    )
    saved_boundary_norm = (
        torch.empty(panels, 4, device=u.device, dtype=torch.float32)
        if return_saved
        else boundary_norm_partial
    )
    _radial_pair_output_kernel[(panels, 4)](
        theta,
        weights,
        strength,
        metadata,
        gram_partial,
        boundary_pair_partial,
        boundary_norm_partial,
        radial_scale,
        radial_q2,
        radial_norm,
        saved_gram,
        saved_boundary_pair,
        saved_boundary_norm,
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
        SAVE_STATISTICS=return_saved,
        num_warps=8,
        num_stages=2,
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
        metadata,
        diagonal,
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
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
    return output, RadialCompactSaved(
        radial_norm,
        saved_gram,
        saved_boundary_pair,
        saved_boundary_norm,
    )


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

    ``u`` is the directly produced bounded FP16 normalization panel and ``h``
    is the once-quantized raw BF16 operand. Persistent boundary states,
    temporal scalars, reductions, and outputs remain FP32. This
    standalone forward intentionally has no autograd registration; its reverse
    is the transpose of the same statistic graph, not a chain of generic VJPs.
    ``valid_count`` contains one value in ``[0,32]`` per panel; omitting it
    declares every panel full.

    Each radial norm is evaluated by the MESA Gram/Hadamard identity over
    strict coordinate tiles. Off-diagonal tiles use two low-precision Gram
    contractions with FP32 accumulation; diagonal 16-by-16 tiles retain their
    exact strict mask. The reverse applies the corresponding pair-statistic,
    off-diagonal tile, and diagonal-prefix transposes without materializing a
    tokenwise dense matrix or a chain of generic VJPs.
    """

    if u.ndim != 3:
        raise ValueError("u must have shape [panels,32,128]")
    panels = u.shape[0]
    _check_tensor(u, "u", (panels, _HOST_CHUNK, _HOST_RANK), torch.float16)
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


def _radial_compact_forward_bthr_trusted(
    u: torch.Tensor,
    h: torch.Tensor,
    log_decay: torch.Tensor,
    strength: torch.Tensor,
    boundary_m: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    *,
    return_saved: bool,
) -> RadialCompactOutput | tuple[RadialCompactOutput, RadialCompactSaved]:
    """Production raw-stride entry; outputs retain panel-major chunk ownership."""

    with torch.cuda.device(u.device):
        return _radial_compact_forward_launch(
            u,
            h,
            log_decay,
            strength,
            boundary_m,
            boundary_j,
            boundary_d,
            None,
            return_saved,
            raw_layout=True,
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
    valid_count: torch.Tensor | None,
    grad_theta_action: torch.Tensor | None,
    grad_weights_action: torch.Tensor | None,
    accumulated_gradients: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
    *,
    raw_layout: bool = False,
) -> RadialCompactGradients:
    if raw_layout:
        batch, length, heads, _ = u.shape
        chunks = triton.cdiv(length, _HOST_CHUNK)
        panels = batch * heads * chunks
        metadata = boundary_m
    else:
        panels = u.shape[0]
        length, heads, chunks = _HOST_CHUNK, 1, 1
        metadata = valid_count
    grad_norm = torch.empty_like(saved.radial_norm)
    grad_strength_radial = torch.empty(
        panels, device=u.device, dtype=torch.float32
    )
    _radial_scalar_reverse_kernel[(panels,)](
        saved.radial_norm,
        output.radial_scale,
        output.radial_q2,
        strength,
        metadata,
        grad_radial_scale,
        grad_norm,
        grad_strength_radial,
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
        num_warps=4,
    )

    grad_theta_partial = torch.empty(
        panels, 4, _HOST_CHUNK, device=u.device, dtype=torch.float32
    )
    grad_weights_partial = torch.empty(
        panels,
        4,
        _HOST_CHUNK,
        _HOST_CHUNK,
        device=u.device,
        dtype=torch.float32,
    )
    boundary_coefficient = torch.empty_like(grad_theta_partial)
    local_coefficient = torch.empty_like(grad_weights_partial)
    _radial_pair_scalar_reverse_kernel[(panels, 4)](
        output.theta,
        output.weights,
        grad_norm,
        metadata,
        saved.gram,
        saved.boundary_pair,
        saved.boundary_norm,
        grad_theta_partial,
        grad_weights_partial,
        boundary_coefficient,
        local_coefficient,
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
        num_warps=8,
        num_stages=2,
    )
    grad_theta_radial = torch.empty_like(output.theta)
    grad_weights_radial = torch.empty_like(output.weights)
    _reduce_radial_pair_scalar_reverse_kernel[(panels,)](
        grad_theta_partial,
        grad_weights_partial,
        grad_theta_radial,
        grad_weights_radial,
        num_warps=8,
        num_stages=2,
    )

    vector_shape = u.shape
    add_to_grad = accumulated_gradients is not None
    if accumulated_gradients is None:
        grad_u = torch.empty(vector_shape, device=u.device, dtype=torch.float32)
        grad_h = torch.empty_like(grad_u)
        grad_boundary_j = torch.empty_like(boundary_j)
        grad_boundary_d = torch.empty_like(boundary_d)
    else:
        grad_u, grad_h, grad_boundary_j, grad_boundary_d = accumulated_gradients
    _radial_pair_boundary_leaf_transpose_kernel[(panels, 8)](
        u,
        h,
        boundary_j,
        boundary_d,
        boundary_coefficient,
        metadata,
        grad_u,
        grad_h,
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
        ADD_TO_GRAD=add_to_grad,
        num_warps=4,
        num_stages=3,
    )
    _radial_pair_boundary_transpose_kernel[(panels, 8, 8)](
        u,
        h,
        boundary_j,
        boundary_d,
        output.theta,
        grad_norm,
        boundary_coefficient,
        metadata,
        grad_boundary_j,
        grad_boundary_d,
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
        ADD_TO_GRAD=add_to_grad,
        num_warps=4,
        num_stages=2,
    )
    _radial_pair_offdiagonal_transpose_kernel[(panels, 8)](
        u,
        h,
        local_coefficient,
        metadata,
        grad_u,
        grad_h,
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
        num_warps=4,
        num_stages=2,
    )
    _radial_pair_diagonal_transpose_kernel[(panels, _HOST_CHUNK)](
        u,
        h,
        local_coefficient,
        metadata,
        grad_u,
        grad_h,
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
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
        metadata,
        grad_log_diagonal,
        grad_moment_j,
        grad_moment_d,
        grad_strength_diagonal_partial,
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
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
        metadata,
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
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
        metadata,
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
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
        metadata,
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
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
        HAS_ACTION=grad_theta_action is not None,
        num_warps=1,
    )

    grad_strength = torch.empty(
        panels, device=u.device, dtype=torch.float32
    )
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
    _check_tensor(u, "u", (panels, _HOST_CHUNK, _HOST_RANK), torch.float16)
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
    _check_tensor(
        saved.gram,
        "saved.gram",
        (panels, 4, _HOST_CHUNK, _HOST_CHUNK),
        torch.float32,
    )
    _check_tensor(
        saved.boundary_pair,
        "saved.boundary_pair",
        (panels, 4, _HOST_CHUNK),
        torch.float32,
    )
    _check_tensor(
        saved.boundary_norm,
        "saved.boundary_norm",
        (panels, 4),
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
            ("saved.gram", saved.gram),
            ("saved.boundary_pair", saved.boundary_pair),
            ("saved.boundary_norm", saved.boundary_norm),
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
    valid_count: torch.Tensor | None,
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
            raw_layout=u.ndim == 4,
        )

__all__ = [
    "RadialCompactGradients",
    "RadialCompactOutput",
    "RadialCompactSaved",
    "radial_compact_forward",
    "radial_compact_reverse",
]
