from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl


_CHUNK = 32
_RANK = 128
_ROUTES = 3
_BOUNDARY_TILE = 16
_BOUNDARY_K = 32
_TOKEN_TILE = 16
_ROW_BLOCKS = _RANK // _BOUNDARY_TILE


@triton.jit
def _panel_count(panel, length, heads: tl.constexpr, chunks: tl.constexpr):
    return tl.minimum(_DEVICE_CHUNK, length - (panel % chunks) * _DEVICE_CHUNK)


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
    tokens = chunk * _DEVICE_CHUNK + local_tokens
    return (
        (batch * length * heads + tokens * heads + head) * _DEVICE_RANK
        + coordinates
    )

_DEVICE_CHUNK = tl.constexpr(32)
_DEVICE_RANK = tl.constexpr(128)
_DEVICE_ROUTES = tl.constexpr(3)
_DEVICE_ROW_BLOCKS = tl.constexpr(8)
_DEVICE_BOUNDARY_TILE = tl.constexpr(16)
_DEVICE_BOUNDARY_K = tl.constexpr(32)
_DEVICE_TOKEN_TILE = tl.constexpr(16)


class StrictChartGradients(NamedTuple):
    """Direct transpose of the lower/upper strict chart coordinates."""

    grad_radial_scale: torch.Tensor
    grad_theta: torch.Tensor
    grad_weights: torch.Tensor
    grad_u: torch.Tensor
    grad_h: torch.Tensor
    grad_boundary_j: torch.Tensor
    grad_boundary_d: torch.Tensor


@triton.jit
def _split_dot(left, right):
    left_fp32 = left.to(tl.float32)
    high = left_fp32.to(tl.bfloat16)
    low = (left_fp32 - high.to(tl.float32)).to(tl.bfloat16)
    return tl.dot(high, right) + tl.dot(low, right)


@triton.jit
def _boundary_projection_kernel(
    lower_left,
    lower_right,
    upper_left,
    upper_right,
    boundary_j,
    boundary_d,
    valid_count,
    projection_partial,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
    UPPER: tl.constexpr,
):
    """Compute paired J/D row-block projections from one descriptor load."""

    panel = tl.program_id(0)
    row_block = tl.program_id(1)
    token_block = tl.program_id(2)
    rows = (
        row_block * _DEVICE_BOUNDARY_TILE
        + tl.arange(0, _DEVICE_BOUNDARY_TILE)
    )
    targets = (
        token_block * _DEVICE_TOKEN_TILE
        + tl.arange(0, _DEVICE_TOKEN_TILE)
    )
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    valid_target = targets < count
    panel_matrix = panel * _DEVICE_RANK * _DEVICE_RANK
    projection_j = tl.zeros((_DEVICE_TOKEN_TILE,), tl.float32)
    projection_d = tl.zeros_like(projection_j)

    for route in tl.static_range(0, _DEVICE_ROUTES):
        action_j = tl.zeros(
            (_DEVICE_BOUNDARY_TILE, _DEVICE_TOKEN_TILE), tl.float32
        )
        action_d = tl.zeros_like(action_j)
        for start in tl.static_range(
            0, _DEVICE_RANK, _DEVICE_BOUNDARY_K
        ):
            columns = start + tl.arange(0, _DEVICE_BOUNDARY_K)
            strict = (
                columns[None, :] > rows[:, None]
                if UPPER
                else rows[:, None] > columns[None, :]
            )
            boundary_offset = (
                panel_matrix
                + rows[:, None] * _DEVICE_RANK
                + columns[None, :]
            )
            local_boundary_j = tl.load(
                boundary_j + boundary_offset, mask=strict, other=0.0
            ).to(tl.float32)
            local_boundary_d = tl.load(
                boundary_d + boundary_offset, mask=strict, other=0.0
            ).to(tl.float32)
            descriptor_pointer = (
                (upper_right if UPPER else lower_right)
                + (
                    (panel * _DEVICE_CHUNK + targets[None, :])
                    * _DEVICE_ROUTES
                    + route
                )
                * _DEVICE_RANK
                + columns[:, None]
            )
            descriptor = tl.load(
                descriptor_pointer,
                mask=valid_target[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            action_j += _split_dot(local_boundary_j, descriptor)
            action_d += _split_dot(local_boundary_d, descriptor)

        left_pointer = (
            (upper_left if UPPER else lower_left)
            + (
                (panel * _DEVICE_CHUNK + targets[None, :])
                * _DEVICE_ROUTES
                + route
            )
            * _DEVICE_RANK
            + rows[:, None]
        )
        left = tl.load(
            left_pointer, mask=valid_target[None, :], other=0.0
        ).to(tl.float32)
        projection_j += tl.sum(left * action_j, axis=0)
        projection_d += tl.sum(left * action_d, axis=0)

    component: tl.constexpr = 2 if UPPER else 0
    output = (
        projection_partial
        + (
            (panel * 4 + component) * _DEVICE_ROW_BLOCKS + row_block
        )
        * _DEVICE_CHUNK
        + targets
    )
    tl.store(output, projection_j, mask=valid_target)
    tl.store(
        output + _DEVICE_ROW_BLOCKS * _DEVICE_CHUNK,
        projection_d,
        mask=valid_target,
    )


@triton.jit
def _boundary_gradient_kernel(
    lower_left,
    lower_right,
    upper_left,
    upper_right,
    theta,
    radial_scale,
    valid_count,
    grad_boundary_j,
    grad_boundary_d,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
    UPPER: tl.constexpr,
):
    """Contract paired J/D boundary gradients from one descriptor load."""

    panel = tl.program_id(0)
    row_block = tl.program_id(1)
    column_block = tl.program_id(2)
    rows = (
        row_block * _DEVICE_BOUNDARY_TILE
        + tl.arange(0, _DEVICE_BOUNDARY_TILE)
    )
    columns = (
        column_block * _DEVICE_BOUNDARY_TILE
        + tl.arange(0, _DEVICE_BOUNDARY_TILE)
    )
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    accumulator_j = tl.zeros(
        (_DEVICE_BOUNDARY_TILE, _DEVICE_BOUNDARY_TILE), tl.float32
    )
    accumulator_d = tl.zeros_like(accumulator_j)
    component: tl.constexpr = 2 if UPPER else 0

    for start in tl.static_range(
        0, _DEVICE_CHUNK * _DEVICE_ROUTES, _DEVICE_BOUNDARY_K
    ):
        basis = start + tl.arange(0, _DEVICE_BOUNDARY_K)
        target = basis // _DEVICE_ROUTES
        route = basis % _DEVICE_ROUTES
        active = target < count
        left_pointer = (
            (upper_left if UPPER else lower_left)
            + (
                (panel * _DEVICE_CHUNK + target[None, :])
                * _DEVICE_ROUTES
                + route[None, :]
            )
            * _DEVICE_RANK
            + rows[:, None]
        )
        right_pointer = (
            (upper_right if UPPER else lower_right)
            + (
                (panel * _DEVICE_CHUNK + target[:, None])
                * _DEVICE_ROUTES
                + route[:, None]
            )
            * _DEVICE_RANK
            + columns[None, :]
        )
        left = tl.load(
            left_pointer, mask=active[None, :], other=0.0
        ).to(tl.float32)
        right = tl.load(
            right_pointer, mask=active[:, None], other=0.0
        ).to(tl.bfloat16)
        local_theta = tl.load(
            theta + panel * _DEVICE_CHUNK + target,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        scale_j = tl.load(
            radial_scale
            + (panel * _DEVICE_CHUNK + target) * 4
            + component,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        scale_d = tl.load(
            radial_scale
            + (panel * _DEVICE_CHUNK + target) * 4
            + component
            + 1,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        weighted_j = left * (local_theta * scale_j)[None, :]
        weighted_d = left * (local_theta * scale_d)[None, :]
        accumulator_j += _split_dot(weighted_j, right)
        accumulator_d += _split_dot(weighted_d, right)

    strict = (
        columns[None, :] > rows[:, None]
        if UPPER
        else rows[:, None] > columns[None, :]
    )
    output_j = grad_boundary_j + panel * _DEVICE_RANK * _DEVICE_RANK
    output_d = grad_boundary_d + panel * _DEVICE_RANK * _DEVICE_RANK
    tl.store(
        output_j
        + rows[:, None] * _DEVICE_RANK
        + columns[None, :],
        accumulator_j,
        mask=strict,
    )
    tl.store(
        output_d
        + rows[:, None] * _DEVICE_RANK
        + columns[None, :],
        accumulator_d,
        mask=strict,
    )


@triton.jit
def _strict_cross_correlation_kernel(
    lower_left,
    lower_right,
    upper_left,
    upper_right,
    u,
    h,
    valid_count,
    correlation,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
    UPPER: tl.constexpr,
):
    """Build strict prefix/suffix correlations once per panel and route."""

    panel = tl.program_id(0)
    route = tl.program_id(1)
    target_vector = tl.arange(0, _DEVICE_CHUNK)
    source_vector = tl.arange(0, _DEVICE_CHUNK)
    target = target_vector[:, None]
    source = source_vector[None, :]
    local = tl.arange(0, _DEVICE_BOUNDARY_TILE)[None, :]
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    valid_target = target < count
    valid_source = source < count
    valid_pair = valid_target & valid_source
    descriptor = upper_right if UPPER else lower_right
    right_u = tl.zeros((_DEVICE_CHUNK, _DEVICE_CHUNK), tl.float32)
    right_h = tl.zeros_like(right_u)

    for step in tl.static_range(0, _DEVICE_ROW_BLOCKS):
        block = _DEVICE_ROW_BLOCKS - 1 - step if UPPER else step
        output_base = (
            (
                ((panel * _DEVICE_ROUTES + route) * 3)
                * _DEVICE_ROW_BLOCKS
                + block
            )
            * _DEVICE_CHUNK
            + target
        ) * _DEVICE_CHUNK + source
        tl.store(correlation + output_base, right_u, mask=valid_pair)
        tl.store(
            correlation
            + output_base
            + _DEVICE_ROW_BLOCKS * _DEVICE_CHUNK * _DEVICE_CHUNK,
            right_h,
            mask=valid_pair,
        )
        coordinate = block * _DEVICE_BOUNDARY_TILE + local
        descriptor_base = (
            (panel * _DEVICE_CHUNK + target) * _DEVICE_ROUTES + route
        ) * _DEVICE_RANK
        right = tl.load(
            descriptor + descriptor_base + coordinate,
            mask=valid_target,
            other=0.0,
        ).to(tl.bfloat16)
        vector_offsets = (
            _raw_vector_offsets(
                panel,
                source_vector[:, None],
                coordinate,
                length,
                heads,
                chunks,
            )
            if RAW_LAYOUT
            else (panel * _DEVICE_CHUNK + source_vector[:, None])
            * _DEVICE_RANK
            + coordinate
        )
        block_u = tl.load(
            u + vector_offsets,
            mask=(source_vector < count)[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        block_h = tl.load(
            h + vector_offsets,
            mask=(source_vector < count)[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        right_u += tl.dot(right, tl.trans(block_u))
        right_h += tl.dot(right, tl.trans(block_h))

    descriptor = upper_left if UPPER else lower_left
    left_u = tl.zeros((_DEVICE_CHUNK, _DEVICE_CHUNK), tl.float32)
    for step in tl.static_range(0, _DEVICE_ROW_BLOCKS):
        block = step if UPPER else _DEVICE_ROW_BLOCKS - 1 - step
        output_base = (
            (
                ((panel * _DEVICE_ROUTES + route) * 3 + 2)
                * _DEVICE_ROW_BLOCKS
                + block
            )
            * _DEVICE_CHUNK
            + target
        ) * _DEVICE_CHUNK + source
        tl.store(correlation + output_base, left_u, mask=valid_pair)
        coordinate = block * _DEVICE_BOUNDARY_TILE + local
        descriptor_base = (
            (panel * _DEVICE_CHUNK + target) * _DEVICE_ROUTES + route
        ) * _DEVICE_RANK
        left = tl.load(
            descriptor + descriptor_base + coordinate,
            mask=valid_target,
            other=0.0,
        ).to(tl.bfloat16)
        vector_offsets = (
            _raw_vector_offsets(
                panel,
                source_vector[:, None],
                coordinate,
                length,
                heads,
                chunks,
            )
            if RAW_LAYOUT
            else (panel * _DEVICE_CHUNK + source_vector[:, None])
            * _DEVICE_RANK
            + coordinate
        )
        block_u = tl.load(
            u + vector_offsets,
            mask=(source_vector < count)[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        left_u += tl.dot(left, tl.trans(block_u))


@triton.jit
def _strict_cross_action_kernel(
    lower_left,
    lower_right,
    upper_left,
    upper_right,
    u,
    h,
    weights,
    radial_scale,
    valid_count,
    correlation,
    pair_partial,
    grad_u,
    grad_h,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
    UPPER: tl.constexpr,
):
    """Apply off-diagonal blocks from pre-aggregated strict correlations."""

    panel = tl.program_id(0)
    coordinate_block = tl.program_id(1)
    target = tl.arange(0, _DEVICE_CHUNK)
    source = tl.arange(0, _DEVICE_CHUNK)
    local = tl.arange(0, _DEVICE_BOUNDARY_TILE)
    coordinate = coordinate_block * _DEVICE_BOUNDARY_TILE + local
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    valid_target = target < count
    valid_source = source < count
    vector_offsets = (
        _raw_vector_offsets(
            panel,
            source[:, None],
            coordinate[None, :],
            length,
            heads,
            chunks,
        )
        if RAW_LAYOUT
        else (panel * _DEVICE_CHUNK + source[:, None]) * _DEVICE_RANK
        + coordinate[None, :]
    )
    current_u = tl.load(
        u + vector_offsets,
        mask=valid_source[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    output_u = tl.zeros(
        (_DEVICE_CHUNK, _DEVICE_BOUNDARY_TILE), tl.float32
    )
    output_h = tl.zeros_like(output_u)
    pair_j = tl.zeros((_DEVICE_CHUNK, _DEVICE_CHUNK), tl.float32)
    pair_d = tl.zeros_like(pair_j)
    component: tl.constexpr = 2 if UPPER else 0

    pair_mask = (
        valid_target[:, None]
        & valid_source[None, :]
        & (source[None, :] <= target[:, None])
    )
    temporal = tl.load(
        weights
        + (panel * _DEVICE_CHUNK + target[:, None])
        * _DEVICE_CHUNK
        + source[None, :],
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    scale_j = tl.load(
        radial_scale
        + (panel * _DEVICE_CHUNK + target) * 4
        + component,
        mask=valid_target,
        other=0.0,
    ).to(tl.float32)
    scale_d = tl.load(
        radial_scale
        + (panel * _DEVICE_CHUNK + target) * 4
        + component
        + 1,
        mask=valid_target,
        other=0.0,
    ).to(tl.float32)

    for route in tl.static_range(0, _DEVICE_ROUTES):
        correlation_base = (
            (
                ((panel * _DEVICE_ROUTES + route) * 3)
                * _DEVICE_ROW_BLOCKS
                + coordinate_block
            )
            * _DEVICE_CHUNK
            + target[:, None]
        ) * _DEVICE_CHUNK + source[None, :]
        corr_right_u = tl.load(
            correlation + correlation_base,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        corr_right_h = tl.load(
            correlation
            + correlation_base
            + _DEVICE_ROW_BLOCKS * _DEVICE_CHUNK * _DEVICE_CHUNK,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        corr_left_u = tl.load(
            correlation
            + correlation_base
            + 2 * _DEVICE_ROW_BLOCKS * _DEVICE_CHUNK * _DEVICE_CHUNK,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        descriptor_base = (
            (panel * _DEVICE_CHUNK + target[:, None])
            * _DEVICE_ROUTES
            + route
        ) * _DEVICE_RANK
        current_left = tl.load(
            (upper_left if UPPER else lower_left)
            + descriptor_base
            + coordinate[None, :],
            mask=valid_target[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        current_right = tl.load(
            (upper_right if UPPER else lower_right)
            + descriptor_base
            + coordinate[None, :],
            mask=valid_target[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        current_left_u = tl.dot(current_left, tl.trans(current_u))

        action_mix = temporal * (
            scale_j[:, None] * corr_right_u
            + scale_d[:, None] * corr_right_h
        )
        output_u += _split_dot(tl.trans(action_mix), current_left)
        transpose_j_mix = temporal * scale_j[:, None] * corr_left_u
        transpose_d_mix = temporal * scale_d[:, None] * corr_left_u
        output_u += _split_dot(
            tl.trans(transpose_j_mix), current_right
        )
        output_h += _split_dot(
            tl.trans(transpose_d_mix), current_right
        )
        pair_j += current_left_u * corr_right_u
        pair_d += current_left_u * corr_right_h

    if UPPER:
        output_u += tl.load(
            grad_u + vector_offsets,
            mask=valid_source[:, None],
            other=0.0,
        ).to(tl.float32)
        output_h += tl.load(
            grad_h + vector_offsets,
            mask=valid_source[:, None],
            other=0.0,
        ).to(tl.float32)
    tl.store(
        grad_u + vector_offsets,
        output_u,
        mask=valid_source[:, None],
    )
    tl.store(
        grad_h + vector_offsets,
        output_h,
        mask=valid_source[:, None],
    )
    tl.store(
        pair_partial
        + (
            (
                (panel * 4 + component) * _DEVICE_ROW_BLOCKS
                + coordinate_block
            )
            * _DEVICE_CHUNK
            + target[:, None]
        )
        * _DEVICE_CHUNK
        + source[None, :],
        tl.where(pair_mask, pair_j, 0.0),
    )
    tl.store(
        pair_partial
        + (
            (
                (panel * 4 + component + 1) * _DEVICE_ROW_BLOCKS
                + coordinate_block
            )
            * _DEVICE_CHUNK
            + target[:, None]
        )
        * _DEVICE_CHUNK
        + source[None, :],
        tl.where(pair_mask, pair_d, 0.0),
    )


@triton.jit
def _strict_diagonal_block_kernel(
    lower_left,
    lower_right,
    upper_left,
    upper_right,
    u,
    h,
    weights,
    radial_scale,
    valid_count,
    pair_partial,
    grad_u,
    grad_h,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
):
    """Add the only coordinate-dependent part: strict 16x16 diagonals."""

    panel = tl.program_id(0)
    coordinate_block = tl.program_id(1)
    source_vector = tl.arange(0, _DEVICE_CHUNK)
    source = source_vector[:, None]
    local = tl.arange(0, _DEVICE_BOUNDARY_TILE)[None, :]
    coordinate = coordinate_block * _DEVICE_BOUNDARY_TILE + local
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    valid_source_vector = source_vector < count
    valid_source = valid_source_vector[:, None]
    vector_offsets = (
        _raw_vector_offsets(
            panel, source, coordinate, length, heads, chunks
        )
        if RAW_LAYOUT
        else (panel * _DEVICE_CHUNK + source) * _DEVICE_RANK + coordinate
    )
    source_u = tl.load(
        u + vector_offsets,
        mask=valid_source,
        other=0.0,
    ).to(tl.float32)
    source_h = tl.load(
        h + vector_offsets,
        mask=valid_source,
        other=0.0,
    ).to(tl.float32)
    output_u = tl.load(
        grad_u + vector_offsets,
        mask=valid_source,
        other=0.0,
    ).to(tl.float32)
    output_h = tl.load(
        grad_h + vector_offsets,
        mask=valid_source,
        other=0.0,
    ).to(tl.float32)

    for target in tl.static_range(0, _DEVICE_CHUNK):
        active_pair = valid_source & (source <= target) & (target < count)
        temporal = tl.load(
            weights
            + (panel * _DEVICE_CHUNK + target) * _DEVICE_CHUNK
            + source,
            mask=active_pair,
            other=0.0,
        ).to(tl.float32)
        scale_lower_j = tl.load(
            radial_scale + (panel * _DEVICE_CHUNK + target) * 4,
            mask=target < count,
            other=0.0,
        ).to(tl.float32)
        scale_lower_d = tl.load(
            radial_scale + (panel * _DEVICE_CHUNK + target) * 4 + 1,
            mask=target < count,
            other=0.0,
        ).to(tl.float32)
        scale_upper_j = tl.load(
            radial_scale + (panel * _DEVICE_CHUNK + target) * 4 + 2,
            mask=target < count,
            other=0.0,
        ).to(tl.float32)
        scale_upper_d = tl.load(
            radial_scale + (panel * _DEVICE_CHUNK + target) * 4 + 3,
            mask=target < count,
            other=0.0,
        ).to(tl.float32)
        lower_j = tl.zeros((_DEVICE_CHUNK,), tl.float32)
        lower_d = tl.zeros((_DEVICE_CHUNK,), tl.float32)
        upper_j = tl.zeros((_DEVICE_CHUNK,), tl.float32)
        upper_d = tl.zeros((_DEVICE_CHUNK,), tl.float32)
        for route in tl.static_range(0, _DEVICE_ROUTES):
            descriptor_base = (
                (panel * _DEVICE_CHUNK + target) * _DEVICE_ROUTES
                + route
            ) * _DEVICE_RANK
            lower_l = tl.load(
                lower_left + descriptor_base + coordinate,
                mask=active_pair,
                other=0.0,
            ).to(tl.float32)
            lower_r = tl.load(
                lower_right + descriptor_base + coordinate,
                mask=active_pair,
                other=0.0,
            ).to(tl.float32)
            upper_l = tl.load(
                upper_left + descriptor_base + coordinate,
                mask=active_pair,
                other=0.0,
            ).to(tl.float32)
            upper_r = tl.load(
                upper_right + descriptor_base + coordinate,
                mask=active_pair,
                other=0.0,
            ).to(tl.float32)

            lower_ru = lower_r * source_u
            lower_rh = lower_r * source_h
            lower_lu = lower_l * source_u
            upper_ru = upper_r * source_u
            upper_rh = upper_r * source_h
            upper_lu = upper_l * source_u
            lower_prefix_ru = tl.cumsum(lower_ru, axis=1) - lower_ru
            lower_prefix_rh = tl.cumsum(lower_rh, axis=1) - lower_rh
            lower_prefix_lu = tl.cumsum(lower_lu, axis=1) - lower_lu
            upper_prefix_ru = tl.cumsum(upper_ru, axis=1) - upper_ru
            upper_prefix_rh = tl.cumsum(upper_rh, axis=1) - upper_rh
            upper_prefix_lu = tl.cumsum(upper_lu, axis=1) - upper_lu
            lower_action_u = lower_l * lower_prefix_ru
            lower_action_h = lower_l * lower_prefix_rh
            lower_transpose_u = lower_r * (
                tl.sum(lower_lu, axis=1)[:, None]
                - lower_lu
                - lower_prefix_lu
            )
            upper_action_u = upper_l * (
                tl.sum(upper_ru, axis=1)[:, None]
                - upper_ru
                - upper_prefix_ru
            )
            upper_action_h = upper_l * (
                tl.sum(upper_rh, axis=1)[:, None]
                - upper_rh
                - upper_prefix_rh
            )
            upper_transpose_u = upper_r * upper_prefix_lu

            output_u += temporal * (
                scale_lower_j * (lower_action_u + lower_transpose_u)
                + scale_lower_d * lower_action_h
                + scale_upper_j * (upper_action_u + upper_transpose_u)
                + scale_upper_d * upper_action_h
            )
            output_h += temporal * (
                scale_lower_d * lower_transpose_u
                + scale_upper_d * upper_transpose_u
            )
            lower_j += tl.sum(source_u * lower_action_u, axis=1)
            lower_d += tl.sum(source_u * lower_action_h, axis=1)
            upper_j += tl.sum(source_u * upper_action_u, axis=1)
            upper_d += tl.sum(source_u * upper_action_h, axis=1)

        for component in tl.static_range(0, 4):
            diagonal_pair = (
                lower_j
                if component == 0
                else lower_d
                if component == 1
                else upper_j
                if component == 2
                else upper_d
            )
            pointer = (
                pair_partial
                + (
                    (
                        (panel * 4 + component) * _DEVICE_ROW_BLOCKS
                        + coordinate_block
                    )
                    * _DEVICE_CHUNK
                    + target
                )
                * _DEVICE_CHUNK
                + source_vector
            )
            previous = tl.load(
                pointer,
                mask=valid_source_vector
                & (source_vector <= target)
                & (target < count),
                other=0.0,
            ).to(tl.float32)
            tl.store(
                pointer,
                previous + diagonal_pair,
                mask=valid_source_vector
                & (source_vector <= target)
                & (target < count),
            )

    tl.store(
        grad_u + vector_offsets,
        output_u,
        mask=valid_source,
    )
    tl.store(
        grad_h + vector_offsets,
        output_h,
        mask=valid_source,
    )


@triton.jit
def _strict_block_scalar_reduce_kernel(
    theta,
    weights,
    radial_scale,
    valid_count,
    boundary_projection_partial,
    pair_partial,
    grad_radial_scale,
    grad_theta,
    grad_weights,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    RAW_LAYOUT: tl.constexpr,
):
    panel = tl.program_id(0)
    target = tl.arange(0, _DEVICE_CHUNK)
    source = tl.arange(0, _DEVICE_CHUNK)
    row_block = tl.arange(0, _DEVICE_ROW_BLOCKS)
    count = (
        _panel_count(panel, length, heads, chunks)
        if RAW_LAYOUT
        else tl.load(valid_count + panel)
    )
    valid_target = target < count
    pair_mask = (
        valid_target[:, None]
        & (source[None, :] < count)
        & (source[None, :] <= target[:, None])
    )
    local_theta = tl.load(
        theta + panel * _DEVICE_CHUNK + target,
        mask=valid_target,
        other=0.0,
    ).to(tl.float32)
    temporal = tl.load(
        weights
        + (panel * _DEVICE_CHUNK + target[:, None]) * _DEVICE_CHUNK
        + source[None, :],
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    weight_gradient = tl.zeros(
        (_DEVICE_CHUNK, _DEVICE_CHUNK), tl.float32
    )
    theta_gradient = tl.zeros((_DEVICE_CHUNK,), tl.float32)

    for component in tl.static_range(0, 4):
        boundary_projection = tl.sum(
            tl.load(
                boundary_projection_partial
                + (
                    (panel * 4 + component) * _DEVICE_ROW_BLOCKS
                    + row_block[:, None]
                )
                * _DEVICE_CHUNK
                + target[None, :],
                mask=valid_target[None, :],
                other=0.0,
            ),
            axis=0,
        )
        pair = tl.zeros((_DEVICE_CHUNK, _DEVICE_CHUNK), tl.float32)
        for block in tl.static_range(0, _DEVICE_ROW_BLOCKS):
            pair += tl.load(
                pair_partial
                + (
                    (
                        (panel * 4 + component) * _DEVICE_ROW_BLOCKS
                        + block
                    )
                    * _DEVICE_CHUNK
                    + target[:, None]
                )
                * _DEVICE_CHUNK
                + source[None, :],
                mask=pair_mask,
                other=0.0,
            ).to(tl.float32)
        scale = tl.load(
            radial_scale
            + (panel * _DEVICE_CHUNK + target) * 4
            + component,
            mask=valid_target,
            other=0.0,
        ).to(tl.float32)
        scale_gradient = (
            local_theta * boundary_projection
            + tl.sum(temporal * pair, axis=1)
        )
        tl.store(
            grad_radial_scale
            + (panel * _DEVICE_CHUNK + target) * 4
            + component,
            tl.where(valid_target, scale_gradient, 0.0),
        )
        theta_gradient += scale * boundary_projection
        weight_gradient += scale[:, None] * pair

    tl.store(
        grad_theta + panel * _DEVICE_CHUNK + target,
        tl.where(valid_target, theta_gradient, 0.0),
    )
    tl.store(
        grad_weights
        + (panel * _DEVICE_CHUNK + target[:, None]) * _DEVICE_CHUNK
        + source[None, :],
        tl.where(pair_mask, weight_gradient, 0.0),
    )


def _check_tensor(
    tensor: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.device != device or device.type != "cuda":
        raise ValueError(f"{name} must be on the shared CUDA device")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_strict_chart_direct_transpose(
    lower_left: torch.Tensor,
    lower_right: torch.Tensor,
    upper_left: torch.Tensor,
    upper_right: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    theta: torch.Tensor,
    weights: torch.Tensor,
    radial_scale: torch.Tensor,
    valid_count: torch.Tensor,
) -> None:
    if not isinstance(u, torch.Tensor) or u.ndim != 3:
        raise ValueError("u must have shape [P,32,128]")
    panels = u.shape[0]
    if panels < 1 or u.shape[1:] != (_CHUNK, _RANK):
        raise ValueError("u must have shape [P,32,128] with P positive")
    device = u.device
    descriptor_shape = (panels, _CHUNK, _ROUTES, _RANK)
    for name, tensor in (
        ("lower_left", lower_left),
        ("lower_right", lower_right),
        ("upper_left", upper_left),
        ("upper_right", upper_right),
    ):
        _check_tensor(tensor, name, descriptor_shape, torch.bfloat16, device)
    _check_tensor(u, "u", (panels, _CHUNK, _RANK), torch.bfloat16, device)
    _check_tensor(h, "h", (panels, _CHUNK, _RANK), torch.bfloat16, device)
    _check_tensor(
        boundary_j,
        "boundary_j",
        (panels, _RANK, _RANK),
        torch.float32,
        device,
    )
    _check_tensor(
        boundary_d,
        "boundary_d",
        (panels, _RANK, _RANK),
        torch.float32,
        device,
    )
    _check_tensor(theta, "theta", (panels, _CHUNK), torch.float32, device)
    _check_tensor(
        weights,
        "weights",
        (panels, _CHUNK, _CHUNK),
        torch.float32,
        device,
    )
    _check_tensor(
        radial_scale,
        "radial_scale",
        (panels, _CHUNK, 4),
        torch.float32,
        device,
    )
    _check_tensor(
        valid_count, "valid_count", (panels,), torch.int32, device
    )
    if bool(torch.any((valid_count < 1) | (valid_count > _CHUNK)).item()):
        raise ValueError("valid_count entries must be in [1,32]")


def _strict_chart_direct_transpose_trusted(
    lower_left: torch.Tensor,
    lower_right: torch.Tensor,
    upper_left: torch.Tensor,
    upper_right: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    theta: torch.Tensor,
    weights: torch.Tensor,
    radial_scale: torch.Tensor,
    valid_count: torch.Tensor | None,
) -> StrictChartGradients:
    """Execute the transpose action for inputs owned by the native pipeline."""

    raw_layout = u.ndim == 4
    if raw_layout:
        batch, length, heads, _ = u.shape
        chunks = triton.cdiv(length, _CHUNK)
        panels = batch * heads * chunks
        metadata = boundary_j
    else:
        panels = u.shape[0]
        length, heads, chunks = _CHUNK, 1, 1
        metadata = valid_count
    device = u.device
    projection_partial = torch.empty(
        panels,
        4,
        _ROW_BLOCKS,
        _CHUNK,
        device=device,
        dtype=torch.float32,
    )
    projection_grid = (panels, _ROW_BLOCKS, _CHUNK // _TOKEN_TILE)
    for upper in (False, True):
        _boundary_projection_kernel[projection_grid](
            lower_left,
            lower_right,
            upper_left,
            upper_right,
            boundary_j,
            boundary_d,
            metadata,
            projection_partial,
            length,
            heads=heads,
            chunks=chunks,
            RAW_LAYOUT=raw_layout,
            UPPER=upper,
            num_warps=2,
            num_stages=3,
        )

    pair_partial = torch.empty(
        panels,
        4,
        _ROW_BLOCKS,
        _CHUNK,
        _CHUNK,
        device=device,
        dtype=torch.float32,
    )
    grad_u = torch.zeros(u.shape, device=device, dtype=torch.float32)
    grad_h = torch.zeros_like(grad_u)
    correlation = torch.empty(
        panels,
        _ROUTES,
        3,
        _ROW_BLOCKS,
        _CHUNK,
        _CHUNK,
        device=device,
        dtype=torch.float32,
    )
    block_grid = (panels, _ROW_BLOCKS)
    correlation_grid = (panels, _ROUTES)
    for upper in (False, True):
        _strict_cross_correlation_kernel[correlation_grid](
            lower_left,
            lower_right,
            upper_left,
            upper_right,
            u,
            h,
            metadata,
            correlation,
            length,
            heads=heads,
            chunks=chunks,
            RAW_LAYOUT=raw_layout,
            UPPER=upper,
            num_warps=4,
            num_stages=3,
        )
        _strict_cross_action_kernel[block_grid](
            lower_left,
            lower_right,
            upper_left,
            upper_right,
            u,
            h,
            weights,
            radial_scale,
            metadata,
            correlation,
            pair_partial,
            grad_u,
            grad_h,
            length,
            heads=heads,
            chunks=chunks,
            RAW_LAYOUT=raw_layout,
            UPPER=upper,
            num_warps=4,
            num_stages=3,
        )
    _strict_diagonal_block_kernel[block_grid](
        lower_left,
        lower_right,
        upper_left,
        upper_right,
        u,
        h,
        weights,
        radial_scale,
        metadata,
        pair_partial,
        grad_u,
        grad_h,
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
        num_warps=4,
    )
    grad_radial_scale = torch.empty(
        panels, _CHUNK, 4, device=device, dtype=torch.float32
    )
    grad_theta = torch.empty_like(theta)
    grad_weights = torch.empty_like(weights)
    _strict_block_scalar_reduce_kernel[(panels,)](
        theta,
        weights,
        radial_scale,
        metadata,
        projection_partial,
        pair_partial,
        grad_radial_scale,
        grad_theta,
        grad_weights,
        length,
        heads=heads,
        chunks=chunks,
        RAW_LAYOUT=raw_layout,
        num_warps=8,
    )

    grad_boundary_j = torch.zeros_like(boundary_j)
    grad_boundary_d = torch.zeros_like(boundary_d)
    boundary_grid = (panels, _ROW_BLOCKS, _ROW_BLOCKS)
    for upper in (False, True):
        _boundary_gradient_kernel[boundary_grid](
            lower_left,
            lower_right,
            upper_left,
            upper_right,
            theta,
            radial_scale,
            metadata,
            grad_boundary_j,
            grad_boundary_d,
            length,
            heads=heads,
            chunks=chunks,
            RAW_LAYOUT=raw_layout,
            UPPER=upper,
            num_warps=2,
            num_stages=3,
        )

    return StrictChartGradients(
        grad_radial_scale,
        grad_theta,
        grad_weights,
        grad_u,
        grad_h,
        grad_boundary_j,
        grad_boundary_d,
    )


def strict_chart_direct_transpose(
    lower_left: torch.Tensor,
    lower_right: torch.Tensor,
    upper_left: torch.Tensor,
    upper_right: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    theta: torch.Tensor,
    weights: torch.Tensor,
    radial_scale: torch.Tensor,
    valid_count: torch.Tensor,
) -> StrictChartGradients:
    """Transpose the direct strict-chart action at the quantized point.

    Each descriptor pair represents ``G_t = sum_a left[t,a] right[t,a]^T``.
    Only the strict lower or upper entries are consumed. The implementation
    uses broad BF16 Tensor Core contractions for boundary terms and triangular
    prefix/suffix identities for the ``O(C^2 r)`` local terms. It never forms
    a tokenwise ``[P,C,r,r]`` tensor.

    This standalone boundary validates descriptors and metadata. The native
    pipeline calls the same execution core with its internally produced,
    structurally valid metadata and therefore performs no device-to-host check.
    """

    _validate_strict_chart_direct_transpose(
        lower_left,
        lower_right,
        upper_left,
        upper_right,
        u,
        h,
        boundary_j,
        boundary_d,
        theta,
        weights,
        radial_scale,
        valid_count,
    )
    return _strict_chart_direct_transpose_trusted(
        lower_left,
        lower_right,
        upper_left,
        upper_right,
        u,
        h,
        boundary_j,
        boundary_d,
        theta,
        weights,
        radial_scale,
        valid_count,
    )


__all__ = ["StrictChartGradients", "strict_chart_direct_transpose"]
