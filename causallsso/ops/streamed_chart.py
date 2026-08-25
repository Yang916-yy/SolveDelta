from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl


_CHUNK = 32
_RANK = 128
_TILE = 16


class StreamedChartGradients(NamedTuple):
    radial_scale: torch.Tensor
    theta: torch.Tensor
    weights: torch.Tensor
    u: torch.Tensor
    h: torch.Tensor


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
    tokens = chunk * 32 + local_tokens
    return ((batch * length + tokens) * heads + head) * 128 + coordinates


@triton.jit
def _raw_dual_offsets(
    panel,
    local_tokens,
    route,
    coordinates,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
):
    return (
        _raw_vector_offsets(
            panel, local_tokens, 0, length, heads, chunks
        )
        * 2
        + route * 128
        + coordinates
    )


@triton.jit
def _left_descriptor(
    key,
    erase,
    query,
    lower_dual,
    action_primal,
    panel,
    target,
    coordinate,
    route,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    UPPER: tl.constexpr,
):
    vector = _raw_vector_offsets(
        panel, target, coordinate, length, heads, chunks
    )
    primal = tl.load(action_primal + vector).to(tl.float32)
    if UPPER:
        dual = tl.load(
            lower_dual
            + _raw_dual_offsets(
                panel,
                target,
                tl.maximum(route - 1, 0),
                coordinate,
                length,
                heads,
                chunks,
            ),
            mask=route > 0,
            other=0.0,
        ).to(tl.float32)
        return tl.where(route == 0, -primal, dual)
    local_key = tl.load(key + vector).to(tl.float32)
    local_erase = tl.load(erase + vector).to(tl.float32)
    local_query = tl.load(query + vector).to(tl.float32)
    return tl.where(
        route == 0,
        -primal,
        tl.where(route == 1, local_key * local_erase, local_query),
    )


@triton.jit
def _right_descriptor(
    d,
    lower_primal,
    original_dual,
    action_dual,
    panel,
    target,
    coordinate,
    route,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    UPPER: tl.constexpr,
):
    vector = _raw_vector_offsets(
        panel, target, coordinate, length, heads, chunks
    )
    if UPPER:
        primal = tl.load(d + vector).to(tl.float32)
        dual = tl.load(
            original_dual
            + _raw_dual_offsets(
                panel,
                target,
                tl.maximum(route - 1, 0),
                coordinate,
                length,
                heads,
                chunks,
            ),
            mask=route > 0,
            other=0.0,
        ).to(tl.float32)
        return tl.where(route == 0, primal, dual)
    primal = tl.load(lower_primal + vector).to(tl.float32)
    dual = tl.load(
        action_dual
        + _raw_dual_offsets(
            panel,
            target,
            tl.maximum(route - 1, 0),
            coordinate,
            length,
            heads,
            chunks,
        ),
        mask=route > 0,
        other=0.0,
    ).to(tl.float32)
    return tl.where(route == 0, primal, dual)


@triton.jit
def _mma(left, right):
    return tl.dot(left.to(tl.bfloat16), right.to(tl.bfloat16))


@triton.jit
def _streamed_local_kernel(
    u,
    h,
    key,
    erase,
    query,
    d,
    lower_primal,
    lower_dual,
    original_dual,
    action_primal,
    action_dual,
    weights,
    radial_scale,
    grad_radial_scale,
    grad_weights,
    grad_u,
    grad_h,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    UPPER: tl.constexpr,
):
    panel = tl.program_id(0)
    coordinate_block = tl.program_id(1)
    route = tl.program_id(2)
    target = tl.arange(0, 32)
    source = tl.arange(0, 32)
    local = tl.arange(0, 16)
    chunk = panel % chunks
    count = tl.minimum(32, length - chunk * 32)
    valid_target = target < count
    valid_source = source < count
    safe_target = tl.minimum(target, count - 1)
    causal = (
        valid_target[:, None]
        & valid_source[None, :]
        & (source[None, :] <= target[:, None])
    )
    temporal = tl.load(
        weights
        + (panel * 32 + target[:, None]) * 32
        + source[None, :],
        mask=causal,
        other=0.0,
    ).to(tl.float32)
    component: tl.constexpr = 2 if UPPER else 0
    scale_j = tl.load(
        radial_scale + (panel * 32 + target) * 4 + component,
        mask=valid_target,
        other=0.0,
    ).to(tl.float32)
    scale_d = tl.load(
        radial_scale + (panel * 32 + target) * 4 + component + 1,
        mask=valid_target,
        other=0.0,
    ).to(tl.float32)

    corr_right_u = tl.zeros((32, 32), tl.float32)
    corr_right_h = tl.zeros((32, 32), tl.float32)
    corr_left_u = tl.zeros((32, 32), tl.float32)
    for block in tl.static_range(0, 8):
        coordinates = block * 16 + local
        vector_offsets = _raw_vector_offsets(
            panel,
            source[:, None],
            coordinates[None, :],
            length,
            heads,
            chunks,
        )
        source_u = tl.load(
            u + vector_offsets,
            mask=valid_source[:, None],
            other=0.0,
        ).to(tl.float32)
        source_h = tl.load(
            h + vector_offsets,
            mask=valid_source[:, None],
            other=0.0,
        ).to(tl.float32)
        left = _left_descriptor(
            key,
            erase,
            query,
            lower_dual,
            action_primal,
            panel,
            safe_target[:, None],
            coordinates[None, :],
            route,
            length,
            heads,
            chunks,
            UPPER,
        )
        right = _right_descriptor(
            d,
            lower_primal,
            original_dual,
            action_dual,
            panel,
            safe_target[:, None],
            coordinates[None, :],
            route,
            length,
            heads,
            chunks,
            UPPER,
        )
        left = tl.where(valid_target[:, None], left, 0.0)
        right = tl.where(valid_target[:, None], right, 0.0)
        right_side = (
            block > coordinate_block if UPPER else block < coordinate_block
        )
        left_side = (
            block < coordinate_block if UPPER else block > coordinate_block
        )
        corr_right_u += _mma(
            tl.where(right_side, right, 0.0), tl.trans(source_u)
        )
        corr_right_h += _mma(
            tl.where(right_side, right, 0.0), tl.trans(source_h)
        )
        corr_left_u += _mma(
            tl.where(left_side, left, 0.0), tl.trans(source_u)
        )

    coordinates = coordinate_block * 16 + local
    vector_offsets = _raw_vector_offsets(
        panel,
        source[:, None],
        coordinates[None, :],
        length,
        heads,
        chunks,
    )
    source_u = tl.load(
        u + vector_offsets,
        mask=valid_source[:, None],
        other=0.0,
    ).to(tl.float32)
    source_h = tl.load(
        h + vector_offsets,
        mask=valid_source[:, None],
        other=0.0,
    ).to(tl.float32)
    left = _left_descriptor(
        key,
        erase,
        query,
        lower_dual,
        action_primal,
        panel,
        safe_target[:, None],
        coordinates[None, :],
        route,
        length,
        heads,
        chunks,
        UPPER,
    )
    right = _right_descriptor(
        d,
        lower_primal,
        original_dual,
        action_dual,
        panel,
        safe_target[:, None],
        coordinates[None, :],
        route,
        length,
        heads,
        chunks,
        UPPER,
    )
    left = tl.where(valid_target[:, None], left, 0.0)
    right = tl.where(valid_target[:, None], right, 0.0)
    output_u = _mma(
        tl.trans(
            temporal
            * (scale_j[:, None] * corr_right_u + scale_d[:, None] * corr_right_h)
        ),
        left,
    )
    output_u += _mma(
        tl.trans(temporal * scale_j[:, None] * corr_left_u), right
    )
    output_h = _mma(
        tl.trans(temporal * scale_d[:, None] * corr_left_u), right
    )
    left_u = _mma(left, tl.trans(source_u))
    pair_j = left_u * corr_right_u
    pair_d = left_u * corr_right_h

    tl.atomic_add(
        grad_u + vector_offsets,
        output_u,
        mask=valid_source[:, None],
        sem="relaxed",
    )
    tl.atomic_add(
        grad_h + vector_offsets,
        output_h,
        mask=valid_source[:, None],
        sem="relaxed",
    )
    tl.atomic_add(
        grad_weights
        + (panel * 32 + target[:, None]) * 32
        + source[None, :],
        scale_j[:, None] * pair_j + scale_d[:, None] * pair_d,
        mask=causal,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_radial_scale + (panel * 32 + target) * 4 + component,
        tl.sum(temporal * pair_j, axis=1),
        mask=valid_target,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_radial_scale + (panel * 32 + target) * 4 + component + 1,
        tl.sum(temporal * pair_d, axis=1),
        mask=valid_target,
        sem="relaxed",
    )


@triton.jit
def _streamed_boundary_scalar_kernel(
    key,
    erase,
    query,
    d,
    lower_primal,
    lower_dual,
    original_dual,
    action_primal,
    action_dual,
    boundary_j,
    boundary_d,
    theta,
    radial_scale,
    grad_radial_scale,
    grad_theta,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    UPPER: tl.constexpr,
):
    panel = tl.program_id(0)
    row_block = tl.program_id(1)
    route = tl.program_id(2)
    target = tl.arange(0, 32)
    rows = row_block * 16 + tl.arange(0, 16)
    chunk = panel % chunks
    count = tl.minimum(32, length - chunk * 32)
    valid = target < count
    safe_target = tl.minimum(target, count - 1)
    projection_j = tl.zeros((16, 32), tl.float32)
    projection_d = tl.zeros((16, 32), tl.float32)
    for start in tl.static_range(0, 128, 32):
        columns = start + tl.arange(0, 32)
        strict = (
            columns[None, :] > rows[:, None]
            if UPPER
            else rows[:, None] > columns[None, :]
        )
        matrix = panel * 128 * 128 + rows[:, None] * 128 + columns[None, :]
        local_j = tl.load(
            boundary_j + matrix, mask=strict, other=0.0
        ).to(tl.float32)
        local_d = tl.load(
            boundary_d + matrix, mask=strict, other=0.0
        ).to(tl.float32)
        right = _right_descriptor(
            d,
            lower_primal,
            original_dual,
            action_dual,
            panel,
            safe_target[:, None],
            columns[None, :],
            route,
            length,
            heads,
            chunks,
            UPPER,
        )
        right = tl.where(valid[:, None], right, 0.0)
        projection_j += _mma(local_j, tl.trans(right))
        projection_d += _mma(local_d, tl.trans(right))
    left = _left_descriptor(
        key,
        erase,
        query,
        lower_dual,
        action_primal,
        panel,
        safe_target[:, None],
        rows[None, :],
        route,
        length,
        heads,
        chunks,
        UPPER,
    )
    left = tl.where(valid[:, None], left, 0.0)
    scalar_j = tl.sum(tl.trans(left) * projection_j, axis=0)
    scalar_d = tl.sum(tl.trans(left) * projection_d, axis=0)
    component: tl.constexpr = 2 if UPPER else 0
    scalar = panel * 32 + target
    local_theta = tl.load(theta + scalar, mask=valid, other=0.0)
    scale_j = tl.load(
        radial_scale + scalar * 4 + component, mask=valid, other=0.0
    )
    scale_d = tl.load(
        radial_scale + scalar * 4 + component + 1, mask=valid, other=0.0
    )
    tl.atomic_add(
        grad_radial_scale + scalar * 4 + component,
        local_theta * scalar_j,
        mask=valid,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_radial_scale + scalar * 4 + component + 1,
        local_theta * scalar_d,
        mask=valid,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_theta + scalar,
        scale_j * scalar_j + scale_d * scalar_d,
        mask=valid,
        sem="relaxed",
    )


@triton.jit
def _streamed_stage_diagonal_kernel(
    u,
    h,
    key,
    erase,
    query,
    d,
    lower_primal,
    lower_dual,
    original_dual,
    action_primal,
    action_dual,
    weights,
    radial_scale,
    grad_radial_scale,
    grad_weights,
    grad_u,
    grad_h,
    length,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    UPPER: tl.constexpr,
):
    panel = tl.program_id(0)
    coordinate_block = tl.program_id(1)
    source = tl.arange(0, 32)
    local = tl.arange(0, 16)
    coordinates = coordinate_block * 16 + local
    chunk = panel % chunks
    count = tl.minimum(32, length - chunk * 32)
    valid_source = source < count
    vector_offsets = _raw_vector_offsets(
        panel,
        source[:, None],
        coordinates[None, :],
        length,
        heads,
        chunks,
    )
    source_u = tl.load(
        u + vector_offsets,
        mask=valid_source[:, None],
        other=0.0,
    ).to(tl.float32)
    source_h = tl.load(
        h + vector_offsets,
        mask=valid_source[:, None],
        other=0.0,
    ).to(tl.float32)
    source_u_t = tl.trans(source_u)
    source_h_t = tl.trans(source_h)
    output_u = tl.trans(
        tl.load(
            grad_u + vector_offsets,
            mask=valid_source[:, None],
            other=0.0,
        ).to(tl.float32)
    )
    output_h = tl.trans(
        tl.load(
            grad_h + vector_offsets,
            mask=valid_source[:, None],
            other=0.0,
        ).to(tl.float32)
    )
    strict = (
        local[:, None] < local[None, :]
        if UPPER
        else local[:, None] > local[None, :]
    )

    for target in tl.range(0, 32):
        active_source = valid_source & (source <= target) & (target < count)
        safe_target = tl.minimum(target, count - 1)
        temporal = tl.load(
            weights + (panel * 32 + target) * 32 + source,
            mask=active_source,
            other=0.0,
        ).to(tl.float32)
        scalar_base = (panel * 32 + target) * 4
        component: tl.constexpr = 2 if UPPER else 0
        scale_j = tl.load(
            radial_scale + scalar_base + component,
            mask=target < count,
            other=0.0,
        ).to(tl.float32)
        scale_d = tl.load(
            radial_scale + scalar_base + component + 1,
            mask=target < count,
            other=0.0,
        ).to(tl.float32)
        matrix = tl.zeros((16, 16), tl.float32)
        for route in tl.static_range(0, 3):
            left = _left_descriptor(
                key,
                erase,
                query,
                lower_dual,
                action_primal,
                panel,
                safe_target,
                coordinates,
                route,
                length,
                heads,
                chunks,
                UPPER,
            )
            right = _right_descriptor(
                d,
                lower_primal,
                original_dual,
                action_dual,
                panel,
                safe_target,
                coordinates,
                route,
                length,
                heads,
                chunks,
                UPPER,
            )
            left = tl.where(target < count, left, 0.0)
            right = tl.where(target < count, right, 0.0)
            matrix += tl.where(
                strict,
                left[:, None] * right[None, :],
                0.0,
            )
        action_u = _mma(matrix, source_u_t)
        action_h = _mma(matrix, source_h_t)
        transpose_u = _mma(tl.trans(matrix), source_u_t)
        output_u += temporal[None, :] * (
            scale_j * (action_u + transpose_u) + scale_d * action_h
        )
        output_h += temporal[None, :] * scale_d * transpose_u
        pair_j = tl.sum(source_u_t * action_u, axis=0)
        pair_d = tl.sum(source_u_t * action_h, axis=0)
        weight_gradient = scale_j * pair_j + scale_d * pair_d
        tl.atomic_add(
            grad_radial_scale + scalar_base + component,
            tl.sum(temporal * pair_j, axis=0),
            mask=target < count,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_radial_scale + scalar_base + component + 1,
            tl.sum(temporal * pair_d, axis=0),
            mask=target < count,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_weights + (panel * 32 + target) * 32 + source,
            weight_gradient,
            mask=active_source,
            sem="relaxed",
        )
    tl.store(
        grad_u + vector_offsets,
        tl.trans(output_u),
        mask=valid_source[:, None],
    )
    tl.store(
        grad_h + vector_offsets,
        tl.trans(output_h),
        mask=valid_source[:, None],
    )


def allocate_streamed_chart_gradients(
    u: torch.Tensor,
    radial_scale: torch.Tensor,
    theta: torch.Tensor,
    weights: torch.Tensor,
) -> StreamedChartGradients:
    return StreamedChartGradients(
        torch.zeros_like(radial_scale),
        torch.zeros_like(theta),
        torch.zeros_like(weights),
        torch.zeros(u.shape, device=u.device, dtype=torch.float32),
        torch.zeros(u.shape, device=u.device, dtype=torch.float32),
    )


def streamed_chart_stage_reverse(
    u: torch.Tensor,
    h: torch.Tensor,
    key: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    d: torch.Tensor,
    lower_primal: torch.Tensor,
    lower_dual: torch.Tensor,
    original_dual: torch.Tensor,
    action_primal: torch.Tensor,
    action_dual: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    theta: torch.Tensor,
    weights: torch.Tensor,
    radial_scale: torch.Tensor,
    *,
    upper: bool,
    gradients: StreamedChartGradients | None = None,
) -> StreamedChartGradients:
    batch, length, heads, _ = u.shape
    chunks = triton.cdiv(length, _CHUNK)
    panels = batch * heads * chunks
    if gradients is None:
        gradients = allocate_streamed_chart_gradients(
            u, radial_scale, theta, weights
        )
    grad_radial_scale = gradients.radial_scale
    grad_theta = gradients.theta
    grad_weights = gradients.weights
    grad_u = gradients.u
    grad_h = gradients.h
    grid = (panels, _RANK // _TILE, 3)
    common = (
        u,
        h,
        key,
        erase,
        query,
        d,
        lower_primal,
        lower_dual,
        original_dual,
    )
    _streamed_local_kernel[grid](
        *common,
        action_primal,
        action_dual,
        weights,
        radial_scale,
        grad_radial_scale,
        grad_weights,
        grad_u,
        grad_h,
        length,
        heads=heads,
        chunks=chunks,
        UPPER=upper,
        num_warps=4,
        num_stages=2,
    )
    _streamed_boundary_scalar_kernel[grid](
        *common[2:],
        action_primal,
        action_dual,
        boundary_j,
        boundary_d,
        theta,
        radial_scale,
        grad_radial_scale,
        grad_theta,
        length,
        heads=heads,
        chunks=chunks,
        UPPER=upper,
        num_warps=4,
        num_stages=2,
    )
    _streamed_stage_diagonal_kernel[(panels, _RANK // _TILE)](
        u,
        h,
        key,
        erase,
        query,
        d,
        lower_primal,
        lower_dual,
        original_dual,
        action_primal,
        action_dual,
        weights,
        radial_scale,
        grad_radial_scale,
        grad_weights,
        grad_u,
        grad_h,
        length,
        heads=heads,
        chunks=chunks,
        UPPER=upper,
        num_warps=4,
        num_stages=2,
    )
    return gradients


__all__ = [
    "StreamedChartGradients",
    "allocate_streamed_chart_gradients",
    "streamed_chart_stage_reverse",
]
