# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
"""Coordinate-gated pre-decay Oja pair and strict transpose.

This is a residual-before-decay specialization of FLA's coordinate-gated Oja
pair schedule. For inclusive prefixes ``G_i`` and token log-retentions ``g_i``
the strict interaction is

``A_ij = gamma_i sum_c u_ic u_jc exp(G_ic-g_ic-G_jc)``, ``j < i``.

The exponent is the retention from ``j+1`` through ``i-1`` and is always
nonpositive. Cross-subchunk pairs use FLA's centered Tensor Core factorization;
diagonal subchunks and the strict transpose evaluate the bounded exponent
directly. No reciprocal-retention panel is formed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp


@triton.jit
def _vector_pair_cross_fwd_kernel(
    source,
    beta,
    log_decay,
    gate,
    pair,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    subpair = tl.program_id(1)
    batch_head = tl.program_id(2).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    row_block = subpair // NC
    column_block = subpair % NC
    if row_block <= column_block:
        return
    if chunk * C + row_block * BC >= T:
        return

    row = chunk * C + row_block * BC + tl.arange(0, BC)
    column = chunk * C + column_block * BC + tl.arange(0, BC)
    valid_row = row < T
    beta_row = tl.load(
        beta + (batch * T + row) * H + head,
        mask=valid_row,
        other=0.0,
    )
    value = tl.zeros((BC, BC), dtype=tl.float32)

    for block in range(0, tl.cdiv(R, BK)):
        coord = block * BK + tl.arange(0, BK)
        valid_r = coord < R
        row_offset = (
            ((batch * T + row[:, None]) * H + head) * R
            + coord[None, :]
        )
        column_offset = (
            ((batch * T + column[None, :]) * H + head) * R
            + coord[:, None]
        )
        row_mask = valid_row[:, None] & valid_r[None, :]
        column_mask = valid_r[:, None] & (column[None, :] < T)

        center_token = chunk * C + row_block * BC
        center_offset = (
            ((batch * T + center_token) * H + head) * R + coord
        )
        center = tl.load(
            gate + center_offset, mask=valid_r, other=0.0
        ).to(tl.float32) - tl.load(
            log_decay + center_offset, mask=valid_r, other=0.0
        ).to(tl.float32)

        row_gate = tl.load(gate + row_offset, mask=row_mask, other=0.0).to(
            tl.float32
        ) - tl.load(
            log_decay + row_offset, mask=row_mask, other=0.0
        ).to(tl.float32)
        column_gate = tl.load(
            gate + column_offset, mask=column_mask, other=0.0
        ).to(tl.float32)
        row_source = tl.load(
            source + row_offset, mask=row_mask, other=0.0
        ) * exp(row_gate - center[None, :])
        column_source = tl.load(
            source + column_offset, mask=column_mask, other=0.0
        ) * exp(center[:, None] - column_gate)
        value += tl.dot(row_source, column_source)

    pair_offset = (
        ((batch * T + row[:, None]) * H + head) * C
        + column_block * BC
        + tl.arange(0, BC)[None, :]
    )
    tl.store(
        pair + pair_offset,
        beta_row[:, None] * value,
        mask=valid_row[:, None] & (column[None, :] < T),
    )


@triton.jit
def _vector_pair_intra_fwd_kernel(
    source,
    beta,
    log_decay,
    gate,
    pair,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    subchunk = tl.program_id(1)
    batch_head = tl.program_id(2).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    local = tl.arange(0, BC)
    row = chunk * C + subchunk * BC + local
    valid_row = row < T
    coord = tl.arange(0, BK)
    valid_r = coord < R
    row_offset = (
        ((batch * T + row[:, None]) * H + head) * R + coord[None, :]
    )
    row_mask = valid_row[:, None] & valid_r[None, :]
    source_row = tl.load(source + row_offset, mask=row_mask, other=0.0)
    exclusive_gate = tl.load(
        gate + row_offset, mask=row_mask, other=0.0
    ).to(tl.float32) - tl.load(
        log_decay + row_offset, mask=row_mask, other=0.0
    ).to(tl.float32)
    beta_row = tl.load(
        beta + (batch * T + row) * H + head,
        mask=valid_row,
        other=0.0,
    ).to(tl.float32)

    for column_local in tl.static_range(0, BC):
        column = chunk * C + subchunk * BC + column_local
        valid_column = (column < T) & valid_r
        column_offset = (
            ((batch * T + column) * H + head) * R + coord
        )
        source_column = tl.load(
            source + column_offset, mask=valid_column, other=0.0
        )
        column_gate = tl.load(
            gate + column_offset, mask=valid_column, other=0.0
        ).to(tl.float32)
        active = (local > column_local) & valid_row & (column < T)
        exponent = exp(
            tl.where(
                active[:, None],
                exclusive_gate - column_gate[None, :],
                0.0,
            )
        )
        value = beta_row * tl.sum(
            source_row * source_column[None, :] * exponent, axis=1
        )
        pair_offset = (
            ((batch * T + row) * H + head) * C
            + subchunk * BC
            + column_local
        )
        tl.store(pair + pair_offset, tl.where(active, value, 0.0), mask=valid_row)


@triton.jit
def _vector_pair_bwd_kernel(
    source,
    beta,
    log_decay,
    gate,
    grad_pair,
    grad_source,
    grad_beta_partial,
    grad_gate,
    grad_log_decay,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    NK: tl.constexpr,
):
    owner = tl.program_id(0).to(tl.int64)
    coord_block = owner % NK
    owner //= NK
    subchunk = owner % NC
    chunk = owner // NC
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H

    local = subchunk * BC + tl.arange(0, BC)
    token = chunk * C + local
    valid_t = token < T
    coord = coord_block * BK + tl.arange(0, BK)
    valid_r = coord < R
    offset = (
        ((batch * T + token[:, None]) * H + head) * R + coord[None, :]
    )
    mask = valid_t[:, None] & valid_r[None, :]
    source_tile = tl.load(source + offset, mask=mask, other=0.0)
    gate_tile = tl.load(gate + offset, mask=mask, other=0.0).to(
        tl.float32
    )
    decay_tile = tl.load(
        log_decay + offset, mask=mask, other=0.0
    ).to(tl.float32)
    beta_tile = tl.load(
        beta + (batch * T + token) * H + head,
        mask=valid_t,
        other=0.0,
    ).to(tl.float32)
    source_cotangent = tl.zeros((BC, BK), dtype=tl.float32)
    gate_cotangent = tl.zeros((BC, BK), dtype=tl.float32)
    decay_cotangent = tl.zeros((BC, BK), dtype=tl.float32)
    beta_cotangent = tl.zeros((BC,), dtype=tl.float32)

    for counterpart_local in tl.static_range(0, C):
        counterpart = chunk * C + counterpart_local
        valid_counterpart = (counterpart < T) & valid_r
        counterpart_offset = (
            ((batch * T + counterpart) * H + head) * R + coord
        )
        source_counterpart = tl.load(
            source + counterpart_offset,
            mask=valid_counterpart,
            other=0.0,
        )
        gate_counterpart = tl.load(
            gate + counterpart_offset,
            mask=valid_counterpart,
            other=0.0,
        ).to(tl.float32)
        decay_counterpart = tl.load(
            log_decay + counterpart_offset,
            mask=valid_counterpart,
            other=0.0,
        ).to(tl.float32)

        row_active = valid_t & (local > counterpart_local) & (counterpart < T)
        row_exponent = exp(
            tl.where(
                row_active[:, None],
                gate_tile - decay_tile - gate_counterpart[None, :],
                0.0,
            )
        )
        row_pair_cotangent = tl.load(
            grad_pair
            + ((batch * T + token) * H + head) * C
            + counterpart_local,
            mask=valid_t,
            other=0.0,
        ).to(tl.float32)
        row_pair_cotangent = tl.where(
            row_active, row_pair_cotangent, 0.0
        )
        row_weight = (
            row_pair_cotangent[:, None]
            * beta_tile[:, None]
            * row_exponent
        )
        source_cotangent += row_weight * source_counterpart[None, :]
        row_common = (
            row_weight * source_tile * source_counterpart[None, :]
        )
        gate_cotangent += row_common
        decay_cotangent -= row_common
        beta_cotangent += row_pair_cotangent * tl.sum(
            source_tile * source_counterpart[None, :] * row_exponent,
            axis=1,
        )

        column_active = (
            valid_t & (counterpart_local > local) & (counterpart < T)
        )
        column_exponent = exp(
            tl.where(
                column_active[:, None],
                gate_counterpart[None, :]
                - decay_counterpart[None, :]
                - gate_tile,
                0.0,
            )
        )
        column_pair_cotangent = tl.load(
            grad_pair
            + ((batch * T + counterpart) * H + head) * C
            + local,
            mask=valid_t & (counterpart < T),
            other=0.0,
        ).to(tl.float32)
        counterpart_beta = tl.load(
            beta + (batch * T + counterpart) * H + head,
            mask=counterpart < T,
            other=0.0,
        ).to(tl.float32)
        column_pair_cotangent = tl.where(
            column_active, column_pair_cotangent, 0.0
        )
        column_weight = (
            column_pair_cotangent[:, None]
            * counterpart_beta
            * column_exponent
        )
        source_cotangent += column_weight * source_counterpart[None, :]
        gate_cotangent -= (
            column_weight * source_tile * source_counterpart[None, :]
        )

    tl.store(grad_source + offset, source_cotangent, mask=mask)
    tl.store(grad_gate + offset, gate_cotangent, mask=mask)
    tl.store(grad_log_decay + offset, decay_cotangent, mask=mask)
    beta_partial_offset = (
        ((batch * NK + coord_block) * T + token) * H + head
    )
    tl.store(
        grad_beta_partial + beta_partial_offset,
        beta_cotangent,
        mask=valid_t,
    )


def vector_pair_forward(
    source: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    gate_cumsum: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    batch, length, heads, rank = source.shape
    if log_decay.shape != source.shape or gate_cumsum.shape != source.shape:
        raise ValueError("vector gate tensors must match source shape")
    if chunk_size % 16 != 0:
        raise ValueError("vector pair chunk size must be divisible by 16")
    pair = torch.zeros(
        batch,
        length,
        heads,
        chunk_size,
        dtype=torch.float32,
        device=source.device,
    )
    chunks = triton.cdiv(length, chunk_size)
    block_time = 16
    subchunks = triton.cdiv(chunk_size, block_time)
    _vector_pair_cross_fwd_kernel[
        (chunks, subchunks * subchunks, batch * heads)
    ](
        source,
        beta,
        log_decay,
        gate_cumsum,
        pair,
        T=length,
        H=heads,
        R=rank,
        C=chunk_size,
        BC=block_time,
        BK=64,
        NC=subchunks,
        num_warps=4,
        num_stages=3,
    )
    _vector_pair_intra_fwd_kernel[(chunks, subchunks, batch * heads)](
        source,
        beta,
        log_decay,
        gate_cumsum,
        pair,
        T=length,
        H=heads,
        R=rank,
        C=chunk_size,
        BC=block_time,
        BK=max(16, triton.next_power_of_2(rank)),
        num_warps=4,
        num_stages=3,
    )
    return pair


def vector_pair_backward(
    source: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    gate_cumsum: torch.Tensor,
    grad_pair: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length, heads, rank = source.shape
    block_time = 16
    block_rank = min(64, triton.next_power_of_2(rank))
    subchunks = triton.cdiv(chunk_size, block_time)
    coord_blocks = triton.cdiv(rank, block_rank)
    chunks = triton.cdiv(length, chunk_size)
    grad_source = torch.empty_like(source, dtype=torch.float32)
    grad_gate = torch.empty_like(gate_cumsum, dtype=torch.float32)
    grad_log_decay = torch.empty_like(log_decay, dtype=torch.float32)
    grad_beta_partial = torch.empty(
        batch,
        coord_blocks,
        length,
        heads,
        dtype=torch.float32,
        device=source.device,
    )
    _vector_pair_bwd_kernel[
        (chunks * subchunks * coord_blocks, batch * heads)
    ](
        source,
        beta,
        log_decay,
        gate_cumsum,
        grad_pair,
        grad_source,
        grad_beta_partial,
        grad_gate,
        grad_log_decay,
        T=length,
        H=heads,
        R=rank,
        C=chunk_size,
        BC=block_time,
        BK=block_rank,
        NC=subchunks,
        NK=coord_blocks,
        num_warps=4,
        num_stages=3,
    )
    grad_beta = grad_beta_partial.sum(dim=1)
    return grad_source, grad_gate, grad_beta, grad_log_decay


__all__ = ["vector_pair_backward", "vector_pair_forward"]
