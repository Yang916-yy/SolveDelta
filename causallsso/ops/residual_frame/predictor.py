# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
# Specialized from FLA's MIT-licensed gated-Oja pair/WY/state owners.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.gated_oja_rule.chunk_h import chunk_oja_fwd_h
from fla.ops.gated_oja_rule.chunk_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from fla.ops.utils import solve_tril


@triton.jit
def _recompute_residual_wy_kernel(
    target,
    source,
    beta,
    inverse,
    w,
    update,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    BR: tl.constexpr,
    TARGET_STRIDE_B: tl.constexpr,
    TARGET_STRIDE_T: tl.constexpr,
    TARGET_STRIDE_H: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    token = chunk * C + tl.arange(0, C)
    valid_t = token < T
    row = tl.arange(0, C)
    inverse_offset = (
        ((batch * T + token[:, None]) * H + head) * C + row[None, :]
    )
    inv = tl.load(
        inverse + inverse_offset,
        mask=valid_t[:, None],
        other=0.0,
    )
    beta_offset = (batch * T + token) * H + head
    scale = tl.load(beta + beta_offset, mask=valid_t, other=0.0)

    for block in range(0, tl.cdiv(R, BR)):
        coord = block * BR + tl.arange(0, BR)
        valid_r = coord < R
        packed_offset = (
            ((batch * T + token[:, None]) * H + head) * R
            + coord[None, :]
        )
        target_offset = (
            batch * TARGET_STRIDE_B
            + token[:, None] * TARGET_STRIDE_T
            + head * TARGET_STRIDE_H
            + coord[None, :]
        )
        src = tl.load(
            source + packed_offset,
            mask=valid_t[:, None] & valid_r[None, :],
            other=0.0,
        )
        tgt = tl.load(
            target + target_offset,
            mask=valid_t[:, None] & valid_r[None, :],
            other=0.0,
        )
        src_out = tl.dot(inv, (src * scale[:, None]).to(src.dtype))
        tgt_out = tl.dot(inv, (tgt * scale[:, None]).to(tgt.dtype))
        tl.store(
            w + packed_offset,
            src_out,
            mask=valid_t[:, None] & valid_r[None, :],
        )
        tl.store(
            update + packed_offset,
            tgt_out,
            mask=valid_t[:, None] & valid_r[None, :],
        )


def recompute_residual_wy(target, source, beta, inverse, *, chunk_size):
    batch, length, heads, rank = target.shape
    w = torch.empty_like(source)
    update = torch.empty_like(target)
    _recompute_residual_wy_kernel[(triton.cdiv(length, chunk_size), batch * heads)](
        target,
        source,
        beta,
        inverse,
        w,
        update,
        T=length,
        H=heads,
        R=rank,
        C=chunk_size,
        BR=64,
        TARGET_STRIDE_B=target.stride(0),
        TARGET_STRIDE_T=target.stride(1),
        TARGET_STRIDE_H=target.stride(2),
        num_warps=4,
        num_stages=3,
    )
    return w, update


@triton.jit
def _residual_source_state_bwd_kernel(
    residual,
    states,
    grad_states,
    grad_update,
    grad_source,
    grad_w,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    BR: tl.constexpr,
):
    value_block = tl.program_id(0)
    chunk = tl.program_id(1).to(tl.int64)
    batch_head = tl.program_id(2).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    token = chunk * C + tl.arange(0, C)
    value = value_block * BR + tl.arange(0, BR)
    valid_t = token < T
    valid_v = value < R
    token_value = (
        ((batch * T + token[:, None]) * H + head) * R
        + value[None, :]
    )
    grad_source_tile = tl.zeros((C, BR), dtype=tl.float32)
    grad_w_tile = tl.zeros((C, BR), dtype=tl.float32)

    for rank_block in range(0, tl.cdiv(R, BR)):
        rank = rank_block * BR + tl.arange(0, BR)
        valid_r = rank < R
        token_rank = (
            ((batch * T + token[:, None]) * H + head) * R
            + rank[None, :]
        )
        state_offset = (
            ((batch * tl.cdiv(T, C) + chunk) * H + head) * R * R
            + rank[:, None] * R
            + value[None, :]
        )
        residual_tile = tl.load(
            residual + token_rank,
            mask=valid_t[:, None] & valid_r[None, :],
            other=0.0,
        )
        update_cotangent = tl.load(
            grad_update + token_rank,
            mask=valid_t[:, None] & valid_r[None, :],
            other=0.0,
        )
        state_tile = tl.load(
            states + state_offset,
            mask=valid_r[:, None] & valid_v[None, :],
            other=0.0,
        )
        grad_state_tile = tl.load(
            grad_states + state_offset,
            mask=valid_r[:, None] & valid_v[None, :],
            other=0.0,
        )
        grad_source_tile += tl.dot(
            residual_tile,
            grad_state_tile.to(residual_tile.dtype),
        )
        grad_w_tile += tl.dot(
            update_cotangent,
            state_tile.to(update_cotangent.dtype),
        )

    tl.store(
        grad_source + token_value,
        grad_source_tile,
        mask=valid_t[:, None] & valid_v[None, :],
    )
    tl.store(
        grad_w + token_value,
        -grad_w_tile,
        mask=valid_t[:, None] & valid_v[None, :],
    )


def residual_source_state_backward(
    residual,
    source,
    states,
    grad_states,
    grad_update,
    *,
    chunk_size,
):
    batch, length, heads, rank = source.shape
    chunks = triton.cdiv(length, chunk_size)
    grad_source = torch.empty_like(source, dtype=torch.float32)
    grad_w = torch.empty_like(source)
    _residual_source_state_bwd_kernel[
        (triton.cdiv(rank, 64), chunks, batch * heads)
    ](
        residual,
        states,
        grad_states,
        grad_update,
        grad_source,
        grad_w,
        T=length,
        H=heads,
        R=rank,
        C=chunk_size,
        BR=64,
        num_warps=4,
        num_stages=3,
    )
    return grad_source, grad_w


@triton.jit
def _residual_wy_bwd_kernel(
    target,
    source,
    beta,
    inverse,
    grad_w,
    grad_update,
    grad_target,
    grad_source,
    grad_beta,
    grad_pair,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    BR: tl.constexpr,
    TARGET_STRIDE_B: tl.constexpr,
    TARGET_STRIDE_T: tl.constexpr,
    TARGET_STRIDE_H: tl.constexpr,
    ACCUMULATE_GRAD_SOURCE: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    token = chunk * C + tl.arange(0, C)
    valid_t = token < T
    row = tl.arange(0, C)
    inverse_offset = (
        ((batch * T + token[None, :]) * H + head) * C
        + row[:, None]
    )
    inverse_tile = tl.load(
        inverse + inverse_offset,
        mask=valid_t[None, :],
        other=0.0,
    )
    beta_offset = (batch * T + token) * H + head
    beta_tile = tl.load(beta + beta_offset, mask=valid_t, other=0.0)
    grad_beta_tile = tl.zeros((C,), dtype=tl.float32)
    grad_pair_tile = tl.zeros((C, C), dtype=tl.float32)

    for block in range(0, tl.cdiv(R, BR)):
        coord = block * BR + tl.arange(0, BR)
        valid_r = coord < R
        packed_offset = (
            ((batch * T + token[:, None]) * H + head) * R
            + coord[None, :]
        )
        target_offset = (
            batch * TARGET_STRIDE_B
            + token[:, None] * TARGET_STRIDE_T
            + head * TARGET_STRIDE_H
            + coord[None, :]
        )
        source_tile = tl.load(
            source + packed_offset,
            mask=valid_t[:, None] & valid_r[None, :],
            other=0.0,
        )
        target_tile = tl.load(
            target + target_offset,
            mask=valid_t[:, None] & valid_r[None, :],
            other=0.0,
        )
        grad_w_tile = tl.load(
            grad_w + packed_offset,
            mask=valid_t[:, None] & valid_r[None, :],
            other=0.0,
        )
        grad_update_tile = tl.load(
            grad_update + packed_offset,
            mask=valid_t[:, None] & valid_r[None, :],
            other=0.0,
        )
        source_scaled = (source_tile * beta_tile[:, None]).to(source_tile.dtype)
        target_scaled = (target_tile * beta_tile[:, None]).to(target_tile.dtype)
        grad_pair_tile += tl.dot(
            grad_w_tile,
            tl.trans(source_scaled),
        )
        grad_pair_tile += tl.dot(
            grad_update_tile,
            tl.trans(target_scaled),
        )
        grad_source_scaled = tl.dot(inverse_tile, grad_w_tile)
        grad_target_scaled = tl.dot(inverse_tile, grad_update_tile)
        grad_source_block = grad_source_scaled * beta_tile[:, None]
        if ACCUMULATE_GRAD_SOURCE:
            grad_source_block += tl.load(
                grad_source + packed_offset,
                mask=valid_t[:, None] & valid_r[None, :],
                other=0.0,
            ).to(tl.float32)
        grad_target_block = grad_target_scaled * beta_tile[:, None]
        grad_beta_tile += tl.sum(
            grad_source_scaled * source_tile
            + grad_target_scaled * target_tile,
            axis=1,
        )
        tl.store(
            grad_source + packed_offset,
            grad_source_block,
            mask=valid_t[:, None] & valid_r[None, :],
        )
        tl.store(
            grad_target + packed_offset,
            grad_target_block,
            mask=valid_t[:, None] & valid_r[None, :],
        )

    strict = (token[:, None] > token[None, :]) & (
        valid_t[:, None] & valid_t[None, :]
    )
    grad_pair_tile = tl.where(strict, grad_pair_tile, 0.0)
    grad_pair_tile = tl.dot(
        grad_pair_tile.to(inverse_tile.dtype),
        inverse_tile,
    )
    grad_pair_tile = tl.dot(
        inverse_tile,
        grad_pair_tile.to(inverse_tile.dtype),
    )
    grad_pair_tile = tl.where(strict, -grad_pair_tile, 0.0)
    pair_offset = (
        ((batch * T + token[:, None]) * H + head) * C
        + row[None, :]
    )
    tl.store(
        grad_pair + pair_offset,
        grad_pair_tile,
        mask=valid_t[:, None],
    )
    tl.store(grad_beta + beta_offset, grad_beta_tile, mask=valid_t)


def residual_wy_backward(
    target,
    source,
    beta,
    inverse,
    grad_w,
    grad_update,
    *,
    grad_source_accumulator=None,
):
    batch, length, heads, rank = source.shape
    chunk_size = inverse.shape[-1]
    grad_target = torch.empty_like(source)
    grad_source = (
        torch.empty_like(source, dtype=torch.float32)
        if grad_source_accumulator is None
        else grad_source_accumulator
    )
    grad_beta = torch.empty_like(beta, dtype=torch.float32)
    grad_pair = torch.empty_like(inverse, dtype=torch.float32)
    _residual_wy_bwd_kernel[(triton.cdiv(length, chunk_size), batch * heads)](
        target,
        source,
        beta,
        inverse,
        grad_w,
        grad_update,
        grad_target,
        grad_source,
        grad_beta,
        grad_pair,
        T=length,
        H=heads,
        R=rank,
        C=chunk_size,
        BR=64,
        TARGET_STRIDE_B=target.stride(0),
        TARGET_STRIDE_T=target.stride(1),
        TARGET_STRIDE_H=target.stride(2),
        ACCUMULATE_GRAD_SOURCE=grad_source_accumulator is not None,
        num_warps=4,
        num_stages=3,
    )
    return grad_target, grad_source, grad_beta, grad_pair


@triton.jit
def _residual_pair_bwd_kernel(
    source,
    beta,
    grad_pair,
    grad_source_base,
    grad_source,
    grad_beta,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    BR: tl.constexpr,
    ACCUMULATE_GRAD_SOURCE: tl.constexpr,
    ACCUMULATE_GRAD_BETA: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    token = chunk * C + tl.arange(0, C)
    valid_t = token < T
    row = tl.arange(0, C)
    pair_offset = (
        ((batch * T + token[:, None]) * H + head) * C
        + row[None, :]
    )
    pair_tile = tl.load(
        grad_pair + pair_offset,
        mask=valid_t[:, None],
        other=0.0,
    )
    strict = (token[:, None] > token[None, :]) & (
        valid_t[:, None] & valid_t[None, :]
    )
    pair_tile = tl.where(strict, pair_tile, 0.0)
    beta_offset = (batch * T + token) * H + head
    beta_tile = tl.load(beta + beta_offset, mask=valid_t, other=0.0)
    weighted_pair = pair_tile * beta_tile[:, None]
    grad_beta_tile = tl.zeros((C,), dtype=tl.float32)

    for block in range(0, tl.cdiv(R, BR)):
        coord = block * BR + tl.arange(0, BR)
        valid_r = coord < R
        source_offset = (
            ((batch * T + token[:, None]) * H + head) * R
            + coord[None, :]
        )
        source_tile = tl.load(
            source + source_offset,
            mask=valid_t[:, None] & valid_r[None, :],
            other=0.0,
        )
        gram_tile = tl.dot(source_tile, tl.trans(source_tile))
        grad_beta_tile += tl.sum(pair_tile * gram_tile, axis=1)
        grad_source_tile = tl.dot(
            weighted_pair.to(source_tile.dtype),
            source_tile,
        )
        grad_source_tile += tl.dot(
            tl.trans(weighted_pair).to(source_tile.dtype),
            source_tile,
        )
        if ACCUMULATE_GRAD_SOURCE:
            grad_source_tile += tl.load(
                grad_source_base + source_offset,
                mask=valid_t[:, None] & valid_r[None, :],
                other=0.0,
            ).to(tl.float32)
        tl.store(
            grad_source + source_offset,
            grad_source_tile,
            mask=valid_t[:, None] & valid_r[None, :],
        )
    if ACCUMULATE_GRAD_BETA:
        grad_beta_tile += tl.load(
            grad_beta + beta_offset, mask=valid_t, other=0.0
        ).to(tl.float32)
    tl.store(grad_beta + beta_offset, grad_beta_tile, mask=valid_t)


def residual_pair_backward(
    source,
    beta,
    grad_pair,
    *,
    chunk_size,
    grad_source_accumulator=None,
    grad_beta_accumulator=None,
    gradient_dtype=None,
):
    batch, length, heads, rank = source.shape
    gradient_dtype = torch.float32 if gradient_dtype is None else gradient_dtype
    grad_source = torch.empty_like(source, dtype=gradient_dtype)
    grad_source_base = (
        grad_source if grad_source_accumulator is None else grad_source_accumulator
    )
    grad_beta = (
        torch.empty_like(beta, dtype=torch.float32)
        if grad_beta_accumulator is None
        else grad_beta_accumulator
    )
    _residual_pair_bwd_kernel[(triton.cdiv(length, chunk_size), batch * heads)](
        source,
        beta,
        grad_pair,
        grad_source_base,
        grad_source,
        grad_beta,
        T=length,
        H=heads,
        R=rank,
        C=chunk_size,
        BR=64,
        ACCUMULATE_GRAD_SOURCE=grad_source_accumulator is not None,
        ACCUMULATE_GRAD_BETA=grad_beta_accumulator is not None,
        num_warps=4,
        num_stages=3,
    )
    return grad_source, grad_beta


@triton.heuristics({
    "USE_FINAL_STATE_GRADIENT": lambda args: args["grad_final"] is not None,
    "STORE_INITIAL_GRADIENT": lambda args: args["grad_initial"] is not None,
})
@triton.jit
def _residual_state_bwd_kernel(
    source,
    w,
    grad_final,
    grad_delta_input,
    grad_states,
    grad_initial,
    grad_update,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STORE_INITIAL_GRADIENT: tl.constexpr,
):
    rank_block = tl.program_id(0)
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    rank = rank_block * BR + tl.arange(0, BR)
    value = tl.arange(0, BV)
    valid_r = rank < R
    chunks = tl.cdiv(T, C)
    state_base = batch_head * R * V + rank[:, None] * V

    dstate_0 = tl.zeros((BR, BV), dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        dstate_0 += tl.load(
            grad_final + state_base + value[None, :],
            mask=valid_r[:, None] & (value[None, :] < V), other=0.0,
        ).to(tl.float32)
    if V > BV:
        dstate_1 = tl.zeros((BR, BV), dtype=tl.float32)
        if USE_FINAL_STATE_GRADIENT:
            dstate_1 += tl.load(
                grad_final + state_base + (BV + value)[None, :],
                mask=valid_r[:, None] & ((BV + value)[None, :] < V), other=0.0,
            ).to(tl.float32)
    if V > 2 * BV:
        dstate_2 = tl.zeros((BR, BV), dtype=tl.float32)
        if USE_FINAL_STATE_GRADIENT:
            dstate_2 += tl.load(
                grad_final + state_base + (2 * BV + value)[None, :],
                mask=valid_r[:, None] & ((2 * BV + value)[None, :] < V), other=0.0,
            ).to(tl.float32)
    if V > 3 * BV:
        dstate_3 = tl.zeros((BR, BV), dtype=tl.float32)
        if USE_FINAL_STATE_GRADIENT:
            dstate_3 += tl.load(
                grad_final + state_base + (3 * BV + value)[None, :],
                mask=valid_r[:, None] & ((3 * BV + value)[None, :] < V), other=0.0,
            ).to(tl.float32)

    lane = tl.arange(0, C)
    for reverse_chunk in range(0, chunks):
        chunk = chunks - 1 - reverse_chunk
        token = chunk * C + lane
        valid_t = token < T
        chunk_base = (
            ((batch * chunks + chunk) * H + head) * R * V
            + rank[:, None] * V
        )
        tl.store(
            grad_states + chunk_base + value[None, :], dstate_0,
            mask=valid_r[:, None] & (value[None, :] < V),
        )
        if V > BV:
            tl.store(
                grad_states + chunk_base + (BV + value)[None, :], dstate_1,
                mask=valid_r[:, None] & ((BV + value)[None, :] < V),
            )
        if V > 2 * BV:
            tl.store(
                grad_states + chunk_base + (2 * BV + value)[None, :], dstate_2,
                mask=valid_r[:, None] & ((2 * BV + value)[None, :] < V),
            )
        if V > 3 * BV:
            tl.store(
                grad_states + chunk_base + (3 * BV + value)[None, :], dstate_3,
                mask=valid_r[:, None] & ((3 * BV + value)[None, :] < V),
            )

        vector_base = ((batch * T + token[:, None]) * H + head) * V
        src_0 = tl.load(
            source + vector_base + value[None, :],
            mask=valid_t[:, None] & (value[None, :] < V), other=0.0,
        )
        dupdate = tl.dot(src_0, tl.trans(dstate_0).to(src_0.dtype))
        if V > BV:
            src_1 = tl.load(
                source + vector_base + (BV + value)[None, :],
                mask=valid_t[:, None] & ((BV + value)[None, :] < V), other=0.0,
            )
            dupdate += tl.dot(src_1, tl.trans(dstate_1).to(src_1.dtype))
        if V > 2 * BV:
            src_2 = tl.load(
                source + vector_base + (2 * BV + value)[None, :],
                mask=valid_t[:, None] & ((2 * BV + value)[None, :] < V), other=0.0,
            )
            dupdate += tl.dot(src_2, tl.trans(dstate_2).to(src_2.dtype))
        if V > 3 * BV:
            src_3 = tl.load(
                source + vector_base + (3 * BV + value)[None, :],
                mask=valid_t[:, None] & ((3 * BV + value)[None, :] < V), other=0.0,
            )
            dupdate += tl.dot(src_3, tl.trans(dstate_3).to(src_3.dtype))

        rank_base = ((batch * T + token[:, None]) * H + head) * R
        dupdate += tl.load(
            grad_delta_input + rank_base + rank[None, :],
            mask=valid_t[:, None] & valid_r[None, :], other=0.0,
        )
        tl.store(
            grad_update + rank_base + rank[None, :], dupdate,
            mask=valid_t[:, None] & valid_r[None, :],
        )

        w_0 = tl.load(
            w + vector_base + value[None, :],
            mask=valid_t[:, None] & (value[None, :] < V), other=0.0,
        )
        dstate_0 -= tl.dot(tl.trans(dupdate).to(w_0.dtype), w_0)
        if V > BV:
            w_1 = tl.load(
                w + vector_base + (BV + value)[None, :],
                mask=valid_t[:, None] & ((BV + value)[None, :] < V), other=0.0,
            )
            dstate_1 -= tl.dot(tl.trans(dupdate).to(w_1.dtype), w_1)
        if V > 2 * BV:
            w_2 = tl.load(
                w + vector_base + (2 * BV + value)[None, :],
                mask=valid_t[:, None] & ((2 * BV + value)[None, :] < V), other=0.0,
            )
            dstate_2 -= tl.dot(tl.trans(dupdate).to(w_2.dtype), w_2)
        if V > 3 * BV:
            w_3 = tl.load(
                w + vector_base + (3 * BV + value)[None, :],
                mask=valid_t[:, None] & ((3 * BV + value)[None, :] < V), other=0.0,
            )
            dstate_3 -= tl.dot(tl.trans(dupdate).to(w_3.dtype), w_3)

    if STORE_INITIAL_GRADIENT:
        tl.store(
            grad_initial + state_base + value[None, :], dstate_0,
            mask=valid_r[:, None] & (value[None, :] < V),
        )
        if V > BV:
            tl.store(
                grad_initial + state_base + (BV + value)[None, :], dstate_1,
                mask=valid_r[:, None] & ((BV + value)[None, :] < V),
            )
        if V > 2 * BV:
            tl.store(
                grad_initial + state_base + (2 * BV + value)[None, :], dstate_2,
                mask=valid_r[:, None] & ((2 * BV + value)[None, :] < V),
            )
        if V > 3 * BV:
            tl.store(
                grad_initial + state_base + (3 * BV + value)[None, :], dstate_3,
                mask=valid_r[:, None] & ((3 * BV + value)[None, :] < V),
            )


def residual_state_backward(
    source,
    w,
    grad_delta,
    grad_final,
    *,
    chunk_size,
    initial_state_required,
):
    batch, length, heads, value_dim = source.shape
    rank = grad_delta.shape[-1]
    chunks = triton.cdiv(length, chunk_size)
    grad_states = torch.empty(
        batch, chunks, heads, rank, value_dim,
        dtype=source.dtype, device=source.device,
    )
    grad_update = torch.empty_like(grad_delta)
    grad_initial = (
        torch.empty(
            batch, heads, rank, value_dim,
            dtype=torch.float32, device=source.device,
        )
        if initial_state_required
        else None
    )
    block_rank = min(32, triton.next_power_of_2(rank))
    _residual_state_bwd_kernel[(triton.cdiv(rank, block_rank), batch * heads)](
        source, w, grad_final, grad_delta, grad_states, grad_initial, grad_update,
        T=length, H=heads, R=rank, V=value_dim, C=chunk_size, BR=block_rank, BV=64,
        num_warps=4, num_stages=3,
    )
    return grad_states, grad_initial, grad_update


def residual_fwd(target, source, beta, initial_state, *, output_final_state, chunk_size):
    # The residual predictor has no vector gate.  Use FLA's mature ungated
    # pair owner directly; passing an all-zero gk selects the slower gated
    # sub-block schedule even though the resulting matrix is identical.
    interaction = chunk_scaled_dot_kkt_fwd(
        k=source,
        beta=beta,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    interaction = solve_tril(A=interaction, output_dtype=target.dtype)
    w, update = recompute_residual_wy(
        target,
        source,
        beta,
        interaction,
        chunk_size=chunk_size,
    )
    states, update_direction, final_state = chunk_oja_fwd_h(
        v=source,
        w=w,
        u=update,
        gv=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        save_new_key=True,
        chunk_size=chunk_size,
    )
    return update_direction, final_state, interaction


class _OjaResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        target,
        source,
        beta,
        initial_state,
        chunk_size,
        output_final_state,
    ):
        update_direction, final_state, interaction = residual_fwd(
            target,
            source,
            beta,
            initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
        )
        ctx.chunk_size = chunk_size
        ctx.has_initial_state = initial_state is not None
        ctx.set_materialize_grads(False)
        saved_initial = target.new_empty(0) if initial_state is None else initial_state
        ctx.save_for_backward(target, source, beta, saved_initial, interaction)
        return update_direction, final_state

    @staticmethod
    def backward(ctx, grad_residual, grad_final_state):
        target, source, beta, saved_initial, interaction = ctx.saved_tensors
        initial_state = saved_initial if ctx.has_initial_state else None
        if grad_residual is None:
            grad_residual = torch.zeros_like(target)
        w, update = recompute_residual_wy(
            target,
            source,
            beta,
            interaction,
            chunk_size=ctx.chunk_size,
        )
        states, residual, _ = chunk_oja_fwd_h(
            v=source,
            w=w,
            u=update,
            gv=None,
            initial_state=initial_state,
            output_final_state=False,
            save_new_key=True,
            chunk_size=ctx.chunk_size,
        )

        grad_states, grad_initial, grad_update = residual_state_backward(
            source,
            w,
            grad_residual,
            grad_final_state,
            chunk_size=ctx.chunk_size,
            initial_state_required=ctx.has_initial_state,
        )
        grad_source, grad_w = residual_source_state_backward(
            residual,
            source,
            states,
            grad_states,
            grad_update,
            chunk_size=ctx.chunk_size,
        )
        (
            grad_target,
            grad_source_wy,
            grad_beta,
            grad_pair,
        ) = residual_wy_backward(
            target,
            source,
            beta,
            interaction,
            grad_w,
            grad_update,
            grad_source_accumulator=grad_source,
        )
        grad_source, grad_beta = residual_pair_backward(
            source,
            beta,
            grad_pair,
            chunk_size=ctx.chunk_size,
            grad_source_accumulator=grad_source,
            grad_beta_accumulator=grad_beta,
            gradient_dtype=source.dtype,
        )
        return (
            grad_target,
            grad_source,
            grad_beta,
            grad_initial,
            None,
            None,
        )


def oja_residual(
    target,
    source,
    beta,
    initial_state,
    *,
    chunk_size=32,
    output_final_state=True,
):
    return _OjaResidual.apply(
        target,
        source,
        beta,
        initial_state,
        chunk_size,
        output_final_state,
    )


__all__ = ["oja_residual"]
