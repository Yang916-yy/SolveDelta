# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
"""Leaky-LMS specialization of FLA's gated-Oja WY owners.

The production recurrence evaluates the residual before coordinatewise
forgetting. Its chunk system therefore uses the exclusive vector-retention
prefix on the source branch and ``gamma*h`` on the target branch. These owners
keep FLA's tile schedule and strict transpose while consuming those stable
operands directly; neither a decay-compensated target nor ``gamma/a`` is
formed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import check_shared_mem


@triton.jit
def _close_vector_gate_bwd_kernel(
    grad_output,
    grad_wy,
    grad_last,
    grad_gate_pair,
    grad_beta_wy,
    grad_beta_pair,
    grad_log_decay_wy,
    grad_log_decay_pair,
    grad_beta,
    grad_log_decay,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    BR: tl.constexpr,
):
    coord_block = tl.program_id(0)
    chunk = tl.program_id(1).to(tl.int64)
    batch_head = tl.program_id(2).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    token = chunk * C + tl.arange(0, C)
    coord = coord_block * BR + tl.arange(0, BR)
    mask = (token[:, None] < T) & (coord[None, :] < R)
    offset = (
        ((batch * T + token[:, None]) * H + head) * R
        + coord[None, :]
    )
    value = tl.load(grad_output + offset, mask=mask, other=0.0).to(
        tl.float32
    )
    value += tl.load(grad_wy + offset, mask=mask, other=0.0).to(
        tl.float32
    )
    value = tl.cumsum(value, axis=0, reverse=True)
    value += tl.load(grad_last + offset, mask=mask, other=0.0).to(
        tl.float32
    )
    pair_value = tl.load(
        grad_gate_pair + offset, mask=mask, other=0.0
    ).to(tl.float32)
    value += tl.cumsum(pair_value, axis=0, reverse=True)
    value += tl.load(
        grad_log_decay_wy + offset, mask=mask, other=0.0
    ).to(tl.float32)
    value += tl.load(
        grad_log_decay_pair + offset, mask=mask, other=0.0
    ).to(tl.float32)
    tl.store(grad_log_decay + offset, value, mask=mask)

    if coord_block == 0:
        beta_offset = (batch * T + token) * H + head
        beta_value = tl.load(
            grad_beta_wy + beta_offset, mask=token < T, other=0.0
        ).to(tl.float32)
        beta_value += tl.load(
            grad_beta_pair + beta_offset, mask=token < T, other=0.0
        ).to(tl.float32)
        tl.store(grad_beta + beta_offset, beta_value, mask=token < T)


@triton.jit
def _merge_source_cotangents_kernel(
    first,
    second,
    third,
    output,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N
    value = tl.load(first + offset, mask=mask, other=0.0).to(tl.float32)
    value += tl.load(second + offset, mask=mask, other=0.0).to(tl.float32)
    value += tl.load(third + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(output + offset, value.to(output.dtype.element_ty), mask=mask)


def merge_source_cotangents(
    first: torch.Tensor,
    second: torch.Tensor,
    third: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Close three or four final-shaped source cotangents in one owner."""
    output = torch.empty_like(first, dtype=output_dtype)
    elements = first.numel()
    _merge_source_cotangents_kernel[(triton.cdiv(elements, 256),)](
        first,
        second,
        third,
        output,
        N=elements,
        BLOCK=256,
        num_warps=4,
    )
    return output


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in (2, 4, 8)
        for num_stages in (2, 3, 4)
    ],
    key=["H", "K", "V", "BT", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def _recompute_leaky_w_u_fwd_kernel(
    target,
    source,
    source_gated,
    beta,
    log_decay,
    w,
    update,
    A,
    gate,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TARGET_STRIDE_B: tl.constexpr,
    TARGET_STRIDE_T: tl.constexpr,
    TARGET_STRIDE_H: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    token = chunk * BT + tl.arange(0, BT)
    valid_t = token < T

    scalar_offset = (batch * T + token) * H + head
    b_beta = tl.load(
        beta + scalar_offset, mask=valid_t, other=0.0
    ).to(tl.float32)
    row = tl.arange(0, BT)
    pair_offset = (
        ((batch * T + token[:, None]) * H + head) * BT
        + row[None, :]
    )
    pair = tl.load(A + pair_offset, mask=valid_t[:, None], other=0.0)

    for block in range(0, tl.cdiv(V, BV)):
        coord = block * BV + tl.arange(0, BV)
        valid_v = coord < V
        offset = (
            ((batch * T + token[:, None]) * H + head) * V
            + coord[None, :]
        )
        source_tile = tl.load(
            source + offset,
            mask=valid_t[:, None] & valid_v[None, :],
            other=0.0,
        )
        gate_tile = tl.load(
            gate + offset,
            mask=valid_t[:, None] & valid_v[None, :],
            other=0.0,
        ).to(tl.float32)
        log_decay_tile = tl.load(
            log_decay + offset,
            mask=valid_t[:, None] & valid_v[None, :],
            other=0.0,
        ).to(tl.float32)
        weighted_source = (
            source_tile
            * b_beta[:, None]
            * exp(gate_tile - log_decay_tile)
        )
        w_tile = tl.dot(pair, weighted_source.to(source_tile.dtype))
        tl.store(
            w + offset,
            w_tile,
            mask=valid_t[:, None] & valid_v[None, :],
        )

        last_token = min(chunk * BT + BT, T) - 1
        last_offset = ((batch * T + last_token) * H + head) * V + coord
        gate_last = tl.load(
            gate + last_offset, mask=valid_v, other=0.0
        ).to(tl.float32)
        source_at_boundary = source_tile * exp(gate_last[None, :] - gate_tile)
        tl.store(
            source_gated + offset,
            source_at_boundary,
            mask=valid_t[:, None] & valid_v[None, :],
        )

    for block in range(0, tl.cdiv(K, BK)):
        coord = block * BK + tl.arange(0, BK)
        valid_k = coord < K
        target_offset = (
            batch * TARGET_STRIDE_B
            + token[:, None] * TARGET_STRIDE_T
            + head * TARGET_STRIDE_H
            + coord[None, :]
        )
        update_offset = (
            ((batch * T + token[:, None]) * H + head) * K
            + coord[None, :]
        )
        target_tile = tl.load(
            target + target_offset,
            mask=valid_t[:, None] & valid_k[None, :],
            other=0.0,
        )
        weighted_target = (target_tile * b_beta[:, None]).to(
            target_tile.dtype
        )
        update_tile = tl.dot(pair, weighted_target, allow_tf32=False)
        tl.store(
            update + update_offset,
            update_tile,
            mask=valid_t[:, None] & valid_k[None, :],
        )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in (2, 4)
        for num_stages in (2, 3, 4)
    ],
    key=["H", "K", "V", "BT", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def _prepare_leaky_wy_bwd_kernel(
    target,
    source,
    beta,
    log_decay,
    gate,
    A,
    grad_A,
    grad_w,
    grad_update,
    grad_target,
    grad_source,
    grad_beta,
    grad_gate,
    grad_log_decay,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TARGET_STRIDE_B: tl.constexpr,
    TARGET_STRIDE_T: tl.constexpr,
    TARGET_STRIDE_H: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    token = chunk * BT + tl.arange(0, BT)
    valid_t = token < T
    row = tl.arange(0, BT)

    scalar_offset = (batch * T + token) * H + head
    b_beta = tl.load(
        beta + scalar_offset, mask=valid_t, other=0.0
    ).to(tl.float32)
    pair_offset = (
        ((batch * T + token[None, :]) * H + head) * BT
        + row[:, None]
    )
    pair = tl.load(A + pair_offset, mask=valid_t[None, :], other=0.0)
    grad_pair = tl.zeros((BT, BT), dtype=tl.float32)
    grad_beta_tile = tl.zeros((BT,), dtype=tl.float32)

    for block in range(0, tl.cdiv(V, BV)):
        coord = block * BV + tl.arange(0, BV)
        valid_v = coord < V
        offset = (
            ((batch * T + token[:, None]) * H + head) * V
            + coord[None, :]
        )
        mask = valid_t[:, None] & valid_v[None, :]
        source_tile = tl.load(source + offset, mask=mask, other=0.0)
        log_decay_tile = tl.load(
            log_decay + offset, mask=mask, other=0.0
        ).to(tl.float32)
        gate_exp = exp(
            tl.load(gate + offset, mask=mask, other=0.0).to(tl.float32)
            - log_decay_tile
        )
        weighted_source = source_tile * b_beta[:, None] * gate_exp
        grad_w_tile = tl.load(grad_w + offset, mask=mask, other=0.0)

        grad_pair += tl.dot(
            grad_w_tile, tl.trans(weighted_source).to(grad_w_tile.dtype)
        )
        grad_weighted = tl.dot(pair, grad_w_tile)
        grad_source_tile = grad_weighted * gate_exp * b_beta[:, None]
        grad_beta_tile += tl.sum(
            grad_weighted * source_tile * gate_exp, axis=1
        )
        grad_gate_tile = grad_weighted * weighted_source
        tl.store(grad_source + offset, grad_source_tile, mask=mask)
        tl.store(grad_gate + offset, grad_gate_tile, mask=mask)
        tl.store(grad_log_decay + offset, -grad_gate_tile, mask=mask)

    for block in range(0, tl.cdiv(K, BK)):
        coord = block * BK + tl.arange(0, BK)
        valid_k = coord < K
        target_offset = (
            batch * TARGET_STRIDE_B
            + token[:, None] * TARGET_STRIDE_T
            + head * TARGET_STRIDE_H
            + coord[None, :]
        )
        contiguous_offset = (
            ((batch * T + token[:, None]) * H + head) * K
            + coord[None, :]
        )
        mask = valid_t[:, None] & valid_k[None, :]
        target_tile = tl.load(target + target_offset, mask=mask, other=0.0)
        weighted_target = (target_tile * b_beta[:, None]).to(
            target_tile.dtype
        )
        grad_update_tile = tl.load(
            grad_update + contiguous_offset, mask=mask, other=0.0
        )

        grad_pair += tl.dot(
            grad_update_tile, tl.trans(weighted_target)
        )
        grad_weighted = tl.dot(pair, grad_update_tile)
        grad_target_tile = grad_weighted * b_beta[:, None]
        grad_beta_tile += tl.sum(grad_weighted * target_tile, axis=1)
        tl.store(grad_target + contiguous_offset, grad_target_tile, mask=mask)

    strict = (token[:, None] > token[None, :]) & (
        valid_t[:, None] & valid_t[None, :]
    )
    grad_pair = tl.where(strict, grad_pair, 0.0)
    grad_pair = tl.dot(grad_pair.to(pair.dtype), pair)
    grad_pair = tl.dot(pair, grad_pair.to(pair.dtype))
    grad_pair = tl.where(strict, -grad_pair, 0.0)

    output_pair_offset = (
        ((batch * T + token[:, None]) * H + head) * BT
        + row[None, :]
    )
    tl.store(
        grad_A + output_pair_offset,
        grad_pair,
        mask=valid_t[:, None],
    )
    tl.store(grad_beta + scalar_offset, grad_beta_tile, mask=valid_t)


def recompute_leaky_w_u_fwd(
    target: torch.Tensor,
    source: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    A: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length, heads, target_width = target.shape
    source_width = source.shape[-1]
    chunk_size = A.shape[-1]
    chunks = triton.cdiv(length, chunk_size)
    block_target = 64
    block_source = 64
    w = torch.empty_like(source)
    update = torch.empty(
        target.shape, dtype=target.dtype, device=target.device
    )
    source_gated = torch.empty_like(source)
    _recompute_leaky_w_u_fwd_kernel[(chunks, batch * heads)](
        target,
        source,
        source_gated,
        beta,
        log_decay,
        w,
        update,
        A,
        gate,
        T=length,
        H=heads,
        K=target_width,
        V=source_width,
        BT=chunk_size,
        BK=block_target,
        BV=block_source,
        TARGET_STRIDE_B=target.stride(0),
        TARGET_STRIDE_T=target.stride(1),
        TARGET_STRIDE_H=target.stride(2),
    )
    return w, update, source_gated


def close_vector_gate_backward(
    grad_output: torch.Tensor,
    grad_wy: torch.Tensor,
    grad_last: torch.Tensor,
    grad_gate_pair: torch.Tensor,
    grad_beta_wy: torch.Tensor,
    grad_beta_pair: torch.Tensor,
    grad_log_decay_wy: torch.Tensor,
    grad_log_decay_pair: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if any(
        value.shape != grad_output.shape
        for value in (
            grad_wy,
            grad_last,
            grad_gate_pair,
            grad_log_decay_wy,
            grad_log_decay_pair,
        )
    ):
        raise ValueError("vector gate cotangents must have matching shapes")
    batch, length, heads, rank = grad_output.shape
    grad_beta = torch.empty_like(grad_beta_wy, dtype=torch.float32)
    grad_log_decay = torch.empty_like(grad_output, dtype=torch.float32)
    block_rank = min(64, triton.next_power_of_2(rank))
    _close_vector_gate_bwd_kernel[
        (
            triton.cdiv(rank, block_rank),
            triton.cdiv(length, chunk_size),
            batch * heads,
        )
    ](
        grad_output,
        grad_wy,
        grad_last,
        grad_gate_pair,
        grad_beta_wy,
        grad_beta_pair,
        grad_log_decay_wy,
        grad_log_decay_pair,
        grad_beta,
        grad_log_decay,
        T=length,
        H=heads,
        R=rank,
        C=chunk_size,
        BR=block_rank,
        num_warps=4,
        num_stages=2,
    )
    return grad_beta, grad_log_decay


def prepare_leaky_wy_bwd(
    target: torch.Tensor,
    source: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    A: torch.Tensor,
    grad_w: torch.Tensor,
    grad_update: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch, length, heads, target_width = target.shape
    source_width = source.shape[-1]
    chunk_size = A.shape[-1]
    chunks = triton.cdiv(length, chunk_size)
    tiling = 64 if check_shared_mem() else 32
    block_target = min(
        max(triton.next_power_of_2(target_width), 16), tiling
    )
    block_source = min(
        max(triton.next_power_of_2(source_width), 16), tiling
    )

    grad_target = torch.empty(
        target.shape, dtype=target.dtype, device=target.device
    )
    grad_source = torch.empty_like(source, dtype=torch.float32)
    grad_beta = torch.empty_like(beta, dtype=torch.float32)
    grad_gate = torch.empty_like(gate, dtype=torch.float32)
    grad_log_decay = torch.empty_like(log_decay, dtype=torch.float32)
    grad_A = torch.empty_like(A, dtype=torch.float32)
    _prepare_leaky_wy_bwd_kernel[(chunks, batch * heads)](
        target,
        source,
        beta,
        log_decay,
        gate,
        A,
        grad_A,
        grad_w,
        grad_update,
        grad_target,
        grad_source,
        grad_beta,
        grad_gate,
        grad_log_decay,
        T=length,
        H=heads,
        K=target_width,
        V=source_width,
        BT=chunk_size,
        BK=block_target,
        BV=block_source,
        TARGET_STRIDE_B=target.stride(0),
        TARGET_STRIDE_T=target.stride(1),
        TARGET_STRIDE_H=target.stride(2),
    )
    return (
        grad_target,
        grad_source,
        grad_beta,
        grad_gate,
        grad_log_decay,
        grad_A,
    )


__all__ = [
    "close_vector_gate_backward",
    "merge_source_cotangents",
    "prepare_leaky_wy_bwd",
    "recompute_leaky_w_u_fwd",
]
