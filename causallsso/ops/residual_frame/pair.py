# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
# Adapted from FLA's MIT-licensed unbounded generalized-DPLR chunk_A owners.
"""Exact unbounded direct-e pair owner for the Residual-Frame exterior.

The generic DPLR owner forms four interaction matrices. Residual-Frame has
``k == b == d`` and the effective source
``a == -exp(log_decay) * e``, so only ``A_qd`` and ``A_ed`` are physical.
The current decay factor is supplied by the inclusive prefix rather than a
materialized ``a`` panel. The kernels retain FLA's exact causal scalar
exponent differences while loading the producer-native rectangular panels
directly. No centered Tensor-Core factorization is legal because coordinate
decay is unbounded.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp2, gather
from fla.utils import IS_GATHER_SUPPORTED, autotune_cache_kwargs


_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=num_warps, num_stages=num_stages)
    for num_warps in (2, 4, 8)
    for num_stages in (2, 3)
]

_BWD_CONFIGS = [triton.Config({}, num_warps=4, num_stages=2)]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["K", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _direct_e_pair_fwd_kernel(
    d,
    paired,
    gi,
    qg,
    dtail,
    ag,
    Aqd,
    Aed,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    FC: tl.constexpr,
    NP: tl.constexpr,
    BK: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_b = tl.program_id(1).to(tl.int64)
    i_h = tl.program_id(2).to(tl.int64)

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]

    panel = (i_b * H + i_h) * NP + o_t // FC
    row = o_t % FC
    p_d = d + panel[:, None] * (FC * K) + row[:, None] * K + o_k[None, :]
    p_e = (
        paired
        + panel[:, None] * (2 * FC * K)
        + row[:, None] * K
        + o_k[None, :]
    )
    p_q = p_e + FC * K

    token_base = (i_b * T + o_t) * (H * K) + i_h * K
    p_gi = gi + token_base[:, None] + o_k[None, :]

    b_d = tl.load(p_d, mask=m_tk, other=0.0)
    b_e = tl.load(p_e, mask=m_tk, other=0.0)
    b_q = tl.load(p_q, mask=m_tk, other=0.0)
    b_a = -b_e
    b_gi = tl.load(p_gi, mask=m_tk, other=0.0).to(tl.float32)

    last = min((i_t + 1) * BT, T) - 1
    p_last = gi + (i_b * T + last) * (H * K) + i_h * K + o_k
    b_last = tl.load(p_last, mask=m_k, other=0.0).to(tl.float32)

    p_qg = qg + token_base[:, None] + o_k[None, :]
    p_ag = ag + token_base[:, None] + o_k[None, :]
    p_dtail = dtail + token_base[:, None] + o_k[None, :]
    tl.store(
        p_qg,
        (b_q * exp2(b_gi)).to(
            p_qg.dtype.element_ty, fp_downcast_rounding="rtne"
        ),
        mask=m_tk,
    )
    tl.store(
        p_ag,
        (b_a * exp2(b_gi)).to(
            p_ag.dtype.element_ty, fp_downcast_rounding="rtne"
        ),
        mask=m_tk,
    )
    tl.store(
        p_dtail,
        (b_d * exp2(b_last[None, :] - b_gi)).to(
            p_dtail.dtype.element_ty, fp_downcast_rounding="rtne"
        ),
        mask=m_tk,
    )

    out_base = (i_b * T + o_t) * (H * BT) + i_h * BT
    valid_len = min(T - i_t * BT, BT)
    for j in range(0, BT):
        if GATHER_SUPPORTED:
            row_index = tl.full((1, BK), j, tl.int16)
            b_dj = gather(b_d, row_index, axis=0)
            b_gij = gather(b_gi, row_index, axis=0)
        else:
            row_mask = o_i == j
            b_dj = tl.sum(tl.where(row_mask[:, None], b_d, 0.0), 0)[None, :]
            b_gij = tl.sum(tl.where(row_mask[:, None], b_gi, 0.0), 0)[None, :]

        inclusive = (o_i[:, None] >= j) & (j < valid_len)
        strict = (o_i[:, None] > j) & (j < valid_len)
        q_exp = exp2(tl.where(inclusive, b_gi - b_gij, float("-inf")))
        e_exp = exp2(tl.where(strict, b_gi - b_gij, float("-inf")))
        b_Aqd = tl.sum(b_q * b_dj * q_exp, 1)
        b_Aed = tl.sum(b_a * b_dj * e_exp, 1)

        p_Aqd = Aqd + out_base + j
        p_Aed = Aed + out_base + j
        tl.store(
            p_Aqd,
            b_Aqd.to(p_Aqd.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=m_t,
        )
        tl.store(p_Aed, b_Aed, mask=m_t)


@triton.autotune(
    configs=_BWD_CONFIGS,
    key=["K", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _direct_e_pair_bwd_kernel(
    d,
    paired,
    gi,
    dAqd,
    dAed0,
    dAed1,
    dqg,
    ddtail,
    dag,
    dg_tail,
    dd,
    dpaired,
    dg,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    FC: tl.constexpr,
    NP: tl.constexpr,
    BK: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    o_k = i_k * BK + tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]

    panel = i_bh * NP + o_t // FC
    row = o_t % FC
    p_d = d + panel[:, None] * (FC * K) + row[:, None] * K + o_k[None, :]
    p_e = (
        paired
        + panel[:, None] * (2 * FC * K)
        + row[:, None] * K
        + o_k[None, :]
    )
    p_q = p_e + FC * K
    token_base = (i_b * T + o_t) * (H * K) + i_h * K
    p_gi = gi + token_base[:, None] + o_k[None, :]

    b_d = tl.load(p_d, mask=m_tk, other=0.0)
    b_e = tl.load(p_e, mask=m_tk, other=0.0)
    b_q = tl.load(p_q, mask=m_tk, other=0.0)
    b_a = -b_e
    b_gi = tl.load(p_gi, mask=m_tk, other=0.0).to(tl.float32)

    o_j = tl.arange(0, BT)
    valid_len = min(T - i_t * BT, BT)
    m_A = m_t[:, None] & (o_j[None, :] < valid_len)
    A_base = (i_b * T + o_t) * (H * BT) + i_h * BT
    p_dAqd = dAqd + A_base[:, None] + o_j[None, :]
    p_dAed0 = dAed0 + A_base[:, None] + o_j[None, :]
    p_dAed1 = dAed1 + A_base[:, None] + o_j[None, :]
    b_dAqd = tl.load(p_dAqd, mask=m_A, other=0.0).to(tl.float32)
    b_dAed = (
        tl.load(p_dAed0, mask=m_A, other=0.0).to(tl.float32)
        + tl.load(p_dAed1, mask=m_A, other=0.0).to(tl.float32)
    )

    b_dq = tl.zeros((BT, BK), tl.float32)
    b_da = tl.zeros((BT, BK), tl.float32)
    b_dd = tl.zeros((BT, BK), tl.float32)
    for j in range(0, BT):
        if GATHER_SUPPORTED:
            row_index_k = tl.full((1, BK), j, tl.int16)
            col_index = tl.full((BT, 1), j, tl.int16)
            row_index_A = tl.full((1, BT), j, tl.int16)
            b_dj = gather(b_d, row_index_k, axis=0)
            b_gij = gather(b_gi, row_index_k, axis=0)
            b_qj = gather(b_q, row_index_k, axis=0)
            b_aj = gather(b_a, row_index_k, axis=0)
            b_dAqd_col = gather(b_dAqd, col_index, axis=1)
            b_dAed_col = gather(b_dAed, col_index, axis=1)
            b_dAqd_row = tl.sum(
                gather(b_dAqd, row_index_A, axis=0), 0
            )[:, None]
            b_dAed_row = tl.sum(
                gather(b_dAed, row_index_A, axis=0), 0
            )[:, None]
        else:
            row_mask = o_i == j
            b_dj = tl.sum(tl.where(row_mask[:, None], b_d, 0.0), 0)[None, :]
            b_gij = tl.sum(tl.where(row_mask[:, None], b_gi, 0.0), 0)[None, :]
            b_qj = tl.sum(tl.where(row_mask[:, None], b_q, 0.0), 0)[None, :]
            b_aj = tl.sum(tl.where(row_mask[:, None], b_a, 0.0), 0)[None, :]
            b_dAqd_col = tl.sum(
                tl.where(row_mask[None, :], b_dAqd, 0.0), 1
            )[:, None]
            b_dAed_col = tl.sum(
                tl.where(row_mask[None, :], b_dAed, 0.0), 1
            )[:, None]
            b_dAqd_row = tl.sum(
                tl.where(row_mask[:, None], b_dAqd, 0.0), 0
            )[:, None]
            b_dAed_row = tl.sum(
                tl.where(row_mask[:, None], b_dAed, 0.0), 0
            )[:, None]

        inclusive_col = (o_i[:, None] >= j) & (j < valid_len)
        strict_col = (o_i[:, None] > j) & (j < valid_len)
        b_dq += b_dAqd_col * b_dj * exp2(
            tl.where(inclusive_col, b_gi - b_gij, float("-inf"))
        )
        b_da += b_dAed_col * b_dj * exp2(
            tl.where(strict_col, b_gi - b_gij, float("-inf"))
        )

        inclusive_row = (o_i[:, None] <= j) & (j < valid_len)
        strict_row = (o_i[:, None] < j) & (j < valid_len)
        b_dd += b_dAqd_row * b_qj * exp2(
            tl.where(inclusive_row, b_gij - b_gi, float("-inf"))
        )
        b_dd += b_dAed_row * b_aj * exp2(
            tl.where(strict_row, b_gij - b_gi, float("-inf"))
        )

    p_dqg = dqg + token_base[:, None] + o_k[None, :]
    p_dag = dag + token_base[:, None] + o_k[None, :]
    p_ddtail = ddtail + token_base[:, None] + o_k[None, :]
    last = min((i_t + 1) * BT, T) - 1
    p_last = gi + (i_b * T + last) * (H * K) + i_h * K + o_k
    b_last = tl.load(p_last, mask=m_k, other=0.0).to(tl.float32)

    b_dq += (
        tl.load(p_dqg, mask=m_tk, other=0.0).to(tl.float32) * exp2(b_gi)
    )
    b_da += (
        tl.load(p_dag, mask=m_tk, other=0.0).to(tl.float32) * exp2(b_gi)
    )
    b_dd += tl.load(p_ddtail, mask=m_tk, other=0.0).to(
        tl.float32
    ) * exp2(b_last[None, :] - b_gi)

    p_dd = dd + panel[:, None] * (FC * K) + row[:, None] * K + o_k[None, :]
    p_de = (
        dpaired
        + panel[:, None] * (2 * FC * K)
        + row[:, None] * K
        + o_k[None, :]
    )
    p_dq = p_de + FC * K
    tl.store(
        p_dd,
        b_dd.to(p_dd.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=m_tk,
    )
    tl.store(
        p_de,
        (-b_da).to(p_de.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=m_tk,
    )
    tl.store(
        p_dq,
        b_dq.to(p_dq.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=m_tk,
    )

    # The erase action consumes the already-decayed state, so its effective
    # DPLR source is ``-e * exp(log_decay)``. Using the inclusive prefix here
    # closes that current-token decay dependency in the same reverse cumsum.
    b_dprefix = b_dq * b_q + b_da * b_a - b_dd * b_d
    NT = tl.cdiv(T, BT)
    p_tail = dg_tail + ((i_b * NT + i_t) * H + i_h) * K + o_k
    b_tail = tl.load(p_tail, mask=m_k, other=0.0).to(tl.float32)
    b_dg = tl.cumsum(b_dprefix, axis=0, reverse=True)
    b_dg += b_tail[None, :]
    p_dg = dg + token_base[:, None] + o_k[None, :]
    tl.store(p_dg, b_dg, mask=m_tk)


def _validate_panels(
    d: torch.Tensor,
    paired: torch.Tensor,
    inclusive: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    panels, edits, frame_chunk, rank = d.shape
    batch, length, heads, gate_rank = inclusive.shape
    if edits != 1 or paired.shape != (panels, 2, frame_chunk, rank):
        raise ValueError(
            "direct-e production panels require one edit and two paired routes"
        )
    if gate_rank != rank or length % frame_chunk != 0:
        raise ValueError("decay and panel layouts do not describe the same token grid")
    frame_chunks = length // frame_chunk
    if panels != batch * heads * frame_chunks:
        raise ValueError("frame panel count does not match rectangular layout")
    return batch, length, heads, rank, frame_chunk, frame_chunks


def direct_e_pair_forward(
    d: torch.Tensor,
    paired: torch.Tensor,
    inclusive: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length, heads, rank, frame_chunk, frame_chunks = _validate_panels(
        d, paired, inclusive
    )
    # The source-native panels are bounded FP16, but multiplying them by the
    # exact unbounded decay factors requires BF16 exponent range downstream.
    q_scaled = torch.empty_like(inclusive, dtype=torch.bfloat16)
    d_tail = torch.empty_like(q_scaled)
    e_scaled = torch.empty_like(q_scaled)
    A_qd = torch.empty(
        batch, length, heads, chunk_size, dtype=torch.bfloat16, device=d.device
    )
    A_ed = torch.empty(
        batch, length, heads, chunk_size, dtype=torch.float32, device=d.device
    )
    grid = (triton.cdiv(length, chunk_size), batch, heads)
    _direct_e_pair_fwd_kernel[grid](
        d,
        paired,
        inclusive,
        q_scaled,
        d_tail,
        e_scaled,
        A_qd,
        A_ed,
        T=length,
        H=heads,
        K=rank,
        BT=chunk_size,
        FC=frame_chunk,
        NP=frame_chunks,
        BK=triton.next_power_of_2(rank),
        GATHER_SUPPORTED=IS_GATHER_SUPPORTED,
    )
    return A_qd, A_ed, q_scaled, d_tail, e_scaled


def direct_e_pair_backward(
    d: torch.Tensor,
    paired: torch.Tensor,
    inclusive: torch.Tensor,
    dA_qd: torch.Tensor,
    dA_ed_0: torch.Tensor,
    dA_ed_1: torch.Tensor,
    dq_scaled: torch.Tensor,
    dd_tail: torch.Tensor,
    de_scaled: torch.Tensor,
    dg_tail: torch.Tensor,
    *,
    chunk_size: int,
    gradient_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length, heads, rank, frame_chunk, frame_chunks = _validate_panels(
        d, paired, inclusive
    )
    gradient_dtype = d.dtype if gradient_dtype is None else gradient_dtype
    dd = torch.empty_like(d, dtype=gradient_dtype)
    dpaired = torch.empty_like(paired, dtype=gradient_dtype)
    dg = torch.empty_like(inclusive, dtype=torch.float32)
    block_rank = min(64, triton.next_power_of_2(rank))
    grid = (
        triton.cdiv(rank, block_rank),
        triton.cdiv(length, chunk_size),
        batch * heads,
    )
    _direct_e_pair_bwd_kernel[grid](
        d,
        paired,
        inclusive,
        dA_qd,
        dA_ed_0,
        dA_ed_1,
        dq_scaled,
        dd_tail,
        de_scaled,
        dg_tail,
        dd,
        dpaired,
        dg,
        T=length,
        H=heads,
        K=rank,
        BT=chunk_size,
        FC=frame_chunk,
        NP=frame_chunks,
        BK=block_rank,
        GATHER_SUPPORTED=IS_GATHER_SUPPORTED,
    )
    return dd, dpaired, dg


__all__ = ["direct_e_pair_forward", "direct_e_pair_backward"]
