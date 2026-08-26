# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source is adapted from flash-linear-attention's MIT-licensed DPLR
# chunk implementation at commit bc3b101dcb713ddc5bd8924b66754eb68b5ccf89.
# SolveDelta specializes the generic (q, k, a, b) interface under
# q=chi, k=b=d, and a=-e*exp(g), reducing four pair matrices to two.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.generalized_delta_rule.dplr.chunk_h_bwd import chunk_dplr_bwd_dhu
from fla.ops.generalized_delta_rule.dplr.chunk_h_fwd import chunk_dplr_fwd_h
from fla.ops.generalized_delta_rule.dplr.chunk_o_bwd import (
    chunk_dplr_bwd_dAu,
    chunk_dplr_bwd_dv,
    chunk_dplr_bwd_o,
)
from fla.ops.generalized_delta_rule.dplr.chunk_o_fwd import chunk_dplr_fwd_o
from fla.ops.generalized_delta_rule.dplr.wy_fast_bwd import chunk_dplr_bwd_wy
from fla.ops.generalized_delta_rule.dplr.wy_fast_fwd import prepare_wy_repr_fwd
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp2, gather
from fla.utils import IS_AMD, IS_GATHER_SUPPORTED, autotune_cache_kwargs, check_shared_mem


_PAIR_WARPS = [2, 4, 8, 16] if IS_AMD else [2, 4, 8, 16, 32]


@triton.autotune(
    configs=[triton.Config({"BR": br}, num_warps=warps) for br in (16, 32, 64) for warps in (2, 4, 8)],
    key=["R", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _frame_gate_cumsum_kernel(
    g,
    cumulative,
    cu_seqlens,
    chunk_indices,
    SG_B: tl.constexpr,
    SG_T: tl.constexpr,
    SG_H: tl.constexpr,
    SG_R: tl.constexpr,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    R: tl.constexpr,
    BT: tl.constexpr,
    BR: tl.constexpr,
    SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_r = tl.program_id(0)
    i_c_global = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    i_c = i_c_global
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_c_global * 2).to(tl.int32)
        i_c = tl.load(chunk_indices + i_c_global * 2 + 1).to(tl.int64)
        source_bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        source_eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        logical_bos = source_bos * E
        logical_length = (source_eos - source_bos) * E
    else:
        source_bos = 0
        logical_bos = i_b * T * E
        logical_length = T * E
    logical = i_c * BT + tl.arange(0, BT)
    coordinate = i_r * BR + tl.arange(0, BR)
    token = logical // E
    edit = logical % E
    mask = (logical[:, None] < logical_length) & (coordinate[None, :] < R)
    source = (
        i_b * SG_B
        + (source_bos + token[:, None]) * SG_T
        + i_h * SG_H
        + coordinate[None, :] * SG_R
    )
    values = tl.load(g + source, mask=mask & (edit[:, None] == 0), other=0.0).to(tl.float32)
    values = tl.cumsum(values, axis=0) * SCALE
    target = ((logical_bos + logical[:, None]) * H + i_h) * R + coordinate[None, :]
    tl.store(cumulative + target, values, mask=mask)


def _frame_gate_cumsum(
    g: torch.Tensor,
    edits: int,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, length, heads, width = g.shape
    logical_length = length * edits
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens * edits, chunk_size)
    chunks = triton.cdiv(logical_length, chunk_size) if chunk_indices is None else len(chunk_indices)
    cumulative = torch.empty(
        batch, logical_length, heads, width, dtype=torch.float32, device=g.device
    )
    _frame_gate_cumsum_kernel[
        lambda meta: (
            triton.cdiv(width, meta["BR"]),
            chunks,
            heads if cu_seqlens is not None else batch * heads,
        )
    ](
        g,
        cumulative,
        cu_seqlens if cu_seqlens is not None else g,
        chunk_indices if chunk_indices is not None else g,
        SG_B=g.stride(0),
        SG_T=g.stride(1),
        SG_H=g.stride(2),
        SG_R=g.stride(3),
        T=length,
        H=heads,
        E=edits,
        R=width,
        BT=chunk_size,
        SCALE=RCP_LN2,
        IS_VARLEN=cu_seqlens is not None,
    )
    return cumulative


@triton.jit(do_not_specialize=["T"])
def _pack_write_kernel(
    values,
    write_raw,
    z,
    SV_B: tl.constexpr,
    SV_T: tl.constexpr,
    SV_H: tl.constexpr,
    SV_E: tl.constexpr,
    SV_V: tl.constexpr,
    SW_B: tl.constexpr,
    SW_T: tl.constexpr,
    SW_H: tl.constexpr,
    SW_E: tl.constexpr,
    SW_V: tl.constexpr,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    logical_length = T * E
    logical = i_t * BT + tl.arange(0, BT)
    coordinate = i_v * BV + tl.arange(0, BV)
    token = logical // E
    edit = logical % E
    mask = (logical[:, None] < logical_length) & (coordinate[None, :] < V)
    value_source = (
        i_b * SV_B
        + token[:, None] * SV_T
        + i_h * SV_H
        + edit[:, None] * SV_E
        + coordinate[None, :] * SV_V
    )
    write_source = (
        i_b * SW_B
        + token[:, None] * SW_T
        + i_h * SW_H
        + edit[:, None] * SW_E
        + coordinate[None, :] * SW_V
    )
    target = ((i_b * logical_length + logical[:, None]) * H + i_h) * V + coordinate[None, :]
    value = tl.load(values + value_source, mask=mask, other=0.0).to(tl.float32)
    logit = tl.load(write_raw + write_source, mask=mask, other=0.0).to(tl.float32)
    gate = 2.0 * tl.sigmoid(logit)
    tl.store(z + target, value * gate, mask=mask)


@triton.jit(do_not_specialize=["T"])
def _pack_write_bwd_kernel(
    dz,
    values,
    write_raw,
    dvalues,
    dwrite_raw,
    SV_B: tl.constexpr,
    SV_T: tl.constexpr,
    SV_H: tl.constexpr,
    SV_E: tl.constexpr,
    SV_V: tl.constexpr,
    SW_B: tl.constexpr,
    SW_T: tl.constexpr,
    SW_H: tl.constexpr,
    SW_E: tl.constexpr,
    SW_V: tl.constexpr,
    SDV_B: tl.constexpr,
    SDV_T: tl.constexpr,
    SDV_H: tl.constexpr,
    SDV_E: tl.constexpr,
    SDV_V: tl.constexpr,
    SDW_B: tl.constexpr,
    SDW_T: tl.constexpr,
    SDW_H: tl.constexpr,
    SDW_E: tl.constexpr,
    SDW_V: tl.constexpr,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    logical_length = T * E
    logical = i_t * BT + tl.arange(0, BT)
    coordinate = i_v * BV + tl.arange(0, BV)
    token = logical // E
    edit = logical % E
    mask = (logical[:, None] < logical_length) & (coordinate[None, :] < V)
    value_source = (
        i_b * SV_B
        + token[:, None] * SV_T
        + i_h * SV_H
        + edit[:, None] * SV_E
        + coordinate[None, :] * SV_V
    )
    write_source = (
        i_b * SW_B
        + token[:, None] * SW_T
        + i_h * SW_H
        + edit[:, None] * SW_E
        + coordinate[None, :] * SW_V
    )
    dvalue_target = (
        i_b * SDV_B
        + token[:, None] * SDV_T
        + i_h * SDV_H
        + edit[:, None] * SDV_E
        + coordinate[None, :] * SDV_V
    )
    dwrite_target = (
        i_b * SDW_B
        + token[:, None] * SDW_T
        + i_h * SDW_H
        + edit[:, None] * SDW_E
        + coordinate[None, :] * SDW_V
    )
    packed = ((i_b * logical_length + logical[:, None]) * H + i_h) * V + coordinate[None, :]
    grad = tl.load(dz + packed, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(values + value_source, mask=mask, other=0.0).to(tl.float32)
    logit = tl.load(write_raw + write_source, mask=mask, other=0.0).to(tl.float32)
    sigmoid = tl.sigmoid(logit)
    gate = 2.0 * sigmoid
    tl.store(dvalues + dvalue_target, grad * gate, mask=mask)
    tl.store(
        dwrite_raw + dwrite_target,
        grad * value * (2.0 * sigmoid * (1.0 - sigmoid)),
        mask=mask,
    )


def _pack_write(values: torch.Tensor, write_raw: torch.Tensor) -> torch.Tensor:
    batch, length, heads, edits, value_width = values.shape
    logical_length = length * edits
    z = torch.empty(
        batch, logical_length, heads, value_width,
        dtype=values.dtype, device=values.device,
    )
    block_value = min(64, triton.next_power_of_2(value_width))
    _pack_write_kernel[(triton.cdiv(value_width, block_value), triton.cdiv(logical_length, 32), batch * heads)](
        values,
        write_raw,
        z,
        SV_B=values.stride(0), SV_T=values.stride(1),
        SV_H=values.stride(2), SV_E=values.stride(3), SV_V=values.stride(4),
        SW_B=write_raw.stride(0), SW_T=write_raw.stride(1),
        SW_H=write_raw.stride(2), SW_E=write_raw.stride(3), SW_V=write_raw.stride(4),
        T=length,
        H=heads,
        E=edits,
        V=value_width,
        BT=32,
        BV=block_value,
        num_warps=4,
    )
    return z


def _pack_write_backward(
    dz: torch.Tensor,
    values: torch.Tensor,
    write_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length, heads, edits, value_width = values.shape
    logical_length = length * edits
    dvalues = torch.empty(values.shape, dtype=values.dtype, device=values.device)
    dwrite_raw = torch.empty(
        write_raw.shape, dtype=write_raw.dtype, device=write_raw.device
    )
    block_value = min(64, triton.next_power_of_2(value_width))
    _pack_write_bwd_kernel[(triton.cdiv(value_width, block_value), triton.cdiv(logical_length, 32), batch * heads)](
        dz,
        values,
        write_raw,
        dvalues,
        dwrite_raw,
        SV_B=values.stride(0), SV_T=values.stride(1),
        SV_H=values.stride(2), SV_E=values.stride(3), SV_V=values.stride(4),
        SW_B=write_raw.stride(0), SW_T=write_raw.stride(1),
        SW_H=write_raw.stride(2), SW_E=write_raw.stride(3), SW_V=write_raw.stride(4),
        SDV_B=dvalues.stride(0), SDV_T=dvalues.stride(1),
        SDV_H=dvalues.stride(2), SDV_E=dvalues.stride(3), SDV_V=dvalues.stride(4),
        SDW_B=dwrite_raw.stride(0), SDW_T=dwrite_raw.stride(1),
        SDW_H=dwrite_raw.stride(2), SDW_E=dwrite_raw.stride(3), SDW_V=dwrite_raw.stride(4),
        T=length,
        H=heads,
        E=edits,
        V=value_width,
        BT=32,
        BV=block_value,
        num_warps=4,
    )
    return dvalues, dwrite_raw


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in _PAIR_WARPS
        for num_stages in (2, 3, 4)
    ],
    key=["BK", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _direct_e_pair_fwd_kernel(
    d,
    paired_dual,
    cumulative,
    q_scaled,
    d_tail,
    e_scaled,
    A_qd,
    A_ed,
    cu_seqlens,
    chunk_indices,
    frame_chunk_offsets,
    T,
    SOURCE_T: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    N: tl.constexpr,
    FC: tl.constexpr,
    R: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t_global = tl.program_id(0).to(tl.int64)
    i_b = tl.program_id(1)
    i_h = tl.program_id(2)
    i_t = i_t_global
    i_n = i_b
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t_global * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t_global * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        sequence_length = eos - bos
    else:
        bos = i_b * T
        sequence_length = T

    if i_t * BT >= sequence_length:
        return

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    o_k = tl.arange(0, BK)
    m_t = o_t < sequence_length
    m_k = o_k < R
    m_tk = m_t[:, None] & m_k[None, :]
    token = o_t // E
    edit = o_t % E
    frame_chunk = token // FC
    frame_row = token % FC
    if IS_VARLEN:
        frame_offset = tl.load(frame_chunk_offsets + i_n).to(tl.int64)
        panel = (frame_offset + frame_chunk) * H + i_h
    else:
        panel = (i_n * H + i_h) * N + frame_chunk
    p_d = ((panel[:, None] * E + edit[:, None]) * FC + frame_row[:, None]) * R + o_k[None, :]
    p_e = ((panel[:, None] * (E + 1) + edit[:, None]) * FC + frame_row[:, None]) * R + o_k[None, :]
    p_chi = ((panel[:, None] * (E + 1) + E) * FC + frame_row[:, None]) * R + o_k[None, :]
    vector_base = (bos * H + i_h) * R
    stride_vector = H * R
    p_g = cumulative + vector_base + o_t[:, None] * stride_vector + o_k[None, :]
    b_chi = tl.load(p_chi + paired_dual, mask=m_tk & (edit[:, None] == E - 1), other=0.0)
    b_d = tl.load(p_d + d, mask=m_tk, other=0.0)
    b_e = tl.load(p_e + paired_dual, mask=m_tk, other=0.0)
    b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)

    last = min((i_t + 1) * BT, sequence_length) - 1
    p_g_last = cumulative + vector_base + last * stride_vector + o_k
    b_g_last = tl.load(p_g_last, mask=m_k, other=0.0).to(tl.float32)
    b_exp_g = exp2(b_g)
    b_q_scaled = b_chi * b_exp_g
    b_e_scaled = -b_e * b_exp_g
    b_d_tail = b_d * exp2(b_g_last[None, :] - b_g)
    tl.store(q_scaled + vector_base + o_t[:, None] * stride_vector + o_k[None, :], b_q_scaled, mask=m_tk)
    tl.store(e_scaled + vector_base + o_t[:, None] * stride_vector + o_k[None, :], b_e_scaled, mask=m_tk)
    tl.store(d_tail + vector_base + o_t[:, None] * stride_vector + o_k[None, :], b_d_tail, mask=m_tk)

    matrix_base = (bos * H + i_h) * BT
    o_A = (bos + o_t) * H * BT + i_h * BT
    m_A = m_t
    for j in range(0, min(BT, sequence_length - i_t * BT)):
        if GATHER_SUPPORTED:
            row = tl.full([1, BK], j, dtype=tl.int16)
            b_d_j = gather(b_d, row, axis=0)
            b_g_j = gather(b_g, row, axis=0)
        else:
            select = o_i == j
            b_d_j = tl.sum(tl.where(select[:, None], b_d, 0.0), axis=0)[None, :]
            b_g_j = tl.sum(tl.where(select[:, None], b_g, 0.0), axis=0)[None, :]
        decay = exp2(b_g - b_g_j)
        pair_qd = tl.sum(b_chi * b_d_j * decay, axis=1)
        pair_ed = tl.sum((-b_e) * b_d_j * decay, axis=1)
        pair_qd = tl.where(o_i >= j, pair_qd, 0.0)
        pair_ed = tl.where(o_i > j, pair_ed, 0.0)
        tl.store(A_qd + o_A + j, pair_qd, mask=m_A)
        tl.store(A_ed + o_A + j, pair_ed, mask=m_A)


def _direct_e_pair_forward(
    d: torch.Tensor,
    paired_dual: torch.Tensor,
    cumulative: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    frame_chunks: int,
    frame_chunk_offsets: torch.Tensor | None,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    panels, edits, frame_chunk_size, width = d.shape
    batch, length, heads, _ = cumulative.shape
    source_length = length // edits
    if cu_seqlens is None and panels != batch * heads * frame_chunks:
        raise ValueError("frame panel count does not match [B,H,Nchunk]")
    if cu_seqlens is not None and frame_chunk_offsets is None:
        raise ValueError("variable-length pair scheduling requires frame chunk offsets")
    chunks = triton.cdiv(length, chunk_size) if chunk_indices is None else len(chunk_indices)
    q_scaled = torch.empty_like(cumulative, dtype=d.dtype)
    d_tail = torch.empty_like(q_scaled)
    e_scaled = torch.empty_like(q_scaled)
    A_qd = torch.empty(batch, length, heads, chunk_size, dtype=d.dtype, device=d.device)
    A_ed = torch.empty(batch, length, heads, chunk_size, dtype=torch.float32, device=d.device)
    block_width = max(triton.next_power_of_2(width), 16)
    _direct_e_pair_fwd_kernel[(chunks, batch if cu_seqlens is None else 1, heads)](
        d=d,
        paired_dual=paired_dual,
        cumulative=cumulative,
        q_scaled=q_scaled,
        d_tail=d_tail,
        e_scaled=e_scaled,
        A_qd=A_qd,
        A_ed=A_ed,
        cu_seqlens=cu_seqlens if cu_seqlens is not None else cumulative,
        chunk_indices=chunk_indices if chunk_indices is not None else cumulative,
        frame_chunk_offsets=(
            frame_chunk_offsets if frame_chunk_offsets is not None else cumulative
        ),
        T=length,
        SOURCE_T=source_length,
        H=heads,
        E=edits,
        N=frame_chunks,
        FC=frame_chunk_size,
        R=width,
        BT=chunk_size,
        BK=block_width,
        GATHER_SUPPORTED=IS_GATHER_SUPPORTED,
        IS_VARLEN=cu_seqlens is not None,
    )
    return A_qd, A_ed, q_scaled, d_tail, e_scaled


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in _PAIR_WARPS
        for num_stages in (2, 3, 4)
    ],
    key=["BK", "BT", "R"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _direct_e_pair_bwd_kernel(
    d,
    paired_dual,
    cumulative,
    dA_qd_0,
    dA_qd_1,
    dA_ed_0,
    dA_ed_1,
    dq_scaled,
    dd_tail_0,
    dd_tail_1,
    de_scaled,
    dg_tail,
    dd,
    dpaired_dual,
    dg,
    cu_seqlens,
    chunk_indices,
    frame_chunk_offsets,
    T,
    SOURCE_T: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    N: tl.constexpr,
    FC: tl.constexpr,
    R: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_t_global = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    i_t = i_t_global
    i_n = i_b
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t_global * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t_global * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        sequence_length = eos - bos
    else:
        chunks = tl.cdiv(T, BT)
        i_t_global = i_b * chunks + i_t
        bos = i_b * T
        sequence_length = T

    if i_t * BT >= sequence_length:
        return

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    o_k = i_k * BK + tl.arange(0, BK)
    m_t = o_t < sequence_length
    m_k = o_k < R
    m_tk = m_t[:, None] & m_k[None, :]
    token = o_t // E
    edit = o_t % E
    frame_chunk = token // FC
    frame_row = token % FC
    if IS_VARLEN:
        frame_offset = tl.load(frame_chunk_offsets + i_n).to(tl.int64)
        panel = (frame_offset + frame_chunk) * H + i_h
    else:
        panel = (i_n * H + i_h) * N + frame_chunk
    p_d = ((panel[:, None] * E + edit[:, None]) * FC + frame_row[:, None]) * R + o_k[None, :]
    p_e = ((panel[:, None] * (E + 1) + edit[:, None]) * FC + frame_row[:, None]) * R + o_k[None, :]
    p_chi = ((panel[:, None] * (E + 1) + E) * FC + frame_row[:, None]) * R + o_k[None, :]
    vector_base = (bos * H + i_h) * R
    stride_vector = H * R
    p_g = cumulative + vector_base + o_t[:, None] * stride_vector + o_k[None, :]
    b_chi = tl.load(paired_dual + p_chi, mask=m_tk & (edit[:, None] == E - 1), other=0.0)
    b_d = tl.load(d + p_d, mask=m_tk, other=0.0)
    b_e = tl.load(paired_dual + p_e, mask=m_tk, other=0.0)
    b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)

    o_A = tl.arange(0, BT)
    matrix_base = (bos * H + i_h) * BT
    m_matrix = m_t[:, None] & (o_A[None, :] < BT)
    p_qd_0 = dA_qd_0 + matrix_base + o_t[:, None] * (H * BT) + o_A[None, :]
    p_qd_1 = dA_qd_1 + matrix_base + o_t[:, None] * (H * BT) + o_A[None, :]
    p_ed_0 = dA_ed_0 + matrix_base + o_t[:, None] * (H * BT) + o_A[None, :]
    p_ed_1 = dA_ed_1 + matrix_base + o_t[:, None] * (H * BT) + o_A[None, :]
    b_dA_qd = tl.load(p_qd_0, mask=m_matrix, other=0.0) + tl.load(p_qd_1, mask=m_matrix, other=0.0)
    b_dA_ed = tl.load(p_ed_0, mask=m_matrix, other=0.0) + tl.load(p_ed_1, mask=m_matrix, other=0.0)

    b_dchi = tl.zeros([BT, BK], dtype=tl.float32)
    b_dd = tl.zeros([BT, BK], dtype=tl.float32)
    b_de = tl.zeros([BT, BK], dtype=tl.float32)
    for j in range(0, min(BT, sequence_length - i_t * BT)):
        if GATHER_SUPPORTED:
            row_k = tl.full([1, BK], j, dtype=tl.int16)
            col_A = tl.full([BT, 1], j, dtype=tl.int16)
            row_A = tl.full([1, BT], j, dtype=tl.int16)
            d_j = gather(b_d, row_k, axis=0)
            chi_j = gather(b_chi, row_k, axis=0)
            e_j = gather(b_e, row_k, axis=0)
            g_j = gather(b_g, row_k, axis=0)
            dA_qd_col = gather(b_dA_qd, col_A, axis=1)
            dA_ed_col = gather(b_dA_ed, col_A, axis=1)
            dA_qd_row = tl.sum(gather(b_dA_qd, row_A, axis=0), axis=0)[:, None]
            dA_ed_row = tl.sum(gather(b_dA_ed, row_A, axis=0), axis=0)[:, None]
        else:
            select = o_i == j
            d_j = tl.sum(tl.where(select[:, None], b_d, 0.0), axis=0)[None, :]
            chi_j = tl.sum(tl.where(select[:, None], b_chi, 0.0), axis=0)[None, :]
            e_j = tl.sum(tl.where(select[:, None], b_e, 0.0), axis=0)[None, :]
            g_j = tl.sum(tl.where(select[:, None], b_g, 0.0), axis=0)[None, :]
            dA_qd_col = tl.sum(tl.where(select[None, :], b_dA_qd, 0.0), axis=1)[:, None]
            dA_ed_col = tl.sum(tl.where(select[None, :], b_dA_ed, 0.0), axis=1)[:, None]
            dA_qd_row = tl.sum(tl.where(select[:, None], b_dA_qd, 0.0), axis=0)[:, None]
            dA_ed_row = tl.sum(tl.where(select[:, None], b_dA_ed, 0.0), axis=0)[:, None]

        row_decay = exp2(b_g - g_j)
        b_dchi += tl.where(o_i[:, None] >= j, dA_qd_col * d_j * row_decay, 0.0)
        b_de += tl.where(o_i[:, None] > j, -dA_ed_col * d_j * row_decay, 0.0)
        column_decay = exp2(g_j - b_g)
        b_dd += tl.where(o_i[:, None] <= j, dA_qd_row * chi_j * column_decay, 0.0)
        b_dd += tl.where(o_i[:, None] < j, -dA_ed_row * e_j * column_decay, 0.0)

    p_dq_scaled = dq_scaled + vector_base + o_t[:, None] * stride_vector + o_k[None, :]
    p_dd_tail_0 = dd_tail_0 + vector_base + o_t[:, None] * stride_vector + o_k[None, :]
    p_dd_tail_1 = dd_tail_1 + vector_base + o_t[:, None] * stride_vector + o_k[None, :]
    p_de_scaled = de_scaled + vector_base + o_t[:, None] * stride_vector + o_k[None, :]
    exp_g = exp2(b_g)
    last = min((i_t + 1) * BT, sequence_length) - 1
    p_g_last = cumulative + vector_base + last * stride_vector + o_k
    g_last = tl.load(p_g_last, mask=m_k, other=0.0).to(tl.float32)
    tail_decay = exp2(g_last[None, :] - b_g)
    b_dchi += tl.load(p_dq_scaled, mask=m_tk, other=0.0).to(tl.float32) * exp_g
    b_de -= tl.load(p_de_scaled, mask=m_tk, other=0.0).to(tl.float32) * exp_g
    b_dd += (
        tl.load(p_dd_tail_0, mask=m_tk, other=0.0).to(tl.float32)
        + tl.load(p_dd_tail_1, mask=m_tk, other=0.0).to(tl.float32)
    ) * tail_decay

    b_dG = b_dchi * b_chi + b_de * b_e - b_dd * b_d
    b_dg = tl.cumsum(b_dG, axis=0, reverse=True)
    tail_base = (i_t_global * H + i_h) * R
    b_dg_tail = tl.load(dg_tail + tail_base + o_k, mask=m_k, other=0.0)
    b_dg += b_dg_tail[None, :]

    source_bos = bos // E
    p_dg = ((source_bos + token[:, None]) * H + i_h) * R + o_k[None, :]
    tl.store(dpaired_dual + p_chi, b_dchi, mask=m_tk & (edit[:, None] == E - 1))
    tl.store(dd + p_d, b_dd, mask=m_tk)
    tl.store(dpaired_dual + p_e, b_de, mask=m_tk)
    tl.store(dg + p_dg, b_dg, mask=m_tk & (edit[:, None] == 0))


def _direct_e_pair_backward(
    d: torch.Tensor,
    paired_dual: torch.Tensor,
    cumulative: torch.Tensor,
    dA_qd_0: torch.Tensor,
    dA_qd_1: torch.Tensor,
    dA_ed_0: torch.Tensor,
    dA_ed_1: torch.Tensor,
    dq_scaled: torch.Tensor,
    dd_tail_0: torch.Tensor,
    dd_tail_1: torch.Tensor,
    de_scaled: torch.Tensor,
    dg_tail: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    frame_chunks: int,
    frame_chunk_offsets: torch.Tensor | None,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    panels, edits, frame_chunk_size, width = d.shape
    batch, length, heads, _ = cumulative.shape
    source_length = length // edits
    chunks = triton.cdiv(length, chunk_size) if chunk_indices is None else len(chunk_indices)
    block_width = min(64, triton.next_power_of_2(width)) if check_shared_mem() else min(32, triton.next_power_of_2(width))
    coordinate_blocks = triton.cdiv(width, block_width)
    dd = torch.zeros_like(d)
    dpaired_dual = torch.zeros_like(paired_dual)
    dg = torch.empty(
        batch, source_length, heads, width, dtype=torch.float32, device=d.device
    )
    _direct_e_pair_bwd_kernel[(
        coordinate_blocks,
        chunks,
        heads if cu_seqlens is not None else batch * heads,
    )](
        d=d,
        paired_dual=paired_dual,
        cumulative=cumulative,
        dA_qd_0=dA_qd_0,
        dA_qd_1=dA_qd_1,
        dA_ed_0=dA_ed_0,
        dA_ed_1=dA_ed_1,
        dq_scaled=dq_scaled,
        dd_tail_0=dd_tail_0,
        dd_tail_1=dd_tail_1,
        de_scaled=de_scaled,
        dg_tail=dg_tail,
        dd=dd,
        dpaired_dual=dpaired_dual,
        dg=dg,
        cu_seqlens=cu_seqlens if cu_seqlens is not None else cumulative,
        chunk_indices=chunk_indices if chunk_indices is not None else cumulative,
        frame_chunk_offsets=(
            frame_chunk_offsets if frame_chunk_offsets is not None else cumulative
        ),
        T=length,
        SOURCE_T=source_length,
        H=heads,
        E=edits,
        N=frame_chunks,
        FC=frame_chunk_size,
        R=width,
        BT=chunk_size,
        BK=block_width,
        GATHER_SUPPORTED=IS_GATHER_SUPPORTED,
        IS_VARLEN=cu_seqlens is not None,
    )
    return dd, dpaired_dual, dg


def _forward_blocks(
    d: torch.Tensor,
    paired_dual: torch.Tensor,
    values: torch.Tensor,
    write_raw: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    *,
    output_final_state: bool,
    compute_output: bool,
    chunk_size: int,
):
    edits = d.shape[1]
    frame_chunks = triton.cdiv(g.shape[1], d.shape[2]) if cu_seqlens is None else 1
    frame_chunk_offsets = (
        prepare_chunk_offsets(cu_seqlens, d.shape[2])
        if cu_seqlens is not None
        else None
    )
    logical_cu_seqlens = cu_seqlens * edits if cu_seqlens is not None else None
    logical_cu_seqlens_cpu = (
        cu_seqlens_cpu * edits if cu_seqlens_cpu is not None else None
    )
    chunk_indices = (
        prepare_chunk_indices(
            logical_cu_seqlens,
            chunk_size,
            cu_seqlens_cpu=logical_cu_seqlens_cpu,
        )
        if logical_cu_seqlens is not None
        else None
    )
    cumulative = _frame_gate_cumsum(
        g,
        edits,
        chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    z = _pack_write(values, write_raw)
    A_qd, A_ed, q_scaled, d_tail, e_scaled = _direct_e_pair_forward(
        d,
        paired_dual,
        cumulative,
        logical_cu_seqlens,
        chunk_indices,
        frame_chunks,
        frame_chunk_offsets,
        chunk_size=chunk_size,
    )
    w, u, A_ed_inv = prepare_wy_repr_fwd(
        ag=e_scaled,
        v=z,
        A_ak=A_ed,
        A_ab=A_ed,
        cu_seqlens=logical_cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    h, z_new, final_state = chunk_dplr_fwd_h(
        kg=d_tail,
        bg=d_tail,
        v=z,
        w=w,
        u=u,
        gk=cumulative,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=logical_cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    output = None
    if compute_output:
        output = chunk_dplr_fwd_o(
            qg=q_scaled,
            v=z,
            v_new=z_new,
            A_qk=A_qd,
            A_qb=A_qd,
            h=h,
            cu_seqlens=logical_cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
    cache = (
        z,
        cumulative,
        A_qd,
        A_ed,
        q_scaled,
        d_tail,
        e_scaled,
        w,
        h,
        z_new,
        A_ed_inv,
        logical_cu_seqlens,
        chunk_indices,
    )
    return output, final_state, cache


class _DirectEChunkFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        d: torch.Tensor,
        paired_dual: torch.Tensor,
        values: torch.Tensor,
        write_raw: torch.Tensor,
        g: torch.Tensor,
        initial_state: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_cpu: torch.Tensor | None,
        output_final_state: bool,
        chunk_size: int,
    ):
        output, final_state, _ = _forward_blocks(
            d,
            paired_dual,
            values,
            write_raw,
            g,
            initial_state,
            cu_seqlens,
            cu_seqlens_cpu,
            output_final_state=output_final_state,
            compute_output=True,
            chunk_size=chunk_size,
        )
        if output is None:
            raise RuntimeError("direct-e forward did not produce an output")
        saved_state = initial_state if initial_state is not None else d.new_empty(0, dtype=torch.float32)
        saved_cu = cu_seqlens if cu_seqlens is not None else d.new_empty(0, dtype=torch.long)
        ctx.save_for_backward(
            d, paired_dual, values, write_raw, g, saved_state, saved_cu
        )
        ctx.has_initial_state = initial_state is not None
        ctx.has_cu_seqlens = cu_seqlens is not None
        ctx.cu_seqlens_cpu = cu_seqlens_cpu
        ctx.chunk_size = chunk_size
        return output.to(d.dtype), final_state

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_final_state: torch.Tensor | None):
        d, paired_dual, values, write_raw, g, saved_state, saved_cu = ctx.saved_tensors
        initial_state = saved_state if ctx.has_initial_state else None
        cu_seqlens = saved_cu if ctx.has_cu_seqlens else None
        chunk_size = ctx.chunk_size
        (
            _,
            _,
            (
                z,
                cumulative,
                A_qd,
                A_ed,
                q_scaled,
                d_tail,
                e_scaled,
                w,
                h,
                z_new,
                A_ed_inv,
                logical_cu_seqlens,
                chunk_indices,
            ),
        ) = _forward_blocks(
            d,
            paired_dual,
            values,
            write_raw,
            g,
            initial_state,
            cu_seqlens,
            ctx.cu_seqlens_cpu,
            output_final_state=False,
            compute_output=False,
            chunk_size=chunk_size,
        )
        dz_new_intra, dA_qd_0, dA_qd_1 = chunk_dplr_bwd_dAu(
            v=z,
            v_new=z_new,
            do=grad_output,
            A_qb=A_qd,
            scale=1.0,
            cu_seqlens=logical_cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        dh, dh0, du = chunk_dplr_bwd_dhu(
            qg=q_scaled,
            bg=d_tail,
            w=w,
            gk=cumulative,
            h0=initial_state,
            dht=grad_final_state,
            do=grad_output,
            dv=dz_new_intra,
            cu_seqlens=logical_cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        dz0 = chunk_dplr_bwd_dv(
            A_qk=A_qd,
            kg=d_tail,
            do=grad_output,
            dh=dh,
            cu_seqlens=logical_cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        dq_scaled, dd_tail_0, dw, dd_tail_1, dg_tail = chunk_dplr_bwd_o(
            k=d_tail,
            b=d_tail,
            v=z,
            v_new=z_new,
            gk=cumulative,
            do=grad_output,
            h=h,
            dh=dh,
            dv=du,
            w=w,
            cu_seqlens=logical_cu_seqlens,
            chunk_size=chunk_size,
            scale=1.0,
            chunk_indices=chunk_indices,
        )
        dA_ed_0, dA_ed_1, dz, de_scaled = chunk_dplr_bwd_wy(
            A_ab_inv=A_ed_inv,
            A_ak=A_ed,
            v=z,
            ag=e_scaled,
            dw=dw,
            du=du,
            dv0=dz0,
            cu_seqlens=logical_cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        dd, dpaired_dual, dg = _direct_e_pair_backward(
            d,
            paired_dual,
            cumulative,
            dA_qd_0,
            dA_qd_1,
            dA_ed_0,
            dA_ed_1,
            dq_scaled,
            dd_tail_0,
            dd_tail_1,
            de_scaled,
            dg_tail,
            logical_cu_seqlens,
            chunk_indices,
            triton.cdiv(g.shape[1], d.shape[2]) if cu_seqlens is None else 1,
            (
                prepare_chunk_offsets(cu_seqlens, d.shape[2])
                if cu_seqlens is not None
                else None
            ),
            chunk_size=chunk_size,
        )
        dvalues, dwrite_raw = _pack_write_backward(dz, values, write_raw)
        return (
            dd,
            dpaired_dual,
            dvalues,
            dwrite_raw,
            dg,
            dh0,
            None,
            None,
            None,
            None,
        )


def chunk_direct_e_delta_rule(
    d: torch.Tensor,
    paired_dual: torch.Tensor,
    values: torch.Tensor,
    write_raw: torch.Tensor,
    g: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    output_final_state: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """FLA chunk-WY specialized to SolveDelta's paired direct-e recurrence."""
    if d.ndim != 4:
        raise ValueError("d must have frame-native [P,K,C,r]")
    panels, edits, frame_chunk_size, width = d.shape
    if paired_dual.shape != (panels, edits + 1, frame_chunk_size, width):
        raise ValueError("paired_dual must have frame-native [P,K+1,C,r]")
    if values.ndim != 5 or write_raw.shape != values.shape:
        raise ValueError("values and write_raw must share [B,T,H,K,d_v]")
    batch, length, heads, value_edits, _ = values.shape
    if value_edits != edits or g.shape != (batch, length, heads, width):
        raise ValueError("frame panels, values, and associative decay disagree")
    expected_frame_chunks = triton.cdiv(length, frame_chunk_size)
    if cu_seqlens is not None and batch != 1:
        raise ValueError("FLA variable-length scheduling requires a flat batch of one")
    if cu_seqlens is None:
        if panels != batch * heads * expected_frame_chunks:
            raise ValueError("frame panel count does not match [B,H,Nchunk]")
    else:
        frame_chunk_indices = prepare_chunk_indices(
            cu_seqlens,
            frame_chunk_size,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
        if panels != len(frame_chunk_indices) * heads:
            raise ValueError("frame panel count does not match packed FLA chunks")
    if d.dtype != values.dtype or d.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("direct-e frame panels and values must share FP16 or BF16 dtype")
    if write_raw.dtype != torch.bfloat16:
        raise TypeError("raw write gates must be BF16")
    if g.dtype != torch.float32:
        raise TypeError("associative log decay must be FP32")
    if not d.is_contiguous() or not paired_dual.is_contiguous():
        raise ValueError("frame-native d and paired_dual panels must be contiguous")
    if any(tensor.stride(-1) != 1 for tensor in (values, write_raw, g)):
        raise ValueError("direct-e source tensors require unit coordinate stride")
    return _DirectEChunkFunction.apply(
        d,
        paired_dual,
        values,
        write_raw,
        g,
        initial_state,
        cu_seqlens,
        cu_seqlens_cpu,
        output_final_state,
        chunk_size,
    )


__all__ = ["chunk_direct_e_delta_rule"]
