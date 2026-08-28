# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
"""Token-major source owner for the selected block-E3 RLS exterior."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.constant import RCP_LN2
from fla.utils import autotune_cache_kwargs


@triton.jit(do_not_specialize=["T"])
def _block_e3_sources_fwd_kernel(
    u,
    h,
    q,
    keys,
    values,
    gain,
    prediction,
    geometry_log_decay,
    associative_log_decay,
    erase_raw,
    write_raw,
    previous_mass,
    current_mass,
    strength,
    out_d,
    out_paired,
    out_z,
    out_diagonal_log,
    out_q_rstd,
    out_key_rstd,
    SH_B: tl.constexpr,
    SH_T: tl.constexpr,
    SH_H: tl.constexpr,
    SQ_B: tl.constexpr,
    SQ_T: tl.constexpr,
    SQ_H: tl.constexpr,
    SK_B: tl.constexpr,
    SK_T: tl.constexpr,
    SK_H: tl.constexpr,
    SV_B: tl.constexpr,
    SV_T: tl.constexpr,
    SV_H: tl.constexpr,
    SE_B: tl.constexpr,
    SE_T: tl.constexpr,
    SE_H: tl.constexpr,
    SW_B: tl.constexpr,
    SW_T: tl.constexpr,
    SW_H: tl.constexpr,
    T,
    H: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    FRAME_CHUNK: tl.constexpr,
    FRAME_CHUNKS: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
):
    token_head = tl.program_id(0).to(tl.int64)
    token_flat, head = token_head // H, token_head % H
    batch, token = token_flat // T, token_flat % T
    panel = (batch * H + head) * FRAME_CHUNKS + token // FRAME_CHUNK
    row = token % FRAME_CHUNK
    total_edits: tl.constexpr = 3

    o_r = tl.arange(0, BR)
    m_r = o_r < R
    vector_base = token_head * R
    b_u = tl.load(u + vector_base + o_r, mask=m_r, other=0.0).to(tl.float32)
    h_offset = batch * SH_B + token * SH_T + head * SH_H + o_r
    q_offset = batch * SQ_B + token * SQ_T + head * SQ_H + o_r
    b_h = tl.load(h + h_offset, mask=m_r, other=0.0).to(tl.float32)
    b_q_raw = tl.load(q + q_offset, mask=m_r, other=0.0).to(tl.float32)
    q_rstd = 1.0 / tl.sqrt(tl.sum(b_q_raw * b_q_raw, axis=0) + 1.0e-24)
    b_q = (b_q_raw * q_rstd).to(q.dtype.element_ty).to(tl.float32)
    b_gain = tl.load(gain + vector_base + o_r, mask=m_r, other=0.0).to(tl.float32)
    b_prediction = tl.load(
        prediction + vector_base + o_r, mask=m_r, other=0.0
    ).to(tl.float32)
    denominator = 1.0 - tl.sum(b_u * b_gain, axis=0)

    geometry_decay = tl.exp(
        tl.load(geometry_log_decay + token_head).to(tl.float32)
    )
    mass_previous = tl.load(previous_mass + token_head).to(tl.float32)
    mass_current = tl.load(current_mass + token_head).to(tl.float32)
    mass_scale = mass_previous / mass_current
    gamma = tl.load(strength + head).to(tl.float32)
    diagonal_h = 1.0 + gamma * (mass_scale * geometry_decay - 1.0)
    b_assoc_log = tl.load(
        associative_log_decay + vector_base + o_r, mask=m_r, other=0.0
    ).to(tl.float32)
    b_assoc = tl.exp(b_assoc_log)
    b_previous_gain = geometry_decay * b_gain / denominator
    b_residual = (b_h - b_prediction) / denominator

    d0 = b_assoc * (gamma * mass_scale / diagonal_h) * b_u
    e0 = -b_previous_gain / b_assoc
    d1 = gamma * b_gain
    e1 = -b_residual
    token_panel = panel * FRAME_CHUNK + row
    p_d0 = (token_panel * total_edits) * R + o_r
    p_d1 = (token_panel * total_edits + 1) * R + o_r
    p_e0 = (token_panel * (total_edits + 1)) * R + o_r
    p_e1 = (token_panel * (total_edits + 1) + 1) * R + o_r
    tl.store(out_d + p_d0, d0, mask=m_r)
    tl.store(out_d + p_d1, d1, mask=m_r)
    tl.store(out_paired + p_e0, e0, mask=m_r)
    tl.store(out_paired + p_e1, e1, mask=m_r)

    p_query = (token_panel * (total_edits + 1) + total_edits) * R + o_r
    tl.store(out_paired + p_query, b_q, mask=m_r)
    tl.store(out_diagonal_log + token_head, tl.log(diagonal_h))

    key_offset = batch * SK_B + token * SK_T + head * SK_H + o_r
    b_key_raw = tl.load(keys + key_offset, mask=m_r, other=0.0).to(tl.float32)
    key_rstd = 1.0 / tl.sqrt(tl.sum(b_key_raw * b_key_raw, axis=0) + 1.0e-24)
    b_key = (b_key_raw * key_rstd).to(keys.dtype.element_ty).to(tl.float32)
    erase_offset = batch * SE_B + token * SE_T + head * SE_H + o_r
    b_erase_raw = tl.load(erase_raw + erase_offset, mask=m_r, other=0.0).to(tl.float32)
    # Preserve the former public BF16 gate boundary in registers.
    b_erase = (2.0 * tl.sigmoid(b_erase_raw)).to(erase_raw.dtype.element_ty).to(tl.float32)
    p_d2 = (token_panel * total_edits + 2) * R + o_r
    p_e2 = (token_panel * (total_edits + 1) + 2) * R + o_r
    tl.store(out_d + p_d2, b_key, mask=m_r)
    tl.store(out_paired + p_e2, b_erase * b_key, mask=m_r)
    tl.store(out_q_rstd + token_head, q_rstd)
    tl.store(out_key_rstd + token_head, key_rstd)

    o_v = tl.arange(0, BV)
    m_v = o_v < V
    value_offset = batch * SV_B + token * SV_T + head * SV_H + o_v
    write_offset = batch * SW_B + token * SW_T + head * SW_H + o_v
    b_value = tl.load(values + value_offset, mask=m_v, other=0.0).to(tl.float32)
    b_write_raw = tl.load(write_raw + write_offset, mask=m_v, other=0.0).to(tl.float32)
    b_write = (2.0 * tl.sigmoid(b_write_raw)).to(write_raw.dtype.element_ty).to(tl.float32)
    tl.store(out_z + token_panel * V + o_v, b_write * b_value, mask=m_v)


def block_e3_sources_forward(
    u,
    h,
    q,
    keys,
    values,
    gain,
    prediction,
    geometry_log_decay,
    associative_log_decay,
    erase_raw,
    write_raw,
    previous_mass,
    current_mass,
    strength,
    *,
    token_chunk_size: int,
):
    """Write the native token-major E=3 source panels."""
    batch, length, heads, rank = u.shape
    edits, value_dim = keys.shape[-2], values.shape[-1]
    if edits != 1:
        raise ValueError("block-E3 requires exactly one ordinary edit")
    chunks = triton.cdiv(length, token_chunk_size)
    panels = batch * heads * chunks
    d = torch.empty(
        panels, token_chunk_size, 3, rank, dtype=u.dtype, device=u.device
    )
    paired = torch.empty(
        panels, token_chunk_size, 4, rank, dtype=u.dtype, device=u.device
    )
    z = torch.empty(
        panels, token_chunk_size, 1, value_dim,
        dtype=values.dtype, device=values.device,
    )
    diagonal_log = torch.empty_like(geometry_log_decay)
    q_rstd = torch.empty_like(geometry_log_decay)
    key_rstd = torch.empty_like(geometry_log_decay)
    block_rank = max(triton.next_power_of_2(rank), 16)
    block_value = max(triton.next_power_of_2(value_dim), 16)
    _block_e3_sources_fwd_kernel[(batch * length * heads,)](
        u,
        h,
        q,
        keys,
        values,
        gain,
        prediction,
        geometry_log_decay,
        associative_log_decay,
        erase_raw,
        write_raw,
        previous_mass,
        current_mass,
        strength.reshape(-1),
        d,
        paired,
        z,
        diagonal_log,
        q_rstd,
        key_rstd,
        SH_B=h.stride(0),
        SH_T=h.stride(1),
        SH_H=h.stride(2),
        SQ_B=q.stride(0),
        SQ_T=q.stride(1),
        SQ_H=q.stride(2),
        SK_B=keys.stride(0),
        SK_T=keys.stride(1),
        SK_H=keys.stride(2),
        SV_B=values.stride(0),
        SV_T=values.stride(1),
        SV_H=values.stride(2),
        SE_B=erase_raw.stride(0),
        SE_T=erase_raw.stride(1),
        SE_H=erase_raw.stride(2),
        SW_B=write_raw.stride(0),
        SW_T=write_raw.stride(1),
        SW_H=write_raw.stride(2),
        T=length,
        H=heads,
        R=rank,
        V=value_dim,
        FRAME_CHUNK=token_chunk_size,
        FRAME_CHUNKS=chunks,
        BR=block_rank,
        BV=block_value,
        num_warps=4,
        num_stages=1,
    )
    return d, paired, z, diagonal_log, q_rstd, key_rstd


@triton.autotune(
    configs=[
        triton.Config({"BR": br}, num_warps=warps)
        for br in (16, 32, 64)
        for warps in (2, 4, 8)
    ],
    key=["R", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _block_e3_gate_cumsum_kernel(
    associative_log_decay,
    diagonal_log,
    cumulative,
    T,
    H: tl.constexpr,
    R: tl.constexpr,
    BT: tl.constexpr,
    BR: tl.constexpr,
    SCALE: tl.constexpr,
):
    coordinate = tl.program_id(0) * BR + tl.arange(0, BR)
    chunk = tl.program_id(1).to(tl.int64)
    batch_head = tl.program_id(2).to(tl.int64)
    batch, head = batch_head // H, batch_head % H
    token = chunk * BT + tl.arange(0, BT)
    mask = (token[:, None] < T) & (coordinate[None, :] < R)
    vector_offset = ((batch * T + token[:, None]) * H + head) * R
    vector_offset += coordinate[None, :]
    scalar_offset = (batch * T + token) * H + head
    values = tl.load(
        associative_log_decay + vector_offset, mask=mask, other=0.0
    ).to(tl.float32)
    values += tl.load(
        diagonal_log + scalar_offset,
        mask=token < T,
        other=0.0,
    )[:, None]
    values = tl.cumsum(values, axis=0) * SCALE
    tl.store(cumulative + vector_offset, values, mask=mask)


def block_e3_gate_cumsum(
    associative_log_decay: torch.Tensor,
    diagonal_log: torch.Tensor,
    *,
    token_chunk_size: int,
) -> torch.Tensor:
    """FLA chunk cumsum with the token scalar broadcast in-register."""
    batch, length, heads, rank = associative_log_decay.shape
    if diagonal_log.shape != (batch, length, heads):
        raise ValueError("diagonal_log must have shape [B,T,H]")
    cumulative = torch.empty_like(associative_log_decay)
    _block_e3_gate_cumsum_kernel[
        lambda meta: (
            triton.cdiv(rank, meta["BR"]),
            triton.cdiv(length, token_chunk_size),
            batch * heads,
        )
    ](
        associative_log_decay,
        diagonal_log,
        cumulative,
        T=length,
        H=heads,
        R=rank,
        BT=token_chunk_size,
        SCALE=RCP_LN2,
    )
    return cumulative


__all__ = ["block_e3_gate_cumsum", "block_e3_sources_forward"]
