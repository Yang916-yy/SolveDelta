# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
# Source ownership follows FLA's MIT-licensed GDN2/KDA reverse schedules.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...reference import RELATIVE_FRAME_RADIUS


@triton.jit
def _source_fwd_kernel(
    u,
    update,
    q,
    key,
    value,
    erase_raw,
    write_raw,
    direct,
    dual,
    query,
    injection,
    rows: tl.constexpr,
    rank: tl.constexpr,
    value_dim: tl.constexpr,
    block_rank: tl.constexpr,
    block_value: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    NT: tl.constexpr,
    value_stride_b: tl.constexpr,
    value_stride_t: tl.constexpr,
    value_stride_h: tl.constexpr,
    erase_stride_b: tl.constexpr,
    erase_stride_t: tl.constexpr,
    erase_stride_h: tl.constexpr,
    write_stride_b: tl.constexpr,
    write_stride_t: tl.constexpr,
    write_stride_h: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    key_stride_b: tl.constexpr,
    key_stride_t: tl.constexpr,
    key_stride_h: tl.constexpr,
    frame_radius: tl.constexpr,
):
    row = tl.program_id(0)
    token_flat = row // H
    head = row % H
    batch = token_flat // T
    token = token_flat % T
    panel = (batch * H + head) * NT + token // C
    panel_row = token % C
    r = tl.arange(0, block_rank)
    mask_r = r < rank
    base_r = row * rank + r
    q_base = (
        batch * q_stride_b
        + token * q_stride_t
        + head * q_stride_h
        + r
    )
    key_base = (
        batch * key_stride_b
        + token * key_stride_t
        + head * key_stride_h
        + r
    )
    u_r = tl.load(u + base_r, mask=mask_r, other=0.0).to(tl.float32)
    update_r = tl.load(update + base_r, mask=mask_r, other=0.0).to(tl.float32)
    q_raw_r = tl.load(q + q_base, mask=mask_r, other=0.0).to(tl.float32)
    key_raw_r = tl.load(key + key_base, mask=mask_r, other=0.0).to(tl.float32)
    q_rstd = 1.0 / tl.sqrt(tl.sum(q_raw_r * q_raw_r, axis=0) + 1.0e-24)
    key_rstd = 1.0 / tl.sqrt(
        tl.sum(key_raw_r * key_raw_r, axis=0) + 1.0e-24
    )
    q_r = q_raw_r * q_rstd
    key_r = key_raw_r * key_rstd
    erase_base = (
        batch * erase_stride_b
        + token * erase_stride_t
        + head * erase_stride_h
        + r
    )
    erase_x = tl.load(erase_raw + erase_base, mask=mask_r, other=0.0).to(tl.float32)
    erase = tl.sigmoid(erase_x)
    erase_key = erase * key_r

    update_norm_sq = tl.sum(update_r * update_r, axis=0)
    radial_denom = frame_radius * frame_radius + update_norm_sq
    frame_scale = frame_radius / tl.sqrt(radial_denom)
    frame_r = frame_scale * update_r
    den = 1.0 + tl.sum(u_r * frame_r, axis=0)
    direct_score = tl.sum(frame_r * key_r, axis=0)
    dual_score = tl.sum(u_r * erase_key, axis=0) / den
    query_score = tl.sum(u_r * q_r, axis=0) / den
    direct_offset = (panel * C + panel_row) * rank + r
    dual_offset = ((panel * 2) * C + panel_row) * rank + r
    query_offset = ((panel * 2 + 1) * C + panel_row) * rank + r
    tl.store(direct + direct_offset, key_r + u_r * direct_score, mask=mask_r)
    tl.store(dual + dual_offset, erase_key - frame_r * dual_score, mask=mask_r)
    tl.store(query + query_offset, q_r - frame_r * query_score, mask=mask_r)

    v = tl.arange(0, block_value)
    mask_v = v < value_dim
    base_v = row * value_dim + v
    value_base = (
        batch * value_stride_b
        + token * value_stride_t
        + head * value_stride_h
        + v
    )
    write_base = (
        batch * write_stride_b
        + token * write_stride_t
        + head * write_stride_h
        + v
    )
    value_v = tl.load(value + value_base, mask=mask_v, other=0.0).to(tl.float32)
    write_x = tl.load(write_raw + write_base, mask=mask_v, other=0.0).to(tl.float32)
    write = tl.sigmoid(write_x)
    tl.store(injection + base_v, write * value_v, mask=mask_v)


@triton.jit
def _source_bwd_kernel(
    u,
    update,
    q,
    key,
    value,
    erase_raw,
    write_raw,
    grad_direct,
    grad_dual,
    grad_query,
    grad_injection,
    grad_u,
    grad_update,
    grad_q,
    grad_key,
    grad_value,
    grad_erase_raw,
    grad_write_raw,
    rows: tl.constexpr,
    rank: tl.constexpr,
    value_dim: tl.constexpr,
    block_rank: tl.constexpr,
    block_value: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    NT: tl.constexpr,
    value_stride_b: tl.constexpr,
    value_stride_t: tl.constexpr,
    value_stride_h: tl.constexpr,
    erase_stride_b: tl.constexpr,
    erase_stride_t: tl.constexpr,
    erase_stride_h: tl.constexpr,
    write_stride_b: tl.constexpr,
    write_stride_t: tl.constexpr,
    write_stride_h: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    key_stride_b: tl.constexpr,
    key_stride_t: tl.constexpr,
    key_stride_h: tl.constexpr,
    frame_radius: tl.constexpr,
):
    row = tl.program_id(0)
    token_flat = row // H
    head = row % H
    batch = token_flat // T
    token = token_flat % T
    panel = (batch * H + head) * NT + token // C
    panel_row = token % C
    r = tl.arange(0, block_rank)
    mask_r = r < rank
    base_r = row * rank + r
    q_base = (
        batch * q_stride_b
        + token * q_stride_t
        + head * q_stride_h
        + r
    )
    key_base = (
        batch * key_stride_b
        + token * key_stride_t
        + head * key_stride_h
        + r
    )
    u_r = tl.load(u + base_r, mask=mask_r, other=0.0).to(tl.float32)
    update_r = tl.load(update + base_r, mask=mask_r, other=0.0).to(tl.float32)
    q_raw_r = tl.load(q + q_base, mask=mask_r, other=0.0).to(tl.float32)
    key_raw_r = tl.load(key + key_base, mask=mask_r, other=0.0).to(tl.float32)
    q_rstd = 1.0 / tl.sqrt(tl.sum(q_raw_r * q_raw_r, axis=0) + 1.0e-24)
    key_rstd = 1.0 / tl.sqrt(
        tl.sum(key_raw_r * key_raw_r, axis=0) + 1.0e-24
    )
    q_r = q_raw_r * q_rstd
    key_r = key_raw_r * key_rstd
    erase_base = (
        batch * erase_stride_b
        + token * erase_stride_t
        + head * erase_stride_h
        + r
    )
    erase_x = tl.load(erase_raw + erase_base, mask=mask_r, other=0.0).to(tl.float32)
    erase = tl.sigmoid(erase_x)
    erase_key = erase * key_r

    direct_offset = (panel * C + panel_row) * rank + r
    dual_offset = ((panel * 2) * C + panel_row) * rank + r
    query_offset = ((panel * 2 + 1) * C + panel_row) * rank + r
    gd = tl.load(grad_direct + direct_offset, mask=mask_r, other=0.0).to(tl.float32)
    ge = tl.load(grad_dual + dual_offset, mask=mask_r, other=0.0).to(tl.float32)
    gchi = tl.load(grad_query + query_offset, mask=mask_r, other=0.0).to(tl.float32)

    update_norm_sq = tl.sum(update_r * update_r, axis=0)
    radial_denom = frame_radius * frame_radius + update_norm_sq
    frame_scale = frame_radius / tl.sqrt(radial_denom)
    frame_r = frame_scale * update_r
    den = 1.0 + tl.sum(u_r * frame_r, axis=0)
    inv_den = 1.0 / den
    direct_score = tl.sum(frame_r * key_r, axis=0)
    dual_numerator = tl.sum(u_r * erase_key, axis=0)
    query_numerator = tl.sum(u_r * q_r, axis=0)
    dual_score = dual_numerator * inv_den
    query_score = query_numerator * inv_den

    gu = gd * direct_score
    gframe = -ge * dual_score - gchi * query_score
    gkey = gd
    gq = gchi

    g_direct_score = tl.sum(gd * u_r, axis=0)
    gframe += g_direct_score * key_r
    gkey += g_direct_score * frame_r

    g_dual_score = -tl.sum(ge * frame_r, axis=0)
    g_query_score = -tl.sum(gchi * frame_r, axis=0)
    g_dual_numerator = g_dual_score * inv_den
    g_query_numerator = g_query_score * inv_den
    g_inv_den = (
        g_dual_score * dual_numerator + g_query_score * query_numerator
    )
    g_den = -g_inv_den * inv_den * inv_den

    gerase_key = ge + g_dual_numerator * u_r
    gu += g_dual_numerator * erase_key
    gq += g_query_numerator * u_r
    gu += g_query_numerator * q_r
    gu += g_den * frame_r
    gframe += g_den * u_r

    g_frame_scale = tl.sum(gframe * update_r, axis=0)
    radial_common = g_frame_scale * frame_scale / radial_denom
    gupdate = (
        frame_scale * gframe
        - radial_common * update_r
    )

    g_erase = gerase_key * key_r
    gkey += gerase_key * erase
    g_erase_x = g_erase * erase * (1.0 - erase)
    tl.store(grad_u + base_r, gu, mask=mask_r)
    tl.store(grad_update + base_r, gupdate, mask=mask_r)
    gq_raw = (gq - tl.sum(gq * q_r, axis=0) * q_r) * q_rstd
    gkey_raw = (
        gkey - tl.sum(gkey * key_r, axis=0) * key_r
    ) * key_rstd
    tl.store(grad_q + base_r, gq_raw, mask=mask_r)
    tl.store(grad_key + base_r, gkey_raw, mask=mask_r)
    tl.store(grad_erase_raw + base_r, g_erase_x, mask=mask_r)

    v = tl.arange(0, block_value)
    mask_v = v < value_dim
    base_v = row * value_dim + v
    value_base = (
        batch * value_stride_b
        + token * value_stride_t
        + head * value_stride_h
        + v
    )
    write_base = (
        batch * write_stride_b
        + token * write_stride_t
        + head * write_stride_h
        + v
    )
    value_v = tl.load(value + value_base, mask=mask_v, other=0.0).to(tl.float32)
    write_x = tl.load(write_raw + write_base, mask=mask_v, other=0.0).to(tl.float32)
    write = tl.sigmoid(write_x)
    gz = tl.load(grad_injection + base_v, mask=mask_v, other=0.0).to(tl.float32)
    g_value = gz * write
    g_write_x = gz * value_v * write * (1.0 - write)
    tl.store(grad_value + base_v, g_value, mask=mask_v)
    tl.store(grad_write_raw + base_v, g_write_x, mask=mask_v)


def relative_sources_forward(
    u, update, q, key, value, erase_raw, write_raw, *, chunk_size
):
    if not all(tensor.is_contiguous() for tensor in (u, update)):
        raise ValueError("normalized geometry source panels must be contiguous")
    if any(
        tensor.stride(-1) != 1
        for tensor in (q, key, value, erase_raw, write_raw)
    ):
        raise ValueError("raw relative sources require unit inner stride")
    rows, rank, value_dim = u.numel() // u.shape[-1], u.shape[-1], value.shape[-1]
    batch, length, heads, _ = u.shape
    chunks = triton.cdiv(length, chunk_size)
    panels = batch * heads * chunks
    # These panels are statically bounded by source normalization and the
    # relative-frame condition bound.  Write FP16 directly from the FP32
    # producer; decay-scaled DPLR operands are materialized separately.
    direct = torch.empty(
        panels, 1, chunk_size, rank, dtype=torch.float16, device=key.device
    )
    paired = torch.empty(
        panels, 2, chunk_size, rank, dtype=torch.float16, device=key.device
    )
    injection = torch.empty_like(value)
    _source_fwd_kernel[(rows,)](
        u,
        update,
        q,
        key,
        value,
        erase_raw,
        write_raw,
        direct,
        paired,
        paired,
        injection,
        rows=rows,
        rank=rank,
        value_dim=value_dim,
        block_rank=triton.next_power_of_2(rank),
        block_value=triton.next_power_of_2(value_dim),
        T=length,
        H=heads,
        C=chunk_size,
        NT=chunks,
        value_stride_b=value.stride(0),
        value_stride_t=value.stride(1),
        value_stride_h=value.stride(2),
        erase_stride_b=erase_raw.stride(0),
        erase_stride_t=erase_raw.stride(1),
        erase_stride_h=erase_raw.stride(2),
        write_stride_b=write_raw.stride(0),
        write_stride_t=write_raw.stride(1),
        write_stride_h=write_raw.stride(2),
        q_stride_b=q.stride(0),
        q_stride_t=q.stride(1),
        q_stride_h=q.stride(2),
        key_stride_b=key.stride(0),
        key_stride_t=key.stride(1),
        key_stride_h=key.stride(2),
        frame_radius=RELATIVE_FRAME_RADIUS,
        num_warps=1,
    )
    return direct, paired, injection


def relative_sources_backward(
    u,
    update,
    q,
    key,
    value,
    erase_raw,
    write_raw,
    grad_direct,
    grad_paired,
    grad_injection,
    *,
    chunk_size,
):
    batch, length, heads, rank = u.shape
    chunks = triton.cdiv(length, chunk_size)
    panels = batch * heads * chunks
    grad_direct = torch.zeros(
        panels, 1, chunk_size, rank, dtype=key.dtype, device=key.device
    ) if grad_direct is None else grad_direct.contiguous()
    grad_paired = torch.zeros(
        panels, 2, chunk_size, rank, dtype=key.dtype, device=key.device
    ) if grad_paired is None else grad_paired.contiguous()
    grad_injection = (
        torch.zeros_like(value)
        if grad_injection is None
        else grad_injection.contiguous()
    )
    outputs = [torch.empty_like(tensor) for tensor in (
        u, update, q, key, value, erase_raw, write_raw
    )]
    rows, value_dim = u.numel() // rank, value.shape[-1]
    _source_bwd_kernel[(rows,)](
        u,
        update,
        q,
        key,
        value,
        erase_raw,
        write_raw,
        grad_direct,
        grad_paired,
        grad_paired,
        grad_injection,
        *outputs,
        rows=rows,
        rank=rank,
        value_dim=value_dim,
        block_rank=triton.next_power_of_2(rank),
        block_value=triton.next_power_of_2(value_dim),
        T=length,
        H=heads,
        C=chunk_size,
        NT=chunks,
        value_stride_b=value.stride(0),
        value_stride_t=value.stride(1),
        value_stride_h=value.stride(2),
        erase_stride_b=erase_raw.stride(0),
        erase_stride_t=erase_raw.stride(1),
        erase_stride_h=erase_raw.stride(2),
        write_stride_b=write_raw.stride(0),
        write_stride_t=write_raw.stride(1),
        write_stride_h=write_raw.stride(2),
        q_stride_b=q.stride(0),
        q_stride_t=q.stride(1),
        q_stride_h=q.stride(2),
        key_stride_b=key.stride(0),
        key_stride_t=key.stride(1),
        key_stride_h=key.stride(2),
        frame_radius=RELATIVE_FRAME_RADIUS,
        num_warps=1,
    )
    return tuple(outputs)


class _RelativeSources(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, update, q, key, value, erase_raw, write_raw, chunk_size):
        outputs = relative_sources_forward(
            u, update, q, key, value, erase_raw, write_raw,
            chunk_size=chunk_size,
        )
        ctx.chunk_size = chunk_size
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(u, update, q, key, value, erase_raw, write_raw)
        return outputs

    @staticmethod
    def backward(ctx, gd, gpaired, gz):
        outputs = relative_sources_backward(
            *ctx.saved_tensors,
            gd,
            gpaired,
            gz,
            chunk_size=ctx.chunk_size,
        )
        return (*outputs, None)


def relative_sources(u, update, q, key, value, erase_raw, write_raw, *, chunk_size):
    return _RelativeSources.apply(
        u, update, q, key, value, erase_raw, write_raw, chunk_size
    )


__all__ = ["relative_sources"]
