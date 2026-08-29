# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
# Source ownership follows FLA's MIT-licensed GDN2/KDA reverse schedules.

from __future__ import annotations

import torch
import triton
import triton.language as tl


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
    u_r = tl.load(u + base_r, mask=mask_r, other=0.0).to(tl.float32)
    update_r = tl.load(update + base_r, mask=mask_r, other=0.0).to(tl.float32)
    q_r = tl.load(q + base_r, mask=mask_r, other=0.0).to(tl.float32)
    key_r = tl.load(key + base_r, mask=mask_r, other=0.0).to(tl.float32)
    erase_base = (
        batch * erase_stride_b
        + token * erase_stride_t
        + head * erase_stride_h
        + r
    )
    erase_x = tl.load(erase_raw + erase_base, mask=mask_r, other=0.0).to(tl.float32)
    erase = (2.0 * tl.sigmoid(erase_x)).to(tl.bfloat16).to(tl.float32)
    erase_key = erase * key_r

    den = 1.0 + tl.sum(u_r * update_r, axis=0)
    direct_score = tl.sum(update_r * key_r, axis=0)
    dual_score = tl.sum(u_r * erase_key, axis=0) / den
    query_score = tl.sum(u_r * q_r, axis=0) / den
    direct_offset = (panel * C + panel_row) * rank + r
    dual_offset = ((panel * 2) * C + panel_row) * rank + r
    query_offset = ((panel * 2 + 1) * C + panel_row) * rank + r
    tl.store(direct + direct_offset, key_r + u_r * direct_score, mask=mask_r)
    tl.store(dual + dual_offset, erase_key - update_r * dual_score, mask=mask_r)
    tl.store(query + query_offset, q_r - update_r * query_score, mask=mask_r)

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
    write = (2.0 * tl.sigmoid(write_x)).to(tl.bfloat16).to(tl.float32)
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
    u_r = tl.load(u + base_r, mask=mask_r, other=0.0).to(tl.float32)
    update_r = tl.load(update + base_r, mask=mask_r, other=0.0).to(tl.float32)
    q_r = tl.load(q + base_r, mask=mask_r, other=0.0).to(tl.float32)
    key_r = tl.load(key + base_r, mask=mask_r, other=0.0).to(tl.float32)
    erase_base = (
        batch * erase_stride_b
        + token * erase_stride_t
        + head * erase_stride_h
        + r
    )
    erase_x = tl.load(erase_raw + erase_base, mask=mask_r, other=0.0).to(tl.float32)
    erase_sigmoid = tl.sigmoid(erase_x)
    erase = (2.0 * erase_sigmoid).to(tl.bfloat16).to(tl.float32)
    erase_key = erase * key_r

    direct_offset = (panel * C + panel_row) * rank + r
    dual_offset = ((panel * 2) * C + panel_row) * rank + r
    query_offset = ((panel * 2 + 1) * C + panel_row) * rank + r
    gd = tl.load(grad_direct + direct_offset, mask=mask_r, other=0.0).to(tl.float32)
    ge = tl.load(grad_dual + dual_offset, mask=mask_r, other=0.0).to(tl.float32)
    gchi = tl.load(grad_query + query_offset, mask=mask_r, other=0.0).to(tl.float32)

    den = 1.0 + tl.sum(u_r * update_r, axis=0)
    inv_den = 1.0 / den
    direct_score = tl.sum(update_r * key_r, axis=0)
    dual_numerator = tl.sum(u_r * erase_key, axis=0)
    query_numerator = tl.sum(u_r * q_r, axis=0)
    dual_score = dual_numerator * inv_den
    query_score = query_numerator * inv_den

    gu = gd * direct_score
    gupdate = -ge * dual_score - gchi * query_score
    gkey = gd
    gq = gchi

    g_direct_score = tl.sum(gd * u_r, axis=0)
    gupdate += g_direct_score * key_r
    gkey += g_direct_score * update_r

    g_dual_score = -tl.sum(ge * update_r, axis=0)
    g_query_score = -tl.sum(gchi * update_r, axis=0)
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
    gu += g_den * update_r
    gupdate += g_den * u_r

    g_erase = gerase_key * key_r
    gkey += gerase_key * erase
    g_erase_x = g_erase * (2.0 * erase_sigmoid * (1.0 - erase_sigmoid))
    tl.store(grad_u + base_r, gu, mask=mask_r)
    tl.store(grad_update + base_r, gupdate, mask=mask_r)
    tl.store(grad_q + base_r, gq, mask=mask_r)
    tl.store(grad_key + base_r, gkey, mask=mask_r)
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
    write_sigmoid = tl.sigmoid(write_x)
    write = (2.0 * write_sigmoid).to(tl.bfloat16).to(tl.float32)
    gz = tl.load(grad_injection + base_v, mask=mask_v, other=0.0).to(tl.float32)
    g_value = gz * write
    g_write_x = gz * value_v * (2.0 * write_sigmoid * (1.0 - write_sigmoid))
    tl.store(grad_value + base_v, g_value, mask=mask_v)
    tl.store(grad_write_raw + base_v, g_write_x, mask=mask_v)


class _RelativeSources(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, update, q, key, value, erase_raw, write_raw, chunk_size):
        if not all(tensor.is_contiguous() for tensor in (u, update, q, key)):
            raise ValueError("normalized relative source panels must be contiguous")
        if any(
            tensor.stride(-1) != 1
            for tensor in (value, erase_raw, write_raw)
        ):
            raise ValueError("raw relative sources require unit inner stride")
        rows, rank, value_dim = u.numel() // u.shape[-1], u.shape[-1], value.shape[-1]
        batch, length, heads, _ = u.shape
        chunks = triton.cdiv(length, chunk_size)
        panels = batch * heads * chunks
        direct = torch.empty(
            panels, 1, chunk_size, rank, dtype=key.dtype, device=key.device
        )
        paired = torch.empty(
            panels, 2, chunk_size, rank, dtype=key.dtype, device=key.device
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
            num_warps=4,
        )
        ctx.chunk_size = chunk_size
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(u, update, q, key, value, erase_raw, write_raw)
        return direct, paired, injection

    @staticmethod
    def backward(ctx, gd, gpaired, gz):
        u, update, q, key, value, erase_raw, write_raw = ctx.saved_tensors
        batch, length, heads, rank = u.shape
        chunks = triton.cdiv(length, ctx.chunk_size)
        panels = batch * heads * chunks
        gd = torch.zeros(
            panels, 1, ctx.chunk_size, rank, dtype=key.dtype, device=key.device
        ) if gd is None else gd.contiguous()
        gpaired = torch.zeros(
            panels, 2, ctx.chunk_size, rank, dtype=key.dtype, device=key.device
        ) if gpaired is None else gpaired.contiguous()
        gz = torch.zeros_like(value) if gz is None else gz.contiguous()
        outputs = [
            torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            for tensor in (u, update, q, key, value, erase_raw, write_raw)
        ]
        rows, rank, value_dim = u.numel() // u.shape[-1], u.shape[-1], value.shape[-1]
        _source_bwd_kernel[(rows,)](
            u,
            update,
            q,
            key,
            value,
            erase_raw,
            write_raw,
            gd,
            gpaired,
            gpaired,
            gz,
            *outputs,
            rows=rows,
            rank=rank,
            value_dim=value_dim,
            block_rank=triton.next_power_of_2(rank),
            block_value=triton.next_power_of_2(value_dim),
            T=length,
            H=heads,
            C=ctx.chunk_size,
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
            num_warps=4,
        )
        return (*outputs, None)


def relative_sources(u, update, q, key, value, erase_raw, write_raw, *, chunk_size):
    return _RelativeSources.apply(
        u, update, q, key, value, erase_raw, write_raw, chunk_size
    )


__all__ = ["relative_sources"]
