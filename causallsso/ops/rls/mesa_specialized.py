# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
"""Dense RLS specializations of FLA MESA's paired state and CG owners.

The donor kernels are MIT-licensed FLA MESA.  This module keeps their tile
ownership and transpose schedules while deleting two structurally constant
inputs of the RLS route: ``beta == 1`` and ``ridge == 0``.  It intentionally
supports only the dense equal-length surface used by the production operator.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp2, safe_dot
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs


_KV_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@triton.jit
def _matrix_action(x, key, value, causal_decay, boundary_decay, boundary):
    local = tl.dot(
        (tl.dot(x.to(key.dtype), tl.trans(key)) * causal_decay).to(value.dtype),
        value,
    )
    return local + tl.dot((x * boundary_decay).to(boundary.dtype), boundary)


@triton.jit(do_not_specialize=["T"])
def _cg_fwd_kernel(
    q,
    key,
    value,
    boundary,
    boundary_kv,
    local_decay,
    gain,
    prediction,
    VALUE_STRIDE_B: tl.constexpr,
    VALUE_STRIDE_T: tl.constexpr,
    VALUE_STRIDE_H: tl.constexpr,
    VALUE_STRIDE_D: tl.constexpr,
    T,
    STEPS: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
):
    chunk, batch_head = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    batch, head = batch_head // H, batch_head % H
    chunks = tl.cdiv(T, C)
    chunk_global = batch * chunks + chunk
    bos = batch * T
    token = chunk * C + tl.arange(0, C)
    coord = tl.arange(0, BK)
    mt = token < T
    mk = coord < K
    mtk = mt[:, None] & mk[None, :]
    mkk = mk[:, None] & mk[None, :]

    q_ptr = q + ((bos + token[:, None]) * H + head) * K + coord[None, :]
    key_ptr = key + ((bos + token[:, None]) * H + head) * K + coord[None, :]
    state_ptr = boundary + (chunk_global * H + head) * K * K
    state_ptr += coord[:, None] * K + coord[None, :]
    kv_ptr = boundary_kv + (chunk_global * H + head) * K * K
    kv_ptr += coord[:, None] * K + coord[None, :]
    g_ptr = local_decay + (bos + token) * H + head

    rhs = tl.load(q_ptr, mask=mtk, other=0.0).to(tl.float32)
    k_tile = tl.load(key_ptr, mask=mtk, other=0.0)
    h_tile = tl.load(state_ptr, mask=mkk, other=0.0)
    g = tl.load(g_ptr, mask=mt, other=0.0).to(tl.float32)
    causal = tl.where(
        (token[:, None] >= token[None, :]) & (mt[:, None] & mt[None, :]),
        exp2(g[:, None] - g[None, :]),
        0.0,
    )
    boundary_scale = exp2(g)[:, None]

    x = tl.zeros((C, BK), tl.float32)
    residual = rhs
    direction = rhs
    residual_norm = tl.sum(residual * residual, axis=1)
    for _ in range(STEPS):
        action = _matrix_action(
            direction, k_tile, k_tile, causal, boundary_scale, h_tile
        )
        alpha = residual_norm / (tl.sum(direction * action, axis=1) + 1e-5)
        x += alpha[:, None] * direction
        residual -= alpha[:, None] * action
        next_norm = tl.sum(residual * residual, axis=1)
        direction = residual + (next_norm / (residual_norm + 1e-5))[:, None] * direction
        residual_norm = next_norm

    gain_ptr = gain + ((bos + token[:, None]) * H + head) * K + coord[None, :]
    tl.store(gain_ptr, x.to(gain_ptr.dtype.element_ty), mask=mtk)

    v_ptr = value + (
        batch * VALUE_STRIDE_B
        + token[:, None] * VALUE_STRIDE_T
        + head * VALUE_STRIDE_H
        + coord[None, :] * VALUE_STRIDE_D
    )
    v_tile = tl.load(v_ptr, mask=mtk, other=0.0)
    h_kv = tl.load(kv_ptr, mask=mkk, other=0.0)
    result = _matrix_action(x, k_tile, v_tile, causal, boundary_scale, h_kv)
    out_ptr = prediction + ((bos + token[:, None]) * H + head) * K + coord[None, :]
    tl.store(out_ptr, result.to(out_ptr.dtype.element_ty), mask=mtk)


@triton.jit(do_not_specialize=["T"])
def _cg_transpose_hkv_kernel(
    gain,
    key,
    value,
    boundary_kv,
    boundary,
    local_decay,
    output_cotangent,
    grad_decay_previous,
    gain_cotangent,
    result,
    grad_decay,
    VALUE_STRIDE_B: tl.constexpr,
    VALUE_STRIDE_T: tl.constexpr,
    VALUE_STRIDE_H: tl.constexpr,
    VALUE_STRIDE_D: tl.constexpr,
    T,
    STEPS: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
):
    chunk, batch_head = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    batch, head = batch_head // H, batch_head % H
    chunks = tl.cdiv(T, C)
    chunk_global = batch * chunks + chunk
    bos = batch * T
    token = chunk * C + tl.arange(0, C)
    coord = tl.arange(0, BK)
    mt = token < T
    mk = coord < K
    mtk = mt[:, None] & mk[None, :]
    mkk = mk[:, None] & mk[None, :]

    q_ptr = gain + ((bos + token[:, None]) * H + head) * K + coord[None, :]
    key_ptr = key + ((bos + token[:, None]) * H + head) * K + coord[None, :]
    value_ptr = value + (
        batch * VALUE_STRIDE_B
        + token[:, None] * VALUE_STRIDE_T
        + head * VALUE_STRIDE_H
        + coord[None, :] * VALUE_STRIDE_D
    )
    do_ptr = output_cotangent + ((bos + token[:, None]) * H + head) * K + coord[None, :]
    hkv_ptr = boundary_kv + (chunk_global * H + head) * K * K
    hkv_ptr += coord[:, None] + coord[None, :] * K
    g_ptr = local_decay + (bos + token) * H + head

    q = tl.load(q_ptr, mask=mtk, other=0.0)
    k_tile = tl.load(key_ptr, mask=mtk, other=0.0)
    v = tl.load(value_ptr, mask=mtk, other=0.0)
    do = tl.load(do_ptr, mask=mtk, other=0.0)
    hkv = tl.load(hkv_ptr, mask=mkk, other=0.0)
    g = tl.load(g_ptr, mask=mt, other=0.0).to(tl.float32)
    causal = tl.where(
        (token[:, None] >= token[None, :]) & (mt[:, None] & mt[None, :]),
        exp2(g[:, None] - g[None, :]),
        0.0,
    )
    dscore = tl.dot(do, tl.trans(v)) * causal
    rhs = tl.dot(do, hkv.to(do.dtype)) * exp2(g)[:, None]
    decay = tl.sum(rhs * q, axis=1)
    rhs += tl.dot(dscore.to(k_tile.dtype), k_tile)
    rhs += tl.load(gain_cotangent + ((bos + token[:, None]) * H + head) * K + coord[None, :], mask=mtk, other=0.0)
    decay += tl.load(grad_decay_previous + (bos + token) * H + head, mask=mt, other=0.0)

    state_ptr = boundary + (chunk_global * H + head) * K * K
    state_ptr += coord[:, None] * K + coord[None, :]
    hkk = tl.load(state_ptr, mask=mkk, other=0.0)
    boundary_scale = exp2(g)[:, None]
    x = tl.zeros((C, BK), tl.float32)
    residual = rhs.to(tl.float32)
    direction = residual
    residual_norm = tl.sum(residual * residual, axis=1)
    for _ in range(STEPS):
        action = _matrix_action(
            direction, k_tile, k_tile, causal, boundary_scale, hkk
        )
        alpha = residual_norm / (tl.sum(direction * action, axis=1) + 1e-5)
        x += alpha[:, None] * direction
        residual -= alpha[:, None] * action
        next_norm = tl.sum(residual * residual, axis=1)
        direction = residual + (next_norm / (residual_norm + 1e-5))[:, None] * direction
        residual_norm = next_norm

    tl.store(result + ((bos + token[:, None]) * H + head) * K + coord[None, :], x.to(result.dtype.element_ty), mask=mtk)
    tl.store(grad_decay + (bos + token) * H + head, decay, mask=mt)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [1, 2, 4, 8]
        for s in [2, 3, 4]
    ],
    key=["C"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _paired_state_fwd_kernel(
    key,
    value,
    local_decay,
    boundary,
    boundary_kv,
    initial,
    initial_kv,
    final,
    final_kv,
    VALUE_STRIDE_B: tl.constexpr,
    VALUE_STRIDE_T: tl.constexpr,
    VALUE_STRIDE_H: tl.constexpr,
    VALUE_STRIDE_D: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    key_block, value_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2).to(tl.int64),
    )
    batch, head = batch_head // H, batch_head % H
    chunks = tl.cdiv(T, C)
    bos = batch * T
    coord_k = key_block * BK + tl.arange(0, BK)
    coord_v = value_block * BV + tl.arange(0, BV)
    mask_state = (coord_k[:, None] < K) & (coord_v[None, :] < K)
    initial_ptr = initial + batch_head * K * K
    initial_ptr += coord_k[:, None] * K + coord_v[None, :]
    initial_kv_ptr = initial_kv + batch_head * K * K
    initial_kv_ptr += coord_k[:, None] * K + coord_v[None, :]
    state = tl.load(initial_ptr, mask=mask_state, other=0.0).to(tl.float32)
    state_kv = tl.load(initial_kv_ptr, mask=mask_state, other=0.0).to(tl.float32)

    for chunk in range(chunks):
        token = chunk * C + tl.arange(0, C)
        mt = token < T
        boundary_ptr = boundary + ((batch * chunks + chunk) * H + head) * K * K
        boundary_ptr += coord_k[:, None] * K + coord_v[None, :]
        boundary_kv_ptr = boundary_kv + ((batch * chunks + chunk) * H + head) * K * K
        boundary_kv_ptr += coord_k[:, None] * K + coord_v[None, :]
        tl.store(boundary_ptr, state.to(boundary_ptr.dtype.element_ty), mask=mask_state)
        tl.store(boundary_kv_ptr, state_kv.to(boundary_kv_ptr.dtype.element_ty), mask=mask_state)

        key_ptr = key + ((bos + token[:, None]) * H + head) * K + coord_k[None, :]
        key_v_ptr = key + ((bos + token[:, None]) * H + head) * K + coord_v[None, :]
        value_ptr = value + (
            batch * VALUE_STRIDE_B
            + token[:, None] * VALUE_STRIDE_T
            + head * VALUE_STRIDE_H
            + coord_v[None, :] * VALUE_STRIDE_D
        )
        k = tl.load(key_ptr, mask=mt[:, None] & (coord_k[None, :] < K), other=0.0)
        k_v = tl.load(key_v_ptr, mask=mt[:, None] & (coord_v[None, :] < K), other=0.0)
        v = tl.load(value_ptr, mask=mt[:, None] & (coord_v[None, :] < K), other=0.0)
        last = tl.minimum((chunk + 1) * C, T) - 1
        g_last = tl.load(local_decay + (bos + last) * H + head)
        g = tl.load(local_decay + (bos + token) * H + head, mask=mt, other=0.0)
        state *= exp2(g_last)
        state_kv *= exp2(g_last)
        k_decay = (k * exp2(g_last - g)[:, None]).to(k_v.dtype)
        state += safe_dot(tl.trans(k_decay), k_v)
        state_kv += safe_dot(tl.trans(k_decay), v.to(k_v.dtype))

    final_ptr = final + batch_head * K * K
    final_ptr += coord_k[:, None] * K + coord_v[None, :]
    final_kv_ptr = final_kv + batch_head * K * K
    final_kv_ptr += coord_k[:, None] * K + coord_v[None, :]
    tl.store(final_ptr, state.to(final_ptr.dtype.element_ty), mask=mask_state)
    tl.store(final_kv_ptr, state_kv.to(final_kv_ptr.dtype.element_ty), mask=mask_state)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in _KV_WARPS
        for s in [2, 3, 4]
    ],
    key=["H", "K", "C", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _hkv_reverse_dkv_kernel(
    gain,
    key,
    value,
    boundary_kv,
    local_decay,
    output_cotangent,
    boundary_cotangent,
    grad_key,
    grad_decay,
    grad_value,
    VALUE_STRIDE_B: tl.constexpr,
    VALUE_STRIDE_T: tl.constexpr,
    VALUE_STRIDE_H: tl.constexpr,
    VALUE_STRIDE_D: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    chunk, batch_head = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    batch, head = batch_head // H, batch_head % H
    chunks = tl.cdiv(T, C)
    bos = batch * T
    chunk_global = batch * chunks + chunk
    token = chunk * C + tl.arange(0, C)
    ck, cv = tl.arange(0, BK), tl.arange(0, BV)
    mt = token < T
    mtk = mt[:, None] & (ck[None, :] < K)
    mtv = mt[:, None] & (cv[None, :] < K)
    mh = (cv[:, None] < K) & (ck[None, :] < K)

    gain_ptr = gain + ((bos + token[:, None]) * H + head) * K + ck[None, :]
    key_ptr = key + ((bos + token[:, None]) * H + head) * K + ck[None, :]
    value_ptr = value + (
        batch * VALUE_STRIDE_B
        + token[:, None] * VALUE_STRIDE_T
        + head * VALUE_STRIDE_H
        + cv[None, :] * VALUE_STRIDE_D
    )
    do_ptr = output_cotangent + ((bos + token[:, None]) * H + head) * K + cv[None, :]
    h_ptr = boundary_kv + (chunk_global * H + head) * K * K
    h_ptr += cv[:, None] + ck[None, :] * K
    dh_ptr = boundary_cotangent + (chunk_global * H + head) * K * K
    dh_ptr += cv[:, None] + ck[None, :] * K
    g_ptr = local_decay + (bos + token) * H + head

    q = tl.load(gain_ptr, mask=mtk, other=0.0)
    k = tl.load(key_ptr, mask=mtk, other=0.0)
    v = tl.load(value_ptr, mask=mtv, other=0.0)
    do = tl.load(do_ptr, mask=mtv, other=0.0)
    h = tl.load(h_ptr, mask=mh, other=0.0)
    dh = tl.load(dh_ptr, mask=mh, other=0.0)
    g = tl.load(g_ptr, mask=mt, other=0.0)
    last = tl.minimum((chunk + 1) * C, T) - 1
    g_last = tl.load(local_decay + (bos + last) * H + head)

    dg = tl.zeros((C,), tl.float32)
    # Hkv and its cotangent are low-precision private states.  Their scalar
    # contraction is a sensitive backward reduction, so accumulate it in FP32.
    dg_last = tl.sum(h.to(tl.float32) * dh.to(tl.float32)) * exp2(g_last)
    causal = tl.where(
        (token[:, None] >= token[None, :]) & (mt[:, None] & mt[None, :]),
        exp2(g[:, None] - g[None, :]),
        0.0,
    )
    score = tl.dot(q, tl.trans(k)) * causal
    dscore = tl.dot(do, tl.trans(v))
    decay_cotangent = tl.where(
        tl.arange(0, C)[:, None] >= tl.arange(0, C)[None, :],
        score * dscore,
        0.0,
    )
    dg += tl.sum(decay_cotangent, axis=1) - tl.sum(decay_cotangent, axis=0)
    dscore *= causal
    key_grad = tl.dot(v, dh.to(v.dtype)) * exp2(-g + g_last)[:, None]
    dg_last += tl.sum(key_grad * k)
    dg -= tl.sum(key_grad * k, axis=1)
    value_grad = tl.dot(k, tl.trans(dh).to(k.dtype)) * exp2(-g + g_last)[:, None]
    value_grad += tl.dot(tl.trans(score.to(do.dtype)), do)
    key_grad += tl.dot(tl.trans(dscore.to(q.dtype)), q)
    dg = tl.where(token < last, dg, dg + dg_last)

    grad_key_ptr = grad_key + ((bos + token[:, None]) * H + head) * K + ck[None, :]
    grad_value_ptr = grad_value + ((bos + token[:, None]) * H + head) * K + cv[None, :]
    grad_decay_ptr = grad_decay + (bos + token) * H + head
    tl.store(grad_key_ptr, key_grad.to(grad_key_ptr.dtype.element_ty), mask=mtk)
    tl.store(grad_value_ptr, value_grad.to(grad_value_ptr.dtype.element_ty), mask=mtv)
    tl.store(grad_decay_ptr, dg.to(grad_decay_ptr.dtype.element_ty), mask=mt)


@triton.jit(do_not_specialize=["T"])
def _hkk_reverse_kernel(
    key,
    boundary,
    boundary_cotangent,
    local_decay,
    gain,
    gain_cotangent,
    key_seed,
    grad_key,
    grad_decay,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
):
    chunk, batch_head = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    batch, head = batch_head // H, batch_head % H
    chunks = tl.cdiv(T, C)
    bos = batch * T
    chunk_global = batch * chunks + chunk
    token = chunk * C + tl.arange(0, C)
    coord = tl.arange(0, BK)
    mt = token < T
    mtk = mt[:, None] & (coord[None, :] < K)
    mkk = (coord[:, None] < K) & (coord[None, :] < K)
    k = tl.load(
        key + ((bos + token[:, None]) * H + head) * K + coord[None, :],
        mask=mtk,
        other=0.0,
    )
    q = tl.load(
        gain + ((bos + token[:, None]) * H + head) * K + coord[None, :],
        mask=mtk,
        other=0.0,
    )
    dq = tl.load(
        gain_cotangent + ((bos + token[:, None]) * H + head) * K + coord[None, :],
        mask=mtk,
        other=0.0,
    )
    dh_ptr = boundary_cotangent + (chunk_global * H + head) * K * K
    dh_ptr += coord[:, None] * K + coord[None, :]
    h_ptr = boundary + (chunk_global * H + head) * K * K
    h_ptr += coord[:, None] * K + coord[None, :]
    h = tl.load(h_ptr, mask=mkk, other=0.0)
    dh = tl.load(dh_ptr, mask=mkk, other=0.0)
    g = tl.load(local_decay + (bos + token) * H + head, mask=mt, other=0.0)
    last = tl.minimum((chunk + 1) * C, T) - 1
    g_last = tl.load(local_decay + (bos + last) * H + head)
    gk = tl.where(mt, exp2(g_last - g), 0.0)
    causal = tl.where(
        (token[:, None] >= token[None, :]) & (mt[:, None] & mt[None, :]),
        exp2(g[:, None] - g[None, :]),
        0.0,
    )

    key_grad = tl.zeros((C, BK), tl.float32)
    value_grad = tl.zeros((C, BK), tl.float32)
    decay_grad = tl.zeros((C,), tl.float32)
    decay_last = tl.zeros((1,), tl.float32)
    score = tl.dot(q, tl.trans(k)) * causal
    dscore = tl.dot(dq, tl.trans(k))
    value_grad += tl.dot(tl.trans(score.to(dq.dtype)), dq)
    dm = tl.where(
        tl.arange(0, C)[:, None] >= tl.arange(0, C)[None, :],
        score * dscore,
        0.0,
    )
    decay_grad += tl.sum(dm, axis=1) - tl.sum(dm, axis=0)
    key_grad += tl.dot(tl.trans((dscore * causal).to(q.dtype)), q)
    decay_grad += tl.sum(tl.dot(dq, tl.trans(h)) * exp2(g)[:, None] * q, axis=1)
    boundary_key = tl.dot(k, dh.to(k.dtype)) * gk[:, None]
    decay_grad -= tl.sum(boundary_key * k, axis=1)
    decay_last += tl.sum(boundary_key * k)
    key_grad += boundary_key
    value_grad += tl.dot(k, tl.trans(dh).to(k.dtype)) * gk[:, None]
    decay_last += tl.sum(
        dh.to(tl.float32) * h.to(tl.float32)
    ) * exp2(g_last)
    seed = tl.load(
        key_seed + ((bos + token[:, None]) * H + head) * K + coord[None, :],
        mask=mtk,
        other=0.0,
    )
    key_grad = -(key_grad - seed + value_grad)
    decay_grad = -tl.where(token < last, decay_grad, decay_grad + decay_last)
    grad_key_ptr = grad_key + ((bos + token[:, None]) * H + head) * K + coord[None, :]
    tl.store(grad_key_ptr, key_grad.to(grad_key_ptr.dtype.element_ty), mask=mtk)
    tl.store(grad_decay + (bos + token) * H + head, decay_grad, mask=mt)


def paired_state_forward(key, value, local_decay, initial, initial_kv, *, chunk_size):
    batch, length, heads, rank = key.shape
    chunks = triton.cdiv(length, chunk_size)
    boundary = torch.empty(
        batch, chunks, heads, rank, rank, dtype=key.dtype, device=key.device
    )
    # The cross-moment contains unnormalized values and has no FP16 range
    # bound. Match the MESA NaN fix by retaining BF16 exponent range even
    # when the diagnostic public path uses FP16.
    kv_dtype = torch.bfloat16 if key.dtype == torch.float16 else key.dtype
    boundary_kv = torch.empty(
        batch, chunks, heads, rank, rank, dtype=kv_dtype, device=key.device
    )
    final = torch.empty_like(initial, dtype=torch.float32)
    final_kv = torch.empty_like(initial_kv, dtype=torch.float32)
    _paired_state_fwd_kernel[
        (triton.cdiv(rank, 64), triton.cdiv(rank, 64), batch * heads)
    ](
        key,
        value,
        local_decay,
        boundary,
        boundary_kv,
        initial,
        initial_kv,
        final,
        final_kv,
        VALUE_STRIDE_B=value.stride(0),
        VALUE_STRIDE_T=value.stride(1),
        VALUE_STRIDE_H=value.stride(2),
        VALUE_STRIDE_D=value.stride(3),
        T=length,
        H=heads,
        K=rank,
        C=chunk_size,
        BK=64,
        BV=64,
    )
    return boundary, boundary_kv, final, final_kv


def cg_forward(q, key, value, boundary, boundary_kv, local_decay, *, chunk_size, steps):
    batch, length, heads, rank = q.shape
    gain = torch.empty_like(q)
    prediction = torch.empty(value.shape, dtype=value.dtype, device=value.device)
    _cg_fwd_kernel[(triton.cdiv(length, chunk_size), batch * heads)](
        q,
        key,
        value,
        boundary,
        boundary_kv,
        local_decay,
        gain,
        prediction,
        VALUE_STRIDE_B=value.stride(0),
        VALUE_STRIDE_T=value.stride(1),
        VALUE_STRIDE_H=value.stride(2),
        VALUE_STRIDE_D=value.stride(3),
        T=length,
        STEPS=steps,
        H=heads,
        K=rank,
        C=chunk_size,
        BK=max(16, triton.next_power_of_2(rank)),
        num_warps=4,
        num_stages=1,
    )
    return gain, prediction


def cg_transpose_hkv(
    gain_cotangent,
    gain,
    key,
    value,
    boundary_kv,
    boundary,
    local_decay,
    output_cotangent,
    grad_decay_previous,
    *,
    chunk_size,
    steps,
):
    batch, length, heads, rank = gain.shape
    result = torch.empty_like(gain)
    grad_decay = torch.empty_like(local_decay)
    _cg_transpose_hkv_kernel[(triton.cdiv(length, chunk_size), batch * heads)](
        gain,
        key,
        value,
        boundary_kv,
        boundary,
        local_decay,
        output_cotangent,
        grad_decay_previous,
        gain_cotangent,
        result,
        grad_decay,
        VALUE_STRIDE_B=value.stride(0),
        VALUE_STRIDE_T=value.stride(1),
        VALUE_STRIDE_H=value.stride(2),
        VALUE_STRIDE_D=value.stride(3),
        T=length,
        STEPS=steps,
        H=heads,
        K=rank,
        C=chunk_size,
        BK=max(16, triton.next_power_of_2(rank)),
        num_warps=4,
        num_stages=1,
    )
    return result, grad_decay


def hkv_reverse_dkv(gain, key, value, boundary_kv, boundary_cotangent, local_decay, output_cotangent, *, chunk_size):
    batch, length, heads, rank = key.shape
    block = max(16, triton.next_power_of_2(rank))
    grid = (triton.cdiv(length, chunk_size), batch * heads)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty(value.shape, dtype=value.dtype, device=value.device)
    grad_decay_first = torch.empty_like(local_decay)
    _hkv_reverse_dkv_kernel[grid](
        gain,
        key,
        value,
        boundary_kv,
        local_decay,
        output_cotangent,
        boundary_cotangent,
        grad_key,
        grad_decay_first,
        grad_value,
        VALUE_STRIDE_B=value.stride(0),
        VALUE_STRIDE_T=value.stride(1),
        VALUE_STRIDE_H=value.stride(2),
        VALUE_STRIDE_D=value.stride(3),
        T=length,
        H=heads,
        K=rank,
        C=chunk_size,
        BK=block,
        BV=block,
    )
    return grad_key, grad_value, grad_decay_first


def hkk_reverse(key, boundary, boundary_cotangent, local_decay, gain, gain_cotangent, key_seed, *, chunk_size):
    batch, length, heads, rank = key.shape
    grad_key = torch.empty_like(key)
    grad_decay = torch.empty_like(local_decay)
    _hkk_reverse_kernel[(triton.cdiv(length, chunk_size), batch * heads)](
        key,
        boundary,
        boundary_cotangent,
        local_decay,
        gain,
        gain_cotangent,
        key_seed,
        grad_key,
        grad_decay,
        T=length,
        H=heads,
        K=rank,
        C=chunk_size,
        BK=max(16, triton.next_power_of_2(rank)),
        num_warps=4,
        num_stages=2,
    )
    return grad_key, grad_decay


__all__ = [
    "paired_state_forward",
    "cg_forward",
    "cg_transpose_hkv",
    "hkv_reverse_dkv",
    "hkk_reverse",
]
