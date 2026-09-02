# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 SolveDelta contributors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import IS_AMD, autotune_cache_kwargs, check_shared_mem

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [2, 4, 8, 16, 32]

"""FLA DPLR output reverse specialized for a shared left interaction."""


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=['BV', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_bwd_kernel_dAu(
    v,
    do,
    v_new,
    A_qb,
    dA_qd,
    dv_new,
    cu_seqlens,
    chunk_indices,
    scale: tl.constexpr,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    b_dA_qd = tl.zeros([BT, BT], dtype=tl.float32)

    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_A_qb = A_qb + (bos * H + i_h) * BT + o_t[:, None] * (H*BT) + o_A[None, :]

    b_A_qb = tl.load(p_A_qb, mask=m_A, other=0.0)
    # causal mask
    b_A_qb = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_A_qb, 0.).to(b_A_qb.dtype)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        m_vt = (o_v[:, None] < V) & m_t[None, :]
        p_do = do + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        p_v = v + (bos*H + i_h) * V + o_v[:, None] + o_t[None, :] * (H*V)
        p_v_new = v_new + (bos*H + i_h) * V + o_v[:, None] + o_t[None, :] * (H*V)
        p_dv_new = dv_new + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_vt, other=0.0)
        b_do = tl.load(p_do, mask=m_v, other=0.0)
        b_v_new = tl.load(p_v_new, mask=m_vt, other=0.0)
        b_dA_qd += tl.dot(b_do, b_v + b_v_new)
        b_dv_new = tl.dot(tl.trans(b_A_qb), b_do)
        # for recurrent
        tl.store(p_dv_new, b_dv_new.to(p_dv_new.dtype.element_ty), mask=m_v)

    p_dA_qd = dA_qd + (bos * H + i_h) * BT + o_t[:, None] * (H*BT) + o_A[None, :]
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_dA_qd = tl.where(m_s, b_dA_qd * scale, 0.)
    tl.store(p_dA_qd, b_dA_qd.to(p_dA_qd.dtype.element_ty), mask=m_A)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit
def chunk_dplr_bwd_o_kernel(
    v,
    v_new,
    h,
    do,
    dh,
    dk,
    w,
    dq,
    dv,
    dw,
    gk,
    dgk_last,
    k,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    v += (bos * H + i_h) * V
    v_new += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h += (i_tg * H + i_h) * K * V
    dh += (i_tg * H + i_h) * K * V
    dk += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    dw += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    dq += (bos * H + i_h) * K
    w += (bos * H + i_h) * K

    dgk_last += (i_tg * H + i_h) * K
    gk += (bos * H + i_h) * K

    stride_qk = H*K
    stride_vo = H*V

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    m_t = o_t < T
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk_last = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        m_hv = (o_v[:, None] < V) & (o_k[None, :] < K)
        p_v = v + o_t[:, None] * stride_vo + o_v[None, :]
        p_v_new = v_new + o_t[:, None] * stride_vo + o_v[None, :]
        p_do = do + o_t[:, None] * stride_vo + o_v[None, :]
        p_h = h + o_v[:, None] + o_k[None, :] * V
        p_dh = dh + o_v[:, None] + o_k[None, :] * V
        # [BT, BV]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_v_new = tl.load(p_v_new, mask=m_v, other=0.0)
        b_do = tl.load(p_do, mask=m_v, other=0.0)
        # [BV, BK]
        b_h = tl.load(p_h, mask=m_hv, other=0.0)
        b_dh = tl.load(p_dh, mask=m_hv, other=0.0)
        b_dgk_last += tl.sum((b_h * b_dh).to(tl.float32), axis=0)

        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk += tl.dot(b_v + b_v_new, b_dh.to(b_v.dtype))
        p_dv = dv + o_t[:, None] * stride_vo + o_v[None, :]
        b_dv = tl.load(p_dv, mask=m_v, other=0.0)
        b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

    m_k = (i_k*BK+tl.arange(0, BK)) < K
    last_idx = min(i_t * BT + BT, T) - 1
    b_gk_last = tl.load(gk + last_idx * stride_qk + i_k*BK + tl.arange(0, BK), mask=m_k, other=float('-inf'))
    b_dgk_last *= exp2(b_gk_last)
    m_kk = m_t[:, None] & m_k[None, :]
    p_k = k + o_t[:, None] * stride_qk + o_k[None, :]
    b_k = tl.load(p_k, mask=m_kk, other=0.0)
    b_dgk_last += tl.sum(b_k * b_dk, axis=0)
    tl.store(dgk_last + tl.arange(0, BK) + i_k * BK, b_dgk_last, mask=m_k)

    p_dw = dw + o_t[:, None] * stride_qk + o_k[None, :]
    p_dk = dk + o_t[:, None] * stride_qk + o_k[None, :]
    p_dq = dq + o_t[:, None] * stride_qk + o_k[None, :]
    tl.store(p_dw, b_dw.to(p_dw.dtype.element_ty), mask=m_kk)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_kk)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_kk)


def chunk_dplr_bwd_o(
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    gk: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    scale: float = 1.0,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, T, H, K, V = *w.shape, v.shape[-1]

    BT = chunk_size
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = min(max(triton.next_power_of_2(K), 16), 64) if check_shared_mem() else min(triton.next_power_of_2(K), 32)
    BV = min(max(triton.next_power_of_2(V), 16), 64) if check_shared_mem() else min(triton.next_power_of_2(K), 32)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(k)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w)
    grid = (NK, NT, B * H)

    dgk_last = torch.empty(B, NT, H, K, dtype=torch.float, device=w.device)

    chunk_dplr_bwd_o_kernel[grid](
        k=k,
        v=v,
        v_new=v_new,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dgk_last=dgk_last,
        w=w,
        dv=dv,
        dw=dw,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dq, dk, dw, dgk_last


def chunk_dplr_bwd_dAu(
    v: torch.Tensor,
    v_new: torch.Tensor,
    do: torch.Tensor,
    A_qb: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, V = v.shape
    BT = chunk_size
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if check_shared_mem('ampere'):  # A100
        BV = min(triton.next_power_of_2(V), 128)
    elif check_shared_mem('ada'):  # 4090
        BV = min(max(triton.next_power_of_2(V), 16), 64)
    else:
        BV = min(triton.next_power_of_2(V), 32)

    grid = (NT, B * H)
    dA_qd = torch.empty(B, T, H, BT, dtype=torch.float, device=v.device)
    dv_new = torch.empty_like(v_new)
    chunk_dplr_bwd_kernel_dAu[grid](
        v=v,
        do=do,
        v_new=v_new,
        A_qb=A_qb,
        dA_qd=dA_qd,
        dv_new=dv_new,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
    )
    return dv_new, dA_qd
