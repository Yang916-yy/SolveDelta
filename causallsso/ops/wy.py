from __future__ import annotations

import torch
import triton
import triton.language as tl


# The surrounding WY staging follows FLA v0.5.2's MIT-licensed DPLR kernels:
# https://github.com/fla-org/flash-linear-attention/tree/v0.5.2/fla/ops/generalized_delta_rule/dplr
# FLA's public operator requires a separately allocated activation tensor.  The
# two kernels below specialize only the intra-chunk boundary to SolveDelta's
# a=-e*exp(g), so the activation bits are generated at use and its pullback is
# folded directly into de/dg.  The mature FLA WY/state/output kernels remain the
# owners of the exterior computation.


@triton.jit(do_not_specialize=["T"])
def _direct_e_fwd_intra_kernel(
    q,
    k,
    e,
    b,
    g,
    gi,
    ge,
    qg,
    kg,
    ag,
    bg,
    A_qk,
    A_qb,
    A_ak,
    A_ab,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    scale: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_b = tl.program_id(1)
    i_h = tl.program_id(2)
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_tk = m_t[:, None] & (o_k[None, :] < K)
    vector_base = (bos * H + i_h) * K
    vector_stride = H * K

    p_q = q + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_k = k + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_e = e + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_b = b + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_g = g + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_gi = gi + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_ge = ge + vector_base + o_t[:, None] * vector_stride + o_k[None, :]

    b_q = tl.load(p_q, mask=m_tk, other=0.0)
    b_k = tl.load(p_k, mask=m_tk, other=0.0)
    b_e = tl.load(p_e, mask=m_tk, other=0.0).to(tl.float32)
    b_b = tl.load(p_b, mask=m_tk, other=0.0)
    b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)
    b_gi = tl.load(p_gi, mask=m_tk, other=0.0).to(tl.float32)
    b_ge = tl.load(p_ge, mask=m_tk, other=0.0).to(tl.float32)

    # This cast is the one declared activation quantization boundary.  No
    # tensor containing a survives this program.
    b_a = (-b_e * tl.exp(b_g)).to(p_e.dtype.element_ty, fp_downcast_rounding="rtne")

    valid_len = min(T - i_t * BT, BT)
    mid_idx = valid_len // 2
    last_idx = min((i_t + 1) * BT, T) - 1
    p_offset = gi + vector_base + (i_t * BT + mid_idx) * vector_stride + o_k
    p_last = gi + vector_base + last_idx * vector_stride + o_k
    m_k = o_k < K
    b_offset = tl.load(p_offset, mask=m_k, other=0.0).to(tl.float32)
    b_last = tl.load(p_last, mask=m_k, other=0.0).to(tl.float32)

    exp_gi = tl.math.exp2(b_gi - b_offset[None, :])
    inv_exp_gi = tl.math.exp2(-b_gi + b_offset[None, :])
    exp_ge = tl.math.exp2(b_ge - b_offset[None, :])
    q_ops = (b_q * scale * exp_gi).to(tl.float32)
    k_ops = (b_k * inv_exp_gi).to(tl.float32)
    a_ops = (b_a * exp_ge).to(tl.float32)
    b_ops = (b_b * inv_exp_gi).to(tl.float32)

    exp_offset = tl.math.exp2(b_offset)
    exp_last_centered = tl.math.exp2(b_last - b_offset)
    p_qg = qg + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_kg = kg + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_ag = ag + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_bg = bg + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    tl.store(
        p_qg,
        (q_ops * exp_offset[None, :]).to(p_qg.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=m_tk,
    )
    tl.store(
        p_ag,
        (a_ops * exp_offset[None, :]).to(p_ag.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=m_tk,
    )
    tl.store(
        p_kg,
        (k_ops * exp_last_centered[None, :]).to(p_kg.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=m_tk,
    )
    tl.store(
        p_bg,
        (b_ops * exp_last_centered[None, :]).to(p_bg.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=m_tk,
    )

    o_A = tl.arange(0, BT)
    inclusive = o_A[:, None] >= o_A[None, :]
    matrix_base = (bos * H + i_h) * BT
    matrix_stride = H * BT
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_A_qk = A_qk + matrix_base + o_t[:, None] * matrix_stride + o_A[None, :]
    p_A_qb = A_qb + matrix_base + o_t[:, None] * matrix_stride + o_A[None, :]

    # Keep each contraction's BF16 operands explicit.  The a-side pair runs in
    # a second kernel from the already-required ag buffer; four simultaneous
    # C32x128 operands otherwise collapse occupancy on consumer Blackwell.
    q_contract = q_ops.to(p_q.dtype.element_ty, fp_downcast_rounding="rtne")
    k_contract = k_ops.to(p_k.dtype.element_ty, fp_downcast_rounding="rtne")
    b_contract = b_ops.to(p_b.dtype.element_ty, fp_downcast_rounding="rtne")
    b_A_qk = tl.where(
        inclusive,
        tl.dot(q_contract, tl.trans(k_contract)),
        0.0,
    )
    b_A_qb = tl.where(
        inclusive,
        tl.dot(q_contract, tl.trans(b_contract)),
        0.0,
    )
    tl.store(p_A_qk, b_A_qk.to(p_A_qk.dtype.element_ty), mask=m_A)
    tl.store(p_A_qb, b_A_qb.to(p_A_qb.dtype.element_ty), mask=m_A)



@triton.jit(do_not_specialize=["T"])
def _direct_e_fwd_activation_matrices_kernel(
    ag,
    k,
    b,
    gi,
    A_ak,
    A_ab,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_b = tl.program_id(1)
    i_h = tl.program_id(2)
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_tk = m_t[:, None] & (o_k[None, :] < K)
    vector_base = (bos * H + i_h) * K
    vector_stride = H * K
    p_ag = ag + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_k = k + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_b = b + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_gi = gi + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    b_ag = tl.load(p_ag, mask=m_tk, other=0.0)
    b_k = tl.load(p_k, mask=m_tk, other=0.0)
    b_b = tl.load(p_b, mask=m_tk, other=0.0)
    b_gi = tl.load(p_gi, mask=m_tk, other=0.0).to(tl.float32)

    valid_len = min(T - i_t * BT, BT)
    mid_idx = valid_len // 2
    p_offset = gi + vector_base + (i_t * BT + mid_idx) * vector_stride + o_k
    b_offset = tl.load(p_offset, mask=o_k < K, other=0.0).to(tl.float32)
    a_contract = (b_ag * tl.math.exp2(-b_offset[None, :])).to(
        p_ag.dtype.element_ty,
        fp_downcast_rounding="rtne",
    )
    k_contract = (b_k * tl.math.exp2(-b_gi + b_offset[None, :])).to(
        p_k.dtype.element_ty,
        fp_downcast_rounding="rtne",
    )
    b_contract = (b_b * tl.math.exp2(-b_gi + b_offset[None, :])).to(
        p_b.dtype.element_ty,
        fp_downcast_rounding="rtne",
    )

    o_A = tl.arange(0, BT)
    strict = o_A[:, None] > o_A[None, :]
    b_A_ak = tl.where(
        strict,
        tl.dot(a_contract, tl.trans(k_contract)),
        0.0,
    )
    b_A_ab = tl.where(
        strict,
        tl.dot(a_contract, tl.trans(b_contract)),
        0.0,
    )
    matrix_base = (bos * H + i_h) * BT
    matrix_stride = H * BT
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_A_ak = A_ak + matrix_base + o_t[:, None] * matrix_stride + o_A[None, :]
    p_A_ab = A_ab + matrix_base + o_t[:, None] * matrix_stride + o_A[None, :]
    tl.store(p_A_ak, b_A_ak.to(p_A_ak.dtype.element_ty), mask=m_A)
    tl.store(p_A_ab, b_A_ab.to(p_A_ab.dtype.element_ty), mask=m_A)


@triton.jit(do_not_specialize=["T"])
def _direct_e_bwd_intra_kernel(
    q,
    k,
    e,
    b,
    g,
    gi,
    ge,
    dA_qk,
    dA_qb,
    dA_ak,
    dA_ab,
    dqg,
    dkg,
    dag,
    dbg,
    dq,
    dk,
    de,
    db,
    dgk,
    dgk_offset,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    scale: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]
    vector_base = (bos * H + i_h) * K
    vector_stride = H * K
    p_q = q + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_k = k + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_e = e + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_b = b + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_g = g + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_gi = gi + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_ge = ge + vector_base + o_t[:, None] * vector_stride + o_k[None, :]

    b_q = tl.load(p_q, mask=m_tk, other=0.0)
    b_k = tl.load(p_k, mask=m_tk, other=0.0)
    b_e = tl.load(p_e, mask=m_tk, other=0.0).to(tl.float32)
    b_b = tl.load(p_b, mask=m_tk, other=0.0)
    b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)
    b_gi = tl.load(p_gi, mask=m_tk, other=0.0).to(tl.float32)
    b_ge = tl.load(p_ge, mask=m_tk, other=0.0).to(tl.float32)
    activation_scale = tl.exp(b_g)
    a_unquantized = -b_e * activation_scale
    b_a = a_unquantized.to(p_e.dtype.element_ty, fp_downcast_rounding="rtne")

    valid_len = min(T - i_t * BT, BT)
    mid_idx = valid_len // 2
    last_idx = min((i_t + 1) * BT, T) - 1
    p_offset = gi + vector_base + (i_t * BT + mid_idx) * vector_stride + o_k
    p_last = gi + vector_base + last_idx * vector_stride + o_k
    b_offset = tl.load(p_offset, mask=m_k, other=0.0).to(tl.float32)
    b_last = tl.load(p_last, mask=m_k, other=0.0).to(tl.float32)
    exp_gi = tl.math.exp2(b_gi - b_offset[None, :])
    inv_exp_gi = tl.math.exp2(-b_gi + b_offset[None, :])
    exp_ge = tl.math.exp2(b_ge - b_offset[None, :])
    q_ops = (b_q * exp_gi).to(tl.float32)
    k_ops = (b_k * inv_exp_gi).to(tl.float32)
    a_ops = (b_a * exp_ge).to(tl.float32)
    b_ops = (b_b * inv_exp_gi).to(tl.float32)

    o_A = tl.arange(0, BT)
    matrix_base = (bos * H + i_h) * BT
    matrix_stride = H * BT
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_dA_qk = dA_qk + matrix_base + o_t[:, None] * matrix_stride + o_A[None, :]
    p_dA_qb = dA_qb + matrix_base + o_t[:, None] * matrix_stride + o_A[None, :]
    p_dA_ak = dA_ak + matrix_base + o_t[:, None] * matrix_stride + o_A[None, :]
    p_dA_ab = dA_ab + matrix_base + o_t[:, None] * matrix_stride + o_A[None, :]
    b_dA_qk = tl.load(p_dA_qk, mask=m_A, other=0.0)
    b_dA_qb = tl.load(p_dA_qb, mask=m_A, other=0.0)
    b_dA_ak = tl.load(p_dA_ak, mask=m_A, other=0.0)
    b_dA_ab = tl.load(p_dA_ab, mask=m_A, other=0.0)
    inclusive = o_A[:, None] >= o_A[None, :]
    strict = o_A[:, None] > o_A[None, :]
    b_dA_qk = tl.where(inclusive, b_dA_qk, 0.0)
    b_dA_qb = tl.where(inclusive, b_dA_qb, 0.0)
    b_dA_ak = tl.where(strict, b_dA_ak, 0.0)
    b_dA_ab = tl.where(strict, b_dA_ab, 0.0)

    dA_qk_contract = b_dA_qk.to(p_q.dtype.element_ty, fp_downcast_rounding="rtne")
    dA_qb_contract = b_dA_qb.to(p_q.dtype.element_ty, fp_downcast_rounding="rtne")
    dA_ak_contract = b_dA_ak.to(p_q.dtype.element_ty, fp_downcast_rounding="rtne")
    dA_ab_contract = b_dA_ab.to(p_q.dtype.element_ty, fp_downcast_rounding="rtne")
    q_contract = q_ops.to(p_q.dtype.element_ty, fp_downcast_rounding="rtne")
    k_contract = k_ops.to(p_k.dtype.element_ty, fp_downcast_rounding="rtne")
    a_contract = a_ops.to(p_e.dtype.element_ty, fp_downcast_rounding="rtne")
    b_contract = b_ops.to(p_b.dtype.element_ty, fp_downcast_rounding="rtne")
    b_dq = tl.dot(dA_qk_contract, k_contract) + tl.dot(
        dA_qb_contract,
        b_contract,
    )
    b_da = tl.dot(dA_ak_contract, k_contract) + tl.dot(
        dA_ab_contract,
        b_contract,
    )
    b_dk = tl.dot(tl.trans(dA_qk_contract), q_contract) + tl.dot(
        tl.trans(dA_ak_contract),
        a_contract,
    )
    b_db = tl.dot(tl.trans(dA_qb_contract), q_contract) + tl.dot(
        tl.trans(dA_ab_contract),
        a_contract,
    )
    b_dq *= exp_gi
    b_da *= exp_ge
    b_dk *= inv_exp_gi
    b_db *= inv_exp_gi

    p_dqg = dqg + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_dkg = dkg + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_dag = dag + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_dbg = dbg + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    b_dq += tl.load(p_dqg, mask=m_tk, other=0.0) * tl.math.exp2(b_gi) * scale
    b_da += tl.load(p_dag, mask=m_tk, other=0.0) * tl.math.exp2(b_ge)
    inter_scale = tl.math.exp2(b_last[None, :] - b_gi)
    b_dk += tl.load(p_dkg, mask=m_tk, other=0.0).to(tl.float32) * inter_scale
    b_db += tl.load(p_dbg, mask=m_tk, other=0.0).to(tl.float32) * inter_scale

    p_dq = dq + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_dk = dk + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_de = de + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_db = db + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_dgk = dgk + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    p_dgk_offset = dgk_offset + vector_base + o_t[:, None] * vector_stride + o_k[None, :]
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_tk)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_tk)
    tl.store(p_de, (-b_da * activation_scale).to(p_de.dtype.element_ty), mask=m_tk)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_tk)

    gate_partial = b_dq * b_q + b_da * b_a - b_dk * b_k - b_db * b_b
    direct_gate = b_da * a_unquantized
    # FLA subtracts dgk_offset after its reverse cumulative sum.  Subtracting
    # direct_gate here folds the a(e,g) chain contribution into that same pass.
    gate_offset = b_da * b_a - direct_gate
    tl.store(p_dgk, gate_partial.to(p_dgk.dtype.element_ty), mask=m_tk)
    tl.store(p_dgk_offset, gate_offset.to(p_dgk_offset.dtype.element_ty), mask=m_tk)


def _direct_e_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    e: torch.Tensor,
    b: torch.Tensor,
    g: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    *,
    chunk_size: int,
    scale: float,
) -> tuple[torch.Tensor, ...]:
    batch, length, heads, rank = q.shape
    block_rank = max(triton.next_power_of_2(rank), 16)
    if block_rank > 256:
        raise ValueError("the direct-e WY specialization requires r <= 256")
    chunks = triton.cdiv(length, chunk_size)
    A_qk = q.new_empty(batch, length, heads, chunk_size)
    A_qb = q.new_empty(batch, length, heads, chunk_size)
    A_ak = q.new_empty(batch, length, heads, chunk_size, dtype=torch.float32)
    A_ab = q.new_empty(batch, length, heads, chunk_size, dtype=torch.float32)
    qg = torch.empty_like(q)
    kg = torch.empty_like(k)
    ag = torch.empty_like(e)
    bg = torch.empty_like(b)
    _direct_e_fwd_intra_kernel[(chunks, batch, heads)](
        q=q,
        k=k,
        e=e,
        b=b,
        g=g,
        gi=gi,
        ge=ge,
        qg=qg,
        kg=kg,
        ag=ag,
        bg=bg,
        A_qk=A_qk,
        A_qb=A_qb,
        A_ak=A_ak,
        A_ab=A_ab,
        T=length,
        H=heads,
        K=rank,
        BT=chunk_size,
        BK=block_rank,
        scale=scale,
        num_warps=4,
        num_stages=2,
    )
    _direct_e_fwd_activation_matrices_kernel[(chunks, batch, heads)](
        ag=ag,
        k=k,
        b=b,
        gi=gi,
        A_ak=A_ak,
        A_ab=A_ab,
        T=length,
        H=heads,
        K=rank,
        BT=chunk_size,
        BK=block_rank,
        num_warps=4,
        num_stages=2,
    )
    return A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg


def _direct_e_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    e: torch.Tensor,
    b: torch.Tensor,
    g: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    dA_qk: torch.Tensor,
    dA_qb: torch.Tensor,
    dA_ak: torch.Tensor,
    dA_ab: torch.Tensor,
    dqg: torch.Tensor,
    dkg: torch.Tensor,
    dag: torch.Tensor,
    dbg: torch.Tensor,
    dgk_last: torch.Tensor,
    *,
    chunk_size: int,
    scale: float,
) -> tuple[torch.Tensor, ...]:
    from fla.ops.generalized_delta_rule.dplr.chunk_A_bwd import (
        chunk_dplr_bwd_dgk_kernel,
    )

    batch, length, heads, rank = q.shape
    chunks = triton.cdiv(length, chunk_size)
    block_rank = min(32, triton.next_power_of_2(rank))
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    de = torch.empty_like(e)
    db = torch.empty_like(b)
    dgk = torch.empty_like(gi, dtype=torch.float32)
    dgk_offset = torch.empty_like(gi, dtype=torch.float32)
    _direct_e_bwd_intra_kernel[(triton.cdiv(rank, block_rank), chunks, batch * heads)](
        q=q,
        k=k,
        e=e,
        b=b,
        g=g,
        gi=gi,
        ge=ge,
        dA_qk=dA_qk,
        dA_qb=dA_qb,
        dA_ak=dA_ak,
        dA_ab=dA_ab,
        dqg=dqg,
        dkg=dkg,
        dag=dag,
        dbg=dbg,
        dq=dq,
        dk=dk,
        de=de,
        db=db,
        dgk=dgk,
        dgk_offset=dgk_offset,
        T=length,
        H=heads,
        K=rank,
        BT=chunk_size,
        BK=block_rank,
        scale=scale,
        num_warps=4,
        num_stages=2,
    )
    dg = torch.empty_like(g, dtype=torch.float32)
    def dg_grid(meta: dict[str, int]) -> tuple[int, int, int]:
        return chunks, triton.cdiv(rank, meta["BK"]), batch * heads

    chunk_dplr_bwd_dgk_kernel[dg_grid](
        dgk=dgk,
        dgk_offset=dgk_offset,
        dgk_last=dgk_last,
        dgk_output=dg,
        cu_seqlens=None,
        chunk_indices=None,
        T=length,
        H=heads,
        K=rank,
        BT=chunk_size,
    )
    return dq, dk, de, db, dg


def _direct_e_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    e: torch.Tensor,
    b: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    output_final_state: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from fla.ops.generalized_delta_rule.dplr.chunk_h_fwd import chunk_dplr_fwd_h
    from fla.ops.generalized_delta_rule.dplr.chunk_o_fwd import chunk_dplr_fwd_o
    from fla.ops.generalized_delta_rule.dplr.wy_fast_fwd import prepare_wy_repr_fwd
    from fla.ops.rwkv6.chunk import chunk_rwkv6_fwd_cumsum
    from fla.ops.utils.constant import RCP_LN2

    gi, ge = chunk_rwkv6_fwd_cumsum(
        g,
        chunk_size,
        scale=RCP_LN2,
    )
    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = _direct_e_fwd_intra(
        q,
        k,
        e,
        b,
        g,
        gi,
        ge,
        chunk_size=chunk_size,
        scale=1.0,
    )
    w, u, _ = prepare_wy_repr_fwd(
        ag=ag,
        A_ab=A_ab,
        A_ak=A_ak,
        v=v,
        cu_seqlens=None,
        chunk_size=chunk_size,
        chunk_indices=None,
    )
    h, v_new, final_state = chunk_dplr_fwd_h(
        kg=kg,
        bg=bg,
        v=v,
        w=w,
        u=u,
        gk=gi,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=None,
        chunk_size=chunk_size,
        chunk_indices=None,
    )
    output = chunk_dplr_fwd_o(
        qg=qg,
        v=v,
        v_new=v_new,
        A_qk=A_qk,
        A_qb=A_qb,
        h=h,
        cu_seqlens=None,
        chunk_size=chunk_size,
        chunk_indices=None,
    )
    return output, final_state


class _DirectEWYFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        e: torch.Tensor,
        b: torch.Tensor,
        g: torch.Tensor,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        output, final_state = _direct_e_fwd(
            q,
            k,
            v,
            e,
            b,
            g,
            initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
        )
        saved_state = (
            initial_state
            if initial_state is not None
            else q.new_empty(0, dtype=torch.float32)
        )
        ctx.save_for_backward(q, k, v, e, b, g, saved_state)
        ctx.has_initial_state = initial_state is not None
        ctx.chunk_size = chunk_size
        return output.to(q.dtype), final_state

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        do: torch.Tensor,
        dht: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        from fla.ops.generalized_delta_rule.dplr.chunk_h_bwd import (
            chunk_dplr_bwd_dhu,
        )
        from fla.ops.generalized_delta_rule.dplr.chunk_h_fwd import chunk_dplr_fwd_h
        from fla.ops.generalized_delta_rule.dplr.chunk_o_bwd import (
            chunk_dplr_bwd_dAu,
            chunk_dplr_bwd_dv,
            chunk_dplr_bwd_o,
        )
        from fla.ops.generalized_delta_rule.dplr.wy_fast_bwd import (
            chunk_dplr_bwd_wy,
        )
        from fla.ops.generalized_delta_rule.dplr.wy_fast_fwd import (
            prepare_wy_repr_fwd,
        )
        from fla.ops.rwkv6.chunk import chunk_rwkv6_fwd_cumsum
        from fla.ops.utils.constant import RCP_LN2

        q, k, v, e, b, g, saved_state = ctx.saved_tensors
        initial_state = saved_state if ctx.has_initial_state else None
        chunk_size = ctx.chunk_size

        # FLA's memory-efficient training path recomputes the WY factors.  The
        # direct-e specialization follows it, but regenerates activation bits
        # inside the intra kernel instead of recovering a saved tensor.
        gi, ge = chunk_rwkv6_fwd_cumsum(
            g,
            chunk_size,
            scale=RCP_LN2,
        )
        A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = _direct_e_fwd_intra(
            q,
            k,
            e,
            b,
            g,
            gi,
            ge,
            chunk_size=chunk_size,
            scale=1.0,
        )
        w, u, A_ab_inv = prepare_wy_repr_fwd(
            ag=ag,
            A_ab=A_ab,
            A_ak=A_ak,
            v=v,
            cu_seqlens=None,
            chunk_size=chunk_size,
            chunk_indices=None,
        )
        h, v_new, _ = chunk_dplr_fwd_h(
            kg=kg,
            bg=bg,
            v=v,
            w=w,
            u=u,
            gk=gi,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=None,
            chunk_size=chunk_size,
            chunk_indices=None,
        )

        dv_new_intra, dA_qk, dA_qb = chunk_dplr_bwd_dAu(
            v=v,
            v_new=v_new,
            do=do,
            A_qb=A_qb,
            scale=1.0,
            cu_seqlens=None,
            chunk_size=chunk_size,
            chunk_indices=None,
        )
        dh, dh0, dv_new = chunk_dplr_bwd_dhu(
            qg=qg,
            bg=bg,
            w=w,
            gk=gi,
            h0=initial_state,
            dht=dht,
            do=do,
            dv=dv_new_intra,
            cu_seqlens=None,
            chunk_size=chunk_size,
            chunk_indices=None,
        )
        dv = chunk_dplr_bwd_dv(
            A_qk=A_qk,
            kg=kg,
            do=do,
            dh=dh,
            cu_seqlens=None,
            chunk_size=chunk_size,
            chunk_indices=None,
        )
        dqg, dkg, dw, dbg, dgk_last = chunk_dplr_bwd_o(
            k=kg,
            b=bg,
            v=v,
            v_new=v_new,
            gk=gi,
            do=do,
            h=h,
            dh=dh,
            dv=dv_new,
            w=w,
            cu_seqlens=None,
            chunk_size=chunk_size,
            scale=1.0,
            chunk_indices=None,
        )
        dA_ab, dA_ak, dv, dag = chunk_dplr_bwd_wy(
            A_ab_inv=A_ab_inv,
            A_ak=A_ak,
            v=v,
            ag=ag,
            dw=dw,
            du=dv_new,
            dv0=dv,
            cu_seqlens=None,
            chunk_size=chunk_size,
            chunk_indices=None,
        )
        dq, dk, de, db, dg = _direct_e_bwd_intra(
            q,
            k,
            e,
            b,
            g,
            gi,
            ge,
            dA_qk,
            dA_qb,
            dA_ak,
            dA_ab,
            dqg,
            dkg,
            dag,
            dbg,
            dgk_last,
            chunk_size=chunk_size,
            scale=1.0,
        )
        return dq, dk, dv, de, db, dg, dh0, None, None


@torch.compiler.disable
def _direct_e_dplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    e: torch.Tensor,
    b: torch.Tensor,
    g: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return _DirectEWYFunction.apply(
        q,
        k,
        v,
        e,
        b,
        g,
        initial_state,
        output_final_state,
        chunk_size,
    )


def _validate_inputs(
    chi: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    z: torch.Tensor,
    log_decay: torch.Tensor,
    initial_state: torch.Tensor | None,
    chunk_size: int,
) -> tuple[int, int, int, int, int, int]:
    if chi.ndim != 4:
        raise ValueError("chi must have shape [B, T, H, r]")
    if d.ndim != 5:
        raise ValueError("d must have shape [B, T, H, K, r]")
    batch, length, heads, edits, rank = d.shape
    if min(batch, length, heads, edits, rank) < 1:
        raise ValueError("B, T, H, K, and r must be positive")
    if chi.shape != (batch, length, heads, rank):
        raise ValueError("chi shape does not match d")
    if e.shape != d.shape:
        raise ValueError("e must match d shape")
    if z.ndim != 5 or z.shape[:4] != (batch, length, heads, edits):
        raise ValueError("z must have shape [B, T, H, K, d_v]")
    value_dim = z.shape[-1]
    if value_dim < 1:
        raise ValueError("d_v must be positive")
    if log_decay.shape != (batch, length, heads, rank):
        raise ValueError("log_decay must have shape [B, T, H, r]")

    tensors = (d, e, z)
    if any(tensor.device != chi.device for tensor in tensors):
        raise ValueError("chi, d, e, and z must share one device")
    if log_decay.device != chi.device:
        raise ValueError("log_decay must share the vector input device")
    if any(tensor.dtype != chi.dtype for tensor in tensors):
        raise TypeError("chi, d, e, and z must share one dtype")
    if chi.device.type != "cuda":
        raise ValueError("the WY exterior requires CUDA tensors")
    if chi.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("the WY exterior requires FP16 or BF16 inputs")
    if log_decay.dtype != torch.float32:
        raise TypeError("log_decay must be FP32")

    if initial_state is not None:
        expected = (batch, heads, rank, value_dim)
        if initial_state.shape != expected:
            raise ValueError(f"initial_state must have shape {expected}")
        if initial_state.device != chi.device:
            raise ValueError("initial_state must share the input CUDA device")
        if initial_state.dtype != torch.float32:
            raise TypeError("initial_state must be FP32")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an int")
    if chunk_size != 32:
        raise ValueError("the direct-e WY specialization requires chunk_size=32")
    return batch, length, heads, edits, rank, value_dim


def _pack_edits(
    chi: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    z: torch.Tensor,
    log_decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand tokens in token-major, edit-minor order for FLA's DPLR ABI."""
    batch, length, heads, edits, rank = d.shape
    value_dim = z.shape[-1]
    slot = torch.arange(edits, device=d.device).view(1, 1, edits, 1, 1)
    expanded_decay = log_decay.unsqueeze(2) * (slot == 0)
    expanded_query = chi.unsqueeze(2) * (slot == edits - 1)

    packed_q = expanded_query.reshape(batch, length * edits, heads, rank)
    packed_k = d.transpose(2, 3).reshape(batch, length * edits, heads, rank)
    packed_v = z.transpose(2, 3).reshape(batch, length * edits, heads, value_dim)
    packed_e = e.transpose(2, 3).reshape(batch, length * edits, heads, rank)
    packed_g = expanded_decay.reshape(batch, length * edits, heads, rank)
    return packed_q, packed_k, packed_v, packed_e, packed_g


def wy_associative(
    chi: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    z: torch.Tensor,
    log_decay: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate SolveDelta's associative state with FLA's chunk/WY kernels.

    The state is stored as ``[B, H, r, d_v]``. At token ``t`` it first
    receives the row-wise decay ``exp(log_decay[t])`` and then the ``K``
    ordered edits

    ``S <- S + d (z - S^T e)^T``.

    FLA's generalized Delta transition is algebraically identical under
    ``k=b=d, v=z, a=-e*exp(g)``. Sharing ``b`` with ``k`` avoids materializing
    the otherwise redundant signed copy of every edit direction. This wrapper
    specializes the DPLR intra-chunk kernels to consume ``e`` and FP32 ``g``
    directly: the BF16 activation bits are generated in registers and the
    backward folds their pullback into ``de`` and ``dg``. No full ``a`` tensor
    is allocated, saved, or replayed.
    For ``K > 1``, token time is expanded in
    token-major, edit-minor order: only edit zero receives ``g`` and only edit
    ``K-1`` receives ``chi``. Thus decay happens once and each output is read
    after the final edit. ``log_decay`` is required to be nonpositive by the
    operator contract; checking that value constraint belongs to the frontend
    so this hot path does not synchronize the CUDA stream.

    Vector inputs are FP16 or BF16; BF16 is SolveDelta's advertised path.
    Log-decay and recurrent states are FP32, matching the mixed-precision
    contract and FLA's returned-state boundary. ``a=-e*exp(g)`` is evaluated
    from the FP32 decay and rounded once to the vector dtype at its first
    contraction use. C32 is the production setting.
    """
    batch, length, heads, edits, rank, value_dim = _validate_inputs(
        chi, d, e, z, log_decay, initial_state, chunk_size
    )
    chi = chi.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    z = z.contiguous()
    log_decay = log_decay.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    try:
        import fla.ops.generalized_delta_rule  # noqa: F401
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("the WY exterior requires flash-linear-attention") from error

    if edits == 1:
        q = chi
        k = d.squeeze(3)
        v = z.squeeze(3)
        g = log_decay
        packed_e = e.squeeze(3)
    else:
        q, k, v, packed_e, g = _pack_edits(chi, d, e, z, log_decay)

    expanded_output, final_state = _direct_e_dplr_delta_rule(
        q=q,
        k=k,
        v=v,
        e=packed_e,
        b=k,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )
    if edits == 1:
        return expanded_output, final_state
    output = expanded_output.reshape(
        batch, length, edits, heads, value_dim
    )[:, :, -1]
    return output, final_state


__all__ = ["wy_associative"]
