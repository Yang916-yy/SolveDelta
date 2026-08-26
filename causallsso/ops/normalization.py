# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Adapted from flash-linear-attention's MIT-licensed L2Norm kernels. The
# SolveDelta specialization implements x / max(||x||_2, eps), accepts the
# original projected strides, and batches geometry, query, and edit keys.

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


_EPSILON = 1.0e-12


@triton.jit
def _normalize_frame_fwd_kernel(
    u,
    h,
    q,
    keys,
    erase_raw,
    out_u_panel,
    out_u_d_panel,
    out_h_panel,
    out_key_panel,
    out_paired_dual,
    signed_inverse,
    cu_seqlens,
    chunk_indices,
    SU_B: tl.constexpr,
    SU_T: tl.constexpr,
    SU_H: tl.constexpr,
    SU_R: tl.constexpr,
    SH_B: tl.constexpr,
    SH_T: tl.constexpr,
    SH_H: tl.constexpr,
    SH_R: tl.constexpr,
    SQ_B: tl.constexpr,
    SQ_T: tl.constexpr,
    SQ_H: tl.constexpr,
    SQ_R: tl.constexpr,
    SK_B: tl.constexpr,
    SK_T: tl.constexpr,
    SK_H: tl.constexpr,
    SK_E: tl.constexpr,
    SK_R: tl.constexpr,
    SE_B: tl.constexpr,
    SE_T: tl.constexpr,
    SE_H: tl.constexpr,
    SE_E: tl.constexpr,
    SE_R: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    BR: tl.constexpr,
    EPSILON: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    vector = tl.program_id(0).to(tl.int64)
    panel_vectors = tl.num_programs(0) // (E + 2)
    u_route = vector < panel_vectors
    q_route = (vector >= panel_vectors) & (vector < 2 * panel_vectors)
    local = tl.where(
        u_route,
        vector,
        tl.where(q_route, vector - panel_vectors, vector - 2 * panel_vectors),
    )
    panel = tl.where(u_route | q_route, local // C, local // (E * C))
    panel_offset = tl.where(u_route | q_route, local % C, local % (E * C))
    edit = tl.where(u_route | q_route, 0, panel_offset // C)
    row = tl.where(u_route | q_route, panel_offset, panel_offset % C)
    if IS_VARLEN:
        global_chunk = panel // H
        head = panel % H
        sequence = tl.load(chunk_indices + global_chunk * 2).to(tl.int32)
        frame_chunk = tl.load(chunk_indices + global_chunk * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + sequence).to(tl.int64)
        eos = tl.load(cu_seqlens + sequence + 1).to(tl.int64)
        batch = 0
        token = bos + frame_chunk * C + row
        valid_token = token < eos
        output_token = token
    else:
        frame_chunk = panel % N
        batch_head = panel // N
        head = batch_head % H
        batch = batch_head // H
        token = frame_chunk * C + row
        valid_token = token < T
        output_token = batch * T + token
    coordinate = tl.arange(0, BR)
    mask = (coordinate < R) & valid_token
    p_u = batch * SU_B + token * SU_T + head * SU_H + coordinate * SU_R
    p_q = batch * SQ_B + token * SQ_T + head * SQ_H + coordinate * SQ_R
    p_h = batch * SH_B + token * SH_T + head * SH_H + coordinate * SH_R
    p_k = (
        batch * SK_B
        + token * SK_T
        + head * SK_H
        + edit * SK_E
        + coordinate * SK_R
    )
    p_e = (
        batch * SE_B
        + token * SE_T
        + head * SE_H
        + edit * SE_E
        + coordinate * SE_R
    )
    source = tl.where(u_route, u + p_u, tl.where(q_route, q + p_q, keys + p_k))
    values = tl.load(source, mask=mask, other=0.0).to(tl.float32)
    norm = tl.sqrt(tl.sum(values * values, axis=0))
    active = norm > EPSILON
    inverse = tl.where(active, 1.0 / norm, 1.0 / EPSILON)
    result = values * inverse
    tl.store(
        out_u_d_panel + local * R + coordinate,
        result,
        mask=(coordinate < R) & u_route,
    )
    tl.store(
        out_u_panel + local * R + coordinate,
        result,
        mask=(coordinate < R) & u_route,
    )
    h_value = tl.load(h + p_h, mask=mask & u_route, other=0.0)
    tl.store(
        out_h_panel + local * R + coordinate,
        h_value,
        mask=(coordinate < R) & u_route,
    )
    tl.store(
        out_key_panel + local * R + coordinate,
        result,
        mask=(coordinate < R) & ~(u_route | q_route),
    )
    erase_logit = tl.load(
        erase_raw + p_e,
        mask=mask & ~(u_route | q_route),
        other=0.0,
    ).to(tl.float32)
    erase_value = 2.0 * tl.sigmoid(erase_logit)
    dual_rhs = tl.where(q_route, E, edit)
    dual_base = ((panel * (E + 1) + dual_rhs) * C + row) * R
    dual_value = tl.where(q_route, result, result * erase_value)
    tl.store(
        out_paired_dual + dual_base + coordinate,
        dual_value,
        mask=(coordinate < R) & ~u_route,
    )
    tl.store(signed_inverse + vector, tl.where(active, inverse, -inverse))


@triton.jit
def _normalize_frame_bwd_kernel(
    u,
    q,
    keys,
    erase_raw,
    signed_inverse,
    grad_u_panel,
    grad_h_panel,
    grad_key_panel,
    grad_paired_dual,
    du,
    dh,
    dq,
    dkeys,
    derase,
    cu_seqlens,
    chunk_indices,
    SU_B: tl.constexpr,
    SU_T: tl.constexpr,
    SU_H: tl.constexpr,
    SU_R: tl.constexpr,
    SQ_B: tl.constexpr,
    SQ_T: tl.constexpr,
    SQ_H: tl.constexpr,
    SQ_R: tl.constexpr,
    SK_B: tl.constexpr,
    SK_T: tl.constexpr,
    SK_H: tl.constexpr,
    SK_E: tl.constexpr,
    SK_R: tl.constexpr,
    SE_B: tl.constexpr,
    SE_T: tl.constexpr,
    SE_H: tl.constexpr,
    SE_E: tl.constexpr,
    SE_R: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    BR: tl.constexpr,
    HAS_GRAD_U_PANEL: tl.constexpr,
    HAS_GRAD_H_PANEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    vector = tl.program_id(0).to(tl.int64)
    panel_vectors = tl.num_programs(0) // (E + 2)
    u_route = vector < panel_vectors
    q_route = (vector >= panel_vectors) & (vector < 2 * panel_vectors)
    local = tl.where(
        u_route,
        vector,
        tl.where(q_route, vector - panel_vectors, vector - 2 * panel_vectors),
    )
    panel = tl.where(u_route | q_route, local // C, local // (E * C))
    panel_offset = tl.where(u_route | q_route, local % C, local % (E * C))
    edit = tl.where(u_route | q_route, 0, panel_offset // C)
    row = tl.where(u_route | q_route, panel_offset, panel_offset % C)
    if IS_VARLEN:
        global_chunk = panel // H
        head = panel % H
        sequence = tl.load(chunk_indices + global_chunk * 2).to(tl.int32)
        frame_chunk = tl.load(chunk_indices + global_chunk * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + sequence).to(tl.int64)
        eos = tl.load(cu_seqlens + sequence + 1).to(tl.int64)
        batch = 0
        token = bos + frame_chunk * C + row
        valid_token = token < eos
        output_token = token
    else:
        frame_chunk = panel % N
        batch_head = panel // N
        head = batch_head % H
        batch = batch_head // H
        token = frame_chunk * C + row
        valid_token = token < T
        output_token = batch * T + token
    coordinate = tl.arange(0, BR)
    mask = (coordinate < R) & valid_token
    p_u = batch * SU_B + token * SU_T + head * SU_H + coordinate * SU_R
    p_q = batch * SQ_B + token * SQ_T + head * SQ_H + coordinate * SQ_R
    p_k = (
        batch * SK_B
        + token * SK_T
        + head * SK_H
        + edit * SK_E
        + coordinate * SK_R
    )
    p_e = (
        batch * SE_B
        + token * SE_T
        + head * SE_H
        + edit * SE_E
        + coordinate * SE_R
    )
    source = tl.where(u_route, u + p_u, tl.where(q_route, q + p_q, keys + p_k))
    values = tl.load(source, mask=mask, other=0.0).to(tl.float32)
    bth = (output_token * H + head) * R + coordinate
    dual_rhs = tl.where(q_route, E, edit)
    dual_base = ((panel * (E + 1) + dual_rhs) * C + row) * R + coordinate
    dual_gradient = tl.load(
        grad_paired_dual + dual_base,
        mask=mask & ~u_route,
        other=0.0,
    ).to(tl.float32)
    erase_logit = tl.load(
        erase_raw + p_e,
        mask=mask & ~(u_route | q_route),
        other=0.0,
    ).to(tl.float32)
    erase_sigmoid = tl.sigmoid(erase_logit)
    erase_value = 2.0 * erase_sigmoid
    u_gradient = tl.zeros([BR], dtype=tl.float32)
    if HAS_GRAD_U_PANEL:
        u_gradient += tl.load(
            grad_u_panel + local * R + coordinate, mask=mask, other=0.0
        ).to(tl.float32)
    key_gradient = tl.load(
        grad_key_panel + local * R + coordinate,
        mask=mask & ~(u_route | q_route),
        other=0.0,
    ).to(tl.float32)
    gradient = tl.where(
        u_route,
        u_gradient,
        tl.where(q_route, dual_gradient, key_gradient + dual_gradient * erase_value),
    )
    signed = tl.load(signed_inverse + vector).to(tl.float32)
    active = signed > 0.0
    inverse = tl.abs(signed)
    projection = tl.sum(gradient * values, axis=0)
    result = tl.where(
        active,
        gradient * inverse - values * projection * inverse * inverse * inverse,
        gradient * inverse,
    )
    output = tl.where(u_route, du, tl.where(q_route, dq, dkeys))
    output_base = tl.where(
        u_route | q_route,
        (output_token * H + head) * R,
        ((output_token * H + head) * E + edit) * R,
    )
    tl.store(output + output_base + coordinate, result, mask=mask)
    h_gradient = tl.zeros([BR], dtype=tl.float32)
    if HAS_GRAD_H_PANEL:
        h_gradient += tl.load(
            grad_h_panel + local * R + coordinate,
            mask=mask & u_route,
            other=0.0,
        ).to(tl.float32)
    tl.store(dh + bth, h_gradient, mask=mask & u_route)
    normalized = values * inverse
    tl.store(
        derase + output_base + coordinate,
        dual_gradient
        * normalized
        * (2.0 * erase_sigmoid * (1.0 - erase_sigmoid)),
        mask=mask & ~(u_route | q_route),
    )


class _NormalizeFrameInputs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,
        h: torch.Tensor,
        q: torch.Tensor,
        keys: torch.Tensor,
        erase_raw: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        chunk_indices: torch.Tensor | None,
        chunk_size: int,
    ):
        batch, length, heads, width = u.shape
        edits = keys.shape[-2]
        chunks = triton.cdiv(length, chunk_size)
        panels = (
            batch * heads * chunks
            if chunk_indices is None
            else len(chunk_indices) * heads
        )
        out_u_panel = torch.empty(
            panels, chunk_size, width, dtype=torch.float16, device=u.device
        )
        out_u_d_panel = torch.empty(
            panels, chunk_size, width, dtype=torch.bfloat16, device=u.device
        )
        out_h_panel = torch.empty(
            panels, chunk_size, width, dtype=torch.bfloat16, device=u.device
        )
        out_key_panel = torch.empty(
            panels, edits, chunk_size, width, dtype=torch.float16, device=u.device
        )
        out_paired_dual = torch.empty(
            panels, edits + 1, chunk_size, width,
            dtype=torch.float16, device=u.device,
        )
        panel_vectors = panels * chunk_size
        vectors = panel_vectors * (edits + 2)
        signed_inverse = torch.empty(vectors, dtype=torch.float32, device=u.device)
        block = triton.next_power_of_2(width)
        _normalize_frame_fwd_kernel[(vectors,)](
            u,
            h,
            q,
            keys,
            erase_raw,
            out_u_panel,
            out_u_d_panel,
            out_h_panel,
            out_key_panel,
            out_paired_dual,
            signed_inverse,
            cu_seqlens if cu_seqlens is not None else u,
            chunk_indices if chunk_indices is not None else u,
            SU_B=u.stride(0), SU_T=u.stride(1), SU_H=u.stride(2), SU_R=u.stride(3),
            SH_B=h.stride(0), SH_T=h.stride(1), SH_H=h.stride(2), SH_R=h.stride(3),
            SQ_B=q.stride(0), SQ_T=q.stride(1), SQ_H=q.stride(2), SQ_R=q.stride(3),
            SK_B=keys.stride(0), SK_T=keys.stride(1), SK_H=keys.stride(2),
            SK_E=keys.stride(3), SK_R=keys.stride(4),
            SE_B=erase_raw.stride(0), SE_T=erase_raw.stride(1), SE_H=erase_raw.stride(2),
            SE_E=erase_raw.stride(3), SE_R=erase_raw.stride(4),
            T=length, H=heads, E=edits, C=chunk_size, N=chunks,
            R=width, BR=block, EPSILON=_EPSILON,
            IS_VARLEN=cu_seqlens is not None,
            num_warps=4, num_stages=1,
        )
        saved_cu = cu_seqlens if cu_seqlens is not None else u.new_empty(0, dtype=torch.long)
        saved_chunks = chunk_indices if chunk_indices is not None else u.new_empty(0, 2, dtype=torch.long)
        ctx.save_for_backward(
            u, q, keys, erase_raw, signed_inverse, saved_cu, saved_chunks
        )
        ctx.shape = (
            length, heads, edits, chunk_size, chunks, width, block, panels,
            cu_seqlens is not None,
        )
        ctx.set_materialize_grads(False)
        return (
            out_u_panel,
            out_u_d_panel,
            out_h_panel,
            out_key_panel,
            out_paired_dual,
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        grad_u_panel: torch.Tensor | None,
        grad_u_d_panel: torch.Tensor | None,
        grad_h_panel: torch.Tensor | None,
        grad_key_panel: torch.Tensor,
        grad_paired_dual: torch.Tensor,
    ):
        u, q, keys, erase_raw, signed_inverse, saved_cu, saved_chunks = ctx.saved_tensors
        (
            length, heads, edits, chunk_size, chunks, width, block, panels,
            is_varlen,
        ) = ctx.shape
        batch = u.shape[0]
        du = torch.empty(u.shape, dtype=u.dtype, device=u.device)
        dh = torch.empty(u.shape, dtype=u.dtype, device=u.device)
        dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        dkeys = torch.empty(keys.shape, dtype=keys.dtype, device=keys.device)
        derase_raw = torch.empty(
            erase_raw.shape, dtype=erase_raw.dtype, device=erase_raw.device
        )
        vectors = panels * chunk_size * (edits + 2)
        _normalize_frame_bwd_kernel[(vectors,)](
            u,
            q,
            keys,
            erase_raw,
            signed_inverse,
            grad_u_panel if grad_u_panel is not None else u,
            grad_h_panel if grad_h_panel is not None else u,
            grad_key_panel,
            grad_paired_dual,
            du,
            dh,
            dq,
            dkeys,
            derase_raw,
            saved_cu if is_varlen else u,
            saved_chunks if is_varlen else u,
            SU_B=u.stride(0), SU_T=u.stride(1), SU_H=u.stride(2), SU_R=u.stride(3),
            SQ_B=q.stride(0), SQ_T=q.stride(1), SQ_H=q.stride(2), SQ_R=q.stride(3),
            SK_B=keys.stride(0), SK_T=keys.stride(1), SK_H=keys.stride(2),
            SK_E=keys.stride(3), SK_R=keys.stride(4),
            SE_B=erase_raw.stride(0), SE_T=erase_raw.stride(1), SE_H=erase_raw.stride(2),
            SE_E=erase_raw.stride(3), SE_R=erase_raw.stride(4),
            T=length, H=heads, E=edits, C=chunk_size, N=chunks,
            R=width, BR=block,
            HAS_GRAD_U_PANEL=grad_u_panel is not None,
            HAS_GRAD_H_PANEL=grad_h_panel is not None,
            IS_VARLEN=is_varlen,
            num_warps=4, num_stages=1,
        )
        return du, dh, dq, dkeys, derase_raw, None, None, None


def normalize_frame_inputs(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    erase_raw: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int,
) -> tuple[torch.Tensor, ...]:
    """Normalize strided BF16 frame inputs with one FLA-style reduction owner."""
    if u.ndim != 4 or h.shape != u.shape or q.shape != u.shape:
        raise ValueError("u, h, and q must share [B,T,H,r]")
    if keys.ndim != 5 or keys.shape[:3] != u.shape[:3] or keys.shape[-1] != u.shape[-1]:
        raise ValueError("keys must have shape [B,T,H,K,r]")
    if erase_raw.shape != keys.shape or erase_raw.dtype != torch.bfloat16:
        raise ValueError("erase_raw must be BF16 with the same shape as keys")
    if (
        u.dtype != torch.bfloat16
        or h.dtype != u.dtype
        or q.dtype != u.dtype
        or keys.dtype != u.dtype
    ):
        raise TypeError("normalization inputs must be BF16")
    if u.device.type != "cuda" or any(
        x.device != u.device for x in (h, q, keys, erase_raw)
    ):
        raise ValueError("normalization inputs must share one CUDA device")
    if any(tensor.stride(-1) != 1 for tensor in (u, h, q, keys, erase_raw)):
        raise ValueError("normalization inputs require unit coordinate stride")
    if cu_seqlens is not None:
        if u.shape[0] != 1:
            raise ValueError("variable-length frame normalization requires batch size one")
        if chunk_indices is None:
            from fla.ops.utils import prepare_chunk_indices

            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    return _NormalizeFrameInputs.apply(
        u, h, q, keys, erase_raw, cu_seqlens, chunk_indices, chunk_size
    )


__all__ = ["normalize_frame_inputs"]
