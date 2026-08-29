# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
# Specialized from FLA's MIT-licensed TileLang generalized-DPLR chunk_A and
# WY owners at commit 5e02dd3a7651f5f2797eb8b12bbec401826031e1.
"""FLA TileLang A-owner specialized to the paired direct-e recurrence.

The generic DPLR owner forms ``A_qk/A_qb/A_ak/A_ab``.  Here ``b == k == d``
exactly, so only ``A_qd`` and ``A_ed`` are physical.  The implementation keeps
FLA's centered Tensor Core operands and FP32 accumulators while loading the
frame-native panel layout directly.
"""

from __future__ import annotations

from functools import lru_cache

import tilelang
import tilelang.language as T
import torch


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise TypeError("direct-e TileLang panels must be BF16 or FP16")


@lru_cache(maxsize=None)
def _pair_fwd_kernel(
    batch: int,
    heads: int,
    rank: int,
    edits: int,
    frame_chunk: int,
    frame_chunks: int,
    logical_length: int,
    chunk_size: int,
    in_dtype: str,
):
    acc_dtype = "float32"
    chunks = (logical_length + chunk_size - 1) // chunk_size
    # Match FLA chunk_A's warp partition: a 16x16 output tile has one valid
    # MMA warp partition, while C32 uses four warps.
    threads = 32 if chunk_size < 32 else 128

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
        },
    )
    def build():
        @T.prim_func
        def pair_fwd(
            d: T.Tensor((batch * heads * frame_chunks, edits, frame_chunk, rank), in_dtype),
            paired: T.Tensor((batch * heads * frame_chunks, edits + 1, frame_chunk, rank), in_dtype),
            cumulative: T.Tensor((batch, logical_length, heads, rank), acc_dtype),
            q_scaled: T.Tensor((batch, logical_length, heads, rank), in_dtype),
            d_tail: T.Tensor((batch, logical_length, heads, rank), in_dtype),
            e_scaled: T.Tensor((batch, logical_length, heads, rank), in_dtype),
            A_qd: T.Tensor((batch, logical_length, heads, chunk_size), in_dtype),
            A_ed: T.Tensor((batch, logical_length, heads, chunk_size), acc_dtype),
        ):
            with T.Kernel(chunks, batch, heads, threads=threads) as (i_c, i_b, i_h):
                bos = i_c * chunk_size
                eos = T.min(bos + chunk_size, logical_length)
                valid_len = eos - bos
                last = T.max(eos - 1, 0)
                mid = valid_len // 2

                q_mat = T.alloc_shared((chunk_size, rank), in_dtype)
                e_mat = T.alloc_shared((chunk_size, rank), in_dtype)
                d_mat = T.alloc_shared((chunk_size, rank), in_dtype)
                gate_offset = T.alloc_shared((rank,), acc_dtype)
                gate_last = T.alloc_shared((rank,), acc_dtype)

                for c in T.Parallel(rank):
                    if bos < eos:
                        gate_offset[c] = cumulative[i_b, bos + mid, i_h, c]
                        gate_last[c] = cumulative[i_b, last, i_h, c]
                    else:
                        gate_offset[c] = 0.0
                        gate_last[c] = 0.0

                for r, c in T.Parallel(chunk_size, rank):
                    logical = bos + r
                    if logical < eos:
                        token = logical // edits
                        edit = logical % edits
                        panel = (i_b * heads + i_h) * frame_chunks + token // frame_chunk
                        row = token % frame_chunk
                        dv = T.Cast(acc_dtype, d[panel, edit, row, c])
                        ev = T.Cast(acc_dtype, paired[panel, edit, row, c])
                        qv = T.Cast(
                            acc_dtype,
                            T.if_then_else(
                                edit == edits - 1,
                                paired[panel, edits, row, c],
                                0.0,
                            ),
                        )
                        gv = cumulative[i_b, logical, i_h, c]
                        centered = gv - gate_offset[c]
                        q_mat[r, c] = T.Cast(in_dtype, qv * T.exp2(centered))
                        e_mat[r, c] = T.Cast(in_dtype, -ev * T.exp2(centered))
                        d_mat[r, c] = T.Cast(in_dtype, dv * T.exp2(-centered))
                        q_scaled[i_b, logical, i_h, c] = T.Cast(in_dtype, qv * T.exp2(gv))
                        e_scaled[i_b, logical, i_h, c] = T.Cast(in_dtype, -ev * T.exp2(gv))
                        d_tail[i_b, logical, i_h, c] = T.Cast(
                            in_dtype, dv * T.exp2(gate_last[c] - gv)
                        )
                    else:
                        q_mat[r, c] = T.Cast(in_dtype, 0.0)
                        e_mat[r, c] = T.Cast(in_dtype, 0.0)
                        d_mat[r, c] = T.Cast(in_dtype, 0.0)

                A_qd_frag = T.alloc_fragment((chunk_size, chunk_size), acc_dtype)
                A_ed_frag = T.alloc_fragment((chunk_size, chunk_size), acc_dtype)
                T.gemm(q_mat, d_mat, A_qd_frag, transpose_B=True, clear_accum=True)
                T.gemm(e_mat, d_mat, A_ed_frag, transpose_B=True, clear_accum=True)

                for r, c in T.Parallel(chunk_size, chunk_size):
                    logical = bos + r
                    if logical < eos:
                        A_qd[i_b, logical, i_h, c] = T.Cast(
                            in_dtype,
                            T.if_then_else(
                                (c < valid_len) and (r >= c), A_qd_frag[r, c], 0.0
                            ),
                        )
                        A_ed[i_b, logical, i_h, c] = T.Cast(
                            acc_dtype,
                            T.if_then_else(
                                (c < valid_len) and (r > c), A_ed_frag[r, c], 0.0
                            ),
                        )

        return pair_fwd

    return build()


@lru_cache(maxsize=None)
def _pair_bwd_kernel(
    batch: int,
    heads: int,
    rank: int,
    edits: int,
    frame_chunk: int,
    frame_chunks: int,
    logical_length: int,
    chunk_size: int,
    in_dtype: str,
):
    acc_dtype = "float32"
    chunks = (logical_length + chunk_size - 1) // chunk_size
    threads = 32 if chunk_size < 32 else 128

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
        },
    )
    def build():
        @T.prim_func
        def pair_bwd(
            d: T.Tensor((batch * heads * frame_chunks, edits, frame_chunk, rank), in_dtype),
            paired: T.Tensor((batch * heads * frame_chunks, edits + 1, frame_chunk, rank), in_dtype),
            cumulative: T.Tensor((batch, logical_length, heads, rank), acc_dtype),
            dA_qd: T.Tensor((batch, logical_length, heads, chunk_size), acc_dtype),
            dA_ed_0: T.Tensor((batch, logical_length, heads, chunk_size), acc_dtype),
            dA_ed_1: T.Tensor((batch, logical_length, heads, chunk_size), acc_dtype),
            dq_scaled: T.Tensor((batch, logical_length, heads, rank), in_dtype),
            dd_tail: T.Tensor((batch, logical_length, heads, rank), in_dtype),
            de_scaled: T.Tensor((batch, logical_length, heads, rank), in_dtype),
            dg_tail: T.Tensor((batch, chunks, heads, rank), acc_dtype),
            dd: T.Tensor((batch * heads * frame_chunks, edits, frame_chunk, rank), in_dtype),
            dpaired: T.Tensor((batch * heads * frame_chunks, edits + 1, frame_chunk, rank), in_dtype),
            dg: T.Tensor((batch, logical_length // edits, heads, rank), acc_dtype),
        ):
            with T.Kernel(chunks, batch, heads, threads=threads) as (i_c, i_b, i_h):
                bos = i_c * chunk_size
                eos = T.min(bos + chunk_size, logical_length)
                valid_len = eos - bos
                mid = valid_len // 2
                gate_offset = T.alloc_shared((rank,), acc_dtype)
                for c in T.Parallel(rank):
                    if bos < eos:
                        gate_offset[c] = cumulative[i_b, bos + mid, i_h, c]
                    else:
                        gate_offset[c] = 0.0

                q_mat = T.alloc_shared((chunk_size, rank), in_dtype)
                e_mat = T.alloc_shared((chunk_size, rank), in_dtype)
                d_mat = T.alloc_shared((chunk_size, rank), in_dtype)
                dA_qd_mat = T.alloc_shared((chunk_size, chunk_size), in_dtype)
                dA_ed_mat = T.alloc_shared((chunk_size, chunk_size), in_dtype)

                for r, c in T.Parallel(chunk_size, rank):
                    logical = bos + r
                    if logical < eos:
                        token = logical // edits
                        edit = logical % edits
                        panel = (i_b * heads + i_h) * frame_chunks + token // frame_chunk
                        row = token % frame_chunk
                        dv = T.Cast(acc_dtype, d[panel, edit, row, c])
                        ev = T.Cast(acc_dtype, paired[panel, edit, row, c])
                        qv = T.Cast(
                            acc_dtype,
                            T.if_then_else(
                                edit == edits - 1,
                                paired[panel, edits, row, c],
                                0.0,
                            ),
                        )
                        gv = cumulative[i_b, logical, i_h, c]
                        centered = gv - gate_offset[c]
                        q_mat[r, c] = T.Cast(in_dtype, qv * T.exp2(centered))
                        e_mat[r, c] = T.Cast(in_dtype, -ev * T.exp2(centered))
                        d_mat[r, c] = T.Cast(in_dtype, dv * T.exp2(-centered))
                    else:
                        q_mat[r, c] = T.Cast(in_dtype, 0.0)
                        e_mat[r, c] = T.Cast(in_dtype, 0.0)
                        d_mat[r, c] = T.Cast(in_dtype, 0.0)

                for r, c in T.Parallel(chunk_size, chunk_size):
                    logical = bos + r
                    if logical < eos:
                        dA_qd_mat[r, c] = T.Cast(
                            in_dtype,
                            T.if_then_else(
                                (c < valid_len) and (r >= c),
                                T.Cast(acc_dtype, dA_qd[i_b, logical, i_h, c]),
                                0.0,
                            ),
                        )
                        dA_ed_mat[r, c] = T.Cast(
                            in_dtype,
                            T.if_then_else(
                                (c < valid_len) and (r > c),
                                T.Cast(acc_dtype, dA_ed_0[i_b, logical, i_h, c])
                                + T.Cast(acc_dtype, dA_ed_1[i_b, logical, i_h, c]),
                                0.0,
                            ),
                        )
                    else:
                        dA_qd_mat[r, c] = T.Cast(in_dtype, 0.0)
                        dA_ed_mat[r, c] = T.Cast(in_dtype, 0.0)

                dchi_frag = T.alloc_fragment((chunk_size, rank), acc_dtype)
                de_frag = T.alloc_fragment((chunk_size, rank), acc_dtype)
                dd_frag = T.alloc_fragment((chunk_size, rank), acc_dtype)
                T.gemm(dA_qd_mat, d_mat, dchi_frag, clear_accum=True)
                T.gemm(dA_ed_mat, d_mat, de_frag, clear_accum=True)
                T.gemm(dA_qd_mat, q_mat, dd_frag, transpose_A=True, clear_accum=True)
                T.gemm(dA_ed_mat, e_mat, dd_frag, transpose_A=True)

                dG = T.alloc_shared((chunk_size, rank), acc_dtype)
                for r, c in T.Parallel(chunk_size, rank):
                    logical = bos + r
                    if logical < eos:
                        token = logical // edits
                        edit = logical % edits
                        panel = (i_b * heads + i_h) * frame_chunks + token // frame_chunk
                        row = token % frame_chunk
                        dv = T.Cast(acc_dtype, d[panel, edit, row, c])
                        ev = T.Cast(acc_dtype, paired[panel, edit, row, c])
                        qv = T.Cast(
                            acc_dtype,
                            T.if_then_else(
                                edit == edits - 1,
                                paired[panel, edits, row, c],
                                0.0,
                            ),
                        )
                        gv = cumulative[i_b, logical, i_h, c]
                        centered = gv - gate_offset[c]
                        exp_center = T.exp2(centered)
                        exp_g = T.exp2(gv)
                        q_center = qv * exp_center
                        e_center = -ev * exp_center
                        d_center = dv * T.exp2(-centered)

                        q_saved = T.Cast(acc_dtype, dq_scaled[i_b, logical, i_h, c])
                        e_saved = T.Cast(acc_dtype, de_scaled[i_b, logical, i_h, c])
                        d_saved = T.Cast(acc_dtype, dd_tail[i_b, logical, i_h, c])
                        last = T.max(eos - 1, 0)
                        tail_scale = T.exp2(cumulative[i_b, last, i_h, c] - gv)
                        gchi = dchi_frag[r, c] * exp_center + q_saved * exp_g
                        ge = -de_frag[r, c] * exp_center - e_saved * exp_g
                        gd = (
                            dd_frag[r, c] * T.exp2(-centered)
                            + d_saved * tail_scale
                        )

                        dd[panel, edit, row, c] = T.Cast(in_dtype, gd)
                        dpaired[panel, edit, row, c] = T.Cast(in_dtype, ge)
                        if edit == edits - 1:
                            dpaired[panel, edits, row, c] = T.Cast(in_dtype, gchi)

                        dG[r, c] = (
                            dchi_frag[r, c] * q_center
                            + de_frag[r, c] * e_center
                            - dd_frag[r, c] * d_center
                            + q_saved * (qv * exp_g)
                            + e_saved * (-ev * exp_g)
                            - d_saved * (dv * tail_scale)
                        )
                    else:
                        dG[r, c] = 0.0

                T.sync_threads()
                gate_acc = T.alloc_fragment((rank,), acc_dtype)
                for c in T.Parallel(rank):
                    gate_acc[c] = dg_tail[i_b, i_c, i_h, c]
                for reverse_row in T.serial(chunk_size):
                    r = chunk_size - 1 - reverse_row
                    logical = bos + r
                    for c in T.Parallel(rank):
                        gate_acc[c] += dG[r, c]
                        if logical < eos:
                            edit = logical % edits
                            if edit == 0:
                                token = logical // edits
                                dg[i_b, token, i_h, c] = gate_acc[c]

        return pair_bwd

    return build()


def direct_e_pair_forward(
    d: torch.Tensor,
    paired: torch.Tensor,
    cumulative: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    panels, edits, frame_chunk, rank = d.shape
    batch, logical_length, heads, _ = cumulative.shape
    source_length = logical_length // edits
    frame_chunks = (source_length + frame_chunk - 1) // frame_chunk
    if panels != batch * heads * frame_chunks:
        raise ValueError("frame panel count does not match rectangular layout")
    dtype_name = _dtype_name(d.dtype)
    q_scaled = torch.empty_like(cumulative, dtype=d.dtype)
    d_tail = torch.empty_like(q_scaled)
    e_scaled = torch.empty_like(q_scaled)
    A_qd = torch.empty(batch, logical_length, heads, chunk_size, dtype=d.dtype, device=d.device)
    A_ed = torch.empty(batch, logical_length, heads, chunk_size, dtype=torch.float32, device=d.device)
    kernel = _pair_fwd_kernel(
        batch, heads, rank, edits, frame_chunk, frame_chunks,
        logical_length, chunk_size, dtype_name,
    )
    kernel(d, paired, cumulative, q_scaled, d_tail, e_scaled, A_qd, A_ed)
    return A_qd, A_ed, q_scaled, d_tail, e_scaled


def direct_e_pair_backward(
    d: torch.Tensor,
    paired: torch.Tensor,
    cumulative: torch.Tensor,
    dA_qd: torch.Tensor,
    dA_ed_0: torch.Tensor,
    dA_ed_1: torch.Tensor,
    dq_scaled: torch.Tensor,
    dd_tail: torch.Tensor,
    de_scaled: torch.Tensor,
    dg_tail: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    panels, edits, frame_chunk, rank = d.shape
    batch, logical_length, heads, _ = cumulative.shape
    source_length = logical_length // edits
    frame_chunks = (source_length + frame_chunk - 1) // frame_chunk
    dd = torch.empty_like(d)
    dpaired = torch.empty_like(paired)
    dg = torch.empty(batch, source_length, heads, rank, dtype=torch.float32, device=d.device)
    kernel = _pair_bwd_kernel(
        batch, heads, rank, edits, frame_chunk, frame_chunks,
        logical_length, chunk_size, _dtype_name(d.dtype),
    )
    kernel(
        d, paired, cumulative, dA_qd, dA_ed_0, dA_ed_1,
        dq_scaled, dd_tail, de_scaled, dg_tail, dd, dpaired, dg,
    )
    return dd, dpaired, dg


__all__ = ["direct_e_pair_forward", "direct_e_pair_backward"]
