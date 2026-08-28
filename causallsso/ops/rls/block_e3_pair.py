# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
#
# The pair owner is specialized from FLA's MIT-licensed generalized-DPLR
# TileLang chunk_A schedule.  It changes the output axes to native
# [token, slot] E=3 ownership and removes the expanded gate panel.
"""Native token-block E=3 pair owner for SolveDelta."""

from __future__ import annotations

from functools import lru_cache

import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl


_E = 3
_TRITON_E = tl.constexpr(3)


@triton.jit
def _query_gauge_recompute_kernel(
    paired,
    cumulative,
    q_global,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    NT: tl.constexpr,
    BR: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    i_bh, chunk = panel // NT, panel % NT
    i_b, i_h = i_bh // H, i_bh % H
    token_bos = chunk * C
    valid = tl.minimum(C, T - token_bos)
    coord = tl.arange(0, BR)
    coord_mask = coord < R
    for token in range(C):
        token_mask = (token < valid) & coord_mask
        gate = tl.load(
            cumulative + ((i_b * T + token_bos + token) * H + i_h) * R + coord,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        query_offset = (
            ((panel * C + token) * (_TRITON_E + 1) + _TRITON_E) * R + coord
        )
        query = tl.load(
            paired + query_offset, mask=token_mask, other=0.0
        ).to(tl.float32)
        tl.store(
            q_global + (panel * C + token) * R + coord,
            query * tl.exp2(gate),
            mask=token_mask,
        )


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise TypeError("block-E3 panels must be BF16 or FP16")


@lru_cache(maxsize=None)
def _pair_fwd_kernel(
    batch: int,
    length: int,
    heads: int,
    rank: int,
    token_chunk: int,
    in_dtype: str,
):
    chunks = (length + token_chunk - 1) // token_chunk
    logical_chunk = _E * token_chunk
    left_rows = (_E + 1) * token_chunk
    threads = 64 if token_chunk <= 16 else 128

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
        },
    )
    def build():
        @T.prim_func
        def pair_fwd(
            d: T.Tensor(
                (batch * heads * chunks, token_chunk, _E, rank), in_dtype
            ),
            paired: T.Tensor(
                (batch * heads * chunks, token_chunk, _E + 1, rank),
                in_dtype,
            ),
            cumulative: T.Tensor((batch, length, heads, rank), "float32"),
            W: T.Tensor(
                (batch * heads * chunks, logical_chunk, logical_chunk),
                in_dtype,
            ),
            A: T.Tensor(
                (batch * heads * chunks, token_chunk, logical_chunk),
                in_dtype,
            ),
            e_global: T.Tensor(
                (batch * heads * chunks, logical_chunk, rank), in_dtype
            ),
            d_tail: T.Tensor(
                (batch * heads * chunks, logical_chunk, rank), in_dtype
            ),
            q_global: T.Tensor(
                (batch * heads * chunks, token_chunk, rank), in_dtype
            ),
        ):
            with T.Kernel(chunks, batch, heads, threads=threads) as (
                i_c,
                i_b,
                i_h,
            ):
                panel = (i_b * heads + i_h) * chunks + i_c
                token_bos = i_c * token_chunk
                valid_tokens = T.min(token_chunk, length - token_bos)
                middle = T.max(valid_tokens // 2, 0)
                last = T.max(valid_tokens - 1, 0)

                g_middle = T.alloc_shared((rank,), "float32")
                g_last = T.alloc_shared((rank,), "float32")
                for c in T.Parallel(rank):
                    if valid_tokens > 0:
                        g_middle[c] = cumulative[
                            i_b, token_bos + middle, i_h, c
                        ]
                        g_last[c] = cumulative[i_b, token_bos + last, i_h, c]
                    else:
                        g_middle[c] = 0.0
                        g_last[c] = 0.0

                left = T.alloc_shared((left_rows, rank), in_dtype)
                right = T.alloc_shared((logical_chunk, rank), in_dtype)
                for t, c in T.Parallel(token_chunk, rank):
                    valid = t < valid_tokens
                    token = token_bos + t
                    g = T.if_then_else(
                        valid, cumulative[i_b, token, i_h, c], 0.0
                    )
                    centered = g - g_middle[c]
                    for slot in T.unroll(_E):
                        dv = T.Cast(
                            "float32",
                            T.if_then_else(valid, d[panel, t, slot, c], 0.0),
                        )
                        ev = T.Cast(
                            "float32",
                            T.if_then_else(
                                valid, paired[panel, t, slot, c], 0.0
                            ),
                        )
                        left[t * (_E + 1) + slot, c] = T.Cast(
                            in_dtype, -ev * T.exp2(centered)
                        )
                        right[t * _E + slot, c] = T.Cast(
                            in_dtype, dv * T.exp2(-centered)
                        )
                        if valid:
                            logical = t * _E + slot
                            e_global[panel, logical, c] = T.Cast(
                                in_dtype, ev * T.exp2(g)
                            )
                            d_tail[panel, logical, c] = T.Cast(
                                in_dtype, dv * T.exp2(g_last[c] - g)
                            )

                    qv = T.Cast(
                        "float32",
                        T.if_then_else(
                            valid, paired[panel, t, _E, c], 0.0
                        ),
                    )
                    left[t * (_E + 1) + _E, c] = T.Cast(
                        in_dtype, qv * T.exp2(centered)
                    )
                    if valid:
                        q_global[panel, t, c] = T.Cast(
                            in_dtype, qv * T.exp2(g)
                        )

                # TileLang's SM120 MMA fragment layouts are power-of-two on
                # their output axes.  Keep the mathematical 64x48 product in
                # this owner, but compose it from mature 16x16 MMA tiles.
                pair_tile = T.alloc_fragment((32, 16), "float32")
                for left_block in T.serial(left_rows // 32):
                    for right_block in T.serial(logical_chunk // 16):
                        T.gemm(
                            left[
                                left_block * 32 : (left_block + 1) * 32,
                                0:rank,
                            ],
                            right[
                                right_block * 16 : (right_block + 1) * 16,
                                0:rank,
                            ],
                            pair_tile,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        for row, column in T.Parallel(32, 16):
                            left_row = left_block * 32 + row
                            pair_column = right_block * 16 + column
                            token_row = left_row // (_E + 1)
                            kind = left_row % (_E + 1)
                            token = token_bos + token_row
                            if (token_row < valid_tokens) and (kind < _E):
                                logical_row = token_row * _E + kind
                                value = T.if_then_else(
                                    pair_column < logical_row,
                                    pair_tile[row, column],
                                    0.0,
                                )
                                W[panel, logical_row, pair_column] = T.Cast(
                                    in_dtype, value
                                )
                            if (token_row < valid_tokens) and (kind == _E):
                                value = T.if_then_else(
                                    pair_column < (token_row + 1) * _E,
                                    pair_tile[row, column],
                                    0.0,
                                )
                                A[panel, token_row, pair_column] = T.Cast(
                                    in_dtype, value
                                )

        return pair_fwd

    return build()


def block_e3_pair_forward(
    d: torch.Tensor,
    paired: torch.Tensor,
    cumulative: torch.Tensor,
    *,
    token_chunk_size: int,
) -> tuple[torch.Tensor, ...]:
    """Build strict WY/read pairs and globally gauged compact panels."""
    panels, panel_tokens, edits, rank = d.shape
    if edits != _E or paired.shape != (
        panels,
        panel_tokens,
        _E + 1,
        rank,
    ):
        raise ValueError("native block-E3 pair expects three slots plus query")
    if panel_tokens != token_chunk_size:
        raise ValueError("source panel and token chunk widths must match")
    batch, length, heads, gate_rank = cumulative.shape
    if gate_rank != rank:
        raise ValueError("cumulative gate width must match source rank")
    chunks = (length + token_chunk_size - 1) // token_chunk_size
    if panels != batch * heads * chunks:
        raise ValueError("source panel count does not match token chunks")
    logical_chunk = _E * token_chunk_size
    W = torch.empty(
        panels, logical_chunk, logical_chunk, dtype=d.dtype, device=d.device
    )
    A = torch.empty(
        panels, token_chunk_size, logical_chunk, dtype=d.dtype, device=d.device
    )
    e_global = torch.empty(
        panels, logical_chunk, rank, dtype=d.dtype, device=d.device
    )
    d_tail = torch.empty_like(e_global)
    q_global = torch.empty(
        panels, token_chunk_size, rank, dtype=d.dtype, device=d.device
    )
    kernel = _pair_fwd_kernel(
        batch,
        length,
        heads,
        rank,
        token_chunk_size,
        _dtype_name(d.dtype),
    )
    kernel(d, paired, cumulative, W, A, e_global, d_tail, q_global)
    return W, A, e_global, d_tail, q_global


def block_e3_recompute_query_gauge(
    paired: torch.Tensor,
    cumulative: torch.Tensor,
    *,
    token_chunk_size: int,
) -> torch.Tensor:
    """Rebuild the short-lived query gauge for the state transpose."""
    panels, _, _, rank = paired.shape
    batch, length, heads, _ = cumulative.shape
    chunks = (length + token_chunk_size - 1) // token_chunk_size
    q_global = torch.empty(
        panels,
        token_chunk_size,
        rank,
        dtype=paired.dtype,
        device=paired.device,
    )
    _query_gauge_recompute_kernel[(panels,)](
        paired,
        cumulative,
        q_global,
        T=length,
        H=heads,
        R=rank,
        C=token_chunk_size,
        NT=chunks,
        BR=max(16, triton.next_power_of_2(rank)),
        num_warps=4,
        num_stages=1,
    )
    return q_global



__all__ = ["block_e3_pair_forward", "block_e3_recompute_query_gauge"]
