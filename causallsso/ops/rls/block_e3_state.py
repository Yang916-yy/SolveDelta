# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
#
# State/output ownership follows FLA generalized-DPLR chunk_h/chunk_o and
# Mamba-3's value-tile resident-state schedule.  The static E=3 compact action
# is SolveDelta-specific specialization glue.
"""Compact token-block state/output owner for SolveDelta."""

from __future__ import annotations

from functools import lru_cache

import tilelang
import tilelang.language as T
import torch


_E = 3


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise TypeError("block-E3 state panels must be BF16 or FP16")


@lru_cache(maxsize=None)
def _state_fwd_kernel(
    batch: int,
    length: int,
    heads: int,
    rank: int,
    value_dim: int,
    token_chunk: int,
    in_dtype: str,
):
    if token_chunk != 16:
        raise ValueError("the first native block-E3 state owner is C16")
    chunks = (length + token_chunk - 1) // token_chunk
    logical_chunk = _E * token_chunk
    value_tiles = (value_dim + 15) // 16
    BV = 16
    token_tile = 32
    threads = 32 if rank <= 32 else 64

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
        },
    )
    def build():
        @T.prim_func
        def state_fwd(
            Y: T.Tensor(
                (batch * heads * chunks, logical_chunk, rank), in_dtype
            ),
            d_tail: T.Tensor(
                (batch * heads * chunks, logical_chunk, rank), in_dtype
            ),
            q_star: T.Tensor(
                (batch * heads * chunks, token_chunk, rank), in_dtype
            ),
            b_z: T.Tensor(
                (batch * heads * chunks, token_chunk, token_chunk), in_dtype
            ),
            k_z: T.Tensor(
                (batch * heads * chunks, rank, token_chunk), in_dtype
            ),
            z: T.Tensor(
                (batch * heads * chunks, token_chunk, 1, value_dim),
                in_dtype,
            ),
            cumulative: T.Tensor((batch, length, heads, rank), "float32"),
            initial_state: T.Tensor(
                (batch, heads, rank, value_dim), "float32"
            ),
            state_cache: T.Tensor(
                (batch * heads * chunks, rank, value_dim), in_dtype
            ),
            output: T.Tensor((batch, length, heads, value_dim), in_dtype),
            final_state: T.Tensor(
                (batch, heads, rank, value_dim), "float32"
            ),
        ):
            with T.Kernel(value_tiles, batch * heads, threads=threads) as (
                i_v,
                i_bh,
            ):
                i_b = i_bh // heads
                i_h = i_bh % heads
                value_bos = i_v * BV
                state = T.alloc_fragment((rank, BV), "float32")
                for r, v in T.Parallel(rank, BV):
                    if value_bos + v < value_dim:
                        state[r, v] = initial_state[
                            i_b, i_h, r, value_bos + v
                        ]
                    else:
                        state[r, v] = 0.0

                state_low = T.alloc_shared((rank, BV), in_dtype)
                y_tile = T.alloc_shared((token_tile, rank), in_dtype)
                d_tile = T.alloc_shared((token_tile, rank), in_dtype)
                vs = T.alloc_fragment((token_tile, BV), "float32")
                vs_low = T.alloc_shared((token_tile, BV), in_dtype)
                state_correction = T.alloc_fragment((rank, BV), "float32")
                q_tile = T.alloc_shared((token_tile, rank), in_dtype)
                bz_tile = T.alloc_shared((token_tile, token_tile), in_dtype)
                kz_tile = T.alloc_shared((rank, token_tile), in_dtype)
                z_tile = T.alloc_shared((token_tile, BV), in_dtype)
                out = T.alloc_fragment((token_tile, BV), "float32")
                kz_update = T.alloc_fragment((rank, BV), "float32")

                for chunk in T.serial(chunks):
                    panel = (i_b * heads + i_h) * chunks + chunk
                    token_bos = chunk * token_chunk
                    valid_tokens = T.min(token_chunk, length - token_bos)
                    valid_rows = valid_tokens * _E
                    for r, v in T.Parallel(rank, BV):
                        if value_bos + v < value_dim:
                            state_cache[panel, r, value_bos + v] = T.Cast(
                                in_dtype, state[r, v]
                            )
                        state_low[r, v] = T.Cast(in_dtype, state[r, v])

                    T.clear(state_correction)
                    for part in T.serial(
                        (logical_chunk + token_tile - 1) // token_tile
                    ):
                        row_bos = part * token_tile
                        for row, r in T.Parallel(token_tile, rank):
                            logical = row_bos + row
                            if logical < valid_rows:
                                y_tile[row, r] = Y[panel, logical, r]
                                d_tile[row, r] = d_tail[panel, logical, r]
                            else:
                                y_tile[row, r] = T.Cast(in_dtype, 0.0)
                                d_tile[row, r] = T.Cast(in_dtype, 0.0)
                        T.gemm(
                            y_tile, state_low, vs, clear_accum=True
                        )
                        for row, v in T.Parallel(token_tile, BV):
                            vs_low[row, v] = T.Cast(in_dtype, vs[row, v])
                        T.gemm(
                            d_tile,
                            vs_low,
                            state_correction,
                            transpose_A=True,
                            clear_accum=False,
                        )

                    for row, r in T.Parallel(token_tile, rank):
                        if row < valid_tokens:
                            q_tile[row, r] = q_star[panel, row, r]
                        else:
                            q_tile[row, r] = T.Cast(in_dtype, 0.0)
                    for row, column in T.Parallel(token_tile, token_tile):
                        if (row < valid_tokens) and (column < valid_tokens):
                            bz_tile[row, column] = b_z[panel, row, column]
                        else:
                            bz_tile[row, column] = T.Cast(in_dtype, 0.0)
                    for r, column in T.Parallel(rank, token_tile):
                        if column < valid_tokens:
                            kz_tile[r, column] = k_z[panel, r, column]
                        else:
                            kz_tile[r, column] = T.Cast(in_dtype, 0.0)
                    for row, v in T.Parallel(token_tile, BV):
                        if (row < valid_tokens) and (
                            value_bos + v < value_dim
                        ):
                            z_tile[row, v] = z[panel, row, 0, value_bos + v]
                        else:
                            z_tile[row, v] = T.Cast(in_dtype, 0.0)

                    T.gemm(q_tile, state_low, out, clear_accum=True)
                    T.gemm(bz_tile, z_tile, out, clear_accum=False)
                    T.gemm(kz_tile, z_tile, kz_update, clear_accum=True)

                    last_token = token_bos + T.max(valid_tokens - 1, 0)
                    for r, v in T.Parallel(rank, BV):
                        decay = T.if_then_else(
                            valid_tokens > 0,
                            T.exp2(cumulative[i_b, last_token, i_h, r]),
                            1.0,
                        )
                        state[r, v] = (
                            decay * state[r, v]
                            - state_correction[r, v]
                            + kz_update[r, v]
                        )

                    for row, v in T.Parallel(token_tile, BV):
                        if (row < valid_tokens) and (
                            value_bos + v < value_dim
                        ):
                            output[
                                i_b,
                                token_bos + row,
                                i_h,
                                value_bos + v,
                            ] = T.Cast(in_dtype, out[row, v])

                for r, v in T.Parallel(rank, BV):
                    if value_bos + v < value_dim:
                        final_state[i_b, i_h, r, value_bos + v] = state[r, v]

        return state_fwd

    return build()


def block_e3_action_statistics(
    A: torch.Tensor,
    Y: torch.Tensor,
    response: torch.Tensor,
    q_global: torch.Tensor,
    d_tail: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build compact Q*, Bz, and Kz with library Tensor-Core GEMMs."""
    ay = torch.bmm(A, Y, out_dtype=torch.float32)
    q_star = (q_global.float() - ay).to(q_global.dtype)
    b_z = torch.bmm(A, response, out_dtype=torch.float32).to(A.dtype)
    k_z = torch.bmm(
        d_tail.transpose(1, 2), response, out_dtype=torch.float32
    ).to(d_tail.dtype)
    return q_star, b_z, k_z


def block_e3_state_forward(
    Y: torch.Tensor,
    d_tail: torch.Tensor,
    q_star: torch.Tensor,
    b_z: torch.Tensor,
    k_z: torch.Tensor,
    z: torch.Tensor,
    cumulative: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    token_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length, heads, rank = cumulative.shape
    value_dim = z.shape[-1]
    chunks = (length + token_chunk_size - 1) // token_chunk_size
    panels = batch * heads * chunks
    if Y.shape[:2] != (panels, _E * token_chunk_size):
        raise ValueError("Y panel shape does not match token chunks")
    if z.shape != (panels, token_chunk_size, 1, value_dim):
        raise ValueError("z must use native token-panel layout")
    state_cache = torch.empty(
        panels, rank, value_dim, dtype=Y.dtype, device=Y.device
    )
    output = torch.empty(
        batch, length, heads, value_dim, dtype=z.dtype, device=z.device
    )
    final_state = torch.empty_like(initial_state, dtype=torch.float32)
    kernel = _state_fwd_kernel(
        batch,
        length,
        heads,
        rank,
        value_dim,
        token_chunk_size,
        _dtype_name(Y.dtype),
    )
    kernel(
        Y,
        d_tail,
        q_star,
        b_z,
        k_z,
        z,
        cumulative,
        initial_state,
        state_cache,
        output,
        final_state,
    )
    return output, final_state, state_cache


__all__ = [
    "block_e3_action_statistics",
    "block_e3_state_forward",
]
