# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
#
# The inverse owner and its row update are specialized from FLA's
# MIT-licensed generalized-DPLR TileLang fast-WY implementation.
"""C48 fast-WY specialization for the token-block E=3 route."""

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
    raise TypeError("block-E3 WY panels must be BF16 or FP16")


@lru_cache(maxsize=None)
def _inverse48_kernel(
    batch: int,
    token_length: int,
    heads: int,
    token_chunk: int,
    in_dtype: str,
):
    if token_chunk != 16:
        raise ValueError("the first native block-E3 WY specialization is C16")
    chunks = (token_length + token_chunk - 1) // token_chunk
    logical_chunk = _E * token_chunk
    block = 16

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
        },
    )
    def build():
        @T.prim_func
        def inverse48(
            A_ab: T.Tensor(
                (batch * heads * chunks, logical_chunk, logical_chunk),
                in_dtype,
            ),
            inverse: T.Tensor(
                (batch * heads * chunks, logical_chunk, logical_chunk),
                in_dtype,
            ),
        ):
            with T.Kernel(chunks, batch, heads, threads=32) as (
                i_c,
                i_b,
                i_h,
            ):
                # Keep the specialized input dtype in the TileLang closure;
                # otherwise postponed annotation evaluation cannot resolve it.
                input_zero = T.Cast(in_dtype, 0.0)
                bos = i_c * logical_chunk
                token_bos = i_c * token_chunk
                valid_tokens = T.min(token_chunk, token_length - token_bos)
                valid_rows = valid_tokens * _E
                panel = (i_b * heads + i_h) * chunks + i_c

                I0 = T.alloc_shared((block, block), "float32")
                I1 = T.alloc_shared((block, block), "float32")
                I2 = T.alloc_shared((block, block), "float32")
                A10 = T.alloc_shared((block, block), "float32")
                A20 = T.alloc_shared((block, block), "float32")
                A21 = T.alloc_shared((block, block), "float32")
                for row, column in T.Parallel(block, block):
                    I0[row, column] = T.if_then_else(
                        (row > column) and (row < valid_rows),
                        A_ab[panel, row, column],
                        input_zero,
                    )
                    I1[row, column] = T.if_then_else(
                        (row > column) and (block + row < valid_rows),
                        A_ab[panel, block + row, block + column],
                        0.0,
                    )
                    I2[row, column] = T.if_then_else(
                        (row > column) and (2 * block + row < valid_rows),
                        A_ab[panel, 2 * block + row, 2 * block + column],
                        0.0,
                    )
                    A10[row, column] = T.if_then_else(
                        block + row < valid_rows,
                        A_ab[panel, block + row, column],
                        0.0,
                    )
                    A20[row, column] = T.if_then_else(
                        2 * block + row < valid_rows,
                        A_ab[panel, 2 * block + row, column],
                        0.0,
                    )
                    A21[row, column] = T.if_then_else(
                        2 * block + row < valid_rows,
                        A_ab[panel, 2 * block + row, block + column],
                        0.0,
                    )

                v0 = T.alloc_shared((block,), "float32")
                v1 = T.alloc_shared((block,), "float32")
                v2 = T.alloc_shared((block,), "float32")
                n0 = T.alloc_fragment((block,), "float32")
                n1 = T.alloc_fragment((block,), "float32")
                n2 = T.alloc_fragment((block,), "float32")
                T.clear(n0)
                T.clear(n1)
                T.clear(n2)
                for i in T.serial(block - 1):
                    row_i = i + 1
                    for column in T.Parallel(block):
                        v0[column] = I0[row_i, column]
                        v1[column] = I1[row_i, column]
                        v2[column] = I2[row_i, column]
                    for j in T.serial(1, row_i):
                        for column in T.Parallel(block):
                            if column < j:
                                n0[column] = n0[column] + v0[j] * I0[j, column]
                                n1[column] = n1[column] + v1[j] * I1[j, column]
                                n2[column] = n2[column] + v2[j] * I2[j, column]
                    for column in T.Parallel(block):
                        if column < row_i:
                            I0[row_i, column] = v0[column] + n0[column]
                            I1[row_i, column] = v1[column] + n1[column]
                            I2[row_i, column] = v2[column] + n2[column]
                        n0[column] = 0.0
                        n1[column] = 0.0
                        n2[column] = 0.0

                for row, column in T.Parallel(block, block):
                    if row == column:
                        I0[row, column] = I0[row, column] + 1.0
                        I1[row, column] = I1[row, column] + 1.0
                        I2[row, column] = I2[row, column] + 1.0

                tmp = T.alloc_fragment((block, block), "float32")
                tmp_shared = T.alloc_shared((block, block), "float32")
                X10 = T.alloc_fragment((block, block), "float32")
                X10_shared = T.alloc_shared((block, block), "float32")
                X21 = T.alloc_fragment((block, block), "float32")
                X20 = T.alloc_fragment((block, block), "float32")

                T.gemm(A10, I0, tmp, clear_accum=True)
                T.copy(tmp, tmp_shared)
                T.gemm(I1, tmp_shared, X10, clear_accum=True)
                T.copy(X10, X10_shared)

                T.gemm(A21, I1, tmp, clear_accum=True)
                T.copy(tmp, tmp_shared)
                T.gemm(I2, tmp_shared, X21, clear_accum=True)

                T.gemm(A20, I0, tmp, clear_accum=True)
                T.copy(tmp, tmp_shared)
                T.gemm(A21, X10_shared, tmp, clear_accum=False)
                T.copy(tmp, tmp_shared)
                T.gemm(I2, tmp_shared, X20, clear_accum=True)

                for row, column in T.Parallel(block, block):
                    if row < valid_rows:
                        inverse[panel, row, column] = I0[
                            row, column
                        ]
                        inverse[panel, row, block + column] = 0.0
                        inverse[panel, row, 2 * block + column] = 0.0
                    if block + row < valid_rows:
                        inverse[panel, block + row, column] = X10[
                            row, column
                        ]
                        inverse[panel, block + row, block + column] = I1[
                            row, column
                        ]
                        inverse[panel, block + row, 2 * block + column] = 0.0
                    if 2 * block + row < valid_rows:
                        inverse[panel, 2 * block + row, column] = X20[
                            row, column
                        ]
                        inverse[panel, 2 * block + row, block + column] = X21[
                            row, column
                        ]
                        inverse[panel, 2 * block + row, 2 * block + column] = I2[
                            row, column
                        ]

        return inverse48

    return build()


@lru_cache(maxsize=None)
def _rhs_fwd_kernel(
    batch: int,
    token_length: int,
    heads: int,
    rank: int,
    token_chunk: int,
    in_dtype: str,
):
    chunks = (token_length + token_chunk - 1) // token_chunk
    logical_chunk = _E * token_chunk
    row_parts = logical_chunk // 16
    threads = 32 if rank <= 32 else 128

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
        },
    )
    def build():
        @T.prim_func
        def rhs_fwd(
            e_global: T.Tensor(
                (batch * heads * chunks, logical_chunk, rank), in_dtype
            ),
            inverse: T.Tensor(
                (batch * heads * chunks, logical_chunk, logical_chunk),
                in_dtype,
            ),
            Y: T.Tensor(
                (batch * heads * chunks, logical_chunk, rank), in_dtype
            ),
            response: T.Tensor(
                (batch * heads * chunks, logical_chunk, token_chunk), in_dtype
            ),
        ):
            with T.Kernel(
                row_parts, chunks, batch * heads, threads=threads
            ) as (i_part, i_c, i_bh):
                i_b = i_bh // heads
                i_h = i_bh % heads
                bos = i_c * logical_chunk
                row_bos = i_part * 16
                token_bos = i_c * token_chunk
                valid_tokens = T.min(token_chunk, token_length - token_bos)
                valid_rows = valid_tokens * _E
                panel = (i_b * heads + i_h) * chunks + i_c

                inv_tile = T.alloc_shared((16, logical_chunk), in_dtype)
                rhs = T.alloc_shared((logical_chunk, rank), in_dtype)
                for row, column in T.Parallel(16, logical_chunk):
                    local_row = row_bos + row
                    logical = bos + local_row
                    if local_row < valid_rows:
                        inv_tile[row, column] = T.Cast(
                            in_dtype,
                            inverse[panel, local_row, column],
                        )
                    else:
                        inv_tile[row, column] = T.Cast(in_dtype, 0.0)

                for row, column in T.Parallel(logical_chunk, rank):
                    logical = bos + row
                    if row < valid_rows:
                        rhs[row, column] = e_global[panel, row, column]
                    else:
                        rhs[row, column] = T.Cast(in_dtype, 0.0)

                solved = T.alloc_fragment((16, rank), "float32")
                T.gemm(inv_tile, rhs, solved, clear_accum=True)
                for row, column in T.Parallel(16, rank):
                    local_row = row_bos + row
                    logical = bos + local_row
                    if local_row < valid_rows:
                        Y[panel, local_row, column] = T.Cast(
                            in_dtype, solved[row, column]
                        )

                # P selects slot two of every token, so W^-1 P is a gather
                # from the mature inverse rather than a value-side GEMM.
                for row, token in T.Parallel(16, token_chunk):
                    local_row = row_bos + row
                    logical = bos + local_row
                    if (local_row < valid_rows) and (token < valid_tokens):
                        response[panel, local_row, token] = T.Cast(
                            in_dtype,
                            inverse[panel, local_row, token * _E + (_E - 1)],
                        )

        return rhs_fwd

    return build()


def block_e3_wy_forward(
    W: torch.Tensor,
    e_global: torch.Tensor,
    *,
    token_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``Y=W^-1 E``, ``R=W^-1 P``, and the consumer-dtype inverse."""
    panels, logical_rows, logical_chunk = W.shape
    if logical_chunk != _E * token_chunk_size:
        raise ValueError("W width does not match E=3 token chunk")
    if logical_rows != logical_chunk:
        raise ValueError("block-E3 W must be square per panel")
    if e_global.shape[:2] != (panels, logical_chunk):
        raise ValueError("e_global shape does not match W")
    rank = e_global.shape[-1]
    # The panel order is fixed by the source owner.  Infer the rectangular
    # geometry from its metadata supplied by the caller-facing shape.
    if panels <= 0:
        raise ValueError("at least one panel is required")
    batch = 1
    heads = 1
    token_length = panels * token_chunk_size
    inverse = torch.empty(
        panels,
        logical_chunk,
        logical_chunk,
        dtype=W.dtype,
        device=W.device,
    )
    inverse_kernel = _inverse48_kernel(
        batch,
        token_length,
        heads,
        token_chunk_size,
        _dtype_name(W.dtype),
    )
    inverse_kernel(W, inverse)

    Y = torch.empty_like(e_global)
    response = torch.empty(
        panels, logical_chunk, token_chunk_size,
        dtype=e_global.dtype, device=e_global.device,
    )
    rhs_kernel = _rhs_fwd_kernel(
        batch,
        token_length,
        heads,
        rank,
        token_chunk_size,
        _dtype_name(e_global.dtype),
    )
    rhs_kernel(e_global, inverse, Y, response)
    return Y, response, inverse



__all__ = ["block_e3_wy_forward"]
