# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
#
# The tile ownership follows FLA generalized-DPLR chunk_A transpose and
# GDN2's fused WY/intra reverse. C48 is composed from C16 MMA tiles.
"""Generate-use-discard WY and direct-e pair transpose for native E=3."""

from __future__ import annotations

from functools import lru_cache

import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl


_E = 3


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise TypeError("block-E3 pair reverse panels must be BF16 or FP16")


@lru_cache(maxsize=None)
def _fused_pair_reverse_kernel(
    batch: int,
    length: int,
    heads: int,
    rank: int,
    token_chunk: int,
    in_dtype: str,
):
    if token_chunk != 16:
        raise ValueError("the fused block-E3 pair transpose is C16")
    chunks = (length + token_chunk - 1) // token_chunk
    logical_chunk = _E * token_chunk
    blocks = logical_chunk // 16
    # The owner contains both 16x16 pair tiles and 16xr source tiles.  Two
    # warps are the largest square-policy partition valid for the former;
    # chunk-level parallelism supplies occupancy at the target shape.
    threads = 32 if rank <= 32 else 64

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
        },
    )
    def build():
        @T.prim_func
        def pair_reverse(
            u: T.Tensor((batch, length, heads, rank), in_dtype),
            gain: T.Tensor((batch, length, heads, rank), in_dtype),
            geometry_log_decay: T.Tensor((batch, length, heads), "float32"),
            associative_log_decay: T.Tensor(
                (batch, length, heads, rank), "float32"
            ),
            previous_mass: T.Tensor((batch, length, heads), "float32"),
            current_mass: T.Tensor((batch, length, heads), "float32"),
            strength: T.Tensor((heads,), "float32"),
            d: T.Tensor(
                (batch * heads * chunks, token_chunk, _E, rank), in_dtype
            ),
            paired: T.Tensor(
                (batch * heads * chunks, token_chunk, _E + 1, rank),
                in_dtype,
            ),
            cumulative: T.Tensor((batch, length, heads, rank), "float32"),
            Y: T.Tensor(
                (batch * heads * chunks, logical_chunk, rank), in_dtype
            ),
            response: T.Tensor(
                (batch * heads * chunks, logical_chunk, token_chunk),
                in_dtype,
            ),
            grad_e: T.Tensor(
                (batch * heads * chunks, logical_chunk, rank), "float32"
            ),
            grad_injection: T.Tensor(
                (batch * heads * chunks, logical_chunk, token_chunk),
                "float32",
            ),
            grad_d: T.Tensor(
                (batch * heads * chunks, logical_chunk, rank), "float32"
            ),
            grad_q: T.Tensor(
                (batch * heads * chunks, token_chunk, rank), "float32"
            ),
            grad_tail_seed: T.Tensor(
                (batch * heads * chunks, rank), "float32"
            ),
            grad_u: T.Tensor((batch, length, heads, rank), in_dtype),
            grad_h: T.Tensor((batch, length, heads, rank), in_dtype),
            grad_gain: T.Tensor((batch, length, heads, rank), in_dtype),
            grad_prediction: T.Tensor(
                (batch, length, heads, rank), in_dtype
            ),
            grad_geometry_log_decay: T.Tensor(
                (batch, length, heads), "float32"
            ),
            grad_associative_log_decay: T.Tensor(
                (batch, length, heads, rank), "float32"
            ),
            grad_previous_mass: T.Tensor(
                (batch, length, heads), "float32"
            ),
            grad_current_mass: T.Tensor(
                (batch, length, heads), "float32"
            ),
            grad_strength_partial: T.Tensor(
                (batch, length, heads), "float32"
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
                valid_rows = valid_tokens * _E
                middle = T.max(valid_tokens // 2, 0)
                last = T.max(valid_tokens - 1, 0)
                g_middle = T.alloc_shared((rank,), "float32")
                for c in T.Parallel(rank):
                    if valid_tokens > 0:
                        g_middle[c] = cumulative[
                            i_b, token_bos + middle, i_h, c
                        ]
                    else:
                        g_middle[c] = 0.0

                by = T.alloc_shared((logical_chunk, rank), in_dtype)
                y = T.alloc_shared((logical_chunk, rank), in_dtype)
                br = T.alloc_shared(
                    (logical_chunk, token_chunk), in_dtype
                )
                response_shared = T.alloc_shared(
                    (logical_chunk, token_chunk), in_dtype
                )
                for logical, c in T.Parallel(logical_chunk, rank):
                    valid = logical < valid_rows
                    by[logical, c] = T.if_then_else(
                        valid, grad_e[panel, logical, c], 0.0
                    )
                    y[logical, c] = T.if_then_else(
                        valid, Y[panel, logical, c], 0.0
                    )
                for logical, token in T.Parallel(
                    logical_chunk, token_chunk
                ):
                    valid = (logical < valid_rows) and (
                        token < valid_tokens
                    )
                    br[logical, token] = T.if_then_else(
                        valid, grad_injection[panel, logical, token], 0.0
                    )
                    response_shared[logical, token] = T.if_then_else(
                        valid, response[panel, logical, token], 0.0
                    )

                grad_w = T.alloc_shared(
                    (logical_chunk, logical_chunk), in_dtype
                )
                pair_tile = T.alloc_fragment((16, 16), "float32")
                for row_block in T.serial(blocks):
                    for column_block in T.serial(blocks):
                        T.gemm(
                            by[row_block * 16 : (row_block + 1) * 16, 0:rank],
                            y[column_block * 16 : (column_block + 1) * 16, 0:rank],
                            pair_tile,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        T.gemm(
                            br[
                                row_block * 16 : (row_block + 1) * 16,
                                0:token_chunk,
                            ],
                            response_shared[
                                column_block * 16 : (column_block + 1) * 16,
                                0:token_chunk,
                            ],
                            pair_tile,
                            transpose_B=True,
                            clear_accum=False,
                        )
                        for row, column in T.Parallel(16, 16):
                            logical_row = row_block * 16 + row
                            logical_column = column_block * 16 + column
                            grad_w[logical_row, logical_column] = T.Cast(
                                in_dtype,
                                T.if_then_else(
                                    (logical_row < valid_rows)
                                    and (logical_column < logical_row),
                                    pair_tile[row, column],
                                    0.0,
                                ),
                            )

                # Keep only one 16-row source operand resident while applying
                # grad_w.  Reloading d/paired tiles costs less than keeping
                # both complete 48xr panels live and allows more CTAs/SM.
                # FLA's pair transpose still keeps the bounded operand tiles
                # in the source dtype with FP32 Tensor Core accumulation.
                grad_left = T.alloc_shared((logical_chunk, rank), in_dtype)
                grad_right = T.alloc_shared((logical_chunk, rank), in_dtype)
                source_operand = T.alloc_shared((16, rank), in_dtype)
                source_tile = T.alloc_fragment((16, rank), "float32")
                for row_block in T.serial(blocks):
                    for reduction_block in T.serial(blocks):
                        for row, c in T.Parallel(16, rank):
                            logical = reduction_block * 16 + row
                            token = logical // _E
                            slot = logical % _E
                            valid = logical < valid_rows
                            gate = T.if_then_else(
                                valid,
                                cumulative[
                                    i_b, token_bos + token, i_h, c
                                ],
                                0.0,
                            )
                            centered = gate - g_middle[c]
                            dv = T.Cast(
                                "float32",
                                T.if_then_else(
                                    valid,
                                    d[panel, token, slot, c],
                                    0.0,
                                ),
                            )
                            source_operand[row, c] = T.Cast(
                                in_dtype, dv * T.exp2(-centered)
                            )
                        T.sync_threads()
                        T.gemm(
                            grad_w[
                                row_block * 16 : (row_block + 1) * 16,
                                reduction_block * 16 : (reduction_block + 1) * 16,
                            ],
                            source_operand[0:16, 0:rank],
                            source_tile,
                            clear_accum=reduction_block == 0,
                        )
                        T.sync_threads()
                    for row, c in T.Parallel(16, rank):
                        grad_left[row_block * 16 + row, c] = T.Cast(
                            in_dtype, source_tile[row, c]
                        )
                    for reduction_block in T.serial(blocks):
                        for row, c in T.Parallel(16, rank):
                            logical = reduction_block * 16 + row
                            token = logical // _E
                            slot = logical % _E
                            valid = logical < valid_rows
                            gate = T.if_then_else(
                                valid,
                                cumulative[
                                    i_b, token_bos + token, i_h, c
                                ],
                                0.0,
                            )
                            centered = gate - g_middle[c]
                            ev = T.Cast(
                                "float32",
                                T.if_then_else(
                                    valid,
                                    paired[panel, token, slot, c],
                                    0.0,
                                ),
                            )
                            source_operand[row, c] = T.Cast(
                                in_dtype, -ev * T.exp2(centered)
                            )
                        T.sync_threads()
                        T.gemm(
                            grad_w[
                                reduction_block * 16 : (reduction_block + 1) * 16,
                                row_block * 16 : (row_block + 1) * 16,
                            ],
                            source_operand[0:16, 0:rank],
                            source_tile,
                            transpose_A=True,
                            clear_accum=reduction_block == 0,
                        )
                        T.sync_threads()
                    for row, c in T.Parallel(16, rank):
                        grad_right[row_block * 16 + row, c] = T.Cast(
                            in_dtype, source_tile[row, c]
                        )

                token_gate = T.alloc_shared(
                    (token_chunk, rank), "float32"
                )
                for t, c in T.Parallel(token_chunk, rank):
                    if t < valid_tokens:
                        gate = cumulative[i_b, token_bos + t, i_h, c]
                        centered = gate - g_middle[c]
                        local_gate = T.alloc_var(T.float32)
                        q_source = T.Cast(
                            "float32", paired[panel, t, _E, c]
                        )
                        local_gate = q_source * grad_q[panel, t, c]
                        if t == last:
                            local_gate = (
                                local_gate + grad_tail_seed[panel, c]
                            )
                        for slot in T.unroll(_E):
                            logical = t * _E + slot
                            dv = T.Cast("float32", d[panel, t, slot, c])
                            ev = T.Cast(
                                "float32", paired[panel, t, slot, c]
                            )
                            dl = T.Cast("float32", grad_left[logical, c])
                            dr = T.Cast("float32", grad_right[logical, c])
                            de = T.Cast("float32", grad_e[panel, logical, c])
                            dd = T.Cast(
                                "float32", grad_d[panel, logical, c]
                            )
                            e_center = -ev * T.exp2(centered)
                            d_center = dv * T.exp2(-centered)
                            e_global = ev * T.exp2(gate)
                            grad_left[logical, c] = T.Cast(
                                in_dtype,
                                -dl * T.exp2(centered) + de * T.exp2(gate),
                            )
                            grad_right[logical, c] = T.Cast(
                                in_dtype,
                                dr * T.exp2(-centered)
                                + dd,
                            )
                            local_gate = (
                                local_gate
                                + dl * e_center
                                + de * e_global
                                - dr * d_center
                                - dd * dv
                            )
                        token_gate[t, c] = local_gate
                    else:
                        token_gate[t, c] = 0.0

                T.sync_threads()
                for c in T.Parallel(rank):
                    accum = T.alloc_var(T.float32)
                    accum = 0.0
                    for reverse_t in T.serial(token_chunk):
                        t = token_chunk - 1 - reverse_t
                        if t < valid_tokens:
                            accum = accum + token_gate[t, c]
                            token_gate[t, c] = accum

                T.sync_threads()
                source_scalars = T.alloc_shared((token_chunk, 2), "float32")
                for t in T.Parallel(token_chunk):
                    if t < valid_tokens:
                        token = token_bos + t
                        denominator = T.alloc_var(T.float32)
                        denominator = 1.0
                        for c in T.serial(rank):
                            uv = T.Cast(
                                "float32", u[i_b, token, i_h, c]
                            )
                            gv = T.Cast(
                                "float32", gain[i_b, token, i_h, c]
                            )
                            denominator = denominator - uv * gv
                        geometry_decay = T.exp(
                            geometry_log_decay[i_b, token, i_h]
                        )
                        mass_previous = previous_mass[i_b, token, i_h]
                        mass_current = current_mass[i_b, token, i_h]
                        mass_scale = mass_previous / mass_current
                        gamma = strength[i_h]
                        diagonal_h = 1.0 + gamma * (
                            mass_scale * geometry_decay - 1.0
                        )

                        grad_d0_scale = T.alloc_var(T.float32)
                        grad_geometry_decay = T.alloc_var(T.float32)
                        grad_denominator = T.alloc_var(T.float32)
                        grad_gamma_direct = T.alloc_var(T.float32)
                        grad_gate_sum = T.alloc_var(T.float32)
                        grad_d0_scale = 0.0
                        grad_geometry_decay = 0.0
                        grad_denominator = 0.0
                        grad_gamma_direct = 0.0
                        grad_gate_sum = 0.0
                        for c in T.serial(rank):
                            uv = T.Cast(
                                "float32", u[i_b, token, i_h, c]
                            )
                            gv = T.Cast(
                                "float32", gain[i_b, token, i_h, c]
                            )
                            assoc = T.exp(
                                associative_log_decay[
                                    i_b, token, i_h, c
                                ]
                            )
                            gd0 = grad_right[t * _E, c]
                            ge0 = grad_left[t * _E, c]
                            gd1 = grad_right[t * _E + 1, c]
                            ge1 = grad_left[t * _E + 1, c]
                            gd2 = grad_right[t * _E + 2, c]
                            ge2 = grad_left[t * _E + 2, c]
                            grad_d0_scale = (
                                grad_d0_scale + gd0 * assoc * uv
                            )
                            grad_geometry_decay = (
                                grad_geometry_decay
                                - ge0 * gv / (denominator * assoc)
                            )
                            grad_denominator = (
                                grad_denominator
                                + ge0
                                * geometry_decay
                                * gv
                                / (denominator * denominator * assoc)
                            )
                            grad_gamma_direct = (
                                grad_gamma_direct + gd1 * gv
                            )
                            grad_gate_sum = (
                                grad_gate_sum + token_gate[t, c]
                            )

                        d0_scale = gamma * mass_scale / diagonal_h
                        grad_gamma = T.alloc_var(T.float32)
                        grad_mass_scale = T.alloc_var(T.float32)
                        grad_gamma = (
                            grad_d0_scale * mass_scale / diagonal_h
                            + grad_gamma_direct
                        )
                        grad_mass_scale = (
                            grad_d0_scale * gamma / diagonal_h
                        )
                        grad_diagonal_h = (
                            -grad_d0_scale * d0_scale / diagonal_h
                            + grad_gate_sum / diagonal_h
                        )
                        grad_gamma = grad_gamma + grad_diagonal_h * (
                            mass_scale * geometry_decay - 1.0
                        )
                        grad_mass_scale = (
                            grad_mass_scale
                            + grad_diagonal_h * gamma * geometry_decay
                        )
                        grad_geometry_decay = (
                            grad_geometry_decay
                            + grad_diagonal_h * gamma * mass_scale
                        )
                        grad_geometry_log_decay[i_b, token, i_h] = (
                            grad_geometry_decay * geometry_decay
                        )
                        grad_previous_mass[i_b, token, i_h] = (
                            grad_mass_scale / mass_current
                        )
                        grad_current_mass[i_b, token, i_h] = (
                            -grad_mass_scale
                            * mass_previous
                            / (mass_current * mass_current)
                        )
                        grad_strength_partial[i_b, token, i_h] = grad_gamma
                        source_scalars[t, 0] = grad_denominator
                        source_scalars[t, 1] = denominator

                T.sync_threads()
                for t, c in T.Parallel(token_chunk, rank):
                    if t < valid_tokens:
                        token = token_bos + t
                        uv = T.Cast("float32", u[i_b, token, i_h, c])
                        gv = T.Cast(
                            "float32", gain[i_b, token, i_h, c]
                        )
                        denom_vector = source_scalars[t, 1]
                        geometry_decay_vector = T.exp(
                            geometry_log_decay[i_b, token, i_h]
                        )
                        mass_scale_vector = (
                            previous_mass[i_b, token, i_h]
                            / current_mass[i_b, token, i_h]
                        )
                        gamma_vector = strength[i_h]
                        diagonal_h_vector = 1.0 + gamma_vector * (
                            mass_scale_vector * geometry_decay_vector - 1.0
                        )
                        assoc_vector = T.exp(
                            associative_log_decay[i_b, token, i_h, c]
                        )
                        d0_scale_vector = (
                            gamma_vector
                            * mass_scale_vector
                            / diagonal_h_vector
                        )
                        gd0 = grad_right[t * _E, c]
                        ge0 = grad_left[t * _E, c]
                        gd1 = grad_right[t * _E + 1, c]
                        ge1 = grad_left[t * _E + 1, c]
                        gd2 = grad_right[t * _E + 2, c]
                        ge2 = grad_left[t * _E + 2, c]
                        denominator_bar = source_scalars[t, 0]
                        grad_u[i_b, token, i_h, c] = (
                            gd0 * assoc_vector * d0_scale_vector
                            - denominator_bar * gv
                        )
                        grad_gain[i_b, token, i_h, c] = (
                            -ge0
                            * geometry_decay_vector
                            / (denom_vector * assoc_vector)
                            + gamma_vector * gd1
                            - denominator_bar * uv
                        )
                        grad_h[i_b, token, i_h, c] = -ge1 / denom_vector
                        grad_prediction[i_b, token, i_h, c] = ge1 / denom_vector
                        e0_vector = (
                            -geometry_decay_vector
                            * gv
                            / (denom_vector * assoc_vector)
                        )
                        d0_vector = assoc_vector * d0_scale_vector * uv
                        grad_associative_log_decay[
                            i_b, token, i_h, c
                        ] = (
                            gd0 * d0_vector
                            - ge0 * e0_vector
                            + token_gate[t, c]
                        )
                        # The strided source epilogue closes the raw gate and
                        # key-normalization VJPs from these final source grads.
                        # It also closes the e1 denominator term from raw h.
                        grad_e[panel, t * _E + 1, c] = ge1
                        grad_d[panel, t * _E + (_E - 1), c] = gd2
                        grad_e[panel, t * _E + (_E - 1), c] = ge2

        return pair_reverse

    return build()


@triton.jit
def _paired_l2norm_bwd_kernel(
    paired,
    d,
    u,
    h,
    gain,
    prediction,
    erase_raw,
    q_rstd,
    key_rstd,
    grad_q_normalized,
    grad_e,
    grad_d,
    grad_q_raw,
    grad_keys,
    grad_erase_raw,
    grad_u,
    grad_gain,
    SH_B: tl.constexpr,
    SH_T: tl.constexpr,
    SH_H: tl.constexpr,
    SE_B: tl.constexpr,
    SE_T: tl.constexpr,
    SE_H: tl.constexpr,
    TOTAL: tl.constexpr,
    LENGTH: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    NT: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    coord = tl.arange(0, BD)
    row_mask = row < TOTAL
    mask = row_mask[:, None] & (coord[None, :] < R)

    token_flat, head = row // H, row % H
    batch, token = token_flat // LENGTH, token_flat % LENGTH
    panel = (batch * H + head) * NT + token // C
    panel_row = token % C
    raw_offset = row[:, None] * R + coord[None, :]
    q_dy_offset = (panel[:, None] * C + panel_row[:, None]) * R + coord[None, :]
    q_y_offset = (
        (panel[:, None] * C * 4 + panel_row[:, None] * 4 + 3) * R
        + coord[None, :]
    )
    key_dy_offset = (
        (panel[:, None] * C * 3 + panel_row[:, None] * 3 + 2) * R
        + coord[None, :]
    )
    e1_dy_offset = (
        (panel[:, None] * C * 3 + panel_row[:, None] * 3 + 1) * R
        + coord[None, :]
    )

    q_scale = tl.load(q_rstd + row, mask=row_mask, other=0.0).to(tl.float32)
    key_scale = tl.load(key_rstd + row, mask=row_mask, other=0.0).to(tl.float32)
    # Match FLA l2norm_bwd: transpose the exact low-precision panel consumed
    # by forward, rather than reconstructing an unrounded FP32 normalization.
    q_y = tl.load(paired + q_y_offset, mask=mask, other=0.0).to(tl.float32)
    key_y = tl.load(d + key_dy_offset, mask=mask, other=0.0).to(tl.float32)
    q_dy = tl.load(grad_q_normalized + q_dy_offset, mask=mask, other=0.0).to(
        tl.float32
    )
    key_dy = tl.load(grad_d + key_dy_offset, mask=mask, other=0.0).to(
        tl.float32
    )
    erase_dy = tl.load(grad_e + key_dy_offset, mask=mask, other=0.0).to(
        tl.float32
    )
    e1_dy = tl.load(grad_e + e1_dy_offset, mask=mask, other=0.0).to(tl.float32)
    erase_offset = (
        batch[:, None] * SE_B
        + token[:, None] * SE_T
        + head[:, None] * SE_H
        + coord[None, :]
    )
    erase_logits = tl.load(erase_raw + erase_offset, mask=mask, other=0.0).to(
        tl.float32
    )
    erase_sigmoid = tl.sigmoid(erase_logits)
    erase_gate = (2.0 * erase_sigmoid).to(erase_raw.dtype.element_ty).to(
        tl.float32
    )
    key_dy += erase_gate * erase_dy

    packed_offset = raw_offset
    b_u = tl.load(u + packed_offset, mask=mask, other=0.0).to(tl.float32)
    b_gain = tl.load(gain + packed_offset, mask=mask, other=0.0).to(tl.float32)
    h_offset = (
        batch[:, None] * SH_B
        + token[:, None] * SH_T
        + head[:, None] * SH_H
        + coord[None, :]
    )
    b_h = tl.load(h + h_offset, mask=mask, other=0.0).to(tl.float32)
    b_prediction = tl.load(
        prediction + packed_offset, mask=mask, other=0.0
    ).to(tl.float32)
    denominator = 1.0 - tl.sum(b_u * b_gain, axis=1)
    denominator_bar = tl.sum(e1_dy * (b_h - b_prediction), axis=1)
    denominator_bar /= denominator * denominator
    b_grad_u = tl.load(grad_u + packed_offset, mask=mask, other=0.0).to(tl.float32)
    b_grad_gain = tl.load(grad_gain + packed_offset, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(
        grad_u + packed_offset,
        b_grad_u - denominator_bar[:, None] * b_gain,
        mask=mask,
    )
    tl.store(
        grad_gain + packed_offset,
        b_grad_gain - denominator_bar[:, None] * b_u,
        mask=mask,
    )

    q_dx = (q_dy - tl.sum(q_dy * q_y, axis=1)[:, None] * q_y) * q_scale[:, None]
    key_dx = (
        key_dy - tl.sum(key_dy * key_y, axis=1)[:, None] * key_y
    ) * key_scale[:, None]
    tl.store(grad_q_raw + raw_offset, q_dx, mask=mask)
    tl.store(grad_keys + raw_offset, key_dx, mask=mask)
    # Match the deleted two-kernel path: source grad rounded to BF16 before
    # applying the FP32 sigmoid derivative, then public grad rounded to BF16.
    erase_gate_grad = (key_y * erase_dy).to(erase_raw.dtype.element_ty).to(
        tl.float32
    )
    erase_raw_grad = erase_gate_grad * 2.0 * erase_sigmoid * (1.0 - erase_sigmoid)
    tl.store(grad_erase_raw + raw_offset, erase_raw_grad, mask=mask)


@triton.jit
def _value_source_reverse_kernel(
    values,
    write_raw,
    grad_z,
    grad_values,
    grad_write_raw,
    SV_B: tl.constexpr,
    SV_T: tl.constexpr,
    SV_H: tl.constexpr,
    SW_B: tl.constexpr,
    SW_T: tl.constexpr,
    SW_H: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    NT: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    token_head = tl.program_id(1).to(tl.int64)
    token_flat, head = token_head // H, token_head % H
    batch, token = token_flat // T, token_flat % T
    chunk, row = token // C, token % C
    panel = (batch * H + head) * NT + chunk
    o_v = i_v * BV + tl.arange(0, BV)
    mask = o_v < V
    raw_offset = (token_head * V) + o_v
    value_offset = batch * SV_B + token * SV_T + head * SV_H + o_v
    write_offset = batch * SW_B + token * SW_T + head * SW_H + o_v
    z_offset = ((panel * C + row) * V) + o_v
    b_value = tl.load(values + value_offset, mask=mask, other=0.0).to(tl.float32)
    b_write_logits = tl.load(write_raw + write_offset, mask=mask, other=0.0).to(
        tl.float32
    )
    b_write_sigmoid = tl.sigmoid(b_write_logits)
    b_write = (2.0 * b_write_sigmoid).to(write_raw.dtype.element_ty).to(
        tl.float32
    )
    b_dz = tl.load(grad_z + z_offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(grad_values + raw_offset, b_write * b_dz, mask=mask)
    b_gate_grad = (b_value * b_dz).to(write_raw.dtype.element_ty).to(tl.float32)
    b_raw_grad = b_gate_grad * 2.0 * b_write_sigmoid * (1.0 - b_write_sigmoid)
    tl.store(grad_write_raw + raw_offset, b_raw_grad, mask=mask)


def block_e3_fused_source_reverse(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    gain: torch.Tensor,
    prediction: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase_raw: torch.Tensor,
    write_raw: torch.Tensor,
    previous_mass: torch.Tensor,
    current_mass: torch.Tensor,
    strength: torch.Tensor,
    d: torch.Tensor,
    paired: torch.Tensor,
    q_rstd: torch.Tensor,
    key_rstd: torch.Tensor,
    cumulative: torch.Tensor,
    Y: torch.Tensor,
    response: torch.Tensor,
    grad_e: torch.Tensor,
    grad_injection: torch.Tensor,
    grad_d: torch.Tensor,
    grad_q: torch.Tensor,
    grad_tail_seed: torch.Tensor,
    grad_z: torch.Tensor,
    *,
    token_chunk_size: int,
) -> tuple[torch.Tensor, ...]:
    batch, _, heads, rank = cumulative.shape
    length = cumulative.shape[1]
    grad_u = torch.empty_like(u)
    grad_h = torch.empty(h.shape, dtype=h.dtype, device=h.device)
    grad_q_raw = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    grad_keys = torch.empty(keys.shape, dtype=keys.dtype, device=keys.device)
    grad_values = torch.empty(values.shape, dtype=values.dtype, device=values.device)
    grad_gain = torch.empty_like(gain)
    grad_prediction = torch.empty_like(prediction)
    grad_geometry_log_decay = torch.empty_like(geometry_log_decay)
    grad_associative_log_decay = torch.empty_like(associative_log_decay)
    grad_erase_raw = torch.empty(
        erase_raw.shape, dtype=erase_raw.dtype, device=erase_raw.device
    )
    grad_write_raw = torch.empty(
        write_raw.shape, dtype=write_raw.dtype, device=write_raw.device
    )
    grad_previous_mass = torch.empty_like(previous_mass)
    grad_current_mass = torch.empty_like(current_mass)
    grad_strength_partial = torch.empty_like(geometry_log_decay)
    kernel = _fused_pair_reverse_kernel(
        batch,
        length,
        heads,
        rank,
        token_chunk_size,
        _dtype_name(d.dtype),
    )
    kernel(
        u,
        gain,
        geometry_log_decay,
        associative_log_decay,
        previous_mass,
        current_mass,
        strength.reshape(-1),
        d,
        paired,
        cumulative,
        Y,
        response,
        grad_e,
        grad_injection,
        grad_d,
        grad_q,
        grad_tail_seed,
        grad_u,
        grad_h,
        grad_gain,
        grad_prediction,
        grad_geometry_log_decay,
        grad_associative_log_decay,
        grad_previous_mass,
        grad_current_mass,
        grad_strength_partial,
    )
    total_rows = batch * length * heads
    block_dim = max(16, triton.next_power_of_2(rank))
    _paired_l2norm_bwd_kernel[(triton.cdiv(total_rows, 16),)](
        paired,
        d,
        u,
        h,
        gain,
        prediction,
        erase_raw,
        q_rstd,
        key_rstd,
        grad_q,
        grad_e,
        grad_d,
        grad_q_raw,
        grad_keys,
        grad_erase_raw,
        grad_u,
        grad_gain,
        SH_B=h.stride(0),
        SH_T=h.stride(1),
        SH_H=h.stride(2),
        SE_B=erase_raw.stride(0),
        SE_T=erase_raw.stride(1),
        SE_H=erase_raw.stride(2),
        TOTAL=total_rows,
        LENGTH=length,
        H=heads,
        R=rank,
        C=token_chunk_size,
        NT=triton.cdiv(length, token_chunk_size),
        BT=16,
        BD=block_dim,
        num_warps=4,
        num_stages=1,
    )
    value_dim = values.shape[-1]
    chunks = triton.cdiv(length, token_chunk_size)
    _value_source_reverse_kernel[
        (triton.cdiv(value_dim, 64), batch * length * heads)
    ](
        values,
        write_raw,
        grad_z,
        grad_values,
        grad_write_raw,
        SV_B=values.stride(0),
        SV_T=values.stride(1),
        SV_H=values.stride(2),
        SW_B=write_raw.stride(0),
        SW_T=write_raw.stride(1),
        SW_H=write_raw.stride(2),
        T=length,
        H=heads,
        V=value_dim,
        C=token_chunk_size,
        NT=chunks,
        BV=64,
        num_warps=4,
        num_stages=1,
    )
    grad_strength = grad_strength_partial.sum(dim=(0, 1)).reshape(
        strength.shape
    )
    return (
        grad_u,
        grad_h,
        grad_q_raw,
        grad_keys,
        grad_values,
        grad_gain,
        grad_prediction,
        grad_geometry_log_decay,
        grad_associative_log_decay,
        grad_erase_raw,
        grad_write_raw,
        grad_previous_mass,
        grad_current_mass,
        grad_strength,
    )


__all__ = ["block_e3_fused_source_reverse"]
