# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
#
# State/output ownership is specialized from FLA generalized-DPLR chunk_h/
# chunk_o and GDN2's fused WY reverse. The source axes remain [token, slot].
"""Mature panel-owned transpose for the compact token-block E=3 exterior."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp2


_E = 3
_TRITON_E = tl.constexpr(3)


@triton.jit
def _state_reverse_owner(
    A,
    Y,
    response,
    q_global,
    d_tail,
    z,
    cumulative,
    state_cache,
    grad_output,
    grad_final_state,
    grad_state,
    grad_residual,
    residual,
    grad_z,
    grad_initial_state,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    NT: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    o_r = tl.arange(0, BR)
    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, C)
    o_n = tl.arange(0, BN)
    m_r = o_r < R
    m_v = o_v < V
    m_rv = m_r[:, None] & m_v[None, :]

    state_offset = i_bh * R * V + o_r[:, None] * V + o_v[None, :]
    dstate = tl.load(
        grad_final_state + state_offset, mask=m_rv, other=0.0
    ).to(tl.float32)

    for reverse_chunk in range(0, NT):
        chunk = NT - 1 - reverse_chunk
        panel = i_bh * NT + chunk
        token_bos = chunk * C
        valid = tl.minimum(C, T - token_bos)
        m_c = o_c < valid
        m_n = o_n < valid * _TRITON_E

        panel_state = panel * R * V + o_r[:, None] * V + o_v[None, :]
        state = tl.load(
            state_cache + panel_state, mask=m_rv, other=0.0
        ).to(tl.float32)
        tl.store(grad_state + panel_state, dstate, mask=m_rv)

        do_offset = (
            ((i_b * T + token_bos + o_c[:, None]) * H + i_h) * V
            + o_v[None, :]
        )
        b_do = tl.load(
            grad_output + do_offset,
            mask=m_c[:, None] & m_v[None, :],
            other=0.0,
        )
        z_offset = (panel * C + o_c[:, None]) * V + o_v[None, :]
        b_z = tl.load(
            z + z_offset,
            mask=m_c[:, None] & m_v[None, :],
            other=0.0,
        )

        a_offset = (panel * C + o_c[:, None]) * N + o_n[None, :]
        b_a = tl.load(
            A + a_offset,
            mask=m_c[:, None] & m_n[None, :],
            other=0.0,
        )
        d_offset = (panel * N + o_n[:, None]) * R + o_r[None, :]
        b_d = tl.load(
            d_tail + d_offset,
            mask=m_n[:, None] & m_r[None, :],
            other=0.0,
        )
        b_du = tl.dot(tl.trans(b_a), b_do)
        b_du += tl.dot(b_d, dstate.to(b_d.dtype))

        response_offset = (panel * N + o_n[:, None]) * C + o_c[None, :]
        b_response = tl.load(
            response + response_offset,
            mask=m_n[:, None] & m_c[None, :],
            other=0.0,
        )
        y_offset = (panel * N + o_n[:, None]) * R + o_r[None, :]
        b_y = tl.load(
            Y + y_offset,
            mask=m_n[:, None] & m_r[None, :],
            other=0.0,
        )
        b_residual = tl.dot(b_response, b_z)
        b_residual -= tl.dot(b_y, state.to(b_y.dtype))

        logical_v = (panel * N + o_n[:, None]) * V + o_v[None, :]
        m_nv = m_n[:, None] & m_v[None, :]
        tl.store(grad_residual + logical_v, b_du, mask=m_nv)
        tl.store(residual + logical_v, b_residual, mask=m_nv)

        b_dz = tl.dot(tl.trans(b_response), b_du.to(b_response.dtype))
        tl.store(grad_z + z_offset, b_dz, mask=m_c[:, None] & m_v[None, :])

        last_token = token_bos + tl.maximum(valid - 1, 0)
        gate_offset = ((i_b * T + last_token) * H + i_h) * R + o_r
        gate = tl.load(
            cumulative + gate_offset,
            mask=(valid > 0) & m_r,
            other=-float("inf"),
        ).to(tl.float32)
        q_offset = (panel * C + o_c[:, None]) * R + o_r[None, :]
        b_q = tl.load(
            q_global + q_offset,
            mask=m_c[:, None] & m_r[None, :],
            other=0.0,
        )
        dstate *= exp2(gate)[:, None]
        dstate += tl.dot(tl.trans(b_q), b_do.to(b_q.dtype))
        dstate -= tl.dot(tl.trans(b_y), b_du.to(b_y.dtype))

    tl.store(grad_initial_state + state_offset, dstate, mask=m_rv)


@triton.jit
def _output_wy_reverse_owner(
    d,
    paired,
    Y,
    response,
    inverse,
    cumulative,
    state_cache,
    grad_output,
    grad_state,
    grad_residual,
    residual,
    z,
    grad_e,
    grad_injection,
    grad_d,
    grad_q,
    grad_tail_seed,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    NT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_r = tl.program_id(0)
    panel = tl.program_id(1).to(tl.int64)
    i_bh, chunk = panel // NT, panel % NT
    i_b, i_h = i_bh // H, i_bh % H
    token_bos = chunk * C
    valid = tl.minimum(C, T - token_bos)
    o_c = tl.arange(0, C)
    o_n = tl.arange(0, BN)
    o_r = i_r * BK + tl.arange(0, BK)
    o_k = tl.arange(0, BN)
    m_c = o_c < valid
    m_n = o_n < valid * _TRITON_E
    m_r = o_r < R

    b_dq_state = tl.zeros((C, BK), dtype=tl.float32)
    b_score = tl.zeros((C, BN), dtype=tl.float32)
    b_dd_tail = tl.zeros((BN, BK), dtype=tl.float32)
    b_dy = tl.zeros((BN, BK), dtype=tl.float32)
    b_dr = tl.zeros((BN, C), dtype=tl.float32)
    b_lambda = tl.zeros((BK,), dtype=tl.float32)

    for i_v in range(0, tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        do_offset = (
            ((i_b * T + token_bos + o_c[:, None]) * H + i_h) * V
            + o_v[None, :]
        )
        b_do = tl.load(
            grad_output + do_offset,
            mask=m_c[:, None] & m_v[None, :],
            other=0.0,
        )
        state_offset = (panel * R + o_r[:, None]) * V + o_v[None, :]
        b_state = tl.load(
            state_cache + state_offset,
            mask=m_r[:, None] & m_v[None, :],
            other=0.0,
        )
        b_dstate = tl.load(
            grad_state + state_offset,
            mask=m_r[:, None] & m_v[None, :],
            other=0.0,
        )
        logical_v = (panel * N + o_n[:, None]) * V + o_v[None, :]
        b_du = tl.load(
            grad_residual + logical_v,
            mask=m_n[:, None] & m_v[None, :],
            other=0.0,
        )
        b_residual = tl.load(
            residual + logical_v,
            mask=m_n[:, None] & m_v[None, :],
            other=0.0,
        )
        z_offset = (panel * C + o_c[:, None]) * V + o_v[None, :]
        b_z = tl.load(
            z + z_offset,
            mask=m_c[:, None] & m_v[None, :],
            other=0.0,
        )
        b_dq_state += tl.dot(b_do, tl.trans(b_state).to(b_do.dtype))
        b_score += tl.dot(b_do, tl.trans(b_residual).to(b_do.dtype))
        b_dd_tail += tl.dot(
            b_residual, tl.trans(b_dstate).to(b_residual.dtype)
        )
        b_dy -= tl.dot(b_du.to(b_state.dtype), tl.trans(b_state))
        b_lambda += tl.sum((b_state * b_dstate).to(tl.float32), axis=1)
        if i_r == 0:
            b_dr += tl.dot(b_du, tl.trans(b_z).to(b_du.dtype))

    causal = o_n[None, :] < (o_c[:, None] + 1) * _TRITON_E
    b_score = tl.where(m_c[:, None] & m_n[None, :] & causal, b_score, 0.0)

    token = o_n // _TRITON_E
    slot = o_n % _TRITON_E
    d_offset = (
        ((panel * C + token[:, None]) * _TRITON_E + slot[:, None]) * R
        + o_r[None, :]
    )
    b_d = tl.load(
        d + d_offset,
        mask=m_n[:, None] & m_r[None, :],
        other=0.0,
    )
    q_offset = (
        ((panel * C + o_c[:, None]) * (_TRITON_E + 1) + _TRITON_E) * R
        + o_r[None, :]
    )
    b_q = tl.load(
        paired + q_offset,
        mask=m_c[:, None] & m_r[None, :],
        other=0.0,
    )
    gate_offset = (
        ((i_b * T + token_bos + o_c[:, None]) * H + i_h) * R
        + o_r[None, :]
    )
    b_gate = tl.load(
        cumulative + gate_offset,
        mask=m_c[:, None] & m_r[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    middle_token = token_bos + tl.maximum(valid // 2, 0)
    middle_offset = ((i_b * T + middle_token) * H + i_h) * R + o_r
    b_middle = tl.load(
        cumulative + middle_offset,
        mask=(valid > 0) & m_r,
        other=0.0,
    ).to(tl.float32)
    d_gate_offset = (
        ((i_b * T + token_bos + token[:, None]) * H + i_h) * R
        + o_r[None, :]
    )
    b_d_gate = tl.load(
        cumulative + d_gate_offset,
        mask=m_n[:, None] & m_r[None, :],
        other=0.0,
    ).to(tl.float32)
    b_q_scale = exp2(b_gate - b_middle[None, :])
    b_d_scale = exp2(b_middle[None, :] - b_d_gate)
    b_q_center = (b_q.to(tl.float32) * b_q_scale).to(b_q.dtype)
    b_d_center = (b_d.to(tl.float32) * b_d_scale).to(b_d.dtype)
    b_dq_pair = tl.dot(b_score.to(b_d.dtype), b_d_center)
    b_dq_pair *= b_q_scale
    b_dd_pair = tl.dot(tl.trans(b_score).to(b_q.dtype), b_q_center)
    b_dd_pair *= b_d_scale

    b_q_state_raw = b_dq_state * exp2(b_gate)
    b_dq = b_q_state_raw + b_dq_pair

    last_token = token_bos + tl.maximum(valid - 1, 0)
    last_gate_offset = ((i_b * T + last_token) * H + i_h) * R + o_r
    b_last_gate = tl.load(
        cumulative + last_gate_offset,
        mask=(valid > 0) & m_r,
        other=0.0,
    ).to(tl.float32)
    b_tail_scale = exp2(b_last_gate[None, :] - b_d_gate)
    b_tail_rows = b_dd_tail * b_d * b_tail_scale
    b_tail_seed = tl.sum(b_tail_rows, axis=0)
    b_tail_seed += b_lambda * exp2(b_last_gate)
    b_d_raw = b_dd_pair + b_dd_tail * b_tail_scale

    inv_offset = (panel * N + o_n[:, None]) * N + o_k[None, :]
    b_inv = tl.load(
        inverse + inv_offset,
        mask=m_n[:, None] & (o_k[None, :] < valid * _TRITON_E),
        other=0.0,
    )
    b_de = tl.dot(
        tl.trans(b_inv).to(b_d.dtype),
        b_dy.to(b_d.dtype),
    )

    out_offset = (panel * N + o_n[:, None]) * R + o_r[None, :]
    tl.store(grad_e + out_offset, b_de, mask=m_n[:, None] & m_r[None, :])
    tl.store(
        grad_d + out_offset,
        b_d_raw,
        mask=m_n[:, None] & m_r[None, :],
    )
    q_out = (panel * C + o_c[:, None]) * R + o_r[None, :]
    tl.store(grad_q + q_out, b_dq, mask=m_c[:, None] & m_r[None, :])
    tl.store(
        grad_tail_seed + panel * R + o_r,
        b_tail_seed,
        mask=m_r,
    )

    if i_r == 0:
        b_dresponse = tl.dot(
            tl.trans(b_inv).to(b_d.dtype),
            b_dr.to(b_d.dtype),
        )
        response_out = (panel * N + o_n[:, None]) * C + o_c[None, :]
        tl.store(
            grad_injection + response_out,
            b_dresponse,
            mask=m_n[:, None] & m_c[None, :],
        )


def block_e3_mature_reverse(
    d: torch.Tensor,
    paired: torch.Tensor,
    z: torch.Tensor,
    cumulative: torch.Tensor,
    A: torch.Tensor,
    d_tail: torch.Tensor,
    Y: torch.Tensor,
    response: torch.Tensor,
    inverse: torch.Tensor,
    q_global: torch.Tensor,
    state_cache: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final_state: torch.Tensor | None,
    initial_state: torch.Tensor,
    *,
    token_chunk_size: int,
) -> tuple[torch.Tensor, ...]:
    batch, length, heads, rank = cumulative.shape
    value_dim = grad_output.shape[-1]
    chunks = triton.cdiv(length, token_chunk_size)
    panels = batch * heads * chunks
    logical_chunk = _E * token_chunk_size
    if token_chunk_size != 16:
        raise ValueError("the mature block-E3 reverse is specialized for C16")
    if grad_final_state is None:
        grad_final_state = torch.zeros_like(initial_state, dtype=torch.float32)
    else:
        grad_final_state = grad_final_state.float().contiguous()

    grad_state = torch.empty_like(state_cache)
    grad_residual = torch.empty(
        panels,
        logical_chunk,
        value_dim,
        dtype=torch.float32,
        device=Y.device,
    )
    residual = torch.empty(
        panels,
        logical_chunk,
        value_dim,
        dtype=Y.dtype,
        device=Y.device,
    )
    grad_z = torch.empty_like(z, dtype=torch.float32)
    grad_initial_state = torch.empty_like(initial_state, dtype=torch.float32)
    block_rank = max(16, triton.next_power_of_2(rank))
    _state_reverse_owner[(triton.cdiv(value_dim, 16), batch * heads)](
        A,
        Y,
        response,
        q_global,
        d_tail,
        z[:, :, 0],
        cumulative,
        state_cache,
        grad_output,
        grad_final_state,
        grad_state,
        grad_residual,
        residual,
        grad_z[:, :, 0],
        grad_initial_state,
        T=length,
        H=heads,
        R=rank,
        V=value_dim,
        C=token_chunk_size,
        N=logical_chunk,
        NT=chunks,
        BN=64,
        BR=block_rank,
        BV=16,
        num_warps=4,
        num_stages=2,
    )

    # These are FP32 backward partials.  The following pair/source owner
    # consumes them directly; no public-panel rounding boundary exists here.
    grad_e = torch.empty_like(Y, dtype=torch.float32)
    grad_injection = torch.empty_like(response, dtype=torch.float32)
    grad_d = torch.empty_like(Y, dtype=torch.float32)
    grad_q = torch.empty_like(q_global, dtype=torch.float32)
    grad_tail_seed = torch.empty(
        panels, rank, dtype=torch.float32, device=Y.device
    )
    _output_wy_reverse_owner[(triton.cdiv(rank, 32), panels)](
        d,
        paired,
        Y,
        response,
        inverse,
        cumulative,
        state_cache,
        grad_output,
        grad_state,
        grad_residual,
        residual,
        z[:, :, 0],
        grad_e,
        grad_injection,
        grad_d,
        grad_q,
        grad_tail_seed,
        T=length,
        H=heads,
        R=rank,
        V=value_dim,
        C=token_chunk_size,
        N=logical_chunk,
        NT=chunks,
        BN=64,
        BK=32,
        BV=32,
        num_warps=4,
        num_stages=2,
    )
    return (
        grad_e,
        grad_injection,
        grad_d,
        grad_q,
        grad_tail_seed,
        grad_z,
        grad_initial_state,
    )


__all__ = ["block_e3_mature_reverse"]
