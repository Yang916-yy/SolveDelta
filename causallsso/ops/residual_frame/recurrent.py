# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
"""Inference-only recurrent owners for one Residual-Frame token.

The state ownership follows FLA's fused recurrent Oja and DPLR kernels.  The
predictor owner changes the order to the SolveDelta contract: compute the
innovation from the unforgotten state, then retain the old state and add the
current rank-one update.  The memory transition is delegated unchanged to
FLA's mature DPLR recurrent owner.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.generalized_delta_rule.dplr.fused_recurrent import (
    fused_recurrent_dplr_delta_rule_fwd,
)

from ...reference import RELATIVE_FRAME_RADIUS, SolveDeltaState


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["initial"] is not None,
        "STORE_FINAL_STATE": lambda args: args["final"] is not None,
    }
)
@triton.jit
def _predictor_recurrent_kernel(
    u,
    target,
    key,
    gamma,
    gate,
    initial,
    final,
    u_norm,
    key_norm,
    update,
    action,
    H: tl.constexpr,
    R: tl.constexpr,
    BR: tl.constexpr,
    SU_B: tl.constexpr,
    SU_H: tl.constexpr,
    ST_B: tl.constexpr,
    ST_H: tl.constexpr,
    SK_B: tl.constexpr,
    SK_H: tl.constexpr,
    SG_B: tl.constexpr,
    SG_H: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    batch_head = tl.program_id(0).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    row = tl.arange(0, BR)
    column = tl.arange(0, BR)
    valid = row < R
    matrix_mask = valid[:, None] & valid[None, :]

    u_offset = batch * SU_B + head * SU_H + row
    target_offset = batch * ST_B + head * ST_H + row
    key_offset = batch * SK_B + head * SK_H + row
    b_u = tl.load(u + u_offset, mask=valid, other=0.0).to(tl.float32)
    b_target = tl.load(target + target_offset, mask=valid, other=0.0).to(
        tl.float32
    )
    b_key = tl.load(key + key_offset, mask=valid, other=0.0).to(tl.float32)
    b_u *= tl.rsqrt(tl.sum(b_u * b_u) + 1.0e-24)
    b_key *= tl.rsqrt(tl.sum(b_key * b_key) + 1.0e-24)

    state_offset = batch_head * R * R + row[:, None] * R + column[None, :]
    b_state = tl.zeros((BR, BR), tl.float32)
    if USE_INITIAL_STATE:
        b_state += tl.load(initial + state_offset, mask=matrix_mask, other=0.0)

    prediction = tl.sum(b_state * b_u[None, :], axis=1)
    b_gamma = tl.load(gamma + batch * SG_B + head * SG_H).to(tl.float32)
    b_update = b_gamma * (b_target - prediction)
    gate_offset = (batch * H + head) * R + column
    b_gate = tl.load(gate + gate_offset, mask=valid, other=0.0).to(tl.float32)
    b_state *= tl.exp(b_gate[None, :])
    b_state += b_update[:, None] * b_u[None, :]
    b_action = tl.sum(b_state * b_key[:, None], axis=0)

    output_offset = batch_head * R + row
    tl.store(u_norm + output_offset, b_u, mask=valid)
    tl.store(key_norm + output_offset, b_key, mask=valid)
    tl.store(update + output_offset, b_update, mask=valid)
    tl.store(action + output_offset, b_action, mask=valid)
    if STORE_FINAL_STATE:
        tl.store(final + state_offset, b_state, mask=matrix_mask)


@triton.jit
def _relative_recurrent_sources_kernel(
    u,
    update,
    q,
    key,
    action,
    value,
    erase_raw,
    write_raw,
    log_decay,
    direct,
    erase_source,
    query,
    injection,
    H: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
    SQ_B: tl.constexpr,
    SQ_H: tl.constexpr,
    SV_B: tl.constexpr,
    SV_H: tl.constexpr,
    SE_B: tl.constexpr,
    SE_H: tl.constexpr,
    SW_B: tl.constexpr,
    SW_H: tl.constexpr,
    SD_B: tl.constexpr,
    SD_H: tl.constexpr,
    FRAME_RADIUS: tl.constexpr,
):
    batch_head = tl.program_id(0).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    r = tl.arange(0, BR)
    valid_r = r < R
    packed_r = batch_head * R + r
    b_u = tl.load(u + packed_r, mask=valid_r, other=0.0).to(tl.float32)
    b_update = tl.load(update + packed_r, mask=valid_r, other=0.0).to(
        tl.float32
    )
    b_key = tl.load(key + packed_r, mask=valid_r, other=0.0).to(tl.float32)
    b_action = tl.load(action + packed_r, mask=valid_r, other=0.0).to(
        tl.float32
    )
    q_offset = batch * SQ_B + head * SQ_H + r
    b_q = tl.load(q + q_offset, mask=valid_r, other=0.0).to(tl.float32)
    b_q *= tl.rsqrt(tl.sum(b_q * b_q) + 1.0e-24)
    erase_offset = batch * SE_B + head * SE_H + r
    b_erase = tl.sigmoid(
        tl.load(erase_raw + erase_offset, mask=valid_r, other=0.0).to(tl.float32)
    )
    erase_key = b_erase * b_key

    scale = FRAME_RADIUS * tl.rsqrt(
        FRAME_RADIUS * FRAME_RADIUS + tl.sum(b_update * b_update)
    )
    frame = scale * b_update
    denominator = 1.0 + tl.sum(b_u * frame)
    dual = erase_key - frame * (tl.sum(b_u * erase_key) / denominator)
    q_dual = b_q - frame * (tl.sum(b_u * b_q) / denominator)
    b_direct = b_key + b_action
    decay_offset = batch * SD_B + head * SD_H + r
    b_decay = tl.load(log_decay + decay_offset, mask=valid_r, other=0.0).to(
        tl.float32
    )
    tl.store(direct + packed_r, b_direct, mask=valid_r)
    tl.store(erase_source + packed_r, -dual * tl.exp(b_decay), mask=valid_r)
    tl.store(query + packed_r, q_dual, mask=valid_r)

    v = tl.arange(0, BV)
    valid_v = v < V
    packed_v = batch_head * V + v
    value_offset = batch * SV_B + head * SV_H + v
    write_offset = batch * SW_B + head * SW_H + v
    b_value = tl.load(value + value_offset, mask=valid_v, other=0.0).to(
        tl.float32
    )
    b_write = tl.sigmoid(
        tl.load(write_raw + write_offset, mask=valid_v, other=0.0).to(
            tl.float32
        )
    )
    tl.store(injection + packed_v, b_write * b_value, mask=valid_v)


def solvedelta_recurrent_inference(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase_raw: torch.Tensor,
    write_raw: torch.Tensor,
    geometry_write: torch.Tensor,
    *,
    initial_state: SolveDeltaState | None,
    return_final_state: bool,
) -> tuple[torch.Tensor, SolveDeltaState | None]:
    """Execute exactly one token with recurrent inference owners."""
    batch, length, heads, rank = u.shape
    if length != 1:
        raise ValueError("recurrent inference owner requires T=1")
    value_dim = values.shape[-1]
    if rank > 128 or value_dim > 128:
        raise ValueError("recurrent inference owner supports widths up to 128")
    if initial_state is not None:
        expected = (
            (batch, heads, rank, rank),
            (batch, heads, rank, value_dim),
        )
        if any(value.shape != shape for value, shape in zip(initial_state, expected)):
            raise ValueError("initial_state shapes do not match recurrent geometry")
        if any(
            value.dtype != torch.float32 or value.device != u.device
            for value in initial_state
        ):
            raise TypeError("recurrent continuation states must be FP32 on input device")
    predictor_initial = None if initial_state is None else initial_state.predictor
    memory_initial = None if initial_state is None else initial_state.S

    if geometry_write.shape != (batch, 1, heads):
        geometry_write = geometry_write.reshape(1, 1, heads).expand(
            batch, 1, heads
        )
    frame_gate = associative_log_decay.contiguous()
    vector_shape = (batch, 1, heads, rank)
    u_norm = torch.empty(vector_shape, dtype=u.dtype, device=u.device)
    key_norm = torch.empty_like(u_norm)
    update = torch.empty_like(u_norm)
    action = torch.empty_like(u_norm)
    final_predictor = (
        torch.empty(
            batch, heads, rank, rank, dtype=torch.float32, device=u.device
        )
        if return_final_state
        else None
    )
    block_rank = triton.next_power_of_2(rank)
    _predictor_recurrent_kernel[(batch * heads,)](
        u,
        h,
        keys.squeeze(-2),
        geometry_write,
        frame_gate,
        predictor_initial,
        final_predictor,
        u_norm,
        key_norm,
        update,
        action,
        H=heads,
        R=rank,
        BR=block_rank,
        SU_B=u.stride(0),
        SU_H=u.stride(2),
        ST_B=h.stride(0),
        ST_H=h.stride(2),
        SK_B=keys.stride(0),
        SK_H=keys.stride(2),
        SG_B=geometry_write.stride(0),
        SG_H=geometry_write.stride(2),
        num_warps=8,
        num_stages=2,
    )

    direct = torch.empty_like(u_norm)
    # The current-token exp(log_decay) factor is unbounded below. Keep its
    # product with the dual covector in FP32 until the recurrent DPLR owner.
    erase_source = torch.empty_like(u_norm, dtype=torch.float32)
    query = torch.empty_like(u_norm)
    injection = torch.empty(
        batch, 1, heads, value_dim, dtype=values.dtype, device=values.device
    )
    _relative_recurrent_sources_kernel[(batch * heads,)](
        u_norm,
        update,
        q,
        key_norm,
        action,
        values.squeeze(-2),
        erase_raw.squeeze(-2),
        write_raw.squeeze(-2),
        associative_log_decay,
        direct,
        erase_source,
        query,
        injection,
        H=heads,
        R=rank,
        V=value_dim,
        BR=block_rank,
        BV=triton.next_power_of_2(value_dim),
        SQ_B=q.stride(0),
        SQ_H=q.stride(2),
        SV_B=values.stride(0),
        SV_H=values.stride(2),
        SE_B=erase_raw.stride(0),
        SE_H=erase_raw.stride(2),
        SW_B=write_raw.stride(0),
        SW_H=write_raw.stride(2),
        SD_B=associative_log_decay.stride(0),
        SD_H=associative_log_decay.stride(2),
        FRAME_RADIUS=RELATIVE_FRAME_RADIUS,
        num_warps=4,
        num_stages=2,
    )

    output, final_memory = fused_recurrent_dplr_delta_rule_fwd(
        q=query,
        k=direct,
        v=injection,
        a=erase_source,
        b=direct,
        gk=associative_log_decay,
        scale=1.0,
        initial_state=memory_initial,
        output_final_state=return_final_state,
    )
    output = output.to(torch.bfloat16)
    if not return_final_state:
        return output, None
    if final_predictor is None or final_memory is None:
        raise RuntimeError("recurrent owners did not return requested state")
    return output, SolveDeltaState(final_predictor.float(), final_memory.float())


__all__ = ["solvedelta_recurrent_inference"]
