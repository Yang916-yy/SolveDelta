# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
"""Connected native token-block E=3 exterior and strict custom transpose."""

from __future__ import annotations

import torch

from .block_e3_pair import block_e3_pair_forward, block_e3_recompute_query_gauge
from .block_e3_pair_reverse import block_e3_fused_source_reverse
from .block_e3_reverse import block_e3_mature_reverse
from .block_e3_sources import block_e3_gate_cumsum, block_e3_sources_forward
from .block_e3_state import (
    block_e3_action_statistics,
    block_e3_state_forward,
)
from .block_e3_wy import block_e3_wy_forward


def _forward_blocks(
    d,
    paired,
    z,
    associative_log_decay,
    diagonal_log,
    initial_state,
    *,
    token_chunk_size,
):
    cumulative = block_e3_gate_cumsum(
        associative_log_decay,
        diagonal_log,
        token_chunk_size=token_chunk_size,
    )
    W, A, e_global, d_tail, q_global = block_e3_pair_forward(
        d, paired, cumulative, token_chunk_size=token_chunk_size
    )
    Y, response, inverse = block_e3_wy_forward(
        W, e_global, token_chunk_size=token_chunk_size
    )
    q_star, b_z, k_z = block_e3_action_statistics(
        A, Y, response, q_global, d_tail
    )
    output, final_state, state_cache = block_e3_state_forward(
        Y,
        d_tail,
        q_star,
        b_z,
        k_z,
        z,
        cumulative,
        initial_state,
        token_chunk_size=token_chunk_size,
    )
    return output, final_state, (
        cumulative,
        A,
        d_tail,
        Y,
        response,
        inverse,
        state_cache,
    )


class _BlockE3DirectE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u,
        h,
        q,
        keys,
        values,
        gain,
        prediction,
        geometry_log_decay,
        associative_log_decay,
        erase_raw,
        write_raw,
        previous_mass,
        current_mass,
        strength,
        initial_state,
        token_chunk_size,
    ):
        d, paired, z, diagonal_log, q_rstd, key_rstd = block_e3_sources_forward(
            u,
            h,
            q,
            keys,
            values,
            gain,
            prediction,
            geometry_log_decay,
            associative_log_decay,
            erase_raw,
            write_raw,
            previous_mass,
            current_mass,
            strength,
            token_chunk_size=token_chunk_size,
        )
        output, final_state, cache = _forward_blocks(
            d,
            paired,
            z,
            associative_log_decay,
            diagonal_log,
            initial_state,
            token_chunk_size=token_chunk_size,
        )
        ctx.token_chunk_size = token_chunk_size
        ctx.strength_shape = strength.shape
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            u,
            h,
            q,
            keys,
            values,
            gain,
            prediction,
            geometry_log_decay,
            associative_log_decay,
            erase_raw,
            write_raw,
            previous_mass,
            current_mass,
            strength,
            d,
            paired,
            z,
            q_rstd,
            key_rstd,
            initial_state,
            *cache,
        )
        return output, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        (
            u,
            h,
            q,
            keys,
            values,
            gain,
            prediction,
            geometry_log_decay,
            associative_log_decay,
            erase_raw,
            write_raw,
            previous_mass,
            current_mass,
            strength,
            d,
            paired,
            z,
            q_rstd,
            key_rstd,
            initial_state,
            cumulative,
            A,
            d_tail,
            Y,
            response,
            inverse,
            state_cache,
        ) = ctx.saved_tensors
        if grad_output is None:
            grad_output = torch.zeros(
                q.shape[0],
                q.shape[1],
                q.shape[2],
                values.shape[-1],
                dtype=values.dtype,
                device=values.device,
            )
        else:
            grad_output = grad_output.contiguous()
        q_global = block_e3_recompute_query_gauge(
            paired,
            cumulative,
            token_chunk_size=ctx.token_chunk_size,
        )
        (
            grad_e,
            grad_injection,
            grad_d,
            grad_q,
            grad_tail_seed,
            grad_z,
            grad_initial_state,
        ) = block_e3_mature_reverse(
            d,
            paired,
            z,
            cumulative,
            A,
            d_tail,
            Y,
            response,
            inverse,
            q_global,
            state_cache,
            grad_output,
            grad_final_state,
            initial_state,
            token_chunk_size=ctx.token_chunk_size,
        )
        source_grads = block_e3_fused_source_reverse(
            u,
            h,
            q,
            keys,
            values,
            gain,
            prediction,
            geometry_log_decay,
            associative_log_decay,
            erase_raw,
            write_raw,
            previous_mass,
            current_mass,
            strength,
            d,
            paired,
            q_rstd,
            key_rstd,
            cumulative,
            Y,
            response,
            grad_e,
            grad_injection,
            grad_d,
            grad_q,
            grad_tail_seed,
            grad_z,
            token_chunk_size=ctx.token_chunk_size,
        )
        return (*source_grads, grad_initial_state, None)


def block_e3_direct_e_delta_rule(
    u,
    h,
    q,
    keys,
    values,
    gain,
    prediction,
    geometry_log_decay,
    associative_log_decay,
    erase_raw,
    write_raw,
    previous_mass,
    current_mass,
    strength,
    initial_state,
    *,
    token_chunk_size: int = 16,
):
    return _BlockE3DirectE.apply(
        u,
        h,
        q,
        keys,
        values,
        gain,
        prediction,
        geometry_log_decay,
        associative_log_decay,
        erase_raw,
        write_raw,
        previous_mass,
        current_mass,
        strength,
        initial_state,
        token_chunk_size,
    )


__all__ = ["block_e3_direct_e_delta_rule"]
