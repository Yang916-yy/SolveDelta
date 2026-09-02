# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
# Specialized from FLA's MIT-licensed gated-Oja pair/WY/state owners.

from __future__ import annotations

import torch

from fla.ops.gated_oja_rule.chunk_h import (
    chunk_oja_bwd_dhu,
    chunk_oja_bwd_dvwg_h,
    chunk_oja_fwd_h,
)
from fla.ops.gated_oja_rule.chunk_o import (
    chunk_oja_bwd_dA,
    chunk_oja_bwd_dqk,
    chunk_oja_bwd_dv_o,
    chunk_oja_fwd_o,
)
from fla.ops.utils import chunk_local_cumsum, solve_tril

from .leaky_wy import (
    close_vector_gate_backward,
    merge_gate_cotangents,
    merge_source_cotangents,
    prepare_leaky_wy_bwd,
    recompute_leaky_w_u_fwd,
)
from .vector_pair import vector_pair_backward, vector_pair_forward


def residual_fwd(
    target,
    source,
    beta,
    log_decay,
    query,
    initial_state,
    *,
    output_final_state,
    chunk_size,
):
    gate_cumsum = chunk_local_cumsum(
        log_decay,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )

    # The source branch uses the exclusive retention prefix while the target
    # branch consumes gamma*h directly. Together they implement the pre-leak
    # residual without forming the numerically unsafe gamma/a ratio.
    interaction = vector_pair_forward(
        source,
        beta,
        log_decay,
        gate_cumsum,
        chunk_size=chunk_size,
    )
    interaction = solve_tril(A=interaction, output_dtype=target.dtype)
    w, update, source_gated = recompute_leaky_w_u_fwd(
        target=target,
        source=source,
        beta=beta,
        log_decay=log_decay,
        A=interaction,
        gate=gate_cumsum,
    )
    states, update_direction, final_state = chunk_oja_fwd_h(
        v=source_gated,
        w=w,
        u=update,
        gv=gate_cumsum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        save_new_key=True,
        chunk_size=chunk_size,
    )
    _, frame_action = chunk_oja_fwd_o(
        q=query,
        k=update_direction,
        v=source,
        gv=gate_cumsum,
        h=states,
        scale=1.0,
        chunk_size=chunk_size,
    )
    return frame_action, update_direction, final_state, interaction


class _OjaResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        target,
        source,
        beta,
        log_decay,
        query,
        initial_state,
        chunk_size,
        output_final_state,
    ):
        frame_action, update_direction, final_state, interaction = residual_fwd(
            target,
            source,
            beta,
            log_decay,
            query,
            initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
        )
        ctx.chunk_size = chunk_size
        ctx.has_initial_state = initial_state is not None
        ctx.set_materialize_grads(False)
        saved_initial = target.new_empty(0) if initial_state is None else initial_state
        ctx.save_for_backward(
            target,
            source,
            beta,
            log_decay,
            query,
            saved_initial,
            interaction,
            frame_action,
        )
        return frame_action, update_direction, final_state

    @staticmethod
    def backward(ctx, grad_frame_action, grad_residual, grad_final_state):
        (
            target,
            source,
            beta,
            log_decay,
            query,
            saved_initial,
            interaction,
            frame_action,
        ) = ctx.saved_tensors
        initial_state = saved_initial if ctx.has_initial_state else None
        if grad_frame_action is None:
            grad_frame_action = torch.zeros_like(frame_action)
        else:
            # The exterior consumes a panel-packed primal. Its transpose
            # returns a token-major logical view with a head-interleaved
            # stride, while FLA's Oja output reverse requires contiguous
            # [B,T,H,r] ownership.
            grad_frame_action = grad_frame_action.contiguous()
        if grad_residual is None:
            grad_residual = torch.zeros_like(target)
        else:
            grad_residual = grad_residual.contiguous()
        gate_cumsum = chunk_local_cumsum(
            log_decay,
            chunk_size=ctx.chunk_size,
            output_dtype=torch.float32,
        )
        w, update, source_gated = recompute_leaky_w_u_fwd(
            target=target,
            source=source,
            beta=beta,
            log_decay=log_decay,
            A=interaction,
            gate=gate_cumsum,
        )
        states, residual, _ = chunk_oja_fwd_h(
            v=source_gated,
            w=w,
            u=update,
            gv=gate_cumsum,
            initial_state=initial_state,
            output_final_state=False,
            save_new_key=True,
            chunk_size=ctx.chunk_size,
        )
        grad_output_pair = chunk_oja_bwd_dA(
            v=source,
            gv=gate_cumsum,
            do=grad_frame_action,
            scale=1.0,
            chunk_size=ctx.chunk_size,
        )
        output_pair, grad_query, grad_residual_output = chunk_oja_bwd_dqk(
            q=query,
            k=residual,
            h=states,
            gv=gate_cumsum,
            dA=grad_output_pair,
            do=grad_frame_action,
            scale=1.0,
            chunk_size=ctx.chunk_size,
        )
        grad_residual_output = grad_residual_output.add_(grad_residual)
        grad_states, grad_initial, grad_update = chunk_oja_bwd_dhu(
            q=query,
            vg=source_gated,
            w=w,
            gv=gate_cumsum,
            h0=initial_state,
            dht=grad_final_state,
            do=grad_frame_action,
            dk=grad_residual_output,
            scale=1.0,
            chunk_size=ctx.chunk_size,
            states_in_fp32=False,
        )
        grad_source, grad_w, grad_gate_last = chunk_oja_bwd_dvwg_h(
            k=residual,
            v=source,
            gv=gate_cumsum,
            h=states,
            dh=grad_states,
            dk=grad_update,
            chunk_size=ctx.chunk_size,
        )
        grad_source, grad_gate_output = chunk_oja_bwd_dv_o(
            v=source,
            gv=gate_cumsum,
            o=frame_action,
            A=output_pair,
            dv=grad_source,
            do=grad_frame_action,
            chunk_size=ctx.chunk_size,
        )
        (
            grad_target,
            grad_source_wy,
            grad_beta_wy,
            grad_gate_wy,
            grad_log_decay_wy,
            grad_pair,
        ) = prepare_leaky_wy_bwd(
            target=target,
            source=source,
            beta=beta,
            log_decay=log_decay,
            A=interaction,
            grad_w=grad_w,
            grad_update=grad_update,
            gate=gate_cumsum,
        )
        (
            grad_source_pair,
            grad_gate_pair,
            grad_beta_pair,
            grad_log_decay_pair,
        ) = vector_pair_backward(
            source,
            beta,
            log_decay,
            gate_cumsum,
            grad_pair,
            chunk_size=ctx.chunk_size,
        )
        grad_source = merge_source_cotangents(
            grad_source,
            grad_source_wy,
            grad_source_pair,
            output_dtype=source.dtype,
        )
        grad_gate_cumsum = merge_gate_cotangents(
            grad_gate_output,
            grad_gate_wy,
            grad_gate_last,
            chunk_size=ctx.chunk_size,
        )

        grad_gate_pair = chunk_local_cumsum(
            grad_gate_pair,
            chunk_size=ctx.chunk_size,
            reverse=True,
            output_dtype=torch.float32,
        )
        grad_beta, grad_log_decay = close_vector_gate_backward(
            grad_gate_cumsum,
            grad_gate_pair,
            grad_beta_wy,
            grad_beta_pair,
            grad_log_decay_wy,
            grad_log_decay_pair,
        )
        return (
            grad_target,
            grad_source,
            grad_beta,
            grad_log_decay,
            grad_query,
            grad_initial,
            None,
            None,
        )


def oja_residual(
    target,
    source,
    beta,
    log_decay,
    query,
    initial_state,
    *,
    chunk_size=32,
    output_final_state=True,
):
    return _OjaResidual.apply(
        target,
        source,
        beta,
        log_decay,
        query,
        initial_state,
        chunk_size,
        output_final_state,
    )


__all__ = ["oja_residual"]
