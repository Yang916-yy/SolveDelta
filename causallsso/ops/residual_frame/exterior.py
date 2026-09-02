from __future__ import annotations

import torch

from .pair import (
    direct_e_pair_backward,
    direct_e_pair_forward,
)
from .common_left_h import chunk_dplr_fwd_h
from .common_left_o_bwd import (
    chunk_dplr_bwd_dAu,
    chunk_dplr_bwd_o,
)
from .common_left_o_fwd import chunk_dplr_fwd_o
from fla.ops.generalized_delta_rule.dplr.chunk_h_bwd import chunk_dplr_bwd_dhu
from fla.ops.generalized_delta_rule.dplr.wy_fast_bwd import chunk_dplr_bwd_wy
from fla.ops.generalized_delta_rule.dplr.wy_fast_fwd import prepare_wy_repr_fwd
from fla.ops.rwkv6.chunk import chunk_rwkv6_fwd_cumsum
from fla.ops.utils.constant import RCP_LN2

def _forward(
    d_panel,
    paired,
    z,
    log_decay,
    initial_state,
    chunk_size,
    *,
    final_state,
    output_required=True,
):
    cumulative, _ = chunk_rwkv6_fwd_cumsum(
        log_decay,
        chunk_size,
        scale=RCP_LN2,
    )
    A_qd, A_ed, q_scaled, d_tail, e_scaled = direct_e_pair_forward(
        d_panel,
        paired,
        cumulative,
        chunk_size=chunk_size,
    )
    w, update, inverse = prepare_wy_repr_fwd(
        ag=e_scaled,
        v=z,
        A_ak=A_ed,
        A_ab=A_ed,
        cu_seqlens=None,
        chunk_size=chunk_size,
        chunk_indices=None,
    )
    states, z_new, state_out = chunk_dplr_fwd_h(
        kg=d_tail,
        v=z,
        w=w,
        u=update,
        gk=cumulative,
        initial_state=initial_state,
        output_final_state=final_state,
        cu_seqlens=None,
        chunk_size=chunk_size,
        chunk_indices=None,
    )
    output = None
    if output_required:
        output = chunk_dplr_fwd_o(
            qg=q_scaled,
            v=z,
            v_new=z_new,
            A_qk=A_qd,
            h=states,
            cu_seqlens=None,
            chunk_size=chunk_size,
            chunk_indices=None,
        )
    return output, state_out, (
        d_panel,
        paired,
        cumulative,
        A_qd,
        A_ed,
        q_scaled,
        d_tail,
        e_scaled,
        w,
        update,
        inverse,
        states,
        z_new,
    )


def _backward(
    d_panel,
    paired,
    z,
    log_decay,
    initial_state,
    grad_output,
    grad_final_state,
    chunk_size,
    *,
    panel_gradient_dtype=None,
):
    if grad_output is None:
        grad_output = torch.zeros_like(z)
    (
        _,
        _,
        (
            d_panel,
            paired,
            cumulative,
            A_qd,
            A_ed,
            q_scaled,
            d_tail,
            e_scaled,
            w,
            update,
            inverse,
            states,
            z_new,
        ),
    ) = _forward(
        d_panel,
        paired,
        z,
        log_decay,
        initial_state,
        chunk_size,
        final_state=False,
        output_required=False,
    )
    grad_z_output, grad_A_qd = chunk_dplr_bwd_dAu(
        v=z,
        v_new=z_new,
        do=grad_output,
        A_qb=A_qd,
        scale=1.0,
        cu_seqlens=None,
        chunk_size=chunk_size,
        chunk_indices=None,
    )
    grad_states, grad_initial, grad_update = chunk_dplr_bwd_dhu(
        qg=q_scaled,
        bg=d_tail,
        w=w,
        gk=cumulative,
        h0=initial_state,
        dht=grad_final_state,
        do=grad_output,
        dv=grad_z_output,
        cu_seqlens=None,
        chunk_size=chunk_size,
        chunk_indices=None,
    )
    del grad_z_output
    (
        grad_q_scaled,
        grad_d_tail,
        grad_w,
        grad_gate_tail,
    ) = chunk_dplr_bwd_o(
        k=d_tail,
        v=z,
        v_new=z_new,
        gk=cumulative,
        do=grad_output,
        h=states,
        dh=grad_states,
        dv=grad_update,
        w=w,
        cu_seqlens=None,
        chunk_size=chunk_size,
        scale=1.0,
        chunk_indices=None,
    )
    del states, z_new, grad_states
    grad_A_ed_0, grad_A_ed_1, grad_z, grad_e_scaled = chunk_dplr_bwd_wy(
        A_ab_inv=inverse,
        A_ak=A_ed,
        v=z,
        ag=e_scaled,
        dw=grad_w,
        du=grad_update,
        dv0=grad_update,
        cu_seqlens=None,
        chunk_size=chunk_size,
        chunk_indices=None,
    )
    del A_ed, e_scaled, w, update, inverse, grad_update, grad_w
    del (
        A_qd,
        q_scaled,
        d_tail,
    )
    grad_d_panel, grad_paired, grad_decay = direct_e_pair_backward(
        d_panel,
        paired,
        cumulative,
        grad_A_qd,
        grad_A_ed_0,
        grad_A_ed_1,
        grad_q_scaled,
        grad_d_tail,
        grad_e_scaled,
        grad_gate_tail,
        chunk_size=chunk_size,
        gradient_dtype=panel_gradient_dtype,
    )
    return grad_d_panel, grad_paired, grad_z, grad_decay, grad_initial


class _DirectEResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        d_panel,
        paired,
        z,
        log_decay,
        initial_state,
        chunk_size,
        output_final_state,
    ):
        output, state_out, _ = _forward(
            d_panel,
            paired,
            z,
            log_decay,
            initial_state,
            chunk_size,
            final_state=output_final_state,
        )
        ctx.chunk_size = chunk_size
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(d_panel, paired, z, log_decay, initial_state)
        return output, state_out

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        d_panel, paired, z, log_decay, initial_state = ctx.saved_tensors
        gradients = _backward(
            d_panel,
            paired,
            z,
            log_decay,
            initial_state,
            grad_output,
            grad_final_state,
            ctx.chunk_size,
        )
        return (
            *gradients,
            None,
            None,
        )


def direct_e_residual(
    d_panel,
    paired,
    z,
    log_decay,
    initial_state,
    *,
    chunk_size=16,
    output_final_state=True,
):
    return _DirectEResidual.apply(
        d_panel,
        paired,
        z,
        log_decay,
        initial_state,
        chunk_size,
        output_final_state,
    )


__all__ = ["direct_e_residual"]
