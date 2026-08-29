# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
# Specialized from FLA's MIT-licensed gated-Oja pair/WY/state owners.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.gated_oja_rule.chunk_h import (
    chunk_oja_bwd_dhu,
    chunk_oja_bwd_dvwg_h,
    chunk_oja_fwd_h,
)
from fla.ops.gated_oja_rule.chunk_kkt import (
    chunk_scaled_dot_kkt_bwd_gk,
    chunk_scaled_dot_kkt_fwd,
)
from fla.ops.gated_oja_rule.wy_fast import (
    prepare_wy_repr_bwd,
    recompute_w_u_fwd,
)
from fla.ops.utils import solve_tril


@triton.jit
def _recompute_residual_wy_kernel(
    target,
    source,
    beta,
    inverse,
    w,
    update,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    BR: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    token = chunk * C + tl.arange(0, C)
    valid_t = token < T
    row = tl.arange(0, C)
    inverse_offset = (
        ((batch * T + token[:, None]) * H + head) * C + row[None, :]
    )
    inv = tl.load(
        inverse + inverse_offset,
        mask=valid_t[:, None],
        other=0.0,
    )
    beta_offset = (batch * T + token) * H + head
    scale = tl.load(beta + beta_offset, mask=valid_t, other=0.0)

    for block in range(0, tl.cdiv(R, BR)):
        coord = block * BR + tl.arange(0, BR)
        valid_r = coord < R
        source_offset = (
            ((batch * T + token[:, None]) * H + head) * R
            + coord[None, :]
        )
        src = tl.load(
            source + source_offset,
            mask=valid_t[:, None] & valid_r[None, :],
            other=0.0,
        )
        tgt = tl.load(
            target + source_offset,
            mask=valid_t[:, None] & valid_r[None, :],
            other=0.0,
        )
        src_out = tl.dot(inv, (src * scale[:, None]).to(src.dtype))
        tgt_out = tl.dot(inv, (tgt * scale[:, None]).to(tgt.dtype))
        tl.store(
            w + source_offset,
            src_out,
            mask=valid_t[:, None] & valid_r[None, :],
        )
        tl.store(
            update + source_offset,
            tgt_out,
            mask=valid_t[:, None] & valid_r[None, :],
        )


def recompute_residual_wy(target, source, beta, inverse, *, chunk_size):
    batch, length, heads, rank = target.shape
    w = torch.empty_like(source)
    update = torch.empty_like(target)
    _recompute_residual_wy_kernel[(triton.cdiv(length, chunk_size), batch * heads)](
        target,
        source,
        beta,
        inverse,
        w,
        update,
        T=length,
        H=heads,
        R=rank,
        C=chunk_size,
        BR=64,
        num_warps=4,
        num_stages=3,
    )
    return w, update


@triton.jit
def _residual_state_bwd_kernel(
    source,
    w,
    grad_final,
    grad_delta_input,
    grad_states,
    grad_initial,
    grad_update,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
):
    rank_block = tl.program_id(0)
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    rank = rank_block * BR + tl.arange(0, BR)
    value = tl.arange(0, BV)
    valid_r = rank < R
    chunks = tl.cdiv(T, C)
    state_base = batch_head * R * V + rank[:, None] * V

    dstate_0 = tl.load(
        grad_final + state_base + value[None, :],
        mask=valid_r[:, None] & (value[None, :] < V), other=0.0,
    ).to(tl.float32)
    if V > BV:
        dstate_1 = tl.load(
            grad_final + state_base + (BV + value)[None, :],
            mask=valid_r[:, None] & ((BV + value)[None, :] < V), other=0.0,
        ).to(tl.float32)
    if V > 2 * BV:
        dstate_2 = tl.load(
            grad_final + state_base + (2 * BV + value)[None, :],
            mask=valid_r[:, None] & ((2 * BV + value)[None, :] < V), other=0.0,
        ).to(tl.float32)
    if V > 3 * BV:
        dstate_3 = tl.load(
            grad_final + state_base + (3 * BV + value)[None, :],
            mask=valid_r[:, None] & ((3 * BV + value)[None, :] < V), other=0.0,
        ).to(tl.float32)

    lane = tl.arange(0, C)
    for reverse_chunk in range(0, chunks):
        chunk = chunks - 1 - reverse_chunk
        token = chunk * C + lane
        valid_t = token < T
        chunk_base = (
            ((batch * chunks + chunk) * H + head) * R * V
            + rank[:, None] * V
        )
        tl.store(
            grad_states + chunk_base + value[None, :], dstate_0,
            mask=valid_r[:, None] & (value[None, :] < V),
        )
        if V > BV:
            tl.store(
                grad_states + chunk_base + (BV + value)[None, :], dstate_1,
                mask=valid_r[:, None] & ((BV + value)[None, :] < V),
            )
        if V > 2 * BV:
            tl.store(
                grad_states + chunk_base + (2 * BV + value)[None, :], dstate_2,
                mask=valid_r[:, None] & ((2 * BV + value)[None, :] < V),
            )
        if V > 3 * BV:
            tl.store(
                grad_states + chunk_base + (3 * BV + value)[None, :], dstate_3,
                mask=valid_r[:, None] & ((3 * BV + value)[None, :] < V),
            )

        vector_base = ((batch * T + token[:, None]) * H + head) * V
        src_0 = tl.load(
            source + vector_base + value[None, :],
            mask=valid_t[:, None] & (value[None, :] < V), other=0.0,
        )
        dupdate = tl.dot(src_0, tl.trans(dstate_0).to(src_0.dtype))
        if V > BV:
            src_1 = tl.load(
                source + vector_base + (BV + value)[None, :],
                mask=valid_t[:, None] & ((BV + value)[None, :] < V), other=0.0,
            )
            dupdate += tl.dot(src_1, tl.trans(dstate_1).to(src_1.dtype))
        if V > 2 * BV:
            src_2 = tl.load(
                source + vector_base + (2 * BV + value)[None, :],
                mask=valid_t[:, None] & ((2 * BV + value)[None, :] < V), other=0.0,
            )
            dupdate += tl.dot(src_2, tl.trans(dstate_2).to(src_2.dtype))
        if V > 3 * BV:
            src_3 = tl.load(
                source + vector_base + (3 * BV + value)[None, :],
                mask=valid_t[:, None] & ((3 * BV + value)[None, :] < V), other=0.0,
            )
            dupdate += tl.dot(src_3, tl.trans(dstate_3).to(src_3.dtype))

        rank_base = ((batch * T + token[:, None]) * H + head) * R
        dupdate += tl.load(
            grad_delta_input + rank_base + rank[None, :],
            mask=valid_t[:, None] & valid_r[None, :], other=0.0,
        )
        tl.store(
            grad_update + rank_base + rank[None, :], dupdate,
            mask=valid_t[:, None] & valid_r[None, :],
        )

        w_0 = tl.load(
            w + vector_base + value[None, :],
            mask=valid_t[:, None] & (value[None, :] < V), other=0.0,
        )
        dstate_0 -= tl.dot(tl.trans(dupdate).to(w_0.dtype), w_0)
        if V > BV:
            w_1 = tl.load(
                w + vector_base + (BV + value)[None, :],
                mask=valid_t[:, None] & ((BV + value)[None, :] < V), other=0.0,
            )
            dstate_1 -= tl.dot(tl.trans(dupdate).to(w_1.dtype), w_1)
        if V > 2 * BV:
            w_2 = tl.load(
                w + vector_base + (2 * BV + value)[None, :],
                mask=valid_t[:, None] & ((2 * BV + value)[None, :] < V), other=0.0,
            )
            dstate_2 -= tl.dot(tl.trans(dupdate).to(w_2.dtype), w_2)
        if V > 3 * BV:
            w_3 = tl.load(
                w + vector_base + (3 * BV + value)[None, :],
                mask=valid_t[:, None] & ((3 * BV + value)[None, :] < V), other=0.0,
            )
            dstate_3 -= tl.dot(tl.trans(dupdate).to(w_3.dtype), w_3)

    tl.store(
        grad_initial + state_base + value[None, :], dstate_0,
        mask=valid_r[:, None] & (value[None, :] < V),
    )
    if V > BV:
        tl.store(
            grad_initial + state_base + (BV + value)[None, :], dstate_1,
            mask=valid_r[:, None] & ((BV + value)[None, :] < V),
        )
    if V > 2 * BV:
        tl.store(
            grad_initial + state_base + (2 * BV + value)[None, :], dstate_2,
            mask=valid_r[:, None] & ((2 * BV + value)[None, :] < V),
        )
    if V > 3 * BV:
        tl.store(
            grad_initial + state_base + (3 * BV + value)[None, :], dstate_3,
            mask=valid_r[:, None] & ((3 * BV + value)[None, :] < V),
        )


def residual_state_backward(source, w, grad_delta, grad_final, *, chunk_size):
    batch, length, heads, value_dim = source.shape
    rank = grad_delta.shape[-1]
    chunks = triton.cdiv(length, chunk_size)
    grad_states = torch.empty(
        batch, chunks, heads, rank, value_dim,
        dtype=source.dtype, device=source.device,
    )
    grad_update = torch.empty_like(grad_delta)
    grad_initial = torch.empty(
        batch, heads, rank, value_dim,
        dtype=torch.float32, device=source.device,
    )
    _residual_state_bwd_kernel[(triton.cdiv(rank, 64), batch * heads)](
        source, w, grad_final, grad_delta, grad_states, grad_initial, grad_update,
        T=length, H=heads, R=rank, V=value_dim, C=chunk_size, BR=64, BV=64,
        num_warps=4, num_stages=3,
    )
    return grad_states, grad_initial, grad_update


def residual_fwd(target, source, beta, initial_state, *, output_final_state, chunk_size):
    # The residual predictor has no vector gate.  Use FLA's mature ungated
    # pair owner directly; passing an all-zero gk selects the slower gated
    # sub-block schedule even though the resulting matrix is identical.
    interaction = chunk_scaled_dot_kkt_fwd(
        k=source,
        beta=beta,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    interaction = solve_tril(A=interaction, output_dtype=target.dtype)
    w, update = recompute_residual_wy(
        target,
        source,
        beta,
        interaction,
        chunk_size=chunk_size,
    )
    states, update_direction, final_state = chunk_oja_fwd_h(
        v=source,
        w=w,
        u=update,
        gv=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        save_new_key=True,
        chunk_size=chunk_size,
    )
    return update_direction, final_state, interaction


class _OjaResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        target,
        source,
        beta,
        initial_state,
        chunk_size,
        output_final_state,
    ):
        update_direction, final_state, interaction = residual_fwd(
            target,
            source,
            beta,
            initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
        )
        ctx.chunk_size = chunk_size
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(target, source, beta, initial_state, interaction)
        return update_direction, final_state

    @staticmethod
    def backward(ctx, grad_residual, grad_final_state):
        target, source, beta, initial_state, interaction = ctx.saved_tensors
        if grad_residual is None:
            grad_residual = torch.zeros_like(target)
        if grad_final_state is None:
            grad_final_state = torch.zeros_like(initial_state)

        w, update = recompute_residual_wy(
            target,
            source,
            beta,
            interaction,
            chunk_size=ctx.chunk_size,
        )
        states, residual, _ = chunk_oja_fwd_h(
            v=source,
            w=w,
            u=update,
            gv=None,
            initial_state=initial_state,
            output_final_state=False,
            save_new_key=True,
            chunk_size=ctx.chunk_size,
        )

        grad_states, grad_initial, grad_update = residual_state_backward(
            source,
            w,
            grad_residual,
            grad_final_state,
            chunk_size=ctx.chunk_size,
        )
        # Two generic FLA reverse helpers still require a vector-gate pointer.
        # Materialize their zero value only after state replay has consumed the
        # ungated specialization; it is not part of the forward cache.
        gv = torch.zeros_like(source, dtype=torch.float32)
        grad_source, grad_w, _ = chunk_oja_bwd_dvwg_h(
            k=residual,
            v=source,
            gv=gv,
            h=states,
            dh=grad_states,
            dk=grad_update,
            dgk=None,
            chunk_size=ctx.chunk_size,
        )
        (
            grad_target,
            grad_source_wy,
            grad_beta,
            _,
            grad_pair,
        ) = prepare_wy_repr_bwd(
            k=target,
            v=source,
            beta=beta,
            gv=gv,
            A=interaction,
            dw=grad_w,
            du=grad_update,
        )
        grad_source_pair, _, grad_beta_pair = chunk_scaled_dot_kkt_bwd_gk(
            k=source,
            g=gv,
            beta=beta,
            dA=grad_pair,
            chunk_size=ctx.chunk_size,
        )
        grad_source = grad_source.add_(grad_source_wy).add_(grad_source_pair)
        grad_beta = grad_beta.add_(grad_beta_pair)
        return (
            grad_target.to(target),
            grad_source.to(source),
            grad_beta.to(beta),
            grad_initial,
            None,
            None,
        )


def oja_residual(
    target,
    source,
    beta,
    initial_state,
    *,
    chunk_size=32,
    output_final_state=True,
):
    return _OjaResidual.apply(
        target,
        source,
        beta,
        initial_state,
        chunk_size,
        output_final_state,
    )


__all__ = ["oja_residual"]
