"""Chunk-affine FP32 effective-mass scan and exact transpose.

The schedule follows FLA's mature split: parallel chunk-local summaries, one
short boundary scan per sequence/head, then parallel chunk-local expansion.
It avoids both a Python token loop and one 1024-step GPU CTA.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _mass_chunk_summary_kernel(
    log_decay,
    chunk_scale,
    chunk_bias,
    T,
    H: tl.constexpr,
    NT: tl.constexpr,
    BT: tl.constexpr,
):
    chunk, batch_head = tl.program_id(0), tl.program_id(1).to(tl.int64)
    batch, head = batch_head // H, batch_head % H
    scale = 1.0
    bias = 0.0
    for local in range(BT):
        token = chunk * BT + local
        valid = token < T
        offset = (batch * T + token) * H + head
        decay = tl.exp(tl.load(log_decay + offset, mask=valid, other=0.0).to(tl.float32))
        scale = tl.where(valid, decay * scale, scale)
        bias = tl.where(valid, decay * bias + 1.0, bias)
    out = (batch * NT + chunk) * H + head
    tl.store(chunk_scale + out, scale)
    tl.store(chunk_bias + out, bias)


@triton.jit
def _mass_boundary_kernel(
    initial,
    chunk_scale,
    chunk_bias,
    chunk_boundary,
    final,
    H: tl.constexpr,
    NT: tl.constexpr,
):
    batch_head = tl.program_id(0).to(tl.int64)
    batch, head = batch_head // H, batch_head % H
    mass = tl.load(initial + batch_head).to(tl.float32)
    for chunk in range(NT):
        offset = (batch * NT + chunk) * H + head
        tl.store(chunk_boundary + offset, mass)
        scale = tl.load(chunk_scale + offset).to(tl.float32)
        bias = tl.load(chunk_bias + offset).to(tl.float32)
        mass = scale * mass + bias
    tl.store(final + batch_head, mass)


@triton.jit(do_not_specialize=["T"])
def _mass_expand_kernel(
    log_decay,
    chunk_boundary,
    previous,
    current,
    T,
    H: tl.constexpr,
    NT: tl.constexpr,
    BT: tl.constexpr,
):
    chunk, batch_head = tl.program_id(0), tl.program_id(1).to(tl.int64)
    batch, head = batch_head // H, batch_head % H
    boundary_offset = (batch * NT + chunk) * H + head
    mass = tl.load(chunk_boundary + boundary_offset).to(tl.float32)
    for local in range(BT):
        token = chunk * BT + local
        valid = token < T
        offset = (batch * T + token) * H + head
        tl.store(previous + offset, mass, mask=valid)
        decay = tl.exp(tl.load(log_decay + offset, mask=valid, other=0.0).to(tl.float32))
        mass = tl.where(valid, decay * mass + 1.0, mass)
        tl.store(current + offset, mass, mask=valid)


@triton.jit(do_not_specialize=["T"])
def _mass_bwd_chunk_summary_kernel(
    log_decay,
    grad_previous,
    grad_current,
    grad_chunk_direct,
    T,
    H: tl.constexpr,
    NT: tl.constexpr,
    BT: tl.constexpr,
):
    chunk, batch_head = tl.program_id(0), tl.program_id(1).to(tl.int64)
    batch, head = batch_head // H, batch_head % H
    adjoint = 0.0
    for reverse_local in range(BT):
        local = BT - 1 - reverse_local
        token = chunk * BT + local
        valid = token < T
        offset = (batch * T + token) * H + head
        grad_cur = tl.load(grad_current + offset, mask=valid, other=0.0).to(tl.float32)
        grad_prev = tl.load(grad_previous + offset, mask=valid, other=0.0).to(tl.float32)
        decay = tl.exp(tl.load(log_decay + offset, mask=valid, other=0.0).to(tl.float32))
        adjoint = tl.where(valid, (adjoint + grad_cur) * decay + grad_prev, adjoint)
    out = (batch * NT + chunk) * H + head
    tl.store(grad_chunk_direct + out, adjoint)


@triton.jit
def _mass_bwd_boundary_kernel(
    chunk_scale,
    grad_chunk_direct,
    grad_final,
    grad_chunk_end,
    grad_initial,
    H: tl.constexpr,
    NT: tl.constexpr,
):
    batch_head = tl.program_id(0).to(tl.int64)
    batch, head = batch_head // H, batch_head % H
    adjoint = tl.load(grad_final + batch_head).to(tl.float32)
    for reverse_chunk in range(NT):
        chunk = NT - 1 - reverse_chunk
        offset = (batch * NT + chunk) * H + head
        tl.store(grad_chunk_end + offset, adjoint)
        scale = tl.load(chunk_scale + offset).to(tl.float32)
        direct = tl.load(grad_chunk_direct + offset).to(tl.float32)
        adjoint = direct + scale * adjoint
    tl.store(grad_initial + batch_head, adjoint)


@triton.jit(do_not_specialize=["T"])
def _mass_bwd_expand_kernel(
    log_decay,
    previous,
    grad_previous,
    grad_current,
    grad_chunk_end,
    grad_log_decay,
    T,
    H: tl.constexpr,
    NT: tl.constexpr,
    BT: tl.constexpr,
):
    chunk, batch_head = tl.program_id(0), tl.program_id(1).to(tl.int64)
    batch, head = batch_head // H, batch_head % H
    boundary_offset = (batch * NT + chunk) * H + head
    adjoint = tl.load(grad_chunk_end + boundary_offset).to(tl.float32)
    for reverse_local in range(BT):
        local = BT - 1 - reverse_local
        token = chunk * BT + local
        valid = token < T
        offset = (batch * T + token) * H + head
        adjoint += tl.load(grad_current + offset, mask=valid, other=0.0).to(tl.float32)
        decay = tl.exp(tl.load(log_decay + offset, mask=valid, other=0.0).to(tl.float32))
        mass_previous = tl.load(previous + offset, mask=valid, other=0.0).to(tl.float32)
        tl.store(grad_log_decay + offset, adjoint * decay * mass_previous, mask=valid)
        grad_prev = tl.load(grad_previous + offset, mask=valid, other=0.0).to(tl.float32)
        adjoint = tl.where(valid, adjoint * decay + grad_prev, adjoint)


class _MassPrefix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_decay: torch.Tensor, initial: torch.Tensor, chunk_size: int):
        if log_decay.ndim != 3:
            raise ValueError("log_decay must have shape [B,T,H]")
        batch, length, heads = log_decay.shape
        if initial.shape != (batch, heads):
            raise ValueError("initial must have shape [B,H]")
        if log_decay.dtype != torch.float32 or initial.dtype != torch.float32:
            raise TypeError("mass scan inputs must be FP32")
        if chunk_size not in (16, 32, 64):
            raise ValueError("mass chunk size must be 16, 32, or 64")
        chunks = triton.cdiv(length, chunk_size)
        chunk_shape = (batch, chunks, heads)
        chunk_scale = torch.empty(chunk_shape, dtype=torch.float32, device=log_decay.device)
        chunk_bias = torch.empty_like(chunk_scale)
        chunk_boundary = torch.empty_like(chunk_scale)
        previous = torch.empty_like(log_decay)
        current = torch.empty_like(log_decay)
        final = torch.empty_like(initial)
        grid = (chunks, batch * heads)
        _mass_chunk_summary_kernel[grid](
            log_decay,
            chunk_scale,
            chunk_bias,
            T=length,
            H=heads,
            NT=chunks,
            BT=chunk_size,
            num_warps=1,
            num_stages=1,
        )
        _mass_boundary_kernel[(batch * heads,)](
            initial,
            chunk_scale,
            chunk_bias,
            chunk_boundary,
            final,
            H=heads,
            NT=chunks,
            num_warps=1,
            num_stages=1,
        )
        _mass_expand_kernel[grid](
            log_decay,
            chunk_boundary,
            previous,
            current,
            T=length,
            H=heads,
            NT=chunks,
            BT=chunk_size,
            num_warps=1,
            num_stages=1,
        )
        ctx.chunk_size = chunk_size
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(log_decay, previous, chunk_scale)
        return previous, current, final

    @staticmethod
    def backward(ctx, grad_previous, grad_current, grad_final):
        log_decay, previous, chunk_scale = ctx.saved_tensors
        batch, length, heads = log_decay.shape
        chunks = triton.cdiv(length, ctx.chunk_size)
        grad_previous = torch.zeros_like(previous) if grad_previous is None else grad_previous.contiguous()
        grad_current = torch.zeros_like(previous) if grad_current is None else grad_current.contiguous()
        grad_final = torch.zeros_like(chunk_scale[:, 0]) if grad_final is None else grad_final.contiguous()
        grad_chunk_direct = torch.empty_like(chunk_scale)
        grad_chunk_end = torch.empty_like(chunk_scale)
        grad_decay = torch.empty_like(log_decay)
        grad_initial = torch.empty((batch, heads), dtype=torch.float32, device=log_decay.device)
        grid = (chunks, batch * heads)
        _mass_bwd_chunk_summary_kernel[grid](
            log_decay,
            grad_previous,
            grad_current,
            grad_chunk_direct,
            T=length,
            H=heads,
            NT=chunks,
            BT=ctx.chunk_size,
            num_warps=1,
            num_stages=1,
        )
        _mass_bwd_boundary_kernel[(batch * heads,)](
            chunk_scale,
            grad_chunk_direct,
            grad_final,
            grad_chunk_end,
            grad_initial,
            H=heads,
            NT=chunks,
            num_warps=1,
            num_stages=1,
        )
        _mass_bwd_expand_kernel[grid](
            log_decay,
            previous,
            grad_previous,
            grad_current,
            grad_chunk_end,
            grad_decay,
            T=length,
            H=heads,
            NT=chunks,
            BT=ctx.chunk_size,
            num_warps=1,
            num_stages=1,
        )
        return grad_decay, grad_initial, None


def mass_prefix(
    log_decay: torch.Tensor,
    initial: torch.Tensor,
    *,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _MassPrefix.apply(log_decay, initial, chunk_size)
