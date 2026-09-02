# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
"""CUDA-Graph-safe fixed-dense specialization of FLA's fused linear CE.

FLA's general owner discovers the number of non-ignored labels with a device
``.item()`` and branches on the scalar output cotangent in Python.  A fixed
dense causal-LM graph has exactly ``B * (T - 1)`` supervised labels, so both
host decisions are unnecessary.  The matmul partitioning and Triton CE kernels
remain FLA's; this wrapper supplies the static denominator and always launches
the small output-cotangent scaling epilogue.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from fla.modules.fused_linear_cross_entropy import (
    MAX_FUSED_SIZE,
    STATIC_WARPS,
    cross_entropy_kernel,
    elementwise_mul_kernel,
    logsumexp_fwd,
)


def _forward(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    ignore_index: int,
    total: int,
    num_chunks: int,
    use_l2warp: bool,
    l2_penalty_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    token_count, hidden = x.shape
    vocab = weight.shape[0]
    if target.shape != (token_count,):
        raise ValueError("target must match the flattened token axis")
    if total <= 0 or total > token_count:
        raise ValueError("fixed dense denominator must be in [1, token_count]")

    block_vocab = min(MAX_FUSED_SIZE, triton.next_power_of_2(vocab))
    chunks = min(num_chunks, triton.cdiv(vocab, hidden))
    chunk_tokens = triton.next_power_of_2(triton.cdiv(token_count, chunks))
    chunks = triton.cdiv(token_count, chunk_tokens)

    grad_x = torch.zeros_like(x)
    grad_weight = torch.zeros_like(weight, dtype=torch.float32)
    grad_bias = (
        None
        if bias is None
        else torch.zeros_like(bias, dtype=torch.float32)
    )
    losses = torch.zeros(token_count, device=x.device, dtype=torch.float32)

    for chunk in range(chunks):
        start = chunk * chunk_tokens
        end = min((chunk + 1) * chunk_tokens, token_count)
        chunk_x = x[start:end]
        logits = F.linear(chunk_x, weight, bias)
        chunk_x_accum = chunk_x.float()
        chunk_target = target[start:end]
        lse = logsumexp_fwd(logits, scale=1.0, softcapping=None, dtype=torch.float32)

        chunk_loss = losses[start:end]
        if use_l2warp:
            max_logit, max_index = torch.max(logits, -1, keepdim=True)

        cross_entropy_kernel[(logits.shape[0],)](
            logits=logits,
            lse=lse,
            target=chunk_target,
            loss=chunk_loss,
            total=total,
            ignore_index=ignore_index,
            label_smoothing=0.0,
            logit_scale=1.0,
            logit_softcapping=None,
            reduction="mean",
            V=vocab,
            BV=block_vocab,
            num_warps=STATIC_WARPS,
        )

        if use_l2warp:
            grad_logits_l2 = torch.zeros_like(logits)
            penalty_grad = max_logit * (l2_penalty_factor / token_count)
            grad_logits_l2.scatter_(-1, max_index, penalty_grad)
            torch.addmm(
                input=grad_weight,
                mat1=grad_logits_l2.t().float(),
                mat2=chunk_x_accum,
                out=grad_weight,
            )
            if grad_bias is not None:
                torch.add(
                    input=grad_bias,
                    other=grad_logits_l2.sum(0, dtype=torch.float32),
                    out=grad_bias,
                )
            grad_x_l2 = torch.mm(grad_logits_l2, weight)
        else:
            grad_x_l2 = 0.0

        grad_x[start:end] = torch.mm(logits, weight) + grad_x_l2
        torch.addmm(
            input=grad_weight,
            mat1=logits.t().float(),
            mat2=chunk_x_accum,
            out=grad_weight,
        )
        if grad_bias is not None:
            torch.add(
                input=grad_bias,
                other=logits.sum(0, dtype=torch.float32),
                out=grad_bias,
            )

    return (
        losses.sum(),
        grad_x,
        grad_weight.to(weight.dtype),
        None if grad_bias is None else grad_bias.to(bias.dtype),
    )


def _scale_gradient_(gradient: torch.Tensor | None, cotangent: torch.Tensor) -> None:
    if gradient is None:
        return
    block = min(MAX_FUSED_SIZE, triton.next_power_of_2(gradient.shape[-1]))
    elementwise_mul_kernel[(triton.cdiv(gradient.numel(), block),)](
        x=gradient,
        g=cotangent,
        N=gradient.numel(),
        B=block,
        num_warps=STATIC_WARPS,
    )


class _FixedDenseLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        ignore_index: int,
        total: int,
        num_chunks: int,
        use_l2warp: bool,
        l2_penalty_factor: float,
    ) -> torch.Tensor:
        loss, grad_x, grad_weight, grad_bias = _forward(
            x,
            target,
            weight,
            bias,
            ignore_index=ignore_index,
            total=total,
            num_chunks=num_chunks,
            use_l2warp=use_l2warp,
            l2_penalty_factor=l2_penalty_factor,
        )
        saved_bias = x.new_empty(0) if grad_bias is None else grad_bias
        ctx.has_bias = grad_bias is not None
        ctx.save_for_backward(grad_x.detach(), grad_weight.detach(), saved_bias.detach())
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        grad_x, grad_weight, saved_bias = ctx.saved_tensors
        grad_bias = saved_bias if ctx.has_bias else None
        _scale_gradient_(grad_x, grad_loss)
        _scale_gradient_(grad_weight, grad_loss)
        _scale_gradient_(grad_bias, grad_loss)
        return grad_x, None, grad_weight, grad_bias, None, None, None, None, None


def fixed_dense_linear_cross_entropy(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
    total: int,
    num_chunks: int = 8,
    use_l2warp: bool = False,
    l2_penalty_factor: float = 1.0e-4,
) -> torch.Tensor:
    """Fused linear CE for a fixed graph with a static valid-label count."""
    return _FixedDenseLinearCrossEntropy.apply(
        x,
        target,
        weight,
        bias,
        ignore_index,
        total,
        num_chunks,
        use_l2warp,
        l2_penalty_factor,
    )


__all__ = ["fixed_dense_linear_cross_entropy"]
