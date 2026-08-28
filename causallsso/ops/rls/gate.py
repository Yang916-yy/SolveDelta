# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
"""FLA beta-sigmoid specialized to a BF16 consumer ABI."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


_BLOCK = 2048


@triton.jit
def _gate_forward_kernel(raw, output, scale, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    value = tl.load(raw + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output + offsets, scale * tl.sigmoid(value), mask=mask)


@triton.jit
def _gate_backward_kernel(raw, grad_output, grad_raw, scale, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    value = tl.load(raw + offsets, mask=mask, other=0.0).to(tl.float32)
    gradient = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
    sigma = tl.sigmoid(value)
    tl.store(grad_raw + offsets, gradient * scale * sigma * (1.0 - sigma), mask=mask)


class _ActivatedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw: torch.Tensor, scale: float) -> torch.Tensor:
        raw = raw.contiguous()
        output = torch.empty_like(raw)
        _gate_forward_kernel[(triton.cdiv(raw.numel(), _BLOCK),)](
            raw, output, scale, raw.numel(), BLOCK=_BLOCK,
            num_warps=8, num_stages=1,
        )
        ctx.save_for_backward(raw)
        ctx.scale = scale
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (raw,) = ctx.saved_tensors
        grad_raw = torch.empty_like(raw)
        _gate_backward_kernel[(triton.cdiv(raw.numel(), _BLOCK),)](
            raw, grad_output.contiguous(), grad_raw, ctx.scale, raw.numel(),
            BLOCK=_BLOCK, num_warps=8, num_stages=1,
        )
        return grad_raw, None


def activated_gate(raw: torch.Tensor, *, scale: float = 2.0) -> torch.Tensor:
    """Return ``scale*sigmoid(raw)`` directly in the raw operand dtype."""
    return _ActivatedGate.apply(raw, scale)


__all__ = ["activated_gate"]
