# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
"""FLA L2Norm specialized for fused-projection views.

The arithmetic and row ownership match FLA's MIT-licensed L2Norm kernels.  The
only specialization is explicit source strides; the normalized panel and its
transpose output stay packed for the native geometry owners.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.utils import autotune_cache_kwargs


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=warps)
        for bt in (8, 16, 32, 64, 128)
        for warps in (1, 2, 4, 8)
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["TOTAL", "LENGTH"])
def _strided_l2norm_fwd_kernel(
    x,
    y,
    rstd,
    TOTAL,
    LENGTH,
    H: tl.constexpr,
    D: tl.constexpr,
    SX_B: tl.constexpr,
    SX_T: tl.constexpr,
    SX_H: tl.constexpr,
    SX_D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    coord = tl.arange(0, BD)
    row_mask = row < TOTAL
    mask = row_mask[:, None] & (coord[None, :] < D)
    token_flat, head = row // H, row % H
    batch, token = token_flat // LENGTH, token_flat % LENGTH
    x_offset = (
        batch[:, None] * SX_B
        + token[:, None] * SX_T
        + head[:, None] * SX_H
        + coord[None, :] * SX_D
    )
    y_offset = row[:, None] * D + coord[None, :]
    b_x = tl.load(x + x_offset, mask=mask, other=0.0).to(tl.float32)
    b_rstd = 1.0 / tl.sqrt(tl.sum(b_x * b_x, axis=1) + 1.0e-24)
    tl.store(y + y_offset, b_x * b_rstd[:, None], mask=mask)
    tl.store(rstd + row, b_rstd, mask=row_mask)


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=warps)
        for bt in (8, 16, 32, 64, 128)
        for warps in (1, 2, 4, 8)
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _strided_l2norm_bwd_kernel(
    y,
    rstd,
    dy,
    dx,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    coord = tl.arange(0, BD)
    row_mask = row < T
    mask = row_mask[:, None] & (coord[None, :] < D)
    offset = row[:, None] * D + coord[None, :]
    b_y = tl.load(y + offset, mask=mask, other=0.0).to(tl.float32)
    b_dy = tl.load(dy + offset, mask=mask, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd + row, mask=row_mask, other=0.0).to(tl.float32)
    b_dx = (
        b_dy - tl.sum(b_dy * b_y, axis=1)[:, None] * b_y
    ) * b_rstd[:, None]
    tl.store(dx + offset, b_dx, mask=mask)


class _StridedL2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.stride(-1) != 1:
            raise ValueError(
                "stride-aware L2Norm requires [B,T,H,D] with unit inner stride"
            )
        batch, length, heads, width = x.shape
        rows = batch * length * heads
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        rstd = torch.empty(batch, length, heads, dtype=torch.float32, device=x.device)
        block = max(16, triton.next_power_of_2(width))
        _strided_l2norm_fwd_kernel[
            lambda meta: (triton.cdiv(rows, meta["BT"]),)
        ](
            x,
            y,
            rstd,
            TOTAL=rows,
            LENGTH=length,
            H=heads,
            D=width,
            SX_B=x.stride(0),
            SX_T=x.stride(1),
            SX_H=x.stride(2),
            SX_D=x.stride(3),
            BD=block,
        )
        ctx.save_for_backward(y, rstd)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        y, rstd = ctx.saved_tensors
        rows, width = y.numel() // y.shape[-1], y.shape[-1]
        grad_x = torch.empty(y.shape, dtype=y.dtype, device=y.device)
        block = max(16, triton.next_power_of_2(width))
        _strided_l2norm_bwd_kernel[
            lambda meta: (triton.cdiv(rows, meta["BT"]),)
        ](
            y,
            rstd,
            grad_y,
            grad_x,
            T=rows,
            D=width,
            BD=block,
        )
        return grad_x


def strided_l2norm(x: torch.Tensor) -> torch.Tensor:
    return _StridedL2Norm.apply(x)


__all__ = ["strided_l2norm"]
