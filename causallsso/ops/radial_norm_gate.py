# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
"""FLA RMSNorm-gate owner specialized with a bounded radial readout.

The row ownership, FP32 reductions, autotuning surface, and strict RMSNorm
transpose are adapted from FLA's MIT-licensed fused norm-gate implementation.
SolveDelta adds one bounded, per-head radial scale while ``rstd`` is resident.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from fla.utils import autotune_cache_kwargs, get_multiprocessor_count


RADIAL_STRENGTH_INIT = 1.0


def radial_rms_norm_gated_reference(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    radial_strength: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Exact PyTorch definition used by CPU/model-reference execution."""
    if x.shape != gate.shape or x.ndim < 2:
        raise ValueError("x and gate must have the same [...,H,D] shape")
    heads, width = x.shape[-2:]
    if weight.shape != (width,):
        raise ValueError("weight must have shape [D]")
    if radial_strength.shape != (heads,):
        raise ValueError("radial_strength must have shape [H]")

    reduction_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    x_fp = x.to(reduction_dtype)
    rstd = torch.rsqrt(x_fp.square().mean(dim=-1, keepdim=True) + eps)
    reference_rms = 1.0 / math.sqrt(width)
    z = (1.0 - reference_rms * rstd) / (1.0 + reference_rms * rstd)
    alpha = torch.sigmoid(2.0 * radial_strength.to(reduction_dtype)) - 0.5
    scale = 1.0 + alpha.view(*((1,) * (x.ndim - 2)), heads, 1) * z
    base = (
        x_fp
        * rstd
        * weight.to(reduction_dtype)
        * torch.sigmoid(gate.to(reduction_dtype))
    )
    return (scale * base).to(x.dtype)


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=warps)
        for bt in (16, 32, 64)
        for warps in (4, 8)
    ],
    key=["D", "H", "NB"],
    **autotune_cache_kwargs,
)
@triton.jit
def _radial_rms_norm_gated_fwd_kernel(
    x,
    gate,
    weight,
    radial_strength,
    y,
    rstd,
    eps,
    T,
    D: tl.constexpr,
    H: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    NB: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    coord = tl.arange(0, BD)
    row_mask = row < T
    mask = row_mask[:, None] & (coord[None, :] < D)
    offset = row[:, None] * D + coord[None, :]

    b_x = tl.load(x + offset, mask=mask, other=0.0).to(tl.float32)
    b_gate = tl.load(gate + offset, mask=mask, other=0.0).to(tl.float32)
    b_weight = tl.load(weight + coord, mask=coord < D, other=0.0).to(
        tl.float32
    )
    b_rstd = 1.0 / tl.sqrt(tl.sum(b_x * b_x, axis=1) / D + eps)
    b_reference_rms = 1.0 / tl.sqrt(D + 0.0)
    b_z = (1.0 - b_reference_rms * b_rstd) / (
        1.0 + b_reference_rms * b_rstd
    )
    head = row % H
    b_strength = tl.load(
        radial_strength + head, mask=row_mask, other=0.0
    ).to(tl.float32)
    b_strength_sigmoid = tl.sigmoid(2.0 * b_strength)
    b_alpha = b_strength_sigmoid - 0.5
    b_scale = 1.0 + b_alpha * b_z
    b_base = (
        b_x
        * b_rstd[:, None]
        * b_weight[None, :]
        * tl.sigmoid(b_gate)
    )

    tl.store(rstd + row, b_rstd, mask=row_mask)
    tl.store(
        y + offset,
        (b_scale[:, None] * b_base).to(y.dtype.element_ty),
        mask=mask,
    )


@triton.heuristics({
    "RECOMPUTE_OUTPUT": lambda args: args["recomputed_y"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=warps)
        for bt in (16, 32, 64)
        for warps in (4, 8)
    ],
    key=["D", "H", "NB"],
    **autotune_cache_kwargs,
)
@triton.jit
def _radial_rms_norm_gated_bwd_kernel(
    x,
    gate,
    weight,
    radial_strength,
    rstd,
    dy,
    dx,
    dgate,
    recomputed_y,
    partial,
    T,
    BS,
    D: tl.constexpr,
    H: tl.constexpr,
    BD: tl.constexpr,
    BH: tl.constexpr,
    PARTIAL_WIDTH: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    BT: tl.constexpr,
    NB: tl.constexpr,
):
    program = tl.program_id(0)
    coord = tl.arange(0, BD)
    head_axis = tl.arange(0, BH)
    coord_mask = coord < D
    head_mask = head_axis < H
    b_weight = tl.load(weight + coord, mask=coord_mask, other=0.0).to(
        tl.float32
    )
    b_dw = tl.zeros((BT, BD), dtype=tl.float32)
    b_dstrength = tl.zeros((BH,), dtype=tl.float32)
    b_reference_rms = 1.0 / tl.sqrt(D + 0.0)

    for start in range(program * BS, program * BS + BS, BT):
        row = (start + tl.arange(0, BT)).to(tl.int64)
        row_mask = row < T
        owner_mask = row < min(program * BS + BS, T)
        mask = row_mask[:, None] & coord_mask[None, :]
        offset = row[:, None] * D + coord[None, :]

        b_x = tl.load(x + offset, mask=mask, other=0.0).to(tl.float32)
        b_gate_raw = tl.load(gate + offset, mask=mask, other=0.0).to(
            tl.float32
        )
        b_dy = tl.load(dy + offset, mask=mask, other=0.0).to(tl.float32)
        b_rstd = tl.load(rstd + row, mask=row_mask, other=0.0).to(
            tl.float32
        )
        head = row % H
        b_strength = tl.load(
            radial_strength + head, mask=row_mask, other=0.0
        ).to(tl.float32)
        b_strength_sigmoid = tl.sigmoid(2.0 * b_strength)
        b_alpha = b_strength_sigmoid - 0.5
        b_z = (1.0 - b_reference_rms * b_rstd) / (
            1.0 + b_reference_rms * b_rstd
        )
        b_scale = 1.0 + b_alpha * b_z

        b_xhat = tl.where(
            coord_mask[None, :], b_x * b_rstd[:, None], 0.0
        )
        b_gate = tl.sigmoid(b_gate_raw)
        b_pre_gate = b_xhat * b_weight[None, :]
        b_base = b_pre_gate * b_gate
        if RECOMPUTE_OUTPUT:
            tl.store(
                recomputed_y + offset,
                (b_scale[:, None] * b_base).to(
                    recomputed_y.dtype.element_ty
                ),
                mask=mask,
            )

        b_dscale = tl.sum(b_dy * b_base, axis=1)
        b_dbase = b_dy * b_scale[:, None]
        b_dgate = b_dbase * b_pre_gate * b_gate * (1.0 - b_gate)
        b_norm_dy = b_dbase * b_gate
        b_weighted_dy = b_norm_dy * b_weight[None, :]
        b_projection = tl.sum(b_xhat * b_weighted_dy, axis=1) / D
        b_dx = (
            b_weighted_dy - b_xhat * b_projection[:, None]
        ) * b_rstd[:, None]

        b_radial_denominator = 1.0 + b_reference_rms * b_rstd
        b_radial_coefficient = (
            b_dscale
            * b_alpha
            * 2.0
            * b_reference_rms
            * b_rstd
            * b_rstd
            / (D * b_radial_denominator * b_radial_denominator)
        )
        b_dx += b_radial_coefficient[:, None] * b_xhat

        tl.store(dx + offset, b_dx.to(dx.dtype.element_ty), mask=mask)
        tl.store(
            dgate + offset,
            b_dgate.to(dgate.dtype.element_ty),
            mask=mask,
        )

        b_dw += tl.where(
            owner_mask[:, None], b_norm_dy * b_xhat, 0.0
        )
        b_dstrength_row = (
            b_dscale
            * b_z
            * 2.0
            * b_strength_sigmoid
            * (1.0 - b_strength_sigmoid)
        )
        b_dstrength += tl.sum(
            tl.where(
                owner_mask[:, None]
                & (head[:, None] == head_axis[None, :]),
                b_dstrength_row[:, None],
                0.0,
            ),
            axis=0,
        )

    partial_offset = program * PARTIAL_WIDTH
    tl.store(
        partial + partial_offset + coord,
        tl.sum(b_dw, axis=0),
        mask=coord_mask,
    )
    tl.store(
        partial + partial_offset + D + head_axis,
        b_dstrength,
        mask=head_mask,
    )


def _radial_rms_norm_gated_fwd(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    radial_strength: torch.Tensor,
    eps: float,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = x.shape
    y = torch.empty_like(x)
    rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
    block = min(65536 // x.element_size(), triton.next_power_of_2(width))
    if width > block or width > 512:
        raise RuntimeError("radial RMSNorm supports head widths up to 512")
    nb = triton.cdiv(rows, 2048 * 32)
    _radial_rms_norm_gated_fwd_kernel[
        lambda meta: (triton.cdiv(rows, meta["BT"]),)
    ](
        x,
        gate,
        weight,
        radial_strength,
        y,
        rstd,
        eps,
        T=rows,
        D=width,
        H=heads,
        BD=block,
        NB=nb,
    )
    return y, rstd


def _radial_rms_norm_gated_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    radial_strength: torch.Tensor,
    rstd: torch.Tensor,
    heads: int,
    *,
    recompute_output: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    rows, width = x.shape
    dx = torch.empty_like(x)
    dgate = torch.empty_like(gate)
    recomputed_y = torch.empty_like(x) if recompute_output else None
    block = min(65536 // x.element_size(), triton.next_power_of_2(width))
    if width > block or width > 512:
        raise RuntimeError("radial RMSNorm supports head widths up to 512")
    head_block = triton.next_power_of_2(heads)
    programs = min(get_multiprocessor_count(x.device.index), rows)
    rows_per_program = math.ceil(rows / programs)
    partial_width = width + heads
    partial = torch.empty(
        programs,
        partial_width,
        dtype=torch.float32,
        device=x.device,
    )
    nb = triton.cdiv(rows, 2048 * 32)
    _radial_rms_norm_gated_bwd_kernel[(programs,)](
        x,
        gate,
        weight,
        radial_strength,
        rstd,
        dy,
        dx,
        dgate,
        recomputed_y,
        partial,
        T=rows,
        BS=rows_per_program,
        D=width,
        H=heads,
        BD=block,
        BH=head_block,
        PARTIAL_WIDTH=partial_width,
        NB=nb,
    )
    reduced = partial.sum(dim=0)
    return (
        dx,
        dgate,
        reduced[:width].to(weight.dtype),
        reduced[width:].to(radial_strength.dtype),
        recomputed_y,
    )


class _RadialRMSNormGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        radial_strength: torch.Tensor,
        eps: float,
        heads: int,
    ) -> torch.Tensor:
        if x.shape != gate.shape or x.ndim < 2:
            raise ValueError("x and gate must have the same [...,H,D] shape")
        if x.shape[-2] != heads:
            raise ValueError("the penultimate input dimension must equal H")
        if not x.is_contiguous() or not gate.is_contiguous():
            raise ValueError("CUDA radial RMSNorm inputs must be contiguous")
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        gate_2d = gate.reshape_as(x_2d)
        y, rstd = _radial_rms_norm_gated_fwd(
            x_2d,
            gate_2d,
            weight,
            radial_strength,
            eps,
            heads,
        )
        ctx.save_for_backward(x_2d, gate_2d, weight, radial_strength, rstd)
        ctx.shape = shape
        ctx.heads = heads
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, gate, weight, radial_strength, rstd = ctx.saved_tensors
        grad_y = grad_y.contiguous().reshape_as(x)
        dx, dgate, dweight, dstrength, _ = _radial_rms_norm_gated_bwd(
            grad_y,
            x,
            gate,
            weight,
            radial_strength,
            rstd,
            ctx.heads,
        )
        return (
            dx.reshape(ctx.shape),
            dgate.reshape(ctx.shape),
            dweight,
            dstrength,
            None,
            None,
        )


class _RadialRMSNormGatedLinearFunction(torch.autograd.Function):
    """FLA norm-linear lifetime ownership with SolveDelta's radial gate."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate: torch.Tensor,
        norm_weight: torch.Tensor,
        radial_strength: torch.Tensor,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor | None,
        shadow_weight: torch.Tensor | None,
        shadow_bias: torch.Tensor | None,
        eps: float,
        heads: int,
    ) -> torch.Tensor:
        if x.shape != gate.shape or x.ndim < 2:
            raise ValueError("x and gate must have the same [...,H,D] shape")
        if (
            x.shape[-2] != heads
            or not x.is_contiguous()
            or not gate.is_contiguous()
        ):
            raise ValueError(
                "CUDA radial norm-linear inputs must be contiguous [...,H,D]"
            )
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        gate_2d = gate.reshape_as(x_2d)
        y, rstd = _radial_rms_norm_gated_fwd(
            x_2d,
            gate_2d,
            norm_weight,
            radial_strength,
            eps,
            heads,
        )
        linear_dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else y.dtype
        )
        packed_weight = (
            linear_weight.to(linear_dtype)
            if shadow_weight is None
            else shadow_weight
        )
        packed_bias = (
            None
            if linear_bias is None
            else (
                linear_bias.to(linear_dtype)
                if shadow_bias is None
                else shadow_bias
            )
        )
        output = F.linear(
            y.reshape(*shape[:-2], -1).to(linear_dtype),
            packed_weight,
            packed_bias,
        )
        ctx.save_for_backward(
            x_2d,
            gate_2d,
            norm_weight,
            radial_strength,
            packed_weight,
            rstd,
        )
        ctx.shape = shape
        ctx.heads = heads
        ctx.linear_weight_dtype = linear_weight.dtype
        ctx.linear_bias_dtype = None if linear_bias is None else linear_bias.dtype
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            x,
            gate,
            norm_weight,
            radial_strength,
            linear_weight,
            rstd,
        ) = ctx.saved_tensors
        grad_output_2d = grad_output.contiguous().reshape(
            -1, grad_output.shape[-1]
        )
        grad_y = F.linear(grad_output_2d, linear_weight.t()).reshape_as(x)
        dx, dgate, dnorm_weight, dstrength, y = _radial_rms_norm_gated_bwd(
            grad_y,
            x,
            gate,
            norm_weight,
            radial_strength,
            rstd,
            ctx.heads,
            recompute_output=True,
        )
        if y is None:
            raise RuntimeError("radial norm-linear reverse did not regenerate output")
        grad_linear_weight = torch.mm(
            grad_output_2d.t(), y.reshape(grad_output_2d.shape[0], -1)
        ).to(ctx.linear_weight_dtype)
        grad_linear_bias = (
            None
            if ctx.linear_bias_dtype is None
            else grad_output_2d.sum(dim=0).to(ctx.linear_bias_dtype)
        )
        return (
            dx.reshape(ctx.shape),
            dgate.reshape(ctx.shape),
            dnorm_weight,
            dstrength,
            grad_linear_weight,
            grad_linear_bias,
            None,
            None,
            None,
            None,
        )


class RadialRMSNormGated(nn.Module):
    """Sigmoid-gated RMSNorm with a bounded learnable per-head radial path."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        eps: float = 1e-6,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.eps = eps
        self.activation = "sigmoid"
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )
        self.radial_strength = nn.Parameter(
            torch.full(
                (num_heads,),
                RADIAL_STRENGTH_INIT,
                device=device,
                dtype=torch.float32,
            )
        )
        self.radial_strength._no_weight_decay = True

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cuda":
            return radial_rms_norm_gated_reference(
                x,
                gate,
                self.weight,
                self.radial_strength,
                self.eps,
            )
        return _RadialRMSNormGatedFunction.apply(
            x,
            gate,
            self.weight,
            self.radial_strength.float(),
            self.eps,
            self.num_heads,
        )

    def forward_linear(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
        linear: nn.Linear,
    ) -> torch.Tensor:
        """Apply the output projection without saving the normalized panel."""
        if x.device.type != "cuda":
            normalized = radial_rms_norm_gated_reference(
                x,
                gate,
                self.weight,
                self.radial_strength,
                self.eps,
            )
            return linear(normalized.reshape(*x.shape[:-2], -1))
        return _RadialRMSNormGatedLinearFunction.apply(
            x,
            gate,
            self.weight,
            self.radial_strength.float(),
            linear.weight,
            linear.bias,
            getattr(linear, "_bf16_shadow_weight", None),
            getattr(linear, "_bf16_shadow_bias", None),
            self.eps,
            self.num_heads,
        )


__all__ = [
    "RADIAL_STRENGTH_INIT",
    "RadialRMSNormGated",
    "radial_rms_norm_gated_reference",
]
