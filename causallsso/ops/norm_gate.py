# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 SolveDelta contributors
# SPDX-License-Identifier: MIT
"""FLA sigmoid-gated RMSNorm with norm-linear lifetime ownership."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fla.modules.fused_norm_gate import (
    layer_norm_gated_bwd,
    layer_norm_gated_fwd,
    rms_norm_gated,
)


def rms_norm_gated_reference(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Exact PyTorch definition used by CPU/model-reference execution."""
    if x.shape != gate.shape or x.ndim < 2:
        raise ValueError("x and gate must have the same [...,H,D] shape")
    width = x.shape[-1]
    if weight.shape != (width,):
        raise ValueError("weight must have shape [D]")

    reduction_dtype = (
        torch.float64 if x.dtype == torch.float64 else torch.float32
    )
    x_fp = x.to(reduction_dtype)
    rstd = torch.rsqrt(x_fp.square().mean(dim=-1, keepdim=True) + eps)
    output = (
        x_fp
        * rstd
        * weight.to(reduction_dtype)
        * torch.sigmoid(gate.to(reduction_dtype))
    )
    return output.to(x.dtype)


class _RMSNormGatedLinearFunction(torch.autograd.Function):
    """FLA norm owner with output recomputation for a following Linear."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate: torch.Tensor,
        norm_weight: torch.Tensor,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor | None,
        shadow_weight: torch.Tensor | None,
        shadow_bias: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        if x.shape != gate.shape or x.ndim < 2:
            raise ValueError("x and gate must have the same [...,H,D] shape")
        if not x.is_contiguous() or not gate.is_contiguous():
            raise ValueError(
                "CUDA norm-linear inputs must be contiguous [...,H,D]"
            )

        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        gate_2d = gate.reshape_as(x_2d)
        y, mean, rstd, _ = layer_norm_gated_fwd(
            x=x_2d,
            g=gate_2d,
            weight=norm_weight,
            bias=None,
            activation="sigmoid",
            eps=eps,
            is_rms_norm=True,
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
            packed_weight,
            mean,
            rstd,
        )
        ctx.shape = shape
        ctx.eps = eps
        ctx.linear_weight_dtype = linear_weight.dtype
        ctx.linear_bias_dtype = None if linear_bias is None else linear_bias.dtype
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, gate, norm_weight, linear_weight, mean, rstd = ctx.saved_tensors
        grad_output_2d = grad_output.contiguous().reshape(
            -1, grad_output.shape[-1]
        )
        grad_y = F.linear(grad_output_2d, linear_weight.t()).reshape_as(x)
        dx, dgate, dnorm_weight, _, _, y = layer_norm_gated_bwd(
            dy=grad_y,
            x=x,
            g=gate,
            weight=norm_weight,
            bias=None,
            activation="sigmoid",
            eps=ctx.eps,
            mean=mean,
            rstd=rstd,
            is_rms_norm=True,
            x_dtype=x.dtype,
            recompute_output=True,
        )
        if y is None:
            raise RuntimeError("norm-linear reverse did not regenerate output")
        grad_linear_weight = torch.mm(
            grad_output_2d.t(), y.reshape(grad_output_2d.shape[0], -1)
        ).to(ctx.linear_weight_dtype)
        grad_linear_bias = (
            None
            if ctx.linear_bias_dtype is None
            else grad_output_2d.float().sum(dim=0).to(ctx.linear_bias_dtype)
        )
        return (
            dx.reshape(ctx.shape),
            dgate.reshape(ctx.shape),
            dnorm_weight,
            grad_linear_weight,
            grad_linear_bias,
            None,
            None,
            None,
        )


class RMSNormGated(nn.Module):
    """Standard sigmoid-gated RMSNorm backed by FLA's fused owner."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.activation = "sigmoid"
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cuda":
            return rms_norm_gated_reference(x, gate, self.weight, self.eps)
        return rms_norm_gated(
            x,
            gate,
            self.weight,
            None,
            activation=self.activation,
            eps=self.eps,
        )

    def forward_linear(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
        linear: nn.Linear,
    ) -> torch.Tensor:
        """Apply the output projection without saving the normalized panel."""
        if x.device.type != "cuda":
            normalized = rms_norm_gated_reference(
                x,
                gate,
                self.weight,
                self.eps,
            )
            return linear(normalized.reshape(*x.shape[:-2], -1))
        return _RMSNormGatedLinearFunction.apply(
            x,
            gate,
            self.weight,
            linear.weight,
            linear.bias,
            getattr(linear, "_bf16_shadow_weight", None),
            getattr(linear, "_bf16_shadow_bias", None),
            self.eps,
        )


__all__ = ["RMSNormGated", "rms_norm_gated_reference"]
