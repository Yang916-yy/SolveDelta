"""Optimizer-step BF16 shadows for FP32 Linear master parameters."""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class _BF16ShadowLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        master_weight: torch.Tensor,
        master_bias: torch.Tensor | None,
        shadow_weight: torch.Tensor,
        shadow_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_low = x.to(shadow_weight.dtype)
        ctx.save_for_backward(x_low, shadow_weight)
        ctx.input_shape = x.shape
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = master_weight.dtype
        ctx.bias_dtype = None if master_bias is None else master_bias.dtype
        ctx.set_materialize_grads(False)
        return F.linear(x_low, shadow_weight, shadow_bias)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        x, shadow_weight = ctx.saved_tensors
        grad_output = grad_output.to(shadow_weight.dtype).contiguous()
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        grad_input = F.linear(
            grad_output_2d, shadow_weight.t()
        ).reshape(ctx.input_shape).to(ctx.input_dtype)
        grad_weight = torch.mm(grad_output_2d.t(), x_2d).to(ctx.weight_dtype)
        grad_bias = (
            None
            if ctx.bias_dtype is None
            else grad_output_2d.sum(dim=0).to(ctx.bias_dtype)
        )
        return grad_input, grad_weight, grad_bias, None, None


def _shadow_linear_forward(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return _BF16ShadowLinearFunction.apply(
        x,
        module.weight,
        module.bias,
        module._bf16_shadow_weight,
        module._bf16_shadow_bias,
    )


class BF16ShadowWeights:
    """Refresh Linear BF16 shadows once after each optimizer update.

    The original FP32 Parameters remain the optimizer and checkpoint owners.
    Nonpersistent BF16 buffers feed mature PyTorch GEMMs during graph replay,
    while the custom transpose returns gradients to the FP32 masters. The
    bound optimizer post-step hook prevents per-microbatch parameter casts.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        stream: torch.cuda.Stream,
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        self._stream = stream
        self._masters: list[torch.Tensor] = []
        self._shadows: list[torch.Tensor] = []
        self._unique_masters: dict[int, torch.Tensor] = {}

        for module in model.modules():
            if type(module) is not nn.Linear:
                continue
            if hasattr(module, "_bf16_shadow_weight"):
                raise ValueError("model already has BF16 Linear shadow weights")
            if (
                module.weight.device.type != "cuda"
                or module.weight.dtype != torch.float32
            ):
                raise TypeError("BF16 shadows require CUDA FP32 Linear masters")
            module.register_buffer(
                "_bf16_shadow_weight",
                torch.empty_like(module.weight, dtype=torch.bfloat16),
                persistent=False,
            )
            self._append(module.weight, module._bf16_shadow_weight)
            if module.bias is None:
                module.register_buffer("_bf16_shadow_bias", None, persistent=False)
            else:
                if (
                    module.bias.device != module.weight.device
                    or module.bias.dtype != torch.float32
                ):
                    raise TypeError("BF16 shadows require CUDA FP32 Linear biases")
                module.register_buffer(
                    "_bf16_shadow_bias",
                    torch.empty_like(module.bias, dtype=torch.bfloat16),
                    persistent=False,
                )
                self._append(module.bias, module._bf16_shadow_bias)
            module.forward = types.MethodType(_shadow_linear_forward, module)

        if not self._masters:
            raise ValueError("model has no CUDA FP32 Linear parameters to shadow")
        optimizer_parameters = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        missing = [
            parameter
            for parameter in self._unique_masters.values()
            if parameter.requires_grad and id(parameter) not in optimizer_parameters
        ]
        if missing:
            raise ValueError("optimizer does not own every trainable Linear master")

        self._versions: dict[int, int] = {}
        self.refresh()
        self._optimizer_hook = optimizer.register_step_post_hook(self._post_step)

    def _append(self, master: torch.Tensor, shadow: torch.Tensor) -> None:
        self._masters.append(master)
        self._shadows.append(shadow)
        self._unique_masters.setdefault(id(master), master)

    def _validate_stream(self) -> None:
        device = self._masters[0].device
        if torch.cuda.current_stream(device).cuda_stream != self._stream.cuda_stream:
            raise RuntimeError(
                "BF16 shadow refresh must use the CUDA Graph replay stream"
            )

    @torch.no_grad()
    def refresh(self) -> None:
        """Refresh every shadow after an optimizer step or state load."""
        self._validate_stream()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("BF16 shadow refresh must remain outside CUDA capture")
        torch._foreach_copy_(self._shadows, self._masters)
        self._versions = {
            key: parameter._version
            for key, parameter in self._unique_masters.items()
        }

    def _post_step(
        self,
        optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del optimizer, args, kwargs
        self.refresh()

    def assert_current(self) -> None:
        stale = any(
            parameter._version != self._versions[key]
            for key, parameter in self._unique_masters.items()
        )
        if stale:
            raise RuntimeError(
                "BF16 shadow weights are stale; call "
                "refresh_bf16_shadow_weights() after modifying parameters"
            )

    @property
    def nbytes(self) -> int:
        return sum(shadow.numel() * shadow.element_size() for shadow in self._shadows)


__all__ = ["BF16ShadowWeights"]
