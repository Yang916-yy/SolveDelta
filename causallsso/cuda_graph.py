"""Fixed-shape CUDA Graph training for the SolveDelta causal LM."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .modeling_solvedelta import SolveDeltaForCausalLM


class _CausalLMLoss(nn.Module):
    def __init__(self, model: SolveDeltaForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        loss = output[0]
        if loss is None:
            raise RuntimeError("the graphed CausalLM forward did not produce a loss")
        return loss


class SolveDeltaGraphedTrainingStep:
    """Capture fixed-shape CausalLM loss forward and backward as CUDA Graphs.

    Calling this object replays the forward graph and returns a scalar loss.
    Calling ``loss.backward()`` replays the matching backward graph. The
    optimizer intentionally remains outside the graph so its implementation,
    gradient accumulation, clipping, and distributed ownership stay under the
    training loop's control.

    Capture is limited to the optimized dense training surface: CUDA token IDs
    and labels with one fixed shape, no masks/resets, no recurrent cache, and
    BF16 autocast. Build a separate instance for every batch/sequence shape.
    Run backward before replaying the same instance again because graph output
    storage is reused.
    """

    def __init__(
        self,
        model: SolveDeltaForCausalLM,
        sample_input_ids: torch.Tensor,
        sample_labels: torch.Tensor,
        *,
        num_warmup_iters: int = 3,
        pool: Any | None = None,
    ) -> None:
        if not isinstance(model, SolveDeltaForCausalLM):
            raise TypeError("model must be a SolveDeltaForCausalLM")
        if not model.training:
            raise ValueError("CUDA Graph training capture requires model.train()")
        if isinstance(num_warmup_iters, bool) or not isinstance(
            num_warmup_iters, int
        ):
            raise TypeError("num_warmup_iters must be an int")
        if num_warmup_iters < 1:
            raise ValueError("num_warmup_iters must be positive")
        if getattr(model.model, "gradient_checkpointing", False):
            raise ValueError("disable gradient checkpointing before CUDA Graph capture")
        if model.config.fuse_linear_cross_entropy:
            raise ValueError(
                "FLA fused linear cross entropy performs a capture-unsafe host "
                "reduction; disable fuse_linear_cross_entropy for CUDA Graph "
                "training"
            )

        self._validate_pair(sample_input_ids, sample_labels)
        parameter_devices = {parameter.device for parameter in model.parameters()}
        if parameter_devices != {sample_input_ids.device}:
            raise ValueError(
                "all model parameters and samples must share one CUDA device"
            )
        for module in model.modules():
            if (
                module._forward_hooks
                or module._forward_pre_hooks
                or module._backward_hooks
            ):
                raise ValueError("remove module hooks before CUDA Graph capture")

        self.model = model
        self.device = sample_input_ids.device
        self.batch_size, self.sequence_length = sample_input_ids.shape
        self._input_dtype = sample_input_ids.dtype
        self._label_dtype = sample_labels.dtype

        # Own the sample storage used as the graph's static input surface.
        static_input_ids = sample_input_ids.detach().contiguous().clone()
        static_labels = sample_labels.detach().contiguous().clone()
        loss_module = _CausalLMLoss(model).train()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            cache_enabled=False,
        ):
            self._graphed_loss = torch.cuda.make_graphed_callables(
                loss_module,
                (static_input_ids, static_labels),
                num_warmup_iters=num_warmup_iters,
                allow_unused_input=False,
                pool=pool,
            )

    @staticmethod
    def _validate_pair(
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        if input_ids.device.type != "cuda" or labels.device != input_ids.device:
            raise ValueError("input_ids and labels must share one CUDA device")
        if input_ids.ndim != 2 or labels.shape != input_ids.shape:
            raise ValueError("input_ids and labels must have the same [B,T] shape")
        if input_ids.dtype != torch.long or labels.dtype != torch.long:
            raise TypeError("input_ids and labels must have torch.long dtype")
        if input_ids.requires_grad or labels.requires_grad:
            raise ValueError("token IDs and labels must not require gradients")

    def __call__(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_pair(input_ids, labels)
        expected_shape = (self.batch_size, self.sequence_length)
        if input_ids.shape != expected_shape:
            raise ValueError(
                f"this CUDA Graph requires input shape {expected_shape}, "
                f"got {tuple(input_ids.shape)}"
            )
        if (
            input_ids.dtype != self._input_dtype
            or labels.dtype != self._label_dtype
        ):
            raise TypeError("input dtypes must match the captured sample dtypes")
        if input_ids.device != self.device:
            raise ValueError("input device must match the captured CUDA device")
        if not self.model.training:
            raise RuntimeError("the captured model must remain in training mode")
        return self._graphed_loss(input_ids, labels)


__all__ = ["SolveDeltaGraphedTrainingStep"]
