"""Fixed-shape CUDA Graph training for the SolveDelta causal LM."""

from __future__ import annotations

import threading
import warnings
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .bf16_shadow import BF16ShadowWeights
from .graph_linear_cross_entropy import fixed_dense_linear_cross_entropy
from .modeling_solvedelta import SolveDeltaForCausalLM


_CAPTURE_LOCK = threading.Lock()
_ACCUMULATE_GRAD_WARNING = (
    "The AccumulateGrad node's stream does not match the stream of the node "
    "that produced the incoming gradient.*"
)


class _CausalLMLoss(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.model.config.fuse_linear_cross_entropy:
            outputs = self.model.model(
                input_ids=input_ids,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )
            hidden_states = outputs[0]
            shifted = torch.cat(
                (
                    labels[..., 1:],
                    torch.full_like(labels[:, :1], -100),
                ),
                dim=1,
            )
            return fixed_dense_linear_cross_entropy(
                hidden_states.reshape(-1, hidden_states.shape[-1]),
                shifted.reshape(-1),
                self.model.lm_head.weight,
                self.model.lm_head.bias,
                total=labels.shape[0] * (labels.shape[1] - 1),
                use_l2warp=self.model.config.use_l2warp,
            )
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


def _capture_loss(
    loss_module: _CausalLMLoss,
    static_input_ids: torch.Tensor,
    static_labels: torch.Tensor,
    *,
    num_warmup_iters: int,
    pool: Any | None,
) -> nn.Module:
    set_override = getattr(
        torch.autograd.graph, "set_override_stale_capture_stream", None
    )
    get_override = getattr(torch._C, "_override_stale_capture_stream", None)
    if set_override is None or get_override is None:
        raise RuntimeError(
            "CUDA Graph training requires a PyTorch build with stale capture "
            "stream override support (validated with PyTorch 2.13+)"
        )

    # make_graphed_callables warms up on a private side stream. Parameters or
    # DDP hooks may retain an AccumulateGrad node created on another stream.
    # PyTorch's scoped override redirects only stale nodes encountered while a
    # producer is actively capturing; ordinary eager stream semantics remain
    # unchanged after this block.
    with _CAPTURE_LOCK:
        previous_override = bool(get_override())
        set_override(True)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=_ACCUMULATE_GRAD_WARNING,
                    category=UserWarning,
                    module=r"torch\.autograd\.graph",
                )
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    cache_enabled=False,
                ):
                    return torch.cuda.make_graphed_callables(
                        loss_module,
                        (static_input_ids, static_labels),
                        num_warmup_iters=num_warmup_iters,
                        allow_unused_input=False,
                        pool=pool,
                    )
        finally:
            set_override(previous_override)


class SolveDeltaGraphedTrainingStep:
    """Capture fixed-shape CausalLM loss forward and backward as CUDA Graphs.

    Calling this object replays the forward graph and returns a scalar loss.
    Calling ``loss.backward()`` replays the matching backward graph. The
    optimizer intentionally remains outside the graph so its implementation,
    gradient accumulation, clipping, and distributed ownership stay under the
    training loop's control.

    With ``distributed=True``, the local loss callable is captured first and
    then wrapped in DistributedDataParallel. DDP's collectives remain outside
    capture, avoiding stale AccumulateGrad streams and keeping communication
    ownership independent of the fixed-shape compute graph.

    Capture is limited to the optimized dense training surface: CUDA token IDs
    and labels with one fixed shape, no masks/resets, no recurrent cache, and
    BF16 autocast. Build a separate instance for every batch/sequence shape.
    Run backward before replaying the same instance again because graph output
    storage is reused.

    Passing ``bf16_shadow_optimizer`` installs nonpersistent BF16 shadows for
    FP32 Linear parameters and refreshes them from that optimizer's post-step
    hook. This is intended for gradient accumulation; factor-one training pays
    the refresh without amortizing it.
    """

    def __init__(
        self,
        model: SolveDeltaForCausalLM,
        sample_input_ids: torch.Tensor,
        sample_labels: torch.Tensor,
        *,
        num_warmup_iters: int = 3,
        pool: Any | None = None,
        distributed: bool = False,
        process_group: Any | None = None,
        bf16_shadow_optimizer: torch.optim.Optimizer | None = None,
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
        if not isinstance(distributed, bool):
            raise TypeError("distributed must be a bool")
        if process_group is not None and not distributed:
            raise ValueError("process_group requires distributed=True")
        if distributed and not dist.is_initialized():
            raise RuntimeError(
                "initialize torch.distributed before distributed graph capture"
            )
        if distributed:
            backend = str(dist.get_backend(process_group)).lower()
            if backend != "nccl":
                raise RuntimeError(
                    "distributed CUDA Graph training requires an NCCL process group"
                )
        if getattr(model.model, "gradient_checkpointing", False):
            raise ValueError("disable gradient checkpointing before CUDA Graph capture")
        self._validate_pair(sample_input_ids, sample_labels)
        if model.config.fuse_linear_cross_entropy:
            if sample_labels.shape[1] < 2:
                raise ValueError(
                    "fixed dense fused-linear loss requires sequence length >= 2"
                )
            if bool(sample_labels.eq(-100).any().item()):
                raise ValueError(
                    "fixed dense fused-linear loss does not accept ignored input labels"
                )
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
        self._replay_stream = torch.cuda.current_stream(self.device)
        self.batch_size, self.sequence_length = sample_input_ids.shape
        self._input_dtype = sample_input_ids.dtype
        self._label_dtype = sample_labels.dtype
        self._bf16_shadows = (
            None
            if bf16_shadow_optimizer is None
            else BF16ShadowWeights(
                model,
                bf16_shadow_optimizer,
                stream=self._replay_stream,
            )
        )

        # AccumulateGrad records the stream on which its node is first created.
        # Keep parameter edges owned by the caller's replay stream so the
        # graphed backward and post-capture DDP hooks do not inherit the private
        # warmup/capture stream used by make_graphed_callables.
        self._gradient_edges = tuple(
            torch.autograd.graph.get_gradient_edge(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        # Own the sample storage used as the graph's static input surface.
        static_input_ids = sample_input_ids.detach().contiguous().clone()
        static_labels = sample_labels.detach().contiguous().clone()
        loss_module = _CausalLMLoss(model).train()
        graphed_loss = _capture_loss(
            loss_module,
            static_input_ids,
            static_labels,
            num_warmup_iters=num_warmup_iters,
            pool=pool,
        )
        self.distributed = distributed
        if distributed:
            device_index = sample_input_ids.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            self._graphed_loss = DistributedDataParallel(
                graphed_loss,
                device_ids=[device_index],
                output_device=device_index,
                forward_sync_buffers=False,
                process_group=process_group,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        else:
            self._graphed_loss = graphed_loss
        if distributed and self._bf16_shadows is not None:
            # DDP initialization may broadcast masters in place. Rebuild the
            # shadows from the synchronized parameters before first replay.
            self._bf16_shadows.refresh()

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
        if (
            torch.cuda.current_stream(self.device).cuda_stream
            != self._replay_stream.cuda_stream
        ):
            raise RuntimeError(
                "CUDA Graph replay must use the stream that constructed this helper"
            )
        if not self.model.training:
            raise RuntimeError("the captured model must remain in training mode")
        if self._bf16_shadows is not None:
            self._bf16_shadows.assert_current()
        return self._graphed_loss(input_ids, labels)

    def refresh_bf16_shadow_weights(self) -> None:
        """Refresh shadows after parameter mutation outside the bound optimizer."""
        if self._bf16_shadows is None:
            raise RuntimeError("this helper does not use BF16 shadow weights")
        self._bf16_shadows.refresh()

    @property
    def bf16_shadow_bytes(self) -> int:
        """Persistent bytes used by optimizer-step BF16 Linear shadows."""
        return 0 if self._bf16_shadows is None else self._bf16_shadows.nbytes

    def no_sync(self) -> Any:
        """Skip DDP gradient reduction for one or more graph replays."""
        if isinstance(self._graphed_loss, DistributedDataParallel):
            return self._graphed_loss.no_sync()
        return nullcontext()


__all__ = ["SolveDeltaGraphedTrainingStep"]
