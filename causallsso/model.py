from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from .config import SolveDeltaConfig
from .ops.operator import solvedelta_native
from .ops.radial_norm_gate import RadialRMSNormGated
from .ops.packing import (
    PackedSegments,
    build_packed_segments,
    pack_tokens,
    unpack_tokens,
)
from .reference import SolveDeltaState, solvedelta_reference


class SolveDeltaLayerState(NamedTuple):
    operator: SolveDeltaState
    conv: torch.Tensor | None


class SolveDelta(nn.Module):
    """Projection, packed conv4, operator, and output owner for SolveDelta."""

    def __init__(self, config: SolveDeltaConfig) -> None:
        super().__init__()
        self.config = config
        heads = config.num_heads
        width = config.resolved_head_k_dim
        value_width = config.resolved_head_v_dim
        edits = config.num_edits
        gate_rank = value_width

        self.projection_sizes = (
            heads * width,
            heads * edits * width,
            heads * edits * value_width,
            heads * width,
            heads * width,
            heads * edits * width,
            heads * edits * value_width,
            heads,
            gate_rank,
            gate_rank,
        )
        self.projection_width = sum(self.projection_sizes)
        # causal-conv1d's channel-last path requires the batch and token
        # strides to be multiples of eight. Padding the packed projection row
        # preserves a direct strided prefix view for conv4 and every consumer.
        self.projection_padding = (
            (-self.projection_width) % 8 if config.use_short_conv else 0
        )
        self.in_proj = nn.Linear(
            config.hidden_size,
            self.projection_width + self.projection_padding,
            bias=config.bias,
        )
        self.output_proj = nn.Linear(
            heads * value_width, config.hidden_size, bias=config.bias
        )
        self.decay_proj = nn.Linear(gate_rank, heads * width, bias=False)
        self.output_gate_proj = nn.Linear(
            gate_rank, heads * value_width, bias=True
        )
        self.output_norm = RadialRMSNormGated(
            value_width,
            heads,
            eps=config.norm_eps,
        )
        if config.use_short_conv:
            conv_width = sum(self.projection_sizes[:3])
            self.conv_weight = nn.Parameter(torch.empty(conv_width, 4))
            nn.init.kaiming_uniform_(self.conv_weight, a=5**0.5)

        # The projected scalar is a token-local normalized-LMS write rate.
        self.geometry_write_bias = nn.Parameter(
            torch.full((heads,), -2.0, dtype=torch.float32)
        )

        # Match the mature GDN2/Mamba decay initialization: a positive rate
        # and log-uniform step size, evaluated in FP32 by the gate owner.
        associative_rate = torch.empty(heads, dtype=torch.float32).uniform_(1, 16)
        self.associative_log_rate = nn.Parameter(
            associative_rate.log()
        )
        associative_step = torch.exp(
            torch.rand(heads, width, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min_(1e-4)
        self.associative_decay_bias = nn.Parameter(
            associative_step + torch.log(-torch.expm1(-associative_step))
        )
        for parameter in (
            self.geometry_write_bias,
            self.associative_log_rate,
            self.associative_decay_bias,
        ):
            parameter._no_weight_decay = True

    def _gate_output(
        self,
        output: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        return self.output_norm(output, gate)

    def _packed_conv(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
        reset_mask: torch.Tensor | None,
        return_final_state: bool,
        packed_segments: PackedSegments | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if packed_segments is None and x.device.type == "cuda":
            try:
                from causal_conv1d import causal_conv1d_fn
            except ImportError as error:
                raise RuntimeError(
                    "SolveDelta conv4 requires causal-conv1d>=1.7.0"
                ) from error
            result = causal_conv1d_fn(
                x=x.transpose(1, 2),
                weight=self.conv_weight,
                initial_states=initial_state,
                return_final_states=return_final_state,
                activation="silu",
            )
            if return_final_state:
                output, final_state = result
                return output.transpose(1, 2), final_state
            return result.transpose(1, 2), None

        if packed_segments is not None and x.device.type == "cuda":
            return self._packed_varlen_conv(
                x,
                initial_state,
                packed_segments,
                return_final_state,
            )

        batch, length, channels = x.shape
        if initial_state is None:
            state = x.new_zeros(batch, channels, 3)
        else:
            state = initial_state
        if valid_mask is None:
            valid_mask = torch.ones(batch, length, dtype=torch.bool, device=x.device)
        if reset_mask is None:
            reset_mask = torch.zeros(batch, length, dtype=torch.bool, device=x.device)
        outputs = []
        for token in range(length):
            valid = valid_mask[:, token]
            reset = reset_mask[:, token] & valid
            state = torch.where(reset[:, None, None], torch.zeros_like(state), state)
            window = torch.cat((state, x[:, token, :, None]), dim=-1)
            candidate = F.silu((window * self.conv_weight[None]).sum(dim=-1))
            outputs.append(torch.where(valid[:, None], candidate, torch.zeros_like(candidate)))
            state = torch.where(valid[:, None, None], window[..., 1:], state)
        return torch.stack(outputs, dim=1), state if return_final_state else None

    def _packed_varlen_conv(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor | None,
        plan: PackedSegments,
        return_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from causal_conv1d import causal_conv1d_fn
        from fla.layers.utils import index_first_axis, index_put_first_axis
        from fla.ops.utils import prepare_sequence_ids

        batch, _, channels = x.shape
        state_width = self.conv_weight.shape[-1] - 1
        if state_width != 3:
            raise RuntimeError("SolveDelta production convolution must be conv4")
        if initial_state is None:
            base_state = x.new_zeros(batch, channels, state_width)
        else:
            if initial_state.shape != (batch, channels, state_width):
                raise ValueError("conv initial state must have shape [B,D,3]")
            if initial_state.device != x.device or initial_state.dtype != x.dtype:
                raise TypeError("conv initial state must match activation device and dtype")
            base_state = initial_state
        if plan.num_tokens == 0:
            return torch.zeros_like(x), base_state if return_final_state else None

        actual = pack_tokens(x, plan).squeeze(0)
        segment_lengths_cpu = torch.diff(plan.cu_seqlens_cpu)
        prefix_lengths_cpu = torch.zeros_like(segment_lengths_cpu)
        if initial_state is not None:
            prefix_lengths_cpu = plan.use_initial_cpu.long() * state_width
        augmented_lengths_cpu = segment_lengths_cpu + prefix_lengths_cpu
        augmented_cu_cpu = F.pad(
            augmented_lengths_cpu.cumsum(0), (1, 0), value=0
        )
        augmented_cu = augmented_cu_cpu.to(x.device)
        prefix_lengths = prefix_lengths_cpu.to(x.device)
        packed_position = torch.arange(
            plan.num_tokens, dtype=torch.long, device=x.device
        ) - plan.cu_seqlens[plan.segment_ids]
        actual_indices = (
            augmented_cu[plan.segment_ids]
            + prefix_lengths[plan.segment_ids]
            + packed_position
        )

        values = [actual]
        target_indices = [actual_indices]
        if initial_state is not None:
            initial_segments_cpu = torch.nonzero(
                plan.use_initial_cpu, as_tuple=False
            ).flatten()
            if initial_segments_cpu.shape[0] != 0:
                initial_segments = initial_segments_cpu.to(x.device)
                initial_batches = plan.segment_batch_cpu[
                    initial_segments_cpu
                ].to(x.device)
                initial_values = index_first_axis(
                    initial_state, initial_batches
                ).transpose(1, 2).reshape(-1, channels)
                initial_indices = (
                    augmented_cu[initial_segments, None]
                    + torch.arange(state_width, device=x.device)[None, :]
                ).reshape(-1)
                values.append(initial_values)
                target_indices.append(initial_indices)

        total_augmented = int(augmented_cu_cpu[-1].item())
        augmented = index_put_first_axis(
            torch.cat(values, dim=0),
            torch.cat(target_indices, dim=0),
            total_augmented,
        )
        sequence_ids = prepare_sequence_ids(
            augmented_cu,
            cu_seqlens_cpu=augmented_cu_cpu,
        ).to(torch.int32).unsqueeze(0)
        convolved = causal_conv1d_fn(
            x=augmented.transpose(0, 1).unsqueeze(0),
            weight=self.conv_weight,
            seq_idx=sequence_ids,
            activation="silu",
        ).transpose(1, 2)
        output = unpack_tokens(
            index_first_axis(convolved.squeeze(0), actual_indices).unsqueeze(0),
            plan,
        )
        if not return_final_state:
            return output, None

        has_segment = plan.last_segment >= 0
        selected_segment = plan.last_segment.clamp_min(0)
        segment_start = augmented_cu[selected_segment]
        segment_end = augmented_cu[selected_segment + 1]
        state_indices = segment_end[:, None] + torch.arange(
            -state_width, 0, dtype=torch.long, device=x.device
        )[None, :]
        state_valid = state_indices >= segment_start[:, None]
        gathered = augmented.index_select(
            0, state_indices.clamp(0, total_augmented - 1).reshape(-1)
        ).view(batch, state_width, channels)
        gathered = torch.where(
            state_valid[:, :, None], gathered, torch.zeros_like(gathered)
        ).transpose(1, 2)
        final_state = torch.where(
            has_segment[:, None, None], gathered, base_state
        )
        return output, final_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: SolveDeltaLayerState | None = None,
        valid_mask: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
        geometry_enabled: bool = True,
        associative_decay_enabled: bool = True,
        return_final_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SolveDeltaLayerState]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B,T,hidden_size]")
        batch, length, hidden_width = hidden_states.shape
        if hidden_width != self.config.hidden_size:
            raise ValueError("hidden state width does not match the configuration")
        heads = self.config.num_heads
        width = self.config.resolved_head_k_dim
        value_width = self.config.resolved_head_v_dim
        edits = self.config.num_edits
        packed_projection = self.in_proj(hidden_states)
        native_inputs = (
            packed_projection.device.type == "cuda"
            and packed_projection.dtype == torch.bfloat16
        )
        use_native = native_inputs and valid_mask is None and reset_mask is None
        packed_segments = None
        if native_inputs and (valid_mask is not None or reset_mask is not None):
            if valid_mask is None:
                valid_mask = torch.ones(
                    batch, length, dtype=torch.bool, device=hidden_states.device
                )
            if reset_mask is None:
                reset_mask = torch.zeros(
                    batch, length, dtype=torch.bool, device=hidden_states.device
                )
            if (
                valid_mask.device != hidden_states.device
                or reset_mask.device != hidden_states.device
            ):
                raise ValueError("masks must share the activation device")
            packed_segments = build_packed_segments(valid_mask, reset_mask)

        conv_state = initial_state.conv if initial_state is not None else None
        if self.config.use_short_conv:
            qkv_width = sum(self.projection_sizes[:3])
            qkv_projection, remaining_projection = packed_projection[
                ..., : self.projection_width
            ].split((qkv_width, self.projection_width - qkv_width), dim=-1)
            qkv, final_conv = self._packed_conv(
                qkv_projection,
                conv_state,
                valid_mask,
                reset_mask,
                return_final_state,
                packed_segments,
            )
            q_raw, key_raw, value_raw = qkv.split(self.projection_sizes[:3], dim=-1)
            (
                u_raw,
                h_raw,
                erase_raw,
                write_raw,
                geometry_raw,
                decay_hidden,
                output_gate_hidden,
            ) = remaining_projection.split(self.projection_sizes[3:], dim=-1)
        else:
            (
                q_raw,
                key_raw,
                value_raw,
                u_raw,
                h_raw,
                erase_raw,
                write_raw,
                geometry_raw,
                decay_hidden,
                output_gate_hidden,
            ) = packed_projection[..., : self.projection_width].split(
                self.projection_sizes, dim=-1
            )
            q_raw, key_raw, value_raw = map(F.silu, (q_raw, key_raw, value_raw))
            final_conv = None

        def heads_view(x: torch.Tensor, size: int) -> torch.Tensor:
            return x.view(batch, length, heads, size)

        u = heads_view(u_raw, width)
        h = heads_view(h_raw, width)
        q = heads_view(q_raw, width)
        keys = key_raw.view(batch, length, heads, edits, width)
        values = value_raw.view(batch, length, heads, edits, value_width)
        associative_raw = self.decay_proj(decay_hidden).view(
            batch, length, heads, width
        )
        output_gate = self.output_gate_proj(output_gate_hidden).view(
            batch, length, heads, value_width
        )
        if native_inputs:
            from fla.ops.kda.gate import fused_kda_gate

            associative_log_decay = fused_kda_gate(
                associative_raw,
                self.associative_log_rate.float(),
                self.associative_decay_bias.float().flatten(),
                output_dtype=torch.float32,
            )
        else:
            associative_log_decay = -torch.exp(
                self.associative_log_rate.float()
            ).view(1, 1, heads, 1)
            associative_log_decay = associative_log_decay * F.softplus(
                associative_raw.float()
                + self.associative_decay_bias.float().view(1, 1, heads, width)
            )
        geometry_write = torch.sigmoid(
            geometry_raw.float()
            + self.geometry_write_bias.float().view(1, 1, heads)
        )
        if not geometry_enabled:
            geometry_write = torch.zeros_like(geometry_write)
        if not associative_decay_enabled:
            associative_log_decay = torch.zeros_like(associative_log_decay)

        operator_initial = initial_state.operator if initial_state is not None else None
        if use_native:
            output, operator_state = solvedelta_native(
                u.to(torch.bfloat16),
                h.to(torch.bfloat16),
                q.to(torch.bfloat16),
                keys.to(torch.bfloat16),
                values.to(torch.bfloat16),
                associative_log_decay.float(),
                erase_raw.view(batch, length, heads, edits, width),
                write_raw.view(batch, length, heads, edits, value_width),
                geometry_write.float(),
                initial_state=operator_initial,
                return_final_state=return_final_state,
            )
        else:
            reference_dtype = (
                torch.float64
                if hidden_states.dtype == torch.float64
                else torch.float32
            )
            erase = torch.sigmoid(
                erase_raw.float().view(batch, length, heads, edits, width)
            )
            write = torch.sigmoid(
                write_raw.float().view(batch, length, heads, edits, value_width)
            )
            reference_initial = (
                None
                if operator_initial is None
                else SolveDeltaState(
                    *(state.to(reference_dtype) for state in operator_initial)
                )
            )
            output, operator_state = solvedelta_reference(
                u.to(reference_dtype),
                h.to(reference_dtype),
                q.to(reference_dtype),
                keys.to(reference_dtype),
                values.to(reference_dtype),
                associative_log_decay.to(reference_dtype),
                erase.to(reference_dtype),
                write.to(reference_dtype),
                geometry_write.to(reference_dtype),
                initial_state=reference_initial,
                valid_mask=valid_mask,
                reset_mask=reset_mask,
            )
            if not return_final_state:
                operator_state = None
        output = self._gate_output(output, output_gate.to(output.dtype))
        output = self.output_proj(
            output.reshape(batch, length, heads * value_width).to(
                hidden_states.dtype
            )
        )
        if not return_final_state:
            return output
        if operator_state is None:
            raise RuntimeError("operator did not return the requested final state")
        return output, SolveDeltaLayerState(operator_state, final_conv)


__all__ = ["SolveDelta", "SolveDeltaLayerState"]
