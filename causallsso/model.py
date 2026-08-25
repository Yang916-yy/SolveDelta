from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from .config import SolveDeltaConfig
from .ops.chunk_wy import chunk_wy_solvedelta
from .reference import SolveDeltaState, solvedelta_reference


class SolveDeltaLayerState(NamedTuple):
    operator: SolveDeltaState
    conv_q: torch.Tensor | None
    conv_k: torch.Tensor | None
    conv_v: torch.Tensor | None


class _CausalShortConvolution(nn.Conv1d):
    """Fixed GDN2-style depthwise conv4 with shared reference/native weights."""

    def __init__(self, width: int) -> None:
        super().__init__(
            in_channels=width,
            out_channels=width,
            kernel_size=4,
            groups=width,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        state_width = self.kernel_size[0] - 1
        expected_cache = (x.shape[0], x.shape[-1], state_width)
        if cache is not None and cache.shape != expected_cache:
            raise ValueError(
                f"short-convolution cache must have shape {expected_cache}, "
                f"got {tuple(cache.shape)}"
            )
        if x.device.type == "cuda":
            try:
                from causal_conv1d import causal_conv1d_fn
            except ImportError as error:
                raise RuntimeError(
                    "CUDA short convolution requires causal-conv1d>=1.7.0"
                ) from error
            transposed = x.transpose(1, 2)
            if transposed.stride(0) % 8 or transposed.stride(2) % 8:
                transposed = x.contiguous().transpose(1, 2)
            if cache is not None and cache.stride(1) != 1:
                cache = cache.transpose(1, 2).contiguous().transpose(1, 2)
            result = causal_conv1d_fn(
                x=transposed,
                weight=self.weight[:, 0, :],
                initial_states=cache,
                return_final_states=output_final_state,
                activation="silu",
            )
            if output_final_state:
                output, final_state = result
                return output.transpose(1, 2), final_state
            return result.transpose(1, 2), None

        batch, length, width = x.shape
        state = (
            x.new_zeros(batch, width, state_width) if cache is None else cache
        )
        outputs = []
        for token in range(length):
            window = torch.cat(
                (state, x[:, token].unsqueeze(-1)), dim=-1
            )
            outputs.append(
                F.silu(
                    torch.einsum(
                        "bdw,dw->bd", window, self.weight[:, 0, :]
                    )
                )
            )
            state = window[..., 1:]
        return torch.stack(outputs, dim=1), state if output_final_state else None


class SolveDelta(nn.Module):
    """Projection owner and dispatcher for the single SolveDelta contract."""

    def __init__(self, config: SolveDeltaConfig) -> None:
        super().__init__()
        self.config = config
        h = config.num_heads
        r = config.resolved_head_k_dim
        v = config.resolved_head_v_dim
        k_edits = config.num_edits

        self.projection_sizes = (
            h * r,                 # geometry feature
            h * r,                 # geometry drive
            h * r,                 # query
            h * k_edits * r,       # edit keys
            h * k_edits * v,       # edit values
            h * k_edits * r,       # erase logits
            h * k_edits * v,       # write logits
            h,                     # geometry decay logits
            h * r,                 # associative decay logits
        )
        self.in_proj = nn.Linear(
            config.hidden_size,
            sum(self.projection_sizes),
            bias=config.bias,
        )
        self.output_proj = nn.Linear(h * v, config.hidden_size, bias=config.bias)

        if config.use_short_conv:
            self.q_conv1d = _CausalShortConvolution(h * r)
            self.k_conv1d = _CausalShortConvolution(h * k_edits * r)
            self.v_conv1d = _CausalShortConvolution(h * k_edits * v)

        self.geometry_log_rate = nn.Parameter(torch.zeros(h, dtype=torch.float32))
        self.associative_log_rate = nn.Parameter(torch.zeros(h, r, dtype=torch.float32))
        self.geometry_strength_logit = nn.Parameter(torch.full((h,), -2.0, dtype=torch.float32))
        self.geometry_decay_bias = nn.Parameter(torch.zeros(h, dtype=torch.float32))
        self.associative_decay_bias = nn.Parameter(torch.zeros(h, r, dtype=torch.float32))

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
            raise ValueError("hidden_states must have shape [B, T, hidden_size]")
        batch, length, width = hidden_states.shape
        if width != self.config.hidden_size:
            raise ValueError(
                f"expected hidden_size={self.config.hidden_size}, got {width}"
            )
        h_count = self.config.num_heads
        r = self.config.resolved_head_k_dim
        v_dim = self.config.resolved_head_v_dim
        edits = self.config.num_edits
        use_native = (
            hidden_states.device.type == "cuda"
            and hidden_states.dtype == torch.bfloat16
        )
        if use_native and (r != 128 or edits != 1):
            raise NotImplementedError(
                "the native BF16 SolveDelta path requires r=128 and K=1"
            )
        if (
            use_native
            and self.config.use_short_conv
            and (h_count * v_dim) % 8
        ):
            raise NotImplementedError(
                "the native CUDA conv4 path requires the projected value "
                "width H*d_v to be divisible by 8"
            )
        if use_native and (valid_mask is not None or reset_mask is not None):
            raise NotImplementedError(
                "the native BF16 SolveDelta path does not yet support "
                "valid_mask or reset_mask"
            )

        def view_head(x: torch.Tensor, dim: int) -> torch.Tensor:
            return x.view(batch, length, h_count, dim)

        projected = self.in_proj(hidden_states).split(self.projection_sizes, dim=-1)
        (
            u_raw, h_raw, q_raw, key_raw, value_raw,
            erase_raw, write_raw, geometry_raw, associative_raw,
        ) = projected

        operator_initial = initial_state.operator if initial_state is not None else None
        conv_q = conv_k = conv_v = None
        if self.config.use_short_conv:
            initial_conv_q = initial_state.conv_q if initial_state is not None else None
            initial_conv_k = initial_state.conv_k if initial_state is not None else None
            initial_conv_v = initial_state.conv_v if initial_state is not None else None
            q_raw, conv_q = self._apply_short_conv(
                self.q_conv1d, q_raw, initial_conv_q,
                valid_mask, reset_mask, return_final_state,
            )
            key_raw, conv_k = self._apply_short_conv(
                self.k_conv1d, key_raw, initial_conv_k,
                valid_mask, reset_mask, return_final_state,
            )
            value_raw, conv_v = self._apply_short_conv(
                self.v_conv1d, value_raw, initial_conv_v,
                valid_mask, reset_mask, return_final_state,
            )
        else:
            q_raw = F.silu(q_raw)
            key_raw = F.silu(key_raw)
            value_raw = F.silu(value_raw)

        u = view_head(u_raw, r)
        h = view_head(h_raw, r)
        q = view_head(q_raw, r)
        keys = key_raw.view(batch, length, h_count, edits, r)
        values = value_raw.view(batch, length, h_count, edits, v_dim)
        if use_native:
            from .ops.fused_gates import fused_native_solvedelta_gates

            u = u.to(torch.bfloat16)
            h = h.to(torch.bfloat16)
            q = q.to(torch.bfloat16)
            keys = keys.to(torch.bfloat16)
            values = values.to(torch.bfloat16)
            erase, write, geometry_log_decay, associative_log_decay = (
                fused_native_solvedelta_gates(
                    erase_raw.view(batch, length, h_count, edits, r),
                    write_raw.view(
                        batch, length, h_count, edits, v_dim
                    ),
                    geometry_raw,
                    associative_raw.view(batch, length, h_count, r),
                    self.geometry_log_rate,
                    self.associative_log_rate,
                    self.geometry_decay_bias,
                    self.associative_decay_bias,
                )
            )
            strength = torch.sigmoid(self.geometry_strength_logit.float())
        else:
            erase = 2.0 * torch.sigmoid(
                erase_raw.view(batch, length, h_count, edits, r)
            )
            write = 2.0 * torch.sigmoid(
                write_raw.view(batch, length, h_count, edits, v_dim)
            )
            geometry_log_decay = -torch.exp(self.geometry_log_rate).view(
                1, 1, h_count
            )
            geometry_log_decay = geometry_log_decay * F.softplus(
                geometry_raw
                + self.geometry_decay_bias.view(1, 1, h_count)
            )
            associative_raw = view_head(associative_raw, r)
            associative_log_decay = -torch.exp(
                self.associative_log_rate
            ).view(1, 1, h_count, r)
            associative_log_decay = associative_log_decay * F.softplus(
                associative_raw
                + self.associative_decay_bias.view(1, 1, h_count, r)
            )
            strength = torch.sigmoid(self.geometry_strength_logit)
        if not associative_decay_enabled:
            associative_log_decay = torch.zeros_like(associative_log_decay)
        if not geometry_enabled:
            strength = torch.zeros_like(strength)

        if use_native:
            outputs, final_state = chunk_wy_solvedelta(
                u,
                h,
                q,
                keys,
                values,
                geometry_log_decay,
                associative_log_decay,
                erase,
                write,
                strength,
                initial_state=operator_initial,
            )
        else:
            reference_dtype = u.dtype
            outputs, final_state = solvedelta_reference(
                u,
                h,
                q,
                keys,
                values,
                geometry_log_decay.to(reference_dtype),
                associative_log_decay.to(reference_dtype),
                erase.to(reference_dtype),
                write.to(reference_dtype),
                strength.to(reference_dtype),
                initial_state=operator_initial,
                valid_mask=valid_mask,
                reset_mask=reset_mask,
            )
        outputs = outputs.reshape(batch, length, h_count * v_dim)
        outputs = self.output_proj(outputs.to(hidden_states.dtype))
        if not return_final_state:
            return outputs
        return outputs, SolveDeltaLayerState(final_state, conv_q, conv_k, conv_v)

    @staticmethod
    def _apply_short_conv(
        convolution: nn.Module,
        x: torch.Tensor,
        initial_cache: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
        reset_mask: torch.Tensor | None,
        output_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply GDN2-style depthwise causal convolution with exact state masks."""
        if valid_mask is None and reset_mask is None:
            return convolution(
                x=x,
                cache=initial_cache,
                output_final_state=output_final_state,
            )

        batch, length, width = x.shape
        kernel_size = convolution.kernel_size[0]
        state_width = kernel_size - 1
        if initial_cache is None:
            cache = x.new_zeros(batch, width, state_width)
        else:
            expected = (batch, width, state_width)
            if initial_cache.shape != expected:
                raise ValueError(
                    f"short-convolution cache must have shape {expected}, "
                    f"got {tuple(initial_cache.shape)}"
                )
            cache = initial_cache
        if valid_mask is None:
            valid_mask = torch.ones(batch, length, dtype=torch.bool, device=x.device)
        if reset_mask is None:
            reset_mask = torch.zeros(batch, length, dtype=torch.bool, device=x.device)

        weight = convolution.weight[:, 0, :]
        bias = convolution.bias
        outputs = []
        for token in range(length):
            valid = valid_mask[:, token]
            reset = reset_mask[:, token] & valid
            cache = torch.where(reset[:, None, None], torch.zeros_like(cache), cache)
            window = torch.cat(
                (cache, x[:, token].unsqueeze(-1)), dim=-1
            )
            preactivation = torch.einsum("bdw,dw->bd", window, weight)
            if bias is not None:
                preactivation = preactivation + bias
            output = F.silu(preactivation)
            outputs.append(torch.where(valid[:, None], output, torch.zeros_like(output)))
            candidate = window[..., 1:]
            cache = torch.where(valid[:, None, None], candidate, cache)
        return torch.stack(outputs, dim=1), cache if output_final_state else None
