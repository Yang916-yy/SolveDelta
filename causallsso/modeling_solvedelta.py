from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from fla.layers.attn import Attention
from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.models.hybrid import get_hybrid_attention_spec
from fla.models.utils import Cache, FLAUnsupportedCacheGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP
from fla.modules.l2warp import l2_warp

from .config import SolveDeltaConfig
from .model import SolveDelta, SolveDeltaLayerState
from .reference import SolveDeltaState


try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer


class SolveDeltaMLP(GatedMLP):
    """FLA GatedMLP with its algebraically identical CPU oracle path."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.device.type != "cpu":
            return super().forward(x, **kwargs)
        gate = self.gate_proj(x)
        value = self.up_proj(x)
        if self.hidden_act == "swish":
            activated = F.silu(gate)
        else:
            sigmoid = torch.sigmoid(gate)
            positive = gate > 0
            safe = torch.where(positive, gate, torch.ones_like(gate))
            exponent = self.powglu_power / (torch.sqrt(safe) + 1.0)
            activated = torch.where(
                positive,
                torch.exp(exponent * torch.log(safe)) * sigmoid,
                gate * sigmoid,
            )
        return self.down_proj(activated * value)


def _state_from_cache(last_state: dict[str, Any] | None) -> SolveDeltaLayerState | None:
    if last_state is None:
        return None
    recurrent_state = last_state.get("recurrent_state")
    if recurrent_state is None:
        return None
    if not isinstance(recurrent_state, (tuple, list)) or len(recurrent_state) != 2:
        raise ValueError("SolveDelta cache recurrent_state must contain (predictor,S)")
    return SolveDeltaLayerState(
        operator=SolveDeltaState(*recurrent_state),
        conv=last_state.get("conv_state"),
    )


def _segment_reset_mask(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.shape[0] != 1:
        raise ValueError("explicit cu_seqlens require packed hidden states [1,T,D]")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be one-dimensional with at least two entries")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must have int32 or int64 dtype")
    starts = cu_seqlens[:-1].to(device=hidden_states.device, dtype=torch.long)
    reset_mask = torch.zeros(
        (1, hidden_states.shape[1]), dtype=torch.bool, device=hidden_states.device
    )
    return reset_mask.scatter(1, starts.unsqueeze(0), True)


class SolveDeltaCacheAdapter(nn.Module):
    """FLA layer/cache ABI around the selected SolveDelta mixer."""

    def __init__(self, config: SolveDeltaConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.mixer = SolveDelta(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, Cache | None]:
        del output_attentions
        batch, length, _ = hidden_states.shape
        last_state = get_layer_cache(self, past_key_values)
        initial_state = _state_from_cache(last_state)

        valid_mask = None
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != batch:
                raise ValueError("attention_mask must have shape [B,T]")
            if attention_mask.shape[1] < length:
                raise ValueError("attention_mask is shorter than the current input")
            valid_mask = attention_mask[:, -length:].to(torch.bool)

        reset_mask = kwargs.get("solvedelta_reset_mask")
        cu_seqlens = kwargs.get("cu_seqlens")
        if reset_mask is None and cu_seqlens is not None:
            reset_mask = _segment_reset_mask(hidden_states, cu_seqlens)
        if reset_mask is not None:
            if reset_mask.shape != (batch, length) or reset_mask.dtype != torch.bool:
                raise ValueError("solvedelta_reset_mask must be bool [B,T]")
            if initial_state is not None and cu_seqlens is not None:
                reset_mask = reset_mask.clone()
                reset_mask[0, 0] = False

        cache_requested = bool(use_cache and past_key_values is not None)
        result = self.mixer(
            hidden_states,
            initial_state=initial_state,
            valid_mask=valid_mask,
            reset_mask=reset_mask,
            return_final_state=cache_requested,
        )
        if cache_requested:
            output, final_state = result
            update_layer_cache(
                self,
                past_key_values,
                recurrent_state=tuple(final_state.operator),
                conv_state=final_state.conv,
                offset=length,
            )
        else:
            output = result
        return output, None, past_key_values


class SolveDeltaBlock(GradientCheckpointingLayer):
    def __init__(self, config: SolveDeltaConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        norm = RMSNorm if config.fuse_norm else nn.RMSNorm
        self.mixer_norm = norm(config.hidden_size, eps=config.norm_eps)

        attn_spec = get_hybrid_attention_spec(config.attn, layer_idx=layer_idx)
        if attn_spec is None:
            self.mixer = SolveDeltaCacheAdapter(config, layer_idx)
        else:
            self.mixer = Attention(
                hidden_size=config.hidden_size,
                num_heads=attn_spec["num_heads"],
                num_kv_heads=attn_spec["num_kv_heads"],
                qkv_bias=attn_spec["qkv_bias"],
                window_size=attn_spec["window_size"],
                rope_theta=attn_spec["rope_theta"],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx,
            )

        self.mlp_norm = norm(config.hidden_size, eps=config.norm_eps)
        self.mlp = SolveDeltaMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        residual = hidden_states
        hidden_states = self.mixer_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, attentions, past_key_values


class SolveDeltaPreTrainedModel(PreTrainedModel):
    config_class = SolveDeltaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SolveDeltaBlock"]
    _supports_cache_class = True

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()


class SolveDeltaModel(SolveDeltaPreTrainedModel):
    def __init__(self, config: SolveDeltaConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            SolveDeltaBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        norm = RMSNorm if config.fuse_norm else nn.RMSNorm
        self.norm = norm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Cache | list[dict[str, Any]] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("specify either input_ids or inputs_embeds, not both")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds is required")
        if output_attentions:
            warnings.warn(
                "SolveDelta does not expose attention weights; disabling output_attentions"
            )
            output_attentions = False
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.use_cache if not self.training else False)
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        hidden_states = (
            self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        )
        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)
        elif not use_cache:
            past_key_values = None

        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is not None and "solvedelta_reset_mask" not in kwargs:
            kwargs["solvedelta_reset_mask"] = _segment_reset_mask(
                hidden_states, cu_seqlens
            )

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attentions, past_key_values = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )
            if output_attentions:
                all_attentions += (attentions,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(
                value
                for value in (
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_attentions,
                )
                if value is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class SolveDeltaForCausalLM(
    SolveDeltaPreTrainedModel, FLAUnsupportedCacheGenerationMixin
):
    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    def __init__(self, config: SolveDeltaConfig) -> None:
        super().__init__(config)
        self.model = SolveDeltaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Cache | list[dict[str, Any]] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = outputs[0]

        loss = None
        logits = None
        keep_all_logits = labels is not None or logits_to_keep in (None, 0)
        logits_input = (
            hidden_states if keep_all_logits else hidden_states[:, -logits_to_keep:]
        )
        if not self.config.fuse_linear_cross_entropy or labels is None:
            logits = self.lm_head(logits_input)
        if labels is not None:
            if self.criterion is not None:
                criterion = self.criterion
            elif self.config.fuse_linear_cross_entropy:
                criterion = FusedLinearCrossEntropyLoss(
                    use_l2warp=self.config.use_l2warp
                )
            elif self.config.fuse_cross_entropy:
                criterion = FusedCrossEntropyLoss(inplace_backward=True)
            else:
                criterion = nn.CrossEntropyLoss()
            labels = labels.to(hidden_states.device)
            labels = torch.cat(
                (
                    labels[..., 1:],
                    torch.full_like(labels[:, :1], criterion.ignore_index),
                ),
                dim=1,
            )
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(
                    hidden_states, labels, self.lm_head.weight, self.lm_head.bias
                )
            else:
                loss = criterion(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
                )
                if self.config.use_l2warp:
                    loss = l2_warp(loss, logits)

        if not return_dict:
            result = (logits,) + outputs[1:]
            return ((loss,) + result) if loss is not None else result
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "SolveDeltaBlock",
    "SolveDeltaCacheAdapter",
    "SolveDeltaForCausalLM",
    "SolveDeltaModel",
    "SolveDeltaPreTrainedModel",
]
