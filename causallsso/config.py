from __future__ import annotations

import math

from transformers.configuration_utils import PretrainedConfig


class SolveDeltaConfig(PretrainedConfig):
    """One configuration for both the SolveDelta mixer and its causal LM."""

    model_type = "solvedelta"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        head_k_dim: int | None = None,
        head_v_dim: int | None = None,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_edits: int = 1,
        use_short_conv: bool = True,
        bias: bool = False,
        *,
        num_hidden_layers: int = 24,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        norm_eps: float = 1e-6,
        max_position_embeddings: int = 2048,
        attn: dict | list[dict] | None = None,
        use_cache: bool = True,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        initializer_range: float = 0.02,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_edits = num_edits
        self.use_short_conv = use_short_conv
        self.bias = bias

        self.num_hidden_layers = num_hidden_layers
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range

        self._validate()
        self.attn = self._normalize_hybrid_attention(attn)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _validate(self) -> None:
        for name in (
            "hidden_size",
            "num_heads",
            "num_edits",
            "num_hidden_layers",
            "vocab_size",
            "max_position_embeddings",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_edits != 1:
            raise ValueError(
                "the current SolveDelta production contract requires num_edits=1"
            )
        for name in ("head_k_dim", "head_v_dim", "intermediate_size"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{name} must be an int or None")
                if value <= 0:
                    raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_ratio is not None:
            if isinstance(self.hidden_ratio, bool) or not isinstance(
                self.hidden_ratio, int
            ):
                raise TypeError("hidden_ratio must be an int or None")
            if self.hidden_ratio <= 0:
                raise ValueError("hidden_ratio must be positive")
        for name in ("expand_k", "expand_v", "norm_eps", "initializer_range"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "use_short_conv",
            "bias",
            "use_cache",
            "fuse_norm",
            "fuse_swiglu",
            "fuse_cross_entropy",
            "fuse_linear_cross_entropy",
            "use_l2warp",
        ):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
        if self.fuse_cross_entropy and self.fuse_linear_cross_entropy:
            raise ValueError(
                "fuse_cross_entropy and fuse_linear_cross_entropy are mutually exclusive"
            )
        if not isinstance(self.hidden_act, str) or not self.hidden_act:
            raise TypeError("hidden_act must be a non-empty string")

        # Resolve during construction so invalid expansion fails before modules
        # allocate parameters.
        _ = self.resolved_head_k_dim
        _ = self.resolved_head_v_dim

    def _normalize_hybrid_attention(
        self, attn: dict | list[dict] | None
    ) -> dict | list[dict] | None:
        if attn is None:
            return None
        try:
            from fla.models.hybrid import normalize_hybrid_attention_config
        except ImportError as error:
            raise ImportError(
                "hybrid attention requires flash-linear-attention"
            ) from error
        return normalize_hybrid_attention_config(
            attn, num_hidden_layers=self.num_hidden_layers
        )

    def _resolve_head_dim(
        self, explicit: int | None, expand: float, name: str
    ) -> int:
        if explicit is not None:
            return explicit
        total_float = self.hidden_size * float(expand)
        total = round(total_float)
        if not math.isclose(total_float, total, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"hidden_size * {name} must be an integer, got {total_float}"
            )
        if total % self.num_heads:
            raise ValueError(
                f"resolved total width {total} must be divisible by "
                f"num_heads={self.num_heads}"
            )
        return total // self.num_heads

    @property
    def resolved_head_k_dim(self) -> int:
        return self._resolve_head_dim(self.head_k_dim, self.expand_k, "expand_k")

    @property
    def resolved_head_v_dim(self) -> int:
        return self._resolve_head_dim(self.head_v_dim, self.expand_v, "expand_v")

    @property
    def geometry_width(self) -> int:
        return self.resolved_head_k_dim

    @property
    def key_dim(self) -> int:
        return self.num_heads * self.resolved_head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_heads * self.resolved_head_v_dim


__all__ = ["SolveDeltaConfig"]
