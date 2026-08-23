from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SolveDeltaConfig:
    hidden_size: int
    num_heads: int
    head_k_dim: int | None = None
    head_v_dim: int | None = None
    expand_k: float = 1.0
    expand_v: float = 1.0
    num_edits: int = 1
    use_short_conv: bool = True
    bias: bool = False

    def __post_init__(self) -> None:
        for name in ("hidden_size", "num_heads", "num_edits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name in ("head_k_dim", "head_v_dim"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{name} must be an int or None")
                if value <= 0:
                    raise ValueError(f"{name} must be positive, got {value}")
        for name in ("expand_k", "expand_v"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("use_short_conv", "bias"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")

        # Force resolution during validation so invalid expansion/divisibility
        # fails at configuration construction rather than at the first forward.
        _ = self.resolved_head_k_dim
        _ = self.resolved_head_v_dim

    def _resolve_head_dim(self, explicit: int | None, expand: float, name: str) -> int:
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
