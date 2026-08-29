"""Public BF16 CUDA entry for Residual-Frame SolveDelta."""

from __future__ import annotations

import torch

from ..reference import SolveDeltaState
from .residual_frame import solvedelta_residual_frame_native


def solvedelta_native(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase_raw: torch.Tensor,
    write_raw: torch.Tensor,
    geometry_write: torch.Tensor,
    *,
    initial_state: SolveDeltaState | None = None,
    return_final_state: bool = False,
) -> tuple[torch.Tensor, SolveDeltaState | None]:
    """Execute dense K=1 SolveDelta from raw fused-projection views."""
    if u.device.type != "cuda" or u.dtype != torch.bfloat16:
        raise TypeError("solvedelta_native requires BF16 CUDA activations")
    if keys.ndim != 5 or keys.shape[-2] != 1:
        raise ValueError("the current production path requires num_edits=1")
    if erase_raw.shape != keys.shape or erase_raw.dtype != torch.bfloat16:
        raise ValueError("erase_raw must be BF16 with the same shape as keys")
    if write_raw.shape != values.shape or write_raw.dtype != torch.bfloat16:
        raise ValueError("write_raw must be BF16 with the same shape as values")
    if any(
        operand.stride(-1) != 1
        for operand in (u, h, q, keys, values, erase_raw, write_raw)
    ):
        raise ValueError("native vector operands require unit innermost stride")

    output, final_state = solvedelta_residual_frame_native(
        u,
        h,
        q,
        keys,
        values,
        associative_log_decay,
        erase_raw,
        write_raw,
        geometry_write,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )
    return output, final_state


__all__ = ["solvedelta_native"]
