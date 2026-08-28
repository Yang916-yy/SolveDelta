"""Public BF16 CUDA entry for the selected SolveDelta RLS path."""

from __future__ import annotations

import torch

from ..reference import SolveDeltaState
from .rls import solvedelta_rls_native
from .rls.gate import activated_gate


def solvedelta_native(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase_raw: torch.Tensor,
    write_raw: torch.Tensor,
    geometry_strength: torch.Tensor,
    *,
    initial_state: SolveDeltaState | None = None,
    return_final_state: bool = False,
) -> tuple[torch.Tensor, SolveDeltaState | None]:
    """Execute dense K=1 SolveDelta with activated FLA gate primitives."""
    if u.device.type != "cuda" or u.dtype != torch.bfloat16:
        raise TypeError("solvedelta_native requires BF16 CUDA activations")
    if keys.ndim != 5 or keys.shape[-2] != 1:
        raise ValueError("the current production path requires num_edits=1")
    if erase_raw.shape != keys.shape or erase_raw.dtype != torch.bfloat16:
        raise ValueError("erase_raw must be BF16 with the same shape as keys")
    if write_raw.shape != values.shape or write_raw.dtype != torch.bfloat16:
        raise ValueError("write_raw must be BF16 with the same shape as values")

    # The MESA and block-E3 owners use packed tensor ABIs. Projection slices
    # retain the wider fused-projection row stride, so canonicalize the five
    # public vector panels once at this boundary.
    u = u.contiguous()
    h = h.contiguous()
    q = q.contiguous()
    keys = keys.contiguous()
    values = values.contiguous()
    erase = activated_gate(erase_raw, scale=2.0)
    write = activated_gate(write_raw, scale=2.0)
    output, final_state = solvedelta_rls_native(
        u,
        h,
        q,
        keys,
        values,
        geometry_log_decay,
        associative_log_decay,
        erase,
        write,
        geometry_strength,
        initial_state=initial_state,
    )
    return output, final_state if return_final_state else None


__all__ = ["solvedelta_native"]
