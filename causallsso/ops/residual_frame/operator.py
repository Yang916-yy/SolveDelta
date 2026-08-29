"""Selected dense Residual-Frame SolveDelta production composition."""

from __future__ import annotations

import torch

from ...reference import SolveDeltaState, solvedelta_zero_state
from .exterior import direct_e_residual
from .l2norm import strided_l2norm
from .predictor import oja_residual
from .sources import relative_sources


PREDICTOR_CHUNK_SIZE = 32
EXTERIOR_CHUNK_SIZE = 16


def solvedelta_residual_frame_native(
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
    return_final_state: bool = True,
) -> tuple[torch.Tensor, SolveDeltaState | None]:
    """Execute the BF16 relative-frame path from raw gate logits."""
    if u.ndim != 4:
        raise ValueError("u must have shape [B,T,H,r]")
    batch, length, heads, rank = u.shape
    if h.shape != u.shape or q.shape != u.shape:
        raise ValueError("h and q must match u")
    if keys.shape != (batch, length, heads, 1, rank):
        raise ValueError("Residual-Frame SolveDelta requires keys [B,T,H,1,r]")
    if values.ndim != 5 or values.shape[:4] != (batch, length, heads, 1):
        raise ValueError("values must have shape [B,T,H,1,d_v]")
    value_dim = values.shape[-1]
    if associative_log_decay.shape != (batch, length, heads, rank):
        raise ValueError("associative_log_decay must have shape [B,T,H,r]")
    if erase_raw.shape != keys.shape or write_raw.shape != values.shape:
        raise ValueError("erase_raw/write_raw shapes must match keys/values")
    if geometry_write.shape not in (
        (heads,),
        (1, heads),
        (batch, length, heads),
    ):
        raise ValueError("geometry_write must have shape [H], [1,H], or [B,T,H]")
    if u.device.type != "cuda" or u.dtype != torch.bfloat16:
        raise TypeError("the production Residual-Frame path requires BF16 CUDA operands")
    if any(x.dtype != u.dtype for x in (h, q, keys, values, erase_raw, write_raw)):
        raise TypeError("all public vector operands must be BF16")
    if associative_log_decay.dtype != torch.float32:
        raise TypeError("associative_log_decay must be FP32")
    if geometry_write.dtype != torch.float32:
        raise TypeError("geometry_write must be FP32")
    if length % EXTERIOR_CHUNK_SIZE:
        raise ValueError("the first native Residual-Frame path requires T divisible by 16")

    neutral = solvedelta_zero_state(
        batch,
        heads,
        rank,
        value_dim,
        dtype=torch.float32,
        device=u.device,
    )
    state = neutral if initial_state is None else initial_state
    if any(value.shape != expected.shape for value, expected in zip(state, neutral)):
        raise ValueError("initial_state shapes do not match the input geometry")
    if any(value.dtype != torch.float32 for value in state):
        raise TypeError("all continuation states must be FP32")

    u_panel = strided_l2norm(u)
    q_panel = strided_l2norm(q)
    key_panel = strided_l2norm(keys.squeeze(-2))
    if geometry_write.shape != (batch, length, heads):
        geometry_write = geometry_write.reshape(1, 1, heads).expand(
            batch, length, heads
        )

    # FLA's current Oja WY helper assumes a packed target. This temporary
    # boundary is removed by the stride-aware donor specialization below.
    update, final_predictor = oja_residual(
        h.contiguous(),
        u_panel,
        geometry_write,
        state.predictor,
        chunk_size=PREDICTOR_CHUNK_SIZE,
        output_final_state=return_final_state,
    )
    direct, paired, injection = relative_sources(
        u_panel,
        update,
        q_panel,
        key_panel,
        values.squeeze(-2),
        erase_raw.squeeze(-2),
        write_raw.squeeze(-2),
        chunk_size=EXTERIOR_CHUNK_SIZE,
    )
    output, final_memory = direct_e_residual(
        direct,
        paired,
        injection,
        associative_log_decay,
        state.S,
        chunk_size=EXTERIOR_CHUNK_SIZE,
        output_final_state=return_final_state,
    )
    output = output.to(torch.bfloat16)
    if not return_final_state:
        return output, None
    if final_predictor is None or final_memory is None:
        raise RuntimeError("native state owners did not return requested endpoints")
    return output, SolveDeltaState(
        predictor=final_predictor.float(),
        S=final_memory.float(),
    )


__all__ = [
    "EXTERIOR_CHUNK_SIZE",
    "PREDICTOR_CHUNK_SIZE",
    "solvedelta_residual_frame_native",
]
