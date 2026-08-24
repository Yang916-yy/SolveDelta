from __future__ import annotations

import torch

from causallsso.ops.native_chunk import native_chunk_solvedelta
from causallsso.ops.normalization import normalize_solvedelta_inputs
from causallsso.reference import SolveDeltaState


_CHUNK_SIZE = 32
_RANK = 128
_EDITS = 1


def _validate_inputs(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    geometry_strength: torch.Tensor,
    initial_state: SolveDeltaState | None,
) -> tuple[int, int, int, int]:
    named_inputs = {
        "u": u,
        "h": h,
        "q": q,
        "keys": keys,
        "values": values,
        "geometry_log_decay": geometry_log_decay,
        "associative_log_decay": associative_log_decay,
        "erase": erase,
        "write": write,
        "geometry_strength": geometry_strength,
    }
    for name, tensor in named_inputs.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if u.ndim != 4:
        raise ValueError("u must have shape [B,T,H,128]")
    batch, length, heads, rank = u.shape
    if min(batch, length, heads) < 1:
        raise ValueError("B, T, and H must be positive")
    if rank != _RANK:
        raise ValueError("the native chunk/WY path requires r=128")
    if h.shape != u.shape or q.shape != u.shape:
        raise ValueError("h and q must match u shape [B,T,H,128]")
    if keys.shape != (batch, length, heads, _EDITS, rank):
        raise ValueError("keys must have shape [B,T,H,1,128]")
    if erase.shape != keys.shape:
        raise ValueError("erase must match keys shape [B,T,H,1,128]")
    if values.ndim != 5 or values.shape[:4] != (
        batch,
        length,
        heads,
        _EDITS,
    ):
        raise ValueError("values must have shape [B,T,H,1,d_v]")
    value_dim = values.shape[-1]
    if value_dim < 1:
        raise ValueError("d_v must be positive")
    if write.shape != values.shape:
        raise ValueError("write must match values shape [B,T,H,1,d_v]")
    if geometry_log_decay.shape != (batch, length, heads):
        raise ValueError("geometry_log_decay must have shape [B,T,H]")
    if associative_log_decay.shape != (batch, length, heads, rank):
        raise ValueError("associative_log_decay must have shape [B,T,H,128]")
    if geometry_strength.shape != (heads,):
        raise ValueError("geometry_strength must have shape [H]")

    for name in ("u", "h", "q", "keys", "values", "erase", "write"):
        if named_inputs[name].dtype != torch.bfloat16:
            raise TypeError(f"{name} must be BF16")
    for name in (
        "geometry_log_decay",
        "associative_log_decay",
        "geometry_strength",
    ):
        if named_inputs[name].dtype != torch.float32:
            raise TypeError(f"{name} must be FP32")
    if u.device.type != "cuda":
        raise ValueError("the native chunk/WY path requires CUDA tensors")
    if any(tensor.device != u.device for tensor in named_inputs.values()):
        raise ValueError("all SolveDelta tensors must share one CUDA device")
    if torch.cuda.get_device_capability(u.device) != (12, 0):
        raise NotImplementedError(
            "the native chunk/WY specialization requires an SM120 CUDA device"
        )

    if initial_state is not None:
        if not isinstance(initial_state, SolveDeltaState):
            raise TypeError("initial_state must be a SolveDeltaState or None")
        expected_shapes = (
            (batch, heads),
            (batch, heads, rank, rank),
            (batch, heads, rank, rank),
            (batch, heads, rank, value_dim),
        )
        for name, tensor, shape in zip(
            SolveDeltaState._fields, initial_state, expected_shapes
        ):
            if tensor.shape != shape:
                raise ValueError(
                    f"initial_state.{name} must have shape {shape}, "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.device != u.device:
                raise ValueError(
                    f"initial_state.{name} must share the input CUDA device"
                )
            if tensor.dtype != torch.float32:
                raise TypeError(f"initial_state.{name} must be FP32")
    return batch, length, heads, value_dim


def chunk_wy_solvedelta(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    geometry_strength: torch.Tensor,
    *,
    initial_state: SolveDeltaState | None = None,
) -> tuple[torch.Tensor, SolveDeltaState]:
    """Evaluate the fixed BF16 C32 SolveDelta frame and C32 WY exterior."""
    _validate_inputs(
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
        initial_state,
    )
    normalized_u, normalized_q, normalized_keys = normalize_solvedelta_inputs(
        u,
        q,
        keys,
    )
    if initial_state is None:
        batch, _, heads, rank = normalized_u.shape
        value_dim = values.shape[-1]
        initial_state = SolveDeltaState(
            torch.zeros(
                batch, heads, device=u.device, dtype=torch.float32
            ),
            torch.zeros(
                batch,
                heads,
                rank,
                rank,
                device=u.device,
                dtype=torch.float32,
            ),
            torch.zeros(
                batch,
                heads,
                rank,
                rank,
                device=u.device,
                dtype=torch.float32,
            ),
            torch.zeros(
                batch,
                heads,
                rank,
                value_dim,
                device=u.device,
                dtype=torch.float32,
            ),
        )
    else:
        initial_state = SolveDeltaState(
            *(tensor.contiguous() for tensor in initial_state)
        )
    return native_chunk_solvedelta(
        normalized_u,
        h.contiguous(),
        normalized_q,
        normalized_keys,
        values.contiguous(),
        geometry_log_decay.contiguous(),
        associative_log_decay.contiguous(),
        erase.contiguous(),
        write.contiguous(),
        geometry_strength.contiguous(),
        initial_state,
    )


__all__ = ["chunk_wy_solvedelta"]
