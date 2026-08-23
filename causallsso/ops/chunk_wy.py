from __future__ import annotations

import torch
import torch.nn.functional as F

from causallsso.ops.native_chunk import native_geometry_frame
from causallsso.ops.wy import wy_associative
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


def _normalize_to_bf16(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1).to(torch.bfloat16)


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
    normalized_u = _normalize_to_bf16(u)
    normalized_q = _normalize_to_bf16(q)
    normalized_keys = _normalize_to_bf16(keys)
    d, e, chi, geometry_final = native_geometry_frame(
        normalized_u,
        h,
        geometry_log_decay,
        normalized_keys,
        erase,
        normalized_q,
        geometry_strength,
        initial_state=initial_state,
    )

    z = write * values
    output, final_s = wy_associative(
        chi,
        d,
        e,
        z,
        associative_log_decay,
        initial_state=initial_state.S if initial_state is not None else None,
        output_final_state=True,
        chunk_size=_CHUNK_SIZE,
    )
    if final_s is None:  # pragma: no cover - fixed output_final_state=True
        raise RuntimeError("the WY exterior did not return its final state")
    return output, SolveDeltaState(
        geometry_final.m,
        geometry_final.J,
        geometry_final.D,
        final_s,
    )


__all__ = ["chunk_wy_solvedelta"]
