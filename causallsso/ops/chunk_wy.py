from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from causallsso.ops.chunk_frame import chunk_frame
from causallsso.ops.native_chunk import native_chunk_frame
from causallsso.ops.triton_geometry import triton_geometry_chunk_scan
from causallsso.ops.wy import wy_associative
from causallsso.reference import SolveDeltaState


ChunkFrameBackend = Literal["reference", "native"]

_FRAME_CHUNK_SIZE = 32


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
    backend: ChunkFrameBackend,
    wy_dtype: torch.dtype,
    wy_chunk_size: int,
) -> tuple[int, int, int, int, int, int]:
    if backend not in ("reference", "native"):
        raise ValueError("backend must be 'reference' or 'native'")
    if wy_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("wy_dtype must be torch.float16 or torch.bfloat16")
    if isinstance(wy_chunk_size, bool) or not isinstance(wy_chunk_size, int):
        raise TypeError("wy_chunk_size must be an int")
    if wy_chunk_size < 1:
        raise ValueError("wy_chunk_size must be positive")

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
        raise ValueError("u must have shape [B,T,H,r]")
    batch, length, heads, rank = u.shape
    if min(batch, length, heads, rank) < 1:
        raise ValueError("B, T, H, and r must be positive")
    if rank % 32:
        raise ValueError("the C32 geometry scan requires r divisible by 32")
    if h.shape != u.shape or q.shape != u.shape:
        raise ValueError("h and q must match u shape [B,T,H,r]")
    if keys.ndim != 5 or keys.shape[:3] != (batch, length, heads):
        raise ValueError("keys must have shape [B,T,H,K,r]")
    edits = keys.shape[-2]
    if edits < 1 or keys.shape[-1] != rank:
        raise ValueError("keys must have shape [B,T,H,K,r] with positive K")
    if values.ndim != 5 or values.shape[:4] != (
        batch,
        length,
        heads,
        edits,
    ):
        raise ValueError("values must have shape [B,T,H,K,d_v]")
    value_dim = values.shape[-1]
    if value_dim < 1:
        raise ValueError("d_v must be positive")
    if geometry_log_decay.shape != (batch, length, heads):
        raise ValueError("geometry_log_decay must have shape [B,T,H]")
    if associative_log_decay.shape != (batch, length, heads, rank):
        raise ValueError("associative_log_decay must have shape [B,T,H,r]")
    if erase.shape != keys.shape:
        raise ValueError("erase must match keys shape [B,T,H,K,r]")
    if write.shape != values.shape:
        raise ValueError("write must match values shape [B,T,H,K,d_v]")
    if geometry_strength.shape not in ((heads,), (1, heads)):
        raise ValueError("geometry_strength must have shape [H] or [1,H]")

    tensors = tuple(named_inputs.values())
    if any(tensor.device != u.device for tensor in tensors):
        raise ValueError("all SolveDelta tensors must share one device")
    if u.device.type != "cuda":
        raise ValueError("the chunk/WY backend requires CUDA tensors")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise TypeError("the chunk/frame composition requires FP32 inputs")

    if backend == "native" and (rank != 128 or edits != 1):
        raise ValueError("the native frame backend requires r=128 and K=1")

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
            ("m", "J", "D", "S"), initial_state, expected_shapes
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"initial_state.{name} must be a torch.Tensor")
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

    return batch, length, heads, edits, rank, value_dim


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
    backend: ChunkFrameBackend = "reference",
    wy_dtype: torch.dtype = torch.float16,
    wy_chunk_size: int = 16,
) -> tuple[torch.Tensor, SolveDeltaState]:
    """Evaluate SolveDelta as a C32 local frame followed by FLA chunk/WY.

    The chart-sensitive geometry and frame actions are evaluated in FP32.
    Only the associative vectors passed to FLA are quantized to ``wy_dtype``;
    its recurrent state remains FP32. ``backend='native'`` selects the one
    current CUDA specialization and fails explicitly when it is unavailable or
    outside its ``r=128, K=1`` contract. ``backend='reference'`` uses the exact
    differentiable chunk-local staging implementation.

    This composition currently owns dense, unmasked training segments. Packed
    masks, resets, and recurrent decode require dedicated boundary semantics
    and are intentionally not accepted here.
    """
    _, _, heads, _, _, _ = _validate_inputs(
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
        backend,
        wy_dtype,
        wy_chunk_size,
    )

    normalized_u = F.normalize(u, p=2, dim=-1)
    normalized_q = F.normalize(q, p=2, dim=-1)
    normalized_keys = F.normalize(keys, p=2, dim=-1)
    strength = geometry_strength.reshape(heads)

    boundary, geometry_final = triton_geometry_chunk_scan(
        normalized_u,
        h,
        geometry_log_decay,
        initial_state=initial_state,
        chunk_size=_FRAME_CHUNK_SIZE,
        input_precision="ieee",
    )
    frame_inputs = (
        normalized_u,
        h,
        geometry_log_decay,
        normalized_keys,
        erase,
        normalized_q,
        strength,
        boundary.m,
        boundary.J,
        boundary.D,
    )
    if backend == "native":
        d, e, chi = native_chunk_frame(*frame_inputs)
    else:
        d, e, chi = chunk_frame(
            *frame_inputs,
            chunk_size=_FRAME_CHUNK_SIZE,
        )

    z = write * values
    associative_initial = initial_state.S if initial_state is not None else None
    output, final_s = wy_associative(
        chi.to(wy_dtype),
        d.to(wy_dtype),
        e.to(wy_dtype),
        z.to(wy_dtype),
        associative_log_decay.to(wy_dtype),
        initial_state=associative_initial,
        output_final_state=True,
        chunk_size=wy_chunk_size,
    )
    if final_s is None:  # pragma: no cover - guarded by output_final_state=True
        raise RuntimeError("the WY exterior did not return its final state")

    final_state = SolveDeltaState(
        geometry_final.m,
        geometry_final.J,
        geometry_final.D,
        final_s,
    )
    return output, final_state


__all__ = ["ChunkFrameBackend", "chunk_wy_solvedelta"]
