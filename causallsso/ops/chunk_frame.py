from __future__ import annotations

import torch

from causallsso.reference import (
    apply_dual_reference,
    apply_primal_reference,
    bounded_ldu_reference,
)


def _validate_chunk_frame_inputs(
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    keys: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    geometry_strength: torch.Tensor,
    boundary_m: torch.Tensor,
    boundary_J: torch.Tensor,
    boundary_D: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int, int]:
    if u.ndim != 4:
        raise ValueError("u must have shape [B,T,H,r]")
    batch, length, heads, rank = u.shape
    if min(batch, length, heads, rank) < 1:
        raise ValueError("B, T, H, and r must be positive")
    if h.shape != u.shape or query.shape != u.shape:
        raise ValueError("h and query must match u shape [B,T,H,r]")
    if keys.ndim != 5 or keys.shape[:3] != (batch, length, heads):
        raise ValueError("keys must have shape [B,T,H,K,r]")
    edits = keys.shape[-2]
    if edits < 1 or keys.shape[-1] != rank:
        raise ValueError("keys must have shape [B,T,H,K,r] with positive K")
    if erase.shape != keys.shape:
        raise ValueError("erase must match keys shape [B,T,H,K,r]")
    if geometry_log_decay.shape != (batch, length, heads):
        raise ValueError("geometry_log_decay must have shape [B,T,H]")
    if geometry_strength.shape not in ((heads,), (1, heads)):
        raise ValueError("geometry_strength must have shape [H] or [1,H]")
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")

    chunks = (length + chunk_size - 1) // chunk_size
    if boundary_m.shape != (batch, heads, chunks):
        raise ValueError("boundary_m must have shape [B,H,ceil(T/chunk_size)]")
    matrix_shape = (batch, heads, chunks, rank, rank)
    if boundary_J.shape != matrix_shape or boundary_D.shape != matrix_shape:
        raise ValueError(
            "boundary_J and boundary_D must have shape "
            "[B,H,ceil(T/chunk_size),r,r]"
        )

    tensors = (
        u,
        h,
        geometry_log_decay,
        keys,
        erase,
        query,
        geometry_strength,
        boundary_m,
        boundary_J,
        boundary_D,
    )
    if any(tensor.device != u.device for tensor in tensors):
        raise ValueError("all chunk-frame tensors must share one device")
    if any(tensor.dtype != u.dtype for tensor in tensors):
        raise TypeError("all chunk-frame tensors must share one dtype")
    if not u.dtype.is_floating_point:
        raise TypeError("chunk-frame tensors must be floating point")
    return batch, length, heads, edits, rank, chunks


def _pad_time(tensor: torch.Tensor, padded_length: int) -> torch.Tensor:
    padding = padded_length - tensor.shape[1]
    if padding == 0:
        return tensor
    zeros = tensor.new_zeros(tensor.shape[0], padding, *tensor.shape[2:])
    return torch.cat((tensor, zeros), dim=1)


def chunk_frame(
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    keys: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    geometry_strength: torch.Tensor,
    boundary_m: torch.Tensor,
    boundary_J: torch.Tensor,
    boundary_D: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage exact SolveDelta frame actions from independent chunk boundaries.

    ``u``, ``keys``, and ``query`` are the already-normalized frontend
    vectors. Each entry in ``boundary_m/J/D`` is the geometry state immediately
    before its chunk. Chunks are batched as independent problems; the only
    recurrence here is over the finite local ``chunk_size`` dimension.

    The returned primal write directions ``d`` and dual erase covectors ``e``
    have shape ``[B,T,H,K,r]``. Read covectors ``chi`` have shape
    ``[B,T,H,r]``. All operations are ordinary differentiable PyTorch and the
    chart and primal/dual mathematics remain owned by ``reference.py``.
    """
    batch, length, heads, edits, rank, chunks = _validate_chunk_frame_inputs(
        u,
        h,
        geometry_log_decay,
        keys,
        erase,
        query,
        geometry_strength,
        boundary_m,
        boundary_J,
        boundary_D,
        chunk_size,
    )
    padded_length = chunks * chunk_size

    local_u = _pad_time(u, padded_length).reshape(
        batch, chunks, chunk_size, heads, rank
    )
    local_h = _pad_time(h, padded_length).reshape_as(local_u)
    local_decay = _pad_time(geometry_log_decay, padded_length).reshape(
        batch, chunks, chunk_size, heads
    )
    local_keys = _pad_time(keys, padded_length).reshape(
        batch, chunks, chunk_size, heads, edits, rank
    )
    local_erase = _pad_time(erase, padded_length).reshape_as(local_keys)
    local_query = _pad_time(query, padded_length).reshape_as(local_u)
    # [B,H,N,...] scan output becomes a panel batch [B,N,H,...]. No state
    # produced below is passed from one panel to the next.
    current_m = boundary_m.permute(0, 2, 1)
    current_J = boundary_J.permute(0, 2, 1, 3, 4)
    current_D = boundary_D.permute(0, 2, 1, 3, 4)
    strength = geometry_strength.reshape(heads)

    staged_d: list[torch.Tensor] = []
    staged_e: list[torch.Tensor] = []
    staged_chi: list[torch.Tensor] = []
    local_steps = length if chunks == 1 else chunk_size
    for local_t in range(local_steps):
        decay = torch.exp(local_decay[:, :, local_t])
        u_t = local_u[:, :, local_t]
        h_t = local_h[:, :, local_t]
        current_m = decay * current_m + 1.0
        current_J = (
            decay[..., None, None] * current_J
            + u_t[..., :, None] * u_t[..., None, :]
        )
        current_D = (
            decay[..., None, None] * current_D
            + u_t[..., :, None] * h_t[..., None, :]
        )

        lower, diagonal, upper = bounded_ldu_reference(
            current_J / current_m[..., None, None],
            current_D / current_m[..., None, None],
            strength,
        )
        key_t = local_keys[:, :, local_t]
        erase_t = local_erase[:, :, local_t]
        b = erase_t * key_t

        staged_d.append(
            apply_primal_reference(
                lower, diagonal, upper, key_t.transpose(-1, -2)
            ).transpose(-1, -2)
        )
        staged_e.append(
            apply_dual_reference(
                lower, diagonal, upper, b.transpose(-1, -2)
            ).transpose(-1, -2)
        )
        staged_chi.append(
            apply_dual_reference(
                lower, diagonal, upper, local_query[:, :, local_t]
            )
        )

    d = torch.stack(staged_d, dim=2).reshape(
        batch, chunks * local_steps, heads, edits, rank
    )
    e = torch.stack(staged_e, dim=2).reshape(
        batch, chunks * local_steps, heads, edits, rank
    )
    chi = torch.stack(staged_chi, dim=2).reshape(
        batch, chunks * local_steps, heads, rank
    )
    return d[:, :length], e[:, :length], chi[:, :length]


__all__ = ["chunk_frame"]
