from __future__ import annotations

import torch

from causallsso.reference import (
    _bounded_ldu_untied_strength_reference,
    apply_dual_reference,
    apply_primal_reference,
    bounded_ldu_reference,
)


def _pad_time(tensor: torch.Tensor, padded_length: int) -> torch.Tensor:
    padding = padded_length - tensor.shape[1]
    if padding == 0:
        return tensor
    zeros = tensor.new_zeros(tensor.shape[0], padding, *tensor.shape[2:])
    return torch.cat((tensor, zeros), dim=1)


def _frame_oracle_impl(
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
    untied_strength: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble independent chunk-local actions from reference.py primitives."""
    batch, length, heads, rank = u.shape
    edits = keys.shape[-2]
    chunks = (length + chunk_size - 1) // chunk_size
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
    current_m = boundary_m.permute(0, 2, 1)
    current_J = boundary_J.permute(0, 2, 1, 3, 4)
    current_D = boundary_D.permute(0, 2, 1, 3, 4)
    if untied_strength:
        if geometry_strength.shape != (6, heads):
            raise ValueError("untied geometry strength must have shape [6,H]")
        strength = geometry_strength
    else:
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
        normalized_J = current_J / current_m[..., None, None]
        normalized_D = current_D / current_m[..., None, None]
        if untied_strength:
            lower, diagonal, upper = _bounded_ldu_untied_strength_reference(
                normalized_J, normalized_D, strength
            )
        else:
            lower, diagonal, upper = bounded_ldu_reference(
                normalized_J, normalized_D, strength
            )
        key_t = local_keys[:, :, local_t]
        erase_t = local_erase[:, :, local_t]
        staged_d.append(
            apply_primal_reference(
                lower, diagonal, upper, key_t.transpose(-1, -2)
            ).transpose(-1, -2)
        )
        staged_e.append(
            apply_dual_reference(
                lower,
                diagonal,
                upper,
                (erase_t * key_t).transpose(-1, -2),
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
    e = torch.stack(staged_e, dim=2).reshape_as(d)
    chi = torch.stack(staged_chi, dim=2).reshape(
        batch, chunks * local_steps, heads, rank
    )
    return d[:, :length], e[:, :length], chi[:, :length]


def frame_oracle(
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
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble independent chunk-local actions from reference.py primitives."""
    return _frame_oracle_impl(
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
        chunk_size=chunk_size,
        untied_strength=False,
    )


def _frame_oracle_untied_strength(
    *inputs: torch.Tensor,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the validation-only six-channel chart diagnostic."""
    return _frame_oracle_impl(
        *inputs,
        chunk_size=chunk_size,
        untied_strength=True,
    )


__all__ = ["frame_oracle"]
