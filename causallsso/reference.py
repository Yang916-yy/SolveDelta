"""FP64/PyTorch oracle for Residual-Frame SolveDelta."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


RELATIVE_FRAME_RADIUS = 5.0 / 8.0


class SolveDeltaState(NamedTuple):
    predictor: torch.Tensor
    S: torch.Tensor


def solvedelta_zero_state(
    batch: int,
    heads: int,
    rank: int,
    value_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> SolveDeltaState:
    return SolveDeltaState(
        predictor=torch.zeros(
            batch, heads, rank, rank, dtype=dtype, device=device
        ),
        S=torch.zeros(
            batch, heads, rank, value_dim, dtype=dtype, device=device
        ),
    )


def _validate(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    geometry_write: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    if u.ndim != 4:
        raise ValueError("u must have shape [B,T,H,r]")
    batch, length, heads, rank = u.shape
    if h.shape != u.shape or q.shape != u.shape:
        raise ValueError("h and q must match u")
    if keys.ndim != 5 or keys.shape[:3] != (batch, length, heads):
        raise ValueError("keys must have shape [B,T,H,K,r]")
    edits, key_rank = keys.shape[-2:]
    if key_rank != rank:
        raise ValueError("key width must equal r")
    if values.ndim != 5 or values.shape[:4] != (
        batch,
        length,
        heads,
        edits,
    ):
        raise ValueError("values must have shape [B,T,H,K,d_v]")
    value_dim = values.shape[-1]
    if associative_log_decay.shape != (batch, length, heads, rank):
        raise ValueError("associative_log_decay must have shape [B,T,H,r]")
    if erase.shape != keys.shape:
        raise ValueError("erase must match keys")
    if write.shape != values.shape:
        raise ValueError("write must match values")
    if geometry_write.shape not in (
        (heads,),
        (1, heads),
        (batch, length, heads),
    ):
        raise ValueError("geometry_write must have shape [H], [1,H], or [B,T,H]")
    return batch, length, heads, edits, rank, value_dim


def solvedelta_reference(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    geometry_write: torch.Tensor,
    *,
    initial_state: SolveDeltaState | None = None,
    valid_mask: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, SolveDeltaState]:
    """Execute Residual-Frame SolveDelta token by token.

    FP64 inputs define the operator. The predictor is stored in the orientation
    ``prediction = C @ u`` and updated by leaky normalized-LMS residual writes.
    The residual is evaluated against the complete old predictor before the
    existing DeltaRule channel retentions right-decay its source coordinates.
    The primal address uses the accumulated frame ``I+C``; dual addresses
    retain the bounded inverse-transpose of the current residual-local factor.
    """
    batch, length, heads, edits, rank, value_dim = _validate(
        u,
        h,
        q,
        keys,
        values,
        associative_log_decay,
        erase,
        write,
        geometry_write,
    )
    if edits != 1:
        raise ValueError("the current SolveDelta contract requires num_edits=1")
    if valid_mask is None:
        valid_mask = torch.ones(
            batch, length, dtype=torch.bool, device=u.device
        )
    if reset_mask is None:
        reset_mask = torch.zeros(
            batch, length, dtype=torch.bool, device=u.device
        )
    if valid_mask.shape != (batch, length) or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool [B,T]")
    if reset_mask.shape != (batch, length) or reset_mask.dtype != torch.bool:
        raise ValueError("reset_mask must be bool [B,T]")

    neutral = solvedelta_zero_state(
        batch,
        heads,
        rank,
        value_dim,
        dtype=u.dtype,
        device=u.device,
    )
    state = neutral if initial_state is None else initial_state
    if (
        state.predictor.shape != neutral.predictor.shape
        or state.S.shape != neutral.S.shape
    ):
        raise ValueError("initial_state shapes do not match inputs")

    u = F.normalize(u, p=2, dim=-1)
    q = F.normalize(q, p=2, dim=-1)
    keys = F.normalize(keys, p=2, dim=-1)
    geometry_write = geometry_write.to(dtype=u.dtype, device=u.device)
    if geometry_write.shape in ((heads,), (1, heads)):
        geometry_write = geometry_write.reshape(1, 1, heads).expand(
            batch, length, heads
        )
    outputs: list[torch.Tensor] = []
    for token in range(length):
        valid = valid_mask[:, token]
        reset = reset_mask[:, token] & valid
        predictor = torch.where(
            reset[:, None, None, None], neutral.predictor, state.predictor
        )
        memory = torch.where(
            reset[:, None, None, None], neutral.S, state.S
        )

        u_t = u[:, token]
        residual = h[:, token] - (
            predictor @ u_t.unsqueeze(-1)
        ).squeeze(-1)
        predictor_update = geometry_write[:, token, :, None] * residual
        predictor_new = predictor * associative_log_decay[:, token].exp()[
            ..., None, :
        ] + predictor_update[..., :, None] * u_t[..., None, :]

        direction_norm_sq = (u_t * u_t).sum(dim=-1, keepdim=True)
        update_norm_sq = (
            predictor_update * predictor_update
        ).sum(dim=-1, keepdim=True)
        frame_scale = RELATIVE_FRAME_RADIUS / torch.sqrt(
            RELATIVE_FRAME_RADIUS**2
            + direction_norm_sq * update_norm_sq
        )
        frame_covector = frame_scale * predictor_update

        denominator = 1.0 + (frame_covector * u_t).sum(dim=-1)
        key = keys[:, token, :, 0]
        erase_key = erase[:, token, :, 0] * key
        # The primal address uses the complete accumulated solution frame.
        # ``predictor`` stores C, so the identity base preserves the exact
        # finite-parameter GDN2 reduction when geometry writing is disabled.
        direct = key + (
            predictor_new.transpose(-1, -2) @ key.unsqueeze(-1)
        ).squeeze(-1)
        dual = erase_key - frame_covector * (
            (u_t * erase_key).sum(dim=-1, keepdim=True)
            / denominator[..., None]
        )
        query = q[:, token] - frame_covector * (
            (u_t * q[:, token]).sum(dim=-1, keepdim=True)
            / denominator[..., None]
        )

        memory_new = (
            associative_log_decay[:, token].exp()[..., None] * memory
        )
        prediction = (
            memory_new.transpose(-1, -2) @ dual.unsqueeze(-1)
        ).squeeze(-1)
        value = write[:, token, :, 0] * values[:, token, :, 0]
        memory_new = memory_new + direct[..., :, None] * (
            value - prediction
        )[..., None, :]
        output = (
            memory_new.transpose(-1, -2) @ query.unsqueeze(-1)
        ).squeeze(-1)
        output = torch.where(
            valid[:, None, None], output, torch.zeros_like(output)
        )
        state = SolveDeltaState(
            predictor=torch.where(
                valid[:, None, None, None], predictor_new, predictor
            ),
            S=torch.where(
                valid[:, None, None, None], memory_new, memory
            ),
        )
        outputs.append(output)

    return torch.stack(outputs, dim=1), state


__all__ = [
    "RELATIVE_FRAME_RADIUS",
    "SolveDeltaState",
    "solvedelta_reference",
    "solvedelta_zero_state",
]
