"""FP64/PyTorch oracle for the current SolveDelta RLS operator."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class SolveDeltaState(NamedTuple):
    m: torch.Tensor
    J: torch.Tensor
    D: torch.Tensor
    S: torch.Tensor


def solvedelta_zero_state(
    batch: int,
    heads: int,
    rank: int,
    value_dim: int,
    *,
    prior_mass: float,
    dtype: torch.dtype,
    device: torch.device,
) -> SolveDeltaState:
    if prior_mass <= 0:
        raise ValueError("prior_mass must be positive")
    identity = torch.eye(rank, dtype=dtype, device=device)
    identity = identity.view(1, 1, rank, rank).expand(batch, heads, -1, -1)
    return SolveDeltaState(
        m=torch.full((batch, heads), prior_mass, dtype=dtype, device=device),
        J=identity * prior_mass,
        D=torch.zeros(batch, heads, rank, rank, dtype=dtype, device=device),
        S=torch.zeros(batch, heads, rank, value_dim, dtype=dtype, device=device),
    )


def _validate(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
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
    if values.ndim != 5 or values.shape[:4] != (batch, length, heads, edits):
        raise ValueError("values must have shape [B,T,H,K,d_v]")
    value_dim = values.shape[-1]
    if geometry_log_decay.shape != (batch, length, heads):
        raise ValueError("geometry_log_decay must have shape [B,T,H]")
    if associative_log_decay.shape != (batch, length, heads, rank):
        raise ValueError("associative_log_decay must have shape [B,T,H,r]")
    if erase.shape != keys.shape:
        raise ValueError("erase must match keys")
    if write.shape != values.shape:
        raise ValueError("write must match values")
    return batch, length, heads, edits, rank, value_dim


def solvedelta_reference(
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
    prior_mass: float = 2.0,
    initial_state: SolveDeltaState | None = None,
    valid_mask: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
    return_gain: bool = False,
) -> tuple[torch.Tensor, SolveDeltaState] | tuple[torch.Tensor, SolveDeltaState, torch.Tensor]:
    """Execute the current SolveDelta recurrence token by token.

    Floating-point operations are deliberately ordinary PyTorch operations so
    FP64 inputs define the operator's numerical oracle.
    """
    batch, length, heads, edits, rank, value_dim = _validate(
        u,
        h,
        q,
        keys,
        values,
        geometry_log_decay,
        associative_log_decay,
        erase,
        write,
    )
    if edits != 1:
        raise ValueError("the current SolveDelta contract requires num_edits=1")
    if geometry_strength.shape not in ((heads,), (1, heads)):
        raise ValueError("geometry_strength must have shape [H] or [1,H]")
    strength = geometry_strength.reshape(heads).to(dtype=u.dtype, device=u.device)
    if valid_mask is None:
        valid_mask = torch.ones(batch, length, dtype=torch.bool, device=u.device)
    if reset_mask is None:
        reset_mask = torch.zeros(batch, length, dtype=torch.bool, device=u.device)
    if valid_mask.shape != (batch, length) or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool [B,T]")
    if reset_mask.shape != (batch, length) or reset_mask.dtype != torch.bool:
        raise ValueError("reset_mask must be bool [B,T]")

    neutral = solvedelta_zero_state(
        batch,
        heads,
        rank,
        value_dim,
        prior_mass=prior_mass,
        dtype=u.dtype,
        device=u.device,
    )
    state = neutral if initial_state is None else initial_state
    if (
        state.m.shape != neutral.m.shape
        or state.J.shape != neutral.J.shape
        or state.D.shape != neutral.D.shape
        or state.S.shape != neutral.S.shape
    ):
        raise ValueError("initial_state shapes do not match inputs")
    if not torch.equal(state.J, state.J.transpose(-1, -2)):
        raise ValueError("initial_state.J must be exactly symmetric")

    # The public operator follows the frontend normalization convention.
    u = F.normalize(u, p=2, dim=-1)
    q = F.normalize(q, p=2, dim=-1)
    keys = F.normalize(keys, p=2, dim=-1)
    identity = torch.eye(rank, dtype=u.dtype, device=u.device).view(1, 1, rank, rank)
    outputs: list[torch.Tensor] = []
    gains: list[torch.Tensor] = []

    for t in range(length):
        valid = valid_mask[:, t]
        reset = reset_mask[:, t] & valid
        state = SolveDeltaState(
            m=torch.where(reset[:, None], neutral.m, state.m),
            J=torch.where(reset[:, None, None, None], neutral.J, state.J),
            D=torch.where(reset[:, None, None, None], neutral.D, state.D),
            S=torch.where(reset[:, None, None, None], neutral.S, state.S),
        )

        u_t = u[:, t]
        h_t = h[:, t]
        lambda_t = geometry_log_decay[:, t].exp()
        m_new = lambda_t * state.m + 1.0
        J_new = lambda_t[..., None, None] * state.J + u_t[..., :, None] * u_t[..., None, :]
        D_new = lambda_t[..., None, None] * state.D + u_t[..., :, None] * h_t[..., None, :]

        gain = torch.linalg.solve(J_new, u_t.unsqueeze(-1)).squeeze(-1)
        previous_gain = torch.linalg.solve(state.J, u_t.unsqueeze(-1)).squeeze(-1)
        previous_c = torch.linalg.solve(state.J, state.D)
        residual = h_t - (previous_c.transpose(-1, -2) @ u_t.unsqueeze(-1)).squeeze(-1)

        mass_scale = state.m / m_new
        F_h = mass_scale[..., None, None] * (
            lambda_t[..., None, None] * identity
            + u_t[..., :, None] * previous_gain[..., None, :]
        )
        gamma = strength.view(1, heads, 1, 1)
        F_h = identity + gamma * (F_h - identity)
        F_c = identity + gamma * (gain[..., :, None] * residual[..., None, :])

        S_new = F_h @ state.S
        S_new = associative_log_decay[:, t].exp()[..., None] * S_new
        S_new = F_c @ S_new
        for edit in range(edits):
            key = keys[:, t, :, edit]
            erase_key = erase[:, t, :, edit] * key
            value = write[:, t, :, edit] * values[:, t, :, edit]
            prediction = (S_new.transpose(-1, -2) @ erase_key.unsqueeze(-1)).squeeze(-1)
            S_new = S_new + key[..., :, None] * (value - prediction)[..., None, :]

        output = (S_new.transpose(-1, -2) @ q[:, t].unsqueeze(-1)).squeeze(-1)
        output = torch.where(valid[:, None, None], output, torch.zeros_like(output))
        state = SolveDeltaState(
            m=torch.where(valid[:, None], m_new, state.m),
            J=torch.where(valid[:, None, None, None], J_new, state.J),
            D=torch.where(valid[:, None, None, None], D_new, state.D),
            S=torch.where(valid[:, None, None, None], S_new, state.S),
        )
        outputs.append(output)
        gains.append(torch.where(valid[:, None, None], gain, torch.zeros_like(gain)))

    result = torch.stack(outputs, dim=1)
    if return_gain:
        return result, state, torch.stack(gains, dim=1)
    return result, state


__all__ = [
    "SolveDeltaState",
    "solvedelta_reference",
    "solvedelta_zero_state",
]
