from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


C_H = 1.0 / 8.0
C_R = 1.0 / 8.0
S_H = 1.0 / 8.0
S_R = 1.0 / 8.0


class SolveDeltaState(NamedTuple):
    m: torch.Tensor
    J: torch.Tensor
    D: torch.Tensor
    S: torch.Tensor


def _radial_bound(x: torch.Tensor, radius: float) -> torch.Tensor:
    norm_sq = x.square().sum(dim=(-2, -1), keepdim=True)
    return radius * x / torch.sqrt(x.new_tensor(radius * radius) + norm_sq)


def _bounded_ldu_untied_strength_reference(
    H: torch.Tensor,
    R: torch.Tensor,
    geometry_strength_channels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expose the six existing chart channels for gradient diagnostics only."""
    if H.shape != R.shape or H.ndim < 2 or H.shape[-1] != H.shape[-2]:
        raise ValueError("H and R must have equal [..., r, r] shapes")
    if (
        geometry_strength_channels.ndim < 1
        or geometry_strength_channels.shape[0] != 6
    ):
        raise ValueError("geometry_strength_channels must have leading size 6")

    r = H.shape[-1]
    eye = torch.eye(r, dtype=H.dtype, device=H.device)
    strengths = []
    for channel_strength in geometry_strength_channels.unbind(dim=0):
        while channel_strength.ndim < H.ndim - 2:
            channel_strength = channel_strength.unsqueeze(0)
        strengths.append(channel_strength.unsqueeze(-1).unsqueeze(-1))

    centered_h = H - eye / r
    lower_h = _radial_bound(
        torch.tril(strengths[0] * centered_h, diagonal=-1), C_H
    )
    lower_r = _radial_bound(
        torch.tril(strengths[1] * R, diagonal=-1), C_R
    )
    upper_h = _radial_bound(
        torch.triu(strengths[2] * centered_h, diagonal=1), C_H
    )
    upper_r = _radial_bound(
        torch.triu(strengths[3] * R, diagonal=1), C_R
    )

    diag_h = torch.diagonal(
        strengths[4] * centered_h, dim1=-2, dim2=-1
    )
    diag_r = torch.diagonal(strengths[5] * R, dim1=-2, dim2=-1)
    log_diagonal = S_H * torch.tanh(diag_h / S_H)
    log_diagonal = log_diagonal + S_R * torch.tanh(diag_r / S_R)
    diagonal = torch.exp(log_diagonal)

    lower = eye + (lower_h + lower_r)
    upper = eye + (upper_h + upper_r)
    return lower, diagonal, upper


def bounded_ldu_reference(
    H: torch.Tensor,
    R: torch.Tensor,
    geometry_strength: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the canonical bounded LDU factors.

    Returns ``(lower, diagonal, upper)`` with ``lower`` and ``upper`` unit
    triangular and ``diagonal`` stored as a vector.
    """
    tied_strength = geometry_strength.unsqueeze(0).expand(
        6, *geometry_strength.shape
    )
    return _bounded_ldu_untied_strength_reference(H, R, tied_strength)


def apply_primal_reference(
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Apply ``M^-1`` for ``M = lower @ diag(diagonal) @ upper``."""
    vector = rhs.ndim == lower.ndim - 1
    if vector:
        rhs = rhs.unsqueeze(-1)
    y = torch.linalg.solve_triangular(lower, rhs, upper=False, unitriangular=True)
    z = y / diagonal.unsqueeze(-1)
    out = torch.linalg.solve_triangular(upper, z, upper=True, unitriangular=True)
    return out.squeeze(-1) if vector else out


def apply_dual_reference(
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact inverse-transpose dual ``P^-T = M^T``."""
    vector = rhs.ndim == lower.ndim - 1
    if vector:
        rhs = rhs.unsqueeze(-1)
    out = lower.transpose(-1, -2) @ rhs
    out = diagonal.unsqueeze(-1) * out
    out = upper.transpose(-1, -2) @ out
    return out.squeeze(-1) if vector else out


def _zero_state(
    batch: int,
    heads: int,
    r: int,
    value_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> SolveDeltaState:
    return SolveDeltaState(
        m=torch.zeros(batch, heads, dtype=dtype, device=device),
        J=torch.zeros(batch, heads, r, r, dtype=dtype, device=device),
        D=torch.zeros(batch, heads, r, r, dtype=dtype, device=device),
        S=torch.zeros(batch, heads, r, value_dim, dtype=dtype, device=device),
    )


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
) -> tuple[int, int, int, int, int, int]:
    if u.ndim != 4:
        raise ValueError("u must have shape [B, T, H, r]")
    batch, length, heads, r = u.shape
    if h.shape != u.shape or q.shape != u.shape:
        raise ValueError("h and q must match u shape [B, T, H, r]")
    if keys.ndim != 5 or keys.shape[:3] != (batch, length, heads) or keys.shape[-1] != r:
        raise ValueError("keys must have shape [B, T, H, K, r]")
    edits = keys.shape[-2]
    if values.ndim != 5 or values.shape[:4] != (batch, length, heads, edits):
        raise ValueError("values must have shape [B, T, H, K, d_v]")
    value_dim = values.shape[-1]
    if geometry_log_decay.shape != (batch, length, heads):
        raise ValueError("geometry_log_decay must have shape [B, T, H]")
    if associative_log_decay.shape != (batch, length, heads, r):
        raise ValueError("associative_log_decay must have shape [B, T, H, r]")
    if erase.shape != keys.shape:
        raise ValueError("erase must match keys shape")
    if write.shape != values.shape:
        raise ValueError("write must match values shape")
    tensors = (
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
    if any(x.device != u.device for x in tensors):
        raise ValueError("all inputs must share one device")
    if any(x.dtype != u.dtype for x in tensors):
        raise ValueError("all inputs must share one floating dtype")
    if not u.dtype.is_floating_point:
        raise TypeError("SolveDelta inputs must be floating point")
    return batch, length, heads, edits, r, value_dim


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
    initial_state: SolveDeltaState | None = None,
    valid_mask: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
    return_state_history: bool = False,
) -> tuple[torch.Tensor, SolveDeltaState] | tuple[torch.Tensor, SolveDeltaState, SolveDeltaState]:
    """Explicit token-by-token SolveDelta recurrence.

    The function is intentionally composed from ordinary PyTorch operations.
    Calling it with FP64 tensors defines numerical truth for optimized paths.
    Inputs ``erase`` and ``write`` are activated gates in ``[0, 2]``;
    log decays are nonpositive. Invalid tokens emit zero and leave every
    recurrent state unchanged. A reset is applied immediately before its valid
    token.
    """
    batch, length, heads, edits, r, value_dim = _validate_inputs(
        u, h, q, keys, values, geometry_log_decay,
        associative_log_decay, erase, write,
    )
    if geometry_strength.shape not in ((heads,), (1, heads)):
        raise ValueError("geometry_strength must have shape [H] or [1, H]")
    geometry_strength = geometry_strength.reshape(heads).to(dtype=u.dtype, device=u.device)
    if valid_mask is None:
        valid_mask = torch.ones(batch, length, dtype=torch.bool, device=u.device)
    if reset_mask is None:
        reset_mask = torch.zeros(batch, length, dtype=torch.bool, device=u.device)
    if valid_mask.shape != (batch, length) or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool [B, T]")
    if reset_mask.shape != (batch, length) or reset_mask.dtype != torch.bool:
        raise ValueError("reset_mask must be bool [B, T]")

    zero = _zero_state(batch, heads, r, value_dim, dtype=u.dtype, device=u.device)
    state = zero if initial_state is None else initial_state
    if state.m.shape != zero.m.shape or state.J.shape != zero.J.shape or state.D.shape != zero.D.shape or state.S.shape != zero.S.shape:
        raise ValueError("initial_state shapes do not match inputs")

    u = F.normalize(u, p=2, dim=-1)
    q = F.normalize(q, p=2, dim=-1)
    keys = F.normalize(keys, p=2, dim=-1)
    outputs: list[torch.Tensor] = []
    histories: list[SolveDeltaState] = []

    for t in range(length):
        valid = valid_mask[:, t]
        reset = reset_mask[:, t] & valid
        state = SolveDeltaState(
            m=torch.where(reset[:, None], zero.m, state.m),
            J=torch.where(reset[:, None, None, None], zero.J, state.J),
            D=torch.where(reset[:, None, None, None], zero.D, state.D),
            S=torch.where(reset[:, None, None, None], zero.S, state.S),
        )

        lambda_g = torch.exp(geometry_log_decay[:, t])
        m_new = lambda_g * state.m + 1.0
        u_t = u[:, t]
        h_t = h[:, t]
        J_new = lambda_g[..., None, None] * state.J + u_t[..., :, None] * u_t[..., None, :]
        D_new = lambda_g[..., None, None] * state.D + u_t[..., :, None] * h_t[..., None, :]
        H_t = J_new / m_new[..., None, None]
        R_t = D_new / m_new[..., None, None]
        lower, diagonal, upper = bounded_ldu_reference(
            H_t, R_t, geometry_strength
        )

        S_new = torch.exp(associative_log_decay[:, t])[..., None] * state.S
        for j in range(edits):
            a = keys[:, t, :, j]
            b = erase[:, t, :, j] * a
            d = apply_primal_reference(lower, diagonal, upper, a)
            e = apply_dual_reference(lower, diagonal, upper, b)
            z = write[:, t, :, j] * values[:, t, :, j]
            prediction = (S_new.transpose(-1, -2) @ e.unsqueeze(-1)).squeeze(-1)
            innovation = z - prediction
            S_new = S_new + d.unsqueeze(-1) * innovation.unsqueeze(-2)

        chi = apply_dual_reference(lower, diagonal, upper, q[:, t])
        output = (S_new.transpose(-1, -2) @ chi.unsqueeze(-1)).squeeze(-1)
        output = torch.where(valid[:, None, None], output, torch.zeros_like(output))
        state = SolveDeltaState(
            m=torch.where(valid[:, None], m_new, state.m),
            J=torch.where(valid[:, None, None, None], J_new, state.J),
            D=torch.where(valid[:, None, None, None], D_new, state.D),
            S=torch.where(valid[:, None, None, None], S_new, state.S),
        )
        outputs.append(output)
        if return_state_history:
            histories.append(state)

    output_tensor = torch.stack(outputs, dim=1)
    if not return_state_history:
        return output_tensor, state
    history = SolveDeltaState(
        m=torch.stack([x.m for x in histories], dim=1),
        J=torch.stack([x.J for x in histories], dim=1),
        D=torch.stack([x.D for x in histories], dim=1),
        S=torch.stack([x.S for x in histories], dim=1),
    )
    return output_tensor, state, history
