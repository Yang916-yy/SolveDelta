"""Selected MESA-RLS geometry and block-E3 associative production path."""

from __future__ import annotations

import torch

from fla.modules.l2norm import l2norm

from ...reference import SolveDeltaState, solvedelta_zero_state
from .block_e3_exterior import block_e3_direct_e_delta_rule
from .mass import mass_prefix
from .mesa_gain import mesa_rls_geometry


PRIOR_MASS = 2.0
GAIN_CHUNK_SIZE = 32
CG_ITERATIONS = 5
TOKEN_CHUNK_SIZE = 16


def _normalize(x: torch.Tensor) -> torch.Tensor:
    # Match FLA GDN2/MESA's FP32 norm reduction and public-dtype rounding.
    return l2norm(x, eps=1.0e-24)


def solvedelta_rls_native(
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
    """Execute the selected dense BF16 RLS path.

    ``erase`` and ``write`` are activated gates in ``[0, 2]``. Raw-logit
    activation is owned by the public operator wrapper.
    """
    if u.ndim != 4:
        raise ValueError("u must have shape [B,T,H,r]")
    batch, length, heads, rank = u.shape
    if h.shape != u.shape or q.shape != u.shape:
        raise ValueError("h and q must match u")
    if keys.shape != (batch, length, heads, 1, rank):
        raise ValueError("the production RLS path requires keys [B,T,H,1,r]")
    if values.ndim != 5 or values.shape[:4] != (batch, length, heads, 1):
        raise ValueError("values must have shape [B,T,H,1,d_v]")
    value_dim = values.shape[-1]
    if geometry_log_decay.shape != (batch, length, heads):
        raise ValueError("geometry_log_decay must have shape [B,T,H]")
    if associative_log_decay.shape != (batch, length, heads, rank):
        raise ValueError("associative_log_decay must have shape [B,T,H,r]")
    if erase.shape != keys.shape or write.shape != values.shape:
        raise ValueError("erase/write shapes must match keys/values")
    if geometry_strength.shape not in ((heads,), (1, heads)):
        raise ValueError("geometry_strength must have shape [H] or [1,H]")
    if u.device.type != "cuda" or u.dtype != torch.bfloat16:
        raise TypeError("the production RLS path requires BF16 CUDA operands")
    if any(x.dtype != u.dtype for x in (h, q, keys, values, erase, write)):
        raise TypeError("all public vector operands must be BF16")
    if geometry_log_decay.dtype != torch.float32:
        raise TypeError("geometry_log_decay must be FP32")
    if associative_log_decay.dtype != torch.float32:
        raise TypeError("associative_log_decay must be FP32")

    neutral = solvedelta_zero_state(
        batch,
        heads,
        rank,
        value_dim,
        prior_mass=PRIOR_MASS,
        dtype=torch.float32,
        device=u.device,
    )
    state = neutral if initial_state is None else initial_state
    if any(value.shape != expected.shape for value, expected in zip(state, neutral)):
        raise ValueError("initial_state shapes do not match the input geometry")
    if any(x.dtype != torch.float32 for x in state):
        raise TypeError("all continuation states must be FP32")
    # CUDA Graph capture reuses the state object validated during its eager
    # warmup; torch.equal itself performs a capture-unsafe device reduction.
    if (
        not torch.cuda.is_current_stream_capturing()
        and not torch.equal(state.J, state.J.transpose(-1, -2))
    ):
        raise ValueError("initial_state.J must be exactly symmetric")

    u_panel = _normalize(u)
    gain, updated_prediction, final_j, final_d = mesa_rls_geometry(
        u_panel,
        h,
        geometry_log_decay,
        state.J,
        state.D,
        chunk_size=GAIN_CHUNK_SIZE,
        cg_iterations=CG_ITERATIONS,
    )
    previous_mass, current_mass, final_mass = mass_prefix(
        geometry_log_decay, state.m
    )
    output, final_s = block_e3_direct_e_delta_rule(
        u_panel,
        h,
        q,
        keys,
        values,
        gain,
        updated_prediction,
        geometry_log_decay,
        associative_log_decay,
        erase,
        write,
        previous_mass,
        current_mass,
        geometry_strength,
        state.S,
        token_chunk_size=TOKEN_CHUNK_SIZE,
    )
    return output.to(torch.bfloat16), SolveDeltaState(
        m=final_mass,
        J=0.5 * (final_j + final_j.transpose(-1, -2)),
        D=final_d,
        S=final_s.float(),
    )


__all__ = [
    "CG_ITERATIONS",
    "GAIN_CHUNK_SIZE",
    "PRIOR_MASS",
    "TOKEN_CHUNK_SIZE",
    "solvedelta_rls_native",
]
