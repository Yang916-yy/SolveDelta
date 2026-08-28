"""MESA-specialized paired RLS geometry with an implicit transpose.

The implementation reuses the MIT-licensed FLA MESA paired state,
matrix-free solve, and covariance reverse blocks while exposing only the
J/D/gain/prediction ABI consumed by SolveDelta.
"""

from __future__ import annotations

import torch

from fla.ops.common.chunk_h import chunk_bwd_dh
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2

from .mesa_specialized import (
    cg_forward,
    cg_transpose_hkv,
    hkk_reverse,
    hkv_reverse_dkv,
    paired_state_forward,
)

def _symmetrize(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


class _MesaRLSGeometry(torch.autograd.Function):
    """Paired MESA Hkk/Hkv specialization for SolveDelta geometry."""

    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,
        h: torch.Tensor,
        geometry_log_decay: torch.Tensor,
        initial_j: torch.Tensor,
        initial_d: torch.Tensor,
        chunk_size: int,
        cg_iterations: int,
    ):
        batch, length, heads, rank = u.shape
        local_decay = chunk_local_cumsum(
            geometry_log_decay,
            chunk_size=chunk_size,
            scale=RCP_LN2,
        )
        h_kk, h_kv, final_j, final_d = paired_state_forward(
            u,
            h,
            local_decay,
            initial_j,
            initial_d,
            chunk_size=chunk_size,
        )
        gain, updated_prediction = cg_forward(
            u,
            u,
            h,
            h_kk,
            h_kv,
            local_decay,
            chunk_size=chunk_size,
            steps=cg_iterations,
        )
        ctx.chunk_size = chunk_size
        ctx.cg_iterations = cg_iterations
        ctx.save_for_backward(
            u,
            h,
            local_decay,
            initial_j,
            initial_d,
            gain,
            updated_prediction,
        )
        return gain, updated_prediction, final_j, final_d

    @staticmethod
    def backward(
        ctx,
        grad_gain: torch.Tensor | None,
        grad_prediction: torch.Tensor | None,
        grad_final_j: torch.Tensor | None,
        grad_final_d: torch.Tensor | None,
    ):
        (
            u,
            h,
            local_decay,
            initial_j,
            initial_d,
            gain,
            prediction,
        ) = ctx.saved_tensors
        grad_gain = torch.zeros_like(gain) if grad_gain is None else grad_gain.to(u.dtype).contiguous()
        grad_prediction = (
            torch.zeros_like(prediction)
            if grad_prediction is None
            else grad_prediction.to(prediction.dtype).contiguous()
        )
        grad_final_j = None if grad_final_j is None else _symmetrize(grad_final_j.float()).contiguous()
        grad_final_d = None if grad_final_d is None else grad_final_d.float().contiguous()

        # Recompute only the two compact chunk-boundary states, matching MESA's
        # selected backward ownership rather than retaining 16 MiB per layer.
        h_kk, h_kv, _, _ = paired_state_forward(
            u,
            h,
            local_decay,
            initial_j,
            initial_d,
            chunk_size=ctx.chunk_size,
        )
        dh_kv, grad_initial_d = chunk_bwd_dh(
            q=gain,
            k=u,
            v=h,
            g=local_decay,
            gk=None,
            gv=None,
            do=grad_prediction,
            h0=initial_d,
            dht=grad_final_d,
            states_in_fp32=False,
            chunk_size=ctx.chunk_size,
            scale=1,
        )
        grad_u_kv, grad_h, grad_decay_first = hkv_reverse_dkv(
            gain,
            u,
            h,
            h_kv,
            dh_kv,
            local_decay,
            grad_prediction,
            chunk_size=ctx.chunk_size,
        )
        grad_rhs, grad_decay = cg_transpose_hkv(
            grad_gain.to(u.dtype),
            gain,
            u,
            h,
            h_kv,
            h_kk,
            local_decay,
            grad_prediction,
            grad_decay_first,
            chunk_size=ctx.chunk_size,
            steps=ctx.cg_iterations,
        )
        dh_kk, grad_initial_j = chunk_bwd_dh(
            q=grad_rhs,
            k=u,
            v=u,
            g=local_decay,
            gk=None,
            gv=None,
            do=gain,
            h0=initial_j,
            dht=-grad_final_j if grad_final_j is not None else None,
            states_in_fp32=False,
            chunk_size=ctx.chunk_size,
            scale=1,
        )
        grad_u_kk, grad_decay_kk = hkk_reverse(
            u,
            h_kk,
            dh_kk,
            local_decay,
            gain,
            grad_rhs,
            grad_u_kv,
            chunk_size=ctx.chunk_size,
        )
        grad_decay.add_(grad_decay_kk)
        grad_geometry_decay = chunk_local_cumsum(
            grad_decay,
            chunk_size=ctx.chunk_size,
            reverse=True,
        ).float()
        grad_u = grad_rhs.to(u.dtype) + grad_u_kk.to(u.dtype)
        grad_initial_j = None if grad_initial_j is None else _symmetrize(-grad_initial_j.float())
        grad_initial_d = None if grad_initial_d is None else grad_initial_d.float()
        return grad_u, grad_h, grad_geometry_decay, grad_initial_j, grad_initial_d, None, None


def mesa_rls_geometry(
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    initial_j: torch.Tensor,
    initial_d: torch.Tensor,
    *,
    chunk_size: int = 32,
    cg_iterations: int = 5,
):
    """Return gain, updated prediction, and FP32 J/D continuation states."""
    return _MesaRLSGeometry.apply(
        u,
        h,
        geometry_log_decay,
        initial_j,
        initial_d,
        chunk_size,
        cg_iterations,
    )
