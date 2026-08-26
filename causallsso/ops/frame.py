from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from ..reference import SolveDeltaState
from .chart import chart_coefficients
from .geometry_scan import geometry_scan
from .normalization import normalize_frame_inputs
from .radial import strict_gram
from .resident_frame import (
    boundary_route_forward,
    boundary_route_vjp,
    factor_local_representation_vjp,
    packed_factor_boundary_vjp,
    resident_dual,
    resident_factor_direct,
    resident_factor_transpose,
    resident_primal,
)


class FramePanels(NamedTuple):
    # Frame-native storage. d is [P,K,C,r] and paired_dual is
    # [P,K+1,C,r] with e routes followed by chi. P indexes [B,H,Nchunk].
    d: torch.Tensor
    paired_dual: torch.Tensor
    m: torch.Tensor
    J: torch.Tensor
    D: torch.Tensor


class _ChunkGeometry(NamedTuple):
    u: torch.Tensor
    h: torch.Tensor
    J: torch.Tensor
    D: torch.Tensor
    decay: torch.Tensor
    boundary_h: torch.Tensor
    boundary_r_lower: torch.Tensor
    boundary_r_upper: torch.Tensor
    sigma: torch.Tensor
    cumulative: torch.Tensor
    mass: torch.Tensor
    kappa_h: torch.Tensor
    kappa_r_lower: torch.Tensor
    kappa_r_upper: torch.Tensor
    shape: tuple[int, int, int, int, int]


class _BoundaryStats(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,
        h: torch.Tensor,
        J: torch.Tensor,
        D: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        norm_h, corr_h = boundary_route_forward(
            u, u, J, lower=True, num_warps=4
        )
        norm_lower, corr_lower = boundary_route_forward(
            u, h, D, lower=True, num_warps=4
        )
        norm_upper, corr_upper = boundary_route_forward(
            u, h, D, lower=False, num_warps=4
        )
        ctx.save_for_backward(u, h, J, D)
        return (
            norm_h,
            norm_lower,
            norm_upper,
            corr_h,
            corr_lower,
            corr_upper,
        )

    @staticmethod
    def backward(
        ctx,
        grad_norm_h: torch.Tensor,
        grad_norm_lower: torch.Tensor,
        grad_norm_upper: torch.Tensor,
        grad_corr_h: torch.Tensor,
        grad_corr_lower: torch.Tensor,
        grad_corr_upper: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        u, h, J, D = ctx.saved_tensors
        grad_u = torch.empty_like(u, dtype=torch.float32)
        grad_h = torch.empty_like(h, dtype=torch.float32)
        grad_j = torch.empty_like(J, dtype=torch.float32)
        grad_d = torch.empty_like(D, dtype=torch.float32)
        boundary_route_vjp(
            u,
            u,
            J,
            grad_norm_h.contiguous(),
            grad_corr_h.contiguous(),
            grad_u,
            grad_u,
            grad_j,
            lower=True,
            accumulate_left=False,
            accumulate_right=False,
            accumulate_matrix=False,
            same_output=True,
            num_warps=4,
        )
        boundary_route_vjp(
            u,
            h,
            D,
            grad_norm_lower.contiguous(),
            grad_corr_lower.contiguous(),
            grad_u,
            grad_h,
            grad_d,
            lower=True,
            accumulate_left=True,
            accumulate_right=False,
            accumulate_matrix=False,
            num_warps=4,
        )
        boundary_route_vjp(
            u,
            h,
            D,
            grad_norm_upper.contiguous(),
            grad_corr_upper.contiguous(),
            grad_u,
            grad_h,
            grad_d,
            lower=False,
            accumulate_left=True,
            accumulate_right=True,
            accumulate_matrix=True,
            num_warps=4,
        )
        return grad_u, grad_h, grad_j, grad_d


def _boundary_stats(
    u: torch.Tensor,
    h: torch.Tensor,
    J: torch.Tensor,
    D: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _BoundaryStats.apply(u, h, J, D)


def _build_geometry(
    u_panel: torch.Tensor,
    u_d_panel: torch.Tensor,
    h_panel: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    geometry_strength: torch.Tensor,
    initial_state: SolveDeltaState | None,
    lengths: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    *,
    chunk_size: int,
) -> tuple[_ChunkGeometry, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    batch, length, heads = geometry_log_decay.shape
    width = u_panel.shape[-1]
    state_batch = batch if cu_seqlens is None else len(cu_seqlens) - 1
    chunks = (length + chunk_size - 1) // chunk_size
    padded_length = chunks * chunk_size
    shape = (batch, length, heads, chunks, chunk_size)

    if initial_state is None:
        initial_m = torch.zeros(
            state_batch, heads, dtype=torch.float32, device=u_panel.device
        )
        initial_j = torch.zeros(
            state_batch,
            heads,
            width,
            width,
            dtype=torch.float32,
            device=u_panel.device,
        )
        initial_d = torch.zeros_like(initial_j)
    else:
        initial_m = initial_state.m.float()
        initial_j = initial_state.J.float()
        initial_d = initial_state.D.float()
    (
        mass,
        J,
        D,
        m_current,
        j_current,
        d_current,
        g,
    ) = geometry_scan(
        u_panel,
        u_d_panel,
        h_panel,
        geometry_log_decay,
        initial_m,
        initial_j,
        initial_d,
        lengths,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_indices,
        chunk_size=chunk_size,
    )
    gram_h, gram_r_lower, gram_r_upper = strict_gram(u_panel, h_panel)
    (
        norm_h,
        norm_r_lower,
        norm_r_upper,
        corr_h,
        corr_r_lower,
        corr_r_upper,
    ) = _boundary_stats(u_panel, h_panel, J, D)
    strength = geometry_strength.float().reshape(heads)
    diagonal_j = J.diagonal(dim1=-2, dim2=-1)
    diagonal_d = D.diagonal(dim1=-2, dim2=-1)
    (
        decay,
        boundary_h,
        boundary_r_lower,
        boundary_r_upper,
        sigma,
        kappa_h,
        kappa_r_lower,
        kappa_r_upper,
    ) = chart_coefficients(
        g,
        mass,
        diagonal_j,
        diagonal_d,
        u_panel,
        h_panel,
        strength,
        gram_h,
        gram_r_lower,
        gram_r_upper,
        corr_h,
        corr_r_lower,
        corr_r_upper,
        norm_h,
        norm_r_lower,
        norm_r_upper,
        heads,
        chunks,
        cu_seqlens is not None,
    )

    geometry = _ChunkGeometry(
        u=u_panel,
        h=h_panel,
        J=J,
        D=D,
        decay=decay,
        boundary_h=boundary_h,
        boundary_r_lower=boundary_r_lower,
        boundary_r_upper=boundary_r_upper,
        sigma=sigma,
        cumulative=g,
        mass=mass,
        kappa_h=kappa_h,
        kappa_r_lower=kappa_r_lower,
        kappa_r_upper=kappa_r_upper,
        shape=shape,
    )
    return geometry, (m_current, j_current, d_current)


@triton.jit
def _primal_sigma_reverse_kernel(
    upper_cotangent,
    lower_cache,
    sigma,
    lower_cotangent,
    grad_sigma,
    N: tl.constexpr,
    CR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    panel = tl.program_id(1).to(tl.int64)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < CR
    scale = tl.load(sigma + panel * CR + offsets, mask=valid, other=1.0).to(
        tl.float32
    )
    scale_cotangent = tl.zeros([BLOCK], dtype=tl.float32)
    for rhs in range(N):
        source = (panel * N + rhs) * CR + offsets
        upper = tl.load(
            upper_cotangent + source, mask=valid, other=0.0
        ).to(tl.float32)
        lower = tl.load(lower_cache + source, mask=valid, other=0.0).to(
            tl.float32
        )
        tl.store(lower_cotangent + source, upper / scale, mask=valid)
        scale_cotangent -= upper * lower / (scale * scale)
    tl.store(grad_sigma + panel * CR + offsets, scale_cotangent, mask=valid)


@triton.jit
def _dual_sigma_forward_kernel(
    value,
    sigma,
    output,
    N: tl.constexpr,
    CR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    panel = tl.program_id(1).to(tl.int64)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < CR
    scale = tl.load(sigma + panel * CR + offsets, mask=valid, other=1.0).to(
        tl.float32
    )
    for rhs in range(N):
        pointer = (panel * N + rhs) * CR + offsets
        x = tl.load(value + pointer, mask=valid, other=0.0).to(tl.float32)
        tl.store(output + pointer, x * scale, mask=valid)


@triton.jit
def _dual_sigma_reverse_kernel(
    grad_output,
    value,
    sigma,
    grad_value,
    grad_sigma,
    N: tl.constexpr,
    CR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    panel = tl.program_id(1).to(tl.int64)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < CR
    scale = tl.load(sigma + panel * CR + offsets, mask=valid, other=1.0).to(
        tl.float32
    )
    scale_cotangent = tl.load(
        grad_sigma + panel * CR + offsets, mask=valid, other=0.0
    ).to(tl.float32)
    for rhs in range(N):
        pointer = (panel * N + rhs) * CR + offsets
        grad = tl.load(grad_output + pointer, mask=valid, other=0.0).to(
            tl.float32
        )
        x = tl.load(value + pointer, mask=valid, other=0.0).to(tl.float32)
        tl.store(grad_value + pointer, grad * scale, mask=valid)
        scale_cotangent += grad * x
    tl.store(grad_sigma + panel * CR + offsets, scale_cotangent, mask=valid)


def _primal_reverse(
    grad_output: torch.Tensor,
    lower_cache: torch.Tensor,
    J: torch.Tensor,
    D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    decay: torch.Tensor,
    kappa_h: torch.Tensor,
    kappa_r_lower: torch.Tensor,
    kappa_r_upper: torch.Tensor,
    boundary_h: torch.Tensor,
    boundary_r_lower: torch.Tensor,
    boundary_r_upper: torch.Tensor,
    sigma: torch.Tensor,
    *,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    upper_cotangent = resident_factor_transpose(
        grad_output,
        J,
        D,
        u,
        h,
        decay,
        kappa_h,
        kappa_r_upper,
        boundary_h,
        boundary_r_upper,
        lower=False,
        num_warps=num_warps,
    )
    lower_output_cotangent = torch.empty_like(
        upper_cotangent, dtype=torch.float32
    )
    grad_sigma = torch.empty_like(sigma, dtype=torch.float32)
    panels, rhs_count, chunk_size, width = upper_cotangent.shape
    cr = chunk_size * width
    block = 256
    grid = (triton.cdiv(cr, block), panels)
    _primal_sigma_reverse_kernel[grid](
        upper_cotangent,
        lower_cache,
        sigma,
        lower_output_cotangent,
        grad_sigma,
        N=rhs_count,
        CR=cr,
        BLOCK=block,
        num_warps=4,
    )
    lower_cotangent = resident_factor_transpose(
        lower_output_cotangent,
        J,
        D,
        u,
        h,
        decay,
        kappa_h,
        kappa_r_lower,
        boundary_h,
        boundary_r_lower,
        lower=True,
        num_warps=num_warps,
    )
    return lower_cotangent, upper_cotangent, grad_sigma


def _dual_reverse(
    grad_output: torch.Tensor,
    rhs: torch.Tensor,
    J: torch.Tensor,
    D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    decay: torch.Tensor,
    kappa_h: torch.Tensor,
    kappa_r_lower: torch.Tensor,
    kappa_r_upper: torch.Tensor,
    boundary_h: torch.Tensor,
    boundary_r_lower: torch.Tensor,
    boundary_r_upper: torch.Tensor,
    sigma: torch.Tensor,
    grad_sigma: torch.Tensor,
    *,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = resident_factor_direct(
        rhs,
        J,
        D,
        u,
        h,
        decay,
        kappa_h,
        kappa_r_lower,
        boundary_h,
        boundary_r_lower,
        lower=True,
        transpose=True,
        num_warps=num_warps,
        output_dtype=torch.float16,
    )
    s = torch.empty_like(t, dtype=torch.float16)
    panels, rhs_count, chunk_size, width = t.shape
    cr = chunk_size * width
    block = 256
    grid = (triton.cdiv(cr, block), panels)
    _dual_sigma_forward_kernel[grid](
        t,
        sigma,
        s,
        N=rhs_count,
        CR=cr,
        BLOCK=block,
        num_warps=4,
    )
    grad_s = resident_factor_direct(
        grad_output,
        J,
        D,
        u,
        h,
        decay,
        kappa_h,
        kappa_r_upper,
        boundary_h,
        boundary_r_upper,
        lower=False,
        transpose=False,
        num_warps=num_warps,
    )
    grad_t = torch.empty_like(grad_s, dtype=torch.float32)
    _dual_sigma_reverse_kernel[grid](
        grad_s,
        t,
        sigma,
        grad_t,
        grad_sigma,
        N=rhs_count,
        CR=cr,
        BLOCK=block,
        num_warps=4,
    )
    grad_rhs = resident_factor_direct(
        grad_t,
        J,
        D,
        u,
        h,
        decay,
        kappa_h,
        kappa_r_lower,
        boundary_h,
        boundary_r_lower,
        lower=True,
        transpose=False,
        num_warps=num_warps,
    )
    return grad_rhs, s, grad_t


def _combine_action_grads(
    grad_j: torch.Tensor,
    grad_d: torch.Tensor,
    grad_u: torch.Tensor,
    grad_h: torch.Tensor,
    upper: tuple[torch.Tensor, ...],
    lower: tuple[torch.Tensor, ...],
    grad_sigma: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        grad_j,
        grad_d,
        grad_u,
        grad_h,
        upper[0],
        lower[1],
        upper[1],
        upper[2],
        grad_sigma,
    )


class _ResidentFrameActions(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        primal_rhs: torch.Tensor,
        dual_rhs: torch.Tensor,
        J: torch.Tensor,
        D: torch.Tensor,
        u: torch.Tensor,
        h: torch.Tensor,
        decay: torch.Tensor,
        boundary_h: torch.Tensor,
        boundary_r_lower: torch.Tensor,
        boundary_r_upper: torch.Tensor,
        sigma: torch.Tensor,
        cumulative: torch.Tensor,
        mass: torch.Tensor,
        kappa_h: torch.Tensor,
        kappa_r_lower: torch.Tensor,
        kappa_r_upper: torch.Tensor,
        num_warps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        primal, lower_cache, final_cache = resident_primal(
            primal_rhs,
            J,
            D,
            u,
            h,
            decay,
            kappa_h,
            kappa_r_lower,
            kappa_r_upper,
            boundary_h,
            boundary_r_lower,
            boundary_r_upper,
            sigma,
            num_warps=num_warps,
        )
        dual = resident_dual(
            dual_rhs,
            J,
            D,
            u,
            h,
            decay,
            kappa_h,
            kappa_r_lower,
            kappa_r_upper,
            boundary_h,
            boundary_r_lower,
            boundary_r_upper,
            sigma,
            num_warps=num_warps,
        )
        ctx.save_for_backward(
            lower_cache,
            final_cache,
            dual_rhs,
            J,
            D,
            u,
            h,
            decay,
            boundary_h,
            boundary_r_lower,
            boundary_r_upper,
            sigma,
            cumulative,
            mass,
            kappa_h,
            kappa_r_lower,
            kappa_r_upper,
        )
        ctx.num_warps = num_warps
        ctx.primal_rhs_dtype = primal_rhs.dtype
        ctx.dual_rhs_dtype = dual_rhs.dtype
        return primal, dual

    @staticmethod
    def backward(ctx, grad_primal: torch.Tensor, grad_dual: torch.Tensor):
        (
            lower_cache,
            final_cache,
            dual_rhs,
            J,
            D,
            u,
            h,
            decay,
            boundary_h,
            boundary_r_lower,
            boundary_r_upper,
            sigma,
            cumulative,
            mass,
            kappa_h,
            kappa_r_lower,
            kappa_r_upper,
        ) = ctx.saved_tensors
        grad_j = torch.empty_like(J, dtype=torch.float32)
        grad_d = torch.empty_like(D, dtype=torch.float32)
        grad_u = torch.empty_like(u, dtype=torch.float32)
        grad_h = torch.empty_like(h, dtype=torch.float32)
        primal_rhs, primal_upper_cotangent, primal_sigma = _primal_reverse(
            grad_primal,
            lower_cache,
            J,
            D,
            u,
            h,
            decay,
            kappa_h,
            kappa_r_lower,
            kappa_r_upper,
            boundary_h,
            boundary_r_lower,
            boundary_r_upper,
            sigma,
            num_warps=ctx.num_warps,
        )
        dual_rhs_grad, dual_upper_cotangent, dual_lower_input = _dual_reverse(
            grad_dual,
            dual_rhs,
            J,
            D,
            u,
            h,
            decay,
            kappa_h,
            kappa_r_lower,
            kappa_r_upper,
            boundary_h,
            boundary_r_lower,
            boundary_r_upper,
            sigma,
            primal_sigma,
            num_warps=ctx.num_warps,
        )
        upper = packed_factor_boundary_vjp(
            final_cache,
            primal_upper_cotangent,
            grad_dual,
            dual_upper_cotangent,
            J,
            D,
            boundary_h,
            boundary_r_upper,
            cumulative,
            mass,
            grad_j,
            grad_d,
            lower=False,
            accumulate=False,
            num_warps=ctx.num_warps,
        )
        factor_local_representation_vjp(
            final_cache,
            primal_upper_cotangent,
            u,
            h,
            decay,
            kappa_h,
            kappa_r_upper,
            mass,
            grad_u,
            grad_h,
            upper[0],
            upper[1],
            upper[2],
            primal=True,
            lower=False,
            accumulate=False,
            num_warps=ctx.num_warps,
        )
        factor_local_representation_vjp(
            grad_dual,
            dual_upper_cotangent,
            u,
            h,
            decay,
            kappa_h,
            kappa_r_upper,
            mass,
            grad_u,
            grad_h,
            upper[0],
            upper[1],
            upper[2],
            primal=False,
            lower=False,
            accumulate=True,
            num_warps=ctx.num_warps,
        )
        lower = packed_factor_boundary_vjp(
            lower_cache,
            primal_rhs,
            dual_lower_input,
            dual_rhs,
            J,
            D,
            boundary_h,
            boundary_r_lower,
            cumulative,
            mass,
            grad_j,
            grad_d,
            lower=True,
            accumulate=True,
            shared_kappa_h=upper[0],
            shared_cumulative=upper[2],
            num_warps=ctx.num_warps,
        )
        factor_local_representation_vjp(
            lower_cache,
            primal_rhs,
            u,
            h,
            decay,
            kappa_h,
            kappa_r_lower,
            mass,
            grad_u,
            grad_h,
            lower[0],
            lower[1],
            lower[2],
            primal=True,
            lower=True,
            accumulate=True,
            num_warps=ctx.num_warps,
        )
        factor_local_representation_vjp(
            dual_lower_input,
            dual_rhs,
            u,
            h,
            decay,
            kappa_h,
            kappa_r_lower,
            mass,
            grad_u,
            grad_h,
            lower[0],
            lower[1],
            lower[2],
            primal=False,
            lower=True,
            accumulate=True,
            num_warps=ctx.num_warps,
        )
        shared = _combine_action_grads(
            grad_j,
            grad_d,
            grad_u,
            grad_h,
            upper,
            lower,
            primal_sigma,
        )
        return (
            primal_rhs.to(ctx.primal_rhs_dtype),
            dual_rhs_grad.to(ctx.dual_rhs_dtype),
            shared[0],
            shared[1],
            shared[2],
            shared[3],
            None,
            None,
            None,
            None,
            shared[8],
            shared[7],
            None,
            shared[4],
            shared[5],
            shared[6],
            None,
        )


def _resident_frame_actions(
    primal_rhs: torch.Tensor,
    dual_rhs: torch.Tensor,
    geometry: _ChunkGeometry,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _ResidentFrameActions.apply(
        primal_rhs,
        dual_rhs,
        geometry.J,
        geometry.D,
        geometry.u,
        geometry.h,
        geometry.decay,
        geometry.boundary_h,
        geometry.boundary_r_lower,
        geometry.boundary_r_upper,
        geometry.sigma,
        geometry.cumulative,
        geometry.mass,
        geometry.kappa_h,
        geometry.kappa_r_lower,
        geometry.kappa_r_upper,
        num_warps,
    )


def bounded_frame_panels(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    erase_raw: torch.Tensor,
    geometry_strength: torch.Tensor,
    *,
    initial_state: SolveDeltaState | None,
    lengths: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int,
    exterior_dtype: torch.dtype = torch.bfloat16,
    num_warps: int = 4,
) -> FramePanels:
    """Construct matrix-free bounded-LDU frame panels for one sequence batch."""
    if u.ndim != 4 or h.shape != u.shape or q.shape != u.shape:
        raise ValueError("u, h, and q must share [B,T,H,r]")
    if keys.ndim != 5 or keys.shape[:3] != u.shape[:3] or keys.shape[-1] != u.shape[-1]:
        raise ValueError("keys must have shape [B,T,H,K,r]")
    if erase_raw.shape != keys.shape:
        raise ValueError("erase_raw must match keys")
    if geometry_log_decay.shape != u.shape[:3]:
        raise ValueError("geometry_log_decay must have shape [B,T,H]")
    if geometry_log_decay.dtype != torch.float32:
        raise TypeError("geometry_log_decay must be FP32")
    if exterior_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("exterior_dtype must be FP16 or BF16")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")

    (
        u_panel,
        u_d_panel,
        h_panel,
        key_panel,
        dual_input,
    ) = normalize_frame_inputs(
        u,
        h,
        q,
        keys,
        erase_raw,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    geometry, final_geometry = _build_geometry(
        u_panel,
        u_d_panel,
        h_panel,
        geometry_log_decay,
        geometry_strength,
        initial_state,
        lengths,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_indices,
        chunk_size=chunk_size,
    )
    d, paired_dual = _resident_frame_actions(
        key_panel, dual_input, geometry, num_warps
    )
    if exterior_dtype != torch.bfloat16:
        d = d.to(exterior_dtype)
        paired_dual = paired_dual.to(exterior_dtype)
    m_final, j_final, d_final = final_geometry
    return FramePanels(d, paired_dual, m_final, j_final, d_final)


__all__ = ["FramePanels", "bounded_frame_panels"]
