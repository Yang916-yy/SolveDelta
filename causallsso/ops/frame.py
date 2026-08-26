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
    boundary_representation_vjp,
    boundary_route_forward,
    boundary_route_vjp,
    local_representation_vjp,
    local_symmetric_representation_vjp,
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
    omega_h: torch.Tensor
    omega_r_lower: torch.Tensor
    omega_r_upper: torch.Tensor
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
    u_j: torch.Tensor,
    u_d: torch.Tensor,
    u_panel: torch.Tensor,
    h: torch.Tensor,
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
    batch, length, heads, width = u_j.shape
    state_batch = batch if cu_seqlens is None else len(cu_seqlens) - 1
    chunks = (length + chunk_size - 1) // chunk_size
    padded_length = chunks * chunk_size
    shape = (batch, length, heads, chunks, chunk_size)

    if initial_state is None:
        initial_m = torch.zeros(
            state_batch, heads, dtype=torch.float32, device=u_j.device
        )
        initial_j = torch.zeros(
            state_batch,
            heads,
            width,
            width,
            dtype=torch.float32,
            device=u_j.device,
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
        u_j,
        u_d,
        u_panel,
        h,
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
    if cu_seqlens is None:
        strength = (
            geometry_strength.float()
            .view(1, heads, 1, 1)
            .expand(batch, -1, chunks, chunk_size)
            .reshape(-1, chunk_size)
        )
    else:
        strength = (
            geometry_strength.float()
            .view(1, heads, 1)
            .expand(u_panel.shape[0] // heads, -1, chunk_size)
            .reshape(-1, chunk_size)
        )
    diagonal_j = J.diagonal(dim1=-2, dim2=-1)
    diagonal_d = D.diagonal(dim1=-2, dim2=-1)
    (
        omega_h,
        omega_r_lower,
        omega_r_upper,
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
    )

    geometry = _ChunkGeometry(
        u=u_panel,
        h=h_panel,
        J=J,
        D=D,
        omega_h=omega_h,
        omega_r_lower=omega_r_lower,
        omega_r_upper=omega_r_upper,
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


def _factor_representation_vjp(
    x: torch.Tensor,
    cotangent: torch.Tensor,
    J: torch.Tensor,
    D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    omega_h: torch.Tensor,
    omega_r: torch.Tensor,
    boundary_h: torch.Tensor,
    boundary_r: torch.Tensor,
    cumulative: torch.Tensor,
    mass: torch.Tensor,
    grad_j: torch.Tensor,
    grad_d: torch.Tensor,
    grad_u: torch.Tensor,
    grad_h: torch.Tensor,
    *,
    lower: bool,
    sign: int,
    accumulate: bool,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim == 3:
        x4 = x[:, None]
        z4 = cotangent[:, None]
    else:
        x4 = x
        z4 = cotangent
    boundary_kappa_h, boundary_kappa_r, boundary_g = (
        boundary_representation_vjp(
            x4,
            z4,
            J,
            D,
            boundary_h,
            boundary_r,
            cumulative,
            mass,
            grad_j,
            grad_d,
            lower=lower,
            sign=sign,
            accumulate=accumulate,
            num_warps=num_warps,
        )
    )

    local_kappa_h, local_g_h = local_symmetric_representation_vjp(
        x4,
        z4,
        u,
        omega_h,
        cumulative,
        mass,
        grad_u,
        lower=lower,
        sign=sign,
        accumulate=accumulate,
        num_warps=num_warps,
    )
    local_kappa_r, local_g_r = local_representation_vjp(
        x4,
        z4,
        u,
        h,
        omega_r,
        cumulative,
        mass,
        grad_u,
        grad_h,
        lower=lower,
        sign=sign,
        accumulate_a=True,
        accumulate_b=accumulate,
        num_warps=num_warps,
    )
    return (
        boundary_kappa_h + local_kappa_h,
        boundary_kappa_r + local_kappa_r,
        boundary_g + local_g_h + local_g_r,
    )


@triton.jit
def _sum_two_kernel(a, b, output, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    value = tl.load(a + offsets, mask=mask, other=0.0).to(tl.float32)
    value += tl.load(b + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _sum_chart_scalars_kernel(
    h0,
    h1,
    h2,
    h3,
    lower0,
    lower1,
    upper0,
    upper1,
    g0,
    g1,
    g2,
    g3,
    out_h,
    out_lower,
    out_upper,
    out_g,
    N,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vh = tl.load(h0 + offsets, mask=mask, other=0.0).to(tl.float32)
    vh += tl.load(h1 + offsets, mask=mask, other=0.0).to(tl.float32)
    vh += tl.load(h2 + offsets, mask=mask, other=0.0).to(tl.float32)
    vh += tl.load(h3 + offsets, mask=mask, other=0.0).to(tl.float32)
    vl = tl.load(lower0 + offsets, mask=mask, other=0.0).to(tl.float32)
    vl += tl.load(lower1 + offsets, mask=mask, other=0.0).to(tl.float32)
    vu = tl.load(upper0 + offsets, mask=mask, other=0.0).to(tl.float32)
    vu += tl.load(upper1 + offsets, mask=mask, other=0.0).to(tl.float32)
    vg = tl.load(g0 + offsets, mask=mask, other=0.0).to(tl.float32)
    vg += tl.load(g1 + offsets, mask=mask, other=0.0).to(tl.float32)
    vg += tl.load(g2 + offsets, mask=mask, other=0.0).to(tl.float32)
    vg += tl.load(g3 + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_h + offsets, vh, mask=mask)
    tl.store(out_lower + offsets, vl, mask=mask)
    tl.store(out_upper + offsets, vu, mask=mask)
    tl.store(out_g + offsets, vg, mask=mask)


def _primal_reverse(
    grad_output: torch.Tensor,
    lower_cache: torch.Tensor,
    final_cache: torch.Tensor,
    J: torch.Tensor,
    D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    omega_h: torch.Tensor,
    omega_r_lower: torch.Tensor,
    omega_r_upper: torch.Tensor,
    boundary_h: torch.Tensor,
    boundary_r_lower: torch.Tensor,
    boundary_r_upper: torch.Tensor,
    sigma: torch.Tensor,
    cumulative: torch.Tensor,
    mass: torch.Tensor,
    grad_j: torch.Tensor,
    grad_d: torch.Tensor,
    grad_u: torch.Tensor,
    grad_h: torch.Tensor,
    *,
    accumulate: bool,
    num_warps: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]:
    upper_cotangent = resident_factor_transpose(
        grad_output,
        J,
        D,
        u,
        h,
        omega_h,
        omega_r_upper,
        boundary_h,
        boundary_r_upper,
        lower=False,
        num_warps=num_warps,
    )
    upper_grads = _factor_representation_vjp(
        final_cache,
        upper_cotangent,
        J,
        D,
        u,
        h,
        omega_h,
        omega_r_upper,
        boundary_h,
        boundary_r_upper,
        cumulative,
        mass,
        grad_j,
        grad_d,
        grad_u,
        grad_h,
        lower=False,
        sign=-1,
        accumulate=accumulate,
        num_warps=num_warps,
    )
    sigma_view = sigma[:, None].float()
    diagonal = lower_cache.float() / sigma_view
    lower_output_cotangent = upper_cotangent / sigma_view
    grad_sigma = -(upper_cotangent * diagonal / sigma_view).sum(dim=1)
    lower_cotangent = resident_factor_transpose(
        lower_output_cotangent,
        J,
        D,
        u,
        h,
        omega_h,
        omega_r_lower,
        boundary_h,
        boundary_r_lower,
        lower=True,
        num_warps=num_warps,
    )
    lower_grads = _factor_representation_vjp(
        lower_cache,
        lower_cotangent,
        J,
        D,
        u,
        h,
        omega_h,
        omega_r_lower,
        boundary_h,
        boundary_r_lower,
        cumulative,
        mass,
        grad_j,
        grad_d,
        grad_u,
        grad_h,
        lower=True,
        sign=-1,
        accumulate=True,
        num_warps=num_warps,
    )
    return lower_cotangent, upper_grads, lower_grads, grad_sigma


def _dual_reverse(
    grad_output: torch.Tensor,
    rhs: torch.Tensor,
    J: torch.Tensor,
    D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    omega_h: torch.Tensor,
    omega_r_lower: torch.Tensor,
    omega_r_upper: torch.Tensor,
    boundary_h: torch.Tensor,
    boundary_r_lower: torch.Tensor,
    boundary_r_upper: torch.Tensor,
    sigma: torch.Tensor,
    cumulative: torch.Tensor,
    mass: torch.Tensor,
    grad_j: torch.Tensor,
    grad_d: torch.Tensor,
    grad_u: torch.Tensor,
    grad_h: torch.Tensor,
    *,
    accumulate: bool,
    num_warps: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]:
    t = resident_factor_direct(
        rhs,
        J,
        D,
        u,
        h,
        omega_h,
        omega_r_lower,
        boundary_h,
        boundary_r_lower,
        lower=True,
        transpose=True,
        num_warps=num_warps,
    ).to(torch.float16)
    sigma_view = sigma[:, None].float()
    s = (t.float() * sigma_view).to(torch.float16)
    grad_s = resident_factor_direct(
        grad_output,
        J,
        D,
        u,
        h,
        omega_h,
        omega_r_upper,
        boundary_h,
        boundary_r_upper,
        lower=False,
        transpose=False,
        num_warps=num_warps,
    )
    upper_grads = _factor_representation_vjp(
        grad_output,
        s,
        J,
        D,
        u,
        h,
        omega_h,
        omega_r_upper,
        boundary_h,
        boundary_r_upper,
        cumulative,
        mass,
        grad_j,
        grad_d,
        grad_u,
        grad_h,
        lower=False,
        sign=1,
        accumulate=accumulate,
        num_warps=num_warps,
    )
    grad_t = grad_s * sigma_view
    grad_sigma = (grad_s * t.float()).sum(dim=1)
    lower_grads = _factor_representation_vjp(
        grad_t,
        rhs,
        J,
        D,
        u,
        h,
        omega_h,
        omega_r_lower,
        boundary_h,
        boundary_r_lower,
        cumulative,
        mass,
        grad_j,
        grad_d,
        grad_u,
        grad_h,
        lower=True,
        sign=1,
        accumulate=True,
        num_warps=num_warps,
    )
    grad_rhs = resident_factor_direct(
        grad_t,
        J,
        D,
        u,
        h,
        omega_h,
        omega_r_lower,
        boundary_h,
        boundary_r_lower,
        lower=True,
        transpose=False,
        num_warps=num_warps,
    )
    return grad_rhs, upper_grads, lower_grads, grad_sigma


def _combine_action_grads(
    grad_j: torch.Tensor,
    grad_d: torch.Tensor,
    grad_u: torch.Tensor,
    grad_h: torch.Tensor,
    primal_upper: tuple[torch.Tensor, ...],
    primal_lower: tuple[torch.Tensor, ...],
    dual_upper: tuple[torch.Tensor, ...],
    dual_lower: tuple[torch.Tensor, ...],
    primal_sigma: torch.Tensor,
    dual_sigma: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    block = 256
    grad_sigma = torch.empty_like(primal_sigma)
    sigma_elements = grad_sigma.numel()
    _sum_two_kernel[(triton.cdiv(sigma_elements, block),)](
        primal_sigma,
        dual_sigma,
        grad_sigma,
        N=sigma_elements,
        BLOCK=block,
        num_warps=4,
    )

    grad_kappa_h = torch.empty_like(primal_upper[0])
    grad_kappa_lower = torch.empty_like(primal_lower[1])
    grad_kappa_upper = torch.empty_like(primal_upper[1])
    grad_cumulative = torch.empty_like(primal_upper[2])
    scalar_elements = grad_kappa_h.numel()
    _sum_chart_scalars_kernel[(triton.cdiv(scalar_elements, block),)](
        primal_upper[0], primal_lower[0], dual_upper[0], dual_lower[0],
        primal_lower[1], dual_lower[1], primal_upper[1], dual_upper[1],
        primal_upper[2], primal_lower[2], dual_upper[2], dual_lower[2],
        grad_kappa_h, grad_kappa_lower, grad_kappa_upper, grad_cumulative,
        N=scalar_elements, BLOCK=block, num_warps=4,
    )
    return (
        grad_j,
        grad_d,
        grad_u,
        grad_h,
        grad_kappa_h,
        grad_kappa_lower,
        grad_kappa_upper,
        grad_cumulative,
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
        omega_h: torch.Tensor,
        omega_r_lower: torch.Tensor,
        omega_r_upper: torch.Tensor,
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
            omega_h,
            omega_r_lower,
            omega_r_upper,
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
            omega_h,
            omega_r_lower,
            omega_r_upper,
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
            omega_h,
            omega_r_lower,
            omega_r_upper,
            boundary_h,
            boundary_r_lower,
            boundary_r_upper,
            sigma,
            cumulative,
            mass,
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
            omega_h,
            omega_r_lower,
            omega_r_upper,
            boundary_h,
            boundary_r_lower,
            boundary_r_upper,
            sigma,
            cumulative,
            mass,
        ) = ctx.saved_tensors
        grad_j = torch.empty_like(J, dtype=torch.float32)
        grad_d = torch.empty_like(D, dtype=torch.float32)
        grad_u = torch.empty_like(u, dtype=torch.float32)
        grad_h = torch.empty_like(h, dtype=torch.float32)
        primal_rhs, primal_upper, primal_lower, primal_sigma = _primal_reverse(
            grad_primal,
            lower_cache,
            final_cache,
            J,
            D,
            u,
            h,
            omega_h,
            omega_r_lower,
            omega_r_upper,
            boundary_h,
            boundary_r_lower,
            boundary_r_upper,
            sigma,
            cumulative,
            mass,
            grad_j,
            grad_d,
            grad_u,
            grad_h,
            accumulate=False,
            num_warps=ctx.num_warps,
        )
        dual_rhs_grad, dual_upper, dual_lower, dual_sigma = _dual_reverse(
            grad_dual,
            dual_rhs,
            J,
            D,
            u,
            h,
            omega_h,
            omega_r_lower,
            omega_r_upper,
            boundary_h,
            boundary_r_lower,
            boundary_r_upper,
            sigma,
            cumulative,
            mass,
            grad_j,
            grad_d,
            grad_u,
            grad_h,
            accumulate=True,
            num_warps=ctx.num_warps,
        )
        shared = _combine_action_grads(
            grad_j,
            grad_d,
            grad_u,
            grad_h,
            primal_upper,
            primal_lower,
            dual_upper,
            dual_lower,
            primal_sigma,
            dual_sigma,
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
        geometry.omega_h,
        geometry.omega_r_lower,
        geometry.omega_r_upper,
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
        u_j,
        u_d,
        u_panel,
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
        u_j,
        u_d,
        u_panel,
        h,
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
