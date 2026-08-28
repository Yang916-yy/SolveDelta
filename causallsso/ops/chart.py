from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..reference import C_H, C_R, S_H, S_R


@triton.jit
def _route_forward_value(
    gram,
    correlation,
    boundary_norm,
    w,
    a,
    safe_m,
    gamma,
    valid,
    pair,
    pair_mask,
    panel,
    oi,
    C: tl.constexpr,
    RADIUS: tl.constexpr,
):
    gram_value = tl.load(
        gram + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    correlation_value = tl.load(
        correlation + panel * C + oi, mask=oi < C, other=0.0
    ).to(tl.float32)
    norm = tl.load(boundary_norm + panel).to(tl.float32)
    product = tl.dot(w, gram_value, input_precision="ieee")
    radial = a * a * norm
    radial += 2.0 * a * tl.sum(w * correlation_value[None, :], axis=1)
    radial += tl.sum(product * w, axis=1)
    denominator = tl.sqrt(RADIUS * RADIUS * safe_m * safe_m + gamma * gamma * radial)
    kappa = tl.where(valid, RADIUS * gamma / denominator, 0.0)
    return kappa * a, kappa


@triton.jit
def _routes_forward_kernel(
    cumulative,
    mass,
    gram_h,
    gram_lower,
    gram_upper,
    corr_h,
    corr_lower,
    corr_upper,
    norm_h,
    norm_lower,
    norm_upper,
    strength,
    decay,
    boundary_h,
    boundary_lower,
    boundary_upper,
    kappa_h,
    kappa_lower,
    kappa_upper,
    C: tl.constexpr,
    BC: tl.constexpr,
    RADIUS: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    oi = tl.arange(0, BC)
    valid = tl.load(mass + panel * C + oi, mask=oi < C, other=0.0) > 0.0
    g = tl.load(cumulative + panel * C + oi, mask=oi < C, other=0.0).to(tl.float32)
    pair_mask = (oi[:, None] < C) & (oi[None, :] < C)
    pair = panel * C * C + oi[:, None] * C + oi[None, :]
    w = tl.exp(g[:, None] - g[None, :])
    w = tl.where(
        (oi[:, None] >= oi[None, :]) & valid[:, None] & valid[None, :],
        w,
        0.0,
    )
    tl.store(decay + pair, w, mask=pair_mask)
    a = tl.exp(g)
    head = panel % H if IS_VARLEN else (panel // N) % H
    gamma_scalar = tl.load(strength + head).to(tl.float32)
    gamma = tl.where(oi < C, gamma_scalar, 0.0)
    m = tl.load(mass + panel * C + oi, mask=oi < C, other=1.0).to(tl.float32)
    safe_m = tl.where(valid, m, 1.0)
    bh, kh = _route_forward_value(
        gram_h, corr_h, norm_h, w, a, safe_m, gamma, valid, pair,
        pair_mask, panel, oi, C, RADIUS,
    )
    bl, kl = _route_forward_value(
        gram_lower, corr_lower, norm_lower, w, a, safe_m, gamma, valid,
        pair, pair_mask, panel, oi, C, RADIUS,
    )
    bu, ku = _route_forward_value(
        gram_upper, corr_upper, norm_upper, w, a, safe_m, gamma, valid,
        pair, pair_mask, panel, oi, C, RADIUS,
    )
    output = panel * C + oi
    output_mask = oi < C
    tl.store(boundary_h + output, bh, mask=output_mask)
    tl.store(boundary_lower + output, bl, mask=output_mask)
    tl.store(boundary_upper + output, bu, mask=output_mask)
    tl.store(kappa_h + output, kh, mask=output_mask)
    tl.store(kappa_lower + output, kl, mask=output_mask)
    tl.store(kappa_upper + output, ku, mask=output_mask)


@triton.jit
def _route_backward_value(
    gram,
    correlation,
    boundary_norm,
    grad_kappa_input,
    w,
    a,
    m,
    safe_m,
    gamma,
    valid,
    pair,
    pair_mask,
    panel,
    oi,
    C: tl.constexpr,
    RADIUS: tl.constexpr,
):
    gv = tl.load(gram + pair, mask=pair_mask, other=0.0).to(tl.float32)
    cv = tl.load(correlation + panel * C + oi, mask=oi < C, other=0.0).to(tl.float32)
    n0 = tl.load(boundary_norm + panel).to(tl.float32)
    wg = tl.dot(w, gv, input_precision="ieee")
    radial = a * a * n0 + 2.0 * a * tl.sum(w * cv[None, :], axis=1)
    radial += tl.sum(wg * w, axis=1)
    den2 = RADIUS * RADIUS * safe_m * safe_m + gamma * gamma * radial
    den = tl.sqrt(den2)
    inv_den3 = 1.0 / (den2 * den)
    grad_kappa = tl.load(
        grad_kappa_input + panel * C + oi, mask=oi < C, other=0.0
    ).to(tl.float32)
    grad_radial = tl.where(
        valid,
        -0.5 * grad_kappa * RADIUS * gamma * gamma * gamma * inv_den3,
        0.0,
    )
    gm = tl.where(valid, -grad_kappa * RADIUS**3 * gamma * m * inv_den3, 0.0)
    gs = tl.where(valid, grad_kappa * RADIUS**3 * m * m * inv_den3, 0.0)

    grad_n0 = tl.sum(grad_radial * a * a, axis=0)
    grad_c = tl.sum(
        2.0 * grad_radial[:, None] * a[:, None] * w,
        axis=0,
    )
    weighted_w = grad_radial[:, None] * w
    grad_gm = tl.dot(tl.trans(w), weighted_w, input_precision="ieee")
    grad_a = grad_radial * (
        2.0 * a * n0 + 2.0 * tl.sum(w * cv[None, :], axis=1)
    )
    grad_w = grad_radial[:, None] * (
        2.0 * a[:, None] * cv[None, :] + wg + tl.dot(w, tl.trans(gv), input_precision="ieee")
    )
    grad_g = grad_a * a
    grad_g += tl.sum(grad_w * w, axis=1) - tl.sum(grad_w * w, axis=0)
    grad_g = tl.where(valid, grad_g, 0.0)
    return gm, gs, grad_g, grad_gm, grad_c, grad_n0


@triton.jit
def _routes_backward_kernel(
    cumulative,
    decay,
    mass,
    gram_h,
    gram_lower,
    gram_upper,
    corr_h,
    corr_lower,
    corr_upper,
    norm_h,
    norm_lower,
    norm_upper,
    strength,
    grad_kappa_h,
    grad_kappa_lower,
    grad_kappa_upper,
    grad_mass,
    grad_gram_h,
    grad_gram_lower,
    grad_gram_upper,
    grad_corr_h,
    grad_corr_lower,
    grad_corr_upper,
    grad_norm_h,
    grad_norm_lower,
    grad_norm_upper,
    grad_strength,
    grad_cumulative,
    C: tl.constexpr,
    BC: tl.constexpr,
    RADIUS: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    oi = tl.arange(0, BC)
    row_mask = oi < C
    valid = tl.load(mass + panel * C + oi, mask=row_mask, other=0.0) > 0.0
    g = tl.load(cumulative + panel * C + oi, mask=row_mask, other=0.0).to(tl.float32)
    pair_mask = row_mask[:, None] & row_mask[None, :]
    pair = panel * C * C + oi[:, None] * C + oi[None, :]
    w = tl.load(decay + pair, mask=pair_mask, other=0.0).to(tl.float32)
    a = tl.exp(g)
    head = panel % H if IS_VARLEN else (panel // N) % H
    gamma_scalar = tl.load(strength + head).to(tl.float32)
    gamma = tl.where(row_mask, gamma_scalar, 0.0)
    m = tl.load(mass + panel * C + oi, mask=row_mask, other=1.0).to(tl.float32)
    safe_m = tl.where(valid, m, 1.0)

    gm_h, gs_h, gg_h, ggram_h, gc_h, gn_h = _route_backward_value(
        gram_h, corr_h, norm_h, grad_kappa_h, w, a, m, safe_m, gamma,
        valid, pair, pair_mask, panel, oi, C, RADIUS,
    )
    gm_l, gs_l, gg_l, ggram_l, gc_l, gn_l = _route_backward_value(
        gram_lower, corr_lower, norm_lower, grad_kappa_lower, w, a, m,
        safe_m, gamma, valid, pair, pair_mask, panel, oi, C, RADIUS,
    )
    gm_u, gs_u, gg_u, ggram_u, gc_u, gn_u = _route_backward_value(
        gram_upper, corr_upper, norm_upper, grad_kappa_upper, w, a, m,
        safe_m, gamma, valid, pair, pair_mask, panel, oi, C, RADIUS,
    )

    gm = gm_h + gm_l + gm_u
    gs = gs_h + gs_l + gs_u
    grad_g = gg_h + gg_l + gg_u
    gm += tl.load(grad_mass + panel * C + oi, mask=row_mask, other=0.0).to(tl.float32)
    gs += tl.load(grad_strength + panel * C + oi, mask=row_mask, other=0.0).to(tl.float32)
    grad_g += tl.load(
        grad_cumulative + panel * C + oi, mask=row_mask, other=0.0
    ).to(tl.float32)

    tl.store(grad_mass + panel * C + oi, gm, mask=row_mask)
    tl.store(grad_strength + panel * C + oi, gs, mask=row_mask)
    tl.store(grad_cumulative + panel * C + oi, grad_g, mask=row_mask)
    tl.store(grad_gram_h + pair, ggram_h, mask=pair_mask)
    tl.store(grad_gram_lower + pair, ggram_l, mask=pair_mask)
    tl.store(grad_gram_upper + pair, ggram_u, mask=pair_mask)
    tl.store(grad_corr_h + panel * C + oi, gc_h, mask=row_mask)
    tl.store(grad_corr_lower + panel * C + oi, gc_l, mask=row_mask)
    tl.store(grad_corr_upper + panel * C + oi, gc_u, mask=row_mask)
    tl.store(grad_norm_h + panel, gn_h)
    tl.store(grad_norm_lower + panel, gn_l)
    tl.store(grad_norm_upper + panel, gn_u)


@triton.jit
def _sigma_forward_kernel(
    cumulative,
    decay,
    mass,
    diagonal_j,
    diagonal_d,
    u,
    h,
    strength,
    sigma,
    C: tl.constexpr,
    R: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    SDJ_P: tl.constexpr,
    SDJ_R: tl.constexpr,
    SDD_P: tl.constexpr,
    SDD_R: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    oi = tl.arange(0, BC)
    ok = tl.arange(0, BK)
    valid_c = tl.load(mass + panel * C + oi, mask=oi < C, other=0.0) > 0.0
    valid_k = ok < R
    g = tl.load(cumulative + panel * C + oi, mask=oi < C, other=0.0).to(tl.float32)
    pair = panel * C * C + oi[:, None] * C + oi[None, :]
    w = tl.load(
        decay + pair,
        mask=(oi[:, None] < C) & (oi[None, :] < C),
        other=0.0,
    ).to(tl.float32)
    a = tl.exp(g)
    base = panel * C * R
    uv = tl.load(
        u + base + oi[:, None] * R + ok[None, :],
        mask=valid_c[:, None] & valid_k[None, :],
        other=0.0,
    ).to(tl.float32)
    hv = tl.load(
        h + base + oi[:, None] * R + ok[None, :],
        mask=valid_c[:, None] & valid_k[None, :],
        other=0.0,
    ).to(tl.float32)
    dj0 = tl.load(
        diagonal_j + panel * SDJ_P + ok * SDJ_R,
        mask=valid_k,
        other=0.0,
    ).to(tl.float32)
    dd0 = tl.load(
        diagonal_d + panel * SDD_P + ok * SDD_R,
        mask=valid_k,
        other=0.0,
    ).to(tl.float32)
    local_j = tl.dot(w.to(tl.float16), (uv * uv).to(tl.float16))
    local_d = tl.dot(w.to(tl.bfloat16), (uv * hv).to(tl.bfloat16))
    dj = a[:, None] * dj0[None, :] + local_j
    dd = a[:, None] * dd0[None, :] + local_d
    m = tl.load(mass + panel * C + oi, mask=oi < C, other=1.0).to(tl.float32)
    m = tl.where(valid_c, m, 1.0)
    head = panel % H if IS_VARLEN else (panel // N) % H
    gamma_scalar = tl.load(strength + head).to(tl.float32)
    gamma = tl.where(oi < C, gamma_scalar, 0.0)
    xh = gamma[:, None] * (dj / m[:, None] - 1.0 / R)
    xr = gamma[:, None] * dd / m[:, None]
    th = 2.0 * tl.sigmoid(2.0 * xh / 0.125) - 1.0
    tr = 2.0 * tl.sigmoid(2.0 * xr / 0.125) - 1.0
    ell = 0.125 * th + 0.125 * tr
    result = tl.where(valid_c[:, None], tl.exp(ell), 1.0)
    tl.store(
        sigma + base + oi[:, None] * R + ok[None, :],
        result,
        mask=(oi[:, None] < C) & valid_k[None, :],
    )


@triton.jit
def _sigma_backward_tiled_kernel(
    cumulative,
    decay,
    mass,
    diagonal_j,
    diagonal_d,
    u,
    h,
    strength,
    sigma,
    grad_sigma,
    grad_mass,
    grad_diagonal_j,
    grad_diagonal_d,
    grad_u,
    grad_h,
    grad_strength,
    grad_cumulative,
    C: tl.constexpr,
    R: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NB: tl.constexpr,
    SDJ_P: tl.constexpr,
    SDJ_R: tl.constexpr,
    SDD_P: tl.constexpr,
    SDD_R: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    oi = tl.arange(0, BC)
    ok = tl.arange(0, BK)
    valid_c = tl.load(mass + panel * C + oi, mask=oi < C, other=0.0) > 0.0
    g = tl.load(cumulative + panel * C + oi, mask=oi < C, other=0.0).to(tl.float32)
    pair = panel * C * C + oi[:, None] * C + oi[None, :]
    w = tl.load(
        decay + pair,
        mask=(oi[:, None] < C) & (oi[None, :] < C),
        other=0.0,
    ).to(tl.float32)
    a = tl.exp(g)
    base = panel * C * R
    m = tl.load(mass + panel * C + oi, mask=oi < C, other=1.0).to(tl.float32)
    m = tl.where(valid_c, m, 1.0)
    head = panel % H if IS_VARLEN else (panel // N) % H
    gamma_scalar = tl.load(strength + head).to(tl.float32)
    gamma = tl.where(oi < C, gamma_scalar, 0.0)
    gm = tl.zeros([BC], dtype=tl.float32)
    ggamma = tl.zeros([BC], dtype=tl.float32)
    gg = tl.zeros([BC], dtype=tl.float32)
    for tile in range(NB):
        coord = tile * BK + ok
        valid_k = coord < R
        uv = tl.load(
            u + base + oi[:, None] * R + coord[None, :],
            mask=valid_c[:, None] & valid_k[None, :],
            other=0.0,
        ).to(tl.float32)
        hv = tl.load(
            h + base + oi[:, None] * R + coord[None, :],
            mask=valid_c[:, None] & valid_k[None, :],
            other=0.0,
        ).to(tl.float32)
        dj0 = tl.load(
            diagonal_j + panel * SDJ_P + coord * SDJ_R,
            mask=valid_k,
            other=0.0,
        ).to(tl.float32)
        dd0 = tl.load(
            diagonal_d + panel * SDD_P + coord * SDD_R,
            mask=valid_k,
            other=0.0,
        ).to(tl.float32)
        local_j = tl.dot(w.to(tl.float16), (uv * uv).to(tl.float16))
        local_d = tl.dot(w.to(tl.bfloat16), (uv * hv).to(tl.bfloat16))
        dj = a[:, None] * dj0[None, :] + local_j
        dd = a[:, None] * dd0[None, :] + local_d
        xh = gamma[:, None] * (dj / m[:, None] - 1.0 / R)
        xr = gamma[:, None] * dd / m[:, None]
        th = 2.0 * tl.sigmoid(2.0 * xh / 0.125) - 1.0
        tr = 2.0 * tl.sigmoid(2.0 * xr / 0.125) - 1.0
        sig = tl.load(
            sigma + base + oi[:, None] * R + coord[None, :],
            mask=valid_c[:, None] & valid_k[None, :],
            other=1.0,
        ).to(tl.float32)
        gs = tl.load(
            grad_sigma + base + oi[:, None] * R + coord[None, :],
            mask=valid_c[:, None] & valid_k[None, :],
            other=0.0,
        ).to(tl.float32)
        gell = gs * sig
        gxh = gell * (1.0 - th * th)
        gxr = gell * (1.0 - tr * tr)
        gdj = gamma[:, None] * gxh / m[:, None]
        gdd = gamma[:, None] * gxr / m[:, None]
        gm += -gamma / (m * m) * tl.sum(gxh * dj + gxr * dd, axis=1)
        ggamma += tl.sum(
            gxh * (dj / m[:, None] - 1.0 / R) + gxr * dd / m[:, None],
            axis=1,
        )
        gj0 = tl.sum(gdj * a[:, None], axis=0)
        gd0 = tl.sum(gdd * a[:, None], axis=0)
        coeff_j = tl.dot(tl.trans(w).to(tl.bfloat16), gdj.to(tl.bfloat16))
        coeff_d = tl.dot(tl.trans(w).to(tl.bfloat16), gdd.to(tl.bfloat16))
        gu = 2.0 * uv * coeff_j + hv * coeff_d
        gh = uv * coeff_d
        gw = tl.dot(gdj.to(tl.bfloat16), tl.trans((uv * uv).to(tl.bfloat16)))
        gw += tl.dot(gdd.to(tl.bfloat16), tl.trans((uv * hv).to(tl.bfloat16)))
        ga = tl.sum(gdj * dj0[None, :] + gdd * dd0[None, :], axis=1)
        gg += ga * a + tl.sum(gw * w, axis=1) - tl.sum(gw * w, axis=0)
        tl.store(grad_diagonal_j + panel * R + coord, gj0, mask=valid_k)
        tl.store(grad_diagonal_d + panel * R + coord, gd0, mask=valid_k)
        output = base + oi[:, None] * R + coord[None, :]
        output_mask = valid_c[:, None] & valid_k[None, :]
        tl.store(grad_u + output, gu, mask=output_mask)
        tl.store(grad_h + output, gh, mask=output_mask)
    gm = tl.where(valid_c, gm, 0.0)
    ggamma = tl.where(valid_c, ggamma, 0.0)
    gg = tl.where(valid_c, gg, 0.0)
    tl.store(grad_mass + panel * C + oi, gm, mask=oi < C)
    tl.store(grad_strength + panel * C + oi, ggamma, mask=oi < C)
    tl.store(grad_cumulative + panel * C + oi, gg, mask=oi < C)


class _ChartCoefficients(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cumulative,
        mass,
        diagonal_j,
        diagonal_d,
        u,
        h,
        strength,
        gram_h,
        gram_lower,
        gram_upper,
        corr_h,
        corr_lower,
        corr_upper,
        norm_h,
        norm_lower,
        norm_upper,
        heads,
        chunks,
        is_varlen,
    ):
        panels, chunk_size = mass.shape
        width = u.shape[-1]
        bc = triton.next_power_of_2(chunk_size)
        bk = triton.next_power_of_2(width)
        decay = torch.empty_like(gram_h)
        boundary_h = torch.empty_like(mass)
        boundary_lower = torch.empty_like(mass)
        boundary_upper = torch.empty_like(mass)
        kappa_h = torch.empty_like(mass)
        kappa_lower = torch.empty_like(mass)
        kappa_upper = torch.empty_like(mass)
        _routes_forward_kernel[(panels,)](
            cumulative,
            mass,
            gram_h,
            gram_lower,
            gram_upper,
            corr_h,
            corr_lower,
            corr_upper,
            norm_h,
            norm_lower,
            norm_upper,
            strength,
            decay,
            boundary_h,
            boundary_lower,
            boundary_upper,
            kappa_h,
            kappa_lower,
            kappa_upper,
            C=chunk_size,
            BC=bc,
            RADIUS=C_H,
            H=heads,
            N=chunks,
            IS_VARLEN=is_varlen,
            num_warps=4,
            num_stages=1,
        )
        sigma = torch.empty_like(u, dtype=torch.float16)
        _sigma_forward_kernel[(panels,)](
            cumulative, decay, mass, diagonal_j, diagonal_d, u, h, strength,
            sigma,
            C=chunk_size,
            R=width,
            BC=bc,
            BK=bk,
            SDJ_P=diagonal_j.stride(0),
            SDJ_R=diagonal_j.stride(1),
            SDD_P=diagonal_d.stride(0),
            SDD_R=diagonal_d.stride(1),
            H=heads,
            N=chunks,
            IS_VARLEN=is_varlen,
            num_warps=4,
            num_stages=1,
        )
        ctx.save_for_backward(
            cumulative, mass, diagonal_j, diagonal_d, u, h, strength,
            gram_h, gram_lower, gram_upper, corr_h, corr_lower, corr_upper,
            norm_h, norm_lower, norm_upper, decay, sigma,
        )
        ctx.bc, ctx.bk = bc, bk
        ctx.layout = heads, chunks, is_varlen
        return (
            decay,
            boundary_h,
            boundary_lower,
            boundary_upper,
            sigma,
            kappa_h,
            kappa_lower,
            kappa_upper,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            cumulative, mass, diagonal_j, diagonal_d, u, h, strength,
            gram_h, gram_lower, gram_upper, corr_h, corr_lower, corr_upper,
            norm_h, norm_lower, norm_upper, decay, sigma,
        ) = ctx.saved_tensors
        panels, chunk_size = mass.shape
        width = u.shape[-1]
        heads, chunks, is_varlen = ctx.layout
        grad_mass = torch.empty_like(mass)
        sigma_dj = torch.empty_like(diagonal_j)
        sigma_dd = torch.empty_like(diagonal_d)
        sigma_u = torch.empty_like(u, dtype=torch.float32)
        sigma_h = torch.empty_like(h, dtype=torch.float32)
        grad_strength_panel = torch.empty_like(mass)
        grad_cumulative = torch.empty_like(cumulative)
        sigma_tile = 32
        _sigma_backward_tiled_kernel[(panels,)](
            cumulative, decay, mass, diagonal_j, diagonal_d, u, h, strength,
            sigma, grad_outputs[4], grad_mass, sigma_dj, sigma_dd, sigma_u, sigma_h,
            grad_strength_panel, grad_cumulative,
            C=chunk_size,
            R=width,
            BC=ctx.bc,
            BK=sigma_tile,
            NB=triton.cdiv(width, sigma_tile),
            SDJ_P=diagonal_j.stride(0),
            SDJ_R=diagonal_j.stride(1),
            SDD_P=diagonal_d.stride(0),
            SDD_R=diagonal_d.stride(1),
            H=heads,
            N=chunks,
            IS_VARLEN=is_varlen,
            num_warps=4,
            num_stages=1,
        )

        gram_gradients = tuple(
            torch.empty_like(gram)
            for gram in (gram_h, gram_lower, gram_upper)
        )
        correlation_gradients = tuple(
            torch.empty_like(corr)
            for corr in (corr_h, corr_lower, corr_upper)
        )
        norm_gradients = tuple(
            torch.empty_like(norm)
            for norm in (norm_h, norm_lower, norm_upper)
        )
        _routes_backward_kernel[(panels,)](
            cumulative,
            decay,
            mass,
            gram_h,
            gram_lower,
            gram_upper,
            corr_h,
            corr_lower,
            corr_upper,
            norm_h,
            norm_lower,
            norm_upper,
            strength,
            grad_outputs[5],
            grad_outputs[6],
            grad_outputs[7],
            grad_mass,
            *gram_gradients,
            *correlation_gradients,
            *norm_gradients,
            grad_strength_panel,
            grad_cumulative,
            C=chunk_size,
            BC=ctx.bc,
            RADIUS=C_H,
            H=heads,
            N=chunks,
            IS_VARLEN=is_varlen,
            num_warps=4,
            num_stages=1,
        )
        if is_varlen:
            grad_strength = grad_strength_panel.view(-1, heads, chunk_size).sum(
                dim=(0, 2)
            )
        else:
            batch = panels // (heads * chunks)
            grad_strength = grad_strength_panel.view(
                batch, heads, chunks, chunk_size
            ).sum(dim=(0, 2, 3))
        return (
            grad_cumulative, grad_mass, sigma_dj, sigma_dd, sigma_u, sigma_h,
            grad_strength,
            *gram_gradients,
            *correlation_gradients,
            *norm_gradients,
            None, None, None,
        )


def chart_coefficients(*args):
    return _ChartCoefficients.apply(*args)


__all__ = ["chart_coefficients"]
