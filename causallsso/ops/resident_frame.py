from __future__ import annotations

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# The blocked substitution and two-dot schedules are adapted from
# flash-linear-attention's MIT-licensed GDN2/KDA and MESA kernels at commit
# bc3b101dcb713ddc5bd8924b66754eb68b5ccf89. SolveDelta supplies the
# structured J/D/u/h pair producer and the bounded-LDU orchestration.

import torch
import triton
import triton.language as tl


@triton.jit
def _exact_coordinate_solve_kernel(
    rhs,
    output,
    J,
    D,
    u,
    h,
    omega_h,
    omega_r,
    boundary_h,
    boundary_r,
    sigma,
    stride_rhs_p: tl.constexpr,
    stride_rhs_n: tl.constexpr,
    stride_rhs_c: tl.constexpr,
    stride_rhs_r: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    NRHS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    LOWER: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    DIVIDE_SIGMA: tl.constexpr,
    BC: tl.constexpr = 16,
):
    """Exact blocked substitution for one structured unit-triangular factor.

    This follows FLA's GDN2/solve_tril split: complete coordinate blocks use
    Tensor-Core products and the 16-coordinate diagonal retains ordered
    substitution. The notation-only generalized-Delta features are generated
    from J/D and u/h in place and never stored.
    """
    chunk = tl.program_id(0).to(tl.int64)
    rows = C * NRHS
    o_row = tl.arange(0, BM)
    o_k = tl.arange(0, BK)
    o_s = tl.arange(0, C)
    o_b = tl.arange(0, BC)
    valid_row = o_row < rows
    valid_k = o_k < R
    token = o_row % C
    rhs_index = o_row // C

    p_rhs = (
        chunk * stride_rhs_p
        + rhs_index[:, None] * stride_rhs_n
        + token[:, None] * stride_rhs_c
        + o_k[None, :] * stride_rhs_r
    )
    solution = tl.load(
        rhs + p_rhs,
        mask=valid_row[:, None] & valid_k[None, :],
        other=0.0,
    ).to(tl.float32)

    panel_vector = chunk * C * R
    panel_pair = chunk * C * C
    panel_matrix = chunk * R * R
    weight_h = tl.load(
        omega_h + panel_pair + token[:, None] * C + o_s[None, :],
        mask=valid_row[:, None],
        other=0.0,
    ).to(tl.float32)
    weight_r = tl.load(
        omega_r + panel_pair + token[:, None] * C + o_s[None, :],
        mask=valid_row[:, None],
        other=0.0,
    ).to(tl.float32)
    coeff_h = tl.load(
        boundary_h + chunk * C + token, mask=valid_row, other=0.0
    ).to(tl.float32)
    coeff_r = tl.load(
        boundary_r + chunk * C + token, mask=valid_row, other=0.0
    ).to(tl.float32)

    prefix_h = tl.zeros([BM, C], dtype=tl.float32)
    prefix_r = tl.zeros([BM, C], dtype=tl.float32)
    reverse = LOWER == TRANSPOSE

    for block in range(tl.cdiv(R, BC)):
        if reverse:
            coord = R - 1 - (block * BC + o_b)
            prior = o_k >= R - block * BC
        else:
            coord = block * BC + o_b
            prior = o_k < block * BC
        valid_b = (coord >= 0) & (coord < R)

        p_block_rhs = (
            chunk * stride_rhs_p
            + rhs_index[:, None] * stride_rhs_n
            + token[:, None] * stride_rhs_c
            + coord[None, :] * stride_rhs_r
        )
        rhs_block = tl.load(
            rhs + p_block_rhs,
            mask=valid_row[:, None] & valid_b[None, :],
            other=0.0,
        ).to(tl.float32)
        if DIVIDE_SIGMA:
            b_sigma = tl.load(
                sigma + panel_vector + token[:, None] * R + coord[None, :],
                mask=valid_row[:, None] & valid_b[None, :],
                other=1.0,
            ).to(tl.float32)
            rhs_block /= b_sigma

        if TRANSPOSE:
            p_j = panel_matrix + o_k[None, :] * R + coord[:, None]
            p_d = panel_matrix + o_k[None, :] * R + coord[:, None]
        else:
            p_j = panel_matrix + coord[:, None] * R + o_k[None, :]
            p_d = panel_matrix + coord[:, None] * R + o_k[None, :]
        matrix_mask = valid_b[:, None] & valid_k[None, :] & prior[None, :]
        block_j = tl.load(J + p_j, mask=matrix_mask, other=0.0).to(tl.bfloat16)
        block_d = tl.load(D + p_d, mask=matrix_mask, other=0.0).to(tl.bfloat16)
        solution_bf16 = solution.to(tl.bfloat16)
        boundary_prior = tl.dot(solution_bf16, tl.trans(block_j)) * coeff_h[:, None]
        boundary_prior += tl.dot(solution_bf16, tl.trans(block_d)) * coeff_r[:, None]

        p_u_block = u + panel_vector + o_s[:, None] * R + coord[None, :]
        p_h_block = h + panel_vector + o_s[:, None] * R + coord[None, :]
        u_block = tl.load(
            p_u_block, mask=valid_b[None, :], other=0.0
        ).to(tl.float16)
        h_block = tl.load(
            p_h_block, mask=valid_b[None, :], other=0.0
        ).to(tl.bfloat16)
        if TRANSPOSE:
            r_input_block = u_block.to(tl.bfloat16)
            r_output_block = h_block
        else:
            r_input_block = h_block
            r_output_block = u_block.to(tl.bfloat16)
        local_prior = tl.dot(
            (prefix_h * weight_h).to(tl.float16), u_block
        )
        local_prior += tl.dot(
            (prefix_r * weight_r).to(tl.bfloat16), r_output_block
        )
        work = rhs_block - boundary_prior - local_prior

        if TRANSPOSE:
            p_j_diag = panel_matrix + coord[None, :] * R + coord[:, None]
            p_d_diag = panel_matrix + coord[None, :] * R + coord[:, None]
        else:
            p_j_diag = panel_matrix + coord[:, None] * R + coord[None, :]
            p_d_diag = panel_matrix + coord[:, None] * R + coord[None, :]
        diag_mask = valid_b[:, None] & valid_b[None, :]
        block_j_diag = tl.load(
            J + p_j_diag, mask=diag_mask, other=0.0
        ).to(tl.float32)
        block_d_diag = tl.load(
            D + p_d_diag, mask=diag_mask, other=0.0
        ).to(tl.float32)
        solved_block = tl.zeros([BM, BC], dtype=tl.float32)
        delta_h = tl.zeros([BM, C], dtype=tl.float32)
        delta_r = tl.zeros([BM, C], dtype=tl.float32)
        for step in range(BC):
            rhs_value = tl.sum(
                tl.where(o_b[None, :] == step, work, 0.0), axis=1
            )
            row_j = tl.sum(
                tl.where(o_b[:, None] == step, block_j_diag, 0.0), axis=0
            )
            row_d = tl.sum(
                tl.where(o_b[:, None] == step, block_d_diag, 0.0), axis=0
            )
            within_boundary = tl.sum(solved_block * row_j[None, :], axis=1)
            within_boundary *= coeff_h
            within_boundary += tl.sum(solved_block * row_d[None, :], axis=1) * coeff_r

            u_value = tl.sum(
                tl.where(o_b[None, :] == step, u_block, 0.0), axis=1
            ).to(tl.float32)
            r_input_value = tl.sum(
                tl.where(o_b[None, :] == step, r_input_block, 0.0), axis=1
            ).to(tl.float32)
            r_output_value = tl.sum(
                tl.where(o_b[None, :] == step, r_output_block, 0.0), axis=1
            ).to(tl.float32)
            within_local = tl.sum(
                weight_h * delta_h * u_value[None, :]
                + weight_r * delta_r * r_output_value[None, :],
                axis=1,
            )
            value = tl.where(
                valid_row,
                rhs_value - within_boundary - within_local,
                0.0,
            )
            solved_block = tl.where(
                o_b[None, :] == step, value[:, None], solved_block
            )
            delta_h += value[:, None] * u_value[None, :]
            delta_r += value[:, None] * r_input_value[None, :]
            target = R - 1 - (block * BC + step) if reverse else block * BC + step
            solution = tl.where(o_k[None, :] == target, value[:, None], solution)

        prefix_h += delta_h
        prefix_r += delta_r

    output_offset = chunk * rows * R + o_row[:, None] * R + o_k[None, :]
    tl.store(
        output + output_offset,
        solution,
        mask=valid_row[:, None] & valid_k[None, :],
    )


@triton.jit
def _direct_prefix_states_kernel(
    rhs,
    u,
    h,
    prefix_h_out,
    prefix_r_out,
    sigma,
    stride_rhs_p: tl.constexpr,
    stride_rhs_n: tl.constexpr,
    stride_rhs_c: tl.constexpr,
    stride_rhs_r: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    NRHS: tl.constexpr,
    BM: tl.constexpr,
    BC: tl.constexpr,
    NB: tl.constexpr,
    LOWER: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    MULTIPLY_SIGMA: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    rows = C * NRHS
    o_row = tl.arange(0, BM)
    o_s = tl.arange(0, C)
    o_b = tl.arange(0, BC)
    valid_row = o_row < rows
    token = o_row % C
    rhs_index = o_row // C
    panel_local = panel * C * R
    panel_state = panel * NB * rows * C
    reverse = LOWER == TRANSPOSE
    prefix_h = tl.zeros([BM, C], dtype=tl.float32)
    prefix_r = tl.zeros([BM, C], dtype=tl.float32)
    for tile in range(NB):
        state_offset = panel_state + tile * rows * C
        pointer = state_offset + o_row[:, None] * C + o_s[None, :]
        tl.store(prefix_h_out + pointer, prefix_h, mask=valid_row[:, None])
        tl.store(prefix_r_out + pointer, prefix_r, mask=valid_row[:, None])
        coord = tile * BC + o_b
        if reverse:
            coord = R - 1 - coord
        p_rhs = (
            panel * stride_rhs_p
            + rhs_index[:, None] * stride_rhs_n
            + token[:, None] * stride_rhs_c
            + coord[None, :] * stride_rhs_r
        )
        bx = tl.load(rhs + p_rhs, mask=valid_row[:, None], other=0.0).to(tl.float32)
        if MULTIPLY_SIGMA:
            bs = tl.load(
                sigma + panel_local + token[:, None] * R + coord[None, :],
                mask=valid_row[:, None],
                other=1.0,
            ).to(tl.float32)
            bx *= bs
        u_block = tl.load(
            u + panel_local + o_s[:, None] * R + coord[None, :]
        ).to(tl.float16)
        h_block = tl.load(
            h + panel_local + o_s[:, None] * R + coord[None, :]
        ).to(tl.bfloat16)
        r_input = u_block.to(tl.bfloat16) if TRANSPOSE else h_block
        prefix_h += tl.dot(bx.to(tl.float16), tl.trans(u_block))
        prefix_r += tl.dot(bx.to(tl.bfloat16), tl.trans(r_input))


@triton.jit
def _direct_block_output_kernel(
    rhs,
    output,
    J,
    D,
    u,
    h,
    omega_h,
    omega_r,
    boundary_h,
    boundary_r,
    prefix_h_in,
    prefix_r_in,
    sigma,
    stride_rhs_p: tl.constexpr,
    stride_rhs_n: tl.constexpr,
    stride_rhs_c: tl.constexpr,
    stride_rhs_r: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    NRHS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    NB: tl.constexpr,
    LOWER: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    MULTIPLY_SIGMA: tl.constexpr,
):
    tile = tl.program_id(0)
    panel = tl.program_id(1).to(tl.int64)
    rows = C * NRHS
    o_row = tl.arange(0, BM)
    o_s = tl.arange(0, C)
    o_k = tl.arange(0, BK)
    o_b = tl.arange(0, BC)
    valid_row = o_row < rows
    valid_k = o_k < R
    token = o_row % C
    rhs_index = o_row // C
    reverse = LOWER == TRANSPOSE
    coord = tile * BC + o_b
    if reverse:
        coord = R - 1 - coord
    panel_local = panel * C * R
    panel_pair = panel * C * C
    panel_matrix = panel * R * R
    state_offset = (panel * NB + tile) * rows * C
    prefix_h = tl.load(
        prefix_h_in + state_offset + o_row[:, None] * C + o_s[None, :],
        mask=valid_row[:, None], other=0.0,
    ).to(tl.float32)
    prefix_r = tl.load(
        prefix_r_in + state_offset + o_row[:, None] * C + o_s[None, :],
        mask=valid_row[:, None], other=0.0,
    ).to(tl.float32)
    wh = tl.load(
        omega_h + panel_pair + token[:, None] * C + o_s[None, :],
        mask=valid_row[:, None], other=0.0,
    ).to(tl.float32)
    wr = tl.load(
        omega_r + panel_pair + token[:, None] * C + o_s[None, :],
        mask=valid_row[:, None], other=0.0,
    ).to(tl.float32)
    ch = tl.load(boundary_h + panel * C + token, mask=valid_row, other=0.0).to(tl.float32)
    cr = tl.load(boundary_r + panel * C + token, mask=valid_row, other=0.0).to(tl.float32)
    p_rhs_all = (
        panel * stride_rhs_p + rhs_index[:, None] * stride_rhs_n
        + token[:, None] * stride_rhs_c + o_k[None, :] * stride_rhs_r
    )
    x_all = tl.load(
        rhs + p_rhs_all, mask=valid_row[:, None] & valid_k[None, :], other=0.0
    ).to(tl.float32)
    if MULTIPLY_SIGMA:
        s_all = tl.load(
            sigma + panel_local + token[:, None] * R + o_k[None, :],
            mask=valid_row[:, None] & valid_k[None, :], other=1.0,
        ).to(tl.float32)
        x_all *= s_all
    if TRANSPOSE:
        p_j = panel_matrix + o_k[None, :] * R + coord[:, None]
        p_d = panel_matrix + o_k[None, :] * R + coord[:, None]
    else:
        p_j = panel_matrix + coord[:, None] * R + o_k[None, :]
        p_d = panel_matrix + coord[:, None] * R + o_k[None, :]
    strict = o_k[None, :] > coord[:, None] if reverse else o_k[None, :] < coord[:, None]
    matrix_mask = valid_k[None, :] & strict
    bj = tl.load(J + p_j, mask=matrix_mask, other=0.0).to(tl.bfloat16)
    bd = tl.load(D + p_d, mask=matrix_mask, other=0.0).to(tl.bfloat16)
    boundary = tl.dot(x_all.to(tl.bfloat16), tl.trans(bj)) * ch[:, None]
    boundary += tl.dot(x_all.to(tl.bfloat16), tl.trans(bd)) * cr[:, None]
    p_rhs_block = (
        panel * stride_rhs_p + rhs_index[:, None] * stride_rhs_n
        + token[:, None] * stride_rhs_c + coord[None, :] * stride_rhs_r
    )
    x_block = tl.load(rhs + p_rhs_block, mask=valid_row[:, None], other=0.0).to(tl.float32)
    if MULTIPLY_SIGMA:
        s_block = tl.load(
            sigma + panel_local + token[:, None] * R + coord[None, :],
            mask=valid_row[:, None], other=1.0,
        ).to(tl.float32)
        x_block *= s_block
    u_block = tl.load(u + panel_local + o_s[:, None] * R + coord[None, :]).to(tl.float32)
    h_block = tl.load(h + panel_local + o_s[:, None] * R + coord[None, :]).to(tl.float32)
    r_input = u_block if TRANSPOSE else h_block
    r_output = h_block if TRANSPOSE else u_block
    result = tl.zeros([BM, BC], dtype=tl.float32)
    for step in range(BC):
        xv = tl.sum(tl.where(o_b[None, :] == step, x_block, 0.0), axis=1)
        uv = tl.sum(tl.where(o_b[None, :] == step, u_block, 0.0), axis=1)
        riv = tl.sum(tl.where(o_b[None, :] == step, r_input, 0.0), axis=1)
        rov = tl.sum(tl.where(o_b[None, :] == step, r_output, 0.0), axis=1)
        local = tl.sum(wh * prefix_h * uv[None, :] + wr * prefix_r * rov[None, :], axis=1)
        bv = tl.sum(tl.where(o_b[None, :] == step, boundary, 0.0), axis=1)
        result = tl.where(o_b[None, :] == step, (xv + bv + local)[:, None], result)
        prefix_h += xv[:, None] * uv[None, :]
        prefix_r += xv[:, None] * riv[None, :]
    output_offset = panel * rows * R + o_row[:, None] * R + coord[None, :]
    tl.store(output + output_offset, result, mask=valid_row[:, None])


@triton.jit
def _local_vjp_resident_kernel(
    x,
    z,
    a,
    b,
    omega,
    cumulative,
    mass,
    grad_a,
    grad_b,
    grad_kappa,
    grad_cumulative,
    R: tl.constexpr,
    C: tl.constexpr,
    NRHS: tl.constexpr,
    BM: tl.constexpr,
    BC: tl.constexpr,
    NB: tl.constexpr,
    LOWER: tl.constexpr,
    SIGN: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    ACCUMULATE_A: tl.constexpr,
    ACCUMULATE_B: tl.constexpr,
):
    """Exact resident transpose of the coordinate generalized-Delta action."""
    panel = tl.program_id(0).to(tl.int64)
    rows = C * NRHS
    o_row = tl.arange(0, BM)
    o_s = tl.arange(0, C)
    o_b = tl.arange(0, BC)
    valid_row = o_row < rows
    token = o_row % C
    panel_rows = panel * rows * R
    panel_local = panel * C * R
    panel_pair = panel * C * C
    bw = tl.load(
        omega + panel_pair + token[:, None] * C + o_s[None, :],
        mask=valid_row[:, None],
        other=0.0,
    ).to(tl.float32)
    prefix = tl.zeros([BM, C], dtype=tl.float32)
    for tile in range(NB):
        coord = tile * BC + o_b
        bx = tl.load(
            x + panel_rows + o_row[:, None] * R + coord[None, :],
            mask=valid_row[:, None] & (coord[None, :] < R),
            other=0.0,
        ).to(tl.bfloat16)
        bb = tl.load(
            b + panel_local + o_s[:, None] * R + coord[None, :],
            mask=coord[None, :] < R,
            other=0.0,
        ).to(tl.bfloat16)
        prefix += tl.dot(bx, tl.trans(bb))
    suffix = tl.zeros([BM, C], dtype=tl.float32)
    grad_w_rows = tl.zeros([BM, C], dtype=tl.float32)
    for reverse_coordinate in range(R):
        coordinate = R - 1 - reverse_coordinate if LOWER else reverse_coordinate
        xc = tl.load(x + panel_rows + o_row * R + coordinate, mask=valid_row, other=0.0).to(tl.float32)
        zc = tl.load(z + panel_rows + o_row * R + coordinate, mask=valid_row, other=0.0).to(tl.float32)
        ac = tl.load(a + panel_local + o_s * R + coordinate).to(tl.float32)
        bc = tl.load(b + panel_local + o_s * R + coordinate).to(tl.float32)
        prefix -= xc[:, None] * bc[None, :]
        ga = SIGN * tl.sum(zc[:, None] * bw * prefix, axis=0)
        gb = SIGN * tl.sum(xc[:, None] * bw * suffix, axis=0)
        output = panel_local + o_s * R + coordinate
        if SYMMETRIC:
            value = ga + gb
            if ACCUMULATE_A:
                value += tl.load(grad_a + output).to(tl.float32)
            tl.store(grad_a + output, value)
        else:
            if ACCUMULATE_A:
                ga += tl.load(grad_a + output).to(tl.float32)
            if ACCUMULATE_B:
                gb += tl.load(grad_b + output).to(tl.float32)
            tl.store(grad_a + output, ga)
            tl.store(grad_b + output, gb)
        grad_w_rows += SIGN * zc[:, None] * ac[None, :] * prefix
        suffix += zc[:, None] * ac[None, :]
    g = tl.load(cumulative + panel * C + o_s, mask=o_s < C, other=0.0).to(tl.float32)
    valid = tl.load(mass + panel * C + o_s, mask=o_s < C, other=0.0) > 0.0
    grad_g_rows = tl.zeros([C], dtype=tl.float32)
    grad_g_columns = tl.zeros([C], dtype=tl.float32)
    for target in range(C):
        value = tl.sum(
            tl.where((token[:, None] == target) & valid_row[:, None], grad_w_rows, 0.0),
            axis=0,
        )
        target_valid = tl.sum(tl.where(o_s == target, valid, 0), axis=0) > 0
        target_g = tl.sum(tl.where(o_s == target, g, 0.0), axis=0)
        weight = tl.where(target_valid & valid & (o_s <= target), tl.exp(target_g - g), 0.0)
        omega_row = tl.load(
            omega + panel_pair + target * C + o_s, mask=o_s < C, other=0.0
        ).to(tl.float32)
        scaled = value * omega_row
        grad_g_rows += tl.where(o_s == target, tl.sum(scaled, axis=0), 0.0)
        grad_g_columns += scaled
        tl.store(
            grad_kappa + panel * C + target,
            tl.where(target_valid, tl.sum(value * weight, axis=0), 0.0),
        )
    tl.store(
        grad_cumulative + panel * C + o_s,
        tl.where(valid, grad_g_rows - grad_g_columns, 0.0),
        mask=o_s < C,
    )


@triton.jit
def _boundary_matrix_vjp_kernel(
    x,
    z,
    boundary_h,
    boundary_r,
    grad_J,
    grad_D,
    R: tl.constexpr,
    C: tl.constexpr,
    NRHS: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    LOWER: tl.constexpr,
    SIGN: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    tile_out = tl.program_id(0)
    tile_in = tl.program_id(1)
    panel = tl.program_id(2).to(tl.int64)
    rows = C * NRHS
    o_out = tile_out * BK + tl.arange(0, BK)
    o_in = tile_in * BK + tl.arange(0, BK)
    o_row = tl.arange(0, BR)
    valid_out = o_out < R
    valid_in = o_in < R
    valid_row = o_row < rows
    token = o_row % C
    panel_rows = panel * rows * R
    bz = tl.load(
        z + panel_rows + o_row[:, None] * R + o_out[None, :],
        mask=valid_row[:, None] & valid_out[None, :],
        other=0.0,
    ).to(tl.float32)
    bx = tl.load(
        x + panel_rows + o_row[:, None] * R + o_in[None, :],
        mask=valid_row[:, None] & valid_in[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    coefficient_h = tl.load(
        boundary_h + panel * C + token, mask=valid_row, other=0.0
    ).to(tl.float32)
    coefficient_r = tl.load(
        boundary_r + panel * C + token, mask=valid_row, other=0.0
    ).to(tl.float32)
    weighted_h = (bz * coefficient_h[:, None]).to(tl.bfloat16)
    weighted_r = (bz * coefficient_r[:, None]).to(tl.bfloat16)
    value_j = SIGN * tl.dot(tl.trans(weighted_h), bx)
    value_d = SIGN * tl.dot(tl.trans(weighted_r), bx)
    valid_matrix = valid_out[:, None] & valid_in[None, :]
    output_mask = valid_matrix
    if LOWER:
        output_mask &= o_out[:, None] > o_in[None, :]
    else:
        output_mask &= o_out[:, None] < o_in[None, :]
    pointer = panel * R * R + o_out[:, None] * R + o_in[None, :]
    value_j = tl.where(output_mask, value_j, 0.0)
    value_d = tl.where(output_mask, value_d, 0.0)
    if ACCUMULATE:
        value_j += tl.load(grad_J + pointer, mask=valid_matrix, other=0.0)
        value_d += tl.load(grad_D + pointer, mask=valid_matrix, other=0.0)
    tl.store(grad_J + pointer, value_j, mask=valid_matrix)
    tl.store(grad_D + pointer, value_d, mask=valid_matrix)


@triton.jit
def _boundary_coefficient_output_kernel(
    x,
    z,
    J,
    D,
    cumulative,
    mass,
    boundary_h,
    boundary_r,
    grad_kappa_h,
    grad_kappa_r,
    grad_cumulative,
    R: tl.constexpr,
    C: tl.constexpr,
    NRHS: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    NT: tl.constexpr,
    LOWER: tl.constexpr,
    SIGN: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BR)
    valid_rows = rows < C * NRHS
    coordinates = tl.arange(0, BK)
    panel_rows = panel * C * NRHS * R
    panel_matrix = panel * R * R
    row_j = tl.zeros([BR], dtype=tl.float32)
    row_d = tl.zeros([BR], dtype=tl.float32)
    for tile_out in range(NT):
        out = tile_out * BK + coordinates
        valid_out = out < R
        bz = tl.load(
            z + panel_rows + rows[:, None] * R + out[None, :],
            mask=valid_rows[:, None] & valid_out[None, :],
            other=0.0,
        ).to(tl.float32)
        for tile_in in range(NT):
            incoming = tile_in * BK + coordinates
            valid_in = incoming < R
            bx = tl.load(
                x + panel_rows + rows[:, None] * R + incoming[None, :],
                mask=valid_rows[:, None] & valid_in[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            factor_mask = valid_out[:, None] & valid_in[None, :]
            if LOWER:
                factor_mask &= out[:, None] > incoming[None, :]
            else:
                factor_mask &= out[:, None] < incoming[None, :]
            pointer = (
                panel_matrix + out[:, None] * R + incoming[None, :]
            )
            factor_j = tl.load(
                J + pointer, mask=factor_mask, other=0.0
            ).to(tl.bfloat16)
            factor_d = tl.load(
                D + pointer, mask=factor_mask, other=0.0
            ).to(tl.bfloat16)
            action_j = tl.dot(bx, tl.trans(factor_j))
            action_d = tl.dot(bx, tl.trans(factor_d))
            row_j += SIGN * tl.sum(bz * action_j, axis=1)
            row_d += SIGN * tl.sum(bz * action_d, axis=1)
    token = rows % C
    o_c = tl.arange(0, C)
    g = tl.load(cumulative + panel * C + o_c).to(tl.float32)
    valid = tl.load(mass + panel * C + o_c).to(tl.float32) > 0.0
    for target in range(C):
        target_mask = valid_rows & (token == target)
        value_j = tl.sum(tl.where(target_mask, row_j, 0.0), axis=0)
        value_d = tl.sum(tl.where(target_mask, row_d, 0.0), axis=0)
        target_valid = tl.sum(tl.where(o_c == target, valid, 0), axis=0) > 0
        target_g = tl.sum(tl.where(o_c == target, g, 0.0), axis=0)
        a = tl.exp(target_g)
        bh = tl.load(boundary_h + panel * C + target).to(tl.float32)
        br = tl.load(boundary_r + panel * C + target).to(tl.float32)
        tl.store(
            grad_kappa_h + panel * C + target,
            tl.where(target_valid, value_j * a, 0.0),
        )
        tl.store(
            grad_kappa_r + panel * C + target,
            tl.where(target_valid, value_d * a, 0.0),
        )
        tl.store(
            grad_cumulative + panel * C + target,
            tl.where(target_valid, value_j * bh + value_d * br, 0.0),
        )


@triton.jit
def _boundary_route_forward_kernel(
    left,
    right,
    matrix,
    norm,
    correlation,
    R: tl.constexpr,
    C: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    LOWER: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    o_c = tl.arange(0, BM)
    o_k = tl.arange(0, BK)
    valid_c = o_c < C
    valid_k = o_k < R
    panel_vector = panel * C * R
    panel_matrix = panel * R * R
    b_left = tl.load(
        left + panel_vector + o_c[:, None] * R + o_k[None, :],
        mask=valid_c[:, None] & valid_k[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    b_right = tl.load(
        right + panel_vector + o_c[:, None] * R + o_k[None, :],
        mask=valid_c[:, None] & valid_k[None, :],
        other=0.0,
    ).to(tl.float32)
    matrix_mask = valid_k[:, None] & valid_k[None, :]
    if LOWER:
        matrix_mask &= o_k[:, None] > o_k[None, :]
    else:
        matrix_mask &= o_k[:, None] < o_k[None, :]
    pointer = panel_matrix + o_k[:, None] * R + o_k[None, :]
    b_matrix = tl.load(
        matrix + pointer, mask=matrix_mask, other=0.0
    ).to(tl.float32)
    action = tl.dot(b_left, b_matrix.to(tl.bfloat16))
    corr = tl.sum(action * b_right, axis=1)
    tl.store(correlation + panel * C + o_c, corr, mask=valid_c)
    tl.store(norm + panel, tl.sum(b_matrix * b_matrix))


@triton.jit
def _boundary_route_vector_vjp_kernel(
    left,
    right,
    matrix,
    grad_correlation,
    grad_left,
    grad_right,
    R: tl.constexpr,
    C: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    LOWER: tl.constexpr,
    ACCUMULATE_LEFT: tl.constexpr,
    ACCUMULATE_RIGHT: tl.constexpr,
    SAME_OUTPUT: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    o_c = tl.arange(0, BM)
    o_k = tl.arange(0, BK)
    valid_c = o_c < C
    valid_k = o_k < R
    panel_vector = panel * C * R
    panel_matrix = panel * R * R
    b_left = tl.load(
        left + panel_vector + o_c[:, None] * R + o_k[None, :],
        mask=valid_c[:, None] & valid_k[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    b_right = tl.load(
        right + panel_vector + o_c[:, None] * R + o_k[None, :],
        mask=valid_c[:, None] & valid_k[None, :],
        other=0.0,
    ).to(tl.float32)
    matrix_mask = valid_k[:, None] & valid_k[None, :]
    if LOWER:
        matrix_mask &= o_k[:, None] > o_k[None, :]
    else:
        matrix_mask &= o_k[:, None] < o_k[None, :]
    pointer = panel_matrix + o_k[:, None] * R + o_k[None, :]
    b_matrix = tl.load(
        matrix + pointer, mask=matrix_mask, other=0.0
    ).to(tl.bfloat16)
    grad_corr = tl.load(
        grad_correlation + panel * C + o_c, mask=valid_c, other=0.0
    ).to(tl.float32)
    action = tl.dot(b_left, b_matrix)
    grad_action = (grad_corr[:, None] * b_right).to(tl.bfloat16)
    b_grad_left = tl.dot(grad_action, tl.trans(b_matrix))
    b_grad_right = grad_corr[:, None] * action
    output_pointer = panel_vector + o_c[:, None] * R + o_k[None, :]
    output_mask = valid_c[:, None] & valid_k[None, :]
    if SAME_OUTPUT:
        value = b_grad_left + b_grad_right
        if ACCUMULATE_LEFT:
            value += tl.load(
                grad_left + output_pointer, mask=output_mask, other=0.0
            ).to(tl.float32)
        tl.store(grad_left + output_pointer, value, mask=output_mask)
    else:
        if ACCUMULATE_LEFT:
            b_grad_left += tl.load(
                grad_left + output_pointer, mask=output_mask, other=0.0
            ).to(tl.float32)
        if ACCUMULATE_RIGHT:
            b_grad_right += tl.load(
                grad_right + output_pointer, mask=output_mask, other=0.0
            ).to(tl.float32)
        tl.store(grad_left + output_pointer, b_grad_left, mask=output_mask)
        tl.store(grad_right + output_pointer, b_grad_right, mask=output_mask)


@triton.jit
def _boundary_route_matrix_vjp_kernel(
    left,
    right,
    matrix,
    grad_norm,
    grad_correlation,
    grad_matrix,
    R: tl.constexpr,
    C: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    LOWER: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    tile_out = tl.program_id(0)
    tile_in = tl.program_id(1)
    panel = tl.program_id(2).to(tl.int64)
    o_out = tile_out * BK + tl.arange(0, BK)
    o_in = tile_in * BK + tl.arange(0, BK)
    o_c = tl.arange(0, BR)
    valid_out = o_out < R
    valid_in = o_in < R
    valid_c = o_c < C
    panel_vector = panel * C * R
    b_left = tl.load(
        left + panel_vector + o_c[:, None] * R + o_out[None, :],
        mask=valid_c[:, None] & valid_out[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    b_right = tl.load(
        right + panel_vector + o_c[:, None] * R + o_in[None, :],
        mask=valid_c[:, None] & valid_in[None, :],
        other=0.0,
    ).to(tl.float32)
    grad_corr = tl.load(
        grad_correlation + panel * C + o_c, mask=valid_c, other=0.0
    ).to(tl.float32)
    grad_action = (grad_corr[:, None] * b_right).to(tl.bfloat16)
    value = tl.dot(tl.trans(b_left), grad_action)
    pointer = panel * R * R + o_out[:, None] * R + o_in[None, :]
    b_matrix = tl.load(
        matrix + pointer,
        mask=valid_out[:, None] & valid_in[None, :],
        other=0.0,
    ).to(tl.float32)
    b_grad_norm = tl.load(grad_norm + panel).to(tl.float32)
    value += 2.0 * b_grad_norm * b_matrix
    valid_matrix = valid_out[:, None] & valid_in[None, :]
    output_mask = valid_matrix
    if LOWER:
        output_mask &= o_out[:, None] > o_in[None, :]
    else:
        output_mask &= o_out[:, None] < o_in[None, :]
    value = tl.where(output_mask, value, 0.0)
    if ACCUMULATE:
        value += tl.load(
            grad_matrix + pointer, mask=valid_matrix, other=0.0
        ).to(tl.float32)
    tl.store(grad_matrix + pointer, value, mask=valid_matrix)


def _launch_shape(chunk_size: int, width: int, rhs_count: int = 1) -> tuple[int, int]:
    return triton.next_power_of_2(chunk_size * rhs_count), triton.next_power_of_2(width)


def _exact_factor_solve(
    rhs: torch.Tensor,
    J: torch.Tensor,
    D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    omega_h: torch.Tensor,
    omega_r: torch.Tensor,
    boundary_h: torch.Tensor,
    boundary_r: torch.Tensor,
    *,
    lower: bool,
    transpose: bool,
    sigma: torch.Tensor | None,
    output_dtype: torch.dtype,
    num_warps: int,
) -> torch.Tensor:
    if rhs.ndim == 3:
        rhs = rhs[:, None]
        squeeze = True
    elif rhs.ndim == 4:
        squeeze = False
    else:
        raise ValueError("rhs must have shape [P,C,R] or [P,N,C,R]")
    chunks, rhs_count, chunk_size, width = rhs.shape
    if width % 16:
        raise ValueError("native exact coordinate solve requires r divisible by 16")
    bm, bk = _launch_shape(chunk_size, width, rhs_count)
    output = torch.empty(rhs.shape, dtype=output_dtype, device=rhs.device)
    sigma_arg = rhs if sigma is None else sigma
    _exact_coordinate_solve_kernel[(chunks,)](
        rhs,
        output,
        J,
        D,
        u,
        h,
        omega_h,
        omega_r,
        boundary_h,
        boundary_r,
        sigma_arg,
        stride_rhs_p=rhs.stride(0),
        stride_rhs_n=rhs.stride(1),
        stride_rhs_c=rhs.stride(2),
        stride_rhs_r=rhs.stride(3),
        R=width,
        C=chunk_size,
        NRHS=rhs_count,
        BM=bm,
        BK=bk,
        LOWER=lower,
        TRANSPOSE=transpose,
        DIVIDE_SIGMA=sigma is not None,
        num_warps=num_warps,
        num_stages=2,
    )
    return output[:, 0] if squeeze else output


def _chunked_factor_direct(
    rhs: torch.Tensor,
    J: torch.Tensor,
    D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    omega_h: torch.Tensor,
    omega_r: torch.Tensor,
    boundary_h: torch.Tensor,
    boundary_r: torch.Tensor,
    *,
    lower: bool,
    transpose: bool,
    sigma: torch.Tensor | None,
    output_dtype: torch.dtype,
    num_warps: int,
) -> torch.Tensor:
    if rhs.ndim == 3:
        rhs = rhs[:, None]
        squeeze = True
    elif rhs.ndim == 4:
        squeeze = False
    else:
        raise ValueError("rhs must have shape [P,C,R] or [P,N,C,R]")
    chunks, rhs_count, chunk_size, width = rhs.shape
    if width % 32:
        raise ValueError("native chunked direct action requires r divisible by 32")
    bm, bk = _launch_shape(chunk_size, width, rhs_count)
    block = 32
    blocks = width // block
    prefix_h = torch.empty(
        chunks, blocks, chunk_size * rhs_count, chunk_size,
        dtype=torch.float32, device=rhs.device,
    )
    prefix_r = torch.empty_like(prefix_h)
    sigma_arg = rhs if sigma is None else sigma
    _direct_prefix_states_kernel[(chunks,)](
        rhs, u, h, prefix_h, prefix_r, sigma_arg,
        stride_rhs_p=rhs.stride(0), stride_rhs_n=rhs.stride(1),
        stride_rhs_c=rhs.stride(2), stride_rhs_r=rhs.stride(3),
        R=width, C=chunk_size, NRHS=rhs_count, BM=bm,
        BC=block, NB=blocks, LOWER=lower, TRANSPOSE=transpose,
        MULTIPLY_SIGMA=sigma is not None,
        num_warps=num_warps, num_stages=2,
    )
    output = torch.empty(rhs.shape, dtype=output_dtype, device=rhs.device)
    _direct_block_output_kernel[(blocks, chunks)](
        rhs, output, J, D, u, h, omega_h, omega_r,
        boundary_h, boundary_r, prefix_h, prefix_r, sigma_arg,
        stride_rhs_p=rhs.stride(0), stride_rhs_n=rhs.stride(1),
        stride_rhs_c=rhs.stride(2), stride_rhs_r=rhs.stride(3),
        R=width, C=chunk_size, NRHS=rhs_count, BM=bm, BK=bk,
        BC=block, NB=blocks, LOWER=lower, TRANSPOSE=transpose,
        MULTIPLY_SIGMA=sigma is not None,
        num_warps=num_warps, num_stages=2,
    )
    return output[:, 0] if squeeze else output


def resident_primal(
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
    *,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rhs.ndim == 3:
        rhs = rhs[:, None]
        squeeze = True
    elif rhs.ndim == 4:
        squeeze = False
    else:
        raise ValueError("rhs must have shape [P,C,R] or [P,N,C,R]")
    lower_cache = _exact_factor_solve(
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
        transpose=False,
        sigma=None,
        output_dtype=torch.float16,
        num_warps=num_warps,
    )
    output = _exact_factor_solve(
        lower_cache,
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
        sigma=sigma,
        output_dtype=torch.bfloat16,
        num_warps=num_warps,
    )
    if squeeze:
        return output[:, 0], lower_cache[:, 0], output[:, 0]
    return output, lower_cache, output


def resident_dual(
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
    *,
    num_warps: int,
) -> torch.Tensor:
    lower_output = _chunked_factor_direct(
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
        sigma=None,
        output_dtype=torch.float16,
        num_warps=num_warps,
    )
    return _chunked_factor_direct(
        lower_output,
        J,
        D,
        u,
        h,
        omega_h,
        omega_r_upper,
        boundary_h,
        boundary_r_upper,
        lower=False,
        transpose=True,
        sigma=sigma,
        output_dtype=torch.bfloat16,
        num_warps=num_warps,
    )


def resident_factor_transpose(
    rhs: torch.Tensor,
    J: torch.Tensor,
    D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    omega_h: torch.Tensor,
    omega_r: torch.Tensor,
    boundary_h: torch.Tensor,
    boundary_r: torch.Tensor,
    *,
    lower: bool,
    num_warps: int,
) -> torch.Tensor:
    return _exact_factor_solve(
        rhs,
        J,
        D,
        u,
        h,
        omega_h,
        omega_r,
        boundary_h,
        boundary_r,
        lower=lower,
        transpose=True,
        sigma=None,
        output_dtype=torch.float32,
        num_warps=num_warps,
    )


def resident_factor_direct(
    rhs: torch.Tensor,
    J: torch.Tensor,
    D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    omega_h: torch.Tensor,
    omega_r: torch.Tensor,
    boundary_h: torch.Tensor,
    boundary_r: torch.Tensor,
    *,
    lower: bool,
    transpose: bool,
    num_warps: int,
) -> torch.Tensor:
    return _chunked_factor_direct(
        rhs,
        J,
        D,
        u,
        h,
        omega_h,
        omega_r,
        boundary_h,
        boundary_r,
        lower=lower,
        transpose=transpose,
        sigma=None,
        output_dtype=torch.float32,
        num_warps=num_warps,
    )


def local_representation_vjp(
    x: torch.Tensor,
    cotangent: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    omega: torch.Tensor,
    cumulative: torch.Tensor,
    mass: torch.Tensor,
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    *,
    lower: bool,
    sign: int,
    accumulate_a: bool,
    accumulate_b: bool,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim == 3:
        x = x[:, None]
        cotangent = cotangent[:, None]
    chunks, rhs_count, chunk_size, width = x.shape
    bm, bk = _launch_shape(chunk_size, width, rhs_count)
    block = 32
    grad_kappa = torch.empty_like(mass, dtype=torch.float32)
    grad_cumulative = torch.empty_like(cumulative, dtype=torch.float32)
    _local_vjp_resident_kernel[(chunks,)](
        x, cotangent, a, b, omega, cumulative, mass,
        grad_a, grad_b, grad_kappa, grad_cumulative,
        R=width, C=chunk_size, NRHS=rhs_count, BM=bm,
        BC=block, NB=triton.cdiv(width, block),
        LOWER=lower, SIGN=sign, SYMMETRIC=False,
        ACCUMULATE_A=accumulate_a, ACCUMULATE_B=accumulate_b,
        num_warps=num_warps, num_stages=2,
    )
    return grad_kappa, grad_cumulative


def local_symmetric_representation_vjp(
    x: torch.Tensor,
    cotangent: torch.Tensor,
    a: torch.Tensor,
    omega: torch.Tensor,
    cumulative: torch.Tensor,
    mass: torch.Tensor,
    grad_a: torch.Tensor,
    *,
    lower: bool,
    sign: int,
    accumulate: bool,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim == 3:
        x = x[:, None]
        cotangent = cotangent[:, None]
    chunks, rhs_count, chunk_size, width = x.shape
    bm, bk = _launch_shape(chunk_size, width, rhs_count)
    block = 32
    grad_kappa = torch.empty_like(mass, dtype=torch.float32)
    grad_cumulative = torch.empty_like(cumulative, dtype=torch.float32)
    _local_vjp_resident_kernel[(chunks,)](
        x,
        cotangent,
        a,
        a,
        omega,
        cumulative,
        mass,
        grad_a,
        grad_a,
        grad_kappa,
        grad_cumulative,
        R=width,
        C=chunk_size,
        NRHS=rhs_count,
        BM=bm,
        BC=block,
        NB=triton.cdiv(width, block),
        LOWER=lower,
        SIGN=sign,
        SYMMETRIC=True,
        ACCUMULATE_A=accumulate,
        ACCUMULATE_B=accumulate,
        num_warps=num_warps,
        num_stages=2,
    )
    return grad_kappa, grad_cumulative


def boundary_representation_vjp(
    x: torch.Tensor,
    cotangent: torch.Tensor,
    J: torch.Tensor,
    D: torch.Tensor,
    boundary_h: torch.Tensor,
    boundary_r: torch.Tensor,
    cumulative: torch.Tensor,
    mass: torch.Tensor,
    grad_j: torch.Tensor,
    grad_d: torch.Tensor,
    *,
    lower: bool,
    sign: int,
    accumulate: bool,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim == 3:
        x = x[:, None]
        cotangent = cotangent[:, None]
    chunks, rhs_count, chunk_size, width = x.shape
    rows = rhs_count * chunk_size
    block_rows = triton.next_power_of_2(rows)
    matrix_tile = 32
    matrix_tiles = triton.cdiv(width, matrix_tile)
    grad_kappa_h = torch.empty_like(mass, dtype=torch.float32)
    grad_kappa_r = torch.empty_like(mass, dtype=torch.float32)
    grad_cumulative = torch.empty_like(cumulative, dtype=torch.float32)
    _boundary_matrix_vjp_kernel[
        (matrix_tiles, matrix_tiles, chunks)
    ](
        x,
        cotangent,
        boundary_h,
        boundary_r,
        grad_j,
        grad_d,
        R=width,
        C=chunk_size,
        NRHS=rhs_count,
        BR=block_rows,
        BK=matrix_tile,
        LOWER=lower,
        SIGN=sign,
        ACCUMULATE=accumulate,
        num_warps=4,
        num_stages=1,
    )
    _boundary_coefficient_output_kernel[(chunks,)](
        x,
        cotangent,
        J,
        D,
        cumulative,
        mass,
        boundary_h,
        boundary_r,
        grad_kappa_h,
        grad_kappa_r,
        grad_cumulative,
        R=width,
        C=chunk_size,
        NRHS=rhs_count,
        BR=block_rows,
        BK=matrix_tile,
        NT=matrix_tiles,
        LOWER=lower,
        SIGN=sign,
        num_warps=num_warps,
        num_stages=1,
    )
    return grad_kappa_h, grad_kappa_r, grad_cumulative


def boundary_route_forward(
    left: torch.Tensor,
    right: torch.Tensor,
    matrix: torch.Tensor,
    *,
    lower: bool,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks, chunk_size, width = left.shape
    block_rows = triton.next_power_of_2(chunk_size)
    block_width = triton.next_power_of_2(width)
    norm = torch.empty(chunks, dtype=torch.float32, device=left.device)
    correlation = torch.empty(
        chunks, chunk_size, dtype=torch.float32, device=left.device
    )
    _boundary_route_forward_kernel[(chunks,)](
        left,
        right,
        matrix,
        norm,
        correlation,
        R=width,
        C=chunk_size,
        BM=block_rows,
        BK=block_width,
        LOWER=lower,
        num_warps=num_warps,
        num_stages=1,
    )
    return norm, correlation


def boundary_route_vjp(
    left: torch.Tensor,
    right: torch.Tensor,
    matrix: torch.Tensor,
    grad_norm: torch.Tensor,
    grad_correlation: torch.Tensor,
    grad_left: torch.Tensor,
    grad_right: torch.Tensor,
    grad_matrix: torch.Tensor,
    *,
    lower: bool,
    accumulate_left: bool,
    accumulate_right: bool,
    accumulate_matrix: bool,
    same_output: bool = False,
    num_warps: int,
) -> None:
    chunks, chunk_size, width = left.shape
    block_rows = triton.next_power_of_2(chunk_size)
    block_width = triton.next_power_of_2(width)
    matrix_tile = 32
    if grad_left.shape != left.shape or grad_right.shape != right.shape:
        raise ValueError("boundary vector gradient owners must match their inputs")
    if grad_matrix.shape != matrix.shape:
        raise ValueError("boundary matrix gradient owner must match its input")
    if same_output and grad_left.data_ptr() != grad_right.data_ptr():
        raise ValueError("same-output boundary VJP requires one shared owner")
    _boundary_route_vector_vjp_kernel[(chunks,)](
        left,
        right,
        matrix,
        grad_correlation,
        grad_left,
        grad_right,
        R=width,
        C=chunk_size,
        BM=block_rows,
        BK=block_width,
        LOWER=lower,
        ACCUMULATE_LEFT=accumulate_left,
        ACCUMULATE_RIGHT=accumulate_right,
        SAME_OUTPUT=same_output,
        num_warps=num_warps,
        num_stages=1,
    )
    _boundary_route_matrix_vjp_kernel[
        (triton.cdiv(width, matrix_tile), triton.cdiv(width, matrix_tile), chunks)
    ](
        left,
        right,
        matrix,
        grad_norm,
        grad_correlation,
        grad_matrix,
        R=width,
        C=chunk_size,
        BR=block_rows,
        BK=matrix_tile,
        LOWER=lower,
        ACCUMULATE=accumulate_matrix,
        num_warps=4,
        num_stages=1,
    )


__all__ = [
    "boundary_representation_vjp",
    "boundary_route_forward",
    "boundary_route_vjp",
    "local_representation_vjp",
    "local_symmetric_representation_vjp",
    "resident_dual",
    "resident_factor_direct",
    "resident_factor_transpose",
    "resident_primal",
]
