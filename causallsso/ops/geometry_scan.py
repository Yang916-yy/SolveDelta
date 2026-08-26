from __future__ import annotations

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# The paired resident state loop below is adapted from MESA's MIT-licensed
# chunk_mesa_net_fwd_kernel_h in Flash Linear Attention 0.5.2. It keeps the
# mature Hkk/Hkv schedule while writing SolveDelta's consumer-native layout;
# one matrix-tile owner also emits the scalar mass route from the same gauge.

import torch
import triton
import triton.language as tl


@triton.jit
def _paired_geometry_forward_kernel(
    u,
    u_d,
    h,
    cumulative,
    initial_j,
    initial_d,
    initial_m,
    j_boundary,
    d_boundary,
    m_boundary,
    mass,
    cumulative_panel,
    tail_weight,
    chunk_decay,
    final_j,
    final_d,
    final_m,
    lengths,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    BR: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    tile_i = tl.program_id(0)
    tile_j = tl.program_id(1)
    bh = tl.program_id(2)
    batch = bh // H
    head = bh % H
    oi = tile_i * BR + tl.arange(0, BR)
    oj = tile_j * BR + tl.arange(0, BR)
    matrix_mask = (oi[:, None] < R) & (oj[None, :] < R)
    state_base = bh * R * R
    d_state = tl.load(
        initial_d + state_base + oi[:, None] * R + oj[None, :],
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)
    j_state = tl.zeros([BR, BR], dtype=tl.float32)
    if tile_i >= tile_j:
        j_state = tl.load(
            initial_j + state_base + oi[:, None] * R + oj[None, :],
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
    mass_owner = (tile_i == 0) & (tile_j == 0)
    current_mass = tl.load(initial_m + bh).to(tl.float32)

    rows = tl.arange(0, C)
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + batch).to(tl.int64)
        eos = tl.load(cu_seqlens + batch + 1).to(tl.int64)
        sequence_length = eos - bos
        chunk_offset = tl.load(chunk_offsets + batch).to(tl.int64)
        chunk_count = tl.load(chunk_offsets + batch + 1).to(tl.int64) - chunk_offset
    else:
        bos = batch * T
        sequence_length = tl.load(lengths + batch) if HAS_LENGTHS else T
        chunk_offset = 0
        chunk_count = N
    for chunk in range(N):
        active_chunk = chunk < chunk_count
        panel = (
            (chunk_offset + chunk) * H + head
            if IS_VARLEN
            else bh * N + chunk
        )
        panel_base = panel * R * R
        d_pointer = d_boundary + panel_base + oi[:, None] * R + oj[None, :]
        tl.store(d_pointer, d_state, mask=matrix_mask & active_chunk)
        if tile_i >= tile_j:
            j_pointer = j_boundary + panel_base + oi[:, None] * R + oj[None, :]
            if tile_i == tile_j:
                j_value = 0.5 * (j_state + tl.trans(j_state))
                tl.store(j_pointer, j_value, mask=matrix_mask & active_chunk)
            else:
                tl.store(j_pointer, j_state, mask=matrix_mask & active_chunk)
                transpose_mask = (oj[:, None] < R) & (oi[None, :] < R)
                tl.store(
                    j_boundary
                    + panel_base
                    + oj[:, None] * R
                    + oi[None, :],
                    tl.trans(j_state),
                    mask=transpose_mask & active_chunk,
                )

        token = chunk * C + rows
        valid = active_chunk & (token < sequence_length)
        g = tl.load(
            cumulative + (bos + token) * H + head,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        tail = tl.minimum(sequence_length - chunk * C, C) - 1
        g_tail = tl.sum(tl.where(rows == tail, g, 0.0), axis=0)
        state_decay = tl.where(active_chunk, tl.exp(g_tail), 1.0)
        observation_decay = tl.where(valid, tl.exp(g_tail - g), 0.0)
        if mass_owner:
            tl.store(m_boundary + panel, current_mass, mask=active_chunk)
            tl.store(
                cumulative_panel + panel * C + rows,
                g,
                mask=(rows < C) & active_chunk,
            )
            pair_weight = tl.exp(g[:, None] - g[None, :])
            pair_weight = tl.where(
                (rows[:, None] >= rows[None, :])
                & valid[:, None]
                & valid[None, :],
                pair_weight,
                0.0,
            )
            token_mass = tl.exp(g) * current_mass + tl.sum(pair_weight, axis=1)
            tl.store(
                mass + panel * C + rows,
                tl.where(valid, token_mass, 0.0),
                mask=(rows < C) & active_chunk,
            )
            tl.store(
                tail_weight + panel * C + rows,
                tl.where(valid, observation_decay, 0.0),
                mask=(rows < C) & active_chunk,
            )
            tl.store(chunk_decay + panel, state_decay, mask=active_chunk)
            next_mass = tl.sum(
                tl.where(rows == tail, token_mass, 0.0), axis=0
            )
            current_mass = tl.where(active_chunk, next_mass, current_mass)

        row_base = panel * C * R + rows[:, None] * R
        u_j_i = tl.zeros([C, BR], dtype=tl.float16)
        u_j_j = tl.zeros([C, BR], dtype=tl.float16)
        if tile_i >= tile_j:
            u_j_i = tl.load(
                u + row_base + oi[None, :],
                mask=valid[:, None] & (oi[None, :] < R),
                other=0.0,
            )
            u_j_j = tl.load(
                u + row_base + oj[None, :],
                mask=valid[:, None] & (oj[None, :] < R),
                other=0.0,
            )
        u_d_i = tl.load(
            u_d + row_base + oi[None, :],
            mask=valid[:, None] & (oi[None, :] < R),
            other=0.0,
        )
        h_j = tl.load(
            h + row_base + oj[None, :],
            mask=valid[:, None] & (oj[None, :] < R),
            other=0.0,
        )
        d_state *= state_decay
        d_state += tl.dot(
            tl.trans((u_d_i * observation_decay[:, None]).to(tl.bfloat16)),
            h_j.to(tl.bfloat16),
        )
        if tile_i >= tile_j:
            j_state *= state_decay
            j_state += tl.dot(
                tl.trans((u_j_i * observation_decay[:, None]).to(tl.float16)),
                u_j_j.to(tl.float16),
            )
            if tile_i == tile_j:
                # Match the full stored symmetric representative at every
                # continuation boundary, including recurrent splits.
                j_state = 0.5 * (j_state + tl.trans(j_state))

    tl.store(
        final_d + state_base + oi[:, None] * R + oj[None, :],
        d_state,
        mask=matrix_mask,
    )
    if tile_i >= tile_j:
        final_pointer = final_j + state_base + oi[:, None] * R + oj[None, :]
        if tile_i == tile_j:
            j_value = 0.5 * (j_state + tl.trans(j_state))
            tl.store(final_pointer, j_value, mask=matrix_mask)
        else:
            tl.store(final_pointer, j_state, mask=matrix_mask)
            transpose_mask = (oj[:, None] < R) & (oi[None, :] < R)
            tl.store(
                final_j + state_base + oj[:, None] * R + oi[None, :],
                tl.trans(j_state),
                mask=transpose_mask,
            )
    if mass_owner:
        tl.store(final_m + bh, current_mass)


@triton.jit
def _matrix_boundary_reverse_kernel(
    grad_j_boundary,
    grad_d_boundary,
    grad_j_final,
    grad_d_final,
    chunk_decay,
    end_j,
    end_d,
    grad_j0,
    grad_d0,
    chunk_offsets,
    N: tl.constexpr,
    R: tl.constexpr,
    H: tl.constexpr,
    BR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    tile_i = tl.program_id(0)
    tile_j = tl.program_id(1)
    bh = tl.program_id(2)
    sequence = bh // H
    head = bh % H
    if IS_VARLEN:
        chunk_offset = tl.load(chunk_offsets + sequence).to(tl.int64)
        chunk_count = tl.load(chunk_offsets + sequence + 1).to(tl.int64) - chunk_offset
    else:
        chunk_offset = 0
        chunk_count = N
    oi = tile_i * BR + tl.arange(0, BR)
    oj = tile_j * BR + tl.arange(0, BR)
    mask = (oi[:, None] < R) & (oj[None, :] < R)
    state_base = bh * R * R
    final_j_direct = tl.load(
        grad_j_final + state_base + oi[:, None] * R + oj[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    final_j_transpose = tl.load(
        grad_j_final + state_base + oj[None, :] * R + oi[:, None],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    dj = 0.5 * (final_j_direct + final_j_transpose)
    dd = tl.load(
        grad_d_final + state_base + oi[:, None] * R + oj[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    for reverse_index in range(N):
        chunk = N - 1 - reverse_index
        active_chunk = chunk < chunk_count
        panel = (
            (chunk_offset + chunk) * H + head
            if IS_VARLEN
            else bh * N + chunk
        )
        chunk_base = panel * R * R
        pointer = chunk_base + oi[:, None] * R + oj[None, :]
        tl.store(end_j + pointer, dj, mask=mask & active_chunk)
        tl.store(end_d + pointer, dd, mask=mask & active_chunk)
        decay = tl.load(
            chunk_decay + panel, mask=active_chunk, other=1.0
        ).to(tl.float32)
        local_j_direct = tl.load(
            grad_j_boundary + pointer, mask=mask & active_chunk, other=0.0
        ).to(tl.float32)
        local_j_transpose = tl.load(
            grad_j_boundary
            + chunk_base
            + oj[None, :] * R
            + oi[:, None],
            mask=mask & active_chunk,
            other=0.0,
        ).to(tl.float32)
        local_j = 0.5 * (local_j_direct + local_j_transpose)
        local_d = tl.load(
            grad_d_boundary + pointer, mask=mask & active_chunk, other=0.0
        )
        dj = tl.where(active_chunk, local_j + decay * dj, dj)
        dd = tl.where(active_chunk, local_d.to(tl.float32) + decay * dd, dd)
    tl.store(grad_j0 + state_base + oi[:, None] * R + oj[None, :], dj, mask=mask)
    tl.store(grad_d0 + state_base + oi[:, None] * R + oj[None, :], dd, mask=mask)


@triton.jit
def _mass_boundary_reverse_kernel(
    grad_boundary,
    grad_final,
    chunk_decay,
    end_cotangent,
    grad_initial,
    chunk_offsets,
    N: tl.constexpr,
    H: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    bh = tl.program_id(0)
    sequence = bh // H
    head = bh % H
    if IS_VARLEN:
        chunk_offset = tl.load(chunk_offsets + sequence).to(tl.int64)
        chunk_count = tl.load(chunk_offsets + sequence + 1).to(tl.int64) - chunk_offset
    else:
        chunk_offset = 0
        chunk_count = N
    cotangent = tl.load(grad_final + bh).to(tl.float32)
    for reverse_index in range(N):
        chunk = N - 1 - reverse_index
        active_chunk = chunk < chunk_count
        offset = (
            (chunk_offset + chunk) * H + head
            if IS_VARLEN
            else bh * N + chunk
        )
        tl.store(end_cotangent + offset, cotangent, mask=active_chunk)
        decay = tl.load(chunk_decay + offset, mask=active_chunk, other=1.0).to(tl.float32)
        local = tl.load(grad_boundary + offset, mask=active_chunk, other=0.0).to(tl.float32)
        cotangent = tl.where(active_chunk, local + decay * cotangent, cotangent)
    tl.store(grad_initial + bh, cotangent)


@triton.jit
def _mass_local_reverse_kernel(
    grad_mass,
    grad_cumulative,
    cumulative,
    tail_weight,
    initial_mass,
    grad_boundary_mass,
    grad_log_decay,
    C: tl.constexpr,
    BC: tl.constexpr,
    HAS_GRAD_MASS: tl.constexpr,
    HAS_GRAD_CUMULATIVE: tl.constexpr,
):
    # FLA's reverse-cumsum primitive specialized to the scalar affine state.
    panel = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BC)
    row_mask = rows < C
    g = tl.load(
        cumulative + panel * C + rows, mask=row_mask, other=0.0
    ).to(tl.float32)
    tail = tl.load(
        tail_weight + panel * C + rows, mask=row_mask, other=0.0
    ).to(tl.float32)
    valid = row_mask & (tail > 0.0)
    gm = tl.zeros([BC], dtype=tl.float32)
    if HAS_GRAD_MASS:
        gm = tl.load(
            grad_mass + panel * C + rows, mask=valid, other=0.0
        ).to(tl.float32)
    exp_g = tl.exp(g)
    tl.store(grad_boundary_mass + panel, tl.sum(gm * exp_g, axis=0))

    weight = tl.exp(g[:, None] - g[None, :])
    causal = rows[:, None] >= rows[None, :]
    pair_mask = causal & valid[:, None] & valid[None, :]
    weight = tl.where(pair_mask, weight, 0.0)
    weighted = gm[:, None] * weight
    m0 = tl.load(initial_mass + panel).to(tl.float32)
    cumulative_cotangent = gm * exp_g * m0
    cumulative_cotangent += tl.sum(weighted, axis=1)
    cumulative_cotangent -= tl.sum(weighted, axis=0)
    if HAS_GRAD_CUMULATIVE:
        cumulative_cotangent += tl.load(
            grad_cumulative + panel * C + rows,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
    result = tl.cumsum(cumulative_cotangent, axis=0, reverse=True)
    tl.store(
        grad_log_decay + panel * C + rows,
        tl.where(valid, result, 0.0),
        mask=row_mask,
    )


@triton.jit
def _panel_scalar_to_bth_kernel(
    panel,
    output,
    cu_seqlens,
    chunk_indices,
    T: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    p = tl.program_id(0)
    if IS_VARLEN:
        global_chunk = p // H
        head = p % H
        sequence = tl.load(chunk_indices + global_chunk * 2).to(tl.int32)
        chunk = tl.load(chunk_indices + global_chunk * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + sequence).to(tl.int64)
        eos = tl.load(cu_seqlens + sequence + 1).to(tl.int64)
    else:
        bh = p // N
        chunk = p % N
        batch = bh // H
        head = bh % H
        bos = batch * T
        eos = bos + T
    rows = tl.arange(0, C)
    token = chunk * C + rows
    output_token = bos + token
    valid = output_token < eos
    value = tl.load(panel + p * C + rows, mask=valid, other=0.0)
    tl.store(
        output + output_token * H + head,
        value,
        mask=valid,
    )


@triton.jit
def _geometry_vector_vjp_kernel(
    u,
    h,
    end_j,
    end_d,
    tail_weight,
    grad_u,
    grad_h,
    C: tl.constexpr,
    R: tl.constexpr,
    BC: tl.constexpr,
    BR: tl.constexpr,
):
    # This is the q/k/v transpose-dot core from MESA's Hkk/Hkv reverse with
    # its model-specific q_star/do/beta interface removed.
    panel = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    rows = tl.arange(0, BC)
    coordinates = tl.arange(0, R)
    outputs = tile * BR + tl.arange(0, BR)
    row_mask = rows < C
    output_mask = outputs < R
    panel_vector = panel * C * R
    panel_matrix = panel * R * R
    u_value = tl.load(
        u + panel_vector + rows[:, None] * R + coordinates[None, :],
        mask=row_mask[:, None] & (coordinates[None, :] < R),
        other=0.0,
    ).to(tl.bfloat16)
    h_value = tl.load(
        h + panel_vector + rows[:, None] * R + coordinates[None, :],
        mask=row_mask[:, None] & (coordinates[None, :] < R),
        other=0.0,
    ).to(tl.bfloat16)
    j_block = tl.load(
        end_j
        + panel_matrix
        + coordinates[:, None] * R
        + outputs[None, :],
        mask=(coordinates[:, None] < R) & output_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    d_block = tl.load(
        end_d
        + panel_matrix
        + coordinates[:, None] * R
        + outputs[None, :],
        mask=(coordinates[:, None] < R) & output_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    d_transpose_block = tl.load(
        end_d
        + panel_matrix
        + outputs[None, :] * R
        + coordinates[:, None],
        mask=output_mask[None, :] & (coordinates[:, None] < R),
        other=0.0,
    ).to(tl.bfloat16)
    u_j = tl.dot(u_value, j_block)
    u_d = tl.dot(u_value, d_block)
    h_dt = tl.dot(h_value, d_transpose_block)
    weight = tl.load(
        tail_weight + panel * C + rows,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    output_pointer = panel_vector + rows[:, None] * R + outputs[None, :]
    output_mask_2d = row_mask[:, None] & output_mask[None, :]
    tl.store(
        grad_u + output_pointer,
        (2.0 * u_j + h_dt) * weight[:, None],
        mask=output_mask_2d,
    )
    tl.store(
        grad_h + output_pointer,
        u_d * weight[:, None],
        mask=output_mask_2d,
    )


@triton.jit
def _geometry_decay_partial_kernel(
    u,
    h,
    end_j,
    end_d,
    boundary_j,
    boundary_d,
    observation_partial,
    boundary_partial,
    C: tl.constexpr,
    R: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NB: tl.constexpr,
):
    """MESA-style state-tile epilogue for sensitive decay contractions."""
    row_tile = tl.program_id(0)
    panel = tl.program_id(1).to(tl.int64)
    rows = tl.arange(0, BC)
    row_coordinates = row_tile * BK + tl.arange(0, BK)
    valid_rows = rows < C
    valid_row_coordinates = row_coordinates < R
    panel_vector = panel * C * R
    panel_matrix = panel * R * R
    u_left = tl.load(
        u + panel_vector + rows[:, None] * R + row_coordinates[None, :],
        mask=valid_rows[:, None] & valid_row_coordinates[None, :],
        other=0.0,
    ).to(tl.float32)
    observation = tl.zeros([BC], dtype=tl.float32)
    boundary = tl.zeros([], dtype=tl.float32)
    for column_tile in range(NB):
        column_coordinates = column_tile * BK + tl.arange(0, BK)
        valid_column_coordinates = column_coordinates < R
        matrix_mask = (
            valid_row_coordinates[:, None]
            & valid_column_coordinates[None, :]
        )
        matrix_pointer = (
            panel_matrix
            + row_coordinates[:, None] * R
            + column_coordinates[None, :]
        )
        ej = tl.load(
            end_j + matrix_pointer, mask=matrix_mask, other=0.0
        ).to(tl.float32)
        ed = tl.load(
            end_d + matrix_pointer, mask=matrix_mask, other=0.0
        ).to(tl.float32)
        j0 = tl.load(
            boundary_j + matrix_pointer, mask=matrix_mask, other=0.0
        ).to(tl.float32)
        d0 = tl.load(
            boundary_d + matrix_pointer, mask=matrix_mask, other=0.0
        ).to(tl.float32)
        u_right = tl.load(
            u
            + panel_vector
            + rows[:, None] * R
            + column_coordinates[None, :],
            mask=valid_rows[:, None] & valid_column_coordinates[None, :],
            other=0.0,
        ).to(tl.float32)
        h_right = tl.load(
            h
            + panel_vector
            + rows[:, None] * R
            + column_coordinates[None, :],
            mask=valid_rows[:, None] & valid_column_coordinates[None, :],
            other=0.0,
        ).to(tl.float32)
        action_j = tl.dot(u_left, ej, input_precision="ieee")
        action_d = tl.dot(u_left, ed, input_precision="ieee")
        observation += tl.sum(
            action_j * u_right + action_d * h_right, axis=1
        )
        boundary += tl.sum(ej * j0 + ed * d0)
    tl.store(
        observation_partial + (panel * NB + row_tile) * C + rows,
        observation,
        mask=valid_rows,
    )
    tl.store(boundary_partial + panel * NB + row_tile, boundary)


@triton.jit
def _geometry_decay_finalize_kernel(
    observation_partial,
    boundary_partial,
    end_m,
    initial_m,
    tail_weight,
    chunk_decay,
    grad_extra,
    grad_panel,
    C: tl.constexpr,
    BC: tl.constexpr,
    NB: tl.constexpr,
    BNB: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BC)
    tiles = tl.arange(0, BNB)
    valid_rows = rows < C
    partial_mask = (tiles[:, None] < NB) & valid_rows[None, :]
    partial = tl.load(
        observation_partial
        + panel * NB * C
        + tiles[:, None] * C
        + rows[None, :],
        mask=partial_mask,
        other=0.0,
    ).to(tl.float32)
    observation = tl.sum(partial, axis=0)
    end_mass = tl.load(end_m + panel).to(tl.float32)
    observation += end_mass
    weight = tl.load(
        tail_weight + panel * C + rows,
        mask=valid_rows,
        other=0.0,
    ).to(tl.float32)
    valid = weight > 0.0
    observation *= weight
    exclusive = tl.sum(
        tl.where(
            rows[:, None] > rows[None, :],
            observation[None, :],
            0.0,
        ),
        axis=1,
    )
    boundary_tiles = tl.load(
        boundary_partial + panel * NB + tiles,
        mask=tiles < NB,
        other=0.0,
    ).to(tl.float32)
    initial_mass = tl.load(initial_m + panel).to(tl.float32)
    decay = tl.load(chunk_decay + panel).to(tl.float32)
    boundary = decay * (
        tl.sum(boundary_tiles, axis=0) + end_mass * initial_mass
    )
    extra = tl.load(
        grad_extra + panel * C + rows,
        mask=valid_rows,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        grad_panel + panel * C + rows,
        tl.where(valid, boundary + exclusive + extra, 0.0),
        mask=valid_rows,
    )


class _GeometryScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_panel: torch.Tensor,
        u_d_panel: torch.Tensor,
        h_panel: torch.Tensor,
        log_decay: torch.Tensor,
        initial_m: torch.Tensor,
        initial_j: torch.Tensor,
        initial_d: torch.Tensor,
        lengths: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_cpu: torch.Tensor | None,
        chunk_indices: torch.Tensor | None,
        chunk_size: int,
    ):
        from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, prepare_chunk_offsets

        batch, length, heads = log_decay.shape
        width = u_panel.shape[-1]
        is_varlen = cu_seqlens is not None
        if is_varlen:
            if batch != 1 or lengths is None:
                raise ValueError("variable-length geometry requires flat input and lengths")
            if chunk_indices is None:
                chunk_indices = prepare_chunk_indices(
                    cu_seqlens,
                    chunk_size,
                    cu_seqlens_cpu=cu_seqlens_cpu,
                )
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
            state_batch = len(cu_seqlens) - 1
            length_source = (
                torch.diff(cu_seqlens_cpu)
                if cu_seqlens_cpu is not None
                else lengths
            )
            chunks = triton.cdiv(int(length_source.max().item()), chunk_size)
            panels = len(chunk_indices) * heads
        else:
            chunk_offsets = None
            state_batch = batch
            chunks = triton.cdiv(length, chunk_size)
            panels = batch * heads * chunks
        if is_varlen:
            cumulative = chunk_local_cumsum(
                log_decay,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        else:
            cumulative = chunk_local_cumsum(log_decay, chunk_size=chunk_size)
        j_boundary = torch.empty(
            panels, width, width, dtype=torch.float32, device=u_panel.device
        )
        d_boundary = torch.empty_like(j_boundary)
        m_boundary = torch.empty(
            panels, dtype=torch.float32, device=u_panel.device
        )
        mass = torch.empty(
            panels,
            chunk_size,
            dtype=torch.float32,
            device=u_panel.device,
        )
        cumulative_panel = torch.empty_like(mass)
        tail_weight = torch.empty_like(mass)
        chunk_decay = torch.empty_like(m_boundary)
        j_final = torch.empty_like(initial_j)
        d_final = torch.empty_like(initial_d)
        m_final = torch.empty_like(initial_m)
        block = min(64, max(16, triton.next_power_of_2(width)))
        warps = 8 if block == 64 else 4
        _paired_geometry_forward_kernel[
            (triton.cdiv(width, block), triton.cdiv(width, block), state_batch * heads)
        ](
            u_panel,
            u_d_panel,
            h_panel,
            cumulative,
            initial_j,
            initial_d,
            initial_m,
            j_boundary,
            d_boundary,
            m_boundary,
            mass,
            cumulative_panel,
            tail_weight,
            chunk_decay,
            j_final,
            d_final,
            m_final,
            lengths if lengths is not None else log_decay,
            cu_seqlens if cu_seqlens is not None else log_decay,
            chunk_offsets if chunk_offsets is not None else log_decay,
            T=length,
            H=heads,
            R=width,
            C=chunk_size,
            N=chunks,
            BR=block,
            HAS_LENGTHS=lengths is not None,
            IS_VARLEN=is_varlen,
            num_warps=warps,
            num_stages=2,
        )
        ctx.save_for_backward(
            u_panel,
            h_panel,
            cumulative_panel,
            tail_weight,
            chunk_decay,
            m_boundary,
            j_boundary,
            d_boundary,
            cu_seqlens if cu_seqlens is not None else log_decay,
            chunk_indices if chunk_indices is not None else log_decay,
            chunk_offsets if chunk_offsets is not None else log_decay,
        )
        ctx.shape = (
            batch, length, heads, width, chunks, chunk_size, state_batch,
            panels, is_varlen,
        )
        return (
            mass,
            j_boundary,
            d_boundary,
            m_final,
            j_final,
            d_final,
            cumulative_panel,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            grad_mass,
            grad_j_panel,
            grad_d_panel,
            grad_m_final,
            grad_j_final,
            grad_d_final,
            grad_cumulative_panel,
        ) = grad_outputs
        (
            u_panel,
            h_panel,
            cumulative_panel,
            tail_weight,
            chunk_decay,
            m_boundary,
            j_boundary,
            d_boundary,
            saved_cu,
            saved_chunk_indices,
            saved_chunk_offsets,
        ) = ctx.saved_tensors
        (
            batch, length, heads, width, chunks, chunk_size, state_batch,
            panels, is_varlen,
        ) = ctx.shape

        grad_j_boundary = grad_j_panel
        grad_d_boundary = grad_d_panel
        end_j = torch.empty_like(j_boundary)
        end_d = torch.empty_like(d_boundary)
        grad_j0 = torch.empty_like(grad_j_final)
        grad_d0 = torch.empty_like(grad_d_final)
        block = 16
        _matrix_boundary_reverse_kernel[
            (triton.cdiv(width, block), triton.cdiv(width, block), state_batch * heads)
        ](
            grad_j_boundary,
            grad_d_boundary,
            grad_j_final,
            grad_d_final,
            chunk_decay,
            end_j,
            end_d,
            grad_j0,
            grad_d0,
            saved_chunk_offsets,
            N=chunks,
            R=width,
            H=heads,
            BR=block,
            IS_VARLEN=is_varlen,
            num_warps=4,
        )

        grad_m_boundary = torch.empty_like(m_boundary)
        grad_g_extra = torch.empty_like(cumulative_panel)
        mass_block = triton.next_power_of_2(chunk_size)
        _mass_local_reverse_kernel[(panels,)](
            grad_mass if grad_mass is not None else cumulative_panel,
            (
                grad_cumulative_panel
                if grad_cumulative_panel is not None
                else cumulative_panel
            ),
            cumulative_panel,
            tail_weight,
            m_boundary,
            grad_m_boundary,
            grad_g_extra,
            C=chunk_size,
            BC=mass_block,
            HAS_GRAD_MASS=grad_mass is not None,
            HAS_GRAD_CUMULATIVE=grad_cumulative_panel is not None,
            num_warps=4,
        )
        end_m = torch.empty_like(m_boundary)
        grad_m0 = torch.empty_like(grad_m_final)
        _mass_boundary_reverse_kernel[(state_batch * heads,)](
            grad_m_boundary,
            grad_m_final,
            chunk_decay,
            end_m,
            grad_m0,
            saved_chunk_offsets,
            N=chunks,
            H=heads,
            IS_VARLEN=is_varlen,
            num_warps=1,
        )

        valid = tail_weight > 0.0
        decay = chunk_decay

        ej = end_j
        ed = end_d
        em = end_m
        j0 = j_boundary
        d0 = d_boundary
        m0 = m_boundary

        grad_u_panel = torch.empty_like(u_panel, dtype=torch.float32)
        grad_h_panel = torch.empty_like(h_panel, dtype=torch.float32)
        vector_block = 32
        _geometry_vector_vjp_kernel[
            (panels, triton.cdiv(width, vector_block))
        ](
            u_panel,
            h_panel,
            ej,
            ed,
            tail_weight,
            grad_u_panel,
            grad_h_panel,
            C=chunk_size,
            R=width,
            BC=triton.next_power_of_2(chunk_size),
            BR=vector_block,
            num_warps=8,
            num_stages=2,
        )

        # The decay scalar retains FP32 operands, but MESA-style state tiles
        # now generate compact partials instead of materializing four full
        # FP32 vector panels around two torch.bmm calls.
        scalar_block = 32
        scalar_tiles = triton.cdiv(width, scalar_block)
        observation_partial = torch.empty(
            panels,
            scalar_tiles,
            chunk_size,
            dtype=torch.float32,
            device=u_panel.device,
        )
        boundary_partial = torch.empty(
            panels,
            scalar_tiles,
            dtype=torch.float32,
            device=u_panel.device,
        )
        _geometry_decay_partial_kernel[(scalar_tiles, panels)](
            u_panel,
            h_panel,
            ej,
            ed,
            j0,
            d0,
            observation_partial,
            boundary_partial,
            C=chunk_size,
            R=width,
            BC=triton.next_power_of_2(chunk_size),
            BK=scalar_block,
            NB=scalar_tiles,
            num_warps=8,
            num_stages=2,
        )
        grad_g_panel = torch.empty_like(cumulative_panel)
        _geometry_decay_finalize_kernel[(panels,)](
            observation_partial,
            boundary_partial,
            em,
            m0,
            tail_weight,
            decay,
            grad_g_extra,
            grad_g_panel,
            C=chunk_size,
            BC=triton.next_power_of_2(chunk_size),
            NB=scalar_tiles,
            BNB=triton.next_power_of_2(scalar_tiles),
            num_warps=4,
            num_stages=1,
        )
        grad_g = torch.empty(
            batch,
            length,
            heads,
            dtype=torch.float32,
            device=u_panel.device,
        )
        _panel_scalar_to_bth_kernel[(panels,)](
            grad_g_panel,
            grad_g,
            saved_cu,
            saved_chunk_indices,
            T=length,
            H=heads,
            C=chunk_size,
            N=chunks,
            IS_VARLEN=is_varlen,
            num_warps=1,
        )
        return (
            grad_u_panel,
            None,
            grad_h_panel,
            grad_g,
            grad_m0,
            grad_j0,
            grad_d0,
            None,
            None,
            None,
            None,
            None,
        )


def geometry_scan(
    u_panel: torch.Tensor,
    u_d_panel: torch.Tensor,
    h_panel: torch.Tensor,
    log_decay: torch.Tensor,
    initial_m: torch.Tensor,
    initial_j: torch.Tensor,
    initial_d: torch.Tensor,
    lengths: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    *,
    chunk_size: int,
):
    """FLA/MESA affine geometry boundaries with a composed transpose."""
    return _GeometryScan.apply(
        u_panel,
        u_d_panel,
        h_panel,
        log_decay,
        initial_m,
        initial_j,
        initial_d,
        lengths,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_indices,
        chunk_size,
    )


__all__ = ["geometry_scan"]
