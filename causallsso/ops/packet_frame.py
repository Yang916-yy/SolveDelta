from __future__ import annotations

import torch
import triton
import triton.language as tl

from .mathdx import _load_mathdx


_CHUNK = 16
_RANK = 128
_DUAL_RHS = 2


@triton.jit
def _pack_dual_rhs_kernel(
    source,
    packed,
    chunk_size: tl.constexpr,
    rank: tl.constexpr,
    rhs_count: tl.constexpr,
):
    item = tl.program_id(0)
    panel = item // (chunk_size * rhs_count)
    local = item % (chunk_size * rhs_count)
    target = local // rhs_count
    rhs = local % rhs_count
    coordinates = tl.arange(0, rank)
    source_base = ((panel * chunk_size + target) * rhs_count + rhs) * rank
    packed_base = panel * rank * chunk_size * rhs_count
    tl.store(
        packed + packed_base + coordinates * (chunk_size * rhs_count)
        + target * rhs_count + rhs,
        tl.load(source + source_base + coordinates).to(tl.float32),
    )


@triton.jit
def _packed_boundary_action_kernel(
    boundary_j,
    boundary_d,
    packed,
    alpha,
    coefficient,
    output,
    rank: tl.constexpr,
    chunk_size: tl.constexpr,
    rhs_count: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    upper: tl.constexpr,
    transpose: tl.constexpr,
):
    panel = tl.program_id(0)
    block_row = tl.program_id(1)
    rows = block_row * block_m + tl.arange(0, block_m)
    columns = tl.arange(0, block_n)
    packed_columns: tl.constexpr = chunk_size * rhs_count
    matrix_base = panel * rank * rank
    packed_base = panel * rank * packed_columns
    j_action = tl.zeros((block_m, block_n), tl.float32)
    d_action = tl.zeros((block_m, block_n), tl.float32)
    for start in tl.static_range(0, rank, block_k):
        inner = start + tl.arange(0, block_k)
        if transpose:
            matrix_rows = inner[None, :]
            matrix_columns = rows[:, None]
            active = matrix_rows < matrix_columns if upper else matrix_rows > matrix_columns
        else:
            matrix_rows = rows[:, None]
            matrix_columns = inner[None, :]
            active = matrix_rows < matrix_columns if upper else matrix_rows > matrix_columns
        j_tile = tl.load(
            boundary_j + matrix_base + matrix_rows * rank + matrix_columns,
            mask=active,
            other=0.0,
        )
        d_tile = tl.load(
            boundary_d + matrix_base + matrix_rows * rank + matrix_columns,
            mask=active,
            other=0.0,
        )
        rhs_tile = tl.load(
            packed + packed_base + inner[:, None] * packed_columns + columns[None, :]
        )
        j_action += tl.dot(j_tile, rhs_tile, input_precision="ieee")
        d_action += tl.dot(d_tile, rhs_tile, input_precision="ieee")
    targets = columns // rhs_count
    scalar = panel * chunk_size + targets
    offset: tl.constexpr = 2 if upper else 0
    scale = tl.load(alpha + scalar).to(tl.float32)
    p = tl.load(coefficient + scalar * 4 + offset).to(tl.float32)
    q = tl.load(coefficient + scalar * 4 + offset + 1).to(tl.float32)
    tl.store(
        output + packed_base + rows[:, None] * packed_columns + columns[None, :],
        scale[None, :] * (p[None, :] * j_action + q[None, :] * d_action),
    )


@triton.jit
def _pack_action_rhs_kernel(
    first,
    second,
    packed,
    chunk_size: tl.constexpr,
    rank: tl.constexpr,
    rhs_count: tl.constexpr,
):
    item = tl.program_id(0)
    panel = item // (chunk_size * rhs_count)
    local = item % (chunk_size * rhs_count)
    target = local // rhs_count
    rhs = local % rhs_count
    coordinates = tl.arange(0, rank)
    source_base = (panel * chunk_size + target) * rank
    values = tl.load(
        tl.where(rhs == 0, first, second) + source_base + coordinates
    ).to(tl.float32)
    packed_base = panel * rank * chunk_size * rhs_count
    tl.store(
        packed + packed_base + coordinates * (chunk_size * rhs_count) + local,
        values,
    )


@triton.jit
def _packed_local_direct_epilogue_kernel(
    u,
    h_value,
    weights,
    coefficient,
    diagonal,
    packed,
    boundary_action,
    intermediate,
    scaled_intermediate,
    first_output,
    second_output,
    chunk_size: tl.constexpr,
    rank: tl.constexpr,
    rhs_count: tl.constexpr,
    upper: tl.constexpr,
):
    column = tl.program_id(0)
    panel = tl.program_id(1)
    target = column // rhs_count
    rhs = column % rhs_count
    coordinates = tl.arange(0, rank)
    packed_columns: tl.constexpr = chunk_size * rhs_count
    vector_base = panel * chunk_size * rank
    packed_offsets = (
        panel * rank * packed_columns + coordinates * packed_columns + column
    )
    value = tl.load(packed + packed_offsets).to(tl.float32)
    scalar = panel * chunk_size + target
    offset: tl.constexpr = 2 if upper else 0
    p = tl.load(coefficient + scalar * 4 + offset).to(tl.float32)
    q = tl.load(coefficient + scalar * 4 + offset + 1).to(tl.float32)
    local_action = tl.zeros((rank,), tl.float32)
    for source in tl.static_range(0, chunk_size):
        active = source <= target
        source_base = vector_base + source * rank
        source_u = tl.load(u + source_base + coordinates).to(tl.float32)
        source_h = tl.load(h_value + source_base + coordinates).to(tl.float32)
        right = p * source_u + q * source_h
        product = right * value
        if upper:
            inclusive = tl.cumsum(product, axis=0, reverse=True)
        else:
            inclusive = tl.cumsum(product, axis=0, reverse=False)
        exclusive = inclusive - product
        weight = tl.load(
            weights + panel * chunk_size * chunk_size + source * chunk_size + target
        ).to(tl.float32)
        local_action += tl.where(active, weight * source_u * exclusive, 0.0)
    transformed = value + tl.load(boundary_action + packed_offsets) + local_action
    output_base = (panel * chunk_size + target) * rank
    if upper:
        intermediate_base = ((panel * rhs_count + rhs) * chunk_size + target) * rank
        tl.store(intermediate + intermediate_base + coordinates, transformed)
        scale = tl.load(diagonal + output_base + coordinates).to(tl.float32)
        scaled = transformed * scale
        tl.store(scaled_intermediate + intermediate_base + coordinates, scaled)
        tl.store(packed + packed_offsets, scaled)
    else:
        tl.store(
            first_output + output_base + coordinates,
            transformed,
            mask=rhs == 0,
        )
        tl.store(
            second_output + output_base + coordinates,
            transformed,
            mask=rhs == 1,
        )


@triton.jit
def _finalize_action_diagonal_kernel(
    c_upper,
    diagonal,
    y,
    dual_lower,
    grad_z,
    grad_diagonal,
    chunk_size: tl.constexpr,
    rank: tl.constexpr,
    rhs_count: tl.constexpr,
):
    scalar = tl.program_id(0)
    panel = scalar // chunk_size
    target = scalar % chunk_size
    coordinates = tl.arange(0, rank)
    vector_base = scalar * rank
    first_grad_base = (panel * rhs_count * chunk_size + target) * rank
    second_grad_base = first_grad_base + chunk_size * rank
    first_dual_base = (scalar * rhs_count) * rank
    second_dual_base = first_dual_base + rank
    grad_diagonal_dual = (
        tl.load(grad_z + first_grad_base + coordinates)
        * tl.load(dual_lower + first_dual_base + coordinates)
        + tl.load(grad_z + second_grad_base + coordinates)
        * tl.load(dual_lower + second_dual_base + coordinates)
    )
    scale = tl.load(diagonal + vector_base + coordinates).to(tl.float32)
    primal = (
        tl.load(c_upper + vector_base + coordinates).to(tl.float32)
        * tl.load(y + vector_base + coordinates).to(tl.float32)
        / (scale * scale)
    )
    tl.store(grad_diagonal + vector_base + coordinates, grad_diagonal_dual - primal)


@triton.jit
def _packed_local_transpose_epilogue_kernel(
    u,
    h_value,
    weights,
    coefficient,
    diagonal,
    packed,
    boundary_action,
    output,
    dual_lower,
    chunk_size: tl.constexpr,
    rank: tl.constexpr,
    rhs_count: tl.constexpr,
    lower: tl.constexpr,
):
    column = tl.program_id(0)
    panel = tl.program_id(1)
    target = column // rhs_count
    rhs = column % rhs_count
    coordinates = tl.arange(0, rank)
    packed_columns: tl.constexpr = chunk_size * rhs_count
    vector_base = panel * chunk_size * rank
    packed_offsets = (
        panel * rank * packed_columns + coordinates * packed_columns + column
    )
    value = tl.load(packed + packed_offsets).to(tl.float32)
    local_action = tl.zeros((rank,), tl.float32)
    scalar = panel * chunk_size + target
    offset: tl.constexpr = 0 if lower else 2
    p = tl.load(coefficient + scalar * 4 + offset).to(tl.float32)
    q = tl.load(coefficient + scalar * 4 + offset + 1).to(tl.float32)
    for source in tl.static_range(0, chunk_size):
        active = source <= target
        source_base = vector_base + source * rank
        source_u = tl.load(u + source_base + coordinates).to(tl.float32)
        source_h = tl.load(h_value + source_base + coordinates).to(tl.float32)
        right = p * source_u + q * source_h
        product = source_u * value
        if lower:
            inclusive = tl.cumsum(product, axis=0, reverse=True)
        else:
            inclusive = tl.cumsum(product, axis=0, reverse=False)
        exclusive = inclusive - product
        weight = tl.load(
            weights + panel * chunk_size * chunk_size + source * chunk_size + target
        ).to(tl.float32)
        local_action += tl.where(active, weight * right * exclusive, 0.0)
    transformed = value + tl.load(boundary_action + packed_offsets) + local_action
    output_base = ((panel * chunk_size + target) * rhs_count + rhs) * rank
    if lower:
        tl.store(dual_lower + output_base + coordinates, transformed)
        scale = tl.load(diagonal + vector_base + target * rank + coordinates)
        tl.store(packed + packed_offsets, transformed * scale)
    else:
        tl.store(output + output_base + coordinates, transformed)


@triton.jit
def _pack_frame_inputs_kernel(
    u,
    h_value,
    key,
    erase,
    query,
    log_decay,
    skew,
    packed_u,
    packed_h,
    packed_key,
    packed_erase,
    packed_query,
    packed_log_decay,
    packed_skew,
    length: tl.constexpr,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    rank: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    local_token = row % chunk_size
    system = row // chunk_size
    chunk = system % chunks
    head = (system // chunks) % heads
    batch = system // (heads * chunks)
    token = chunk * chunk_size + local_token
    valid = token < length
    coordinates = tl.arange(0, block)
    coordinate_mask = coordinates < rank
    source_base = ((batch * length + token) * heads + head) * rank
    target_base = row * rank
    mask = valid & coordinate_mask
    tl.store(
        packed_u + target_base + coordinates,
        tl.load(u + source_base + coordinates, mask=mask, other=0.0),
        mask=coordinate_mask,
    )
    tl.store(
        packed_h + target_base + coordinates,
        tl.load(h_value + source_base + coordinates, mask=mask, other=0.0),
        mask=coordinate_mask,
    )
    tl.store(
        packed_key + target_base + coordinates,
        tl.load(key + source_base + coordinates, mask=mask, other=0.0),
        mask=coordinate_mask,
    )
    tl.store(
        packed_erase + target_base + coordinates,
        tl.load(erase + source_base + coordinates, mask=mask, other=0.0),
        mask=coordinate_mask,
    )
    tl.store(
        packed_query + target_base + coordinates,
        tl.load(query + source_base + coordinates, mask=mask, other=0.0),
        mask=coordinate_mask,
    )
    scalar_source = (batch * length + token) * heads + head
    tl.store(
        packed_log_decay + row,
        tl.load(log_decay + scalar_source, mask=valid, other=0.0),
    )
    tl.store(
        packed_skew + row,
        tl.load(skew + scalar_source, mask=valid, other=0.0),
    )


@triton.jit
def _pack_frame_output_grads_kernel(
    grad_d,
    grad_e,
    grad_chi,
    packed_grad_d,
    packed_grad_e,
    packed_grad_chi,
    length: tl.constexpr,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    rank: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    local_token = row % chunk_size
    system = row // chunk_size
    chunk = system % chunks
    head = (system // chunks) % heads
    batch = system // (heads * chunks)
    token = chunk * chunk_size + local_token
    valid = token < length
    coordinates = tl.arange(0, block)
    coordinate_mask = coordinates < rank
    source_base = ((batch * length + token) * heads + head) * rank
    target_base = row * rank
    mask = valid & coordinate_mask
    tl.store(
        packed_grad_d + target_base + coordinates,
        tl.load(grad_d + source_base + coordinates, mask=mask, other=0.0),
        mask=coordinate_mask,
    )
    tl.store(
        packed_grad_e + target_base + coordinates,
        tl.load(grad_e + source_base + coordinates, mask=mask, other=0.0),
        mask=coordinate_mask,
    )
    tl.store(
        packed_grad_chi + target_base + coordinates,
        tl.load(grad_chi + source_base + coordinates, mask=mask, other=0.0),
        mask=coordinate_mask,
    )


@triton.jit
def _unpack_frame_input_grads_kernel(
    packed_grad_u,
    packed_grad_h,
    packed_grad_log_decay,
    packed_grad_key,
    packed_grad_erase,
    packed_grad_query,
    packed_grad_skew,
    grad_u,
    grad_h,
    grad_log_decay,
    grad_key,
    grad_erase,
    grad_query,
    grad_skew,
    length: tl.constexpr,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    rank: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    head = row % heads
    token_batch = row // heads
    token = token_batch % length
    batch = token_batch // length
    chunk = token // chunk_size
    local_token = token % chunk_size
    system = (batch * heads + head) * chunks + chunk
    coordinates = tl.arange(0, block)
    mask = coordinates < rank
    packed_base = (system * chunk_size + local_token) * rank
    output_base = row * rank
    tl.store(
        grad_u + output_base + coordinates,
        tl.load(packed_grad_u + packed_base + coordinates, mask=mask),
        mask=mask,
    )
    tl.store(
        grad_h + output_base + coordinates,
        tl.load(packed_grad_h + packed_base + coordinates, mask=mask),
        mask=mask,
    )
    tl.store(
        grad_key + output_base + coordinates,
        tl.load(packed_grad_key + packed_base + coordinates, mask=mask),
        mask=mask,
    )
    tl.store(
        grad_erase + output_base + coordinates,
        tl.load(packed_grad_erase + packed_base + coordinates, mask=mask),
        mask=mask,
    )
    tl.store(
        grad_query + output_base + coordinates,
        tl.load(packed_grad_query + packed_base + coordinates, mask=mask),
        mask=mask,
    )
    scalar_source = system * chunk_size + local_token
    tl.store(grad_log_decay + row, tl.load(packed_grad_log_decay + scalar_source))
    tl.store(grad_skew + row, tl.load(packed_grad_skew + scalar_source))


@triton.jit
def _reduce_strength_grad_kernel(
    panel_grad,
    grad_strength,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    panels_per_head: tl.constexpr,
    reduce_block: tl.constexpr,
):
    head = tl.program_id(0)
    offsets = tl.arange(0, reduce_block)
    mask = offsets < panels_per_head
    batch = offsets // chunks
    chunk = offsets % chunks
    panel = (batch * heads + head) * chunks + chunk
    values = tl.load(panel_grad + panel, mask=mask, other=0.0)
    tl.store(grad_strength + head, tl.sum(values))


@triton.jit
def _prefix_coefficients_kernel(
    boundary_m,
    log_decay,
    alpha,
    weights,
    length: tl.constexpr,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
):
    system = tl.program_id(0)
    chunk = system % chunks
    offsets = tl.arange(0, chunk_size)
    tokens = chunk * chunk_size + offsets
    valid = tokens < length
    logs = tl.load(
        log_decay + system * chunk_size + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    prefix = tl.cumsum(logs, axis=0)
    targets = offsets[:, None]
    sources = offsets[None, :]
    causal = (sources <= targets) & valid[:, None] & valid[None, :]
    unnormalized = tl.where(
        causal,
        tl.exp(prefix[:, None] - prefix[None, :]),
        0.0,
    )
    mass = tl.exp(prefix) * tl.load(boundary_m + system).to(tl.float32)
    mass += tl.sum(unnormalized, axis=1)
    alpha_value = tl.where(valid, tl.exp(prefix) / mass, 0.0)
    tl.store(alpha + system * chunk_size + offsets, alpha_value)
    normalized = tl.where(causal, unnormalized / mass[:, None], 0.0)
    # Native packet kernels use source-major / target-minor storage.
    tl.store(
        weights
        + system * chunk_size * chunk_size
        + sources.T * chunk_size
        + targets.T,
        normalized.T,
    )


@triton.jit
def _boundary_omega_kernel(
    boundary_j,
    boundary_d,
    omega_input,
    primal_key,
    alpha,
    coefficient,
    output,
    qbar_partial,
    rank: tl.constexpr,
    chunk_size: tl.constexpr,
    block_r: tl.constexpr,
    block_k: tl.constexpr,
):
    row_block = tl.program_id(0)
    system = tl.program_id(1)
    rows = row_block * block_r + tl.arange(0, block_r)
    targets = tl.arange(0, chunk_size)
    matrix_base = system * rank * rank
    vector_base = system * chunk_size * rank
    scalar_base = system * chunk_size
    h_lower = tl.load(coefficient + (scalar_base + targets) * 4 + 0).to(tl.float32)
    r_lower = tl.load(coefficient + (scalar_base + targets) * 4 + 1).to(tl.float32)
    h_upper = tl.load(coefficient + (scalar_base + targets) * 4 + 2).to(tl.float32)
    r_upper = tl.load(coefficient + (scalar_base + targets) * 4 + 3).to(tl.float32)
    alpha_t = tl.load(alpha + scalar_base + targets).to(tl.float32)
    lower_j_action = tl.zeros((block_r, chunk_size), tl.float32)
    lower_r_action = tl.zeros((block_r, chunk_size), tl.float32)
    upper_j_action = tl.zeros((block_r, chunk_size), tl.float32)
    upper_r_action = tl.zeros((block_r, chunk_size), tl.float32)
    for col_block in tl.static_range(0, rank // block_k):
        cols = col_block * block_k + tl.arange(0, block_k)
        j_rc = tl.load(
            boundary_j + matrix_base + rows[:, None] * rank + cols[None, :]
        ).to(tl.float32)
        j_cr = tl.load(
            boundary_j + matrix_base + cols[:, None] * rank + rows[None, :]
        ).to(tl.float32)
        d_rc = tl.load(
            boundary_d + matrix_base + rows[:, None] * rank + cols[None, :]
        ).to(tl.float32)
        d_cr = tl.load(
            boundary_d + matrix_base + cols[:, None] * rank + rows[None, :]
        ).to(tl.float32)
        below = rows[:, None] > cols[None, :]
        above = rows[:, None] < cols[None, :]
        j_lower_skew = tl.where(below, j_rc, 0.0) - tl.where(
            above, tl.trans(j_cr), 0.0
        )
        j_upper_skew = tl.where(above, j_rc, 0.0) - tl.where(
            below, tl.trans(j_cr), 0.0
        )
        d_lower_skew = tl.where(below, d_rc, 0.0) - tl.where(
            above, tl.trans(d_cr), 0.0
        )
        d_upper_skew = tl.where(above, d_rc, 0.0) - tl.where(
            below, tl.trans(d_cr), 0.0
        )
        key_tile = tl.load(
            omega_input + vector_base + targets[None, :] * rank + cols[:, None]
        ).to(tl.float32)
        lower_j_action += tl.dot(j_lower_skew, key_tile, input_precision="ieee")
        lower_r_action += tl.dot(d_lower_skew, key_tile, input_precision="ieee")
        upper_j_action += tl.dot(j_upper_skew, key_tile, input_precision="ieee")
        upper_r_action += tl.dot(d_upper_skew, key_tile, input_precision="ieee")
    result = (
        lower_j_action * h_lower[None, :]
        + lower_r_action * r_lower[None, :]
        + upper_j_action * h_upper[None, :]
        + upper_r_action * r_upper[None, :]
    )
    result *= 0.5 * alpha_t[None, :]
    tl.store(
        output + vector_base + targets[None, :] * rank + rows[:, None],
        result,
    )
    contraction_key = tl.load(
        primal_key + vector_base + targets[None, :] * rank + rows[:, None]
    ).to(tl.float32)
    lower_h_bar = -0.5 * tl.sum(
        contraction_key * lower_j_action, axis=0
    )
    lower_r_bar = -0.5 * tl.sum(
        contraction_key * lower_r_action, axis=0
    )
    upper_h_bar = -0.5 * tl.sum(
        contraction_key * upper_j_action, axis=0
    )
    upper_r_bar = -0.5 * tl.sum(
        contraction_key * upper_r_action, axis=0
    )
    partial_base = (
        ((system * (rank // block_r) + row_block) * chunk_size + targets) * 4
    )
    tl.store(qbar_partial + partial_base + 0, lower_h_bar)
    tl.store(qbar_partial + partial_base + 1, lower_r_bar)
    tl.store(qbar_partial + partial_base + 2, upper_h_bar)
    tl.store(qbar_partial + partial_base + 3, upper_r_bar)


@triton.jit
def _unpack_frame_outputs_kernel(
    packed_d,
    packed_dual,
    d,
    e,
    chi,
    length: tl.constexpr,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    rank: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    head = row % heads
    token_batch = row // heads
    token = token_batch % length
    batch = token_batch // length
    chunk = token // chunk_size
    local_token = token % chunk_size
    system = (batch * heads + head) * chunks + chunk
    coordinates = tl.arange(0, block)
    mask = coordinates < rank
    packed_vector_base = (system * chunk_size + local_token) * rank
    output_base = row * rank
    tl.store(
        d + output_base + coordinates,
        tl.load(packed_d + packed_vector_base + coordinates, mask=mask),
        mask=mask,
    )
    packed_dual_base = (system * chunk_size + local_token) * 2 * rank
    tl.store(
        e + output_base + coordinates,
        tl.load(packed_dual + packed_dual_base + coordinates, mask=mask),
        mask=mask,
    )
    tl.store(
        chi + output_base + coordinates,
        tl.load(packed_dual + packed_dual_base + rank + coordinates, mask=mask),
        mask=mask,
    )


def _pack_frame_inputs(
    u: torch.Tensor,
    h: torch.Tensor,
    key: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    skew: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    batch, length, heads, rank = u.shape
    chunks = triton.cdiv(length, _CHUNK)
    programs = batch * heads * chunks
    vectors = tuple(
        torch.empty(programs, _CHUNK, rank, device=u.device, dtype=torch.float32)
        for _ in range(5)
    )
    packed_log_decay = torch.empty(
        programs, _CHUNK, device=u.device, dtype=torch.float32
    )
    packed_skew = torch.empty_like(packed_log_decay)
    block = triton.next_power_of_2(rank)
    _pack_frame_inputs_kernel[(programs * _CHUNK,)](
        u,
        h,
        key,
        erase,
        query,
        geometry_log_decay,
        skew,
        *vectors,
        packed_log_decay,
        packed_skew,
        length=length,
        heads=heads,
        chunks=chunks,
        chunk_size=_CHUNK,
        rank=rank,
        block=block,
        num_warps=4,
    )
    return (*vectors, packed_log_decay, packed_skew)


def _pack_frame_output_grads(
    grad_d: torch.Tensor,
    grad_e: torch.Tensor,
    grad_chi: torch.Tensor,
    *,
    batch: int,
    length: int,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunks = triton.cdiv(length, _CHUNK)
    programs = batch * heads * chunks
    packed = tuple(
        torch.empty(
            programs,
            _CHUNK,
            _RANK,
            device=grad_chi.device,
            dtype=torch.float32,
        )
        for _ in range(3)
    )
    _pack_frame_output_grads_kernel[(programs * _CHUNK,)](
        grad_d.contiguous(),
        grad_e.contiguous(),
        grad_chi.contiguous(),
        *packed,
        length=length,
        heads=heads,
        chunks=chunks,
        chunk_size=_CHUNK,
        rank=_RANK,
        block=_RANK,
        num_warps=4,
    )
    return packed


def _unpack_frame_input_grads(
    packed_grad_u: torch.Tensor,
    packed_grad_h: torch.Tensor,
    packed_grad_log_decay: torch.Tensor,
    packed_grad_key: torch.Tensor,
    packed_grad_erase: torch.Tensor,
    packed_grad_query: torch.Tensor,
    packed_grad_skew: torch.Tensor,
    *,
    batch: int,
    length: int,
    heads: int,
) -> tuple[torch.Tensor, ...]:
    chunks = triton.cdiv(length, _CHUNK)
    vector_shape = (batch, length, heads, _RANK)
    edit_shape = (batch, length, heads, 1, _RANK)
    grad_u = torch.empty(vector_shape, device=packed_grad_u.device, dtype=torch.float32)
    grad_h = torch.empty_like(grad_u)
    grad_log_decay = torch.empty(
        batch, length, heads, device=packed_grad_u.device, dtype=torch.float32
    )
    grad_key = torch.empty(edit_shape, device=packed_grad_u.device, dtype=torch.float32)
    grad_erase = torch.empty_like(grad_key)
    grad_query = torch.empty_like(grad_u)
    grad_skew = torch.empty(
        batch, length, heads, 1, device=packed_grad_u.device, dtype=torch.float32
    )
    _unpack_frame_input_grads_kernel[(batch * length * heads,)](
        packed_grad_u,
        packed_grad_h,
        packed_grad_log_decay,
        packed_grad_key,
        packed_grad_erase,
        packed_grad_query,
        packed_grad_skew,
        grad_u,
        grad_h,
        grad_log_decay,
        grad_key,
        grad_erase,
        grad_query,
        grad_skew,
        length=length,
        heads=heads,
        chunks=chunks,
        chunk_size=_CHUNK,
        rank=_RANK,
        block=_RANK,
        num_warps=4,
    )
    return (
        grad_u,
        grad_h,
        grad_log_decay,
        grad_key,
        grad_erase,
        grad_query,
        grad_skew,
    )


def _reduce_strength_grad(
    panel_grad: torch.Tensor,
    *,
    batch: int,
    heads: int,
    chunks: int,
) -> torch.Tensor:
    panels_per_head = batch * chunks
    grad_strength = torch.empty(
        heads, device=panel_grad.device, dtype=torch.float32
    )
    _reduce_strength_grad_kernel[(heads,)](
        panel_grad,
        grad_strength,
        heads=heads,
        chunks=chunks,
        panels_per_head=panels_per_head,
        reduce_block=triton.next_power_of_2(panels_per_head),
        num_warps=4,
        num_stages=1,
    )
    return grad_strength


def _packet_parameters(
    boundary_m: torch.Tensor,
    boundary_J: torch.Tensor,
    boundary_D: torch.Tensor,
    packed_u: torch.Tensor,
    packed_h: torch.Tensor,
    packed_log_decay: torch.Tensor,
    geometry_strength: torch.Tensor,
    *,
    length: int,
    heads: int,
    return_aux: bool = False,
) -> tuple[torch.Tensor, ...]:
    _load_mathdx()
    programs = boundary_m.numel()
    chunks = boundary_m.shape[2]
    flat_m = boundary_m.reshape(programs)
    flat_J = boundary_J.reshape(programs, _RANK, _RANK)
    flat_D = boundary_D.reshape_as(flat_J)
    alpha = torch.empty(programs, _CHUNK, device=boundary_m.device, dtype=torch.float32)
    weights = torch.empty(
        programs, _CHUNK, _CHUNK, device=boundary_m.device, dtype=torch.float32
    )
    _prefix_coefficients_kernel[(programs,)](
        flat_m,
        packed_log_decay,
        alpha,
        weights,
        length=length,
        heads=heads,
        chunks=chunks,
        chunk_size=_CHUNK,
        num_warps=4,
        num_stages=1,
    )

    coefficient, diagonal, norm_sq, diagonal_h, diagonal_r = (
        torch.ops.causallsso.packet_frame_radial_forward128(
            flat_J,
            flat_D,
            packed_u,
            packed_h,
            alpha,
            weights,
            geometry_strength,
            heads,
            chunks,
            length,
        )
    )
    result = (alpha, weights, coefficient, diagonal)
    if return_aux:
        return (*result, norm_sq, diagonal_h, diagonal_r)
    return result


def _packet_dual2(
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    packed_u: torch.Tensor,
    packed_h: torch.Tensor,
    weights: torch.Tensor,
    alpha: torch.Tensor,
    coefficient: torch.Tensor,
    diagonal: torch.Tensor,
    dual_rhs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact two-RHS transpose-dual through a C16 packet."""
    programs = boundary_j.shape[0]
    packed_columns = _CHUNK * _DUAL_RHS
    packed = torch.empty(
        programs,
        _RANK,
        packed_columns,
        device=boundary_j.device,
        dtype=torch.float32,
    )
    boundary_action = torch.empty_like(packed)
    dual = torch.empty_like(dual_rhs)
    dual_lower = torch.empty_like(dual_rhs)
    _pack_dual_rhs_kernel[(programs * _CHUNK * _DUAL_RHS,)](
        dual_rhs,
        packed,
        chunk_size=_CHUNK,
        rank=_RANK,
        rhs_count=_DUAL_RHS,
        num_warps=4,
        num_stages=1,
    )
    _packed_boundary_action_kernel[(programs, _RANK // 64)](
        boundary_j,
        boundary_d,
        packed,
        alpha,
        coefficient,
        boundary_action,
        rank=_RANK,
        chunk_size=_CHUNK,
        rhs_count=_DUAL_RHS,
        block_m=64,
        block_n=packed_columns,
        block_k=32,
        upper=False,
        transpose=True,
        num_warps=8,
        num_stages=1,
    )
    _packed_local_transpose_epilogue_kernel[(packed_columns, programs)](
        packed_u,
        packed_h,
        weights,
        coefficient,
        diagonal,
        packed,
        boundary_action,
        dual,
        dual_lower,
        chunk_size=_CHUNK,
        rank=_RANK,
        rhs_count=_DUAL_RHS,
        lower=True,
        num_warps=1,
        num_stages=1,
    )
    _packed_boundary_action_kernel[(programs, _RANK // 64)](
        boundary_j,
        boundary_d,
        packed,
        alpha,
        coefficient,
        boundary_action,
        rank=_RANK,
        chunk_size=_CHUNK,
        rhs_count=_DUAL_RHS,
        block_m=64,
        block_n=packed_columns,
        block_k=32,
        upper=True,
        transpose=True,
        num_warps=8,
        num_stages=1,
    )
    _packed_local_transpose_epilogue_kernel[(packed_columns, programs)](
        packed_u,
        packed_h,
        weights,
        coefficient,
        diagonal,
        packed,
        boundary_action,
        dual,
        dual_lower,
        chunk_size=_CHUNK,
        rank=_RANK,
        rhs_count=_DUAL_RHS,
        lower=False,
        num_warps=1,
        num_stages=1,
    )
    return dual, dual_lower


def _packet_direct2(
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    packed_u: torch.Tensor,
    packed_h: torch.Tensor,
    weights: torch.Tensor,
    alpha: torch.Tensor,
    coefficient: torch.Tensor,
    diagonal: torch.Tensor,
    dual_rhs: torch.Tensor,
    dual_lower: torch.Tensor,
    first_rhs: torch.Tensor,
    second_rhs: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Apply the exact two-RHS direct action used by the dual VJP."""
    programs = boundary_j.shape[0]
    packed_columns = _CHUNK * _DUAL_RHS
    packed = torch.empty(
        programs,
        _RANK,
        packed_columns,
        device=boundary_j.device,
        dtype=torch.float32,
    )
    boundary_action = torch.empty_like(packed)
    grad_z = torch.empty(
        programs,
        _DUAL_RHS,
        _CHUNK,
        _RANK,
        device=boundary_j.device,
        dtype=torch.float32,
    )
    grad_dual_lower = torch.empty_like(grad_z)
    grad_first = torch.empty_like(first_rhs)
    grad_second = torch.empty_like(second_rhs)
    _pack_action_rhs_kernel[(programs * packed_columns,)](
        first_rhs,
        second_rhs,
        packed,
        chunk_size=_CHUNK,
        rank=_RANK,
        rhs_count=_DUAL_RHS,
        num_warps=4,
        num_stages=1,
    )
    _packed_boundary_action_kernel[(programs, _RANK // 64)](
        boundary_j,
        boundary_d,
        packed,
        alpha,
        coefficient,
        boundary_action,
        rank=_RANK,
        chunk_size=_CHUNK,
        rhs_count=_DUAL_RHS,
        block_m=64,
        block_n=packed_columns,
        block_k=32,
        upper=True,
        transpose=False,
        num_warps=8,
        num_stages=1,
    )
    _packed_local_direct_epilogue_kernel[(packed_columns, programs)](
        packed_u,
        packed_h,
        weights,
        coefficient,
        diagonal,
        packed,
        boundary_action,
        grad_z,
        grad_dual_lower,
        grad_first,
        grad_second,
        chunk_size=_CHUNK,
        rank=_RANK,
        rhs_count=_DUAL_RHS,
        upper=True,
        num_warps=1,
        num_stages=1,
    )
    _packed_boundary_action_kernel[(programs, _RANK // 64)](
        boundary_j,
        boundary_d,
        packed,
        alpha,
        coefficient,
        boundary_action,
        rank=_RANK,
        chunk_size=_CHUNK,
        rhs_count=_DUAL_RHS,
        block_m=64,
        block_n=packed_columns,
        block_k=32,
        upper=False,
        transpose=False,
        num_warps=8,
        num_stages=1,
    )
    _packed_local_direct_epilogue_kernel[(packed_columns, programs)](
        packed_u,
        packed_h,
        weights,
        coefficient,
        diagonal,
        packed,
        boundary_action,
        grad_z,
        grad_dual_lower,
        grad_first,
        grad_second,
        chunk_size=_CHUNK,
        rank=_RANK,
        rhs_count=_DUAL_RHS,
        upper=False,
        num_warps=1,
        num_stages=1,
    )
    return grad_z, grad_dual_lower, grad_first, grad_second


def _packet_frame_forward(
    boundary_m: torch.Tensor,
    boundary_J: torch.Tensor,
    boundary_D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    keys: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    skew: torch.Tensor,
    geometry_strength: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    _load_mathdx()
    batch, length, heads, rank = u.shape
    chunks = boundary_m.shape[2]
    programs = batch * heads * chunks
    key = keys.squeeze(-2)
    erase_vector = erase.squeeze(-2)
    skew_scalar = skew.squeeze(-1)
    (
        packed_u,
        packed_h,
        packed_key,
        packed_erase,
        packed_query,
        packed_log_decay,
        packed_skew,
    ) = _pack_frame_inputs(
        u,
        h,
        key,
        erase_vector,
        query,
        geometry_log_decay,
        skew_scalar,
    )
    (
        alpha,
        weights,
        coefficient,
        diagonal,
        norm_sq,
        diagonal_h,
        diagonal_r,
    ) = _packet_parameters(
        boundary_m,
        boundary_J,
        boundary_D,
        packed_u,
        packed_h,
        packed_log_decay,
        geometry_strength,
        length=length,
        heads=heads,
        return_aux=True,
    )
    flat_J = boundary_J.reshape(programs, rank, rank)
    flat_D = boundary_D.reshape_as(flat_J)
    boundary_omega = torch.empty_like(packed_key)
    packed_d, y, dual_rhs = torch.ops.causallsso.packet_frame128(
        flat_J,
        flat_D,
        packed_u,
        packed_h,
        weights,
        alpha,
        coefficient,
        diagonal,
        packed_key,
        packed_erase,
        packed_query,
        packed_skew,
        boundary_omega,
    )
    packed_dual, dual_lower = _packet_dual2(
        flat_J,
        flat_D,
        packed_u,
        packed_h,
        weights,
        alpha,
        coefficient,
        diagonal,
        dual_rhs,
    )
    d = torch.empty_like(keys)
    e = torch.empty_like(keys)
    chi = torch.empty_like(query)
    block = triton.next_power_of_2(rank)
    _unpack_frame_outputs_kernel[(batch * length * heads,)](
        packed_d,
        packed_dual,
        d,
        e,
        chi,
        length=length,
        heads=heads,
        chunks=chunks,
        chunk_size=_CHUNK,
        rank=rank,
        block=block,
        num_warps=4,
    )
    saved = (
        boundary_m.reshape(programs),
        flat_J,
        flat_D,
        packed_u,
        packed_h,
        packed_key,
        packed_erase,
        packed_log_decay,
        packed_skew,
        geometry_strength,
        alpha,
        weights,
        coefficient,
        diagonal,
        norm_sq,
        diagonal_h,
        diagonal_r,
        y,
        boundary_omega,
        dual_rhs,
        dual_lower,
        packed_d,
    )
    return (d, e, chi), saved


def _packet_frame_backward(
    saved: tuple[torch.Tensor, ...],
    grad_d: torch.Tensor,
    grad_e: torch.Tensor,
    grad_chi: torch.Tensor,
    *,
    batch: int,
    length: int,
    heads: int,
    chunks: int,
) -> tuple[torch.Tensor, ...]:
    (
        boundary_m,
        boundary_j,
        boundary_d,
        packed_u,
        packed_h,
        packed_key,
        packed_erase,
        packed_log_decay,
        packed_skew,
        strength,
        alpha,
        weights,
        coefficient,
        diagonal,
        norm_sq,
        diagonal_h,
        diagonal_r,
        y,
        omega_key,
        dual_rhs,
        dual_lower,
        packed_d,
    ) = saved
    packed_grad_d, packed_grad_e, packed_grad_chi = _pack_frame_output_grads(
        grad_d,
        grad_e,
        grad_chi,
        batch=batch,
        length=length,
        heads=heads,
    )

    c_upper, c_lower = torch.ops.causallsso.packet_frame_action_vjp128(
        boundary_j,
        boundary_d,
        packed_u,
        packed_h,
        weights,
        alpha,
        coefficient,
        diagonal,
        packed_grad_d,
    )
    grad_z, grad_dual_lower, grad_b, grad_query = _packet_direct2(
        boundary_j,
        boundary_d,
        packed_u,
        packed_h,
        weights,
        alpha,
        coefficient,
        diagonal,
        dual_rhs,
        dual_lower,
        packed_grad_e,
        packed_grad_chi,
    )
    grad_diagonal = torch.empty_like(diagonal)
    _finalize_action_diagonal_kernel[(boundary_j.shape[0] * _CHUNK,)](
        c_upper,
        diagonal,
        y,
        dual_lower,
        grad_z,
        grad_diagonal,
        chunk_size=_CHUNK,
        rank=_RANK,
        rhs_count=_DUAL_RHS,
        num_warps=4,
        num_stages=1,
    )
    action = (
        c_upper,
        c_lower,
        grad_dual_lower,
        grad_z,
        grad_diagonal,
        c_lower,
        grad_b,
        grad_query,
    )
    descriptors = torch.ops.causallsso.packet_frame_descriptor_vjp128(
        packed_key,
        packed_erase,
        packed_skew,
        omega_key,
        dual_rhs,
        diagonal,
        y,
        packed_d,
        dual_lower,
        packed_grad_e,
        packed_grad_chi,
        c_upper,
        c_lower,
        grad_dual_lower,
        action[6],
    )
    factor = torch.ops.causallsso.packet_frame_rank5_vjp128(
        boundary_j,
        boundary_d,
        packed_u,
        packed_h,
        weights,
        alpha,
        coefficient,
        *descriptors[:4],
    )
    boundary_contraction = factor[6]

    boundary_omega_action = torch.empty_like(descriptors[4])
    boundary_omega_qbar = torch.empty(
        boundary_m.numel(),
        _RANK // 16,
        _CHUNK,
        4,
        device=boundary_m.device,
        dtype=torch.float32,
    )
    _boundary_omega_kernel[(_RANK // 16, boundary_m.numel())](
        boundary_j,
        boundary_d,
        descriptors[4],
        packed_key,
        alpha,
        coefficient,
        boundary_omega_action,
        boundary_omega_qbar,
        rank=_RANK,
        chunk_size=_CHUNK,
        block_r=16,
        block_k=16,
        num_warps=8,
        num_stages=1,
    )
    packed_grad_key, _ = torch.ops.causallsso.packet_frame_omega_vjp128(
        packed_u,
        packed_h,
        weights,
        coefficient,
        descriptors[4],
        boundary_omega_action,
        boundary_omega_qbar,
        descriptors[5],
    )

    chart = torch.ops.causallsso.packet_frame_chart_vjp128(
        boundary_contraction,
        factor[5],
        alpha,
        norm_sq,
        strength,
        diagonal_h,
        diagonal_r,
        diagonal,
        action[4],
        heads,
        chunks,
    )
    correction = torch.ops.causallsso.packet_frame_radial_vjp128(
        boundary_j,
        boundary_d,
        packed_u,
        packed_h,
        weights,
        alpha,
        chart[0],
        chart[1],
        chart[2],
    )
    grad_alpha = chart[4] + correction[5]
    grad_weights = factor[4] + correction[4]
    _, packed_grad_log_decay = (
        torch.ops.causallsso.packet_frame_prefix_vjp128(
            boundary_m,
            packed_log_decay,
            grad_alpha,
            grad_weights,
        )
    )
    grad_boundary_m = chart[5]

    packed_grads = _unpack_frame_input_grads(
        factor[2] + correction[2],
        factor[3] + correction[3],
        packed_grad_log_decay,
        packed_grad_key,
        descriptors[6],
        action[7],
        descriptors[7],
        batch=batch,
        length=length,
        heads=heads,
    )
    grad_strength = _reduce_strength_grad(
        chart[3], batch=batch, heads=heads, chunks=chunks
    )
    boundary_shape = (batch, heads, chunks)
    matrix_shape = (*boundary_shape, _RANK, _RANK)
    return (
        grad_boundary_m.reshape(boundary_shape),
        (factor[0] + correction[0]).reshape(matrix_shape),
        (factor[1] + correction[1]).reshape(matrix_shape),
        *packed_grads,
        grad_strength,
    )


class _PacketFrame128(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *tensors):
        outputs, saved = _packet_frame_forward(*tensors)
        ctx.save_for_backward(*saved)
        batch, length, heads, _ = tensors[3].shape
        ctx.input_layout = (batch, length, heads, tensors[0].shape[2])
        return outputs

    @staticmethod
    def backward(ctx, grad_d, grad_e, grad_chi):
        batch, length, heads, chunks = ctx.input_layout
        return _packet_frame_backward(
            ctx.saved_tensors,
            grad_d,
            grad_e,
            grad_chi,
            batch=batch,
            length=length,
            heads=heads,
            chunks=chunks,
        )


def packet_frame128(
    boundary_m: torch.Tensor,
    boundary_J: torch.Tensor,
    boundary_D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    keys: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    skew: torch.Tensor,
    geometry_strength: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact fixed-C16 packet frame forward for ``r=128, K=1``.

    Geometry boundary states use ``[B,H,chunks,...]`` layout. Vector inputs use
    the public ``[B,T,H,...]`` layout and are packed once by a fused Triton copy;
    the returned frame vectors are unpacked together in one kernel. The native
    CUDA operator never materializes a tokenwise frame.
    """
    if u.ndim != 4 or u.shape[-1] != _RANK:
        raise ValueError("packet_frame128 requires u [B,T,H,128]")
    batch, length, heads, rank = u.shape
    chunks = triton.cdiv(length, _CHUNK)
    if boundary_m.shape != (batch, heads, chunks):
        raise ValueError("boundary_m must be [B,H,ceil(T/16)]")
    matrix_shape = (batch, heads, chunks, rank, rank)
    if boundary_J.shape != matrix_shape or boundary_D.shape != matrix_shape:
        raise ValueError("boundary_J and boundary_D shape mismatch")
    if h.shape != u.shape or query.shape != u.shape:
        raise ValueError("h and query must match u")
    if keys.shape != (batch, length, heads, 1, rank) or erase.shape != keys.shape:
        raise ValueError("keys and erase must be [B,T,H,1,128]")
    if geometry_log_decay.shape != (batch, length, heads):
        raise ValueError("geometry_log_decay must be [B,T,H]")
    if skew.shape != (batch, length, heads, 1):
        raise ValueError("skew must be [B,T,H,1]")
    if geometry_strength.shape != (heads,):
        raise ValueError("geometry_strength must be [H]")
    tensors = (
        boundary_m,
        boundary_J,
        boundary_D,
        u,
        h,
        geometry_log_decay,
        keys,
        erase,
        query,
        skew,
        geometry_strength,
    )
    if any(x.device != u.device or x.device.type != "cuda" for x in tensors):
        raise ValueError("all packet-frame tensors must share one CUDA device")
    if any(x.dtype != torch.float32 for x in tensors):
        raise TypeError("packet_frame128 supports FP32 inputs and states only")
    return _PacketFrame128.apply(*tensors)
