from __future__ import annotations

from typing import Literal

import torch
import triton
import triton.language as tl

from causallsso.reference import SolveDeltaState


@triton.jit
def _twofold_product_add(high, low, left, right):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 product, product_err, summed, virtual, scratch;
            .reg .f32 sum_err, correction;
            mul.rn.f32 product, $4, $5;
            neg.f32 scratch, product;
            fma.rn.f32 product_err, $4, $5, scratch;
            add.rn.f32 summed, $2, product;
            sub.rn.f32 virtual, summed, $2;
            sub.rn.f32 scratch, summed, virtual;
            sub.rn.f32 scratch, $2, scratch;
            sub.rn.f32 sum_err, product, virtual;
            add.rn.f32 sum_err, scratch, sum_err;
            add.rn.f32 correction, $3, product_err;
            add.rn.f32 correction, correction, sum_err;
            add.rn.f32 $0, summed, correction;
            sub.rn.f32 scratch, $0, summed;
            sub.rn.f32 $1, correction, scratch;
        }
        """,
        "=f,=f,f,f,f,f",
        [high, low, left, right],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _chunk_weights_kernel(
    log_decay,
    weights,
    chunk_lambda,
    chunk_mass,
    length: tl.constexpr,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    stride_b: tl.constexpr,
    stride_t: tl.constexpr,
    stride_h: tl.constexpr,
    write_mass: tl.constexpr,
):
    program = tl.program_id(0)
    chunk = program % chunks
    head = (program // chunks) % heads
    batch = program // (chunks * heads)
    offsets = tl.arange(0, chunk_size)
    tokens = chunk * chunk_size + offsets
    mask = tokens < length
    pointer = log_decay + batch * stride_b + tokens * stride_t + head * stride_h
    logs = tl.load(pointer, mask=mask, other=0.0).to(tl.float32)
    suffix_inclusive = tl.cumsum(logs, axis=0, reverse=True)
    suffix_exclusive = suffix_inclusive - logs
    local_weights = tl.exp(suffix_exclusive)
    local_weights = tl.where(mask, local_weights, 0.0)
    base = ((batch * heads + head) * chunks + chunk) * chunk_size
    tl.store(weights + base + offsets, local_weights)
    summary_index = (batch * heads + head) * chunks + chunk
    tl.store(chunk_lambda + summary_index, tl.exp(tl.sum(logs, axis=0)))
    if write_mass:
        tl.store(chunk_mass + summary_index, tl.sum(local_weights, axis=0))


@triton.jit
def _chunk_matrix_summary_kernel(
    u,
    h_value,
    weights,
    summary_J,
    summary_D,
    length: tl.constexpr,
    heads: tl.constexpr,
    rank: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    block_r: tl.constexpr,
    input_precision: tl.constexpr,
    stride_ub: tl.constexpr,
    stride_ut: tl.constexpr,
    stride_uh: tl.constexpr,
    stride_ur: tl.constexpr,
    stride_hb: tl.constexpr,
    stride_ht: tl.constexpr,
    stride_hh: tl.constexpr,
    stride_hr: tl.constexpr,
):
    chunk = tl.program_id(0)
    head_batch = tl.program_id(1)
    row_block = tl.program_id(2) // tl.cdiv(rank, block_r)
    col_block = tl.program_id(2) % tl.cdiv(rank, block_r)
    batch = head_batch // heads
    head = head_batch % heads
    rows = row_block * block_r + tl.arange(0, block_r)
    cols = col_block * block_r + tl.arange(0, block_r)
    local_t = tl.arange(0, chunk_size)
    tokens = chunk * chunk_size + local_t
    token_mask = tokens < length

    u_left_ptr = (
        u
        + batch * stride_ub
        + tokens[None, :] * stride_ut
        + head * stride_uh
        + rows[:, None] * stride_ur
    )
    u_right_ptr = (
        u
        + batch * stride_ub
        + tokens[:, None] * stride_ut
        + head * stride_uh
        + cols[None, :] * stride_ur
    )
    h_right_ptr = (
        h_value
        + batch * stride_hb
        + tokens[:, None] * stride_ht
        + head * stride_hh
        + cols[None, :] * stride_hr
    )
    mask_left = (rows[:, None] < rank) & token_mask[None, :]
    mask_right = token_mask[:, None] & (cols[None, :] < rank)
    u_left = tl.load(u_left_ptr, mask=mask_left, other=0.0)
    u_right = tl.load(u_right_ptr, mask=mask_right, other=0.0)
    h_right = tl.load(h_right_ptr, mask=mask_right, other=0.0)
    weight_base = (head_batch * chunks + chunk) * chunk_size
    local_weights = tl.load(weights + weight_base + local_t)
    # Preserve the tensor-core input dtype while accumulating the dot product
    # in FP32. Multiplying BF16/FP16 by FP32 otherwise promotes only the left
    # operand and makes the two dot operands ill-typed.
    weighted_left = u_left * local_weights.to(u_left.dtype)[None, :]
    j_tile = tl.dot(weighted_left, u_right, input_precision=input_precision)
    # D keeps the raw h operand's representation. In the production mixed
    # schedule this is BF16, so the already bounded FP16 u panel is packed to
    # BF16 only for this named contraction; J retains both FP16 operands.
    d_tile = tl.dot(
        weighted_left.to(h_right.dtype),
        h_right,
        input_precision=input_precision,
    )

    output_base = (head_batch * chunks + chunk) * rank * rank
    output_offsets = output_base + rows[:, None] * rank + cols[None, :]
    output_mask = (rows[:, None] < rank) & (cols[None, :] < rank)
    tl.store(summary_J + output_offsets, j_tile, mask=output_mask)
    tl.store(summary_D + output_offsets, d_tile, mask=output_mask)


@triton.jit
def _matrix_boundary_scan_kernel(
    chunk_J,
    chunk_D,
    chunk_lambda,
    initial_J,
    initial_D,
    final_J,
    final_D,
    heads: tl.constexpr,
    rank: tl.constexpr,
    chunks: tl.constexpr,
    block: tl.constexpr,
):
    tile = tl.program_id(0)
    head_batch = tl.program_id(1)
    offsets = tile * block + tl.arange(0, block)
    matrix_size = rank * rank
    mask = offsets < matrix_size
    current_J = tl.load(initial_J + head_batch * matrix_size + offsets, mask=mask, other=0.0)
    current_D = tl.load(initial_D + head_batch * matrix_size + offsets, mask=mask, other=0.0)
    for chunk in tl.range(0, chunks):
        boundary_base = (head_batch * chunks + chunk) * matrix_size
        summary_J = tl.load(
            chunk_J + boundary_base + offsets, mask=mask, other=0.0
        )
        summary_D = tl.load(
            chunk_D + boundary_base + offsets, mask=mask, other=0.0
        )
        factor = tl.load(chunk_lambda + head_batch * chunks + chunk)
        tl.store(chunk_J + boundary_base + offsets, current_J, mask=mask)
        tl.store(chunk_D + boundary_base + offsets, current_D, mask=mask)
        current_J = factor * current_J + summary_J
        current_D = factor * current_D + summary_D
    tl.store(final_J + head_batch * matrix_size + offsets, current_J, mask=mask)
    tl.store(final_D + head_batch * matrix_size + offsets, current_D, mask=mask)


@triton.jit
def _mass_boundary_scan_kernel(
    chunk_lambda,
    chunk_mass,
    initial_m,
    boundary_m,
    final_m,
    heads: tl.constexpr,
    chunks: tl.constexpr,
):
    head_batch = tl.program_id(0)
    current = tl.load(initial_m + head_batch)
    for chunk in tl.range(0, chunks):
        index = head_batch * chunks + chunk
        tl.store(boundary_m + index, current)
        current = tl.load(chunk_lambda + index) * current + tl.load(chunk_mass + index)
    tl.store(final_m + head_batch, current)


@triton.jit
def _matrix_adjoint_scan_kernel(
    boundary_J,
    boundary_D,
    grad_boundary_J,
    grad_boundary_D,
    grad_final_J,
    grad_final_D,
    chunk_lambda,
    grad_summary_J,
    grad_summary_D,
    grad_initial_J,
    grad_initial_D,
    lambda_partial,
    rank: tl.constexpr,
    chunks: tl.constexpr,
    tile_size: tl.constexpr,
    tiles: tl.constexpr,
):
    tile = tl.program_id(0)
    head_batch = tl.program_id(1)
    matrix_size = rank * rank
    offsets = tile * tile_size + tl.arange(0, tile_size)
    mask = offsets < matrix_size
    rows = offsets // rank
    columns = offsets % rank
    transpose_offsets = columns * rank + rows
    state_base = head_batch * matrix_size
    carry_J_direct = tl.load(
        grad_final_J + state_base + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    carry_J_transpose = tl.load(
        grad_final_J + state_base + transpose_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    carry_J = 0.5 * (carry_J_direct + carry_J_transpose)
    carry_D = tl.load(
        grad_final_D + state_base + offsets, mask=mask, other=0.0
    ).to(tl.float32)

    for reverse in tl.range(0, chunks):
        chunk = chunks - 1 - reverse
        panel = head_batch * chunks + chunk
        panel_base = panel * matrix_size
        tl.store(grad_summary_J + panel_base + offsets, carry_J, mask=mask)
        tl.store(grad_summary_D + panel_base + offsets, carry_D, mask=mask)
        state_J = tl.load(
            boundary_J + panel_base + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        state_D = tl.load(
            boundary_D + panel_base + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        tl.store(
            lambda_partial + panel * tiles + tile,
            tl.sum(carry_J * state_J + carry_D * state_D, axis=0),
        )
        decay = tl.load(chunk_lambda + panel).to(tl.float32)
        local_J_direct = tl.load(
            grad_boundary_J + panel_base + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        local_J_transpose = tl.load(
            grad_boundary_J + panel_base + transpose_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        local_J = 0.5 * (local_J_direct + local_J_transpose)
        local_D = tl.load(
            grad_boundary_D + panel_base + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        carry_J = local_J + decay * carry_J
        carry_D = local_D + decay * carry_D

    tl.store(grad_initial_J + state_base + offsets, carry_J, mask=mask)
    tl.store(grad_initial_D + state_base + offsets, carry_D, mask=mask)


@triton.jit
def _mass_adjoint_scan_kernel(
    boundary_m,
    grad_boundary_m,
    grad_final_m,
    chunk_lambda,
    grad_summary_m,
    grad_initial_m,
    lambda_mass,
    chunks: tl.constexpr,
):
    head_batch = tl.program_id(0)
    carry = tl.load(grad_final_m + head_batch).to(tl.float32)
    for reverse in tl.range(0, chunks):
        chunk = chunks - 1 - reverse
        panel = head_batch * chunks + chunk
        tl.store(grad_summary_m + panel, carry)
        tl.store(lambda_mass + panel, carry * tl.load(boundary_m + panel))
        carry = (
            tl.load(grad_boundary_m + panel).to(tl.float32)
            + tl.load(chunk_lambda + panel).to(tl.float32) * carry
        )
    tl.store(grad_initial_m + head_batch, carry)


@triton.jit
def _reduce_chunk_lambda_grad_kernel(
    lambda_partial,
    lambda_mass,
    grad_lambda,
    tiles: tl.constexpr,
    reduce_block: tl.constexpr,
):
    panel = tl.program_id(0)
    offsets = tl.arange(0, reduce_block)
    values = tl.load(
        lambda_partial + panel * tiles + offsets,
        mask=offsets < tiles,
        other=0.0,
    )
    tl.store(
        grad_lambda + panel,
        tl.sum(values, axis=0) + tl.load(lambda_mass + panel),
    )


@triton.jit
def _chunk_summary_vector_vjp_kernel(
    grad_summary_J,
    grad_summary_D,
    u,
    h_value,
    weights,
    grad_u,
    grad_h,
    grad_weight_partial,
    local_grad_u,
    local_grad_h,
    length: tl.constexpr,
    heads: tl.constexpr,
    rank: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    block_r: tl.constexpr,
    row_blocks: tl.constexpr,
    stride_ub: tl.constexpr,
    stride_ut: tl.constexpr,
    stride_uh: tl.constexpr,
    stride_ur: tl.constexpr,
    stride_hb: tl.constexpr,
    stride_ht: tl.constexpr,
    stride_hh: tl.constexpr,
    stride_hr: tl.constexpr,
    use_low_precision_mma: tl.constexpr,
    u_is_fp16: tl.constexpr,
    h_is_fp16: tl.constexpr,
    add_local_partials: tl.constexpr,
):
    row_block = tl.program_id(0)
    panel = tl.program_id(1)
    chunk = panel % chunks
    head_batch = panel // chunks
    head = head_batch % heads
    batch = head_batch // heads
    rows = row_block * block_r + tl.arange(0, block_r)
    local_t = tl.arange(0, chunk_size)
    tokens = chunk * chunk_size + local_t
    valid = tokens < length
    matrix_base = panel * rank * rank

    action_J_symmetric = tl.zeros((block_r, chunk_size), tl.float32)
    action_D = tl.zeros((block_r, chunk_size), tl.float32)
    action_Dt = tl.zeros((block_r, chunk_size), tl.float32)
    for col_block in tl.static_range(0, row_blocks):
        cols = col_block * block_r + tl.arange(0, block_r)
        grad_J = tl.load(
            grad_summary_J
            + matrix_base
            + rows[:, None] * rank
            + cols[None, :]
        ).to(tl.float32)
        grad_Jt_source = tl.load(
            grad_summary_J
            + matrix_base
            + cols[:, None] * rank
            + rows[None, :]
        ).to(tl.float32)
        grad_D = tl.load(
            grad_summary_D
            + matrix_base
            + rows[:, None] * rank
            + cols[None, :]
        ).to(tl.float32)
        grad_Dt_source = tl.load(
            grad_summary_D
            + matrix_base
            + cols[:, None] * rank
            + rows[None, :]
        ).to(tl.float32)
        source_u = tl.load(
            u
            + batch * stride_ub
            + tokens[None, :] * stride_ut
            + head * stride_uh
            + cols[:, None] * stride_ur,
            mask=valid[None, :],
            other=0.0,
        )
        source_h = tl.load(
            h_value
            + batch * stride_hb
            + tokens[None, :] * stride_ht
            + head * stride_hh
            + cols[:, None] * stride_hr,
            mask=valid[None, :],
            other=0.0,
        )
        grad_J_symmetric = grad_J + tl.trans(grad_Jt_source)
        grad_D_transposed = tl.trans(grad_Dt_source)
        if use_low_precision_mma:
            if u_is_fp16:
                grad_J_operand = grad_J_symmetric.to(
                    tl.float16, fp_downcast_rounding="rtne"
                )
                source_u_j = source_u.to(
                    tl.float16, fp_downcast_rounding="rtne"
                )
            else:
                grad_J_operand = grad_J_symmetric.to(
                    tl.bfloat16, fp_downcast_rounding="rtne"
                )
                source_u_j = source_u.to(
                    tl.bfloat16, fp_downcast_rounding="rtne"
                )
            if h_is_fp16:
                grad_D_operand = grad_D.to(
                    tl.float16, fp_downcast_rounding="rtne"
                )
                grad_Dt_operand = grad_D_transposed.to(
                    tl.float16, fp_downcast_rounding="rtne"
                )
                source_h_d = source_h.to(
                    tl.float16, fp_downcast_rounding="rtne"
                )
                source_u_d = source_u.to(
                    tl.float16, fp_downcast_rounding="rtne"
                )
            else:
                grad_D_operand = grad_D.to(
                    tl.bfloat16, fp_downcast_rounding="rtne"
                )
                grad_Dt_operand = grad_D_transposed.to(
                    tl.bfloat16, fp_downcast_rounding="rtne"
                )
                source_h_d = source_h.to(
                    tl.bfloat16, fp_downcast_rounding="rtne"
                )
                source_u_d = source_u.to(tl.bfloat16)
        else:
            grad_J_operand = grad_J_symmetric
            grad_D_operand = grad_D
            grad_Dt_operand = grad_D_transposed
            source_u_j = source_u.to(tl.float32)
            source_u_d = source_u_j
            source_h_d = source_h.to(tl.float32)
        action_J_symmetric += tl.dot(
            grad_J_operand,
            source_u_j,
            input_precision="ieee",
            out_dtype=tl.float32,
        )
        action_D += tl.dot(
            grad_D_operand,
            source_h_d,
            input_precision="ieee",
            out_dtype=tl.float32,
        )
        action_Dt += tl.dot(
            grad_Dt_operand,
            source_u_d,
            input_precision="ieee",
            out_dtype=tl.float32,
        )

    local_u = tl.load(
        u
        + batch * stride_ub
        + tokens[None, :] * stride_ut
        + head * stride_uh
        + rows[:, None] * stride_ur,
        mask=valid[None, :],
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(
        weights + panel * chunk_size + local_t, mask=valid, other=0.0
    ).to(tl.float32)
    output_u = (
        batch * stride_ub
        + tokens[None, :] * stride_ut
        + head * stride_uh
        + rows[:, None] * stride_ur
    )
    output_h = (
        batch * stride_hb
        + tokens[None, :] * stride_ht
        + head * stride_hh
        + rows[:, None] * stride_hr
    )
    result_u = (action_J_symmetric + action_D) * weight[None, :]
    result_h = action_Dt * weight[None, :]
    if add_local_partials:
        result_u += tl.load(
            local_grad_u + output_u, mask=valid[None, :], other=0.0
        ).to(tl.float32)
        result_h += tl.load(
            local_grad_h + output_h, mask=valid[None, :], other=0.0
        ).to(tl.float32)
    tl.store(grad_u + output_u, result_u, mask=valid[None, :])
    tl.store(grad_h + output_h, result_h, mask=valid[None, :])
    grad_weight = tl.sum(
        local_u * (0.5 * action_J_symmetric + action_D), axis=0
    )
    tl.store(
        grad_weight_partial
        + (panel * row_blocks + row_block) * chunk_size
        + local_t,
        tl.where(valid, grad_weight, 0.0),
    )


@triton.jit
def _chunk_summary_scalar_vjp_kernel(
    weights,
    chunk_lambda,
    grad_summary_m,
    grad_lambda,
    grad_weight_partial,
    grad_log_decay,
    local_grad_log_decay,
    length: tl.constexpr,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    row_blocks: tl.constexpr,
    stride_b: tl.constexpr,
    stride_t: tl.constexpr,
    stride_h: tl.constexpr,
    add_local_partial: tl.constexpr,
):
    panel = tl.program_id(0)
    chunk = panel % chunks
    head_batch = panel // chunks
    head = head_batch % heads
    batch = head_batch // heads
    local_t = tl.arange(0, chunk_size)
    tokens = chunk * chunk_size + local_t
    valid = tokens < length
    high = tl.zeros((chunk_size,), tl.float32)
    low = tl.zeros((chunk_size,), tl.float32)
    grad_lambda_value = tl.load(grad_lambda + panel).to(tl.float32)
    lambda_value = tl.load(chunk_lambda + panel).to(tl.float32)
    high, low = _twofold_product_add(
        high, low, grad_lambda_value, lambda_value
    )
    for source in tl.static_range(0, chunk_size):
        source_grad_weight = tl.load(grad_summary_m + panel).to(tl.float32)
        for row_block in tl.static_range(0, row_blocks):
            source_grad_weight += tl.load(
                grad_weight_partial
                + (panel * row_blocks + row_block) * chunk_size
                + source
            ).to(tl.float32)
        source_weight = tl.load(
            weights + panel * chunk_size + source
        ).to(tl.float32)
        include = source < local_t
        high, low = _twofold_product_add(
            high,
            low,
            tl.where(include, source_grad_weight, 0.0),
            source_weight,
        )
    output = (
        batch * stride_b + tokens * stride_t + head * stride_h
    )
    result = high + low
    if add_local_partial:
        result += tl.load(
            local_grad_log_decay + output, mask=valid, other=0.0
        ).to(tl.float32)
    tl.store(
        grad_log_decay + output,
        tl.where(valid, result, 0.0),
        mask=valid,
    )


def _triton_geometry_chunk_scan_forward(
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    *,
    initial_state: SolveDeltaState | None = None,
    chunk_size: int = 64,
    input_precision: Literal["ieee", "tf32"] = "ieee",
) -> tuple[SolveDeltaState, SolveDeltaState]:
    """Compute chunk-start and final geometry states without tokenwise matrices.

    ``u`` must already be normalized. The production pair is bounded FP16
    ``u`` with raw BF16 ``h``; equal FP32/BF16/FP16 dtypes remain oracle and
    provenance modes. Log-decay and all boundary states are FP32. Boundary
    ``S`` tensors are empty because this operation owns only ``m,J,D``.
    """
    if not torch.cuda.is_available() or u.device.type != "cuda":
        raise ValueError("Triton geometry scan requires CUDA tensors")
    if u.ndim != 4 or h.shape != u.shape:
        raise ValueError("u and h must have equal [B, T, H, r] shapes")
    if geometry_log_decay.shape != u.shape[:3]:
        raise ValueError("geometry_log_decay must have shape [B, T, H]")
    if u.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("Triton geometry scan supports FP32, BF16, or FP16 inputs")
    if h.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("Triton geometry scan supports FP32, BF16, or FP16 inputs")
    if h.dtype != u.dtype and (u.dtype, h.dtype) != (
        torch.float16,
        torch.bfloat16,
    ):
        raise TypeError("mixed geometry operands require FP16 u and BF16 h")
    if geometry_log_decay.dtype != torch.float32:
        raise TypeError("geometry_log_decay must be FP32")
    if chunk_size not in (16, 32, 64):
        raise ValueError("Triton geometry chunk_size must be one of 16, 32, 64")
    if input_precision not in ("ieee", "tf32"):
        raise ValueError("input_precision must be 'ieee' or 'tf32'")
    batch, length, heads, rank = u.shape
    if rank % 32:
        raise ValueError("the first Triton specialization requires r divisible by 32")
    chunks = triton.cdiv(length, chunk_size)
    if chunks == 0:
        raise ValueError("length must be positive")
    weights = torch.empty(batch, heads, chunks, chunk_size, device=u.device, dtype=torch.float32)
    chunk_lambda = torch.empty(batch, heads, chunks, device=u.device, dtype=torch.float32)
    chunk_mass = torch.empty_like(chunk_lambda)
    _chunk_weights_kernel[(batch * heads * chunks,)](
        geometry_log_decay,
        weights,
        chunk_lambda,
        chunk_mass,
        length=length,
        heads=heads,
        chunks=chunks,
        chunk_size=chunk_size,
        stride_b=geometry_log_decay.stride(0),
        stride_t=geometry_log_decay.stride(1),
        stride_h=geometry_log_decay.stride(2),
        write_mass=True,
    )

    # The summary kernel fills the public boundary storage first. The affine
    # scan then consumes each summary before replacing it with that chunk's
    # incoming state, so no second pair of chunk matrices is required.
    boundary_J = torch.empty(batch, heads, chunks, rank, rank, device=u.device, dtype=torch.float32)
    boundary_D = torch.empty_like(boundary_J)
    rank_blocks = triton.cdiv(rank, 64)
    _chunk_matrix_summary_kernel[(chunks, batch * heads, rank_blocks * rank_blocks)](
        u,
        h,
        weights,
        boundary_J,
        boundary_D,
        length=length,
        heads=heads,
        rank=rank,
        chunks=chunks,
        chunk_size=chunk_size,
        block_r=64,
        input_precision=input_precision,
        stride_ub=u.stride(0),
        stride_ut=u.stride(1),
        stride_uh=u.stride(2),
        stride_ur=u.stride(3),
        stride_hb=h.stride(0),
        stride_ht=h.stride(1),
        stride_hh=h.stride(2),
        stride_hr=h.stride(3),
        num_warps=4,
    )

    state_dtype = torch.float32
    if initial_state is None:
        initial_m = torch.zeros(batch, heads, device=u.device, dtype=state_dtype)
        initial_J = torch.zeros(batch, heads, rank, rank, device=u.device, dtype=state_dtype)
        initial_D = torch.zeros_like(initial_J)
    else:
        for name, tensor in zip(SolveDeltaState._fields, initial_state):
            if tensor.device != u.device:
                raise ValueError(f"initial_state.{name} must share the input device")
            if tensor.dtype != state_dtype:
                raise TypeError(f"initial_state.{name} must be FP32")
        initial_m = initial_state.m.contiguous()
        initial_J = initial_state.J.contiguous()
        initial_D = initial_state.D.contiguous()
        if (
            initial_m.shape != (batch, heads)
            or initial_J.shape != (batch, heads, rank, rank)
            or initial_D.shape != initial_J.shape
        ):
            raise ValueError("initial geometry state shapes do not match inputs")
        if not torch.equal(initial_J, initial_J.transpose(-1, -2)):
            raise ValueError("initial_state.J must be exactly symmetric")

    boundary_m = torch.empty(batch, heads, chunks, device=u.device, dtype=state_dtype)
    final_m = torch.empty(batch, heads, device=u.device, dtype=state_dtype)
    final_J = torch.empty(batch, heads, rank, rank, device=u.device, dtype=state_dtype)
    final_D = torch.empty_like(final_J)
    _mass_boundary_scan_kernel[(batch * heads,)](
        chunk_lambda,
        chunk_mass,
        initial_m,
        boundary_m,
        final_m,
        heads=heads,
        chunks=chunks,
    )
    _matrix_boundary_scan_kernel[(triton.cdiv(rank * rank, 256), batch * heads)](
        boundary_J,
        boundary_D,
        chunk_lambda,
        initial_J,
        initial_D,
        final_J,
        final_D,
        heads=heads,
        rank=rank,
        chunks=chunks,
        block=256,
        num_warps=4,
    )
    empty_boundary_S = torch.empty(0, device=u.device, dtype=state_dtype)
    empty_final_S = torch.empty(0, device=u.device, dtype=state_dtype)
    return (
        SolveDeltaState(boundary_m, boundary_J, boundary_D, empty_boundary_S),
        SolveDeltaState(final_m, final_J, final_D, empty_final_S),
    )


def _triton_geometry_chunk_scan_backward(
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    boundary_m: torch.Tensor,
    boundary_J: torch.Tensor,
    boundary_D: torch.Tensor,
    grad_boundary_m: torch.Tensor,
    grad_boundary_J: torch.Tensor,
    grad_boundary_D: torch.Tensor,
    grad_final_m: torch.Tensor,
    grad_final_J: torch.Tensor,
    grad_final_D: torch.Tensor,
    chunk_size: int,
    *,
    local_grad_u: torch.Tensor | None = None,
    local_grad_h: torch.Tensor | None = None,
    local_grad_log_decay: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Return affine-scan partials for a surrounding composed backward.

    Standalone and composed scan partials are FP32. When resident FP32 local
    partials are supplied, they are accumulated before the composed owner
    performs the only low-precision cast at its public autograd boundary.
    """
    batch, length, heads, rank = u.shape
    chunks = triton.cdiv(length, chunk_size)
    panels = batch * heads * chunks
    head_batches = batch * heads
    local_partials = (
        local_grad_u,
        local_grad_h,
        local_grad_log_decay,
    )
    add_local_partials = any(value is not None for value in local_partials)
    if add_local_partials:
        if not all(value is not None for value in local_partials):
            raise ValueError("all local geometry partials must be provided")
        expected = (u.shape, h.shape, geometry_log_decay.shape)
        for name, value, shape in zip(
            ("local_grad_u", "local_grad_h", "local_grad_log_decay"),
            local_partials,
            expected,
        ):
            if value.shape != shape or value.dtype != torch.float32:
                raise ValueError(f"{name} must be contiguous FP32 with shape {shape}")
            if value.device != u.device or not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous on the input device")

    weights = torch.empty(
        batch,
        heads,
        chunks,
        chunk_size,
        device=u.device,
        dtype=torch.float32,
    )
    chunk_lambda = torch.empty(
        batch, heads, chunks, device=u.device, dtype=torch.float32
    )
    _chunk_weights_kernel[(panels,)](
        geometry_log_decay,
        weights,
        chunk_lambda,
        chunk_lambda,
        length=length,
        heads=heads,
        chunks=chunks,
        chunk_size=chunk_size,
        stride_b=geometry_log_decay.stride(0),
        stride_t=geometry_log_decay.stride(1),
        stride_h=geometry_log_decay.stride(2),
        write_mass=False,
        num_warps=4,
    )

    grad_summary_J = torch.empty_like(boundary_J)
    grad_summary_D = torch.empty_like(boundary_D)
    grad_initial_J = torch.empty_like(grad_final_J)
    grad_initial_D = torch.empty_like(grad_final_D)
    matrix_tile = 256
    matrix_tiles = triton.cdiv(rank * rank, matrix_tile)
    lambda_partial = torch.empty(
        panels, matrix_tiles, device=u.device, dtype=torch.float32
    )
    _matrix_adjoint_scan_kernel[(matrix_tiles, head_batches)](
        boundary_J,
        boundary_D,
        grad_boundary_J,
        grad_boundary_D,
        grad_final_J,
        grad_final_D,
        chunk_lambda,
        grad_summary_J,
        grad_summary_D,
        grad_initial_J,
        grad_initial_D,
        lambda_partial,
        rank=rank,
        chunks=chunks,
        tile_size=matrix_tile,
        tiles=matrix_tiles,
        num_warps=8,
        num_stages=1,
    )

    grad_summary_m = torch.empty_like(boundary_m)
    grad_initial_m = torch.empty_like(grad_final_m)
    lambda_mass = torch.empty_like(chunk_lambda)
    _mass_adjoint_scan_kernel[(head_batches,)](
        boundary_m,
        grad_boundary_m,
        grad_final_m,
        chunk_lambda,
        grad_summary_m,
        grad_initial_m,
        lambda_mass,
        chunks=chunks,
        num_warps=1,
        num_stages=1,
    )
    grad_lambda = torch.empty_like(chunk_lambda)
    _reduce_chunk_lambda_grad_kernel[(panels,)](
        lambda_partial,
        lambda_mass,
        grad_lambda,
        tiles=matrix_tiles,
        reduce_block=triton.next_power_of_2(matrix_tiles),
        num_warps=4,
        num_stages=1,
    )

    # Mixed FP16/BF16 operands still share FP32 scan and frame partials. The
    # composed owners perform the only casts at their actual autograd leaves.
    activation_dtype = torch.float32
    grad_u = torch.empty(u.shape, device=u.device, dtype=activation_dtype)
    grad_h = torch.empty(h.shape, device=h.device, dtype=activation_dtype)
    local_u_pointer = local_grad_u if add_local_partials else grad_u
    local_h_pointer = local_grad_h if add_local_partials else grad_h
    row_blocks = rank // 64
    grad_weight_partial = torch.empty(
        panels,
        row_blocks,
        chunk_size,
        device=u.device,
        dtype=torch.float32,
    )
    _chunk_summary_vector_vjp_kernel[(row_blocks, panels)](
        grad_summary_J,
        grad_summary_D,
        u,
        h,
        weights,
        grad_u,
        grad_h,
        grad_weight_partial,
        local_u_pointer,
        local_h_pointer,
        length=length,
        heads=heads,
        rank=rank,
        chunks=chunks,
        chunk_size=chunk_size,
        block_r=64,
        row_blocks=row_blocks,
        stride_ub=u.stride(0),
        stride_ut=u.stride(1),
        stride_uh=u.stride(2),
        stride_ur=u.stride(3),
        stride_hb=h.stride(0),
        stride_ht=h.stride(1),
        stride_hh=h.stride(2),
        stride_hr=h.stride(3),
        use_low_precision_mma=u.dtype != torch.float32,
        u_is_fp16=u.dtype == torch.float16,
        h_is_fp16=h.dtype == torch.float16,
        add_local_partials=add_local_partials,
        num_warps=4,
        num_stages=1,
    )
    grad_log_decay = torch.empty(
        geometry_log_decay.shape,
        device=geometry_log_decay.device,
        dtype=torch.float32,
    )
    local_decay_pointer = (
        local_grad_log_decay if add_local_partials else grad_log_decay
    )
    _chunk_summary_scalar_vjp_kernel[(panels,)](
        weights,
        chunk_lambda,
        grad_summary_m,
        grad_lambda,
        grad_weight_partial,
        grad_log_decay,
        local_decay_pointer,
        length=length,
        heads=heads,
        chunks=chunks,
        chunk_size=chunk_size,
        row_blocks=row_blocks,
        stride_b=grad_log_decay.stride(0),
        stride_t=grad_log_decay.stride(1),
        stride_h=grad_log_decay.stride(2),
        add_local_partial=add_local_partials,
        num_warps=1,
        num_stages=1,
    )
    return (
        grad_u,
        grad_h,
        grad_log_decay,
        grad_initial_m,
        grad_initial_J,
        grad_initial_D,
    )


class _TritonGeometryChunkScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u,
        h,
        geometry_log_decay,
        initial_m,
        initial_J,
        initial_D,
        chunk_size,
        input_precision,
    ):
        needs_backward = any(ctx.needs_input_grad[:6])
        if needs_backward:
            # The scan kernels accept arbitrary strides. Canonicalize once for
            # both the forward kernels and their saved backward inputs instead
            # of staging the same projection view twice.
            u = u.contiguous()
            h = h.contiguous()
            geometry_log_decay = geometry_log_decay.contiguous()
        initial = SolveDeltaState(
            initial_m,
            initial_J,
            initial_D,
            torch.empty(0, device=u.device, dtype=torch.float32),
        )
        boundary, final = _triton_geometry_chunk_scan_forward(
            u,
            h,
            geometry_log_decay,
            initial_state=initial,
            chunk_size=chunk_size,
            input_precision=input_precision,
        )
        if needs_backward:
            ctx.save_for_backward(
                u,
                h,
                geometry_log_decay,
                boundary.m,
                boundary.J,
                boundary.D,
            )
        ctx.chunk_size = chunk_size
        return boundary.m, boundary.J, boundary.D, final.m, final.J, final.D

    @staticmethod
    def backward(ctx, *grad_outputs):
        u, h, geometry_log_decay, boundary_m, boundary_J, boundary_D = (
            ctx.saved_tensors
        )
        gradients = _triton_geometry_chunk_scan_backward(
            u,
            h,
            geometry_log_decay,
            boundary_m,
            boundary_J,
            boundary_D,
            *(gradient.contiguous() for gradient in grad_outputs),
            ctx.chunk_size,
        )
        # This standalone autograd boundary owns the public activation dtype.
        # A fused geometry/frame owner calls the lower-level adjoint directly
        # and therefore receives the unrounded FP32 partials above.
        grad_u, grad_h, *state_gradients = gradients
        return (
            grad_u.to(u.dtype),
            grad_h.to(h.dtype),
            *state_gradients,
            None,
            None,
        )


def triton_geometry_chunk_scan(
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    *,
    initial_state: SolveDeltaState | None = None,
    chunk_size: int = 64,
    input_precision: Literal["ieee", "tf32"] = "ieee",
) -> tuple[SolveDeltaState, SolveDeltaState]:
    """Autograd-capable mixed-precision chunk-boundary geometry scan."""
    if u.ndim != 4:
        raise ValueError("u must have shape [B,T,H,r]")
    batch, _, heads, rank = u.shape
    if initial_state is None:
        initial_m = torch.zeros(batch, heads, device=u.device, dtype=torch.float32)
        initial_J = torch.zeros(batch, heads, rank, rank, device=u.device, dtype=torch.float32)
        initial_D = torch.zeros_like(initial_J)
    else:
        for name, tensor in zip(SolveDeltaState._fields, initial_state):
            if tensor.device != u.device:
                raise ValueError(f"initial_state.{name} must share the input device")
            if tensor.dtype != torch.float32:
                raise TypeError(f"initial_state.{name} must be FP32")
        initial_m = initial_state.m
        initial_J = initial_state.J
        initial_D = initial_state.D
    outputs = _TritonGeometryChunkScan.apply(
        u,
        h,
        geometry_log_decay,
        initial_m,
        initial_J,
        initial_D,
        chunk_size,
        input_precision,
    )
    empty = torch.empty(0, device=u.device, dtype=torch.float32)
    return (
        SolveDeltaState(outputs[0], outputs[1], outputs[2], empty),
        SolveDeltaState(outputs[3], outputs[4], outputs[5], empty),
    )
