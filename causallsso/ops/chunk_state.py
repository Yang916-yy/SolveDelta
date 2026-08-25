from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl


_CHUNK = 32
_RANK = 128


class ChunkStateForward(NamedTuple):
    output: torch.Tensor
    boundaries: torch.Tensor
    residual: torch.Tensor
    final_state: torch.Tensor


class ChunkStateBackward(NamedTuple):
    grad_Y: torch.Tensor
    grad_U_z: torch.Tensor
    grad_D_tail: torch.Tensor
    grad_Q_gamma: torch.Tensor
    grad_A_qd: torch.Tensor
    grad_G_last: torch.Tensor
    grad_initial_state: torch.Tensor


@triton.jit(do_not_specialize=["T"])
def _decay_cumsum_forward_kernel(
    log_decay,
    inclusive,
    T,
    H: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    head = tl.program_id(2).to(tl.int64)
    rows = chunk * C + tl.arange(0, C)
    coordinates = tl.arange(0, K)
    mask = (rows[:, None] < T) & (coordinates[None, :] < K)
    base = (batch * T * H + head) * K
    offsets = base + rows[:, None] * H * K + coordinates[None, :]
    values = tl.load(log_decay + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(inclusive + offsets, tl.cumsum(values, axis=0), mask=mask)


@triton.jit(do_not_specialize=["T"])
def _decay_cumsum_backward_kernel(
    grad_inclusive,
    grad_log_decay,
    T,
    H: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    head = tl.program_id(2).to(tl.int64)
    rows = chunk * C + tl.arange(0, C)
    coordinates = tl.arange(0, K)
    mask = (rows[:, None] < T) & (coordinates[None, :] < K)
    base = (batch * T * H + head) * K
    offsets = base + rows[:, None] * H * K + coordinates[None, :]
    values = tl.load(grad_inclusive + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(
        grad_log_decay + offsets,
        tl.cumsum(values, axis=0, reverse=True),
        mask=mask,
    )


@triton.jit(do_not_specialize=["T"])
def _state_forward_kernel(
    Y,
    U_z,
    D_tail,
    G_last,
    initial_state,
    boundaries,
    residual,
    final_state,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    N,
    C: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
):
    value_block = tl.program_id(0)
    head_batch = tl.program_id(1).to(tl.int64)
    batch = head_batch // H
    head = head_batch % H
    coordinates = tl.arange(0, K)
    values = value_block * BV + tl.arange(0, BV)
    state_mask = (coordinates[:, None] < K) & (values[None, :] < V)
    state = tl.load(
        initial_state
        + head_batch * K * V
        + coordinates[:, None] * V
        + values[None, :],
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)

    for chunk in range(0, N):
        panel = head_batch * N + chunk
        tl.store(
            boundaries
            + panel * K * V
            + coordinates[:, None] * V
            + values[None, :],
            state,
            mask=state_mask,
        )
        rows = chunk * C + tl.arange(0, C)
        token_mask = rows < T
        vector_mask = token_mask[:, None] & (coordinates[None, :] < K)
        value_mask = token_mask[:, None] & (values[None, :] < V)
        vector_offsets = (
            (batch * T * H + head) * K
            + rows[:, None] * H * K
            + coordinates[None, :]
        )
        value_offsets = (
            (batch * T * H + head) * V
            + rows[:, None] * H * V
            + values[None, :]
        )
        local_Y = tl.load(Y + vector_offsets, mask=vector_mask, other=0.0)
        local_U = tl.load(U_z + value_offsets, mask=value_mask, other=0.0).to(
            tl.float32
        )
        local_residual = local_U - tl.dot(local_Y, state.to(tl.bfloat16))
        tl.store(
            residual + value_offsets,
            local_residual.to(tl.bfloat16),
            mask=value_mask,
        )
        local_D = tl.load(
            D_tail + vector_offsets, mask=vector_mask, other=0.0
        )
        update = tl.dot(
            tl.trans(local_D.to(tl.bfloat16)),
            local_residual.to(tl.bfloat16),
        )
        gate = tl.load(
            G_last + panel * K + coordinates,
            mask=coordinates < K,
            other=float("-inf"),
        ).to(tl.float32)
        state = tl.exp(gate)[:, None] * state + update

    tl.store(
        final_state
        + head_batch * K * V
        + coordinates[:, None] * V
        + values[None, :],
        state,
        mask=state_mask,
    )


@triton.jit(do_not_specialize=["T"])
def _output_forward_kernel(
    Q_gamma,
    A_qd,
    boundaries,
    residual,
    output,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    N,
    C: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
):
    value_block = tl.program_id(0)
    chunk = tl.program_id(1).to(tl.int64)
    head_batch = tl.program_id(2).to(tl.int64)
    batch = head_batch // H
    head = head_batch % H
    panel = head_batch * N + chunk
    rows = chunk * C + tl.arange(0, C)
    columns = tl.arange(0, C)
    coordinates = tl.arange(0, K)
    values = value_block * BV + tl.arange(0, BV)
    token_mask = rows < T
    vector_mask = token_mask[:, None] & (coordinates[None, :] < K)
    value_mask = token_mask[:, None] & (values[None, :] < V)
    vector_offsets = (
        (batch * T * H + head) * K
        + rows[:, None] * H * K
        + coordinates[None, :]
    )
    value_offsets = (
        (batch * T * H + head) * V
        + rows[:, None] * H * V
        + values[None, :]
    )
    query = tl.load(Q_gamma + vector_offsets, mask=vector_mask, other=0.0)
    state = tl.load(
        boundaries
        + panel * K * V
        + coordinates[:, None] * V
        + values[None, :],
        mask=(coordinates[:, None] < K) & (values[None, :] < V),
        other=0.0,
    ).to(tl.float32)
    result = tl.dot(query.to(tl.bfloat16), state.to(tl.bfloat16))
    interaction = tl.load(
        A_qd
        + panel * C * C
        + tl.arange(0, C)[:, None] * C
        + columns[None, :]
    ).to(tl.bfloat16)
    local_residual = tl.load(
        residual + value_offsets, mask=value_mask, other=0.0
    ).to(tl.bfloat16)
    result += tl.dot(interaction, local_residual)
    tl.store(
        output + value_offsets,
        result.to(tl.bfloat16),
        mask=value_mask,
    )


@triton.jit(do_not_specialize=["T"])
def _state_backward_kernel(
    Y,
    D_tail,
    Q_gamma,
    A_qd,
    G_last,
    grad_output,
    grad_final_state,
    grad_state_next,
    grad_residual,
    grad_initial_state,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    N,
    C: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
):
    value_block = tl.program_id(0)
    head_batch = tl.program_id(1).to(tl.int64)
    batch = head_batch // H
    head = head_batch % H
    coordinates = tl.arange(0, K)
    values = value_block * BV + tl.arange(0, BV)
    state_mask = (coordinates[:, None] < K) & (values[None, :] < V)
    carry = tl.load(
        grad_final_state
        + head_batch * K * V
        + coordinates[:, None] * V
        + values[None, :],
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)

    for reverse in range(0, N):
        chunk = N - 1 - reverse
        panel = head_batch * N + chunk
        tl.store(
            grad_state_next
            + panel * K * V
            + coordinates[:, None] * V
            + values[None, :],
            carry,
            mask=state_mask,
        )
        rows = chunk * C + tl.arange(0, C)
        columns = tl.arange(0, C)
        token_mask = rows < T
        vector_mask = token_mask[:, None] & (coordinates[None, :] < K)
        value_mask = token_mask[:, None] & (values[None, :] < V)
        vector_offsets = (
            (batch * T * H + head) * K
            + rows[:, None] * H * K
            + coordinates[None, :]
        )
        value_offsets = (
            (batch * T * H + head) * V
            + rows[:, None] * H * V
            + values[None, :]
        )
        local_grad_output = tl.load(
            grad_output + value_offsets, mask=value_mask, other=0.0
        ).to(tl.float32)
        local_D = tl.load(
            D_tail + vector_offsets, mask=vector_mask, other=0.0
        )
        local_grad_residual = tl.dot(
            local_D.to(tl.bfloat16), carry.to(tl.bfloat16)
        )
        interaction = tl.load(
            A_qd
            + panel * C * C
            + tl.arange(0, C)[:, None] * C
            + columns[None, :]
        ).to(tl.bfloat16)
        local_grad_residual += tl.dot(
            tl.trans(interaction), local_grad_output.to(tl.bfloat16)
        )
        tl.store(
            grad_residual + value_offsets,
            local_grad_residual,
            mask=value_mask,
        )
        local_Q = tl.load(
            Q_gamma + vector_offsets, mask=vector_mask, other=0.0
        )
        local_Y = tl.load(Y + vector_offsets, mask=vector_mask, other=0.0)
        gate = tl.load(
            G_last + panel * K + coordinates,
            mask=coordinates < K,
            other=float("-inf"),
        ).to(tl.float32)
        carry = (
            tl.dot(
                tl.trans(local_Q.to(tl.bfloat16)),
                local_grad_output.to(tl.bfloat16),
            )
            + tl.exp(gate)[:, None] * carry
            - tl.dot(
                tl.trans(local_Y), local_grad_residual.to(tl.bfloat16)
            )
        )

    tl.store(
        grad_initial_state
        + head_batch * K * V
        + coordinates[:, None] * V
        + values[None, :],
        carry,
        mask=state_mask,
    )


@triton.jit(do_not_specialize=["T"])
def _factor_cr_backward_kernel(
    boundaries,
    residual,
    grad_output,
    grad_state_next,
    grad_residual,
    grad_Y,
    grad_Q_gamma,
    grad_D_tail,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    rank_block = tl.program_id(1)
    head_batch = panel // N
    chunk = panel % N
    batch = head_batch // H
    head = head_batch % H
    rows = chunk * C + tl.arange(0, C)
    coordinates = rank_block * BR + tl.arange(0, BR)
    token_mask = rows < T
    factor_mask = token_mask[:, None] & (coordinates[None, :] < K)
    grad_y = tl.zeros((C, BR), tl.float32)
    grad_q = tl.zeros((C, BR), tl.float32)
    grad_d = tl.zeros((C, BR), tl.float32)
    for value_start in tl.static_range(0, V, BV):
        values = value_start + tl.arange(0, BV)
        value_mask = token_mask[:, None] & (values[None, :] < V)
        state_mask = (coordinates[:, None] < K) & (values[None, :] < V)
        value_offsets = (
            (batch * T * H + head) * V
            + rows[:, None] * H * V
            + values[None, :]
        )
        state_offsets = (
            panel * K * V
            + coordinates[:, None] * V
            + values[None, :]
        )
        local_state = tl.load(
            boundaries + state_offsets, mask=state_mask, other=0.0
        ).to(tl.bfloat16)
        local_grad_state = tl.load(
            grad_state_next + state_offsets, mask=state_mask, other=0.0
        ).to(tl.bfloat16)
        local_grad_output = tl.load(
            grad_output + value_offsets, mask=value_mask, other=0.0
        ).to(tl.bfloat16)
        local_grad_residual = tl.load(
            grad_residual + value_offsets, mask=value_mask, other=0.0
        ).to(tl.bfloat16)
        local_residual = tl.load(
            residual + value_offsets, mask=value_mask, other=0.0
        ).to(tl.bfloat16)
        grad_q += tl.dot(local_grad_output, tl.trans(local_state))
        grad_y -= tl.dot(local_grad_residual, tl.trans(local_state))
        grad_d += tl.dot(local_residual, tl.trans(local_grad_state))
    vector_offsets = (
        (batch * T * H + head) * K
        + rows[:, None] * H * K
        + coordinates[None, :]
    )
    tl.store(grad_Y + vector_offsets, grad_y, mask=factor_mask)
    tl.store(grad_Q_gamma + vector_offsets, grad_q, mask=factor_mask)
    tl.store(grad_D_tail + vector_offsets, grad_d, mask=factor_mask)


@triton.jit(do_not_specialize=["T"])
def _factor_a_backward_kernel(
    residual,
    grad_output,
    grad_A_qd,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    BV: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    head_batch = panel // N
    chunk = panel % N
    batch = head_batch // H
    head = head_batch % H
    rows = chunk * C + tl.arange(0, C)
    token_mask = rows < T
    gradient = tl.zeros((C, C), tl.float32)
    for value_start in tl.static_range(0, V, BV):
        values = value_start + tl.arange(0, BV)
        mask = token_mask[:, None] & (values[None, :] < V)
        offsets = (
            (batch * T * H + head) * V
            + rows[:, None] * H * V
            + values[None, :]
        )
        local_grad = tl.load(
            grad_output + offsets, mask=mask, other=0.0
        ).to(tl.bfloat16)
        local_residual = tl.load(
            residual + offsets, mask=mask, other=0.0
        ).to(tl.bfloat16)
        gradient += tl.dot(local_grad, tl.trans(local_residual))
    causal = (
        tl.arange(0, C)[:, None] >= tl.arange(0, C)[None, :]
    ) & token_mask[:, None] & token_mask[None, :]
    tl.store(
        grad_A_qd
        + panel * C * C
        + tl.arange(0, C)[:, None] * C
        + tl.arange(0, C)[None, :],
        tl.where(causal, gradient, 0.0),
    )


@triton.jit
def _factor_gate_backward_kernel(
    boundaries,
    grad_state_next,
    G_last,
    grad_G_last,
    V: tl.constexpr,
    K: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    rank_block = tl.program_id(1)
    coordinates = rank_block * BR + tl.arange(0, BR)
    gate = tl.load(
        G_last + panel * K + coordinates,
        mask=coordinates < K,
        other=float("-inf"),
    ).to(tl.float32)
    gradient = tl.zeros((BR,), tl.float32)
    for value_start in tl.static_range(0, V, BV):
        values = value_start + tl.arange(0, BV)
        mask = (coordinates[:, None] < K) & (values[None, :] < V)
        offsets = panel * K * V + coordinates[:, None] * V + values[None, :]
        state = tl.load(boundaries + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        grad_state = tl.load(
            grad_state_next + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        gradient += tl.sum(grad_state * (tl.exp(gate)[:, None] * state), axis=1)
    tl.store(
        grad_G_last + panel * K + coordinates,
        gradient,
        mask=coordinates < K,
    )


def decay_cumsum_forward(log_decay: torch.Tensor) -> torch.Tensor:
    if log_decay.ndim != 4 or log_decay.shape[-1] != _RANK:
        raise ValueError("log_decay must have shape [B,T,H,128]")
    if log_decay.dtype != torch.float32 or log_decay.device.type != "cuda":
        raise TypeError("log_decay must be contiguous FP32 CUDA")
    source = log_decay.contiguous()
    batch, length, heads, rank = source.shape
    output = torch.empty_like(source)
    _decay_cumsum_forward_kernel[(triton.cdiv(length, _CHUNK), batch, heads)](
        source,
        output,
        T=length,
        H=heads,
        C=_CHUNK,
        K=rank,
        num_warps=4,
        num_stages=2,
    )
    return output


def decay_cumsum_backward(grad_inclusive: torch.Tensor) -> torch.Tensor:
    if grad_inclusive.ndim != 4 or grad_inclusive.shape[-1] != _RANK:
        raise ValueError("grad_inclusive must have shape [B,T,H,128]")
    source = grad_inclusive.contiguous()
    batch, length, heads, rank = source.shape
    output = torch.empty_like(source, dtype=torch.float32)
    _decay_cumsum_backward_kernel[(triton.cdiv(length, _CHUNK), batch, heads)](
        source,
        output,
        T=length,
        H=heads,
        C=_CHUNK,
        K=rank,
        num_warps=4,
        num_stages=2,
    )
    return output


def chunk_state_forward(
    Y: torch.Tensor,
    U_z: torch.Tensor,
    D_tail: torch.Tensor,
    Q_gamma: torch.Tensor,
    A_qd: torch.Tensor,
    G_last: torch.Tensor,
    initial_state: torch.Tensor,
) -> ChunkStateForward:
    batch, length, heads, rank = Y.shape
    value_dim = U_z.shape[-1]
    chunks = triton.cdiv(length, _CHUNK)
    if rank != _RANK or value_dim < 1 or value_dim > _RANK:
        raise ValueError("native state path requires r=128 and 1 <= d_v <= 128")
    boundaries = torch.empty(
        batch,
        heads,
        chunks,
        rank,
        value_dim,
        device=Y.device,
        dtype=torch.float32,
    )
    residual = torch.empty_like(U_z)
    final_state = torch.empty_like(initial_state, dtype=torch.float32)
    output = torch.empty_like(U_z)
    block_v = 32
    _state_forward_kernel[(triton.cdiv(value_dim, block_v), batch * heads)](
        Y,
        U_z,
        D_tail,
        G_last,
        initial_state,
        boundaries,
        residual,
        final_state,
        T=length,
        H=heads,
        V=value_dim,
        N=chunks,
        C=_CHUNK,
        K=rank,
        BV=block_v,
        num_warps=8,
        num_stages=3,
    )
    _output_forward_kernel[
        (triton.cdiv(value_dim, block_v), chunks, batch * heads)
    ](
        Q_gamma,
        A_qd,
        boundaries,
        residual,
        output,
        T=length,
        H=heads,
        V=value_dim,
        N=chunks,
        C=_CHUNK,
        K=rank,
        BV=block_v,
        num_warps=8,
        num_stages=3,
    )
    return ChunkStateForward(output, boundaries, residual, final_state)


def chunk_state_backward(
    Y: torch.Tensor,
    D_tail: torch.Tensor,
    Q_gamma: torch.Tensor,
    A_qd: torch.Tensor,
    G_last: torch.Tensor,
    boundaries: torch.Tensor,
    residual: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final_state: torch.Tensor,
) -> ChunkStateBackward:
    batch, length, heads, rank = Y.shape
    value_dim = residual.shape[-1]
    chunks = triton.cdiv(length, _CHUNK)
    grad_state_next = torch.empty_like(boundaries)
    grad_residual = torch.empty_like(residual, dtype=torch.float32)
    grad_initial = torch.empty_like(grad_final_state, dtype=torch.float32)
    block_v = 16
    _state_backward_kernel[(triton.cdiv(value_dim, block_v), batch * heads)](
        Y,
        D_tail,
        Q_gamma,
        A_qd,
        G_last,
        grad_output,
        grad_final_state,
        grad_state_next,
        grad_residual,
        grad_initial,
        T=length,
        H=heads,
        V=value_dim,
        N=chunks,
        C=_CHUNK,
        K=rank,
        BV=block_v,
        num_warps=4,
        num_stages=3,
    )
    grad_Y = torch.empty_like(Y, dtype=torch.float32)
    grad_Q = torch.empty_like(Y, dtype=torch.float32)
    grad_D = torch.empty_like(Y, dtype=torch.float32)
    panels = batch * heads * chunks
    block_r = 16
    _factor_cr_backward_kernel[(panels, triton.cdiv(rank, block_r))](
        boundaries,
        residual,
        grad_output,
        grad_state_next,
        grad_residual,
        grad_Y,
        grad_Q,
        grad_D,
        T=length,
        H=heads,
        V=value_dim,
        N=chunks,
        C=_CHUNK,
        K=rank,
        BR=block_r,
        BV=32,
        num_warps=4,
        num_stages=3,
    )
    grad_A = torch.empty_like(A_qd, dtype=torch.float32)
    _factor_a_backward_kernel[(panels,)](
        residual,
        grad_output,
        grad_A,
        T=length,
        H=heads,
        V=value_dim,
        N=chunks,
        C=_CHUNK,
        BV=32,
        num_warps=8,
        num_stages=3,
    )
    grad_G_last = torch.empty_like(G_last, dtype=torch.float32)
    _factor_gate_backward_kernel[(panels, triton.cdiv(rank, block_r))](
        boundaries,
        grad_state_next,
        G_last,
        grad_G_last,
        V=value_dim,
        K=rank,
        BR=block_r,
        BV=32,
        num_warps=4,
    )
    return ChunkStateBackward(
        grad_Y,
        grad_residual,
        grad_D,
        grad_Q,
        grad_A,
        grad_G_last,
        grad_initial,
    )


__all__ = [
    "ChunkStateBackward",
    "ChunkStateForward",
    "chunk_state_backward",
    "chunk_state_forward",
    "decay_cumsum_backward",
    "decay_cumsum_forward",
]
