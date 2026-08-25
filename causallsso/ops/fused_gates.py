from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus


_BETA_BLOCK = 1024
_DECAY_TILE = 128
_REDUCE_BLOCK = 256


@triton.jit
def _paired_beta_forward_kernel(
    erase_raw,
    write_raw,
    erase,
    write,
    ERASE_SIZE: tl.constexpr,
    WRITE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    erase_mask = offsets < ERASE_SIZE
    write_mask = offsets < WRITE_SIZE
    erase_values = tl.load(erase_raw + offsets, mask=erase_mask, other=0.0)
    write_values = tl.load(write_raw + offsets, mask=write_mask, other=0.0)
    tl.store(erase + offsets, 2.0 * tl.sigmoid(erase_values.to(tl.float32)), mask=erase_mask)
    tl.store(write + offsets, 2.0 * tl.sigmoid(write_values.to(tl.float32)), mask=write_mask)


@triton.jit
def _paired_beta_backward_kernel(
    erase_raw,
    write_raw,
    grad_erase,
    grad_write,
    grad_erase_raw,
    grad_write_raw,
    ERASE_SIZE: tl.constexpr,
    WRITE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_ERASE: tl.constexpr,
    HAS_WRITE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if HAS_ERASE:
        erase_mask = offsets < ERASE_SIZE
        erase_values = tl.sigmoid(
            tl.load(erase_raw + offsets, mask=erase_mask, other=0.0).to(tl.float32)
        )
        grad_erase_values = tl.load(
            grad_erase + offsets, mask=erase_mask, other=0.0
        ).to(tl.float32)
        tl.store(
            grad_erase_raw + offsets,
            2.0 * grad_erase_values * erase_values * (1.0 - erase_values),
            mask=erase_mask,
        )
    if HAS_WRITE:
        write_mask = offsets < WRITE_SIZE
        write_values = tl.sigmoid(
            tl.load(write_raw + offsets, mask=write_mask, other=0.0).to(tl.float32)
        )
        grad_write_values = tl.load(
            grad_write + offsets, mask=write_mask, other=0.0
        ).to(tl.float32)
        tl.store(
            grad_write_raw + offsets,
            2.0 * grad_write_values * write_values * (1.0 - write_values),
            mask=write_mask,
        )


@triton.jit
def _decay_forward_kernel(
    raw,
    log_rate,
    bias,
    output,
    TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    feature = tl.program_id(1).to(tl.int64)
    tokens = tile * TILE + tl.arange(0, TILE).to(tl.int64)
    mask = tokens < TOKENS
    offsets = tokens * WIDTH + feature
    values = tl.load(raw + offsets, mask=mask, other=0.0).to(tl.float32)
    values += tl.load(bias + feature).to(tl.float32)
    scale = -exp(tl.load(log_rate + feature).to(tl.float32))
    tl.store(output + offsets, scale * softplus(values), mask=mask)


@triton.jit
def _decay_backward_kernel(
    raw,
    log_rate,
    bias,
    grad_output,
    grad_raw,
    partial_rate,
    partial_bias,
    TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    feature = tl.program_id(1).to(tl.int64)
    tokens = tile * TILE + tl.arange(0, TILE).to(tl.int64)
    mask = tokens < TOKENS
    offsets = tokens * WIDTH + feature
    values = tl.load(raw + offsets, mask=mask, other=0.0).to(tl.float32)
    values += tl.load(bias + feature).to(tl.float32)
    output_gradient = tl.load(
        grad_output + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    scale = -exp(tl.load(log_rate + feature).to(tl.float32))
    decay = scale * softplus(values)
    raw_gradient = output_gradient * scale * tl.sigmoid(values)
    tl.store(grad_raw + offsets, raw_gradient, mask=mask)
    tl.store(
        partial_rate + tile * WIDTH + feature,
        tl.sum(output_gradient * decay, axis=0),
    )
    tl.store(
        partial_bias + tile * WIDTH + feature,
        tl.sum(raw_gradient, axis=0),
    )


@triton.jit
def _decay_partial_reduce_kernel(
    partial_rate,
    partial_bias,
    grad_log_rate,
    grad_bias,
    TILES: tl.constexpr,
    TILES_PADDED: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    features = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tiles = tl.arange(0, TILES_PADDED)[:, None]
    feature_matrix = features[None, :]
    mask = (tiles < TILES) & (feature_matrix < WIDTH)
    offsets = tiles * WIDTH + feature_matrix
    rate = tl.load(partial_rate + offsets, mask=mask, other=0.0)
    bias = tl.load(partial_bias + offsets, mask=mask, other=0.0)
    tl.store(grad_log_rate + features, tl.sum(rate, axis=0), mask=features < WIDTH)
    tl.store(grad_bias + features, tl.sum(bias, axis=0), mask=features < WIDTH)


def _decay_forward(
    raw: torch.Tensor,
    log_rate: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    width = log_rate.numel()
    tokens = raw.numel() // width
    output = torch.empty_like(raw, dtype=torch.float32)
    _decay_forward_kernel[(triton.cdiv(tokens, _DECAY_TILE), width)](
        raw,
        log_rate,
        bias,
        output,
        TOKENS=tokens,
        WIDTH=width,
        TILE=_DECAY_TILE,
        num_warps=4,
        num_stages=1,
    )
    return output


def _decay_backward(
    raw: torch.Tensor,
    log_rate: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = log_rate.numel()
    tokens = raw.numel() // width
    tiles = triton.cdiv(tokens, _DECAY_TILE)
    grad_raw = torch.empty_like(raw)
    partial_rate = torch.empty(
        tiles, width, device=raw.device, dtype=torch.float32
    )
    partial_bias = torch.empty_like(partial_rate)
    _decay_backward_kernel[(tiles, width)](
        raw,
        log_rate,
        bias,
        grad_output.contiguous(),
        grad_raw,
        partial_rate,
        partial_bias,
        TOKENS=tokens,
        WIDTH=width,
        TILE=_DECAY_TILE,
        num_warps=4,
        num_stages=1,
    )
    grad_log_rate = torch.empty_like(log_rate)
    grad_bias = torch.empty_like(bias)
    _decay_partial_reduce_kernel[(triton.cdiv(width, _REDUCE_BLOCK),)](
        partial_rate,
        partial_bias,
        grad_log_rate,
        grad_bias,
        TILES=tiles,
        TILES_PADDED=triton.next_power_of_2(tiles),
        WIDTH=width,
        BLOCK=_REDUCE_BLOCK,
        num_warps=8,
        num_stages=1,
    )
    return grad_raw, grad_log_rate, grad_bias


class _NativeSolveDeltaGates(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        erase_raw: torch.Tensor,
        write_raw: torch.Tensor,
        geometry_raw: torch.Tensor,
        associative_raw: torch.Tensor,
        geometry_log_rate: torch.Tensor,
        associative_log_rate: torch.Tensor,
        geometry_bias: torch.Tensor,
        associative_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        erase = torch.empty_like(erase_raw, dtype=torch.bfloat16)
        write = torch.empty_like(write_raw, dtype=torch.bfloat16)
        maximum = max(erase_raw.numel(), write_raw.numel())
        _paired_beta_forward_kernel[(triton.cdiv(maximum, _BETA_BLOCK),)](
            erase_raw,
            write_raw,
            erase,
            write,
            ERASE_SIZE=erase_raw.numel(),
            WRITE_SIZE=write_raw.numel(),
            BLOCK=_BETA_BLOCK,
            num_warps=8,
            num_stages=1,
        )
        geometry_decay = _decay_forward(
            geometry_raw, geometry_log_rate, geometry_bias
        )
        associative_decay = _decay_forward(
            associative_raw, associative_log_rate, associative_bias
        )
        ctx.save_for_backward(
            erase_raw,
            write_raw,
            geometry_raw,
            associative_raw,
            geometry_log_rate,
            associative_log_rate,
            geometry_bias,
            associative_bias,
        )
        ctx.set_materialize_grads(False)
        return erase, write, geometry_decay, associative_decay

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        grad_erase: torch.Tensor | None,
        grad_write: torch.Tensor | None,
        grad_geometry_decay: torch.Tensor | None,
        grad_associative_decay: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        (
            erase_raw,
            write_raw,
            geometry_raw,
            associative_raw,
            geometry_log_rate,
            associative_log_rate,
            geometry_bias,
            associative_bias,
        ) = ctx.saved_tensors
        grad_erase_raw = (
            torch.empty_like(erase_raw) if grad_erase is not None else None
        )
        grad_write_raw = (
            torch.empty_like(write_raw) if grad_write is not None else None
        )
        if grad_erase is not None or grad_write is not None:
            maximum = max(erase_raw.numel(), write_raw.numel())
            _paired_beta_backward_kernel[(triton.cdiv(maximum, _BETA_BLOCK),)](
                erase_raw,
                write_raw,
                grad_erase.contiguous() if grad_erase is not None else erase_raw,
                grad_write.contiguous() if grad_write is not None else write_raw,
                grad_erase_raw if grad_erase_raw is not None else erase_raw,
                grad_write_raw if grad_write_raw is not None else write_raw,
                ERASE_SIZE=erase_raw.numel(),
                WRITE_SIZE=write_raw.numel(),
                BLOCK=_BETA_BLOCK,
                HAS_ERASE=grad_erase is not None,
                HAS_WRITE=grad_write is not None,
                num_warps=8,
                num_stages=1,
            )
        geometry_gradients = (
            _decay_backward(
                geometry_raw,
                geometry_log_rate,
                geometry_bias,
                grad_geometry_decay,
            )
            if grad_geometry_decay is not None
            else (None, None, None)
        )
        associative_gradients = (
            _decay_backward(
                associative_raw,
                associative_log_rate,
                associative_bias,
                grad_associative_decay,
            )
            if grad_associative_decay is not None
            else (None, None, None)
        )
        return (
            grad_erase_raw,
            grad_write_raw,
            geometry_gradients[0],
            associative_gradients[0],
            geometry_gradients[1],
            associative_gradients[1],
            geometry_gradients[2],
            associative_gradients[2],
        )


def fused_native_solvedelta_gates(
    erase_raw: torch.Tensor,
    write_raw: torch.Tensor,
    geometry_raw: torch.Tensor,
    associative_raw: torch.Tensor,
    geometry_log_rate: torch.Tensor,
    associative_log_rate: torch.Tensor,
    geometry_bias: torch.Tensor,
    associative_bias: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return _NativeSolveDeltaGates.apply(
        erase_raw.contiguous(),
        write_raw.contiguous(),
        geometry_raw.contiguous(),
        associative_raw.contiguous(),
        geometry_log_rate.contiguous(),
        associative_log_rate.contiguous(),
        geometry_bias.contiguous(),
        associative_bias.contiguous(),
    )


__all__ = ["fused_native_solvedelta_gates"]
