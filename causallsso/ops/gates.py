# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Adapted from flash-linear-attention's MIT-licensed GDN gate kernels. The
# SolveDelta specialization reads strided projection views directly and
# reduces log-rate and bias cotangents from the same FP32 register value.

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
from torch.autograd.function import once_differentiable

_REDUCE_BLOCK = 256


@triton.jit
def _precise_softplus(x):
    return tl.where(
        x > 20.0,
        x,
        tl.where(x < -20.0, tldevice.exp(x), tldevice.log1p(tldevice.exp(x))),
    )


@triton.jit(do_not_specialize=["T"])
def _decay_gate_fwd_kernel(
    raw,
    log_rate,
    bias,
    output,
    T,
    H: tl.constexpr,
    S_G: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    tile_t = tl.program_id(0).to(tl.int64)
    tile_d = tl.program_id(1).to(tl.int64)
    rows = tile_t * BT + tl.arange(0, BT).to(tl.int64)
    features = tile_d * BD + tl.arange(0, BD).to(tl.int64)
    mask = (rows[:, None] < T) & (features[None, :] < H)
    value = tl.load(
        raw + rows[:, None] * S_G + features[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    value += tl.load(
        bias + features[None, :],
        mask=features[None, :] < H,
        other=0.0,
    ).to(tl.float32)
    scale = -tldevice.exp(
        tl.load(
            log_rate + features[None, :],
            mask=features[None, :] < H,
            other=0.0,
        ).to(tl.float32)
    )
    tl.store(
        output + rows[:, None] * H + features[None, :],
        scale * _precise_softplus(value),
        mask=mask,
    )


@triton.heuristics({"HAS_BIAS": lambda args: args["bias"] is not None})
@triton.jit(do_not_specialize=["T"])
def _decay_gate_bwd_kernel(
    raw,
    log_rate,
    bias,
    grad_output,
    grad_raw,
    partial_rate,
    partial_bias,
    T,
    H: tl.constexpr,
    S_G: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    tile_t = tl.program_id(0).to(tl.int64)
    tile_d = tl.program_id(1).to(tl.int64)
    rows = tile_t * BT + tl.arange(0, BT).to(tl.int64)
    features = tile_d * BD + tl.arange(0, BD).to(tl.int64)
    mask = (rows[:, None] < T) & (features[None, :] < H)
    value = tl.load(
        raw + rows[:, None] * S_G + features[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_BIAS:
        value += tl.load(
            bias + features[None, :],
            mask=features[None, :] < H,
            other=0.0,
        ).to(tl.float32)
    output_gradient = tl.load(
        grad_output + rows[:, None] * H + features[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = -tldevice.exp(
        tl.load(
            log_rate + features[None, :],
            mask=features[None, :] < H,
            other=0.0,
        ).to(tl.float32)
    )
    decay = scale * _precise_softplus(value)
    sigmoid = 1.0 / (1.0 + tldevice.exp(-value))
    raw_gradient = output_gradient * scale * sigmoid
    tl.store(
        grad_raw + rows[:, None] * H + features[None, :],
        raw_gradient,
        mask=mask,
    )
    partial_offset = tile_t * H + features
    tl.store(
        partial_rate + partial_offset,
        tl.sum(output_gradient * decay, axis=0),
        mask=features < H,
    )
    tl.store(
        partial_bias + partial_offset,
        tl.sum(raw_gradient, axis=0),
        mask=features < H,
    )


@triton.jit
def _decay_gate_reduce_kernel(
    partial_rate,
    partial_bias,
    grad_log_rate,
    grad_bias,
    TILES: tl.constexpr,
    TILES_PADDED: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    features = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tiles = tl.arange(0, TILES_PADDED)[:, None]
    feature_matrix = features[None, :]
    mask = (tiles < TILES) & (feature_matrix < H)
    offsets = tiles * H + feature_matrix
    rate = tl.load(partial_rate + offsets, mask=mask, other=0.0)
    bias = tl.load(partial_bias + offsets, mask=mask, other=0.0)
    tl.store(
        grad_log_rate + features,
        tl.sum(rate, axis=0),
        mask=features < H,
    )
    tl.store(grad_bias + features, tl.sum(bias, axis=0), mask=features < H)


def _validate_row_major_view(raw: torch.Tensor, width: int) -> int:
    if raw.ndim < 2 or raw.shape[-1] != width or raw.stride(-1) != 1:
        raise ValueError("raw gate input must have unit-stride trailing feature width")
    row_stride = raw.stride(-2)
    expected = row_stride * raw.shape[-2]
    for axis in range(raw.ndim - 3, -1, -1):
        if raw.stride(axis) != expected:
            raise ValueError("raw gate leading dimensions must form a regular row view")
        expected *= raw.shape[axis]
    return row_stride


def _tile_shape(width: int) -> tuple[int, int, int]:
    if width <= 16:
        return 128, 16, 4
    return 32, 32, 4


class _FusedDecayGate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        raw: torch.Tensor,
        log_rate: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        width = log_rate.numel()
        if bias.numel() != width:
            raise ValueError("log_rate and bias must have the raw trailing width")
        if raw.device.type != "cuda" or any(
            tensor.device != raw.device for tensor in (log_rate, bias)
        ):
            raise ValueError("decay gate tensors must share one CUDA device")
        row_stride = _validate_row_major_view(raw, width)
        rows = raw.numel() // width
        output = torch.empty(raw.shape, dtype=torch.float32, device=raw.device)
        block_t, block_d, num_warps = _tile_shape(width)
        _decay_gate_fwd_kernel[
            (triton.cdiv(rows, block_t), triton.cdiv(width, block_d))
        ](
            raw,
            log_rate,
            bias,
            output,
            T=rows,
            H=width,
            S_G=row_stride,
            BT=block_t,
            BD=block_d,
            num_warps=num_warps,
            num_stages=1,
        )
        ctx.save_for_backward(raw, log_rate, bias)
        ctx.row_stride = row_stride
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor | None):
        if grad_output is None:
            return None, None, None
        raw, log_rate, bias = ctx.saved_tensors
        width = log_rate.numel()
        rows = raw.numel() // width
        block_t, block_d, num_warps = _tile_shape(width)
        tiles = triton.cdiv(rows, block_t)
        grad_raw = torch.empty(raw.shape, dtype=raw.dtype, device=raw.device)
        partial = torch.empty(2, tiles, width, dtype=torch.float32, device=raw.device)
        grad_output = grad_output.contiguous()
        _decay_gate_bwd_kernel[(tiles, triton.cdiv(width, block_d))](
            raw,
            log_rate,
            bias,
            grad_output,
            grad_raw,
            partial[0],
            partial[1],
            T=rows,
            H=width,
            S_G=ctx.row_stride,
            BT=block_t,
            BD=block_d,
            num_warps=num_warps,
            num_stages=1,
        )
        grad_log_rate = torch.empty_like(log_rate)
        grad_bias = torch.empty_like(bias)
        _decay_gate_reduce_kernel[(triton.cdiv(width, _REDUCE_BLOCK),)](
            partial[0],
            partial[1],
            grad_log_rate,
            grad_bias,
            TILES=tiles,
            TILES_PADDED=triton.next_power_of_2(tiles),
            H=width,
            BLOCK=_REDUCE_BLOCK,
            num_warps=8,
            num_stages=1,
        )
        return grad_raw, grad_log_rate, grad_bias


def fused_decay_gate(
    raw: torch.Tensor,
    log_rate: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a GDN decay gate without materializing an FP32 raw panel."""
    return _FusedDecayGate.apply(raw, log_rate, bias)


__all__ = ["fused_decay_gate"]
