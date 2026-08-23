from __future__ import annotations

import torch
import triton
import triton.language as tl


_RANK = 128
_MATRIX_SIZE = _RANK * _RANK
_BLOCK = 2048
_CTA_WARPS = 16


@triton.jit
def _bounded_ldu_vjp128_kernel(
    H,
    R,
    strength,
    grad_lower,
    grad_diagonal,
    grad_upper,
    grad_omega,
    grad_H,
    grad_R,
    grad_strength_partial,
    MATRIX_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    system = tl.program_id(0)
    matrix_base = system * MATRIX_SIZE
    diagonal_base = system * 128
    geometry_strength = tl.load(strength + system)

    norm_h_lower = 0.0
    norm_h_upper = 0.0
    norm_r_lower = 0.0
    norm_r_upper = 0.0
    projection_h_lower = 0.0
    projection_h_upper = 0.0
    projection_r_lower = 0.0
    projection_r_upper = 0.0

    for start in tl.range(0, MATRIX_SIZE, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        rows = offsets // 128
        cols = offsets % 128
        lower_mask = rows > cols
        upper_mask = rows < cols

        h_value = tl.load(H + matrix_base + offsets)
        r_value = tl.load(R + matrix_base + offsets)
        centered_h = h_value
        x_h = geometry_strength * centered_h
        x_r = geometry_strength * r_value

        omega_value = tl.load(grad_omega + matrix_base + offsets)
        transpose_offsets = cols * 128 + rows
        omega_transpose = tl.load(
            grad_omega + matrix_base + transpose_offsets
        )
        skew_grad = 0.5 * (omega_value - omega_transpose)
        lower_grad = tl.where(
            lower_mask,
            tl.load(
                grad_lower + matrix_base + offsets,
                mask=lower_mask,
                other=0.0,
            )
            + skew_grad,
            0.0,
        )
        upper_grad = tl.where(
            upper_mask,
            tl.load(
                grad_upper + matrix_base + offsets,
                mask=upper_mask,
                other=0.0,
            )
            + skew_grad,
            0.0,
        )

        x_h_lower = tl.where(lower_mask, x_h, 0.0)
        x_h_upper = tl.where(upper_mask, x_h, 0.0)
        x_r_lower = tl.where(lower_mask, x_r, 0.0)
        x_r_upper = tl.where(upper_mask, x_r, 0.0)
        norm_h_lower += tl.sum(x_h_lower * x_h_lower, axis=0)
        norm_h_upper += tl.sum(x_h_upper * x_h_upper, axis=0)
        norm_r_lower += tl.sum(x_r_lower * x_r_lower, axis=0)
        norm_r_upper += tl.sum(x_r_upper * x_r_upper, axis=0)
        projection_h_lower += tl.sum(lower_grad * x_h_lower, axis=0)
        projection_h_upper += tl.sum(upper_grad * x_h_upper, axis=0)
        projection_r_lower += tl.sum(lower_grad * x_r_lower, axis=0)
        projection_r_upper += tl.sum(upper_grad * x_r_upper, axis=0)

    radius = 1.0 / 8.0
    radius_sq = radius * radius
    denominator_h_lower = radius_sq + norm_h_lower
    denominator_h_upper = radius_sq + norm_h_upper
    denominator_r_lower = radius_sq + norm_r_lower
    denominator_r_upper = radius_sq + norm_r_upper
    scale_h_lower = radius * tl.rsqrt(denominator_h_lower)
    scale_h_upper = radius * tl.rsqrt(denominator_h_upper)
    scale_r_lower = radius * tl.rsqrt(denominator_r_lower)
    scale_r_upper = radius * tl.rsqrt(denominator_r_upper)

    strength_partial = 0.0
    for start in tl.range(0, MATRIX_SIZE, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        rows = offsets // 128
        cols = offsets % 128
        lower_mask = rows > cols
        upper_mask = rows < cols

        h_value = tl.load(H + matrix_base + offsets)
        r_value = tl.load(R + matrix_base + offsets)
        centered_h = h_value
        x_h = geometry_strength * centered_h
        x_r = geometry_strength * r_value

        omega_value = tl.load(grad_omega + matrix_base + offsets)
        transpose_offsets = cols * 128 + rows
        omega_transpose = tl.load(
            grad_omega + matrix_base + transpose_offsets
        )
        skew_grad = 0.5 * (omega_value - omega_transpose)
        lower_grad = tl.load(
            grad_lower + matrix_base + offsets,
            mask=lower_mask,
            other=0.0,
        ) + skew_grad
        upper_grad = tl.load(
            grad_upper + matrix_base + offsets,
            mask=upper_mask,
            other=0.0,
        ) + skew_grad

        radial_h_lower = scale_h_lower * (
            lower_grad - x_h * projection_h_lower / denominator_h_lower
        )
        radial_h_upper = scale_h_upper * (
            upper_grad - x_h * projection_h_upper / denominator_h_upper
        )
        radial_r_lower = scale_r_lower * (
            lower_grad - x_r * projection_r_lower / denominator_r_lower
        )
        radial_r_upper = scale_r_upper * (
            upper_grad - x_r * projection_r_upper / denominator_r_upper
        )
        grad_x_h = tl.where(
            lower_mask,
            radial_h_lower,
            tl.where(upper_mask, radial_h_upper, 0.0),
        )
        grad_x_r = tl.where(
            lower_mask,
            radial_r_lower,
            tl.where(upper_mask, radial_r_upper, 0.0),
        )

        strict_mask = lower_mask | upper_mask
        grad_h_value = geometry_strength * grad_x_h
        grad_r_value = geometry_strength * grad_x_r
        tl.store(
            grad_H + matrix_base + offsets,
            grad_h_value,
            mask=strict_mask,
        )
        tl.store(
            grad_R + matrix_base + offsets,
            grad_r_value,
            mask=strict_mask,
        )
        strength_partial += tl.sum(
            tl.where(
                strict_mask,
                grad_x_h * centered_h + grad_x_r * r_value,
                0.0,
            ),
            axis=0,
        )

    diagonal_offsets = tl.arange(0, 128)
    matrix_diagonal_offsets = diagonal_offsets * 129
    h_diagonal = tl.load(H + matrix_base + matrix_diagonal_offsets)
    r_diagonal = tl.load(R + matrix_base + matrix_diagonal_offsets)
    centered_h_diagonal = h_diagonal - 1.0 / 128.0
    x_h_diagonal = geometry_strength * centered_h_diagonal
    x_r_diagonal = geometry_strength * r_diagonal
    tanh_h = tl.extra.libdevice.tanh(x_h_diagonal / radius)
    tanh_r = tl.extra.libdevice.tanh(x_r_diagonal / radius)
    diagonal = tl.exp(radius * tanh_h + radius * tanh_r)
    diagonal_grad = tl.load(
        grad_diagonal + diagonal_base + diagonal_offsets
    )
    grad_log_diagonal = diagonal_grad * diagonal
    grad_x_h_diagonal = grad_log_diagonal * (1.0 - tanh_h * tanh_h)
    grad_x_r_diagonal = grad_log_diagonal * (1.0 - tanh_r * tanh_r)
    grad_h_diagonal = geometry_strength * grad_x_h_diagonal
    grad_r_diagonal = geometry_strength * grad_x_r_diagonal
    tl.store(
        grad_H + matrix_base + matrix_diagonal_offsets,
        grad_h_diagonal,
    )
    tl.store(
        grad_R + matrix_base + matrix_diagonal_offsets,
        grad_r_diagonal,
    )
    strength_partial += tl.sum(
        grad_x_h_diagonal * centered_h_diagonal
        + grad_x_r_diagonal * r_diagonal,
        axis=0,
    )

    tl.store(grad_strength_partial + system, strength_partial)


def bounded_ldu_vjp128(
    H: torch.Tensor,
    R: torch.Tensor,
    strength: torch.Tensor,
    grad_lower: torch.Tensor,
    grad_diagonal: torch.Tensor,
    grad_upper: torch.Tensor,
    grad_omega: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the FP32 VJP of the canonical ``r=128`` bounded LDU chart.

    ``H`` and ``R`` have shape ``[..., 128, 128]``. ``strength`` may be any
    tensor broadcastable to ``...``. The returned strength gradient is one
    deterministic partial per system, with shape ``...``; callers that share a
    strength parameter across systems own the final reduction.
    """
    if not torch.cuda.is_available() or H.device.type != "cuda":
        raise ValueError("bounded_ldu_vjp128 requires CUDA tensors")
    if H.shape != R.shape or H.ndim < 2 or H.shape[-2:] != (128, 128):
        raise ValueError("H and R must have equal [..., 128, 128] shapes")
    if grad_lower.shape != H.shape or grad_upper.shape != H.shape:
        raise ValueError("grad_lower and grad_upper must match H")
    if grad_omega.shape != H.shape:
        raise ValueError("grad_omega must match H")
    if grad_diagonal.shape != H.shape[:-1]:
        raise ValueError("grad_diagonal must have shape [..., 128]")
    tensors = (H, R, strength, grad_lower, grad_diagonal, grad_upper, grad_omega)
    if any(tensor.device != H.device for tensor in tensors):
        raise ValueError("all bounded LDU VJP inputs must share one CUDA device")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise TypeError("bounded_ldu_vjp128 supports FP32 inputs only")

    system_shape = H.shape[:-2]
    try:
        system_strength = torch.broadcast_to(strength, system_shape)
    except RuntimeError as error:
        raise ValueError("strength must be broadcastable to H.shape[:-2]") from error
    systems = H.numel() // _MATRIX_SIZE
    if systems == 0:
        raise ValueError("bounded_ldu_vjp128 requires at least one system")

    H = H.contiguous()
    R = R.contiguous()
    system_strength = system_strength.contiguous()
    grad_lower = grad_lower.contiguous()
    grad_diagonal = grad_diagonal.contiguous()
    grad_upper = grad_upper.contiguous()
    grad_omega = grad_omega.contiguous()
    grad_H = torch.empty_like(H)
    grad_R = torch.empty_like(R)
    grad_strength_partial = torch.empty(
        system_shape, device=H.device, dtype=torch.float32
    )
    _bounded_ldu_vjp128_kernel[(systems,)](
        H,
        R,
        system_strength,
        grad_lower,
        grad_diagonal,
        grad_upper,
        grad_omega,
        grad_H,
        grad_R,
        grad_strength_partial,
        MATRIX_SIZE=_MATRIX_SIZE,
        # Sixteen warps make one 512-thread CTA. The logical tile gives each
        # thread four entries per reduction round without changing the CTA.
        BLOCK=_BLOCK,
        num_warps=_CTA_WARPS,
        num_stages=1,
    )
    return grad_H, grad_R, grad_strength_partial
