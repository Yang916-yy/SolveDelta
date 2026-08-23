from __future__ import annotations

import torch
import triton
import triton.language as tl


_CHUNK = tl.constexpr(32)
_HOST_CHUNK = 32
_HOST_RANK = 128
_RADIUS = 1.0 / 8.0


def _panelize_tokens(tensor: torch.Tensor, padded_length: int) -> torch.Tensor:
    batch, length, heads = tensor.shape[:3]
    if length != padded_length:
        tensor = torch.cat(
            (
                tensor,
                tensor.new_zeros(
                    batch,
                    padded_length - length,
                    heads,
                    *tensor.shape[3:],
                ),
            ),
            dim=1,
        )
    chunks = padded_length // _HOST_CHUNK
    tail = tensor.shape[3:]
    return (
        tensor.reshape(batch, chunks, _HOST_CHUNK, heads, *tail)
        .permute(0, 3, 1, 2, *range(4, 4 + len(tail)))
        .reshape(batch * heads * chunks, _HOST_CHUNK, *tail)
        .contiguous()
    )


def _unpanelize_tokens(
    tensor: torch.Tensor,
    *,
    batch: int,
    length: int,
    heads: int,
) -> torch.Tensor:
    chunks = (length + _HOST_CHUNK - 1) // _HOST_CHUNK
    tail = tensor.shape[2:]
    return (
        tensor.reshape(batch, heads, chunks, _HOST_CHUNK, *tail)
        .permute(0, 2, 3, 1, *range(4, 4 + len(tail)))
        .reshape(batch, chunks * _HOST_CHUNK, heads, *tail)[:, :length]
        .contiguous()
    )


def _bf16_bmm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.bmm(
        left.to(torch.bfloat16),
        right.to(torch.bfloat16),
        out_dtype=torch.float32,
    )


def _strict_boundary(
    boundary: torch.Tensor, *, upper: bool
) -> torch.Tensor:
    return (
        torch.triu(boundary, diagonal=1)
        if upper
        else torch.tril(boundary, diagonal=-1)
    )


def _boundary_action_setup(
    boundary: torch.Tensor,
    descriptor_right: torch.Tensor,
    local_right: torch.Tensor,
    *,
    upper: bool,
) -> tuple[torch.Tensor, ...]:
    panels = boundary.shape[0]
    strict = _strict_boundary(boundary, upper=upper)
    strict_high = strict.to(torch.bfloat16)
    strict_low = (strict - strict_high.float()).to(torch.bfloat16)
    right = torch.cat(
        (
            descriptor_right.reshape(
                panels, _HOST_CHUNK * 3, _HOST_RANK
            ),
            local_right,
        ),
        dim=1,
    )
    packed_right = right.transpose(1, 2).to(torch.bfloat16)
    action = (
        torch.bmm(strict_high, packed_right, out_dtype=torch.float32)
        + torch.bmm(strict_low, packed_right, out_dtype=torch.float32)
    ).transpose(1, 2)
    return strict, action, strict_high, strict_low, right


def _compact_boundary_gradient(
    descriptor_left: torch.Tensor,
    local_left: torch.Tensor,
    direct: torch.Tensor,
    mix: torch.Tensor,
    packed_right: torch.Tensor,
    strict: torch.Tensor,
    *,
    upper: bool,
) -> torch.Tensor:
    panels = descriptor_left.shape[0]
    weighted_descriptor = (
        descriptor_left.float() * direct[:, 0, :, None, None]
    ).reshape(panels, _HOST_CHUNK * 3, _HOST_RANK)
    gradient = _bf16_bmm(
        torch.cat(
            (
                weighted_descriptor,
                local_left.float() * mix[:, 0, 1:, None],
            ),
            dim=1,
        ).transpose(1, 2),
        packed_right,
    )
    return _strict_boundary(gradient, upper=upper) + (
        mix[:, 0, 0, None, None] * strict
    )


def _boundary_transpose_action(
    strict_high: torch.Tensor,
    strict_low: torch.Tensor,
    local_left: torch.Tensor,
) -> torch.Tensor:
    packed_left = local_left.transpose(1, 2).to(torch.bfloat16)
    return (
        torch.bmm(
            strict_high.transpose(1, 2),
            packed_left,
            out_dtype=torch.float32,
        )
        + torch.bmm(
            strict_low.transpose(1, 2),
            packed_left,
            out_dtype=torch.float32,
        )
    ).transpose(1, 2)


def _side_geometry_backward(
    descriptor_left: torch.Tensor,
    descriptor_right: torch.Tensor,
    panel_u: torch.Tensor,
    panel_h: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    temporal: torch.Tensor,
    inverse_mass: torch.Tensor,
    radial_scale: torch.Tensor,
    radial_q2: torch.Tensor,
    strength: torch.Tensor,
    *,
    upper: bool,
) -> tuple[torch.Tensor, ...]:
    descriptor_j, descriptor_d, gram_j, gram_d = (
        torch.ops.causallsso.c32_frame_compact_pair(
            descriptor_left,
            descriptor_right,
            panel_u,
            panel_h,
            upper,
        )
    )
    j_strict, j_action, j_high, j_low, j_right = _boundary_action_setup(
        boundary_j,
        descriptor_right,
        panel_u,
        upper=upper,
    )
    d_strict, d_action, d_high, d_low, d_right = _boundary_action_setup(
        boundary_d,
        descriptor_right,
        panel_h,
        upper=upper,
    )
    direct_j, direct_d, mix_j, mix_d, grad_temporal, grad_strength = (
        torch.ops.causallsso.c32_frame_compact_coefficients(
            descriptor_left,
            panel_u,
            boundary_j,
            boundary_d,
            j_action,
            d_action,
            descriptor_j,
            descriptor_d,
            gram_j,
            gram_d,
            temporal,
            inverse_mass,
            radial_scale,
            radial_q2,
            strength,
            upper,
        )
    )
    grad_boundary_j = _compact_boundary_gradient(
        descriptor_left,
        panel_u,
        direct_j,
        mix_j,
        j_right,
        j_strict,
        upper=upper,
    )
    grad_boundary_d = _compact_boundary_gradient(
        descriptor_left,
        panel_u,
        direct_d,
        mix_d,
        d_right,
        d_strict,
        upper=upper,
    )
    j_transpose = _boundary_transpose_action(j_high, j_low, panel_u)
    d_transpose = _boundary_transpose_action(d_high, d_low, panel_u)
    grad_u, grad_h = torch.ops.causallsso.c32_frame_compact_leaf(
        descriptor_left,
        descriptor_right,
        direct_j,
        direct_d,
        mix_j,
        mix_d,
        panel_u,
        panel_h,
        j_action[:, _HOST_CHUNK * 3 :],
        j_transpose,
        d_action[:, _HOST_CHUNK * 3 :],
        d_transpose,
        upper,
    )
    return (
        grad_boundary_j,
        grad_boundary_d,
        grad_u,
        grad_h,
        grad_temporal,
        grad_strength,
    )


def _diagonal_geometry_backward(
    panel_u: torch.Tensor,
    panel_h: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    temporal: torch.Tensor,
    grad_log_diagonal: torch.Tensor,
    strength: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    temporal_boundary = temporal[:, :, :1]
    weights = temporal[:, :, 1:]
    local_j = panel_u.float().square()
    local_d = panel_u.float() * panel_h.float()
    diagonal_j = (
        temporal_boundary
        * torch.diagonal(boundary_j, dim1=-2, dim2=-1)[:, None]
        + torch.bmm(weights, local_j)
    )
    diagonal_d = (
        temporal_boundary
        * torch.diagonal(boundary_d, dim1=-2, dim2=-1)[:, None]
        + torch.bmm(weights, local_d)
    )
    centered_j = diagonal_j - 1.0 / _HOST_RANK
    tanh_j = torch.tanh(
        strength[:, None, None] * centered_j / _RADIUS
    )
    tanh_d = torch.tanh(
        strength[:, None, None] * diagonal_d / _RADIUS
    )
    sech_j = 1.0 - tanh_j.square()
    sech_d = 1.0 - tanh_d.square()
    adjoint_j = (
        strength[:, None, None] * sech_j * grad_log_diagonal
    )
    adjoint_d = (
        strength[:, None, None] * sech_d * grad_log_diagonal
    )
    basis_j = torch.cat(
        (
            torch.diagonal(boundary_j, dim1=-2, dim2=-1)[:, None],
            local_j,
        ),
        dim=1,
    )
    basis_d = torch.cat(
        (
            torch.diagonal(boundary_d, dim1=-2, dim2=-1)[:, None],
            local_d,
        ),
        dim=1,
    )
    grad_temporal = torch.bmm(
        adjoint_j, basis_j.transpose(1, 2)
    ) + torch.bmm(adjoint_d, basis_d.transpose(1, 2))
    grad_boundary_j = (
        temporal_boundary * adjoint_j
    ).sum(dim=1)
    grad_boundary_d = (
        temporal_boundary * adjoint_d
    ).sum(dim=1)
    grad_local_j = torch.bmm(weights.transpose(1, 2), adjoint_j)
    grad_local_d = torch.bmm(weights.transpose(1, 2), adjoint_d)
    grad_u = 2.0 * panel_u.float() * grad_local_j
    grad_u += panel_h.float() * grad_local_d
    grad_h = panel_u.float() * grad_local_d
    grad_strength = (
        grad_log_diagonal
        * (sech_j * centered_j + sech_d * diagonal_d)
    ).sum(dim=(1, 2))
    return (
        grad_boundary_j,
        grad_boundary_d,
        grad_u,
        grad_h,
        grad_temporal,
        grad_strength,
    )


@triton.jit
def _temporal_scalar_backward_kernel(
    panel_log_decay,
    boundary_mass,
    inverse_mass,
    weights,
    theta,
    grad_temporal,
    valid_count,
    grad_log_decay,
    grad_boundary_mass,
):
    panel = tl.program_id(0)
    source = tl.arange(0, _CHUNK)
    propagated_row = tl.zeros((_CHUNK,), tl.float32)
    propagated_theta = 0.0
    grad_mass = 0.0
    count = tl.load(valid_count + panel)
    boundary = tl.load(boundary_mass + panel).to(tl.float32)
    for step in tl.static_range(0, _CHUNK):
        target = _CHUNK - 1 - step
        active = target < count
        current_row = tl.load(
            grad_temporal
            + (panel * _CHUNK + target) * (_CHUNK + 1)
            + 1
            + source
        ).to(tl.float32) + propagated_row
        current_theta = tl.load(
            grad_temporal
            + (panel * _CHUNK + target) * (_CHUNK + 1)
        ).to(tl.float32) + propagated_theta
        inverse = tl.load(
            inverse_mass + panel * _CHUNK + target
        ).to(tl.float32)
        decay_value = tl.load(
            panel_log_decay + panel * _CHUNK + target
        ).to(tl.float32)
        decay = tl.exp(decay_value)
        if target == 0:
            grad_inverse = tl.sum(
                tl.where(source == 0, current_row, 0.0), axis=0
            ) + current_theta * decay
            grad_lambda = current_theta * inverse
            mass_previous = boundary
            propagated_row = tl.zeros((_CHUNK,), tl.float32)
            propagated_theta = 0.0
        else:
            retain = 1.0 - inverse
            previous_theta = tl.load(
                theta + panel * _CHUNK + target - 1
            ).to(tl.float32)
            previous_row = tl.load(
                weights
                + (panel * _CHUNK + target - 1) * _CHUNK
                + source
            ).to(tl.float32)
            grad_retain = current_theta * previous_theta
            grad_retain += tl.sum(
                tl.where(source < target, current_row * previous_row, 0.0),
                axis=0,
            )
            grad_diagonal = tl.sum(
                tl.where(source == target, current_row, 0.0), axis=0
            )
            grad_inverse = grad_diagonal - grad_retain
            grad_lambda = 0.0
            previous_inverse = tl.load(
                inverse_mass + panel * _CHUNK + target - 1
            ).to(tl.float32)
            mass_previous = tl.where(
                previous_inverse > 0.0, 1.0 / previous_inverse, 0.0
            )
            propagated_row = tl.where(
                active & (source < target), retain * current_row, 0.0
            )
            propagated_theta = tl.where(
                active, retain * current_theta, 0.0
            )
        total_mass = grad_mass - grad_inverse * inverse * inverse
        grad_lambda += total_mass * mass_previous
        tl.store(
            grad_log_decay + panel * _CHUNK + target,
            tl.where(active, grad_lambda * decay, 0.0),
        )
        grad_mass = tl.where(active, total_mass * decay, grad_mass)
    tl.store(grad_boundary_mass + panel, grad_mass)


def _temporal_scalar_backward(
    panel_log_decay: torch.Tensor,
    boundary_mass: torch.Tensor,
    inverse_mass: torch.Tensor,
    weights: torch.Tensor,
    theta: torch.Tensor,
    grad_temporal: torch.Tensor,
    valid_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    panels = panel_log_decay.shape[0]
    grad_log_decay = torch.zeros_like(panel_log_decay)
    grad_mass = torch.empty(
        panels, device=panel_log_decay.device, dtype=torch.float32
    )
    _temporal_scalar_backward_kernel[(panels,)](
        panel_log_decay,
        boundary_mass,
        inverse_mass,
        weights,
        theta,
        grad_temporal,
        valid_count,
        grad_log_decay,
        grad_mass,
        num_warps=1,
    )
    return grad_log_decay, grad_mass


@triton.jit
def _frame_weights_from_alpha_kernel(
    inverse_mass,
    alpha0,
    weights,
    theta,
):
    panel = tl.program_id(0)
    source = tl.arange(0, _CHUNK)
    previous = tl.zeros((_CHUNK,), tl.float32)
    previous_theta = tl.load(alpha0 + panel).to(tl.float32)
    for target in tl.static_range(0, _CHUNK):
        inverse = tl.load(
            inverse_mass + panel * _CHUNK + target
        ).to(tl.float32)
        active = inverse > 0.0
        retain = 1.0 - inverse
        row = retain * previous + tl.where(source == target, inverse, 0.0)
        row = tl.where(active, row, 0.0)
        local_theta = (
            previous_theta if target == 0 else previous_theta * retain
        )
        local_theta = tl.where(active, local_theta, 0.0)
        tl.store(
            weights + (panel * _CHUNK + target) * _CHUNK + source,
            row,
        )
        tl.store(theta + panel * _CHUNK + target, local_theta)
        previous = row
        previous_theta = local_theta


def _frame_weights_from_alpha(
    inverse_mass: torch.Tensor,
    alpha0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    panels = alpha0.numel()
    flat_inverse = inverse_mass.reshape(panels, _HOST_CHUNK)
    weights = torch.empty(
        panels,
        _HOST_CHUNK,
        _HOST_CHUNK,
        device=inverse_mass.device,
        dtype=torch.float32,
    )
    theta = torch.empty(
        panels,
        _HOST_CHUNK,
        device=inverse_mass.device,
        dtype=torch.float32,
    )
    _frame_weights_from_alpha_kernel[(panels,)](
        flat_inverse,
        alpha0,
        weights,
        theta,
        num_warps=1,
    )
    return weights, theta


def _finish_c32_frame_backward(
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    key: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    geometry_strength: torch.Tensor,
    boundary_m: torch.Tensor,
    boundary_J: torch.Tensor,
    boundary_D: torch.Tensor,
    lower_primal: torch.Tensor,
    scaled_dual: torch.Tensor,
    write: torch.Tensor,
    inverse_mass: torch.Tensor,
    coefficient: torch.Tensor,
    radial_q2: torch.Tensor,
    weights: torch.Tensor,
    theta: torch.Tensor,
    grad_e: torch.Tensor,
    grad_chi: torch.Tensor,
    action: tuple[torch.Tensor, ...],
    *,
    retain_fp32_vector_partials: bool = False,
) -> tuple[torch.Tensor, ...]:
    grad_key, grad_erase, grad_query = action[:3]
    x_upper, upper_direct, x_lower, lower_direct, grad_log = action[3:]
    batch, length, heads, rank = u.shape
    chunks = (length + _HOST_CHUNK - 1) // _HOST_CHUNK
    padded_length = chunks * _HOST_CHUNK
    panels = batch * heads * chunks
    local_b = erase.squeeze(-2).float() * key.squeeze(-2).float()

    upper_left = torch.stack(
        (-x_upper, scaled_dual[:, :, :, 0], scaled_dual[:, :, :, 1]),
        dim=3,
    ).to(torch.bfloat16)
    upper_right = torch.stack(
        (write, grad_e.squeeze(-2).float(), grad_chi.float()), dim=3
    ).to(torch.bfloat16)
    lower_left = torch.stack(
        (-x_lower, local_b, query.float()), dim=3
    ).to(torch.bfloat16)
    lower_right = torch.stack(
        (lower_primal, lower_direct[:, :, :, 0], lower_direct[:, :, :, 1]),
        dim=3,
    ).to(torch.bfloat16)

    panel_u = _panelize_tokens(u, padded_length)
    panel_h = _panelize_tokens(h, padded_length)
    panel_upper_left = _panelize_tokens(upper_left, padded_length)
    panel_upper_right = _panelize_tokens(upper_right, padded_length)
    panel_lower_left = _panelize_tokens(lower_left, padded_length)
    panel_lower_right = _panelize_tokens(lower_right, padded_length)
    panel_grad_log = _panelize_tokens(grad_log, padded_length)
    panel_boundary_j = boundary_J.reshape(panels, rank, rank)
    panel_boundary_d = boundary_D.reshape(panels, rank, rank)
    temporal = torch.cat((theta[:, :, None], weights), dim=-1)
    panel_strength = (
        geometry_strength[None, :, None]
        .expand(batch, heads, chunks)
        .reshape(panels)
        .contiguous()
    )

    lower_side = _side_geometry_backward(
        panel_lower_left,
        panel_lower_right,
        panel_u,
        panel_h,
        panel_boundary_j,
        panel_boundary_d,
        temporal,
        inverse_mass.reshape(panels, _HOST_CHUNK),
        coefficient,
        radial_q2,
        panel_strength,
        upper=False,
    )
    upper_side = _side_geometry_backward(
        panel_upper_left,
        panel_upper_right,
        panel_u,
        panel_h,
        panel_boundary_j,
        panel_boundary_d,
        temporal,
        inverse_mass.reshape(panels, _HOST_CHUNK),
        coefficient,
        radial_q2,
        panel_strength,
        upper=True,
    )
    diagonal_pullback = _diagonal_geometry_backward(
        panel_u,
        panel_h,
        panel_boundary_j,
        panel_boundary_d,
        temporal,
        panel_grad_log,
        panel_strength,
    )

    grad_boundary_j = lower_side[0] + upper_side[0]
    grad_boundary_d = lower_side[1] + upper_side[1]
    grad_boundary_j.diagonal(dim1=-2, dim2=-1).add_(diagonal_pullback[0])
    grad_boundary_d.diagonal(dim1=-2, dim2=-1).add_(diagonal_pullback[1])
    panel_grad_u = lower_side[2] + upper_side[2] + diagonal_pullback[2]
    panel_grad_h = lower_side[3] + upper_side[3] + diagonal_pullback[3]
    grad_temporal = lower_side[4] + upper_side[4] + diagonal_pullback[4]
    panel_grad_strength = lower_side[5] + upper_side[5] + diagonal_pullback[5]
    panel_decay = _panelize_tokens(geometry_log_decay, padded_length)
    valid_count = (
        length
        - torch.arange(chunks, device=u.device, dtype=torch.int64)
        * _HOST_CHUNK
    ).clamp(min=0, max=_HOST_CHUNK)
    valid_count = (
        valid_count[None, None]
        .expand(batch, heads, chunks)
        .reshape(panels)
        .contiguous()
    )
    panel_grad_decay, panel_grad_mass = _temporal_scalar_backward(
        panel_decay,
        boundary_m.reshape(panels),
        inverse_mass.reshape(panels, _HOST_CHUNK),
        weights,
        theta,
        grad_temporal,
        valid_count,
    )
    grad_u = _unpanelize_tokens(
        panel_grad_u, batch=batch, length=length, heads=heads
    )
    grad_h = _unpanelize_tokens(
        panel_grad_h, batch=batch, length=length, heads=heads
    )
    if not retain_fp32_vector_partials:
        grad_u = grad_u.to(torch.bfloat16)
        grad_h = grad_h.to(torch.bfloat16)
    grad_decay = _unpanelize_tokens(
        panel_grad_decay, batch=batch, length=length, heads=heads
    )
    grad_strength = panel_grad_strength.reshape(
        batch, heads, chunks
    ).sum(dim=(0, 2))
    return (
        grad_u,
        grad_h,
        grad_decay,
        grad_key,
        grad_erase,
        grad_query,
        grad_strength,
        panel_grad_mass.reshape_as(boundary_m),
        grad_boundary_j.reshape_as(boundary_J),
        grad_boundary_d.reshape_as(boundary_D),
    )


def resident_c32_frame_backward(
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    key: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    geometry_strength: torch.Tensor,
    boundary_m: torch.Tensor,
    boundary_J: torch.Tensor,
    boundary_D: torch.Tensor,
    lower_primal: torch.Tensor,
    lower_dual_scaled: torch.Tensor,
    write: torch.Tensor,
    inverse_mass: torch.Tensor,
    coefficient: torch.Tensor,
    radial_q2: torch.Tensor,
    diagonal: torch.Tensor,
    alpha0: torch.Tensor,
    grad_d: torch.Tensor,
    grad_e: torch.Tensor,
    grad_chi: torch.Tensor,
    *,
    retain_fp32_vector_partials: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Run the resident transpose action and compact chart VJP."""
    action = torch.ops.causallsso.c32_frame_resident_action_backward(
        u,
        h,
        key,
        erase,
        boundary_J,
        boundary_D,
        lower_primal,
        lower_dual_scaled,
        inverse_mass,
        coefficient,
        diagonal,
        alpha0,
        grad_d,
        grad_e,
        grad_chi,
    )
    weights, theta = _frame_weights_from_alpha(inverse_mass, alpha0)
    return _finish_c32_frame_backward(
        u,
        h,
        geometry_log_decay,
        key,
        erase,
        query,
        geometry_strength,
        boundary_m,
        boundary_J,
        boundary_D,
        lower_primal,
        lower_dual_scaled,
        write,
        inverse_mass,
        coefficient,
        radial_q2,
        weights,
        theta,
        grad_e,
        grad_chi,
        action,
        retain_fp32_vector_partials=retain_fp32_vector_partials,
    )


__all__ = ["resident_c32_frame_backward"]
