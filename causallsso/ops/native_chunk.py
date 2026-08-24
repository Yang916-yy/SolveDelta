from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from causallsso.ops.chunk_state import (
    chunk_state_backward,
    chunk_state_forward,
    decay_cumsum_backward,
    decay_cumsum_forward,
)
from causallsso.ops.radial_compact import (
    RadialCompactOutput,
    RadialCompactSaved,
    _radial_compact_reverse_accumulate_trusted,
)
from causallsso.ops.strict_chart import _strict_chart_direct_transpose_trusted
from causallsso.ops.triton_geometry import (
    _triton_geometry_chunk_scan_backward,
    _triton_geometry_chunk_scan_forward,
)
from causallsso.reference import SolveDeltaState


_CHUNK = 32
_RANK = 128
_LOADED = False


class _FrameGradients(NamedTuple):
    u: torch.Tensor
    h: torch.Tensor
    log_decay: torch.Tensor
    key: torch.Tensor
    erase: torch.Tensor
    query: torch.Tensor
    strength: torch.Tensor
    boundary_m: torch.Tensor
    boundary_j: torch.Tensor
    boundary_d: torch.Tensor


def _library_candidates() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[2]
    return (
        root / "build" / "native-release" / "libcausallsso_chunk.so",
        root / "build" / "native" / "libcausallsso_chunk.so",
        root / "build" / "libcausallsso_chunk.so",
    )


def _ops_registered() -> bool:
    return all(
        hasattr(torch.ops.causallsso, name)
        for name in (
            "c32_solvedelta_prepare_forward",
            "c32_solvedelta_prepare_backward",
        )
    )


def _load_chunk_library() -> None:
    global _LOADED
    if _LOADED or _ops_registered():
        _LOADED = True
        return
    for path in _library_candidates():
        if path.is_file():
            torch.ops.load_library(str(path))
            if not _ops_registered():
                raise RuntimeError(f"{path} does not register the C32 native ABI")
            _LOADED = True
            return
    locations = ", ".join(str(path) for path in _library_candidates())
    raise RuntimeError(
        "the SolveDelta C32 native library is not built; expected it at one "
        f"of: {locations}"
    )


@triton.jit
def _frame_temporal_coefficients_kernel(
    inverse_mass,
    alpha0,
    weights,
    theta,
    C: tl.constexpr,
):
    panel = tl.program_id(0)
    sources = tl.arange(0, C)
    previous = tl.zeros((C,), tl.float32)
    previous_theta = tl.load(alpha0 + panel).to(tl.float32)
    for target in tl.static_range(0, C):
        inverse = tl.load(inverse_mass + panel * C + target).to(tl.float32)
        active = inverse > 0.0
        retain = 1.0 - inverse
        row = retain * previous + tl.where(sources == target, inverse, 0.0)
        row = tl.where(active, row, 0.0)
        local_theta = (
            previous_theta if target == 0 else previous_theta * retain
        )
        local_theta = tl.where(active, local_theta, 0.0)
        tl.store(weights + (panel * C + target) * C + sources, row)
        tl.store(theta + panel * C + target, local_theta)
        previous = row
        previous_theta = local_theta


def _panelize(tensor: torch.Tensor, padded_length: int) -> torch.Tensor:
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
    chunks = padded_length // _CHUNK
    tail = tensor.shape[3:]
    return (
        tensor.reshape(batch, chunks, _CHUNK, heads, *tail)
        .permute(0, 3, 1, 2, *range(4, 4 + len(tail)))
        .reshape(batch * heads * chunks, _CHUNK, *tail)
        .contiguous()
    )


def _frame_temporal_coefficients(
    inverse_mass: torch.Tensor,
    alpha0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    panels = alpha0.numel()
    inverse = inverse_mass.reshape(panels, _CHUNK).contiguous()
    weights = torch.empty(
        panels,
        _CHUNK,
        _CHUNK,
        device=inverse.device,
        dtype=torch.float32,
    )
    theta = torch.empty_like(inverse)
    _frame_temporal_coefficients_kernel[(panels,)](
        inverse,
        alpha0,
        weights,
        theta,
        C=_CHUNK,
        num_warps=1,
    )
    return weights, theta


def _valid_counts(
    batch: int,
    length: int,
    heads: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = (length + _CHUNK - 1) // _CHUNK
    count = (
        length
        - torch.arange(chunks, device=device, dtype=torch.int32) * _CHUNK
    ).clamp_(min=1, max=_CHUNK)
    return (
        count[None, None]
        .expand(batch, heads, chunks)
        .reshape(batch * heads * chunks)
        .contiguous()
    )


def _cotangent(
    gradient: torch.Tensor | None,
    reference: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if gradient is None:
        return torch.zeros_like(reference, dtype=dtype)
    if gradient.shape != reference.shape or gradient.device != reference.device:
        raise RuntimeError("native output cotangent shape/device mismatch")
    return gradient.to(dtype=dtype).contiguous()


def _finish_frame_backward(
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    geometry_strength: torch.Tensor,
    boundary_m: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    frame: tuple[torch.Tensor, ...],
    action: tuple[torch.Tensor, ...],
) -> _FrameGradients:
    (
        _d,
        _e,
        _chi,
        lower_primal,
        lower_dual_scaled,
        inverse_mass,
        radial_scale,
        radial_q2,
        radial_norm,
        diagonal,
        alpha0,
    ) = frame
    (
        grad_key,
        grad_erase,
        grad_query,
        grad_log_diagonal,
        descriptor_bundle,
    ) = action

    batch, length, heads, rank = u.shape
    chunks = (length + _CHUNK - 1) // _CHUNK
    padded_length = chunks * _CHUNK
    panels = batch * heads * chunks
    lower_left, lower_right, upper_left, upper_right = descriptor_bundle.unbind(0)

    panel_u = _panelize(u, padded_length)
    panel_h = _panelize(h, padded_length)
    panel_boundary_j = boundary_j.reshape(panels, rank, rank)
    panel_boundary_d = boundary_d.reshape(panels, rank, rank)
    weights, theta = _frame_temporal_coefficients(inverse_mass, alpha0)
    valid_count = _valid_counts(batch, length, heads, u.device)
    strict = _strict_chart_direct_transpose_trusted(
        lower_left,
        lower_right,
        upper_left,
        upper_right,
        panel_u,
        panel_h,
        panel_boundary_j,
        panel_boundary_d,
        theta,
        weights,
        radial_scale,
        valid_count,
    )

    panel_strength = (
        geometry_strength[None, :, None]
        .expand(batch, heads, chunks)
        .reshape(panels)
        .contiguous()
    )
    radial = _radial_compact_reverse_accumulate_trusted(
        panel_u,
        panel_h,
        _panelize(geometry_log_decay[..., None], padded_length).squeeze(-1),
        panel_strength,
        boundary_m.reshape(panels),
        panel_boundary_j,
        panel_boundary_d,
        RadialCompactOutput(
            inverse_mass.reshape(panels, _CHUNK),
            theta,
            weights,
            radial_scale,
            radial_q2,
            diagonal,
        ),
        RadialCompactSaved(radial_norm),
        strict.grad_radial_scale,
        _panelize(grad_log_diagonal, padded_length),
        valid_count,
        strict.grad_theta,
        strict.grad_weights,
        strict.grad_u,
        strict.grad_h,
        strict.grad_boundary_j,
        strict.grad_boundary_d,
    )
    return _FrameGradients(
        radial.grad_u,
        radial.grad_h,
        radial.grad_log_decay,
        grad_key,
        grad_erase,
        grad_query,
        radial.grad_strength.reshape(batch, heads, chunks).sum(dim=(0, 2)),
        radial.grad_boundary_m.reshape_as(boundary_m),
        radial.grad_boundary_j.reshape_as(boundary_j),
        radial.grad_boundary_d.reshape_as(boundary_d),
    )


class _NativeChunkSolveWY(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,
        h: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        geometry_log_decay: torch.Tensor,
        associative_log_decay: torch.Tensor,
        erase: torch.Tensor,
        write: torch.Tensor,
        geometry_strength: torch.Tensor,
        initial_m: torch.Tensor,
        initial_j: torch.Tensor,
        initial_d: torch.Tensor,
        initial_s: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        _load_chunk_library()
        empty_s = initial_s.new_empty(0)
        boundary, geometry_final = _triton_geometry_chunk_scan_forward(
            u,
            h,
            geometry_log_decay,
            initial_state=SolveDeltaState(initial_m, initial_j, initial_d, empty_s),
            chunk_size=_CHUNK,
            input_precision="ieee",
        )
        inclusive_decay = decay_cumsum_forward(associative_log_decay)
        prepare = tuple(
            torch.ops.causallsso.c32_solvedelta_prepare_forward(
                u,
                h,
                geometry_log_decay,
                key,
                erase,
                query,
                geometry_strength,
                boundary.m,
                boundary.J,
                boundary.D,
                inclusive_decay,
                write,
                value,
            )
        )
        if len(prepare) != 18:
            raise RuntimeError(
                "c32_solvedelta_prepare_forward returned an invalid cache"
            )
        W, A_qd, Q_gamma, D_tail, G_last, Y, U_z = prepare[:7]
        frame = prepare[7:]
        state = chunk_state_forward(
            Y, U_z, D_tail, Q_gamma, A_qd, G_last, initial_s
        )
        ctx.save_for_backward(
            u,
            h,
            query,
            key,
            value,
            geometry_log_decay,
            associative_log_decay,
            erase,
            write,
            geometry_strength,
            initial_m,
            initial_j,
            initial_d,
            initial_s,
            boundary.m,
            boundary.J,
            boundary.D,
            *frame,
            inclusive_decay,
            W,
            A_qd,
            Q_gamma,
            D_tail,
            G_last,
            Y,
            U_z,
            state.boundaries,
            state.residual,
        )
        ctx.set_materialize_grads(False)
        return (
            state.output,
            geometry_final.m,
            geometry_final.J,
            geometry_final.D,
            state.final_state,
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        grad_output,
        grad_final_m,
        grad_final_j,
        grad_final_d,
        grad_final_s,
    ) -> tuple[torch.Tensor | None, ...]:
        saved = ctx.saved_tensors
        inputs = saved[:14]
        (
            u,
            h,
            query,
            key,
            value,
            geometry_log_decay,
            associative_log_decay,
            erase,
            write,
            geometry_strength,
            initial_m,
            initial_j,
            initial_d,
            initial_s,
        ) = inputs
        boundary_m, boundary_j, boundary_d = saved[14:17]
        frame = tuple(saved[17:28])
        (
            inclusive_decay,
            W,
            A_qd,
            Q_gamma,
            D_tail,
            G_last,
            Y,
            U_z,
            state_boundaries,
            residual,
        ) = saved[28:38]

        grad_output = _cotangent(grad_output, U_z, dtype=torch.bfloat16)
        grad_final_s = _cotangent(grad_final_s, initial_s, dtype=torch.float32)
        state_grad = chunk_state_backward(
            Y,
            D_tail,
            Q_gamma,
            A_qd,
            G_last,
            state_boundaries,
            residual,
            grad_output,
            grad_final_s,
        )
        prepare_grad = tuple(
            torch.ops.causallsso.c32_solvedelta_prepare_backward(
                u,
                h,
                key,
                erase,
                query,
                boundary_j,
                boundary_d,
                frame[0],
                frame[1],
                frame[2],
                frame[3],
                frame[4],
                frame[5],
                frame[6],
                frame[9],
                frame[10],
                inclusive_decay,
                W,
                D_tail,
                Q_gamma,
                Y,
                U_z,
                write,
                value,
                state_grad.grad_Y,
                state_grad.grad_U_z,
                state_grad.grad_A_qd,
                state_grad.grad_Q_gamma,
                state_grad.grad_D_tail,
                state_grad.grad_G_last,
            )
        )
        if len(prepare_grad) != 8:
            raise RuntimeError(
                "c32_solvedelta_prepare_backward returned invalid gradients"
            )
        action = prepare_grad[:5]
        grad_inclusive, grad_write, grad_value = prepare_grad[5:]
        grad_associative_decay = decay_cumsum_backward(grad_inclusive)

        frame_grad = _finish_frame_backward(
            u,
            h,
            geometry_log_decay,
            geometry_strength,
            boundary_m,
            boundary_j,
            boundary_d,
            frame,
            action,
        )
        scan_grad = _triton_geometry_chunk_scan_backward(
            u,
            h,
            geometry_log_decay,
            boundary_m,
            boundary_j,
            boundary_d,
            frame_grad.boundary_m,
            frame_grad.boundary_j,
            frame_grad.boundary_d,
            _cotangent(grad_final_m, initial_m, dtype=torch.float32),
            _cotangent(grad_final_j, initial_j, dtype=torch.float32),
            _cotangent(grad_final_d, initial_d, dtype=torch.float32),
            _CHUNK,
            panel_gradients=(
                frame_grad.u,
                frame_grad.h,
                frame_grad.log_decay,
            ),
        )
        gradients = (
            scan_grad[0],
            scan_grad[1],
            frame_grad.query,
            frame_grad.key,
            grad_value,
            scan_grad[2],
            grad_associative_decay,
            frame_grad.erase,
            grad_write,
            frame_grad.strength,
            scan_grad[3],
            scan_grad[4],
            scan_grad[5],
            state_grad.grad_initial_state,
        )
        return tuple(
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad)
        )


def native_chunk_solvedelta(
    u: torch.Tensor,
    h: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    geometry_strength: torch.Tensor,
    initial_state: SolveDeltaState,
) -> tuple[torch.Tensor, SolveDeltaState]:
    outputs = _NativeChunkSolveWY.apply(
        u,
        h,
        query,
        key,
        value,
        geometry_log_decay,
        associative_log_decay,
        erase,
        write,
        geometry_strength,
        initial_state.m,
        initial_state.J,
        initial_state.D,
        initial_state.S,
    )
    return outputs[0], SolveDeltaState(*outputs[1:])


__all__ = ["native_chunk_solvedelta"]
