from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
from torch.autograd.function import once_differentiable

from causallsso.ops.chunk_state import (
    chunk_state_backward,
    chunk_state_forward,
    decay_cumsum_backward,
    decay_cumsum_forward,
)
from causallsso.ops.paired_wy import paired_wy_backward, paired_wy_forward
from causallsso.ops.radial_compact import (
    RadialCompactOutput,
    RadialCompactSaved,
    _radial_compact_reverse_accumulate_trusted,
    radial_compact_forward,
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


class _FrameCache(NamedTuple):
    d: torch.Tensor
    e: torch.Tensor
    chi: torch.Tensor
    lower_primal: torch.Tensor
    lower_dual_scaled: torch.Tensor
    inverse_mass: torch.Tensor
    radial_scale: torch.Tensor
    radial_q2: torch.Tensor
    radial_norm: torch.Tensor
    radial_gram: torch.Tensor
    radial_boundary_pair: torch.Tensor
    radial_boundary_norm: torch.Tensor
    diagonal: torch.Tensor
    alpha0: torch.Tensor
    theta: torch.Tensor
    weights: torch.Tensor


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
    frame: _FrameCache,
    action: tuple[torch.Tensor, ...],
) -> _FrameGradients:
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
        frame.theta,
        frame.weights,
        frame.radial_scale,
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
            frame.inverse_mass,
            frame.theta,
            frame.weights,
            frame.radial_scale,
            frame.radial_q2,
            frame.diagonal,
        ),
        RadialCompactSaved(
            frame.radial_norm,
            frame.radial_gram,
            frame.radial_boundary_pair,
            frame.radial_boundary_norm,
        ),
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
        batch, length, heads, rank = u.shape
        chunks = (length + _CHUNK - 1) // _CHUNK
        padded_length = chunks * _CHUNK
        panels = batch * heads * chunks
        valid_count = _valid_counts(batch, length, heads, u.device)
        panel_strength = (
            geometry_strength[None, :, None]
            .expand(batch, heads, chunks)
            .reshape(panels)
            .contiguous()
        )
        radial, radial_saved = radial_compact_forward(
            _panelize(u, padded_length),
            _panelize(h, padded_length),
            _panelize(geometry_log_decay[..., None], padded_length).squeeze(-1),
            panel_strength,
            boundary.m.reshape(panels),
            boundary.J.reshape(panels, rank, rank),
            boundary.D.reshape(panels, rank, rank),
            valid_count=valid_count,
            return_saved=True,
        )
        alpha0 = radial.theta[:, 0].contiguous()
        prepare = tuple(
            torch.ops.causallsso.c32_solvedelta_prepare_forward(
                u,
                h,
                key,
                erase,
                query,
                boundary.J,
                boundary.D,
                radial.inverse_mass,
                radial.radial_scale,
                radial.diagonal,
                alpha0,
                inclusive_decay,
            )
        )
        if len(prepare) != 10:
            raise RuntimeError(
                "c32_solvedelta_prepare_forward returned an invalid cache"
            )
        W, A_qd, Q_gamma, D_tail, G_last = prepare[:5]
        frame = _FrameCache(
            *prepare[5:],
            radial.inverse_mass,
            radial.radial_scale,
            radial.radial_q2,
            radial_saved.radial_norm,
            radial_saved.gram,
            radial_saved.boundary_pair,
            radial_saved.boundary_norm,
            radial.diagonal,
            alpha0,
            radial.theta,
            radial.weights,
        )
        wy = paired_wy_forward(
            W, frame.e, inclusive_decay, write, value
        )
        Y, U_z = wy
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
        frame_end = 17 + len(_FrameCache._fields)
        frame = _FrameCache(*saved[17:frame_end])
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
        ) = saved[frame_end : frame_end + 10]

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
        wy_grad = paired_wy_backward(
            W,
            Y,
            U_z,
            write,
            value,
            state_grad.grad_Y,
            state_grad.grad_U_z,
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
                frame.d,
                frame.e,
                frame.chi,
                frame.lower_primal,
                frame.lower_dual_scaled,
                frame.inverse_mass,
                frame.radial_scale,
                frame.diagonal,
                frame.alpha0,
                inclusive_decay,
                D_tail,
                Q_gamma,
                wy_grad.system,
                wy_grad.edit_rhs,
                state_grad.grad_A_qd,
                state_grad.grad_Q_gamma,
                state_grad.grad_D_tail,
                state_grad.grad_G_last,
            )
        )
        if len(prepare_grad) != 6:
            raise RuntimeError(
                "c32_solvedelta_prepare_backward returned invalid gradients"
            )
        action = prepare_grad[:5]
        grad_inclusive = prepare_grad[5]
        grad_write, grad_value = wy_grad.write, wy_grad.value
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
