from __future__ import annotations

from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from causallsso.ops.triton_geometry import (
    _triton_geometry_chunk_scan_backward,
    _triton_geometry_chunk_scan_forward,
)
from causallsso.reference import SolveDeltaState


_CHUNK_SIZE = 32
_RANK = 128
_EDITS = 1
_LOADED = False

_GEOMETRY_INPUT_NAMES = (
    "u",
    "h",
    "geometry_log_decay",
    "key",
    "erase",
    "query",
    "geometry_strength",
    "initial_m",
    "initial_J",
    "initial_D",
)


def _library_candidates() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[2]
    return (
        root / "build" / "native" / "libcausallsso_chunk.so",
        root / "build" / "libcausallsso_chunk.so",
    )


def _ops_registered() -> bool:
    required = (
        "c32_frame_resident_forward",
        "c32_frame_resident_action_backward",
        "c32_frame_compact_pair",
        "c32_frame_compact_coefficients",
        "c32_frame_compact_leaf",
    )
    return all(hasattr(torch.ops.causallsso, name) for name in required)


def _load_chunk_library() -> None:
    global _LOADED
    if _LOADED:
        return
    if _ops_registered():
        _LOADED = True
        return

    candidates = _library_candidates()
    for path in candidates:
        if not path.is_file():
            continue
        torch.ops.load_library(str(path))
        if not _ops_registered():
            raise RuntimeError(
                f"{path} does not register the resident C32 forward/reverse ABI"
            )
        _LOADED = True
        return

    locations = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        "the SolveDelta C32 native library is not built; expected "
        f"libcausallsso_chunk.so at one of: {locations}"
    )


def _validate_forward_outputs(
    outputs: object,
    inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 11:
        raise RuntimeError(
            "c32_frame_resident_forward must return 3 public tensors and "
            "8 compact saved tensors"
        )
    tensors = tuple(outputs)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise RuntimeError("resident forward returned a non-Tensor value")

    u, _, _, key, _, query, _, _, _, _ = inputs
    batch, length, heads, rank = u.shape
    chunks = (length + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    panels = batch * heads * chunks
    expected = (
        ("d", key.shape, torch.bfloat16),
        ("e", key.shape, torch.bfloat16),
        ("chi", query.shape, torch.bfloat16),
        ("lower_primal", u.shape, torch.float32),
        (
            "lower_dual_scaled",
            (batch, length, heads, 2, rank),
            torch.float32,
        ),
        ("write_fp32", u.shape, torch.float32),
        (
            "inverse_mass",
            (batch, heads, chunks, _CHUNK_SIZE),
            torch.float32,
        ),
        ("radial_scale", (panels, _CHUNK_SIZE, 4), torch.float32),
        ("radial_q2", (panels, _CHUNK_SIZE, 4), torch.float32),
        ("diagonal", (panels, _CHUNK_SIZE, rank), torch.float32),
        ("alpha0", (panels,), torch.float32),
    )
    for (name, shape, dtype), tensor in zip(expected, tensors):
        if tensor.shape != shape:
            raise RuntimeError(
                f"resident forward returned {name} with shape "
                f"{tuple(tensor.shape)}; expected {shape}"
            )
        if tensor.dtype != dtype or tensor.device != u.device:
            raise RuntimeError(
                f"resident forward returned {name} outside its dtype/device ABI"
            )
    return tensors


def _output_cotangent(
    name: str,
    gradient: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    if gradient is None:
        return torch.zeros_like(reference)
    if gradient.shape != reference.shape:
        raise RuntimeError(
            f"{name} has shape {tuple(gradient.shape)}; "
            f"expected {tuple(reference.shape)}"
        )
    if gradient.dtype != torch.bfloat16 or gradient.device != reference.device:
        raise RuntimeError(f"{name} must match the BF16 resident output")
    return gradient.contiguous()


def _fp32_cotangent(
    name: str,
    gradient: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    if gradient is None:
        return torch.zeros_like(reference)
    if gradient.shape != reference.shape:
        raise RuntimeError(
            f"{name} has shape {tuple(gradient.shape)}; "
            f"expected {tuple(reference.shape)}"
        )
    if gradient.dtype != torch.float32 or gradient.device != reference.device:
        raise RuntimeError(f"{name} must match the FP32 geometry state")
    return gradient.contiguous()


def _validate_native_geometry_inputs(
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    key: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    geometry_strength: torch.Tensor,
    initial_m: torch.Tensor,
    initial_J: torch.Tensor,
    initial_D: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    inputs = (
        u,
        h,
        geometry_log_decay,
        key,
        erase,
        query,
        geometry_strength,
        initial_m,
        initial_J,
        initial_D,
    )
    for name, tensor in zip(_GEOMETRY_INPUT_NAMES, inputs):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if u.ndim != 4:
        raise ValueError("u must have shape [B,T,H,128]")
    batch, length, heads, rank = u.shape
    if min(batch, length, heads) < 1:
        raise ValueError("B, T, and H must be positive")
    if rank != _RANK:
        raise ValueError("the native geometry/frame requires r=128")
    expected_shapes = {
        "h": u.shape,
        "geometry_log_decay": (batch, length, heads),
        "key": (batch, length, heads, _EDITS, rank),
        "erase": (batch, length, heads, _EDITS, rank),
        "query": u.shape,
        "geometry_strength": (heads,),
        "initial_m": (batch, heads),
        "initial_J": (batch, heads, rank, rank),
        "initial_D": (batch, heads, rank, rank),
    }
    named_inputs = dict(zip(_GEOMETRY_INPUT_NAMES, inputs))
    for name, shape in expected_shapes.items():
        if named_inputs[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for name in ("u", "h", "key", "erase", "query"):
        if named_inputs[name].dtype != torch.bfloat16:
            raise TypeError(f"{name} must be BF16")
    for name in (
        "geometry_log_decay",
        "geometry_strength",
        "initial_m",
        "initial_J",
        "initial_D",
    ):
        if named_inputs[name].dtype != torch.float32:
            raise TypeError(f"{name} must be FP32")
    if u.device.type != "cuda":
        raise ValueError("the native geometry/frame requires CUDA tensors")
    if any(tensor.device != u.device for tensor in inputs):
        raise ValueError("all native geometry/frame tensors must share one device")
    return tuple(tensor.contiguous() for tensor in inputs)


class _NativeGeometryFrame(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,
        h: torch.Tensor,
        geometry_log_decay: torch.Tensor,
        key: torch.Tensor,
        erase: torch.Tensor,
        query: torch.Tensor,
        geometry_strength: torch.Tensor,
        initial_m: torch.Tensor,
        initial_J: torch.Tensor,
        initial_D: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        inputs = _validate_native_geometry_inputs(
            u,
            h,
            geometry_log_decay,
            key,
            erase,
            query,
            geometry_strength,
            initial_m,
            initial_J,
            initial_D,
        )
        (
            u,
            h,
            geometry_log_decay,
            key,
            erase,
            query,
            geometry_strength,
            initial_m,
            initial_J,
            initial_D,
        ) = inputs
        empty_s = torch.empty(0, device=u.device, dtype=torch.float32)
        boundary, final = _triton_geometry_chunk_scan_forward(
            u,
            h,
            geometry_log_decay,
            initial_state=SolveDeltaState(
                initial_m,
                initial_J,
                initial_D,
                empty_s,
            ),
            chunk_size=_CHUNK_SIZE,
            input_precision="ieee",
        )
        frame_inputs = (
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
        )
        _load_chunk_library()
        frame_outputs = _validate_forward_outputs(
            torch.ops.causallsso.c32_frame_resident_forward(*frame_inputs),
            frame_inputs,
        )
        ctx.save_for_backward(
            *inputs,
            boundary.m,
            boundary.J,
            boundary.D,
            *frame_outputs[3:],
        )
        ctx.set_materialize_grads(False)
        return (
            *frame_outputs[:3],
            final.m,
            final.J,
            final.D,
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        grad_d,
        grad_e,
        grad_chi,
        grad_final_m,
        grad_final_J,
        grad_final_D,
    ) -> tuple[torch.Tensor | None, ...]:
        from causallsso.ops.resident_frame import resident_c32_frame_backward

        saved = ctx.saved_tensors
        inputs = saved[: len(_GEOMETRY_INPUT_NAMES)]
        boundary_m, boundary_J, boundary_D = saved[
            len(_GEOMETRY_INPUT_NAMES) : len(_GEOMETRY_INPUT_NAMES) + 3
        ]
        auxiliaries = saved[len(_GEOMETRY_INPUT_NAMES) + 3 :]
        (
            u,
            h,
            geometry_log_decay,
            key,
            erase,
            query,
            geometry_strength,
            initial_m,
            initial_J,
            initial_D,
        ) = inputs
        grad_d = _output_cotangent("grad_d", grad_d, key)
        grad_e = _output_cotangent("grad_e", grad_e, key)
        grad_chi = _output_cotangent("grad_chi", grad_chi, query)
        grad_final_m = _fp32_cotangent(
            "grad_final_m", grad_final_m, initial_m
        )
        grad_final_J = _fp32_cotangent(
            "grad_final_J", grad_final_J, initial_J
        )
        grad_final_D = _fp32_cotangent(
            "grad_final_D", grad_final_D, initial_D
        )

        frame_gradients = resident_c32_frame_backward(
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
            *auxiliaries,
            grad_d,
            grad_e,
            grad_chi,
            retain_fp32_vector_partials=True,
        )
        scan_gradients = _triton_geometry_chunk_scan_backward(
            u,
            h,
            geometry_log_decay,
            boundary_m,
            boundary_J,
            boundary_D,
            frame_gradients[7],
            frame_gradients[8],
            frame_gradients[9],
            grad_final_m,
            grad_final_J,
            grad_final_D,
            _CHUNK_SIZE,
        )
        gradients = (
            (frame_gradients[0] + scan_gradients[0]).to(u.dtype),
            (frame_gradients[1] + scan_gradients[1]).to(h.dtype),
            frame_gradients[2] + scan_gradients[2],
            frame_gradients[3],
            frame_gradients[4],
            frame_gradients[5],
            frame_gradients[6],
            scan_gradients[3],
            scan_gradients[4],
            scan_gradients[5],
        )
        return tuple(
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad)
        )


def native_geometry_frame(
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    key: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    geometry_strength: torch.Tensor,
    *,
    initial_state: SolveDeltaState | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SolveDeltaState]:
    """Run the fixed C32 geometry scan and resident frame as one VJP owner."""
    if u.ndim != 4:
        raise ValueError("u must have shape [B,T,H,128]")
    batch, _, heads, rank = u.shape
    if initial_state is None:
        initial_m = torch.zeros(
            batch, heads, device=u.device, dtype=torch.float32
        )
        initial_J = torch.zeros(
            batch, heads, rank, rank, device=u.device, dtype=torch.float32
        )
        initial_D = torch.zeros_like(initial_J)
    else:
        if not isinstance(initial_state, SolveDeltaState):
            raise TypeError("initial_state must be a SolveDeltaState or None")
        initial_m = initial_state.m
        initial_J = initial_state.J
        initial_D = initial_state.D
    outputs = _NativeGeometryFrame.apply(
        u,
        h,
        geometry_log_decay,
        key,
        erase,
        query,
        geometry_strength,
        initial_m,
        initial_J,
        initial_D,
    )
    empty_s = torch.empty(0, device=u.device, dtype=torch.float32)
    final = SolveDeltaState(outputs[3], outputs[4], outputs[5], empty_s)
    return outputs[0], outputs[1], outputs[2], final


__all__ = ["native_geometry_frame"]
