from __future__ import annotations

from pathlib import Path

import torch
from torch.autograd.function import once_differentiable


_CHUNK_SIZE = 32
_RANK = 128
_EDITS = 1
_LOADED = False

_INPUT_NAMES = (
    "u",
    "h",
    "geometry_log_decay",
    "key",
    "erase",
    "query",
    "geometry_strength",
    "boundary_m",
    "boundary_J",
    "boundary_D",
)


def _library_candidates() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[2]
    return (
        root / "build" / "native" / "libcausallsso_chunk.so",
        root / "build" / "libcausallsso_chunk.so",
    )


def _ops_registered() -> bool:
    try:
        torch.ops.causallsso.c32_frame_forward
        torch.ops.causallsso.c32_frame_backward
    except AttributeError:
        return False
    return True


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
                f"{path} does not register c32_frame_forward and "
                "c32_frame_backward"
            )
        _LOADED = True
        return

    locations = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        "the SolveDelta C32 native library is not built; expected "
        f"libcausallsso_chunk.so at one of: {locations}"
    )


def _validate_native_chunk_inputs(
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
) -> None:
    inputs = (
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
    )
    for name, tensor in zip(_INPUT_NAMES, inputs):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if u.ndim != 4:
        raise ValueError("u must have shape [B,T,H,128]")
    batch, length, heads, rank = u.shape
    if batch < 1 or length < 1 or heads < 1:
        raise ValueError("B, T, and H must be positive")
    if rank != _RANK:
        raise ValueError("the native chunk frame requires r=128")

    chunks = (length + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    expected_shapes = {
        "h": (batch, length, heads, rank),
        "geometry_log_decay": (batch, length, heads),
        "key": (batch, length, heads, _EDITS, rank),
        "erase": (batch, length, heads, _EDITS, rank),
        "query": (batch, length, heads, rank),
        "geometry_strength": (heads,),
        "boundary_m": (batch, heads, chunks),
        "boundary_J": (batch, heads, chunks, rank, rank),
        "boundary_D": (batch, heads, chunks, rank, rank),
    }
    named_inputs = dict(zip(_INPUT_NAMES, inputs))
    for name, shape in expected_shapes.items():
        if named_inputs[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}")

    bad_dtypes = [
        f"{name}={tensor.dtype}"
        for name, tensor in zip(_INPUT_NAMES, inputs)
        if tensor.dtype != torch.float32
    ]
    if bad_dtypes:
        raise TypeError(
            "the native C32 geometry/frame ABI requires FP32 inputs; got "
            + ", ".join(bad_dtypes)
        )

    device = u.device
    if any(tensor.device != device for tensor in inputs):
        raise ValueError("all native chunk frame inputs must share one device")
    if device.type != "cuda":
        raise ValueError("the native chunk frame requires CUDA tensors")


def _validate_forward_outputs(
    outputs: object,
    inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
        raise RuntimeError(
            "c32_frame_forward must return "
            "(d,e,chi,lower_primal,lower_dual_scaled)"
        )
    tensors = tuple(outputs)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise RuntimeError("c32_frame_forward returned a non-Tensor value")

    u, _, _, key, _, query, _, _, _, _ = inputs
    batch, length, heads, rank = u.shape
    expected_shapes = (
        key.shape,
        key.shape,
        query.shape,
        (batch, length, heads, _EDITS, rank),
        (batch, length, heads, 2, rank),
    )
    names = (
        "d",
        "e",
        "chi",
        "lower_primal",
        "lower_dual_scaled",
    )
    for name, tensor, shape in zip(names, tensors, expected_shapes):
        if tensor.shape != shape:
            raise RuntimeError(
                f"c32_frame_forward returned {name} with shape "
                f"{tuple(tensor.shape)}; expected {shape}"
            )
        if tensor.dtype != torch.float32 or tensor.device != u.device:
            raise RuntimeError(
                f"c32_frame_forward returned {name} outside the FP32 CUDA ABI"
            )
    return tensors


def _validate_output_cotangent(
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
        raise RuntimeError(f"{name} must match the FP32 CUDA frame output")
    return gradient.contiguous()


def _validate_backward_outputs(
    outputs: object,
    inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != len(inputs):
        raise RuntimeError(
            "c32_frame_backward must return one gradient for each forward tensor "
            "input"
        )
    gradients = tuple(outputs)
    if not all(isinstance(gradient, torch.Tensor) for gradient in gradients):
        raise RuntimeError("c32_frame_backward returned a non-Tensor value")
    for name, gradient, reference in zip(_INPUT_NAMES, gradients, inputs):
        if gradient.shape != reference.shape:
            raise RuntimeError(
                f"c32_frame_backward returned d{name} with shape "
                f"{tuple(gradient.shape)}; expected {tuple(reference.shape)}"
            )
        if gradient.dtype != torch.float32 or gradient.device != reference.device:
            raise RuntimeError(
                f"c32_frame_backward returned d{name} outside the FP32 CUDA ABI"
            )
    return gradients


class _NativeChunkFrame(torch.autograd.Function):
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
        boundary_m: torch.Tensor,
        boundary_J: torch.Tensor,
        boundary_D: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_native_chunk_inputs(
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
        )
        _load_chunk_library()

        inputs = tuple(
            tensor.contiguous()
            for tensor in (
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
            )
        )
        outputs = _validate_forward_outputs(
            torch.ops.causallsso.c32_frame_forward(*inputs),
            inputs,
        )
        d, e, chi, lower_primal, lower_dual_scaled = outputs
        ctx.save_for_backward(
            *inputs,
            lower_primal,
            lower_dual_scaled,
            d,
        )
        ctx.set_materialize_grads(False)
        return d, e, chi

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_d, grad_e, grad_chi):
        saved = ctx.saved_tensors
        inputs = saved[: len(_INPUT_NAMES)]
        lower_primal, lower_dual_scaled, d = saved[len(_INPUT_NAMES) :]

        grad_d = _validate_output_cotangent("grad_d", grad_d, d)
        grad_e = _validate_output_cotangent("grad_e", grad_e, inputs[3])
        grad_chi = _validate_output_cotangent("grad_chi", grad_chi, inputs[5])
        gradients = _validate_backward_outputs(
            torch.ops.causallsso.c32_frame_backward(
                *inputs,
                lower_primal,
                lower_dual_scaled,
                d,
                grad_d,
                grad_e,
                grad_chi,
            ),
            inputs,
        )
        return tuple(
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad)
        )


def native_chunk_frame(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the exact CUDA C32, K=1, r=128 frame specialization."""
    return _NativeChunkFrame.apply(
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
    )


__all__ = ["native_chunk_frame"]
