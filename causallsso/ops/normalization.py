from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


_RANK = 128
_EPSILON = 1.0e-12


@triton.jit
def _normalize_forward_kernel(
    source,
    normalized,
    signed_inverse_norm,
    R: tl.constexpr,
    EPSILON: tl.constexpr,
):
    vector = tl.program_id(0).to(tl.int64)
    coordinate = tl.arange(0, R)
    values = tl.load(source + vector * R + coordinate).to(tl.float32)
    norm = tl.sqrt(tl.sum(values * values, axis=0))
    active = norm > EPSILON
    inverse_norm = tl.where(active, 1.0 / norm, 1.0 / EPSILON)
    tl.store(
        normalized + vector * R + coordinate,
        (values * inverse_norm).to(
            normalized.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
    )
    tl.store(
        signed_inverse_norm + vector,
        tl.where(active, inverse_norm, -inverse_norm),
    )


@triton.jit
def _normalize_backward_kernel(
    source,
    signed_inverse_norm,
    grad_normalized,
    grad_source,
    R: tl.constexpr,
):
    vector = tl.program_id(0).to(tl.int64)
    coordinate = tl.arange(0, R)
    values = tl.load(source + vector * R + coordinate).to(tl.float32)
    gradient = tl.load(
        grad_normalized + vector * R + coordinate
    ).to(tl.float32)
    signed_inverse = tl.load(signed_inverse_norm + vector)
    active = signed_inverse > 0.0
    inverse_norm = tl.abs(signed_inverse)
    projection = tl.sum(gradient * values, axis=0)
    active_gradient = (
        gradient * inverse_norm
        - values * projection * inverse_norm * inverse_norm * inverse_norm
    )
    result = tl.where(active, active_gradient, gradient * inverse_norm)
    tl.store(
        grad_source + vector * R + coordinate,
        result.to(
            grad_source.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
    )


def _normalize_forward(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = torch.empty_like(source, dtype=torch.float16)
    vectors = source.numel() // _RANK
    signed_inverse_norm = torch.empty(
        vectors,
        device=source.device,
        dtype=torch.float32,
    )
    _normalize_forward_kernel[(vectors,)](
        source,
        normalized,
        signed_inverse_norm,
        R=_RANK,
        EPSILON=_EPSILON,
        num_warps=4,
        num_stages=1,
    )
    return normalized, signed_inverse_norm


def _normalize_backward(
    source: torch.Tensor,
    signed_inverse_norm: torch.Tensor,
    gradient: torch.Tensor | None,
) -> torch.Tensor | None:
    if gradient is None:
        return None
    grad_source = torch.empty_like(source)
    vectors = source.numel() // _RANK
    _normalize_backward_kernel[(vectors,)](
        source,
        signed_inverse_norm,
        gradient.contiguous(),
        grad_source,
        R=_RANK,
        num_warps=4,
        num_stages=1,
    )
    return grad_source


class _NormalizeSolveDeltaInputs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        geometry_feature: torch.Tensor,
        query: torch.Tensor,
        edit_key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_geometry, inverse_geometry = _normalize_forward(
            geometry_feature
        )
        normalized_query, inverse_query = _normalize_forward(query)
        normalized_key, inverse_key = _normalize_forward(edit_key)
        ctx.save_for_backward(
            geometry_feature,
            query,
            edit_key,
            inverse_geometry,
            inverse_query,
            inverse_key,
        )
        ctx.set_materialize_grads(False)
        return normalized_geometry, normalized_query, normalized_key

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_geometry: torch.Tensor | None,
        grad_query: torch.Tensor | None,
        grad_key: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        (
            geometry_feature,
            query,
            edit_key,
            inverse_geometry,
            inverse_query,
            inverse_key,
        ) = ctx.saved_tensors
        return (
            _normalize_backward(
                geometry_feature, inverse_geometry, grad_geometry
            ),
            _normalize_backward(query, inverse_query, grad_query),
            _normalize_backward(edit_key, inverse_key, grad_key),
        )


def normalize_solvedelta_inputs(
    geometry_feature: torch.Tensor,
    query: torch.Tensor,
    edit_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize BF16 inputs directly into certified FP16 private panels."""
    for name, tensor in (
        ("geometry_feature", geometry_feature),
        ("query", query),
        ("edit_key", edit_key),
    ):
        if tensor.shape[-1] != _RANK:
            raise ValueError(f"{name} must have trailing dimension 128")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be BF16")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.device != geometry_feature.device:
            raise ValueError("normalization inputs must share one CUDA device")
    if query.numel() != geometry_feature.numel():
        raise ValueError("query must contain one vector per geometry feature")
    if edit_key.numel() != geometry_feature.numel():
        raise ValueError("K=1 edit_key must contain one vector per token")
    return _NormalizeSolveDeltaInputs.apply(
        geometry_feature.contiguous(),
        query.contiguous(),
        edit_key.contiguous(),
    )


__all__ = ["normalize_solvedelta_inputs"]
