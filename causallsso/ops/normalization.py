from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


_RANK = 128
_EPSILON = 1.0e-12


@triton.jit
def _normalize_three_forward_kernel(
    source_geometry,
    source_query,
    source_key,
    normalized_geometry,
    normalized_query,
    normalized_key,
    signed_inverse_norms,
    VECTORS: tl.constexpr,
    R: tl.constexpr,
    EPSILON: tl.constexpr,
):
    vector = tl.program_id(0).to(tl.int64)
    coordinate = tl.arange(0, R)
    for route in tl.static_range(3):
        source = tl.where(
            route == 0,
            source_geometry,
            tl.where(route == 1, source_query, source_key),
        )
        normalized = tl.where(
            route == 0,
            normalized_geometry,
            tl.where(route == 1, normalized_query, normalized_key),
        )
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
            signed_inverse_norms + route * VECTORS + vector,
            tl.where(active, inverse_norm, -inverse_norm),
        )


@triton.jit
def _normalize_three_backward_kernel(
    source_geometry,
    source_query,
    source_key,
    signed_inverse_norms,
    grad_normalized_geometry,
    grad_normalized_query,
    grad_normalized_key,
    grad_source_geometry,
    grad_source_query,
    grad_source_key,
    VECTORS: tl.constexpr,
    R: tl.constexpr,
    HAS_GEOMETRY: tl.constexpr,
    HAS_QUERY: tl.constexpr,
    HAS_KEY: tl.constexpr,
):
    vector = tl.program_id(0).to(tl.int64)
    coordinate = tl.arange(0, R)
    for route in tl.static_range(3):
        if ((route == 0 and HAS_GEOMETRY) or
            (route == 1 and HAS_QUERY) or
            (route == 2 and HAS_KEY)):
            source = tl.where(
                route == 0,
                source_geometry,
                tl.where(route == 1, source_query, source_key),
            )
            grad_normalized = tl.where(
                route == 0,
                grad_normalized_geometry,
                tl.where(
                    route == 1,
                    grad_normalized_query,
                    grad_normalized_key,
                ),
            )
            grad_source = tl.where(
                route == 0,
                grad_source_geometry,
                tl.where(route == 1, grad_source_query, grad_source_key),
            )
            values = tl.load(source + vector * R + coordinate).to(tl.float32)
            gradient = tl.load(
                grad_normalized + vector * R + coordinate
            ).to(tl.float32)
            signed_inverse = tl.load(
                signed_inverse_norms + route * VECTORS + vector
            )
            active = signed_inverse > 0.0
            inverse_norm = tl.abs(signed_inverse)
            projection = tl.sum(gradient * values, axis=0)
            active_gradient = (
                gradient * inverse_norm
                - values
                * projection
                * inverse_norm
                * inverse_norm
                * inverse_norm
            )
            result = tl.where(
                active,
                active_gradient,
                gradient * inverse_norm,
            )
            tl.store(
                grad_source + vector * R + coordinate,
                result.to(
                    grad_source.dtype.element_ty,
                    fp_downcast_rounding="rtne",
                ),
            )


class _NormalizeSolveDeltaInputs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        geometry_feature: torch.Tensor,
        query: torch.Tensor,
        edit_key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_geometry = torch.empty_like(
            geometry_feature, dtype=torch.float16
        )
        normalized_query = torch.empty_like(query, dtype=torch.float16)
        normalized_key = torch.empty_like(edit_key, dtype=torch.float16)
        vectors = geometry_feature.numel() // _RANK
        signed_inverse_norms = torch.empty(
            3,
            vectors,
            device=geometry_feature.device,
            dtype=torch.float32,
        )
        _normalize_three_forward_kernel[(vectors,)](
            geometry_feature,
            query,
            edit_key,
            normalized_geometry,
            normalized_query,
            normalized_key,
            signed_inverse_norms,
            VECTORS=vectors,
            R=_RANK,
            EPSILON=_EPSILON,
            num_warps=4,
            num_stages=1,
        )
        ctx.save_for_backward(
            geometry_feature,
            query,
            edit_key,
            signed_inverse_norms,
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
            signed_inverse_norms,
        ) = ctx.saved_tensors
        gradients = (grad_geometry, grad_query, grad_key)
        if all(gradient is None for gradient in gradients):
            return None, None, None
        sources = (geometry_feature, query, edit_key)
        grad_sources = tuple(
            torch.empty_like(source) if gradient is not None else None
            for source, gradient in zip(sources, gradients)
        )
        pointer_gradients = tuple(
            gradient.contiguous() if gradient is not None else geometry_feature
            for gradient in gradients
        )
        pointer_outputs = tuple(
            output if output is not None else geometry_feature
            for output in grad_sources
        )
        vectors = geometry_feature.numel() // _RANK
        _normalize_three_backward_kernel[(vectors,)](
            geometry_feature,
            query,
            edit_key,
            signed_inverse_norms,
            *pointer_gradients,
            *pointer_outputs,
            VECTORS=vectors,
            R=_RANK,
            HAS_GEOMETRY=grad_geometry is not None,
            HAS_QUERY=grad_query is not None,
            HAS_KEY=grad_key is not None,
            num_warps=4,
            num_stages=1,
        )
        return grad_sources


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
