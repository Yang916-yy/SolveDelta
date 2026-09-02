"""Selected dense Residual-Frame SolveDelta production composition."""

from __future__ import annotations

import torch

from ...reference import SolveDeltaState
from .exterior import _backward as exterior_backward
from .exterior import _forward as exterior_forward
from .l2norm import strided_l2norm
from .predictor import oja_residual
from .recurrent import solvedelta_recurrent_inference
from .sources import relative_sources_backward, relative_sources_forward


PREDICTOR_CHUNK_SIZE = 32
EXTERIOR_CHUNK_SIZE = 16


class _RelativeExterior(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        direct,
        u,
        update,
        q,
        key,
        value,
        erase_raw,
        write_raw,
        log_decay,
        initial_state,
        chunk_size,
        output_final_state,
    ):
        paired, injection = relative_sources_forward(
            u,
            update,
            q,
            key,
            value,
            erase_raw,
            write_raw,
            chunk_size=chunk_size,
        )
        output, state_out, _ = exterior_forward(
            direct,
            paired,
            injection,
            log_decay,
            initial_state,
            chunk_size,
            final_state=output_final_state,
        )
        ctx.chunk_size = chunk_size
        ctx.has_initial_state = initial_state is not None
        ctx.set_materialize_grads(False)
        saved_initial = u.new_empty(0) if initial_state is None else initial_state
        ctx.save_for_backward(
            direct,
            u,
            update,
            q,
            key,
            value,
            erase_raw,
            write_raw,
            log_decay,
            saved_initial,
        )
        return output, state_out

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        (
            direct,
            u,
            update,
            q,
            key,
            value,
            erase_raw,
            write_raw,
            log_decay,
            saved_initial,
        ) = ctx.saved_tensors
        initial_state = saved_initial if ctx.has_initial_state else None
        paired, injection = relative_sources_forward(
            u,
            update,
            q,
            key,
            value,
            erase_raw,
            write_raw,
            chunk_size=ctx.chunk_size,
        )
        (
            grad_direct,
            grad_paired,
            grad_injection,
            grad_decay,
            grad_initial,
        ) = exterior_backward(
            direct,
            paired,
            injection,
            log_decay,
            initial_state,
            grad_output,
            grad_final_state,
            ctx.chunk_size,
            # These are final-shaped source cotangents, not cross-owner
            # reduction partials. Both adjacent owners accumulate in FP32;
            # BF16 keeps exponent range while halving this HBM handoff.
            panel_gradient_dtype=torch.bfloat16,
        )
        source_gradients = relative_sources_backward(
            u,
            update,
            q,
            key,
            value,
            erase_raw,
            write_raw,
            grad_paired,
            grad_injection,
            chunk_size=ctx.chunk_size,
        )
        return (
            grad_direct,
            *source_gradients,
            grad_decay,
            grad_initial,
            None,
            None,
        )


def _pack_primal(direct: torch.Tensor, chunk_size: int) -> torch.Tensor:
    batch, length, heads, rank = direct.shape
    if length % chunk_size:
        raise ValueError("primal panel packing requires an aligned sequence")
    chunks = length // chunk_size
    return (
        direct.view(batch, chunks, chunk_size, heads, rank)
        .permute(0, 3, 1, 2, 4)
        .reshape(batch * heads * chunks, 1, chunk_size, rank)
        .contiguous()
    )


def solvedelta_residual_frame_native(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase_raw: torch.Tensor,
    write_raw: torch.Tensor,
    geometry_write: torch.Tensor,
    *,
    initial_state: SolveDeltaState | None = None,
    return_final_state: bool = True,
) -> tuple[torch.Tensor, SolveDeltaState | None]:
    """Execute the BF16 relative-frame path from raw gate logits."""
    if u.ndim != 4:
        raise ValueError("u must have shape [B,T,H,r]")
    batch, length, heads, rank = u.shape
    if h.shape != u.shape or q.shape != u.shape:
        raise ValueError("h and q must match u")
    if keys.shape != (batch, length, heads, 1, rank):
        raise ValueError("Residual-Frame SolveDelta requires keys [B,T,H,1,r]")
    if values.ndim != 5 or values.shape[:4] != (batch, length, heads, 1):
        raise ValueError("values must have shape [B,T,H,1,d_v]")
    value_dim = values.shape[-1]
    if associative_log_decay.shape != (batch, length, heads, rank):
        raise ValueError("associative_log_decay must have shape [B,T,H,r]")
    if erase_raw.shape != keys.shape or write_raw.shape != values.shape:
        raise ValueError("erase_raw/write_raw shapes must match keys/values")
    if geometry_write.shape not in (
        (heads,),
        (1, heads),
        (batch, length, heads),
    ):
        raise ValueError("geometry_write must have shape [H], [1,H], or [B,T,H]")
    if u.device.type != "cuda" or u.dtype != torch.bfloat16:
        raise TypeError("the production Residual-Frame path requires BF16 CUDA operands")
    if any(x.dtype != u.dtype for x in (h, q, keys, values, erase_raw, write_raw)):
        raise TypeError("all public vector operands must be BF16")
    if associative_log_decay.dtype != torch.float32:
        raise TypeError("associative_log_decay must be FP32")
    if geometry_write.dtype != torch.float32:
        raise TypeError("geometry_write must be FP32")

    if (
        length == 1
        and rank <= 128
        and value_dim <= 128
        and not torch.is_grad_enabled()
    ):
        return solvedelta_recurrent_inference(
            u,
            h,
            q,
            keys,
            values,
            associative_log_decay,
            erase_raw,
            write_raw,
            geometry_write,
            initial_state=initial_state,
            return_final_state=return_final_state,
        )

    tail = length % EXTERIOR_CHUNK_SIZE
    if tail:
        padding = EXTERIOR_CHUNK_SIZE - tail

        def pad_tokens(tensor: torch.Tensor) -> torch.Tensor:
            shape = list(tensor.shape)
            shape[1] = padding
            return torch.cat((tensor, tensor.new_zeros(shape)), dim=1)

        padded_geometry_write = (
            pad_tokens(geometry_write)
            if geometry_write.shape == (batch, length, heads)
            else geometry_write
        )
        padded_output, padded_state = solvedelta_residual_frame_native(
            pad_tokens(u),
            pad_tokens(h),
            pad_tokens(q),
            pad_tokens(keys),
            pad_tokens(values),
            pad_tokens(associative_log_decay),
            pad_tokens(erase_raw),
            pad_tokens(write_raw),
            padded_geometry_write,
            initial_state=initial_state,
            return_final_state=return_final_state,
        )
        return padded_output[:, :length], padded_state

    if initial_state is None:
        predictor_initial = None
        memory_initial = None
    else:
        expected_shapes = (
            (batch, heads, rank, rank),
            (batch, heads, rank, value_dim),
        )
        if any(
            value.shape != expected
            for value, expected in zip(initial_state, expected_shapes)
        ):
            raise ValueError("initial_state shapes do not match the input geometry")
        if any(value.dtype != torch.float32 for value in initial_state):
            raise TypeError("all continuation states must be FP32")
        predictor_initial, memory_initial = initial_state

    u_panel = strided_l2norm(u)
    key_panel = strided_l2norm(keys.squeeze(-2))
    if geometry_write.shape != (batch, length, heads):
        geometry_write = geometry_write.reshape(1, 1, heads).expand(
            batch, length, heads
        )

    frame_action, update, final_predictor = oja_residual(
        h,
        u_panel,
        geometry_write,
        associative_log_decay,
        key_panel,
        predictor_initial,
        chunk_size=PREDICTOR_CHUNK_SIZE,
        output_final_state=return_final_state,
    )
    direct = _pack_primal(key_panel + frame_action, EXTERIOR_CHUNK_SIZE)
    output, final_memory = _RelativeExterior.apply(
        direct,
        u_panel,
        update,
        q,
        key_panel,
        values.squeeze(-2),
        erase_raw.squeeze(-2),
        write_raw.squeeze(-2),
        associative_log_decay,
        memory_initial,
        EXTERIOR_CHUNK_SIZE,
        return_final_state,
    )
    output = output.to(torch.bfloat16)
    if not return_final_state:
        return output, None
    if final_predictor is None or final_memory is None:
        raise RuntimeError("native state owners did not return requested endpoints")
    return output, SolveDeltaState(
        predictor=final_predictor.float(),
        S=final_memory.float(),
    )


__all__ = [
    "EXTERIOR_CHUNK_SIZE",
    "PREDICTOR_CHUNK_SIZE",
    "solvedelta_residual_frame_native",
]
