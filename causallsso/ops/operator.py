"""Public BF16 CUDA entry for Residual-Frame SolveDelta."""

from __future__ import annotations

import torch
from fla.layers.utils import index_first_axis, index_put_first_axis

from ..reference import SolveDeltaState
from .packing import PackedSegments, pack_tokens, unpack_tokens
from .residual_frame import solvedelta_residual_frame_native


def solvedelta_native(
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
    return_final_state: bool = False,
) -> tuple[torch.Tensor, SolveDeltaState | None]:
    """Execute dense K=1 SolveDelta from raw fused-projection views."""
    if u.device.type != "cuda" or u.dtype != torch.bfloat16:
        raise TypeError("solvedelta_native requires BF16 CUDA activations")
    if keys.ndim != 5 or keys.shape[-2] != 1:
        raise ValueError("the current production path requires num_edits=1")
    if erase_raw.shape != keys.shape or erase_raw.dtype != torch.bfloat16:
        raise ValueError("erase_raw must be BF16 with the same shape as keys")
    if write_raw.shape != values.shape or write_raw.dtype != torch.bfloat16:
        raise ValueError("write_raw must be BF16 with the same shape as values")
    if any(
        operand.stride(-1) != 1
        for operand in (u, h, q, keys, values, erase_raw, write_raw)
    ):
        raise ValueError("native vector operands require unit innermost stride")

    output, final_state = solvedelta_residual_frame_native(
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
    return output, final_state


def _segment_batch(
    tensor: torch.Tensor,
    plan: PackedSegments,
    padded_length: int,
) -> torch.Tensor:
    actual = pack_tokens(tensor, plan).squeeze(0)
    position = torch.arange(
        plan.num_tokens, dtype=torch.long, device=tensor.device
    ) - plan.cu_seqlens[plan.segment_ids]
    destination = plan.segment_ids * padded_length + position
    rectangular = index_put_first_axis(
        actual,
        destination,
        plan.num_segments * padded_length,
    )
    return rectangular.view(
        plan.num_segments,
        padded_length,
        *tensor.shape[2:],
    )


def _restore_segment_output(
    output: torch.Tensor,
    plan: PackedSegments,
) -> torch.Tensor:
    padded_length = output.shape[1]
    position = torch.arange(
        plan.num_tokens, dtype=torch.long, device=output.device
    ) - plan.cu_seqlens[plan.segment_ids]
    source = plan.segment_ids * padded_length + position
    actual = index_first_axis(
        output.reshape(plan.num_segments * padded_length, *output.shape[2:]),
        source,
    )
    return unpack_tokens(actual.unsqueeze(0), plan)


def solvedelta_segmented_native(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase_raw: torch.Tensor,
    write_raw: torch.Tensor,
    geometry_write: torch.Tensor,
    plan: PackedSegments,
    *,
    initial_state: SolveDeltaState | None = None,
    return_final_state: bool = False,
) -> tuple[torch.Tensor, SolveDeltaState | None]:
    """Run reset-free packed segments as one neutral-padded native batch."""
    batch, _, heads, rank = u.shape
    value_dim = values.shape[-1]
    if (plan.batch, plan.length) != u.shape[:2]:
        raise ValueError("packed segment plan does not match the operator input")
    if plan.num_tokens == 0:
        output = values.new_zeros(batch, plan.length, heads, value_dim)
        if not return_final_state:
            return output, None
        if initial_state is not None:
            return output, initial_state
        return output, SolveDeltaState(
            predictor=torch.zeros(
                batch, heads, rank, rank, dtype=torch.float32, device=u.device
            ),
            S=torch.zeros(
                batch,
                heads,
                rank,
                value_dim,
                dtype=torch.float32,
                device=u.device,
            ),
        )

    padded_length = (plan.max_seqlen + 15) // 16 * 16
    segmented = tuple(
        _segment_batch(tensor, plan, padded_length)
        for tensor in (
            u,
            h,
            q,
            keys,
            values,
            associative_log_decay,
            erase_raw,
            write_raw,
            geometry_write,
        )
    )

    segment_initial = None
    if initial_state is not None:
        use_initial = plan.use_initial[..., None, None, None]
        # Multiple reset-delimited segments may belong to one batch row.
        # index_select has the required additive transpose for duplicate
        # segment_batch entries; FLA's index_first_axis assumes uniqueness.
        predictor = initial_state.predictor.index_select(
            0, plan.segment_batch
        )
        memory = initial_state.S.index_select(0, plan.segment_batch)
        segment_initial = SolveDeltaState(
            predictor=torch.where(use_initial, predictor, torch.zeros_like(predictor)),
            S=torch.where(use_initial, memory, torch.zeros_like(memory)),
        )

    segment_output, segment_final = solvedelta_native(
        *segmented,
        initial_state=segment_initial,
        return_final_state=return_final_state,
    )
    output = _restore_segment_output(segment_output, plan)
    if not return_final_state:
        return output, None
    if segment_final is None:
        raise RuntimeError("segmented native owners did not return final state")

    has_segment = plan.last_segment >= 0
    selected = plan.last_segment.clamp_min(0)
    predictor = segment_final.predictor.index_select(0, selected)
    memory = segment_final.S.index_select(0, selected)
    if initial_state is None:
        base_predictor = torch.zeros_like(predictor)
        base_memory = torch.zeros_like(memory)
    else:
        base_predictor = initial_state.predictor
        base_memory = initial_state.S
    return output, SolveDeltaState(
        predictor=torch.where(
            has_segment[..., None, None, None], predictor, base_predictor
        ),
        S=torch.where(has_segment[..., None, None, None], memory, base_memory),
    )


__all__ = ["solvedelta_native", "solvedelta_segmented_native"]
