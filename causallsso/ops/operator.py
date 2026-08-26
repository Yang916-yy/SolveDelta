from __future__ import annotations

import torch

from ..reference import SolveDeltaState
from .exterior import chunk_wy_exterior
from .frame import bounded_frame_panels
from .packing import (
    PackedSegments,
    build_packed_segments,
    pack_tokens,
    unpack_tokens,
)


def _all_valid_solvedelta(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase_raw: torch.Tensor,
    write_raw: torch.Tensor,
    geometry_strength: torch.Tensor,
    *,
    initial_state: SolveDeltaState | None,
    return_final_state: bool,
    chunk_size: int,
    lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, SolveDeltaState | None]:
    frame = bounded_frame_panels(
        u,
        h,
        q,
        keys,
        geometry_log_decay,
        erase_raw,
        geometry_strength,
        initial_state=initial_state,
        lengths=lengths,
        chunk_size=chunk_size,
        exterior_dtype=torch.bfloat16,
    )
    batch, length, heads, edits, width = keys.shape
    value_width = values.shape[-1]

    state_s = initial_state.S if initial_state is not None else None
    output, final_s = chunk_wy_exterior(
        frame.d,
        frame.paired_dual,
        values,
        write_raw,
        associative_log_decay,
        initial_state=state_s,
        output_final_state=return_final_state,
        chunk_size=chunk_size,
    )
    if edits != 1:
        output = output.view(batch, length, edits, heads, value_width)[:, :, -1]
    if not return_final_state:
        return output.to(torch.bfloat16), None
    if final_s is None:
        raise RuntimeError("FLA did not return the requested associative state")
    state = SolveDeltaState(frame.m, frame.J, frame.D, final_s.float())
    return output.to(torch.bfloat16), state


def _zero_state(
    *,
    batch: int,
    heads: int,
    width: int,
    value_width: int,
    device: torch.device,
) -> SolveDeltaState:
    return SolveDeltaState(
        torch.zeros(batch, heads, dtype=torch.float32, device=device),
        torch.zeros(batch, heads, width, width, dtype=torch.float32, device=device),
        torch.zeros(batch, heads, width, width, dtype=torch.float32, device=device),
        torch.zeros(batch, heads, width, value_width, dtype=torch.float32, device=device),
    )


def _packed_segment_solvedelta(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase_raw: torch.Tensor,
    write_raw: torch.Tensor,
    geometry_strength: torch.Tensor,
    *,
    initial_state: SolveDeltaState | None,
    plan: PackedSegments,
    return_final_state: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, SolveDeltaState | None]:
    """Compact reset-free valid segments into FLA's variable-length schedule."""
    batch, length, heads, width = u.shape
    value_width = values.shape[-1]
    if initial_state is None:
        base_state = _zero_state(
            batch=batch,
            heads=heads,
            width=width,
            value_width=value_width,
            device=u.device,
        )
    else:
        base_state = SolveDeltaState(
            initial_state.m,
            0.5 * (initial_state.J + initial_state.J.transpose(-1, -2)),
            initial_state.D,
            initial_state.S,
        )
    if plan.num_tokens == 0:
        output = torch.zeros(
            batch,
            length,
            heads,
            value_width,
            dtype=torch.bfloat16,
            device=u.device,
        )
        return output, base_state if return_final_state else None

    def pack_initial(tensor: torch.Tensor) -> torch.Tensor:
        # A batch may own multiple reset-delimited segments. Unlike the token
        # gather, this index is not unique and its transpose must accumulate.
        selected = tensor.index_select(0, plan.segment_batch)
        shape = (plan.num_segments,) + (1,) * (selected.ndim - 1)
        return torch.where(
            plan.use_initial.view(shape), selected, torch.zeros_like(selected)
        )

    from fla.ops.utils import prepare_chunk_indices

    frame_chunk_indices = prepare_chunk_indices(
        plan.cu_seqlens,
        chunk_size,
        cu_seqlens_cpu=plan.cu_seqlens_cpu,
    )
    packed_initial = (
        None
        if initial_state is None
        else SolveDeltaState(*(pack_initial(tensor) for tensor in base_state))
    )
    frame = bounded_frame_panels(
        pack_tokens(u, plan),
        pack_tokens(h, plan),
        pack_tokens(q, plan),
        pack_tokens(keys, plan),
        pack_tokens(geometry_log_decay, plan),
        pack_tokens(erase_raw, plan),
        geometry_strength,
        initial_state=packed_initial,
        lengths=plan.segment_lengths,
        cu_seqlens=plan.cu_seqlens,
        cu_seqlens_cpu=plan.cu_seqlens_cpu,
        chunk_indices=frame_chunk_indices,
        chunk_size=chunk_size,
        exterior_dtype=torch.bfloat16,
    )

    packed_output, final_s = chunk_wy_exterior(
        frame.d,
        frame.paired_dual,
        pack_tokens(values, plan),
        pack_tokens(write_raw, plan),
        pack_tokens(associative_log_decay, plan),
        initial_state=packed_initial.S if packed_initial is not None else None,
        cu_seqlens=plan.cu_seqlens,
        cu_seqlens_cpu=plan.cu_seqlens_cpu,
        output_final_state=return_final_state,
        chunk_size=chunk_size,
    )
    edits = keys.shape[-2]
    if edits != 1:
        packed_output = packed_output.view(
            1, plan.num_tokens, edits, heads, value_width
        )[:, :, -1]
    output = unpack_tokens(packed_output, plan)
    if not return_final_state:
        return output, None
    if final_s is None:
        raise RuntimeError("packed SolveDelta did not return final state")
    packed_final = SolveDeltaState(frame.m, frame.J, frame.D, final_s.float())
    has_segment = plan.last_segment >= 0
    selected_index = plan.last_segment.clamp_min(0)
    final_state = SolveDeltaState(
        *(
            torch.where(
                has_segment.view((batch,) + (1,) * (base.ndim - 1)),
                packed.index_select(0, selected_index),
                base,
            )
            for packed, base in zip(packed_final, base_state)
        )
    )
    return output, final_state


def solvedelta_native(
    u: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase_raw: torch.Tensor,
    write_raw: torch.Tensor,
    geometry_strength: torch.Tensor,
    *,
    initial_state: SolveDeltaState | None = None,
    valid_mask: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
    _packed_segments: PackedSegments | None = None,
    return_final_state: bool = False,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, SolveDeltaState | None]:
    """Execute the production graph from BF16 activations and raw gate logits."""
    if u.device.type != "cuda" or u.dtype != torch.bfloat16:
        raise TypeError("solvedelta_native requires BF16 CUDA activations")
    batch, length, heads, width = u.shape
    if h.shape != u.shape or q.shape != u.shape:
        raise ValueError("u, h, and q must share [B,T,H,r]")
    if erase_raw.shape != keys.shape or erase_raw.dtype != torch.bfloat16:
        raise ValueError("erase_raw must be BF16 with the same shape as keys")
    if write_raw.shape != values.shape or write_raw.dtype != torch.bfloat16:
        raise ValueError("write_raw must be BF16 with the same shape as values")
    if chunk_size not in (16, 32, 64):
        raise ValueError("chunk_size must be one of 16, 32, or 64")
    if initial_state is not None:
        if initial_state.m.dtype != torch.float32:
            raise TypeError("continuation states must be FP32")
        if not torch.equal(initial_state.J, initial_state.J.transpose(-1, -2)):
            raise ValueError("initial_state.J must be exactly symmetric")

    masks_supplied = valid_mask is not None or reset_mask is not None
    if not masks_supplied:
        if _packed_segments is not None:
            raise ValueError("packed metadata requires explicit masks")
        return _all_valid_solvedelta(
            u,
            h,
            q,
            keys,
            values,
            geometry_log_decay,
            associative_log_decay,
            erase_raw,
            write_raw,
            geometry_strength,
            initial_state=initial_state,
            return_final_state=return_final_state,
            chunk_size=chunk_size,
        )

    if valid_mask is None:
        valid_mask = torch.ones(batch, length, dtype=torch.bool, device=u.device)
    if reset_mask is None:
        reset_mask = torch.zeros(batch, length, dtype=torch.bool, device=u.device)
    if valid_mask.shape != (batch, length) or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool [B,T]")
    if reset_mask.shape != (batch, length) or reset_mask.dtype != torch.bool:
        raise ValueError("reset_mask must be bool [B,T]")

    if valid_mask.device != u.device or reset_mask.device != u.device:
        raise ValueError("masks must share the activation device")
    plan = (
        build_packed_segments(valid_mask, reset_mask)
        if _packed_segments is None
        else _packed_segments
    )
    if plan.batch != batch or plan.length != length:
        raise ValueError("packed metadata does not match the activation shape")

    return _packed_segment_solvedelta(
        u,
        h,
        q,
        keys,
        values,
        geometry_log_decay,
        associative_log_decay,
        erase_raw,
        write_raw,
        geometry_strength,
        initial_state=initial_state,
        plan=plan,
        return_final_state=return_final_state,
        chunk_size=chunk_size,
    )


__all__ = ["solvedelta_native"]
