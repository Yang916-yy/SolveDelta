from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fla.layers.utils import index_first_axis, pad_input


@dataclass(frozen=True)
class PackedSegments:
    """One reset-delimited packing plan shared by conv, frame, and WY."""

    batch: int
    length: int
    indices: torch.Tensor
    segment_ids: torch.Tensor
    segment_batch: torch.Tensor
    use_initial: torch.Tensor
    segment_lengths: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor
    segment_batch_cpu: torch.Tensor
    use_initial_cpu: torch.Tensor
    last_segment: torch.Tensor

    @property
    def num_tokens(self) -> int:
        return self.indices.shape[0]

    @property
    def num_segments(self) -> int:
        return self.segment_batch.shape[0]

    @property
    def max_seqlen(self) -> int:
        if self.num_segments == 0:
            return 0
        return int(torch.diff(self.cu_seqlens_cpu).max().item())


def build_packed_segments(
    valid_mask: torch.Tensor,
    reset_mask: torch.Tensor,
) -> PackedSegments:
    """Build reset-free segment metadata and one shared CPU mirror.

    Dynamic compaction necessarily discovers an output size on the host. The
    small metadata transfer produced here is then reused by every FLA utility
    so none of them independently reads CUDA lengths back to the host.
    """
    if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool [B,T]")
    if reset_mask.shape != valid_mask.shape or reset_mask.dtype != torch.bool:
        raise ValueError("reset_mask must be bool [B,T]")
    if reset_mask.device != valid_mask.device:
        raise ValueError("valid_mask and reset_mask must share one device")

    batch, length = valid_mask.shape
    indices = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).flatten()
    empty_long = torch.empty(0, dtype=torch.long, device=valid_mask.device)
    if indices.shape[0] == 0:
        cu_cpu = torch.zeros(1, dtype=torch.long, device="cpu")
        return PackedSegments(
            batch=batch,
            length=length,
            indices=indices,
            segment_ids=empty_long,
            segment_batch=empty_long,
            use_initial=torch.empty(0, dtype=torch.bool, device=valid_mask.device),
            segment_lengths=torch.empty(
                0, dtype=torch.int32, device=valid_mask.device
            ),
            cu_seqlens=cu_cpu.to(valid_mask.device),
            cu_seqlens_cpu=cu_cpu,
            segment_batch_cpu=torch.empty(0, dtype=torch.long, device="cpu"),
            use_initial_cpu=torch.empty(0, dtype=torch.bool, device="cpu"),
            last_segment=torch.full(
                (batch,), -1, dtype=torch.long, device=valid_mask.device
            ),
        )

    batch_index = torch.div(indices, length, rounding_mode="floor")
    reset_valid = index_first_axis(
        reset_mask.reshape(-1, 1), indices
    ).squeeze(-1)
    starts = torch.cat(
        (
            torch.ones(1, dtype=torch.bool, device=indices.device),
            (batch_index[1:] != batch_index[:-1]) | reset_valid[1:],
        )
    )
    segment_ids = starts.cumsum(0, dtype=torch.long) - 1
    start_positions = torch.nonzero(starts, as_tuple=False).flatten()
    segment_batch = index_first_axis(
        batch_index[:, None], start_positions
    ).squeeze(-1)
    segment_reset = index_first_axis(
        reset_valid[:, None], start_positions
    ).squeeze(-1)
    end_positions = torch.cat(
        (
            start_positions[1:],
            start_positions.new_full((1,), indices.shape[0]),
        )
    )
    segment_lengths = (end_positions - start_positions).to(torch.int32)
    first_for_batch = torch.cat(
        (
            torch.ones(1, dtype=torch.bool, device=indices.device),
            segment_batch[1:] != segment_batch[:-1],
        )
    )
    use_initial = first_for_batch & ~segment_reset

    segment_number = torch.arange(
        segment_batch.shape[0], dtype=torch.long, device=indices.device
    )
    last_segment = torch.full(
        (batch,), -1, dtype=torch.long, device=indices.device
    ).scatter_reduce(
        0,
        segment_batch,
        segment_number,
        reduce="amax",
        include_self=True,
    )

    host_metadata = torch.stack(
        (segment_lengths.long(), segment_batch, use_initial.long()), dim=-1
    ).cpu()
    lengths_cpu = host_metadata[:, 0]
    segment_batch_cpu = host_metadata[:, 1]
    use_initial_cpu = host_metadata[:, 2].bool()
    cu_cpu = F.pad(lengths_cpu.cumsum(0), (1, 0), value=0)
    return PackedSegments(
        batch=batch,
        length=length,
        indices=indices,
        segment_ids=segment_ids,
        segment_batch=segment_batch,
        use_initial=use_initial,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_cpu.to(valid_mask.device),
        cu_seqlens_cpu=cu_cpu,
        segment_batch_cpu=segment_batch_cpu,
        use_initial_cpu=use_initial_cpu,
        last_segment=last_segment,
    )


def pack_tokens(tensor: torch.Tensor, plan: PackedSegments) -> torch.Tensor:
    if tensor.shape[:2] != (plan.batch, plan.length):
        raise ValueError("packed tensor must share the plan's [B,T] prefix")
    flat = tensor.reshape(plan.batch * plan.length, *tensor.shape[2:])
    return index_first_axis(flat, plan.indices).unsqueeze(0)


def unpack_tokens(tensor: torch.Tensor, plan: PackedSegments) -> torch.Tensor:
    if tensor.ndim < 2 or tensor.shape[0] != 1:
        raise ValueError("packed tensor must have a singleton batch axis")
    return pad_input(
        tensor.squeeze(0), plan.indices, plan.batch, plan.length
    )


__all__ = [
    "PackedSegments",
    "build_packed_segments",
    "pack_tokens",
    "unpack_tokens",
]
