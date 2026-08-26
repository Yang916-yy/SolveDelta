from __future__ import annotations

import torch

from .direct_e import chunk_direct_e_delta_rule


def chunk_wy_exterior(
    d: torch.Tensor,
    paired_dual: torch.Tensor,
    values: torch.Tensor,
    write_raw: torch.Tensor,
    associative_log_decay: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    output_final_state: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply the direct-e associative exterior with FLA's chunk schedules."""
    if d.ndim != 4 or paired_dual.ndim != 4:
        raise ValueError("frame/exterior boundary must use resident panel storage")
    if values.ndim != 5 or write_raw.shape != values.shape:
        raise ValueError("values and write_raw must share [B,T,H,K,d_v]")
    if d.device.type != "cuda":
        raise ValueError("the native chunk-WY exterior requires CUDA tensors")
    if d.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("frame panels must be FP16 or BF16")
    if values.dtype != d.dtype:
        raise TypeError(
            "FLA 0.5.2 DPLR requires frame panels and values to share one dtype"
        )
    if associative_log_decay.dtype != torch.float32:
        raise TypeError("associative log decay must be FP32")
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError("associative continuation state must be FP32")

    output, final_state = chunk_direct_e_delta_rule(
        d=d,
        paired_dual=paired_dual,
        values=values,
        write_raw=write_raw,
        g=associative_log_decay,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )
    return output, final_state


__all__ = ["chunk_wy_exterior"]
