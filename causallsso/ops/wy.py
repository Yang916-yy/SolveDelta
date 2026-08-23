from __future__ import annotations

import torch


def _validate_inputs(
    chi: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    z: torch.Tensor,
    log_decay: torch.Tensor,
    initial_state: torch.Tensor | None,
    chunk_size: int,
) -> tuple[int, int, int, int, int, int]:
    if chi.ndim != 4:
        raise ValueError("chi must have shape [B, T, H, r]")
    if d.ndim != 5:
        raise ValueError("d must have shape [B, T, H, K, r]")
    batch, length, heads, edits, rank = d.shape
    if min(batch, length, heads, edits, rank) < 1:
        raise ValueError("B, T, H, K, and r must be positive")
    if chi.shape != (batch, length, heads, rank):
        raise ValueError("chi shape does not match d")
    if e.shape != d.shape:
        raise ValueError("e must match d shape")
    if z.ndim != 5 or z.shape[:4] != (batch, length, heads, edits):
        raise ValueError("z must have shape [B, T, H, K, d_v]")
    value_dim = z.shape[-1]
    if value_dim < 1:
        raise ValueError("d_v must be positive")
    if log_decay.shape != (batch, length, heads, rank):
        raise ValueError("log_decay must have shape [B, T, H, r]")

    tensors = (d, e, z, log_decay)
    if any(tensor.device != chi.device for tensor in tensors):
        raise ValueError("chi, d, e, z, and log_decay must share one device")
    if any(tensor.dtype != chi.dtype for tensor in tensors):
        raise TypeError("chi, d, e, z, and log_decay must share one dtype")
    if chi.device.type != "cuda":
        raise ValueError("the WY exterior requires CUDA tensors")
    if chi.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("the WY exterior requires FP16 or BF16 inputs")

    if initial_state is not None:
        expected = (batch, heads, rank, value_dim)
        if initial_state.shape != expected:
            raise ValueError(f"initial_state must have shape {expected}")
        if initial_state.device != chi.device:
            raise ValueError("initial_state must share the input CUDA device")
        if initial_state.dtype != torch.float32:
            raise TypeError("initial_state must be FP32")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an int")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return batch, length, heads, edits, rank, value_dim


def _materialize_fla_activation(e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Build ``a=-e*exp(g)`` without a second sign-only allocation."""
    a = e * torch.exp(g)
    return a.neg_()


def _pack_edits(
    chi: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    z: torch.Tensor,
    log_decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand tokens in token-major, edit-minor order for FLA's DPLR ABI."""
    batch, length, heads, edits, rank = d.shape
    value_dim = z.shape[-1]
    slot = torch.arange(edits, device=d.device).view(1, 1, edits, 1, 1)
    expanded_decay = log_decay.unsqueeze(2) * (slot == 0)
    expanded_query = chi.unsqueeze(2) * (slot == edits - 1)

    packed_q = expanded_query.reshape(batch, length * edits, heads, rank)
    packed_k = d.transpose(2, 3).reshape(batch, length * edits, heads, rank)
    packed_v = z.transpose(2, 3).reshape(batch, length * edits, heads, value_dim)
    packed_e = e.transpose(2, 3).reshape(batch, length * edits, heads, rank)
    packed_g = expanded_decay.reshape(batch, length * edits, heads, rank)
    packed_a = _materialize_fla_activation(packed_e, packed_g)
    return packed_q, packed_k, packed_v, packed_a, packed_g


def wy_associative(
    chi: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    z: torch.Tensor,
    log_decay: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate SolveDelta's associative state with FLA's chunk/WY kernels.

    The state is stored as ``[B, H, r, d_v]``. At token ``t`` it first
    receives the row-wise decay ``exp(log_decay[t])`` and then the ``K``
    ordered edits

    ``S <- S + d (z - S^T e)^T``.

    FLA's generalized Delta transition is algebraically identical under
    ``k=b=d, v=z, a=-e*exp(g)``. Sharing ``b`` with ``k`` avoids materializing
    the otherwise redundant signed copy of every edit direction. FLA's public
    DPLR ABI still requires materialized ``a``; eliminating it requires a
    direct-``e`` specialization inside the WY kernels rather than this wrapper.
    For ``K > 1``, token time is expanded in
    token-major, edit-minor order: only edit zero receives ``g`` and only edit
    ``K-1`` receives ``chi``. Thus decay happens once and each output is read
    after the final edit. ``log_decay`` is required to be nonpositive by the
    operator contract; checking that value constraint belongs to the frontend
    so this hot path does not synchronize the CUDA stream.

    Vector inputs are FP16 or BF16. Recurrent states are FP32, matching FLA's
    accumulation and returned-state contract.
    """
    batch, length, heads, edits, rank, value_dim = _validate_inputs(
        chi, d, e, z, log_decay, initial_state, chunk_size
    )
    try:
        from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("the WY exterior requires flash-linear-attention") from error

    if edits == 1:
        q = chi
        k = d.squeeze(3)
        v = z.squeeze(3)
        g = log_decay
        a = _materialize_fla_activation(e.squeeze(3), g)
    else:
        q, k, v, a, g = _pack_edits(chi, d, e, z, log_decay)

    expanded_output, final_state = chunk_dplr_delta_rule(
        q=q,
        k=k,
        v=v,
        a=a,
        b=k,
        gk=g,
        scale=1.0,
        initial_state=initial_state,
        output_final_state=output_final_state,
        safe_gate=False,
        chunk_size=chunk_size,
        disable_recompute=False,
    )
    if edits == 1:
        return expanded_output, final_state
    output = expanded_output.reshape(
        batch, length, edits, heads, value_dim
    )[:, :, -1]
    return output, final_state


__all__ = ["wy_associative"]
