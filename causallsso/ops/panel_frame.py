from __future__ import annotations

import torch
import triton
import triton.language as tl

from .mathdx import _load_mathdx


_CHUNK = 32
_RANK = 128


@triton.jit
def _pack_inputs_kernel(
    u,
    h_value,
    key,
    erase,
    query,
    log_decay,
    packed_u,
    packed_h,
    packed_key,
    packed_erase,
    packed_query,
    packed_log_decay,
    length: tl.constexpr,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    rank: tl.constexpr,
):
    row = tl.program_id(0)
    local_token = row % chunk_size
    panel = row // chunk_size
    chunk = panel % chunks
    head = (panel // chunks) % heads
    batch = panel // (heads * chunks)
    token = chunk * chunk_size + local_token
    valid = token < length
    coordinates = tl.arange(0, rank)
    source = ((batch * length + token) * heads + head) * rank + coordinates
    target = row * rank + coordinates
    tl.store(packed_u + target, tl.load(u + source, mask=valid, other=0.0))
    tl.store(
        packed_h + target,
        tl.load(h_value + source, mask=valid, other=0.0),
    )
    tl.store(packed_key + target, tl.load(key + source, mask=valid, other=0.0))
    tl.store(
        packed_erase + target,
        tl.load(erase + source, mask=valid, other=0.0),
    )
    tl.store(
        packed_query + target,
        tl.load(query + source, mask=valid, other=0.0),
    )
    scalar_source = (batch * length + token) * heads + head
    tl.store(
        packed_log_decay + row,
        tl.load(log_decay + scalar_source, mask=valid, other=0.0),
    )


@triton.jit
def _pack_output_grads_kernel(
    grad_d,
    grad_e,
    grad_chi,
    packed_grad_d,
    packed_grad_e,
    packed_grad_chi,
    length: tl.constexpr,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    rank: tl.constexpr,
):
    row = tl.program_id(0)
    local_token = row % chunk_size
    panel = row // chunk_size
    chunk = panel % chunks
    head = (panel // chunks) % heads
    batch = panel // (heads * chunks)
    token = chunk * chunk_size + local_token
    valid = token < length
    coordinates = tl.arange(0, rank)
    source = ((batch * length + token) * heads + head) * rank + coordinates
    target = row * rank + coordinates
    tl.store(
        packed_grad_d + target,
        tl.load(grad_d + source, mask=valid, other=0.0),
    )
    tl.store(
        packed_grad_e + target,
        tl.load(grad_e + source, mask=valid, other=0.0),
    )
    tl.store(
        packed_grad_chi + target,
        tl.load(grad_chi + source, mask=valid, other=0.0),
    )


@triton.jit
def _unpack_outputs_kernel(
    packed_d,
    packed_e,
    packed_chi,
    d,
    e,
    chi,
    length: tl.constexpr,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    rank: tl.constexpr,
):
    row = tl.program_id(0)
    head = row % heads
    token_batch = row // heads
    token = token_batch % length
    batch = token_batch // length
    chunk = token // chunk_size
    local_token = token % chunk_size
    panel = (batch * heads + head) * chunks + chunk
    coordinates = tl.arange(0, rank)
    source = (panel * chunk_size + local_token) * rank + coordinates
    target = row * rank + coordinates
    tl.store(d + target, tl.load(packed_d + source))
    tl.store(e + target, tl.load(packed_e + source))
    tl.store(chi + target, tl.load(packed_chi + source))


@triton.jit
def _unpack_input_grads_kernel(
    packed_grad_u,
    packed_grad_h,
    packed_grad_log_decay,
    packed_grad_key,
    packed_grad_erase,
    packed_grad_query,
    grad_u,
    grad_h,
    grad_log_decay,
    grad_key,
    grad_erase,
    grad_query,
    length: tl.constexpr,
    heads: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
    rank: tl.constexpr,
):
    row = tl.program_id(0)
    head = row % heads
    token_batch = row // heads
    token = token_batch % length
    batch = token_batch // length
    chunk = token // chunk_size
    local_token = token % chunk_size
    panel = (batch * heads + head) * chunks + chunk
    coordinates = tl.arange(0, rank)
    source = (panel * chunk_size + local_token) * rank + coordinates
    target = row * rank + coordinates
    tl.store(grad_u + target, tl.load(packed_grad_u + source))
    tl.store(grad_h + target, tl.load(packed_grad_h + source))
    tl.store(grad_key + target, tl.load(packed_grad_key + source))
    tl.store(grad_erase + target, tl.load(packed_grad_erase + source))
    tl.store(grad_query + target, tl.load(packed_grad_query + source))
    scalar_source = panel * chunk_size + local_token
    tl.store(grad_log_decay + row, tl.load(packed_grad_log_decay + scalar_source))


def _pack_inputs(
    u: torch.Tensor,
    h: torch.Tensor,
    key: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    log_decay: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    batch, length, heads, rank = u.shape
    chunks = triton.cdiv(length, _CHUNK)
    panels = batch * heads * chunks
    vectors = tuple(
        torch.empty(panels, _CHUNK, rank, device=u.device, dtype=torch.float32)
        for _ in range(5)
    )
    packed_log_decay = torch.empty(
        panels, _CHUNK, device=u.device, dtype=torch.float32
    )
    _pack_inputs_kernel[(panels * _CHUNK,)](
        u.contiguous(),
        h.contiguous(),
        key.contiguous(),
        erase.contiguous(),
        query.contiguous(),
        log_decay.contiguous(),
        *vectors,
        packed_log_decay,
        length=length,
        heads=heads,
        chunks=chunks,
        chunk_size=_CHUNK,
        rank=rank,
        num_warps=4,
    )
    return (*vectors, packed_log_decay)


def _pack_output_grads(
    grad_d: torch.Tensor | None,
    grad_e: torch.Tensor | None,
    grad_chi: torch.Tensor | None,
    *,
    batch: int,
    length: int,
    heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    public_vector = (batch, length, heads, _RANK)
    if grad_d is None:
        grad_d = torch.zeros(public_vector, device=device, dtype=torch.float32)
    if grad_e is None:
        grad_e = torch.zeros(public_vector, device=device, dtype=torch.float32)
    if grad_chi is None:
        grad_chi = torch.zeros(public_vector, device=device, dtype=torch.float32)
    chunks = triton.cdiv(length, _CHUNK)
    panels = batch * heads * chunks
    packed = tuple(
        torch.empty(
            panels, _CHUNK, _RANK, device=device, dtype=torch.float32
        )
        for _ in range(3)
    )
    _pack_output_grads_kernel[(panels * _CHUNK,)](
        grad_d.contiguous(),
        grad_e.contiguous(),
        grad_chi.contiguous(),
        *packed,
        length=length,
        heads=heads,
        chunks=chunks,
        chunk_size=_CHUNK,
        rank=_RANK,
        num_warps=4,
    )
    return packed


def _unpack_outputs(
    packed_d: torch.Tensor,
    packed_e: torch.Tensor,
    packed_chi: torch.Tensor,
    *,
    batch: int,
    length: int,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vector_shape = (batch, length, heads, _RANK)
    d = torch.empty(vector_shape, device=packed_d.device, dtype=torch.float32)
    e = torch.empty_like(d)
    chi = torch.empty_like(d)
    chunks = triton.cdiv(length, _CHUNK)
    _unpack_outputs_kernel[(batch * length * heads,)](
        packed_d,
        packed_e,
        packed_chi,
        d,
        e,
        chi,
        length=length,
        heads=heads,
        chunks=chunks,
        chunk_size=_CHUNK,
        rank=_RANK,
        num_warps=4,
    )
    return d.unsqueeze(-2), e.unsqueeze(-2), chi


def _unpack_input_grads(
    packed: tuple[torch.Tensor, ...],
    *,
    batch: int,
    length: int,
    heads: int,
) -> tuple[torch.Tensor, ...]:
    device = packed[0].device
    vector_shape = (batch, length, heads, _RANK)
    vector_grads = tuple(
        torch.empty(vector_shape, device=device, dtype=torch.float32)
        for _ in range(5)
    )
    grad_log_decay = torch.empty(
        batch, length, heads, device=device, dtype=torch.float32
    )
    chunks = triton.cdiv(length, _CHUNK)
    _unpack_input_grads_kernel[(batch * length * heads,)](
        packed[0],
        packed[1],
        packed[2],
        packed[3],
        packed[4],
        packed[5],
        vector_grads[0],
        vector_grads[1],
        grad_log_decay,
        vector_grads[2],
        vector_grads[3],
        vector_grads[4],
        length=length,
        heads=heads,
        chunks=chunks,
        chunk_size=_CHUNK,
        rank=_RANK,
        num_warps=4,
    )
    return (
        vector_grads[0],
        vector_grads[1],
        grad_log_decay,
        vector_grads[2].unsqueeze(-2),
        vector_grads[3].unsqueeze(-2),
        vector_grads[4],
    )


class _PanelFrame128(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *tensors):
        (
            boundary_m,
            boundary_j,
            boundary_d,
            u,
            h,
            log_decay,
            keys,
            erase,
            query,
            strength,
        ) = tensors
        _load_mathdx()
        batch, length, heads, rank = u.shape
        chunks = triton.cdiv(length, _CHUNK)
        panels = batch * heads * chunks
        packed_u, packed_h, key, packed_erase, packed_query, packed_decay = (
            _pack_inputs(
                u,
                h,
                keys.squeeze(-2),
                erase.squeeze(-2),
                query,
                log_decay,
            )
        )
        flat_m = boundary_m.reshape(panels)
        flat_j = boundary_j.reshape(panels, rank, rank)
        flat_d = boundary_d.reshape_as(flat_j)
        packed_strength = strength.contiguous()
        alpha0, inverse_mass, coefficient, diagonal, norm_sq = (
            torch.ops.causallsso.panel_frame32_parameters128(
                flat_m,
                flat_j,
                flat_d,
                packed_u,
                packed_h,
                packed_decay,
                packed_strength,
                heads,
                chunks,
                length,
            )
        )
        packed_d, packed_e, packed_chi, lower_solved, dual_scaled = (
            torch.ops.causallsso.panel_frame32_action128(
                flat_j,
                flat_d,
                packed_u,
                packed_h,
                alpha0,
                inverse_mass,
                coefficient,
                diagonal,
                key,
                packed_erase,
                packed_query,
            )
        )
        ctx.save_for_backward(
            flat_m,
            flat_j,
            flat_d,
            packed_u,
            packed_h,
            packed_decay,
            key,
            packed_erase,
            packed_query,
            packed_strength,
            alpha0,
            inverse_mass,
            coefficient,
            diagonal,
            norm_sq,
            packed_d,
            lower_solved,
            dual_scaled,
        )
        ctx.layout = (batch, length, heads, chunks)
        return _unpack_outputs(
            packed_d,
            packed_e,
            packed_chi,
            batch=batch,
            length=length,
            heads=heads,
        )

    @staticmethod
    def backward(ctx, grad_d, grad_e, grad_chi):
        (
            boundary_m,
            boundary_j,
            boundary_d,
            packed_u,
            packed_h,
            packed_decay,
            key,
            erase,
            query,
            strength,
            alpha0,
            inverse_mass,
            coefficient,
            diagonal,
            norm_sq,
            write_direction,
            lower_solved,
            dual_scaled,
        ) = ctx.saved_tensors
        batch, length, heads, chunks = ctx.layout
        packed_grad_d, packed_grad_e, packed_grad_chi = _pack_output_grads(
            grad_d,
            grad_e,
            grad_chi,
            batch=batch,
            length=length,
            heads=heads,
            device=key.device,
        )
        (
            upper_left,
            upper_right,
            lower_left,
            lower_right,
            grad_diagonal,
            grad_key,
            grad_erase,
            grad_query,
        ) = torch.ops.causallsso.panel_frame32_action_vjp128(
            boundary_j,
            boundary_d,
            packed_u,
            packed_h,
            alpha0,
            inverse_mass,
            coefficient,
            diagonal,
            key,
            erase,
            query,
            write_direction,
            lower_solved,
            dual_scaled,
            packed_grad_d,
            packed_grad_e,
            packed_grad_chi,
        )
        radial_grads = torch.ops.causallsso.panel_frame32_radial_vjp128(
            boundary_m,
            boundary_j,
            boundary_d,
            packed_u,
            packed_h,
            packed_decay,
            strength,
            alpha0,
            inverse_mass,
            coefficient,
            norm_sq,
            upper_left,
            upper_right,
            lower_left,
            lower_right,
            grad_diagonal,
            heads,
            chunks,
            length,
        )
        unpacked = _unpack_input_grads(
            (
                radial_grads[3],
                radial_grads[4],
                radial_grads[5],
                grad_key,
                grad_erase,
                grad_query,
            ),
            batch=batch,
            length=length,
            heads=heads,
        )
        matrix_shape = (batch, heads, chunks, _RANK, _RANK)
        return (
            radial_grads[0].reshape(batch, heads, chunks),
            radial_grads[1].reshape(matrix_shape),
            radial_grads[2].reshape(matrix_shape),
            unpacked[0],
            unpacked[1],
            unpacked[2],
            unpacked[3],
            unpacked[4],
            unpacked[5],
            radial_grads[6],
        )


def panel_frame128(
    boundary_m: torch.Tensor,
    boundary_j: torch.Tensor,
    boundary_d: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    keys: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    geometry_strength: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the exact skew-off C32 frame specialization for ``r=128, K=1``."""
    if u.ndim != 4 or u.shape[-1] != _RANK:
        raise ValueError("panel_frame128 requires u [B,T,H,128]")
    batch, length, heads, rank = u.shape
    chunks = triton.cdiv(length, _CHUNK)
    if length < 1:
        raise ValueError("length must be positive")
    if boundary_m.shape != (batch, heads, chunks):
        raise ValueError("boundary_m must be [B,H,ceil(T/32)]")
    matrix_shape = (batch, heads, chunks, rank, rank)
    if boundary_j.shape != matrix_shape or boundary_d.shape != matrix_shape:
        raise ValueError("boundary_j and boundary_d shape mismatch")
    if h.shape != u.shape or query.shape != u.shape:
        raise ValueError("h and query must match u")
    edit_shape = (batch, length, heads, 1, rank)
    if keys.shape != edit_shape or erase.shape != edit_shape:
        raise ValueError("keys and erase must be [B,T,H,1,128]")
    if geometry_log_decay.shape != (batch, length, heads):
        raise ValueError("geometry_log_decay must be [B,T,H]")
    if geometry_strength.shape != (heads,):
        raise ValueError("geometry_strength must be [H]")
    tensors = (
        boundary_m,
        boundary_j,
        boundary_d,
        u,
        h,
        geometry_log_decay,
        keys,
        erase,
        query,
        geometry_strength,
    )
    if any(value.device != u.device or value.device.type != "cuda" for value in tensors):
        raise ValueError("all panel-frame tensors must share one CUDA device")
    if any(value.dtype != torch.float32 for value in tensors):
        raise TypeError("panel_frame128 supports FP32 inputs and states only")
    return _PanelFrame128.apply(*tensors)
