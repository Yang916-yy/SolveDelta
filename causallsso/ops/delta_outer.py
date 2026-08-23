from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from causallsso.reference import SolveDeltaState


@triton.jit
def _pack_dplr_fwd_kernel(
    chi, d, e, z, g,
    q_out, k_out, v_out, a_out, b_out, g_out,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
    R: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    h = row % H
    edit = (row // H) % K
    token_batch = row // (H * K)
    t = token_batch % T
    b = token_batch // T
    offsets = tl.arange(0, BLOCK)
    mask_r = offsets < R
    mask_v = offsets < V
    source_r = ((b * T + t) * H + h) * K * R + edit * R + offsets
    source_z = ((b * T + t) * H + h) * K * V + edit * V + offsets
    token_r = ((b * T + t) * H + h) * R + offsets
    target_r = row * R + offsets
    target_v = row * V + offsets
    d_value = tl.load(d + source_r, mask=mask_r, other=0.0)
    e_value = tl.load(e + source_r, mask=mask_r, other=0.0)
    g_value = tl.load(g + token_r, mask=mask_r, other=0.0)
    active_g = tl.where(edit == 0, g_value, 0.0)
    q_value = tl.load(chi + token_r, mask=mask_r, other=0.0)
    q_value = tl.where(edit == K - 1, q_value, 0.0)
    tl.store(q_out + target_r, q_value, mask=mask_r)
    tl.store(k_out + target_r, d_value, mask=mask_r)
    tl.store(a_out + target_r, e_value * tl.exp(active_g.to(tl.float32)), mask=mask_r)
    tl.store(b_out + target_r, -d_value, mask=mask_r)
    tl.store(g_out + target_r, active_g, mask=mask_r)
    tl.store(v_out + target_v, tl.load(z + source_z, mask=mask_v, other=0.0), mask=mask_v)


@triton.jit
def _pack_dplr_bwd_kernel(
    a, g_input, dq, dk, dv, da, db, dg_out,
    dchi, dd, de, dz, dg,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
    R: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    h = row % H
    edit = (row // H) % K
    token_batch = row // (H * K)
    t = token_batch % T
    b = token_batch // T
    offsets = tl.arange(0, BLOCK)
    mask_r = offsets < R
    mask_v = offsets < V
    source_r = row * R + offsets
    source_v = row * V + offsets
    target_r = ((b * T + t) * H + h) * K * R + edit * R + offsets
    target_v = ((b * T + t) * H + h) * K * V + edit * V + offsets
    token_r = ((b * T + t) * H + h) * R + offsets
    tl.store(dd + target_r, tl.load(dk + source_r, mask=mask_r, other=0.0)
             - tl.load(db + source_r, mask=mask_r, other=0.0), mask=mask_r)
    da_value = tl.load(da + source_r, mask=mask_r, other=0.0)
    a_value = tl.load(a + source_r, mask=mask_r, other=0.0)
    gate_value = tl.load(g_input + token_r, mask=mask_r, other=0.0)
    scale = tl.where(edit == 0, tl.exp(gate_value.to(tl.float32)), 1.0)
    tl.store(de + target_r, da_value * scale, mask=mask_r)
    tl.store(dz + target_v, tl.load(dv + source_v, mask=mask_v, other=0.0), mask=mask_v)
    tl.store(
        dchi + token_r,
        tl.load(dq + source_r, mask=mask_r, other=0.0),
        mask=mask_r & (edit == K - 1),
    )
    local_dg = tl.load(dg_out + source_r, mask=mask_r, other=0.0) + da_value * a_value
    tl.store(dg + token_r, local_dg, mask=mask_r & (edit == 0))


class _PackDPLR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, chi, d, e, z, g):
        batch, length, heads, edits, rank = d.shape
        value_dim = z.shape[-1]
        expanded = length * edits
        q = torch.empty(batch, expanded, heads, rank, dtype=d.dtype, device=d.device)
        k = torch.empty_like(q)
        a = torch.empty_like(q)
        b = torch.empty_like(q)
        g_out = torch.empty_like(q)
        v = torch.empty(batch, expanded, heads, value_dim, dtype=z.dtype, device=z.device)
        block = triton.next_power_of_2(max(rank, value_dim))
        _pack_dplr_fwd_kernel[(batch * length * edits * heads,)](
            chi, d, e, z, g, q, k, v, a, b, g_out,
            T=length, H=heads, K=edits, R=rank, V=value_dim, BLOCK=block,
            num_warps=4,
        )
        ctx.save_for_backward(a, g)
        ctx.input_shapes = (chi.shape, d.shape, z.shape, g.shape)
        return q, k, v, a, b, g_out

    @staticmethod
    def backward(ctx, dq, dk, dv, da, db, dg_out):
        a, g_input = ctx.saved_tensors
        chi_shape, d_shape, z_shape, g_shape = ctx.input_shapes
        dchi = torch.empty(chi_shape, dtype=a.dtype, device=a.device)
        dd = torch.empty(d_shape, dtype=a.dtype, device=a.device)
        de = torch.empty_like(dd)
        dz = torch.empty(z_shape, dtype=dv.dtype, device=dv.device)
        dg = torch.empty(g_shape, dtype=a.dtype, device=a.device)
        batch, length, heads, edits, rank = d_shape
        value_dim = z_shape[-1]
        block = triton.next_power_of_2(max(rank, value_dim))
        _pack_dplr_bwd_kernel[(batch * length * edits * heads,)](
            a, g_input, dq, dk, dv, da, db, dg_out,
            dchi, dd, de, dz, dg,
            T=length, H=heads, K=edits, R=rank, V=value_dim, BLOCK=block,
            num_warps=4,
        )
        return dchi, dd, de, dz, dg


def fla_dplr_delta_outer(
    chi: torch.Tensor,
    write_direction: torch.Tensor,
    erase_direction: torch.Tensor,
    write_value: torch.Tensor,
    associative_log_decay: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the SolveDelta associative recurrence with FLA's DPLR kernels.

    The exact identification is

    ``k=d, v=z, a=e*exp(g), b=-d``

    because FLA's update is ``S <- exp(g) S + k v^T + b a^T S``.
    For ``K`` edits, token time is expanded by ``K``; forgetting is applied
    only to the first edit and the output is read only after the final edit.
    This keeps edit count a model hyperparameter while reusing the mature
    chunk/WY outer recurrence and its backward implementation.
    """
    try:
        from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule
    except ImportError as error:  # pragma: no cover - environment boundary
        raise RuntimeError("FLA is required for the optimized Delta outer path") from error

    if chi.ndim != 4:
        raise ValueError("chi must have shape [B, T, H, r]")
    if write_direction.ndim != 5:
        raise ValueError("write_direction must have shape [B, T, H, K, r]")
    batch, length, heads, edits, rank = write_direction.shape
    if chi.shape != (batch, length, heads, rank):
        raise ValueError("chi shape does not match write_direction")
    if erase_direction.shape != write_direction.shape:
        raise ValueError("erase_direction must match write_direction")
    if write_value.shape[:4] != (batch, length, heads, edits):
        raise ValueError("write_value must have shape [B, T, H, K, d_v]")
    if associative_log_decay.shape != (batch, length, heads, rank):
        raise ValueError("associative_log_decay must have shape [B, T, H, r]")
    if edits < 1:
        raise ValueError("edit count must be positive")
    tensors = (chi, write_direction, erase_direction, write_value, associative_log_decay)
    if any(x.device != chi.device or x.dtype != chi.dtype for x in tensors):
        raise ValueError("all outer-path tensors must share device and dtype")
    if chi.device.type != "cuda" or chi.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("FLA DPLR outer path requires CUDA FP16 or BF16 tensors")

    if edits == 1:
        # With one edit there is no micro-time expansion or read selection.
        # Feed FLA's DPLR ABI directly and avoid the pack/unpack kernels.
        q = chi
        d = write_direction.squeeze(-2)
        z = write_value.squeeze(-2)
        e = erase_direction.squeeze(-2)
        g = associative_log_decay
        a = e * torch.exp(g)
        b = -d
    else:
        # One fused pack writes the K-expanded DPLR ABI directly while
        # retaining the exact token-major/edit-minor order.
        q, d, z, a, b, g = _PackDPLR.apply(
            chi, write_direction, erase_direction, write_value, associative_log_decay
        )
    expanded_output, final_state = chunk_dplr_delta_rule(
        q=q,
        k=d,
        v=z,
        a=a,
        b=b,
        gk=g,
        scale=1.0,
        initial_state=initial_state,
        output_final_state=output_final_state,
        safe_gate=True,
        chunk_size=chunk_size,
        # The DPLR cache is small relative to SolveDelta's geometry workspace.
        # On the r=128,K=1 target, saving it makes outer forward+backward about
        # 12% faster for roughly 2.75 MiB additional allocation.
        disable_recompute=True,
    )
    output = expanded_output if edits == 1 else expanded_output.reshape(
        batch, length, edits, heads, write_value.shape[-1]
    )[:, :, -1]
    return output, final_state


def solvedelta_fused(
    u: torch.Tensor,
    h: torch.Tensor,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    associative_log_decay: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    skew: torch.Tensor,
    geometry_strength: torch.Tensor,
    *,
    initial_state: SolveDeltaState | None = None,
    output_final_state: bool = False,
    outer_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, SolveDeltaState | None]:
    """Run the selected differentiable r=128, K=1 fused CUDA schedule.

    Geometry summaries use a fixed 16-token Triton chunk scan. Exact packet
    frame actions are split across Triton prefix/radial tiles and dedicated
    CUDA primal/dual kernels without materializing tokenwise frames. The
    associative recurrence uses FLA's mature DPLR implementation. Geometry and
    frame accumulation are FP32; only transformed vectors and the outer
    recurrence are stored in ``outer_dtype``.

    Backward recomputes chunk summaries and local frame actions from saved
    vector inputs. This establishes the complete gradient contract without
    saving tokenwise matrices; native fused backward kernels may replace the
    recomputation implementation without changing this API.
    """
    tensors = (
        u, h, query, keys, values, geometry_log_decay,
        associative_log_decay, erase, write, skew, geometry_strength,
    )
    if outer_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("outer_dtype must be FP16 or BF16")
    if u.ndim != 4 or u.shape[-1] != 128:
        raise ValueError("the fused specialization requires u [B,T,H,128]")
    batch, length, heads, rank = u.shape
    if keys.shape != (batch, length, heads, 1, rank):
        raise ValueError("the fused specialization requires exactly K=1 edit")
    value_dim = values.shape[-1]
    expected = {
        "h": (batch, length, heads, rank),
        "query": (batch, length, heads, rank),
        "values": (batch, length, heads, 1, value_dim),
        "geometry_log_decay": (batch, length, heads),
        "associative_log_decay": (batch, length, heads, rank),
        "erase": keys.shape,
        "write": values.shape,
        "skew": (batch, length, heads, 1),
        "geometry_strength": (heads,),
    }
    named = {
        "h": h, "query": query, "values": values,
        "geometry_log_decay": geometry_log_decay,
        "associative_log_decay": associative_log_decay,
        "erase": erase, "write": write, "skew": skew,
        "geometry_strength": geometry_strength,
    }
    for name, shape in expected.items():
        if named[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if any(x.device != u.device or x.device.type != "cuda" for x in tensors):
        raise ValueError("all fused inputs must share one CUDA device")
    frame_tensors = (
        u, h, query, keys, geometry_log_decay, erase, skew, geometry_strength,
    )
    if any(x.dtype != torch.float32 for x in frame_tensors):
        raise TypeError("the fused geometry/frame boundary requires FP32 inputs")
    if values.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("values must be FP16, BF16, or FP32")
    if write.dtype != torch.float32 or associative_log_decay.dtype != torch.float32:
        raise TypeError("write gates and associative log decay must be FP32")

    # Local imports keep the independently testable staging operators free of
    # an import cycle while this function owns only their composition.
    from .packet_frame import packet_frame128
    from .triton_geometry import triton_geometry_chunk_scan

    normalized_u = F.normalize(u, p=2, dim=-1)
    normalized_query = F.normalize(query, p=2, dim=-1)
    normalized_keys = F.normalize(keys, p=2, dim=-1)
    boundaries, final_geometry = triton_geometry_chunk_scan(
        normalized_u,
        h,
        geometry_log_decay,
        initial_state=initial_state,
        chunk_size=16,
        input_precision="ieee",
    )
    d, e, chi = packet_frame128(
        boundaries.m,
        boundaries.J,
        boundaries.D,
        normalized_u,
        h,
        geometry_log_decay,
        normalized_keys,
        erase,
        normalized_query,
        skew,
        geometry_strength,
    )
    initial_associative = None
    if initial_state is not None:
        initial_associative = initial_state.S.to(outer_dtype)
    output, final_associative = fla_dplr_delta_outer(
        chi.to(outer_dtype),
        d.to(outer_dtype),
        e.to(outer_dtype),
        write.to(outer_dtype) * values.to(outer_dtype),
        associative_log_decay.to(outer_dtype),
        initial_state=initial_associative,
        output_final_state=output_final_state,
    )
    if not output_final_state:
        return output, None
    return output, SolveDeltaState(
        final_geometry.m,
        final_geometry.J,
        final_geometry.D,
        final_associative,
    )
