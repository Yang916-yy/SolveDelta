from __future__ import annotations

from pathlib import Path

import torch

from causallsso.reference import (
    apply_dual_reference,
    apply_primal_reference,
    bounded_ldu_reference,
)

_LOADED = False
_BACKWARD_SUBCHUNK = 8


def _library_candidates() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[2]
    return (
        root / "build" / "native" / "libcausallsso_mathdx.so",
        root / "build" / "libcausallsso_mathdx.so",
    )


def _load_mathdx() -> None:
    global _LOADED
    if _LOADED:
        return
    for path in _library_candidates():
        if path.is_file():
            torch.ops.load_library(str(path))
            _LOADED = True
            return
    raise RuntimeError(
        "SolveDelta MathDx extension is not built; configure native/CMakeLists.txt "
        "into build/native first"
    )


def mathdx_available() -> bool:
    try:
        _load_mathdx()
    except (OSError, RuntimeError):
        return False
    return True


class _MathDxTRSM128(torch.autograd.Function):
    @staticmethod
    def forward(ctx, factor: torch.Tensor, rhs: torch.Tensor, upper: bool) -> torch.Tensor:
        _load_mathdx()
        if factor.shape[-2:] != (128, 128):
            raise ValueError("MathDx validation path supports only 128x128 factors")
        if rhs.shape[-2:] != (128, 2):
            raise ValueError("MathDx validation path supports exactly two right-hand sides")
        if factor.dtype != torch.float32 or rhs.dtype != torch.float32:
            raise TypeError("MathDx TRSM factors and right-hand sides must be FP32")
        if factor.device != rhs.device or factor.device.type != "cuda":
            raise ValueError("MathDx TRSM inputs must share one CUDA device")
        batch_shape = torch.broadcast_shapes(factor.shape[:-2], rhs.shape[:-2])
        factor = factor.expand(*batch_shape, 128, 128)
        rhs = rhs.expand(*batch_shape, 128, 2)
        # The native validation operator accepts packed column-major buffers.
        # These copies disappear in the final chunk-fused kernel.
        factor_col = factor.transpose(-1, -2).contiguous().reshape(-1, 128, 128)
        rhs_col = rhs.transpose(-1, -2).contiguous().reshape(-1, 2, 128)
        out_col = torch.ops.causallsso.mathdx_trsm128(factor_col, rhs_col, upper)
        out = out_col.reshape(*batch_shape, 2, 128).transpose(-1, -2).contiguous()
        ctx.save_for_backward(factor, out)
        ctx.upper = upper
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        factor, out = ctx.saved_tensors
        # dB = A^-T dX. Reuse MathDx with the transposed triangular factor.
        grad_rhs = _MathDxTRSM128.apply(
            factor.transpose(-1, -2).contiguous(), grad_out.contiguous(), not ctx.upper
        )
        grad_factor = -(grad_rhs @ out.transpose(-1, -2))
        grad_factor = (
            torch.triu(grad_factor, diagonal=1)
            if ctx.upper
            else torch.tril(grad_factor, diagonal=-1)
        )
        return grad_factor, grad_rhs, None


class _MathDxSolveFrame128(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lower, diagonal, upper, keys, erase, query):
        _load_mathdx()
        dual_rhs = torch.cat((erase, query.unsqueeze(-2)), dim=-2).contiguous()
        write_direction, dual = torch.ops.causallsso.mathdx_solve_frame128(
            lower.contiguous(), diagonal.contiguous(), upper.contiguous(),
            keys.contiguous(), dual_rhs,
        )
        ctx.save_for_backward(lower, diagonal, upper, keys, erase, query)
        return write_direction, dual[..., :2, :], dual[..., 2, :]

    @staticmethod
    def backward(ctx, grad_d, grad_e, grad_chi):
        # The standalone oracle recomputes its exact factorwise graph. The
        # training path below instead recomputes factors from chunk boundaries.
        saved = ctx.saved_tensors
        with torch.enable_grad():
            inputs = tuple(x.detach().requires_grad_(True) for x in saved)
            lower, diagonal, upper, keys, erase, query = inputs
            d = apply_primal_reference(
                lower, diagonal, upper, keys.transpose(-1, -2)
            ).transpose(-1, -2)
            dual_rhs = torch.cat((erase, query.unsqueeze(-2)), dim=-2)
            dual = apply_dual_reference(
                lower, diagonal, upper, dual_rhs.transpose(-1, -2)
            ).transpose(-1, -2)
            gradients = torch.autograd.grad(
                (d, dual[..., :2, :], dual[..., 2, :]),
                inputs,
                (grad_d, grad_e, grad_chi),
            )
        return gradients


def mathdx_trsm128(
    factor: torch.Tensor,
    rhs: torch.Tensor,
    *,
    upper: bool,
) -> torch.Tensor:
    """Validation-stage MathDx unit-triangular solve for ``r=128, nrhs=2``.

    This operator intentionally exposes factor packing so its forward and
    backward can be checked independently. It is not the final fused path.
    """
    return _MathDxTRSM128.apply(factor, rhs, upper)


def mathdx_solve_frame128(
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    keys: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    *,
    dual_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the r=128, K in {1,2} SolveDelta frame with MathDx TRSM.

    K1 is padded with an exact zero second RHS for the standalone MathDx
    ``nrhs=2`` specialization. The primal write directions share that solve. The
    erase directions and query use the exact inverse-transpose dual
    ``M.T @ x`` and therefore require only GEMMs, not additional solves.
    Factors and primal solves stay FP32. Dual GEMMs may use BF16/FP16 inputs;
    FP32 remains the default until those casts are fused with factor creation.
    """
    if lower.shape[-2:] != (128, 128) or upper.shape != lower.shape:
        raise ValueError("lower and upper must have equal [..., 128, 128] shapes")
    if diagonal.shape != lower.shape[:-1]:
        raise ValueError("diagonal must have shape [..., 128]")
    if keys.ndim != lower.ndim or keys.shape[:-2] != lower.shape[:-2]:
        raise ValueError("keys must have shape [..., K, 128]")
    edits = keys.shape[-2]
    if edits not in (1, 2) or keys.shape[-1] != 128:
        raise ValueError("the standalone MathDx frame supports K in {1, 2}")
    if erase.shape != keys.shape or query.shape != (*lower.shape[:-2], 128):
        raise ValueError("erase/query shapes do not match the frame batch")
    tensors = (lower, diagonal, upper, keys, erase, query)
    if any(x.device != lower.device for x in tensors):
        raise ValueError("all frame tensors must share one device")
    if any(x.dtype != torch.float32 for x in tensors):
        raise TypeError("MathDx frame factors and inputs must be FP32")
    if dual_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("dual_dtype must be FP32, BF16, or FP16")
    padded_keys = keys
    padded_erase = erase
    if edits == 1:
        padded_keys = torch.cat((keys, torch.zeros_like(keys)), dim=-2)
        padded_erase = torch.cat((erase, torch.zeros_like(erase)), dim=-2)
    if dual_dtype == torch.float32:
        write_direction, erase_direction, solved_query = _MathDxSolveFrame128.apply(
            lower, diagonal, upper, padded_keys, padded_erase, query
        )
        return (
            write_direction[..., :edits, :],
            erase_direction[..., :edits, :],
            solved_query,
        )

    rhs = padded_keys.transpose(-1, -2).contiguous()
    primal = mathdx_trsm128(lower, rhs, upper=False)
    primal = primal / diagonal.unsqueeze(-1)
    primal = mathdx_trsm128(upper, primal, upper=True)
    write_direction = primal.transpose(-1, -2).contiguous()[..., :edits, :]

    dual_rhs = torch.cat((padded_erase, query.unsqueeze(-2)), dim=-2)
    # Quantize only tensor-core operands. CUDA tensor cores accumulate the
    # GEMM internally in FP32; the BF16/FP16 result is immediately widened
    # before diagonal scaling, so the recurrent state itself stays FP32.
    l_t = lower.transpose(-1, -2).to(dual_dtype)
    u_t = upper.transpose(-1, -2).to(dual_dtype)
    dual_input = dual_rhs.transpose(-1, -2).to(dual_dtype)
    dual_rhs_count = 3
    batch_count = lower.numel() // (128 * 128)
    dual = torch.bmm(
        l_t.reshape(batch_count, 128, 128),
        dual_input.reshape(batch_count, 128, dual_rhs_count),
    ).float().reshape(*lower.shape[:-2], 128, dual_rhs_count)
    dual = diagonal.unsqueeze(-1) * dual
    dual = torch.bmm(
        u_t.reshape(batch_count, 128, 128),
        dual.to(dual_dtype).reshape(batch_count, 128, dual_rhs_count),
    ).float().reshape(*lower.shape[:-2], 128, dual_rhs_count)
    dual = dual.transpose(-1, -2).contiguous()
    return write_direction, dual[..., :edits, :], dual[..., 2, :]


def _neumann4(factor: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Apply the degree-four inverse polynomial for a unit-triangular factor."""
    strict = factor - torch.eye(128, dtype=factor.dtype, device=factor.device)
    term = rhs
    result = rhs
    for _ in range(4):
        term = -(strict @ term)
        result = result + term
    return result


def _chunk_frame_recompute(
    boundary_m: torch.Tensor,
    boundary_J: torch.Tensor,
    boundary_D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    keys: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    skew: torch.Tensor,
    strength: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable engineering oracle for the chunk-owned CUDA kernel."""
    batch, length, heads, rank = u.shape
    native_subchunk = 8
    ds: list[torch.Tensor | None] = [None] * length
    es: list[torch.Tensor | None] = [None] * length
    chis: list[torch.Tensor | None] = [None] * length
    for block_start in range(0, length, native_subchunk):
        chunk = block_start // chunk_size
        chunk_start = chunk * chunk_size
        start_local = block_start - chunk_start
        end_local = min(start_local + native_subchunk, chunk_size, length - chunk_start)
        mass = boundary_m[:, :, chunk]
        # Backward uses the FP32 chart as a straight-through estimator for the
        # forward-only storage quantization. Differentiating through every
        # BF16/FP16 rounding boundary made the tiny geometry-decay gradient
        # dominated by quantization noise.
        moment_j = boundary_J[:, :, chunk]
        moment_d = boundary_D[:, :, chunk]
        for local_t in range(end_local):
            token = chunk_start + local_t
            decay = torch.exp(geometry_log_decay[:, token])
            mass = decay * mass + 1.0
            ut, ht = u[:, token], h[:, token]
            moment_j = (
                decay[..., None, None] * moment_j
                + ut[..., :, None] * ut[..., None, :]
            )
            moment_d = (
                decay[..., None, None] * moment_d
                + ut[..., :, None] * ht[..., None, :]
            )
            if local_t < start_local:
                continue
            lower, diagonal, upper, _ = bounded_ldu_reference(
                moment_j / mass[..., None, None],
                moment_d / mass[..., None, None],
                strength,
            )
            omega = 0.5 * (
                lower + upper - lower.transpose(-1, -2) - upper.transpose(-1, -2)
            )
            token_d, token_e = [], []
            for edit in range(keys.shape[-2]):
                key = keys[:, token, :, edit]
                b0 = erase[:, token, :, edit] * key
                tau = (b0 * key).sum(-1)
                direction = (omega @ key.unsqueeze(-1)).squeeze(-1)
                direction = direction / torch.sqrt(
                    1.0 + direction.square().sum(-1, keepdim=True)
                )
                b = b0 + (
                    tau * (2.0 - tau) * skew[:, token, :, edit]
                )[..., None] * direction
                primal = _neumann4(lower, key.unsqueeze(-1))
                primal = primal / diagonal.unsqueeze(-1)
                primal = _neumann4(upper, primal)
                dual = lower.transpose(-1, -2) @ b.unsqueeze(-1)
                dual = diagonal.unsqueeze(-1) * dual
                dual = upper.transpose(-1, -2) @ dual
                token_d.append(primal.squeeze(-1))
                token_e.append(dual.squeeze(-1))
            dual_query = lower.transpose(-1, -2) @ query[:, token].unsqueeze(-1)
            dual_query = diagonal.unsqueeze(-1) * dual_query
            dual_query = upper.transpose(-1, -2) @ dual_query
            ds[token] = torch.stack(token_d, dim=2)
            es[token] = torch.stack(token_e, dim=2)
            chis[token] = dual_query.squeeze(-1)
    return (
        torch.stack([x for x in ds if x is not None], dim=1),
        torch.stack([x for x in es if x is not None], dim=1),
        torch.stack([x for x in chis if x is not None], dim=1),
    )


def _chunk_frame_recompute_one(
    initial_m: torch.Tensor,
    initial_J: torch.Tensor,
    initial_D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    keys: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    skew: torch.Tensor,
    strength: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Vectorized local VJP graph, including its terminal geometry state."""
    mass, moment_j, moment_d = initial_m, initial_J, initial_D
    masses, moments_j, moments_d = [], [], []
    for token in range(u.shape[1]):
        decay = torch.exp(geometry_log_decay[:, token])
        ut, ht = u[:, token], h[:, token]
        mass = decay * mass + 1.0
        moment_j = decay[..., None, None] * moment_j + ut[..., :, None] * ut[..., None, :]
        moment_d = decay[..., None, None] * moment_d + ut[..., :, None] * ht[..., None, :]
        masses.append(mass)
        moments_j.append(moment_j)
        moments_d.append(moment_d)
    mass = torch.stack(masses, dim=1)
    moment_j = torch.stack(moments_j, dim=1)
    moment_d = torch.stack(moments_d, dim=1)
    lower, diagonal, upper, omega = bounded_ldu_reference(
        moment_j / mass[..., None, None],
        moment_d / mass[..., None, None],
        strength,
    )
    lower_action = lower.to(torch.float16)
    upper_action = upper.to(torch.float16)
    omega = 0.5 * (
        lower_action.float() + upper_action.float()
        - lower_action.transpose(-1, -2).float()
        - upper_action.transpose(-1, -2).float()
    )
    b0 = erase * keys
    tau = (b0 * keys).sum(-1)
    direction = torch.einsum("bthrs,bthks->bthkr", omega, keys)
    direction = direction / torch.sqrt(1.0 + direction.square().sum(-1, keepdim=True))
    b = b0 + (tau * (2.0 - tau) * skew)[..., None] * direction

    primal = _neumann4(lower_action, keys.transpose(-1, -2).to(torch.float16))
    primal = (primal.float() / diagonal.unsqueeze(-1)).to(torch.float16)
    primal = _neumann4(upper_action, primal).transpose(-1, -2).float()
    dual_rhs = torch.cat((b, query.unsqueeze(-2)), dim=-2).transpose(-1, -2)
    dual = lower_action.transpose(-1, -2) @ dual_rhs.to(torch.float16)
    dual = (diagonal.unsqueeze(-1) * dual.float()).to(torch.float16)
    dual = (upper_action.transpose(-1, -2) @ dual).transpose(-1, -2).float()
    edits = keys.shape[-2]
    return (
        primal,
        dual[..., :edits, :],
        dual[..., edits, :],
        mass[:, -1],
        moment_j[:, -1],
        moment_d[:, -1],
    )


def _neumann4_terms(
    factor: torch.Tensor,
    rhs: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    strict = factor - torch.eye(128, dtype=factor.dtype, device=factor.device)
    terms = [rhs]
    for _ in range(4):
        terms.append(-(strict @ terms[-1]))
    return strict, terms, sum(terms[1:], start=terms[0])


def _neumann4_vjp(
    strict: torch.Tensor,
    terms: list[torch.Tensor],
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_strict = torch.zeros_like(strict)
    grad_term = grad_output
    for power in range(4, 0, -1):
        grad_strict = grad_strict - grad_term @ terms[power - 1].transpose(-1, -2)
        grad_term = grad_output - strict.transpose(-1, -2) @ grad_term
    return grad_strict, grad_term


def _radial_vjp(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    denominator_sq = x.square().sum(dim=(-2, -1), keepdim=True) + radius * radius
    scale = radius * torch.rsqrt(denominator_sq)
    projection = (grad_output * x).sum(dim=(-2, -1), keepdim=True)
    return scale * (grad_output - x * projection / denominator_sq)


def _chunk_frame_vjp_one(
    initial_m: torch.Tensor,
    initial_J: torch.Tensor,
    initial_D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    keys: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    skew: torch.Tensor,
    strength: torch.Tensor,
    grad_d: torch.Tensor,
    grad_e: torch.Tensor,
    grad_chi: torch.Tensor,
    grad_final_m: torch.Tensor,
    grad_final_J: torch.Tensor,
    grad_final_D: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Explicit adjoint for one packed geometry/frame subchunk."""
    masses = [initial_m]
    moments_j = [initial_J]
    moments_d = [initial_D]
    decays = []
    mass, moment_j, moment_d = initial_m, initial_J, initial_D
    for token in range(u.shape[1]):
        decay = torch.exp(geometry_log_decay[:, token])
        ut, ht = u[:, token], h[:, token]
        mass = decay * mass + 1.0
        moment_j = decay[..., None, None] * moment_j + ut[..., :, None] * ut[..., None, :]
        moment_d = decay[..., None, None] * moment_d + ut[..., :, None] * ht[..., None, :]
        decays.append(decay)
        masses.append(mass)
        moments_j.append(moment_j)
        moments_d.append(moment_d)

    mass_tokens = torch.stack(masses[1:], dim=1)
    moment_j_tokens = torch.stack(moments_j[1:], dim=1)
    moment_d_tokens = torch.stack(moments_d[1:], dim=1)
    normalized_j = moment_j_tokens / mass_tokens[..., None, None]
    normalized_d = moment_d_tokens / mass_tokens[..., None, None]
    eye = torch.eye(128, dtype=u.dtype, device=u.device)
    strength_matrix = strength[None, None, :, None, None]
    x_h = strength_matrix * (normalized_j - eye / 128.0)
    x_r = strength_matrix * normalized_d
    x_h_lower = torch.tril(x_h, diagonal=-1)
    x_h_upper = torch.triu(x_h, diagonal=1)
    x_r_lower = torch.tril(x_r, diagonal=-1)
    x_r_upper = torch.triu(x_r, diagonal=1)

    def radial(x: torch.Tensor, radius: float) -> torch.Tensor:
        norm_sq = x.square().sum(dim=(-2, -1), keepdim=True)
        return radius * x / torch.sqrt(norm_sq + radius * radius)

    n_lower = radial(x_h_lower, 1.0 / 8.0) + radial(x_r_lower, 1.0 / 8.0)
    n_upper = radial(x_h_upper, 1.0 / 8.0) + radial(x_r_upper, 1.0 / 8.0)
    diag_h = torch.diagonal(x_h, dim1=-2, dim2=-1)
    diag_r = torch.diagonal(x_r, dim1=-2, dim2=-1)
    tanh_h = torch.tanh(diag_h / (1.0 / 8.0))
    tanh_r = torch.tanh(diag_r / (1.0 / 8.0))
    diagonal = torch.exp((1.0 / 8.0) * tanh_h + (1.0 / 8.0) * tanh_r)
    lower = (eye + n_lower).to(torch.float16)
    upper = (eye + n_upper).to(torch.float16)
    omega = 0.5 * (
        lower.float() + upper.float()
        - lower.transpose(-1, -2).float()
        - upper.transpose(-1, -2).float()
    )

    key = keys.squeeze(-2)
    erase_gate = erase.squeeze(-2)
    skew_gate = skew.squeeze(-1)
    b0 = erase_gate * key
    tau = (b0 * key).sum(-1)
    omega_key = (omega @ key.unsqueeze(-1)).squeeze(-1)
    inverse_norm = torch.rsqrt(1.0 + omega_key.square().sum(-1, keepdim=True))
    direction = omega_key * inverse_norm
    coefficient = tau * (2.0 - tau) * skew_gate
    b = b0 + coefficient[..., None] * direction

    lower_strict, lower_terms, lower_result = _neumann4_terms(
        lower, key.unsqueeze(-1).to(torch.float16)
    )
    scaled_primal_float = lower_result.float() / diagonal.unsqueeze(-1)
    scaled_primal = scaled_primal_float.to(torch.float16)
    upper_strict, upper_terms, _ = _neumann4_terms(upper, scaled_primal)

    grad_upper_primal, grad_scaled_primal = _neumann4_vjp(
        upper_strict,
        upper_terms,
        grad_d.squeeze(-2).unsqueeze(-1).to(torch.float16),
    )
    grad_scaled_primal_float = grad_scaled_primal.float()
    grad_diagonal = -(
        grad_scaled_primal_float
        * lower_result.float()
        / diagonal.unsqueeze(-1).square()
    ).sum(-1)
    grad_lower_result = (
        grad_scaled_primal_float / diagonal.unsqueeze(-1)
    ).to(torch.float16)
    grad_lower_primal, grad_key_primal = _neumann4_vjp(
        lower_strict, lower_terms, grad_lower_result
    )

    dual_rhs = torch.stack((b, query), dim=-1).to(torch.float16)
    dual_lower = lower.transpose(-1, -2) @ dual_rhs
    dual_scaled_float = diagonal.unsqueeze(-1) * dual_lower.float()
    dual_scaled = dual_scaled_float.to(torch.float16)
    grad_dual = torch.stack((grad_e.squeeze(-2), grad_chi), dim=-1).to(torch.float16)
    grad_upper_dual = dual_scaled @ grad_dual.transpose(-1, -2)
    grad_dual_scaled = upper @ grad_dual
    grad_dual_scaled_float = grad_dual_scaled.float()
    grad_diagonal = grad_diagonal + (
        grad_dual_scaled_float * dual_lower.float()
    ).sum(-1)
    grad_dual_lower = (
        diagonal.unsqueeze(-1) * grad_dual_scaled_float
    ).to(torch.float16)
    grad_lower_dual = dual_rhs @ grad_dual_lower.transpose(-1, -2)
    grad_dual_rhs = lower @ grad_dual_lower
    grad_b = grad_dual_rhs[..., 0].float()
    grad_query = grad_dual_rhs[..., 1].float()

    grad_coefficient = (grad_b * direction).sum(-1)
    grad_direction = coefficient[..., None] * grad_b
    direction_projection = (grad_direction * omega_key).sum(-1, keepdim=True)
    grad_omega_key = (
        inverse_norm * grad_direction
        - inverse_norm.pow(3) * omega_key * direction_projection
    )
    grad_omega = grad_omega_key.unsqueeze(-1) * key.unsqueeze(-2)
    grad_key = (
        omega.transpose(-1, -2) @ grad_omega_key.unsqueeze(-1)
    ).squeeze(-1)
    grad_tau = grad_coefficient * (2.0 - 2.0 * tau) * skew_gate
    grad_skew = grad_coefficient * tau * (2.0 - tau)
    grad_b0 = grad_b + grad_tau[..., None] * key
    grad_key = grad_key + grad_tau[..., None] * b0
    grad_erase = grad_b0 * key
    grad_key = grad_key + grad_b0 * erase_gate + grad_key_primal.squeeze(-1).float()

    grad_skew_factor = 0.5 * (grad_omega - grad_omega.transpose(-1, -2))
    grad_lower = (
        grad_lower_primal
        + grad_lower_dual
        + grad_skew_factor.to(torch.float16)
    ).float()
    grad_upper = (
        grad_upper_primal
        + grad_upper_dual
        + grad_skew_factor.to(torch.float16)
    ).float()
    grad_n_lower = torch.tril(grad_lower, diagonal=-1)
    grad_n_upper = torch.triu(grad_upper, diagonal=1)
    grad_x_h = _radial_vjp(x_h_lower, grad_n_lower, 1.0 / 8.0)
    grad_x_h = grad_x_h + _radial_vjp(x_h_upper, grad_n_upper, 1.0 / 8.0)
    grad_x_r = _radial_vjp(x_r_lower, grad_n_lower, 1.0 / 8.0)
    grad_x_r = grad_x_r + _radial_vjp(x_r_upper, grad_n_upper, 1.0 / 8.0)
    grad_log_diagonal = grad_diagonal * diagonal
    grad_x_h = grad_x_h + torch.diag_embed(
        grad_log_diagonal * (1.0 - tanh_h.square())
    )
    grad_x_r = grad_x_r + torch.diag_embed(
        grad_log_diagonal * (1.0 - tanh_r.square())
    )

    grad_normalized_j = strength_matrix * grad_x_h
    grad_normalized_d = strength_matrix * grad_x_r
    grad_strength = (
        grad_x_h * (normalized_j - eye / 128.0)
        + grad_x_r * normalized_d
    ).sum(dim=(0, 1, 3, 4))
    grad_m_tokens = -(
        (grad_normalized_j * moment_j_tokens).sum(dim=(-2, -1))
        + (grad_normalized_d * moment_d_tokens).sum(dim=(-2, -1))
    ) / mass_tokens.square()
    grad_j_tokens = grad_normalized_j / mass_tokens[..., None, None]
    grad_d_tokens = grad_normalized_d / mass_tokens[..., None, None]

    grad_u_tokens: list[torch.Tensor] = [u[:, 0]] * u.shape[1]
    grad_h_tokens: list[torch.Tensor] = [h[:, 0]] * h.shape[1]
    grad_log_tokens: list[torch.Tensor] = [geometry_log_decay[:, 0]] * u.shape[1]
    adjoint_m = grad_final_m
    adjoint_j = grad_final_J
    adjoint_d = grad_final_D
    for token in range(u.shape[1] - 1, -1, -1):
        adjoint_m = adjoint_m + grad_m_tokens[:, token]
        adjoint_j = adjoint_j + grad_j_tokens[:, token]
        adjoint_d = adjoint_d + grad_d_tokens[:, token]
        previous_m = masses[token]
        previous_j = moments_j[token]
        previous_d = moments_d[token]
        decay = decays[token]
        ut, ht = u[:, token], h[:, token]
        grad_decay = (
            adjoint_m * previous_m
            + (adjoint_j * previous_j).sum(dim=(-2, -1))
            + (adjoint_d * previous_d).sum(dim=(-2, -1))
        )
        grad_u_tokens[token] = (
            (adjoint_j + adjoint_j.transpose(-1, -2)) @ ut.unsqueeze(-1)
            + adjoint_d @ ht.unsqueeze(-1)
        ).squeeze(-1)
        grad_h_tokens[token] = (
            adjoint_d.transpose(-1, -2) @ ut.unsqueeze(-1)
        ).squeeze(-1)
        grad_log_tokens[token] = grad_decay * decay
        adjoint_m = decay * adjoint_m
        adjoint_j = decay[..., None, None] * adjoint_j
        adjoint_d = decay[..., None, None] * adjoint_d

    return (
        adjoint_m,
        adjoint_j,
        adjoint_d,
        torch.stack(grad_u_tokens, dim=1),
        torch.stack(grad_h_tokens, dim=1),
        torch.stack(grad_log_tokens, dim=1),
        grad_key.unsqueeze(-2),
        grad_erase.unsqueeze(-2),
        grad_query,
        grad_skew.unsqueeze(-1),
        grad_strength,
    )


# The explicit adjoint is compiled once per concrete subchunk shape. Its
# differentiable recompute counterpart remains only as the local VJP oracle.
_compiled_chunk_frame_vjp_one = torch.compile(
    _chunk_frame_vjp_one,
    fullgraph=True,
    dynamic=False,
)


class _CudaChunkSolveFrame128(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        *tensors, chunk_size = args
        _load_mathdx()
        outputs = torch.ops.causallsso.cuda_chunk_solve_frame128(
            *(x.contiguous() for x in tensors), chunk_size
        )
        ctx.save_for_backward(*tensors)
        ctx.chunk_size = chunk_size
        return outputs

    @staticmethod
    def backward(ctx, grad_d, grad_e, grad_chi):
        saved = ctx.saved_tensors
        boundary_m, boundary_J, boundary_D = saved[:3]
        u, h, geometry_log_decay, keys, erase, query, skew, strength = saved[3:]
        gradients = [torch.zeros_like(x) for x in saved]
        chunks = boundary_m.shape[2]
        batch = u.shape[0]

        # The forward owns 64-token chunks but the reverse pass checkpoints
        # only their eight-token geometry boundaries. Processing one local
        # offset across every independent chunk exposes a large GEMM batch,
        # while the reverse boundary cotangent carries all later-token effects
        # without retaining tokenwise r-by-r charts.
        from .triton_geometry import _compiled_geometry_scan_recompute

        def backprop_packed(initial, sequence, output_grads):
            local_length = sequence[0].shape[1]
            with torch.no_grad():
                sub_m, sub_J, sub_D, _, _, _ = _compiled_geometry_scan_recompute(
                    sequence[0], sequence[1], sequence[2], *initial,
                    _BACKWARD_SUBCHUNK,
                )
            sequence_grads = [torch.zeros_like(x) for x in sequence]
            adjoint = (
                torch.zeros_like(initial[0]),
                torch.zeros_like(initial[1]),
                torch.zeros_like(initial[2]),
            )
            strength_grad = torch.zeros_like(strength)
            subchunks = sub_m.shape[2]
            for subchunk in range(subchunks - 1, -1, -1):
                start = subchunk * _BACKWARD_SUBCHUNK
                end = min(start + _BACKWARD_SUBCHUNK, local_length)
                local = (
                    sub_m[:, :, subchunk],
                    sub_J[:, :, subchunk],
                    sub_D[:, :, subchunk],
                    *(x[:, start:end] for x in sequence),
                    strength,
                )
                with torch.no_grad():
                    local_grads = _compiled_chunk_frame_vjp_one(
                        *local,
                        output_grads[0][:, start:end],
                        output_grads[1][:, start:end],
                        output_grads[2][:, start:end],
                        *adjoint,
                    )
                adjoint = local_grads[:3]
                for index in range(7):
                    sequence_grads[index][:, start:end] = local_grads[index + 3]
                strength_grad += local_grads[10]
            return adjoint, tuple(sequence_grads), strength_grad

        with torch.no_grad():
            full_chunks = u.shape[1] // ctx.chunk_size
            if full_chunks:
                full_length = full_chunks * ctx.chunk_size
                packed_initial = (
                    boundary_m[:, :, :full_chunks].permute(0, 2, 1).reshape(
                        batch * full_chunks, *boundary_m.shape[1:2]
                    ),
                    boundary_J[:, :, :full_chunks].permute(0, 2, 1, 3, 4).reshape(
                        batch * full_chunks, *boundary_J.shape[1:2], *boundary_J.shape[3:]
                    ),
                    boundary_D[:, :, :full_chunks].permute(0, 2, 1, 3, 4).reshape(
                        batch * full_chunks, *boundary_D.shape[1:2], *boundary_D.shape[3:]
                    ),
                )

                def pack_full(x):
                    return x[:, :full_length].reshape(
                        batch * full_chunks, ctx.chunk_size, *x.shape[2:]
                    )

                packed_sequence = tuple(pack_full(x) for x in saved[3:10])
                packed_output_grads = (
                    pack_full(grad_d), pack_full(grad_e), pack_full(grad_chi)
                )
                initial_grads, sequence_grads, strength_grad = backprop_packed(
                    packed_initial, packed_sequence, packed_output_grads
                )
                gradients[0][:, :, :full_chunks] = initial_grads[0].reshape(
                    batch, full_chunks, *boundary_m.shape[1:2]
                ).permute(0, 2, 1)
                gradients[1][:, :, :full_chunks] = initial_grads[1].reshape(
                    batch, full_chunks, *boundary_J.shape[1:2], *boundary_J.shape[3:]
                ).permute(0, 2, 1, 3, 4)
                gradients[2][:, :, :full_chunks] = initial_grads[2].reshape(
                    batch, full_chunks, *boundary_D.shape[1:2], *boundary_D.shape[3:]
                ).permute(0, 2, 1, 3, 4)
                for index, local_grad in enumerate(sequence_grads, start=3):
                    gradients[index][:, :full_length] = local_grad.reshape(
                        batch, full_length, *saved[index].shape[2:]
                    )
                gradients[10] += strength_grad

            if full_chunks < chunks:
                start = full_chunks * ctx.chunk_size
                tail_initial = (
                    boundary_m[:, :, full_chunks],
                    boundary_J[:, :, full_chunks],
                    boundary_D[:, :, full_chunks],
                )
                tail_sequence = tuple(x[:, start:] for x in saved[3:10])
                tail_output_grads = (
                    grad_d[:, start:], grad_e[:, start:], grad_chi[:, start:]
                )
                initial_grads, sequence_grads, strength_grad = backprop_packed(
                    tail_initial, tail_sequence, tail_output_grads
                )
                gradients[0][:, :, full_chunks] = initial_grads[0]
                gradients[1][:, :, full_chunks] = initial_grads[1]
                gradients[2][:, :, full_chunks] = initial_grads[2]
                for index, local_grad in enumerate(sequence_grads, start=3):
                    gradients[index][:, start:] = local_grad
                gradients[10] += strength_grad
        return (*gradients, None)


def cuda_chunk_solve_frame128(
    boundary_m: torch.Tensor,
    boundary_J: torch.Tensor,
    boundary_D: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    keys: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    skew: torch.Tensor,
    strength: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunk-owned fused geometry reconstruction and Solve-Frame autograd path.

    Inputs ``u``, ``keys``, and ``query`` are already L2-normalized. Geometry
    boundaries come from :func:`triton_geometry_chunk_scan`. The first native
    specialization is CUDA FP32, ``r=128``, ``K=1``, and SM120.
    """
    _load_mathdx()
    tensors = (
        boundary_m, boundary_J, boundary_D, u, h, geometry_log_decay,
        keys, erase, query, skew, strength,
    )
    if any(x.device != u.device or x.device.type != "cuda" for x in tensors):
        raise ValueError("all fused chunk-frame inputs must share one CUDA device")
    if any(x.dtype != torch.float32 for x in tensors):
        raise TypeError("fused chunk-frame inputs must be FP32")
    return _CudaChunkSolveFrame128.apply(*tensors, chunk_size)
