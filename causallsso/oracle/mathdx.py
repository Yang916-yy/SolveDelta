from __future__ import annotations

from pathlib import Path

import torch

from causallsso.reference import apply_dual_reference, apply_primal_reference


_LOADED = False


def _library_candidates() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[2]
    return (
        root / "build" / "native" / "libcausallsso_mathdx_oracle.so",
        root / "build" / "libcausallsso_mathdx_oracle.so",
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
        "the optional SolveDelta MathDx oracle is not built; configure with "
        "-DCAUSALLSSO_BUILD_MATHDX_ORACLE=ON"
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
            raise ValueError("MathDx oracle supports only 128x128 factors")
        if rhs.shape[-2:] != (128, 2):
            raise ValueError("MathDx oracle requires exactly two right-hand sides")
        if factor.dtype != torch.float32 or rhs.dtype != torch.float32:
            raise TypeError("MathDx oracle inputs must be FP32")
        if factor.device != rhs.device or factor.device.type != "cuda":
            raise ValueError("MathDx oracle inputs must share one CUDA device")
        batch_shape = torch.broadcast_shapes(factor.shape[:-2], rhs.shape[:-2])
        factor = factor.expand(*batch_shape, 128, 128)
        rhs = rhs.expand(*batch_shape, 128, 2)
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
        grad_rhs = _MathDxTRSM128.apply(
            factor.transpose(-1, -2).contiguous(),
            grad_out.contiguous(),
            not ctx.upper,
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
            lower.contiguous(),
            diagonal.contiguous(),
            upper.contiguous(),
            keys.contiguous(),
            dual_rhs,
        )
        ctx.save_for_backward(lower, diagonal, upper, keys, erase, query)
        return write_direction, dual[..., :2, :], dual[..., 2, :]

    @staticmethod
    def backward(ctx, grad_d, grad_e, grad_chi):
        with torch.enable_grad():
            inputs = tuple(x.detach().requires_grad_(True) for x in ctx.saved_tensors)
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
    """Run the optional exact r=128, nrhs=2 triangular oracle."""
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
    """Apply an exact r=128 frame for validation only."""
    if lower.shape[-2:] != (128, 128) or upper.shape != lower.shape:
        raise ValueError("lower and upper must have equal [..., 128, 128] shapes")
    if diagonal.shape != lower.shape[:-1]:
        raise ValueError("diagonal must have shape [..., 128]")
    if keys.ndim != lower.ndim or keys.shape[:-2] != lower.shape[:-2]:
        raise ValueError("keys must have shape [..., K, 128]")
    edits = keys.shape[-2]
    if edits not in (1, 2) or keys.shape[-1] != 128:
        raise ValueError("the MathDx oracle supports K in {1, 2}")
    if erase.shape != keys.shape or query.shape != (*lower.shape[:-2], 128):
        raise ValueError("erase/query shapes do not match the frame batch")
    tensors = (lower, diagonal, upper, keys, erase, query)
    if any(x.device != lower.device for x in tensors):
        raise ValueError("all frame tensors must share one device")
    if any(x.dtype != torch.float32 for x in tensors):
        raise TypeError("MathDx frame inputs must be FP32")
    if dual_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("dual_dtype must be FP32, BF16, or FP16")

    padded_keys = keys
    padded_erase = erase
    if edits == 1:
        padded_keys = torch.cat((keys, torch.zeros_like(keys)), dim=-2)
        padded_erase = torch.cat((erase, torch.zeros_like(erase)), dim=-2)
    if dual_dtype == torch.float32:
        d, e, chi = _MathDxSolveFrame128.apply(
            lower, diagonal, upper, padded_keys, padded_erase, query
        )
        return d[..., :edits, :], e[..., :edits, :], chi

    primal = mathdx_trsm128(
        lower, padded_keys.transpose(-1, -2).contiguous(), upper=False
    )
    primal = primal / diagonal.unsqueeze(-1)
    primal = mathdx_trsm128(upper, primal, upper=True)
    d = primal.transpose(-1, -2).contiguous()[..., :edits, :]

    dual_rhs = torch.cat((padded_erase, query.unsqueeze(-2)), dim=-2)
    batch_count = lower.numel() // (128 * 128)
    dual = torch.bmm(
        lower.transpose(-1, -2).to(dual_dtype).reshape(batch_count, 128, 128),
        dual_rhs.transpose(-1, -2).to(dual_dtype).reshape(batch_count, 128, 3),
    ).float().reshape(*lower.shape[:-2], 128, 3)
    dual = diagonal.unsqueeze(-1) * dual
    dual = torch.bmm(
        upper.transpose(-1, -2).to(dual_dtype).reshape(batch_count, 128, 128),
        dual.to(dual_dtype).reshape(batch_count, 128, 3),
    ).float().reshape(*lower.shape[:-2], 128, 3)
    dual = dual.transpose(-1, -2).contiguous()
    return d, dual[..., :edits, :], dual[..., 2, :]
