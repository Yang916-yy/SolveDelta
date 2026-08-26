from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parent / "csrc" / "local_transpose.cu"
    if not source.is_file():
        raise RuntimeError(f"SolveDelta native source is missing: {source}")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return load(
        name=f"causallsso_local_transpose_{source_hash}",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )


def local_transpose(
    x: torch.Tensor,
    cotangent: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    decay: torch.Tensor,
    kappa_h: torch.Tensor,
    kappa_r: torch.Tensor,
    mass: torch.Tensor,
    grad_u: torch.Tensor,
    grad_h: torch.Tensor,
    grad_kappa_h: torch.Tensor,
    grad_kappa_r: torch.Tensor,
    grad_cumulative: torch.Tensor,
    *,
    primal: bool,
    lower: bool,
    accumulate: bool,
) -> None:
    _extension().local_transpose(
        x,
        cotangent,
        u,
        h,
        decay,
        kappa_h,
        kappa_r,
        mass,
        grad_u,
        grad_h,
        grad_kappa_h,
        grad_kappa_r,
        grad_cumulative,
        lower,
        primal,
        accumulate,
    )


__all__ = ["local_transpose"]
