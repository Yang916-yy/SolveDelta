from __future__ import annotations

# The resident Gram/Hadamard organization is specialized from the MIT-licensed
# MESA kernels in Flash Linear Attention 0.5.2 (Copyright 2023-2026 Songlin
# Yang, Yu Zhang, Zhiyuan Li). SolveDelta supplies the strict-coordinate masks.

import torch
import triton
import triton.language as tl


@triton.jit
def _h_gram_forward_kernel(
    u,
    gram,
    R: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    oi = tl.arange(0, BC)
    ok = tl.arange(0, BK)
    mask_c = oi < C
    mask_k = ok < R
    pointer = u + panel * C * R + oi[:, None] * R + ok[None, :]
    value = tl.load(
        pointer, mask=mask_c[:, None] & mask_k[None, :], other=0.0
    ).to(tl.float32)
    u16 = value.to(tl.float16)
    u2 = (value * value).to(tl.float16)
    ku = tl.dot(u16, tl.trans(u16))
    ku2 = tl.dot(u2, tl.trans(u2))
    result = 0.5 * (ku * ku - ku2)
    tl.store(
        gram + panel * C * C + oi[:, None] * C + oi[None, :],
        result,
        mask=mask_c[:, None] & mask_c[None, :],
    )


@triton.jit
def _r_gram_forward_block_prefix_kernel(
    u,
    h,
    gram_lower,
    gram_upper,
    R: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NB: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    oi = tl.arange(0, BC)
    ok = tl.arange(0, BK)
    mask_c = oi < C
    prefix_u = tl.zeros([BC, BC], dtype=tl.float32)
    prefix_h = tl.zeros([BC, BC], dtype=tl.float32)
    lower = tl.zeros([BC, BC], dtype=tl.float32)
    upper = tl.zeros([BC, BC], dtype=tl.float32)
    base = panel * C * R
    for block in range(NB):
        coordinate = block * BK + ok
        mask_k = coordinate < R
        pointer = base + oi[:, None] * R + coordinate[None, :]
        u_block = tl.load(
            u + pointer,
            mask=mask_c[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        h_block = tl.load(
            h + pointer,
            mask=mask_c[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        ku = tl.dot(u_block, tl.trans(u_block))
        kh = tl.dot(h_block, tl.trans(h_block))
        lower += ku * prefix_h
        upper += kh * prefix_u

        local_prefix_u = tl.zeros([BC, BC], dtype=tl.float32)
        local_prefix_h = tl.zeros([BC, BC], dtype=tl.float32)
        for step in range(BK):
            uv = tl.sum(
                tl.where(ok[None, :] == step, u_block.to(tl.float32), 0.0),
                axis=1,
            )
            hv = tl.sum(
                tl.where(ok[None, :] == step, h_block.to(tl.float32), 0.0),
                axis=1,
            )
            valid_step = block * BK + step < R
            uu = tl.where(
                valid_step, uv[:, None] * uv[None, :], 0.0
            )
            hh = tl.where(
                valid_step, hv[:, None] * hv[None, :], 0.0
            )
            lower += uu * local_prefix_h
            upper += hh * local_prefix_u
            local_prefix_u += uu
            local_prefix_h += hh
        prefix_u += ku
        prefix_h += kh
    pair_mask = mask_c[:, None] & mask_c[None, :]
    pointer = panel * C * C + oi[:, None] * C + oi[None, :]
    tl.store(gram_lower + pointer, lower, mask=pair_mask)
    tl.store(gram_upper + pointer, upper, mask=pair_mask)


@triton.jit
def _h_gram_backward_kernel(
    u,
    grad_gram,
    grad_u,
    R: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    oi = tl.arange(0, BC)
    ok = tl.arange(0, BK)
    mask_c = oi < C
    mask_k = ok < R
    base = panel * C * R
    value = tl.load(
        u + base + oi[:, None] * R + ok[None, :],
        mask=mask_c[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)
    u16 = value.to(tl.float16)
    k = tl.dot(u16, tl.trans(u16))
    pair = panel * C * C + oi[:, None] * C + oi[None, :]
    z = tl.load(
        grad_gram + pair,
        mask=mask_c[:, None] & mask_c[None, :],
        other=0.0,
    ).to(tl.float32)
    zs = z + tl.trans(z)
    first = tl.dot((zs * k).to(tl.float16), u16)
    u2 = (value * value).to(tl.float16)
    second = value * tl.dot(zs.to(tl.float16), u2)
    tl.store(
        grad_u + base + oi[:, None] * R + ok[None, :],
        first - second,
        mask=mask_c[:, None] & mask_k[None, :],
    )


@triton.jit
def _r_gram_backward_block_scan_kernel(
    u,
    h,
    grad_lower,
    grad_upper,
    grad_u,
    grad_h,
    R: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NB: tl.constexpr,
    REVERSE: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    oi = tl.arange(0, BC)
    ok = tl.arange(0, BK)
    mask_c = oi < C
    pair = panel * C * C + oi[:, None] * C + oi[None, :]
    pair_mask = mask_c[:, None] & mask_c[None, :]
    zl = tl.load(grad_lower + pair, mask=pair_mask, other=0.0).to(tl.float32)
    zu = tl.load(grad_upper + pair, mask=pair_mask, other=0.0).to(tl.float32)
    zl = zl + tl.trans(zl)
    zu = zu + tl.trans(zu)
    base = panel * C * R
    running_u = tl.zeros([BC, BC], dtype=tl.float32)
    running_h = tl.zeros([BC, BC], dtype=tl.float32)
    for iteration in range(NB):
        block = NB - 1 - iteration if REVERSE else iteration
        coordinate = block * BK + ok
        mask_k = coordinate < R
        pointer = base + oi[:, None] * R + coordinate[None, :]
        u_block = tl.load(
            u + pointer,
            mask=mask_c[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        h_block = tl.load(
            h + pointer,
            mask=mask_c[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        z_u = zu if REVERSE else zl
        z_h = zl if REVERSE else zu
        result_u = tl.dot((z_u * running_h).to(tl.bfloat16), u_block)
        result_h = tl.dot((z_h * running_u).to(tl.bfloat16), h_block)
        local_u = tl.zeros([BC, BC], dtype=tl.float32)
        local_h = tl.zeros([BC, BC], dtype=tl.float32)
        within_u = tl.zeros([BC, BK], dtype=tl.float32)
        within_h = tl.zeros([BC, BK], dtype=tl.float32)
        for local_iteration in range(BK):
            step = BK - 1 - local_iteration if REVERSE else local_iteration
            uv = tl.sum(
                tl.where(
                    ok[None, :] == step, u_block.to(tl.float32), 0.0
                ),
                axis=1,
            )
            hv = tl.sum(
                tl.where(
                    ok[None, :] == step, h_block.to(tl.float32), 0.0
                ),
                axis=1,
            )
            valid_step = block * BK + step < R
            gu = tl.sum((z_u * local_h) * uv[None, :], axis=1)
            gh = tl.sum((z_h * local_u) * hv[None, :], axis=1)
            within_u = tl.where(
                ok[None, :] == step,
                tl.where(valid_step, gu[:, None], 0.0),
                within_u,
            )
            within_h = tl.where(
                ok[None, :] == step,
                tl.where(valid_step, gh[:, None], 0.0),
                within_h,
            )
            local_u += tl.where(
                valid_step, uv[:, None] * uv[None, :], 0.0
            )
            local_h += tl.where(
                valid_step, hv[:, None] * hv[None, :], 0.0
            )
        result_u += within_u
        result_h += within_h
        if ACCUMULATE:
            result_u += tl.load(
                grad_u + pointer,
                mask=mask_c[:, None] & mask_k[None, :],
                other=0.0,
            )
            result_h += tl.load(
                grad_h + pointer,
                mask=mask_c[:, None] & mask_k[None, :],
                other=0.0,
            )
        tl.store(
            grad_u + pointer,
            result_u,
            mask=mask_c[:, None] & mask_k[None, :],
        )
        tl.store(
            grad_h + pointer,
            result_h,
            mask=mask_c[:, None] & mask_k[None, :],
        )
        running_u += tl.dot(u_block, tl.trans(u_block))
        running_h += tl.dot(h_block, tl.trans(h_block))


class _StrictGram(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: torch.Tensor, h: torch.Tensor):
        panels, chunk_size, width = u.shape
        block_c = triton.next_power_of_2(chunk_size)
        h_block_k = triton.next_power_of_2(width)
        gram_h = torch.empty(
            panels, chunk_size, chunk_size, dtype=torch.float32, device=u.device
        )
        gram_lower = torch.empty_like(gram_h)
        gram_upper = torch.empty_like(gram_h)
        _h_gram_forward_kernel[(panels,)](
            u,
            gram_h,
            R=width,
            C=chunk_size,
            BC=block_c,
            BK=h_block_k,
            num_warps=4,
            num_stages=1,
        )
        block_k = 16
        blocks = triton.cdiv(width, block_k)
        _r_gram_forward_block_prefix_kernel[(panels,)](
            u,
            h,
            gram_lower,
            gram_upper,
            R=width,
            C=chunk_size,
            BC=block_c,
            BK=block_k,
            NB=blocks,
            num_warps=4,
            num_stages=2,
        )
        ctx.save_for_backward(u, h)
        ctx.block_c = block_c
        ctx.h_block_k = h_block_k
        return gram_h, gram_lower, gram_upper

    @staticmethod
    def backward(ctx, grad_h_gram, grad_lower, grad_upper):
        u, h = ctx.saved_tensors
        panels, chunk_size, width = u.shape
        grad_u_r = torch.empty_like(u, dtype=torch.float32)
        grad_h = torch.empty_like(h, dtype=torch.float32)
        if grad_h_gram is None:
            grad_u_h = torch.zeros_like(u, dtype=torch.float32)
        else:
            grad_u_h = torch.empty_like(u, dtype=torch.float32)
            _h_gram_backward_kernel[(panels,)](
                u,
                grad_h_gram,
                grad_u_h,
                R=width,
                C=chunk_size,
                BC=ctx.block_c,
                BK=ctx.h_block_k,
                num_warps=4,
                num_stages=1,
            )
        block_k = 16
        blocks = triton.cdiv(width, block_k)
        _r_gram_backward_block_scan_kernel[(panels,)](
            u,
            h,
            grad_lower,
            grad_upper,
            grad_u_r,
            grad_h,
            R=width,
            C=chunk_size,
            BC=ctx.block_c,
            BK=block_k,
            NB=blocks,
            REVERSE=False,
            ACCUMULATE=False,
            num_warps=4,
            num_stages=2,
        )
        _r_gram_backward_block_scan_kernel[(panels,)](
            u,
            h,
            grad_lower,
            grad_upper,
            grad_u_r,
            grad_h,
            R=width,
            C=chunk_size,
            BC=ctx.block_c,
            BK=block_k,
            NB=blocks,
            REVERSE=True,
            ACCUMULATE=True,
            num_warps=4,
            num_stages=2,
        )
        return grad_u_h + grad_u_r, grad_h


def strict_gram(
    u: torch.Tensor, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _StrictGram.apply(u, h)


__all__ = ["strict_gram"]
