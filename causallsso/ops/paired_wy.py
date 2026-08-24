# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in LICENSES/MIT.txt.
# The fixed C32 inverse, wide-RHS application, and matrix reverse below adapt
# kernel blocks from Flash Linear Attention v0.5.2; see
# THIRD_PARTY_NOTICES.md for exact provenance.

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl


_CHUNK = 32
_RANK = 128


class PairedWYSolution(NamedTuple):
    edit: torch.Tensor
    value: torch.Tensor


class PairedWYAdjoint(NamedTuple):
    edit_rhs: torch.Tensor
    system: torch.Tensor
    write: torch.Tensor
    value: torch.Tensor


# Adapted from FLA v0.5.2:
# fla/ops/utils/solve_tril.py::merge_16x16_to_32x32_inverse_kernel and
# fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py::wu_fwd_kernel.
@triton.jit
def _inverse_c32_blocks(system, panel):
    offsets = tl.arange(0, 16)
    strict_lower = offsets[:, None] > offsets[None, :]
    identity = offsets[:, None] == offsets[None, :]
    base = system + panel * 32 * 32
    system_00 = tl.load(
        base + offsets[:, None] * 32 + offsets[None, :]
    ).to(tl.float32)
    system_10 = tl.load(
        base
        + (offsets[:, None] + 16) * 32
        + offsets[None, :]
    ).to(tl.float32)
    system_11 = tl.load(
        base
        + (offsets[:, None] + 16) * 32
        + offsets[None, :]
        + 16
    ).to(tl.float32)

    inverse_00 = -tl.where(strict_lower, system_00, 0.0)
    inverse_11 = -tl.where(strict_lower, system_11, 0.0)
    for row in range(2, 16):
        current_00 = -tl.load(base + row * 32 + offsets)
        current_00 = tl.where(offsets < row, current_00, 0.0)
        current_00 += tl.sum(
            current_00[:, None] * inverse_00, axis=0
        )
        inverse_00 = tl.where(
            (offsets == row)[:, None], current_00, inverse_00
        )

        current_11 = -tl.load(
            base + (row + 16) * 32 + offsets + 16
        )
        current_11 = tl.where(offsets < row, current_11, 0.0)
        current_11 += tl.sum(
            current_11[:, None] * inverse_11, axis=0
        )
        inverse_11 = tl.where(
            (offsets == row)[:, None], current_11, inverse_11
        )

    inverse_00 += identity
    inverse_11 += identity
    inverse_10 = -tl.dot(
        tl.dot(inverse_11, system_10, input_precision="ieee"),
        inverse_00,
        input_precision="ieee",
    )
    return inverse_00, inverse_10, inverse_11


@triton.jit
def _bf16_dot(left, right):
    return tl.dot(
        left.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        right.to(tl.bfloat16, fp_downcast_rounding="rtne"),
    )


@triton.jit(do_not_specialize=["T"])
def _paired_wy_forward_kernel(
    system,
    erase_dual,
    inclusive_decay,
    write,
    value,
    edit_solution,
    value_solution,
    T,
    H: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    N: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    inverse_00, inverse_10, inverse_11 = _inverse_c32_blocks(
        system, panel
    )

    offsets = tl.arange(0, 16)
    chunk = panel % N
    head_batch = panel // N
    batch = head_batch // H
    head = head_batch % H
    tokens_0 = chunk * 32 + offsets
    tokens_1 = tokens_0 + 16
    valid_0 = tokens_0 < T
    valid_1 = tokens_1 < T

    for start in tl.static_range(0, R, BR):
        coordinates = start + tl.arange(0, BR)
        mask_0 = valid_0[:, None] & (coordinates[None, :] < R)
        mask_1 = valid_1[:, None] & (coordinates[None, :] < R)
        vector_base = (batch * T * H + head) * R
        locations_0 = (
            vector_base
            + tokens_0[:, None] * H * R
            + coordinates[None, :]
        )
        locations_1 = (
            vector_base
            + tokens_1[:, None] * H * R
            + coordinates[None, :]
        )
        erase_0 = tl.load(
            erase_dual + locations_0, mask=mask_0, other=0.0
        ).to(tl.float32)
        erase_1 = tl.load(
            erase_dual + locations_1, mask=mask_1, other=0.0
        ).to(tl.float32)
        gate_0 = tl.load(
            inclusive_decay + locations_0,
            mask=mask_0,
            other=float("-inf"),
        ).to(tl.float32)
        gate_1 = tl.load(
            inclusive_decay + locations_1,
            mask=mask_1,
            other=float("-inf"),
        ).to(tl.float32)
        rhs_0 = erase_0 * tl.exp(gate_0)
        rhs_1 = erase_1 * tl.exp(gate_1)
        result_0 = _bf16_dot(inverse_00, rhs_0)
        result_1 = (
            _bf16_dot(inverse_10, rhs_0)
            + _bf16_dot(inverse_11, rhs_1)
        )
        tl.store(
            edit_solution + locations_0,
            result_0.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=mask_0,
        )
        tl.store(
            edit_solution + locations_1,
            result_1.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=mask_1,
        )

    for start in tl.static_range(0, V, BV):
        coordinates = start + tl.arange(0, BV)
        mask_0 = valid_0[:, None] & (coordinates[None, :] < V)
        mask_1 = valid_1[:, None] & (coordinates[None, :] < V)
        value_base = (batch * T * H + head) * V
        locations_0 = (
            value_base
            + tokens_0[:, None] * H * V
            + coordinates[None, :]
        )
        locations_1 = (
            value_base
            + tokens_1[:, None] * H * V
            + coordinates[None, :]
        )
        write_0 = tl.load(
            write + locations_0, mask=mask_0, other=0.0
        ).to(tl.float32)
        write_1 = tl.load(
            write + locations_1, mask=mask_1, other=0.0
        ).to(tl.float32)
        value_0 = tl.load(
            value + locations_0, mask=mask_0, other=0.0
        ).to(tl.float32)
        value_1 = tl.load(
            value + locations_1, mask=mask_1, other=0.0
        ).to(tl.float32)
        rhs_0 = write_0 * value_0
        rhs_1 = write_1 * value_1
        result_0 = _bf16_dot(inverse_00, rhs_0)
        result_1 = (
            _bf16_dot(inverse_10, rhs_0)
            + _bf16_dot(inverse_11, rhs_1)
        )
        tl.store(
            value_solution + locations_0,
            result_0.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=mask_0,
        )
        tl.store(
            value_solution + locations_1,
            result_1.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=mask_1,
        )


# Matrix reverse adapted from FLA v0.5.2
# fla/ops/generalized_delta_rule/dplr/wy_fast_bwd.py.
@triton.jit(do_not_specialize=["T"])
def _paired_wy_backward_kernel(
    system,
    edit_solution,
    value_solution,
    write,
    value,
    grad_edit_solution,
    grad_value_solution,
    grad_edit_rhs,
    grad_system,
    grad_write,
    grad_value,
    T,
    H: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    N: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
):
    panel = tl.program_id(0).to(tl.int64)
    inverse_00, inverse_10, inverse_11 = _inverse_c32_blocks(
        system, panel
    )
    inverse_t_00 = tl.trans(inverse_00)
    inverse_t_10 = tl.trans(inverse_10)
    inverse_t_11 = tl.trans(inverse_11)

    offsets = tl.arange(0, 16)
    chunk = panel % N
    head_batch = panel // N
    batch = head_batch // H
    head = head_batch % H
    tokens_0 = chunk * 32 + offsets
    tokens_1 = tokens_0 + 16
    valid_0 = tokens_0 < T
    valid_1 = tokens_1 < T
    system_00 = tl.zeros((16, 16), tl.float32)
    system_10 = tl.zeros((16, 16), tl.float32)
    system_11 = tl.zeros((16, 16), tl.float32)

    for start in tl.static_range(0, R, BR):
        coordinates = start + tl.arange(0, BR)
        mask_0 = valid_0[:, None] & (coordinates[None, :] < R)
        mask_1 = valid_1[:, None] & (coordinates[None, :] < R)
        vector_base = (batch * T * H + head) * R
        locations_0 = (
            vector_base
            + tokens_0[:, None] * H * R
            + coordinates[None, :]
        )
        locations_1 = (
            vector_base
            + tokens_1[:, None] * H * R
            + coordinates[None, :]
        )
        solution_0 = tl.load(
            edit_solution + locations_0, mask=mask_0, other=0.0
        )
        solution_1 = tl.load(
            edit_solution + locations_1, mask=mask_1, other=0.0
        )
        output_bar_0 = tl.load(
            grad_edit_solution + locations_0, mask=mask_0, other=0.0
        ).to(tl.float32)
        output_bar_1 = tl.load(
            grad_edit_solution + locations_1, mask=mask_1, other=0.0
        ).to(tl.float32)
        rhs_bar_0 = (
            _bf16_dot(inverse_t_00, output_bar_0)
            + _bf16_dot(inverse_t_10, output_bar_1)
        )
        rhs_bar_1 = _bf16_dot(inverse_t_11, output_bar_1)
        tl.store(grad_edit_rhs + locations_0, rhs_bar_0, mask=mask_0)
        tl.store(grad_edit_rhs + locations_1, rhs_bar_1, mask=mask_1)
        system_00 -= tl.dot(
            rhs_bar_0.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            tl.trans(solution_0),
        )
        system_10 -= tl.dot(
            rhs_bar_1.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            tl.trans(solution_0),
        )
        system_11 -= tl.dot(
            rhs_bar_1.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            tl.trans(solution_1),
        )

    for start in tl.static_range(0, V, BV):
        coordinates = start + tl.arange(0, BV)
        mask_0 = valid_0[:, None] & (coordinates[None, :] < V)
        mask_1 = valid_1[:, None] & (coordinates[None, :] < V)
        value_base = (batch * T * H + head) * V
        locations_0 = (
            value_base
            + tokens_0[:, None] * H * V
            + coordinates[None, :]
        )
        locations_1 = (
            value_base
            + tokens_1[:, None] * H * V
            + coordinates[None, :]
        )
        solution_0 = tl.load(
            value_solution + locations_0, mask=mask_0, other=0.0
        )
        solution_1 = tl.load(
            value_solution + locations_1, mask=mask_1, other=0.0
        )
        output_bar_0 = tl.load(
            grad_value_solution + locations_0, mask=mask_0, other=0.0
        ).to(tl.float32)
        output_bar_1 = tl.load(
            grad_value_solution + locations_1, mask=mask_1, other=0.0
        ).to(tl.float32)
        rhs_bar_0 = (
            _bf16_dot(inverse_t_00, output_bar_0)
            + _bf16_dot(inverse_t_10, output_bar_1)
        )
        rhs_bar_1 = _bf16_dot(inverse_t_11, output_bar_1)
        write_0 = tl.load(
            write + locations_0, mask=mask_0, other=0.0
        ).to(tl.float32)
        write_1 = tl.load(
            write + locations_1, mask=mask_1, other=0.0
        ).to(tl.float32)
        value_0 = tl.load(
            value + locations_0, mask=mask_0, other=0.0
        ).to(tl.float32)
        value_1 = tl.load(
            value + locations_1, mask=mask_1, other=0.0
        ).to(tl.float32)
        tl.store(
            grad_write + locations_0,
            (rhs_bar_0 * value_0).to(
                tl.bfloat16, fp_downcast_rounding="rtne"
            ),
            mask=mask_0,
        )
        tl.store(
            grad_write + locations_1,
            (rhs_bar_1 * value_1).to(
                tl.bfloat16, fp_downcast_rounding="rtne"
            ),
            mask=mask_1,
        )
        tl.store(
            grad_value + locations_0,
            (rhs_bar_0 * write_0).to(
                tl.bfloat16, fp_downcast_rounding="rtne"
            ),
            mask=mask_0,
        )
        tl.store(
            grad_value + locations_1,
            (rhs_bar_1 * write_1).to(
                tl.bfloat16, fp_downcast_rounding="rtne"
            ),
            mask=mask_1,
        )
        system_00 -= tl.dot(
            rhs_bar_0.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            tl.trans(solution_0),
        )
        system_10 -= tl.dot(
            rhs_bar_1.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            tl.trans(solution_0),
        )
        system_11 -= tl.dot(
            rhs_bar_1.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            tl.trans(solution_1),
        )

    strict_lower = offsets[:, None] > offsets[None, :]
    base = grad_system + panel * 32 * 32
    tl.store(
        base + offsets[:, None] * 32 + offsets[None, :],
        tl.where(
            strict_lower & valid_0[:, None] & valid_0[None, :],
            system_00,
            0.0,
        ),
    )
    tl.store(
        base
        + (offsets[:, None] + 16) * 32
        + offsets[None, :],
        tl.where(valid_1[:, None] & valid_0[None, :], system_10, 0.0),
    )
    tl.store(
        base
        + (offsets[:, None] + 16) * 32
        + offsets[None, :]
        + 16,
        tl.where(
            strict_lower & valid_1[:, None] & valid_1[None, :],
            system_11,
            0.0,
        ),
    )
    tl.store(
        base
        + offsets[:, None] * 32
        + offsets[None, :]
        + 16,
        0.0,
    )


def paired_wy_forward(
    system: torch.Tensor,
    erase_dual: torch.Tensor,
    inclusive_decay: torch.Tensor,
    write: torch.Tensor,
    value: torch.Tensor,
) -> PairedWYSolution:
    batch, length, heads, rank = erase_dual.shape
    value_dim = value.shape[-1]
    chunks = triton.cdiv(length, _CHUNK)
    # The solve result is not analytically FP16-bounded. Accumulate in FP32
    # and cross the private/public boundary only once when storing BF16.
    edit_solution = torch.empty_like(erase_dual, dtype=torch.bfloat16)
    value_solution = torch.empty(
        batch,
        length,
        heads,
        value_dim,
        device=value.device,
        dtype=value.dtype,
    )
    _paired_wy_forward_kernel[(batch * heads * chunks,)](
        system,
        erase_dual,
        inclusive_decay,
        write,
        value,
        edit_solution,
        value_solution,
        T=length,
        H=heads,
        R=rank,
        V=value_dim,
        N=chunks,
        BR=32,
        BV=32,
        num_warps=2,
        num_stages=2,
    )
    return PairedWYSolution(edit_solution, value_solution)


def paired_wy_backward(
    system: torch.Tensor,
    edit_solution: torch.Tensor,
    value_solution: torch.Tensor,
    write: torch.Tensor,
    value: torch.Tensor,
    grad_edit_solution: torch.Tensor,
    grad_value_solution: torch.Tensor,
) -> PairedWYAdjoint:
    batch, length, heads, rank = edit_solution.shape
    value_dim = value_solution.shape[-1]
    chunks = triton.cdiv(length, _CHUNK)
    grad_edit_rhs = torch.empty_like(
        grad_edit_solution, dtype=torch.float32
    )
    grad_system = torch.empty_like(system, dtype=torch.float32)
    grad_write = torch.empty_like(write)
    grad_value = torch.empty_like(value)
    _paired_wy_backward_kernel[(batch * heads * chunks,)](
        system,
        edit_solution,
        value_solution,
        write,
        value,
        grad_edit_solution,
        grad_value_solution,
        grad_edit_rhs,
        grad_system,
        grad_write,
        grad_value,
        T=length,
        H=heads,
        R=rank,
        V=value_dim,
        N=chunks,
        BR=32,
        BV=32,
        num_warps=2,
        num_stages=2,
    )
    return PairedWYAdjoint(
        grad_edit_rhs, grad_system, grad_write, grad_value
    )


__all__ = ["paired_wy_backward", "paired_wy_forward"]
