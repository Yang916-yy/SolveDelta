from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import triton

from causallsso.ops.strict_chart import strict_chart_direct_transpose


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

_CHUNK = 32
_RANK = 128
_ROUTES = 3


def _inputs(
    panels: int = 2,
    *,
    valid_count: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(20260824)
    descriptor_shape = (panels, _CHUNK, _ROUTES, _RANK)
    descriptors = tuple(
        (
            torch.randn(
                descriptor_shape, generator=generator, device="cuda"
            )
            / _RANK**0.5
        )
        .bfloat16()
        .contiguous()
        for _ in range(4)
    )
    u = (
        torch.randn(
            panels,
            _CHUNK,
            _RANK,
            generator=generator,
            device="cuda",
        )
        / _RANK**0.5
    ).bfloat16()
    h = (
        torch.randn(
            panels,
            _CHUNK,
            _RANK,
            generator=generator,
            device="cuda",
        )
        / _RANK**0.5
    ).bfloat16()
    boundary_j = 0.025 * torch.randn(
        panels, _RANK, _RANK, generator=generator, device="cuda"
    )
    boundary_d = 0.025 * torch.randn(
        panels, _RANK, _RANK, generator=generator, device="cuda"
    )
    theta = 0.1 + 0.8 * torch.rand(
        panels, _CHUNK, generator=generator, device="cuda"
    )
    weights = 0.04 * torch.tril(
        torch.rand(
            panels,
            _CHUNK,
            _CHUNK,
            generator=generator,
            device="cuda",
        )
    )
    radial_scale = 0.12 * (
        torch.rand(
            panels, _CHUNK, 4, generator=generator, device="cuda"
        )
        - 0.5
    )
    if valid_count is None:
        valid_count = torch.full(
            (panels,), _CHUNK, device="cuda", dtype=torch.int32
        )
    return tuple(
        tensor.contiguous()
        for tensor in (
            *descriptors,
            u,
            h,
            boundary_j,
            boundary_d,
            theta,
            weights,
            radial_scale,
            valid_count,
        )
    )


def _dense_descriptors(
    left: torch.Tensor,
    right: torch.Tensor,
    valid: torch.Tensor,
    *,
    upper: bool,
) -> torch.Tensor:
    dense = torch.einsum(
        "ptai,ptaj->ptij", left.double(), right.double()
    )
    dense = torch.triu(dense, diagonal=1) if upper else torch.tril(
        dense, diagonal=-1
    )
    return dense * valid[:, :, None, None]


def _fp64_oracle(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    (
        lower_left,
        lower_right,
        upper_left,
        upper_right,
        u,
        h,
        boundary_j,
        boundary_d,
        theta,
        weights,
        radial_scale,
        valid_count,
    ) = inputs
    token = torch.arange(_CHUNK, device=u.device)
    valid = token[None, :] < valid_count[:, None]
    causal = (
        (token[None, None, :] <= token[None, :, None])
        & valid[:, :, None]
        & valid[:, None, :]
    )
    u64 = u.double()
    h64 = h.double()
    theta64 = torch.where(valid, theta.double(), 0.0)
    weights64 = torch.where(causal, weights.double(), 0.0)
    scale64 = torch.where(valid[:, :, None], radial_scale.double(), 0.0)
    boundary_j64 = boundary_j.double()
    boundary_d64 = boundary_d.double()
    lower = _dense_descriptors(
        lower_left, lower_right, valid, upper=False
    )
    upper = _dense_descriptors(
        upper_left, upper_right, valid, upper=True
    )
    moment_j = theta64[:, :, None, None] * boundary_j64[:, None]
    moment_j += torch.einsum(
        "pts,psi,psj->ptij", weights64, u64, u64
    )
    moment_d = theta64[:, :, None, None] * boundary_d64[:, None]
    moment_d += torch.einsum(
        "pts,psi,psj->ptij", weights64, u64, h64
    )

    grad_radial_scale = torch.stack(
        (
            (lower * moment_j).sum(dim=(-2, -1)),
            (lower * moment_d).sum(dim=(-2, -1)),
            (upper * moment_j).sum(dim=(-2, -1)),
            (upper * moment_d).sum(dim=(-2, -1)),
        ),
        dim=-1,
    )
    direct_j = (
        scale64[:, :, 0, None, None] * lower
        + scale64[:, :, 2, None, None] * upper
    )
    direct_d = (
        scale64[:, :, 1, None, None] * lower
        + scale64[:, :, 3, None, None] * upper
    )
    grad_theta = (direct_j * boundary_j64[:, None]).sum(dim=(-2, -1))
    grad_theta += (direct_d * boundary_d64[:, None]).sum(dim=(-2, -1))
    grad_weights = torch.einsum(
        "ptij,psi,psj->pts", direct_j, u64, u64
    )
    grad_weights += torch.einsum(
        "ptij,psi,psj->pts", direct_d, u64, h64
    )
    grad_weights = torch.where(causal, grad_weights, 0.0)
    grad_u = torch.einsum(
        "pts,ptij,psj->psi", weights64, direct_j, u64
    )
    grad_u += torch.einsum(
        "pts,ptji,psj->psi", weights64, direct_j, u64
    )
    grad_u += torch.einsum(
        "pts,ptij,psj->psi", weights64, direct_d, h64
    )
    grad_h = torch.einsum(
        "pts,ptij,psi->psj", weights64, direct_d, u64
    )
    grad_u = torch.where(valid[:, :, None], grad_u, 0.0)
    grad_h = torch.where(valid[:, :, None], grad_h, 0.0)
    grad_boundary_j = torch.einsum(
        "pt,ptij->pij", theta64, direct_j
    )
    grad_boundary_d = torch.einsum(
        "pt,ptij->pij", theta64, direct_d
    )
    return (
        grad_radial_scale,
        grad_theta,
        grad_weights,
        grad_u,
        grad_h,
        grad_boundary_j,
        grad_boundary_d,
    )


def _rho(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.double() - expected.double()
    return float(
        difference.square().mean().sqrt()
        / (expected.double().square().mean().sqrt() + 1e-8)
    )


def _assert_matches_oracle(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    *,
    ceiling: float = 2.0e-4,
) -> None:
    for name, actual_tensor, expected_tensor in zip(
        (
            "radial_scale",
            "theta",
            "weights",
            "u",
            "h",
            "boundary_j",
            "boundary_d",
        ),
        actual,
        expected,
        strict=True,
    ):
        assert torch.isfinite(actual_tensor).all(), name
        rho = _rho(actual_tensor, expected_tensor)
        assert rho <= ceiling, f"{name} rho={rho:.3e}"


def test_strict_chart_matches_explicit_fp64_transpose() -> None:
    valid_count = torch.tensor([32, 19], device="cuda", dtype=torch.int32)
    inputs = _inputs(valid_count=valid_count)
    actual = strict_chart_direct_transpose(*inputs)
    expected = _fp64_oracle(inputs)
    _assert_matches_oracle(actual, expected)


def test_strict_chart_nan_tail_is_structurally_ignored() -> None:
    valid_count = torch.tensor([5, 19], device="cuda", dtype=torch.int32)
    clean = _inputs(valid_count=valid_count)
    dirty = [tensor.clone() for tensor in clean]
    token = torch.arange(_CHUNK, device="cuda")
    for panel, count in enumerate(valid_count.tolist()):
        for descriptor in dirty[:4]:
            descriptor[panel, count:] = torch.nan
        dirty[4][panel, count:] = torch.nan
        dirty[5][panel, count:] = torch.nan
        dirty[8][panel, count:] = torch.nan
        dirty[10][panel, count:] = torch.nan
        invalid_weight = (
            (token[:, None] >= count)
            | (token[None, :] >= count)
            | (token[None, :] > token[:, None])
        )
        dirty[9][panel][invalid_weight] = torch.nan
    dirty = tuple(tensor.contiguous() for tensor in dirty)

    clean_output = strict_chart_direct_transpose(*clean)
    dirty_output = strict_chart_direct_transpose(*dirty)
    for clean_tensor, dirty_tensor in zip(
        clean_output, dirty_output, strict=True
    ):
        assert torch.equal(clean_tensor, dirty_tensor)
        assert torch.isfinite(dirty_tensor).all()

    for panel, count in enumerate(valid_count.tolist()):
        assert torch.count_nonzero(dirty_output.grad_radial_scale[panel, count:]) == 0
        assert torch.count_nonzero(dirty_output.grad_theta[panel, count:]) == 0
        assert torch.count_nonzero(dirty_output.grad_weights[panel, count:]) == 0
        assert torch.count_nonzero(dirty_output.grad_weights[panel, :, count:]) == 0
        assert torch.count_nonzero(dirty_output.grad_u[panel, count:]) == 0
        assert torch.count_nonzero(dirty_output.grad_h[panel, count:]) == 0


def test_strict_chart_preserves_4095_to_one_cancellation() -> None:
    first, second = 0.6953125, 0.71875
    shape = (1, _CHUNK, _ROUTES, _RANK)
    lower_left = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    lower_right = torch.zeros_like(lower_left)
    upper_left = torch.zeros_like(lower_left)
    upper_right = torch.zeros_like(lower_left)
    lower_left[0, 0, 0, 1] = 1.0
    lower_right[0, 0, 0, 0] = 1.0
    upper_left[0, 0, 0, 0] = 1.0
    upper_right[0, 0, 0, 1] = 1.0
    u = torch.zeros(1, _CHUNK, _RANK, device="cuda", dtype=torch.bfloat16)
    h = torch.zeros_like(u)
    u[0, 0, 0], u[0, 0, 1] = first, second
    h[0, 0, 0], h[0, 0, 1] = first, second
    boundary_j = torch.zeros(1, _RANK, _RANK, device="cuda")
    boundary_d = torch.zeros_like(boundary_j)
    boundary_j[0, 0, 1] = -0.5
    boundary_j[0, 1, 0] = -0.5
    boundary_d.copy_(boundary_j)
    theta = torch.zeros(1, _CHUNK, device="cuda")
    theta[0, 0] = 0.5
    weights = torch.zeros(1, _CHUNK, _CHUNK, device="cuda")
    weights[0, 0, 0] = 0.5
    radial_scale = torch.zeros(1, _CHUNK, 4, device="cuda")
    radial_scale[0, 0] = torch.tensor([0.7, -0.4, 0.3, -0.2], device="cuda")
    valid_count = torch.tensor([1], device="cuda", dtype=torch.int32)
    inputs = tuple(
        tensor.contiguous()
        for tensor in (
            lower_left,
            lower_right,
            upper_left,
            upper_right,
            u,
            h,
            boundary_j,
            boundary_d,
            theta,
            weights,
            radial_scale,
            valid_count,
        )
    )
    actual = strict_chart_direct_transpose(*inputs)
    expected = _fp64_oracle(inputs)
    residual = 0.5 - first * second
    assert (0.5 + first * second) / residual == 4095.0
    expected_scale = torch.full(
        (4,), -0.5 * residual, device="cuda", dtype=torch.double
    )
    torch.testing.assert_close(
        actual.grad_radial_scale[0, 0].double(),
        expected_scale,
        rtol=1e-6,
        atol=1e-10,
    )
    _assert_matches_oracle(actual, expected, ceiling=2.0e-5)


def test_strict_chart_transpose_inner_product() -> None:
    valid_count = torch.tensor([7], device="cuda", dtype=torch.int32)
    inputs = _inputs(panels=1, valid_count=valid_count)
    actual = strict_chart_direct_transpose(*inputs)
    generator = torch.Generator(device="cuda").manual_seed(20260825)
    u, h = inputs[4], inputs[5]
    tangents = (
        torch.randn_like(inputs[10], generator=generator),
        torch.randn_like(inputs[8], generator=generator),
        torch.randn_like(inputs[9], generator=generator),
        torch.randn(u.shape, device="cuda", generator=generator),
        torch.randn(h.shape, device="cuda", generator=generator),
        torch.randn_like(inputs[6], generator=generator),
        torch.randn_like(inputs[7], generator=generator),
    )
    token = torch.arange(_CHUNK, device="cuda")
    valid = token[None, :] < valid_count[:, None]
    causal = (
        (token[None, None, :] <= token[None, :, None])
        & valid[:, :, None]
        & valid[:, None, :]
    )
    tangents = list(tangents)
    tangents[0] = torch.where(valid[:, :, None], tangents[0], 0.0)
    tangents[1] = torch.where(valid, tangents[1], 0.0)
    tangents[2] = torch.where(causal, tangents[2], 0.0)
    tangents[3] = torch.where(valid[:, :, None], tangents[3], 0.0)
    tangents[4] = torch.where(valid[:, :, None], tangents[4], 0.0)
    left_inner = sum(
        (gradient.double() * tangent.double()).sum()
        for gradient, tangent in zip(actual, tangents, strict=True)
    )

    lower = _dense_descriptors(inputs[0], inputs[1], valid, upper=False)
    upper = _dense_descriptors(inputs[2], inputs[3], valid, upper=True)
    u64, h64 = inputs[4].double(), inputs[5].double()
    boundary_j, boundary_d = inputs[6].double(), inputs[7].double()
    theta = torch.where(valid, inputs[8].double(), 0.0)
    weights = torch.where(causal, inputs[9].double(), 0.0)
    scale = torch.where(valid[:, :, None], inputs[10].double(), 0.0)
    dscale, dtheta, dweights, du, dh, dbj, dbd = (
        tangent.double() for tangent in tangents
    )
    moment_j = theta[:, :, None, None] * boundary_j[:, None]
    moment_j += torch.einsum("pts,psi,psj->ptij", weights, u64, u64)
    moment_d = theta[:, :, None, None] * boundary_d[:, None]
    moment_d += torch.einsum("pts,psi,psj->ptij", weights, u64, h64)
    dmoment_j = dtheta[:, :, None, None] * boundary_j[:, None]
    dmoment_j += theta[:, :, None, None] * dbj[:, None]
    dmoment_j += torch.einsum("pts,psi,psj->ptij", dweights, u64, u64)
    dmoment_j += torch.einsum("pts,psi,psj->ptij", weights, du, u64)
    dmoment_j += torch.einsum("pts,psi,psj->ptij", weights, u64, du)
    dmoment_d = dtheta[:, :, None, None] * boundary_d[:, None]
    dmoment_d += theta[:, :, None, None] * dbd[:, None]
    dmoment_d += torch.einsum("pts,psi,psj->ptij", dweights, u64, h64)
    dmoment_d += torch.einsum("pts,psi,psj->ptij", weights, du, h64)
    dmoment_d += torch.einsum("pts,psi,psj->ptij", weights, u64, dh)
    lower_jvp = (
        dscale[:, :, 0, None, None] * moment_j
        + scale[:, :, 0, None, None] * dmoment_j
        + dscale[:, :, 1, None, None] * moment_d
        + scale[:, :, 1, None, None] * dmoment_d
    )
    upper_jvp = (
        dscale[:, :, 2, None, None] * moment_j
        + scale[:, :, 2, None, None] * dmoment_j
        + dscale[:, :, 3, None, None] * moment_d
        + scale[:, :, 3, None, None] * dmoment_d
    )
    right_inner = (lower * lower_jvp).sum() + (upper * upper_jvp).sum()
    relative = (left_inner - right_inner).abs() / (right_inner.abs() + 1e-8)
    assert float(relative) < 2.0e-4


def test_strict_chart_is_bitwise_repeatable_and_standalone() -> None:
    valid_count = torch.tensor(
        [32, 7, 31, 1], device="cuda", dtype=torch.int32
    )
    inputs = _inputs(panels=4, valid_count=valid_count)
    first = strict_chart_direct_transpose(*inputs)
    second = strict_chart_direct_transpose(*inputs)
    for first_tensor, second_tensor in zip(first, second, strict=True):
        assert torch.equal(first_tensor, second_tensor)

    source = (
        Path(__file__).resolve().parents[2]
        / "causallsso"
        / "ops"
        / "strict_chart.py"
    ).read_text()
    assert "tl.dot" in source
    assert "tl.cumsum" not in source
    assert "diagonal_grad_u" not in source
    assert "_strict_diagonal_reduce_kernel" not in source
    assert "torch.autograd" not in source
    assert "c32_frame_compact_pair" not in source
    assert "c32_frame_compact_coefficients" not in source
    assert "c32_frame_compact_leaf" not in source


@pytest.mark.skipif(
    os.environ.get("CAUSALLSSO_RUN_BENCHMARKS") != "1",
    reason="set CAUSALLSSO_RUN_BENCHMARKS=1 to run target benchmark",
)
def test_strict_chart_target_benchmark() -> None:
    inputs = _inputs(panels=256)
    strict_chart_direct_transpose(*inputs)
    milliseconds = triton.testing.do_bench(
        lambda: strict_chart_direct_transpose(*inputs)
    )
    print(f"strict_chart transpose P256 C32 r128: {milliseconds:.3f} ms")
    assert milliseconds < 20.0
