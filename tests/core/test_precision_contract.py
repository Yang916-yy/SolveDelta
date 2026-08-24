import inspect

import torch
import torch.nn.functional as F

from causallsso.ops.chunk_wy import chunk_wy_solvedelta


def test_chunk_wy_exposes_no_precision_or_backend_switch() -> None:
    parameters = inspect.signature(chunk_wy_solvedelta).parameters
    assert "backend" not in parameters
    assert "wy_dtype" not in parameters
    assert "chunk_size" not in parameters


def test_bf16_quantized_operands_retain_a_2pow12_cancellation() -> None:
    master = F.normalize(
        torch.tensor([0.6953125, 0.71875], dtype=torch.float64), dim=0
    )
    operand = master.to(torch.bfloat16).double()
    assert torch.equal(
        operand,
        torch.tensor([0.6953125, 0.71875], dtype=torch.float64),
    )

    local = operand[0] * operand[1]
    boundary = torch.tensor(-0.5, dtype=torch.float64)
    residual = boundary + local
    kappa = (boundary.abs() + local.abs()) / residual.abs()

    assert local == 0.5 - 2.0**-12
    assert residual == -(2.0**-12)
    assert kappa == 4095.0


def test_bf16_observability_distinguishes_zero_centered_and_diagonal_residuals() -> None:
    radius = 2.0**-3
    residual = radius * 2.0**-10
    assert residual == 2.0**-13

    # A zero-centered coordinate follows its own exponent and remains visible.
    assert torch.tensor(residual, dtype=torch.bfloat16).float() == residual

    # The same perturbation is below one BF16 ulp when merged with identity.
    identity_centered = torch.tensor(1.0 + residual, dtype=torch.bfloat16)
    assert identity_centered.float() == 1.0
    assert torch.finfo(torch.bfloat16).eps == 2.0**-7

    # The radial quadratic sees the square of the normalized residual. It is
    # far below BF16 unit roundoff even though the zero-centered factor remains
    # representable, so q2 is not an independent observable at this fixture.
    quadratic_effect = (residual / radius) ** 2
    assert quadratic_effect == 2.0**-20
    assert quadratic_effect < 2.0**-8


def test_fp32_normalization_can_write_a_more_accurate_fp16_private_panel() -> None:
    source = torch.tensor([1.0, 0.3, -0.2, 0.1], dtype=torch.bfloat16)
    normalized_fp32 = F.normalize(source.float(), dim=0)
    private_fp16 = normalized_fp32.to(torch.float16)
    public_bf16 = normalized_fp32.to(torch.bfloat16)

    fp16_error = torch.linalg.vector_norm(private_fp16.float() - normalized_fp32)
    bf16_error = torch.linalg.vector_norm(public_bf16.float() - normalized_fp32)
    assert not torch.equal(private_fp16.float(), public_bf16.float())
    assert fp16_error <= bf16_error


def test_bf16_to_fp16_cast_does_not_recover_precision_for_representable_values() -> None:
    source = torch.tensor(
        [0.6953125, 0.71875, -2.0, 2.0**-20], dtype=torch.bfloat16
    )
    round_trip = source.to(torch.float16).to(torch.bfloat16)
    assert torch.equal(round_trip, source)
    assert torch.equal(source.to(torch.float16).float(), source.float())
    assert torch.finfo(torch.float16).eps == 2.0**-10


def test_initial_private_fp16_panels_have_analytic_range_certificates() -> None:
    torch.manual_seed(0)
    normalized = F.normalize(torch.randn(8, 128, dtype=torch.float32), dim=-1)
    erase_gate = 2.0 * torch.rand_like(normalized)
    erase_source = erase_gate * normalized

    assert torch.all(torch.linalg.vector_norm(normalized, dim=-1) <= 1.0 + 1e-6)
    assert torch.all(torch.linalg.vector_norm(erase_source, dim=-1) <= 2.0 + 1e-6)

    c = 0.25
    s_max = 0.25
    primal_bound = torch.exp(torch.tensor(s_max)) / (1.0 - c) ** 2
    dual_bound = (1.0 + c) ** 2 * torch.exp(torch.tensor(s_max))
    assert primal_bound.item() < 2.283
    assert dual_bound.item() < 2.007
    assert (2.0 * dual_bound).item() < 4.013
    assert (2.0 * dual_bound).item() < torch.finfo(torch.float16).max

    rounded_strict_radius = (1.0 + 2.0**-11) * c
    rounded_primal_bound = torch.exp(torch.tensor(s_max)) / (
        1.0 - rounded_strict_radius
    ) ** 2
    rounded_dual_bound = (1.0 + rounded_strict_radius) ** 2 * torch.exp(
        torch.tensor(s_max)
    )
    assert rounded_primal_bound.item() < 2.284
    assert rounded_dual_bound.item() < 2.007
    assert (2.0 * rounded_dual_bound).item() < 4.014


def test_exact_zero_j_cancellation_witness_has_a_psd_boundary() -> None:
    u = torch.full((4,), 0.5, dtype=torch.bfloat16).double()
    boundary_j = torch.full((4, 4), -0.25, dtype=torch.float64)
    boundary_j.diagonal().fill_(0.75)
    assert torch.linalg.eigvalsh(boundary_j).min() >= -1e-12

    updated_j = boundary_j + torch.outer(u, u)
    strict_mask = ~torch.eye(4, dtype=torch.bool)
    assert torch.count_nonzero(updated_j[strict_mask]) == 0

    boundary_d = -torch.outer(u, u)
    assert torch.count_nonzero(boundary_d + torch.outer(u, u)) == 0
