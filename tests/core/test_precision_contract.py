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
