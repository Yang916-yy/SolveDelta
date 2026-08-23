from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from causallsso.ops.chunk_frame import chunk_frame
from causallsso.reference import (
    apply_dual_reference,
    apply_primal_reference,
    bounded_ldu_reference,
)


_INPUT_NAMES = (
    "u",
    "h",
    "geometry_log_decay",
    "keys",
    "erase",
    "query",
    "geometry_strength",
    "boundary_m",
    "boundary_J",
    "boundary_D",
)


def _inputs(
    *,
    length: int,
    edits: int,
    chunk_size: int,
    heads: int = 2,
    rank: int = 4,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(916 + length + edits)
    batch = 1
    chunks = (length + chunk_size - 1) // chunk_size
    dtype = torch.float64
    raw_J = 0.08 * torch.randn(
        batch, heads, chunks, rank, rank, dtype=dtype
    )
    boundary_J = raw_J @ raw_J.transpose(-1, -2)
    return {
        "u": F.normalize(
            torch.randn(batch, length, heads, rank, dtype=dtype), dim=-1
        ),
        "h": 0.2 * torch.randn(batch, length, heads, rank, dtype=dtype),
        "geometry_log_decay": -0.3
        * torch.rand(batch, length, heads, dtype=dtype),
        "keys": F.normalize(
            torch.randn(batch, length, heads, edits, rank, dtype=dtype), dim=-1
        ),
        "erase": 2.0
        * torch.rand(batch, length, heads, edits, rank, dtype=dtype),
        "query": F.normalize(
            torch.randn(batch, length, heads, rank, dtype=dtype), dim=-1
        ),
        "geometry_strength": torch.sigmoid(
            torch.randn(heads, dtype=dtype)
        ),
        "boundary_m": 0.5
        + torch.rand(batch, heads, chunks, dtype=dtype),
        "boundary_J": boundary_J,
        "boundary_D": 0.08
        * torch.randn(batch, heads, chunks, rank, rank, dtype=dtype),
    }


def _token_local_oracle(
    u: torch.Tensor,
    h: torch.Tensor,
    geometry_log_decay: torch.Tensor,
    keys: torch.Tensor,
    erase: torch.Tensor,
    query: torch.Tensor,
    geometry_strength: torch.Tensor,
    boundary_m: torch.Tensor,
    boundary_J: torch.Tensor,
    boundary_D: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = u.shape[1]
    edits = keys.shape[-2]
    d_tokens: list[torch.Tensor] = []
    e_tokens: list[torch.Tensor] = []
    chi_tokens: list[torch.Tensor] = []
    for chunk, start in enumerate(range(0, length, chunk_size)):
        current_m = boundary_m[:, :, chunk]
        current_J = boundary_J[:, :, chunk]
        current_D = boundary_D[:, :, chunk]
        for token in range(start, min(start + chunk_size, length)):
            decay = torch.exp(geometry_log_decay[:, token])
            u_t = u[:, token]
            h_t = h[:, token]
            current_m = decay * current_m + 1.0
            current_J = (
                decay[..., None, None] * current_J
                + u_t[..., :, None] * u_t[..., None, :]
            )
            current_D = (
                decay[..., None, None] * current_D
                + u_t[..., :, None] * h_t[..., None, :]
            )
            lower, diagonal, upper = bounded_ldu_reference(
                current_J / current_m[..., None, None],
                current_D / current_m[..., None, None],
                geometry_strength,
            )

            token_d: list[torch.Tensor] = []
            token_e: list[torch.Tensor] = []
            for edit in range(edits):
                key = keys[:, token, :, edit]
                b = erase[:, token, :, edit] * key
                token_d.append(
                    apply_primal_reference(lower, diagonal, upper, key)
                )
                token_e.append(
                    apply_dual_reference(lower, diagonal, upper, b)
                )
            d_tokens.append(torch.stack(token_d, dim=-2))
            e_tokens.append(torch.stack(token_e, dim=-2))
            chi_tokens.append(
                apply_dual_reference(lower, diagonal, upper, query[:, token])
            )
    return (
        torch.stack(d_tokens, dim=1),
        torch.stack(e_tokens, dim=1),
        torch.stack(chi_tokens, dim=1),
    )


@pytest.mark.parametrize(
    ("length", "edits", "chunk_size"),
    ((1, 1, 4), (7, 1, 4), (8, 2, 4), (9, 2, 4)),
)
def test_chunk_frame_matches_token_local_oracle(
    length: int, edits: int, chunk_size: int
) -> None:
    inputs = _inputs(length=length, edits=edits, chunk_size=chunk_size)
    expected = _token_local_oracle(**inputs, chunk_size=chunk_size)
    actual = chunk_frame(**inputs, chunk_size=chunk_size)
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(
            actual_tensor, expected_tensor, rtol=2e-12, atol=2e-12
        )


def test_chunk_frame_identity_chart_is_exact() -> None:
    inputs = _inputs(length=5, edits=2, chunk_size=3)
    inputs["geometry_strength"] = torch.zeros_like(
        inputs["geometry_strength"]
    )
    d, e, chi = chunk_frame(**inputs, chunk_size=3)
    torch.testing.assert_close(d, inputs["keys"], rtol=0, atol=0)
    torch.testing.assert_close(
        e, inputs["erase"] * inputs["keys"], rtol=0, atol=0
    )
    torch.testing.assert_close(chi, inputs["query"], rtol=0, atol=0)


def test_chunk_frame_preserves_primal_dual_pairing() -> None:
    inputs = _inputs(length=5, edits=2, chunk_size=3)
    d, e, _ = chunk_frame(**inputs, chunk_size=3)
    expected = (inputs["erase"] * inputs["keys"].square()).sum(dim=-1)
    torch.testing.assert_close((d * e).sum(dim=-1), expected, rtol=2e-12, atol=2e-12)


def test_chunk_frame_has_no_cross_chunk_recurrence() -> None:
    chunk_size = 3
    inputs = _inputs(length=6, edits=2, chunk_size=chunk_size)
    expected = chunk_frame(**inputs, chunk_size=chunk_size)
    changed = dict(inputs)
    changed["u"] = inputs["u"].clone()
    changed["h"] = inputs["h"].clone()
    changed["geometry_log_decay"] = inputs["geometry_log_decay"].clone()
    changed["u"][:, :chunk_size] = F.normalize(
        10.0 + changed["u"][:, :chunk_size], dim=-1
    )
    changed["h"][:, :chunk_size] *= -100.0
    changed["geometry_log_decay"][:, :chunk_size] = -1000.0
    actual = chunk_frame(**changed, chunk_size=chunk_size)
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(
            actual_tensor[:, chunk_size:],
            expected_tensor[:, chunk_size:],
            rtol=0,
            atol=0,
        )


def _cancellation_inputs() -> dict[str, torch.Tensor]:
    inputs = _inputs(length=5, edits=1, chunk_size=3, heads=1, rank=4)
    dtype = inputs["u"].dtype
    left = F.normalize(torch.tensor([1.0, -2.0, 3.0, 4.0], dtype=dtype), dim=0)
    right = F.normalize(torch.tensor([-3.0, 1.0, 4.0, -2.0], dtype=dtype), dim=0)
    inputs["boundary_m"] = torch.ones_like(inputs["boundary_m"])
    inputs["boundary_D"] = (
        4096.0 * torch.outer(left, right)[None, None, None]
    ).expand_as(inputs["boundary_D"]).clone()
    for token in (0, 3):
        inputs["u"][:, token, 0] = left
        inputs["h"][:, token, 0] = (-4096.0 + 0.01) * right
        inputs["geometry_log_decay"][:, token, 0] = 0.0
    inputs["geometry_log_decay"][:, 1, 0] = -1000.0
    inputs["geometry_log_decay"][:, 4, 0] = -1000.0
    return inputs


def test_chunk_frame_underflow_and_driven_cancellation_remain_exact_and_finite() -> None:
    inputs = _cancellation_inputs()
    expected = _token_local_oracle(**inputs, chunk_size=3)
    actual = chunk_frame(**inputs, chunk_size=3)
    for expected_tensor, actual_tensor in zip(expected, actual):
        assert torch.isfinite(actual_tensor).all()
        torch.testing.assert_close(
            actual_tensor, expected_tensor, rtol=2e-11, atol=2e-11
        )


def _vjp(
    implementation,
    inputs: dict[str, torch.Tensor],
    output_grads: tuple[torch.Tensor, ...],
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, ...]:
    variables = {
        name: value.detach().requires_grad_(True) for name, value in inputs.items()
    }
    outputs = implementation(**variables, chunk_size=chunk_size)
    return torch.autograd.grad(
        outputs,
        tuple(variables[name] for name in _INPUT_NAMES),
        output_grads,
        allow_unused=False,
    )


@pytest.mark.parametrize("case", ("tail_k2", "identity_k1", "cancellation"))
def test_chunk_frame_all_vjps_match_token_local_oracle(case: str) -> None:
    if case == "tail_k2":
        chunk_size = 3
        inputs = _inputs(length=5, edits=2, chunk_size=chunk_size)
    elif case == "identity_k1":
        chunk_size = 4
        inputs = _inputs(length=7, edits=1, chunk_size=chunk_size)
        inputs["geometry_strength"] = torch.zeros_like(
            inputs["geometry_strength"]
        )
    else:
        chunk_size = 3
        inputs = _cancellation_inputs()

    with torch.no_grad():
        outputs = _token_local_oracle(**inputs, chunk_size=chunk_size)
        torch.manual_seed(20260823 + len(case))
        output_grads = tuple(torch.randn_like(output) for output in outputs)
    expected = _vjp(
        _token_local_oracle, inputs, output_grads, chunk_size=chunk_size
    )
    actual = _vjp(chunk_frame, inputs, output_grads, chunk_size=chunk_size)
    for name, expected_grad, actual_grad in zip(
        _INPUT_NAMES, expected, actual
    ):
        assert torch.isfinite(actual_grad).all(), name
        torch.testing.assert_close(
            actual_grad, expected_grad, rtol=3e-10, atol=3e-10, msg=name
        )


def test_chunk_frame_rejects_mismatched_boundary_count() -> None:
    inputs = _inputs(length=5, edits=1, chunk_size=3)
    inputs["boundary_m"] = inputs["boundary_m"][:, :, :1]
    with pytest.raises(ValueError, match="boundary_m"):
        chunk_frame(**inputs, chunk_size=3)
