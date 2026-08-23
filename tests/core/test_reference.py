from __future__ import annotations

import torch
import torch.nn.functional as F

from causallsso import (
    SolveDeltaState,
    apply_dual_reference,
    apply_primal_reference,
    bounded_ldu_reference,
    solvedelta_reference,
)


def _inputs(
    *,
    batch: int = 2,
    length: int = 5,
    heads: int = 2,
    edits: int = 2,
    r: int = 4,
    value_dim: int = 3,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(916)
    dtype = torch.float64
    return {
        "u": torch.randn(batch, length, heads, r, dtype=dtype),
        "h": torch.randn(batch, length, heads, r, dtype=dtype) * 0.3,
        "q": torch.randn(batch, length, heads, r, dtype=dtype),
        "keys": torch.randn(batch, length, heads, edits, r, dtype=dtype),
        "values": torch.randn(batch, length, heads, edits, value_dim, dtype=dtype),
        "geometry_log_decay": -torch.rand(batch, length, heads, dtype=dtype),
        "associative_log_decay": -torch.rand(batch, length, heads, r, dtype=dtype),
        "erase": 2.0 * torch.rand(batch, length, heads, edits, r, dtype=dtype),
        "write": 2.0 * torch.rand(batch, length, heads, edits, value_dim, dtype=dtype),
        "skew": 2.0 * torch.rand(batch, length, heads, edits, dtype=dtype) - 1.0,
        "geometry_strength": torch.sigmoid(torch.randn(heads, dtype=dtype)),
    }


def _slice_time(inputs: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in inputs.items():
        result[name] = value if name == "geometry_strength" else value[:, start:end]
    return result


def test_identity_chart_and_exact_primal_dual_pairing() -> None:
    torch.manual_seed(1)
    batch, heads, r, rhs_count = 2, 3, 5, 4
    H = torch.randn(batch, heads, r, r, dtype=torch.float64)
    H = H @ H.transpose(-1, -2)
    H = H / torch.diagonal(H, dim1=-2, dim2=-1).sum(-1)[..., None, None]
    R = torch.randn_like(H)
    lower, diagonal, upper, omega = bounded_ldu_reference(
        H, R, torch.zeros(heads, dtype=torch.float64)
    )
    eye = torch.eye(r, dtype=torch.float64)
    torch.testing.assert_close(lower, eye.expand_as(lower), rtol=0, atol=0)
    torch.testing.assert_close(upper, eye.expand_as(upper), rtol=0, atol=0)
    torch.testing.assert_close(diagonal, torch.ones_like(diagonal), rtol=0, atol=0)
    torch.testing.assert_close(omega, torch.zeros_like(omega), rtol=0, atol=0)

    strength = torch.full((heads,), 0.7, dtype=torch.float64)
    lower, diagonal, upper, _ = bounded_ldu_reference(H, R, strength)
    a = torch.randn(batch, heads, r, rhs_count, dtype=torch.float64)
    b = torch.randn_like(a)
    d = apply_primal_reference(lower, diagonal, upper, a)
    e = apply_dual_reference(lower, diagonal, upper, b)
    M = lower @ torch.diag_embed(diagonal) @ upper
    expected_d = torch.linalg.solve(M, a)
    expected_e = M.transpose(-1, -2) @ b
    torch.testing.assert_close(d, expected_d, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(e, expected_e, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        (e * d).sum(-2), (b * a).sum(-2), rtol=1e-11, atol=1e-11
    )


def test_factor_bounds_hold_for_saturated_coordinates() -> None:
    torch.manual_seed(2)
    batch, heads, r = 4, 2, 8
    raw = 1e5 * torch.randn(batch, heads, r, r, dtype=torch.float64)
    H = raw @ raw.transpose(-1, -2)
    H = H / torch.diagonal(H, dim1=-2, dim2=-1).sum(-1)[..., None, None]
    R = 1e5 * torch.randn_like(H)
    lower, diagonal, upper, _ = bounded_ldu_reference(
        H, R, torch.ones(heads, dtype=torch.float64)
    )
    eye = torch.eye(r, dtype=torch.float64)
    assert torch.linalg.matrix_norm(lower - eye, ord="fro").max() < 0.25
    assert torch.linalg.matrix_norm(upper - eye, ord="fro").max() < 0.25
    assert diagonal.max() < torch.exp(torch.tensor(0.25, dtype=torch.float64))
    assert diagonal.min() > torch.exp(torch.tensor(-0.25, dtype=torch.float64))
    condition = torch.linalg.cond(lower @ torch.diag_embed(diagonal) @ upper)
    assert condition.max() < 4.58


def test_identity_geometry_is_exact_gdn2_recurrence() -> None:
    x = _inputs(edits=1)
    x["geometry_strength"] = torch.zeros_like(x["geometry_strength"])
    x["skew"] = torch.zeros_like(x["skew"])
    output, state = solvedelta_reference(**x)

    keys = F.normalize(x["keys"], dim=-1)
    query = F.normalize(x["q"], dim=-1)
    S = torch.zeros_like(state.S)
    expected = []
    for t in range(x["u"].shape[1]):
        S = torch.exp(x["associative_log_decay"][:, t])[..., None] * S
        key = keys[:, t, :, 0]
        erase = x["erase"][:, t, :, 0] * key
        target = x["write"][:, t, :, 0] * x["values"][:, t, :, 0]
        prediction = (S.transpose(-1, -2) @ erase.unsqueeze(-1)).squeeze(-1)
        S = S + key.unsqueeze(-1) * (target - prediction).unsqueeze(-2)
        expected.append(
            (S.transpose(-1, -2) @ query[:, t].unsqueeze(-1)).squeeze(-1)
        )
    torch.testing.assert_close(output, torch.stack(expected, dim=1), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state.S, S, rtol=1e-12, atol=1e-12)


def test_identity_geometry_contains_ordered_deltaproduct() -> None:
    x = _inputs(edits=4)
    x["geometry_strength"] = torch.zeros_like(x["geometry_strength"])
    x["skew"] = torch.zeros_like(x["skew"])
    beta = 2.0 * torch.sigmoid(torch.randn(2, 5, 2, 4, 1, dtype=torch.float64))
    x["erase"] = beta.expand_as(x["erase"])
    x["write"] = beta.expand_as(x["write"])
    output, state = solvedelta_reference(**x)

    keys = F.normalize(x["keys"], dim=-1)
    query = F.normalize(x["q"], dim=-1)
    S = torch.zeros_like(state.S)
    expected = []
    for t in range(5):
        S = torch.exp(x["associative_log_decay"][:, t])[..., None] * S
        for j in range(4):
            key = keys[:, t, :, j]
            scale = beta[:, t, :, j]
            value = x["values"][:, t, :, j]
            prediction = (S.transpose(-1, -2) @ key.unsqueeze(-1)).squeeze(-1)
            S = S + key.unsqueeze(-1) * (scale * (value - prediction)).unsqueeze(-2)
        expected.append((S.transpose(-1, -2) @ query[:, t].unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(output, torch.stack(expected, 1), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state.S, S, rtol=1e-12, atol=1e-12)


def test_recurrent_split_matches_whole_sequence() -> None:
    x = _inputs(length=7, r=5)
    whole_output, whole_state = solvedelta_reference(**x)
    first_output, first_state = solvedelta_reference(**_slice_time(x, 0, 3))
    second_output, split_state = solvedelta_reference(
        **_slice_time(x, 3, 7), initial_state=first_state
    )
    torch.testing.assert_close(
        whole_output, torch.cat((first_output, second_output), dim=1), rtol=1e-12, atol=1e-12
    )
    for whole, split in zip(whole_state, split_state):
        torch.testing.assert_close(whole, split, rtol=1e-12, atol=1e-12)


def test_masks_and_resets_have_declared_state_semantics() -> None:
    x = _inputs(batch=1, length=6)
    valid = torch.tensor([[True, True, False, True, True, True]])
    reset = torch.tensor([[False, False, False, True, False, False]])
    output, state, history = solvedelta_reference(
        **x, valid_mask=valid, reset_mask=reset, return_state_history=True
    )
    assert torch.count_nonzero(output[:, 2]) == 0
    for tensor in history:
        torch.testing.assert_close(tensor[:, 2], tensor[:, 1], rtol=0, atol=0)

    suffix = _slice_time(x, 3, 6)
    expected_output, expected_state = solvedelta_reference(**suffix)
    torch.testing.assert_close(output[:, 3:], expected_output, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(state, expected_state):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_reference_has_finite_fp64_gradients() -> None:
    x = _inputs(batch=1, length=3, heads=1, edits=2, r=3, value_dim=2)
    differentiable = (
        "u", "h", "q", "keys", "values", "geometry_log_decay",
        "associative_log_decay", "erase", "write", "skew", "geometry_strength",
    )
    for name in differentiable:
        x[name].requires_grad_(True)
    output, state = solvedelta_reference(**x)
    loss = output.square().sum() + sum(t.square().sum() for t in state)
    gradients = torch.autograd.grad(loss, [x[name] for name in differentiable])
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_primal_dual_actions_pass_gradcheck() -> None:
    torch.manual_seed(3)
    r = 3
    H = torch.eye(r, dtype=torch.float64).reshape(1, 1, r, r) / r
    R = (0.1 * torch.randn(1, 1, r, r, dtype=torch.float64)).requires_grad_()
    strength = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    a = torch.randn(1, 1, r, 2, dtype=torch.float64, requires_grad=True)
    b = torch.randn(1, 1, r, 2, dtype=torch.float64, requires_grad=True)

    def actions(R_: torch.Tensor, strength_: torch.Tensor, a_: torch.Tensor, b_: torch.Tensor):
        lower, diagonal, upper, _ = bounded_ldu_reference(H, R_, strength_)
        return (
            apply_primal_reference(lower, diagonal, upper, a_),
            apply_dual_reference(lower, diagonal, upper, b_),
        )

    assert torch.autograd.gradcheck(actions, (R, strength, a, b), rtol=1e-5, atol=1e-7)
