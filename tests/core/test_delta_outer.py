import pytest
import torch

from causallsso.ops import fla_dplr_delta_outer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _rho(reference: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    reference = reference.double()
    actual = actual.double()
    return (actual - reference).square().mean().sqrt() / (
        reference.square().mean().sqrt() + 1e-8
    )


def _outer_oracle(chi, d, e, z, log_decay, initial_state):
    state = initial_state
    outputs = []
    for t in range(chi.shape[1]):
        state = torch.exp(log_decay[:, t])[..., None] * state
        for edit in range(d.shape[3]):
            innovation = z[:, t, :, edit] - torch.einsum(
                "bhrv,bhr->bhv", state, e[:, t, :, edit]
            )
            state = state + torch.einsum(
                "bhr,bhv->bhrv", d[:, t, :, edit], innovation
            )
        outputs.append(torch.einsum("bhrv,bhr->bhv", state, chi[:, t]))
    return torch.stack(outputs, dim=1), state


@pytest.mark.parametrize("edits", [1, 2, 4])
def test_fla_dplr_outer_matches_fp64_oracle(edits: int) -> None:
    torch.manual_seed(916 + edits)
    batch, length, heads, rank, value_dim = 1, 33, 2, 32, 24
    shape = (batch, length, heads)
    chi = torch.randn(*shape, rank, device="cuda", dtype=torch.bfloat16)
    d = (0.05 * torch.randn(*shape, edits, rank, device="cuda")).to(torch.bfloat16)
    e = (0.05 * torch.randn_like(d.float())).to(torch.bfloat16)
    z = (0.1 * torch.randn(*shape, edits, value_dim, device="cuda")).to(torch.bfloat16)
    log_decay = (-0.05 * torch.rand(*shape, rank, device="cuda")).to(torch.bfloat16)
    initial = (0.1 * torch.randn(batch, heads, rank, value_dim, device="cuda"))
    actual, final = fla_dplr_delta_outer(
        chi, d, e, z, log_decay,
        initial_state=initial,
        output_final_state=True,
    )
    expected, expected_final = _outer_oracle(
        chi.double(), d.double(), e.double(), z.double(), log_decay.double(), initial.double()
    )
    assert _rho(expected, actual) < 5e-3
    assert _rho(expected_final, final) < 5e-3


def test_fla_dplr_outer_backward_is_finite() -> None:
    torch.manual_seed(919)
    batch, length, heads, edits, rank, value_dim = 1, 17, 1, 2, 32, 16
    def variable(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    chi = variable(batch, length, heads, rank)
    d = variable(batch, length, heads, edits, rank)
    e = variable(batch, length, heads, edits, rank)
    z = variable(batch, length, heads, edits, value_dim)
    raw_decay = variable(batch, length, heads, rank)
    log_decay = -torch.nn.functional.softplus(raw_decay.float()).to(torch.bfloat16)
    output, final = fla_dplr_delta_outer(
        chi, d, e, z, log_decay, output_final_state=True
    )
    loss = output.float().square().mean() + final.float().square().mean()
    loss.backward()
    for tensor in (chi, d, e, z, raw_decay):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("edits", [1, 2, 4])
def test_fla_dplr_outer_backward_matches_fp64_oracle(edits: int) -> None:
    """The fused K-expansion must preserve every DPLR input VJP."""
    torch.manual_seed(1200 + edits)
    batch, length, heads, rank, value_dim = 1, 19, 1, 24, 16
    shape = (batch, length, heads)

    def runtime_variable(value: torch.Tensor) -> torch.Tensor:
        return value.to(torch.float16).detach().requires_grad_(True)

    runtime = (
        runtime_variable(torch.randn(*shape, rank, device="cuda")),
        runtime_variable(0.04 * torch.randn(*shape, edits, rank, device="cuda")),
        runtime_variable(0.04 * torch.randn(*shape, edits, rank, device="cuda")),
        runtime_variable(0.08 * torch.randn(*shape, edits, value_dim, device="cuda")),
        runtime_variable(-0.03 * torch.rand(*shape, rank, device="cuda")),
        runtime_variable(0.03 * torch.randn(batch, heads, rank, value_dim, device="cuda")),
    )
    reference = tuple(x.detach().double().requires_grad_(True) for x in runtime)
    upstream_output = torch.randn(*shape, value_dim, device="cuda")
    upstream_state = torch.randn(batch, heads, rank, value_dim, device="cuda")

    actual_output, actual_state = fla_dplr_delta_outer(
        *runtime[:5], initial_state=runtime[5], output_final_state=True
    )
    actual_loss = (
        (actual_output.float() * upstream_output).sum()
        + (actual_state.float() * upstream_state).sum()
    )
    actual_loss.backward()

    expected_output, expected_state = _outer_oracle(
        *reference[:5], reference[5]
    )
    expected_loss = (
        (expected_output * upstream_output.double()).sum()
        + (expected_state * upstream_state.double()).sum()
    )
    expected_loss.backward()

    for expected, actual in zip(reference, runtime):
        assert actual.grad is not None
        assert torch.isfinite(actual.grad).all()
        assert _rho(expected.grad, actual.grad) < 1e-2
