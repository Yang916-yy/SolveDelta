# SolveDelta

**A CUDA-native recurrent memory layer that learns the geometry of its prefix
before deciding how to edit memory.**

Most Delta-rule layers write in a fixed coordinate system. SolveDelta adds an
online matrix solver. Directional observations `u -> h` define second and
cross moments whose least-squares equation is `C J = G`; a normalized-LMS
step advances the solution `C` directly. The accumulated solution directs the
primal write, while the current solve residual supplies a bounded dual action
for erase and read:

```text
prefix observations -> online geometry solve -> accumulated primal write
                    -> residual-local dual erase/read -> Delta memory
```

The recurrent state is two fixed-size FP32 matrices: `C`, the geometry
solution, and `S`, the Delta memory. This solution-coordinate form captures
the useful history of a second-order fit while keeping recurrence causal and
streamable. Setting the geometry rate to zero recovers the ordinary GDN2
edit/read path exactly.

This repository includes the FP64 mathematical oracle, a BF16/FP32 CUDA path
specialized from mature Flash Linear Attention primitives, and a complete
Hugging Face causal LM with recurrent cache and CUDA Graph training.

Keywords: **recurrent memory**, **Delta rule**, **online least squares**,
**second-order statistics**, **linear attention**, **causal language
modeling**, **PyTorch**, **Triton**, and **CUDA**.

## Highlights

- **Second-order geometry in solution coordinates.** Prefix covariance and
  cross-moment fitting supply the problem; normalized-LMS supplies a fast
  rank-one iteration. The full `r x r` solution evolves without carrying a
  covariance inverse through the sequence.
- **History-bearing primal, stable residual dual.** The write direction uses
  the full accumulated frame `I+C`, while erase/query use an analytically
  conditioned rank-one inverse-transpose aligned with the current residual.
  No inverse of the accumulated matrix is required.
- **A strict GDN2 reduction.** At finite `gamma=0`, `F=I` and the memory path
  becomes GDN2's channel-wise Delta edit/read. Geometry is an added capability,
  not a replacement memory rule.
- **Full matrix history from rank-one work.** Each token contributes one
  normal-equation residual, while chunked pair/WY/state kernels expose enough
  parallelism for training.
- **Production model integration.** The repository includes a BF16/FP32 native
  path, complete transpose, recurrent cache, Hugging Face CausalLM, conv4,
  fused loss, CUDA Graph training, and DDP integration.
- **An executable oracle.** `causallsso/reference.py` defines the FP64
  recurrence used to verify native outputs, continuation states, and VJPs.

## Installation

The optimized path targets Linux on an NVIDIA CUDA GPU with native BF16
support. Python `3.10+` is supported; install a CUDA-enabled PyTorch build for
your driver first, then install SolveDelta:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Install the appropriate CUDA-enabled torch build from pytorch.org first.
python -m pip install -e ".[native,test]"
python -m pytest -q -s
```

The current validated stack is Python `3.12`, PyTorch `2.13.0+cu130`, Triton
`3.7.1`, Transformers `5.15.0`, FLA `0.6.0`, and causal-conv1d `1.7.0` on an
RTX 5070 Ti. These are validated versions, not a claim that older supported
versions have identical performance. See [Environment and Installation](docs/ENVIRONMENT.md)
for the dependency matrix, source-build requirements, verification commands,
and native-path constraints.

## Quick Start

Construct a causal LM through the Hugging Face auto classes:

```python
import torch
from transformers import AutoModelForCausalLM

from causallsso import SolveDeltaConfig

config = SolveDeltaConfig(
    hidden_size=1024,
    num_heads=8,
    head_k_dim=128,
    head_v_dim=128,
    num_hidden_layers=24,
    vocab_size=32000,
)
model = AutoModelForCausalLM.from_config(config).cuda().train()
input_ids = torch.randint(0, config.vocab_size, (2, 1024), device="cuda")

with torch.autocast("cuda", dtype=torch.bfloat16):
    result = model(input_ids=input_ids, labels=input_ids)
result.loss.backward()
```

Use FP32 master parameters with BF16 autocast. The fused projection emits
BF16 public operands while geometry parameters and recurrent state remain
FP32.

For the standalone mixer:

```python
import torch

from causallsso import SolveDelta, SolveDeltaConfig

mixer = SolveDelta(
    SolveDeltaConfig(
        hidden_size=256,
        num_heads=4,
        head_k_dim=64,
        head_v_dim=64,
    )
).cuda()
hidden = torch.randn(2, 128, 256, device="cuda")

with torch.autocast("cuda", dtype=torch.bfloat16):
    output = mixer(hidden)
```

## CUDA Graph Training

Fixed-shape dense CausalLM training can capture the loss forward and matching
backward:

```python
from causallsso import SolveDeltaGraphedTrainingStep

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
graph_step = SolveDeltaGraphedTrainingStep(
    model,
    sample_input_ids=input_ids,
    sample_labels=input_ids,
)

for batch_ids, batch_labels in batches:
    optimizer.zero_grad(set_to_none=True)
    loss = graph_step(batch_ids, batch_labels)
    loss.backward()
    optimizer.step()
```

One helper instance owns one `[B,T]` shape and CUDA device. Backward must run
before its next replay. The optimizer, clipping, gradient accumulation, and
distributed reduction remain outside capture. Masks, resets, recurrent cache,
gradient checkpointing, and module hooks are outside this dense graph surface.
Ordinary fused cross entropy and fixed-dense fused-linear cross entropy are
both graph-safe; the latter requires replay labels without user-supplied
ignore entries. Construct and replay the helper on one CUDA stream.

Two optional training modes are available: optimizer-bound BF16 Linear shadows
amortize weight casts across gradient-accumulation microbatches, and
`distributed=True` installs DDP after local graph capture so NCCL reduction
stays outside the graph. Their memory tradeoffs, stream rules, and complete
examples live in [Environment and Installation](docs/ENVIRONMENT.md).

## How It Works

For a prefix of directional observations, define

```text
J_t = sum_{s<=t} u_s u_s^T
G_t = sum_{s<=t} h_s u_s^T.
```

The stationary equation of `1/2 sum ||h_s-C u_s||^2` is `C J_t = G_t`.
SolveDelta advances `C` with one online residual step for each normalized
direction `u_t`, target `h_t`, and learned relaxation
`gamma_t in (0,1)`:

```text
D_t = Diag(exp(log_alpha_t))
r_t = h_t - C_{t-1} u_t
delta_t = gamma_t r_t
C_t = C_{t-1} D_t + delta_t u_t^T.
```

This is a leaky normalized-LMS step in solution coordinates. The residual is
measured against the complete old solution before forgetting, so current
innovation is unchanged. `C` reuses DeltaRule's complete channel-retention
field on its source/address axis; it adds no gate projection, scalar reduction,
or parameter. This orientation is aligned with the primal readout because
`(C_t D_t)^T k = D_t C_t^T k`. Ordering, relaxation, and coordinate-selective
turnover make `C_t` a recurrent online solution rather than a batch closed form.
Equivalently, the geometry retention exponent is fixed at `rho_h=1`.

The accumulated primal frame and residual-local dual factor are

```text
rho = 5/8
phi_t = rho delta_t / sqrt(rho^2 + ||u_t||^2 ||delta_t||^2)
F_t = I + u_t phi_t^T
F_t^-T x = x - phi_t (u_t^T x) / (1 + phi_t^T u_t).
```

The radial map is identity to first order and gives
`3/8 < 1+phi_t^T u_t < 13/8`, while leaving the predictor update unchanged.

With normalized edit key `k_t`, erase covector `b_t`, and query `q_t`:

```text
e_t   = F_t^-T b_t
chi_t = F_t^-T q_t.
```

The primal line is evaluated from the complete solution:

```text
P_t = I + C_t
d_t = P_t^T k_t.
```

The asymmetry is deliberate: accumulated history controls the stable primal
action, while the dual follows the current innovation without forming
`C_t^-1`.

The channel-decayed memory applies GDN2's independent channel-wise write and
erase gates in that relative frame and reads with `chi_t`. The core output then
passes through a low-rank sigmoid gate and standard RMSNorm. The geometry bias
initializes to `logit(0.9)`, treating `gamma` as normalized-LMS relaxation. See
`docs/INNOVATION_PROGRAM.md` for the derivation and
`docs/FROM_SCRATCH_REBUILD.md` for exact execution ownership.

## Native Implementation

The CUDA path specializes concrete mature primitives around SolveDelta's
private panel ownership:

- stride-aware FLA L2Norm for the predictor source;
- FLA coordinate-gated Oja pair, triangular WY, matrix-state/output forward,
  and matching transpose for the pre-decay residual predictor and accumulated
  primal action;
- a shared key normalization owner and fused source owner for residual-local
  dual/query panels plus erase/write transposes;
- an exact unbounded Triton specialization of FLA's direct-`e` pair
  forward/transpose;
- FLA generalized-DPLR WY, state/output, and output-owned reverse;
- FLA KDA coordinate-decay and output-gate projections plus FLA's
  sigmoid-gated RMSNorm owner and exact transpose.

Each producer writes its consumer's panel layout directly. State and output
stay selectively split to preserve chunk/rank/value CTA parallelism.

## Model and Cache

`SolveDeltaForCausalLM` uses an FLA-style prenorm block, a packed gate-up
GatedMLP, final RMSNorm, LM head, and optional fused cross entropy. It registers
with `AutoConfig`, `AutoModel`, and `AutoModelForCausalLM`.

`past_key_values` stores each layer's FP32 `(C,S)` and conv4 state. Packed
training may provide `cu_seqlens`; explicit resets may provide
`solvedelta_reset_mask`. Invalid tokens and resets are compacted into native
reset-free segment batches. Non-C16 tails use neutral private padding, while
no-grad single-token cache steps use recurrent predictor and DPLR owners.

## Performance Snapshot

The current accumulated-primal core was measured on an RTX 5070 Ti at
`B=1,T=1024,H=8,r=V=128`, BF16 autocast, FP32 master parameters/state, and
CUDA Graph replay. Final continuation output is disabled.

| Scope | Forward median/p95 | Backward median/p95 | F+B median/p95 | Graph allocated |
| --- | ---: | ---: | ---: | ---: |
| Core operator | 0.264 / 0.286 ms | 0.639 / 0.837 ms | 0.905 / 1.114 ms | 63.0 MiB |
| Projected mixer | 0.515 / 0.696 ms | 1.222 / 1.423 ms | 1.760 / 1.993 ms | 208.1 MiB |
| Full model block | 0.761 / 0.959 ms | 1.910 / 1.954 ms | 2.695 / 2.764 ms | 276.1 MiB |

These measurements cover the right-coordinate forgetting contract. Its
initial matched landing comparison increased core F+B from `0.796` to
`0.908 ms` and projected-mixer F+B from `1.592` to `1.753 ms`, with unchanged
allocation. The cost was localized to the coordinate-aware pair and strict
transpose, not an added gate projection or continuation state. The table above
includes the subsequent implementation audit.

Scope, numerical quality, the exploratory 200M-token comparison, and complete
measurement conditions are recorded in
[Evaluation and Performance](docs/RESULTS.md).

## Correctness

The acceptance suite covers:

- FP64 residual prediction and exact relative-frame identities;
- the analytic `3/8` denominator lower bound under adversarial residuals;
- finite-parameter GDN2 reduction at `gamma=0`;
- masks, resets, recurrent splits, and a non-128 width;
- BF16 native output and FP32 final `(C,S)`;
- complete VJPs including initial and final state;
- public CausalLM loss/backward, cache, checkpoint, native dispatch, and CUDA
  Graph gradient equality.

Run it with:

```bash
python -m pytest -q -s
```

## Current Limits

- C16-aligned dense training is the tuned surface; tail and reset-density
  performance has not received the same breadth of profiling.
- CUDA Graph training remains fixed-shape and excludes masks, resets, cache,
  hooks, and gradient checkpointing.
- Broader multi-seed and history-conditioned retrieval evaluation remains the
  main model-quality target.

## Contributing

Read `AGENTS.md` before using an AI coding tool or changing operator code. It
defines the authority order, mathematical and precision contracts, provenance
requirements, test gates, and benchmark reporting rules. Material upstream
reuse is documented in `docs/PRIOR_ART.md` and
`THIRD_PARTY_NOTICES.md`.

SolveDelta is research software under active development. No repository-wide
license has yet been selected for original code; review
`THIRD_PARTY_NOTICES.md` before redistribution.

## Repository Map

- `causallsso/reference.py`: sole FP64 mathematical oracle;
- `causallsso/model.py`: projected SolveDelta mixer;
- `causallsso/modeling_solvedelta.py`: FLA/Hugging Face causal LM;
- `causallsso/ops/residual_frame/`: selected native forward and transpose;
- `docs/FROM_SCRATCH_REBUILD.md`: native implementation blueprint;
- `docs/INNOVATION_PROGRAM.md`: derivation and interpretation;
- `docs/ENVIRONMENT.md`: supported environment, installation, and diagnosis;
- `docs/RESULTS.md`: scoped performance and exploratory model evidence;
- `docs/PRIOR_ART.md`: upstream provenance and production decisions;
- `THIRD_PARTY_NOTICES.md`: adapted-source attribution.
