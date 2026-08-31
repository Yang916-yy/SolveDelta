# SolveDelta

**A CUDA-native recurrent memory layer that solves prefix geometry online and
uses the solution as coordinates for Delta edits.**

Each prefix supplies directional observations `u -> h`. Their second moment
`J=sum(u u^T)` and cross moment `G=sum(h u^T)` define the least-squares normal
equation `C J = G`. SolveDelta does not materialize or invert those moment
matrices. It advances the full matrix solution `C` directly with an online
normalized-LMS residual step, then turns each new solve residual into a
bounded rank-one frame that transforms the current memory edit and query:

```text
prefix moments -> online residual solve -> relative frame
       -> channel-decayed Delta edit -> query read
```

The result is causal, recurrent, and streamable. Its fixed-size FP32
continuation is only `(C,S)`: the current geometry solution and the Delta
memory. Second-order prefix information is retained in solution coordinates;
no explicit covariance, inverse, attention matrix, or token history is
carried. At finite `gamma=0`, its memory path is exactly GDN2; enabling the
solver adds history-conditioned coordinates without changing the ordinary
Delta edit.

This repository includes the FP64 mathematical oracle, a BF16/FP32 CUDA path
specialized from mature Flash Linear Attention primitives, and a complete
Hugging Face causal LM with recurrent cache and CUDA Graph training.

Keywords: **recurrent memory**, **Delta rule**, **linear attention**,
**online least squares**, **second-order statistics**, **causal language
modeling**, **PyTorch**, **Triton**, and **CUDA**.

## Highlights

- **Online second-order solve.** Prefix moments define `C J = G`, while a
  normalized-LMS/Oja residual step updates the full `r x r` solution directly.
  The operator captures the geometry of covariance and cross-moment fitting
  without storing or inverting a covariance, running CG, requiring an SPD
  prior, or globally forgetting unobserved directions.
- **Exact local primal/dual frame.** The factor
  `F=I+u phi^T` uses a statically bounded radial covector, maps the edit key
  with `F`, and maps erase/query covectors with `F^-T`. The resulting Delta
  transition is an exact, analytically well-conditioned similarity transform,
  not an approximate inverse action.
- **GDN2 is structurally contained.** At finite `gamma=0`, `F=I` and the
  independent channel-wise erase/write gates give the GDN2 edit/read exactly.
  The model retains GDN2's low-rank coordinate decay and standard
  sigmoid-gated RMSNorm readout.
- **Full matrix solution, rank-one work.** The solver has `r^2` recurrent
  capacity while each token contributes one rank-one normal-equation residual
  that maps to mature pair/WY/state kernels.
- **Mixed-precision native path.** Tensor Core contractions use BF16 operands
  and FP32 accumulation; predictor, memory, gates, denominators, reductions,
  and backward partials stay FP32.
- **Standard model surface.** `AutoModelForCausalLM`, fused loss, FLA
  RMSNorm/GatedMLP, hybrid attention, conv4, checkpoint save/load, recurrent
  generation, and fixed-shape CUDA Graph training are connected.
- **One mathematical owner.** `causallsso/reference.py` is the sole
  executable FP64 operator definition.

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
Use ordinary fused cross entropy; FLA fused linear cross entropy currently
performs a host-synchronizing label count and is rejected before capture.
Construct and replay a helper on the same CUDA stream.

When each optimizer update accumulates at least two fixed-shape
microbatches, bind the optimizer to an optional BF16 Linear shadow:

```python
accumulation_steps = 4
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
graph_step = SolveDeltaGraphedTrainingStep(
    model,
    sample_input_ids=input_ids,
    sample_labels=input_ids,
    bf16_shadow_optimizer=optimizer,
)

optimizer.zero_grad(set_to_none=True)
for batch_ids, batch_labels in microbatches:
    loss = graph_step(batch_ids, batch_labels) / accumulation_steps
    loss.backward()
optimizer.step()  # refreshes every BF16 shadow exactly once
```

FP32 parameters remain the optimizer and checkpoint owners. Nonpersistent
BF16 buffers remove repeated autocast conversions and the optimizer post-step
hook refreshes them on the replay stream. This is a throughput-for-memory
option, not the default: at the documented one-layer profile it adds about
`43.6 MiB` after capture, is slightly slower without accumulation, and becomes
useful from two or more microbatches. Direct parameter mutation is detected;
call `graph_step.refresh_bf16_shadow_weights()` after loading or otherwise
changing weights outside the bound optimizer.

For DDP, initialize NCCL and select one CUDA device per process, then ask the
helper to install DDP after local graph capture:

```python
import os
import torch.distributed as dist

dist.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
model = model.cuda().train()

graph_step = SolveDeltaGraphedTrainingStep(
    model,
    sample_input_ids=input_ids,
    sample_labels=input_ids,
    distributed=True,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

Do not wrap `model` in DDP before constructing the helper. The captured graph
owns fixed-shape local compute; DDP parameter hooks and NCCL reductions execute
after graph replay. This avoids coupling CUDA Graph capture to reducer buckets
or collective capture support while preserving ordinary DDP gradient
accumulation semantics. For accumulation without a reduction on every
microbatch, wrap both replay and backward in `with graph_step.no_sync():`, just
as with ordinary DDP.

## How It Works

For a prefix of directional observations, define

```text
J_t = sum_{s<=t} u_s u_s^T
G_t = sum_{s<=t} h_s u_s^T.
```

The stationary equation of `1/2 sum ||h_s-C u_s||^2` is `C J_t = G_t`.
Instead of forming `J_t`, `G_t`, or `J_t^-1`, SolveDelta performs one online
residual solve step for each normalized direction `u_t`, target `h_t`, and
learned relaxation `gamma_t in (0,1)`:

```text
r_t = h_t - C_{t-1} u_t
delta_t = gamma_t r_t
C_t = C_{t-1} + delta_t u_t^T.
```

This is the negative instantaneous least-squares gradient in solution
coordinates. It is not a claim that every intermediate `C_t` equals the batch
closed-form solution; ordering and learned relaxation are part of the
recurrent model.

The relative frame is

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
d_t   = F_t k_t
e_t   = F_t^-T b_t
chi_t = F_t^-T q_t.
```

This gives

```text
e_t^T d_t = b_t^T k_t
I-d_t e_t^T = F_t (I-k_t b_t^T) F_t^-1.
```

The channel-decayed memory applies GDN2's independent channel-wise write and
erase gates in that relative frame and reads with `chi_t`. The core output then
passes through a low-rank sigmoid gate and standard RMSNorm. The geometry bias
initializes to `logit(0.9)`, treating `gamma` as normalized-LMS relaxation. See
`docs/INNOVATION_PROGRAM.md` for the derivation and
`docs/FROM_SCRATCH_REBUILD.md` for exact execution ownership.

## Native Implementation

The CUDA path specializes concrete mature primitives instead of retaining
their upstream public ABIs:

- stride-aware FLA L2Norm for the predictor source;
- FLA gated-Oja pair, triangular WY, matrix-state forward, and an ungated
  transpose specialization for the residual predictor;
- a fused source owner that normalizes strided query/key views and writes
  direct/dual/query exterior panels while closing erase/write transposes;
- an exact unbounded Triton specialization of FLA's direct-`e` pair
  forward/transpose;
- FLA generalized-DPLR WY, state/output, and output-owned reverse;
- FLA KDA coordinate-decay and output-gate projections plus FLA's
  sigmoid-gated RMSNorm owner and exact transpose.

The producer writes the consumer's panel layout directly, so no token-major
`d/e/chi` ABI or panelization copy remains. State and output stay selectively
split to preserve chunk/rank/value CTA parallelism.

## Model and Cache

`SolveDeltaForCausalLM` uses an FLA-style prenorm block, GatedMLP, final
RMSNorm, LM head, and optional fused cross entropy. It registers with
`AutoConfig`, `AutoModel`, and `AutoModelForCausalLM`.

`past_key_values` stores each layer's FP32 `(C,S)` and conv4 state. Packed
training may provide `cu_seqlens`; explicit resets may provide
`solvedelta_reset_mask`. Dense batches without padding should omit an
all-ones `attention_mask` so they stay on the optimized native path.

## Performance Snapshot

Development measurements use an RTX 5070 Ti at
`B=1,T=1024,H=8,r=V=128`, BF16 autocast, FP32 master parameters/state, and
CUDA Graph replay. Both paths omit final continuation output in this table.

| Scope | Path | Forward median/p95 | F+B median/p95 | Graph allocated |
| --- | --- | ---: | ---: | ---: |
| Core operator | SolveDelta | 0.173 / 0.178 ms | 0.610 / 0.785 ms | 58.1 MiB |
| Core operator | FLA GDN2 | 0.107 / 0.112 ms | 0.467 / 0.643 ms | 46.0 MiB |
| Projected mixer | SolveDelta | 0.428 / 0.614 ms | 1.437 / 1.632 ms | 245.8 MiB |
| Projected mixer | FLA GDN2 | 0.364 / 0.534 ms | 1.276 / 1.468 ms | 233.2 MiB |

At the projected-mixer boundary, SolveDelta currently pays about `18%` forward
and `12%` F+B latency over GDN2, plus `12.6 MiB` active Graph allocation, for
the residual predictor and relative-frame actions. The mixer comparison adds
projection, conv4, gates, normalization, and output projection, but excludes
the MLP, LM head, optimizer, and distributed communication.

The core rows expose different mathematical contracts: SolveDelta includes its
predictor and initial `(C,S)` cotangents, while GDN2 has no predictor. The
projected-mixer rows are the primary hidden-to-hidden comparison.

This is a development snapshot, not a hardware-independent claim. Scope,
numerical quality, the exploratory 200M-token comparison, and measurement
caveats are recorded in [Evaluation and Performance](docs/RESULTS.md).

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

- Dense native execution currently requires sequence lengths divisible by 16.
- Masks and resets use the same reference operator rather than a packed native
  kernel.
- Recurrent cache semantics are connected, but a dedicated optimized decode
  kernel is not yet claimed.
- Residual-Frame is less instantaneously expressive than the archived
  bounded-LDU chart. Its quality/latency trade must be established by matched
  training.

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
