# SolveDelta

**History-conditioned Delta memory for causal language models.**

SolveDelta learns a full matrix predictor from the observed prefix. Each new
prediction residual generates an exact rank-one coordinate frame for the
current Delta edit and query:

```text
prefix -> residual predictor -> relative frame
       -> channel-decayed Delta edit -> query read
```

The operator remains causal and recurrent. Its fixed-size FP32 continuation is
only the predictor and memory state `(C,S)`; it carries no covariance,
inverse, or token history.

The repository includes an FP64 oracle, a BF16/FP32 CUDA training path built
from mature FLA/TileLang blocks, and a complete Hugging Face causal LM with
recurrent cache and CUDA Graph training support.

## Highlights

- **Residual geometry learning.** A normalized-LMS/Oja update writes each
  observed `u -> h` residual directly into a full `r x r` predictor.
  History accumulates in solution coordinates without covariance inversion,
  CG, an SPD prior, or global forgetting of unobserved directions.
- **Exact local primal/dual frame.** The factor
  `F=I+u delta^T` maps the edit key with `F` and erase/query covectors with
  `F^-T`. The resulting Delta transition is an exact similarity transform,
  not an approximate inverse action.
- **GDN2 is structurally contained.** At finite `gamma=0`, `F=I` and the
  memory path reduces exactly to the ordinary gated Delta edit/read.
- **Full matrix history, rank-one work.** The predictor has `r^2` recurrent
  capacity while each token contributes one rank-one residual write that maps
  to mature pair/WY/state kernels.
- **Mixed-precision native path.** Tensor Core contractions use BF16 operands
  and FP32 accumulation; predictor, memory, gates, denominators, reductions,
  and backward partials stay FP32.
- **Standard model surface.** `AutoModelForCausalLM`, fused loss, FLA
  RMSNorm/GatedMLP, hybrid attention, conv4, checkpoint save/load, recurrent
  generation, and fixed-shape CUDA Graph training are connected.
- **One mathematical owner.** `causallsso/reference.py` is the sole
  executable FP64 operator definition.

## Quick Start

Install the native dependencies and tests:

```bash
python -m pip install -e ".[native,test]"
```

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

## How It Works

For normalized geometry direction `u_t`, target `h_t`, and token-local
`gamma_t in (0,1)`:

```text
r_t = h_t - C_{t-1} u_t
delta_t = gamma_t r_t
C_t = C_{t-1} + delta_t u_t^T.
```

The relative frame is

```text
F_t = I + u_t delta_t^T
F_t^-T x = x - delta_t (u_t^T x) / (1 + delta_t^T u_t).
```

With normalized edit key `k_t`, erase covector `b_t), and query `q_t`:

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

The channel-decayed memory then performs one ordinary Delta write in that
relative frame and reads with `chi_t`. See
`docs/INNOVATION_PROGRAM.md` for the derivation and
`docs/FROM_SCRATCH_REBUILD.md` for exact execution ownership.

## Native Implementation

The CUDA path specializes concrete mature primitives instead of retaining
their upstream public ABIs:

- stride-aware FLA L2Norm;
- FLA gated-Oja pair, triangular WY, matrix-state forward, and transpose for
  the residual predictor;
- a fused source owner that writes direct/dual/query exterior panels;
- TileLang direct-`e` pair forward/transpose;
- FLA generalized-DPLR WY, state/output, and output-owned reverse.

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

## Measured Core Performance

Development measurements use an RTX 5070 Ti at
`B=1,T=1024,H=8,r=V=128`, BF16 public operands and FP32 state:

| Execution | Forward median/p95 | F+B median/p95 | Peak allocated |
| --- | ---: | ---: | ---: |
| Eager core | 0.498 / 0.644 ms | 1.861 / 2.056 ms | 105.1 MiB incremental |
| CUDA Graph core | 0.187 / 0.193 ms | 0.717 / 0.918 ms | capture-owned |
| CUDA Graph projected mixer | 0.406 / 0.518 ms | 1.428 / 1.639 ms | 120.0 MiB capture increment |

The core F+B number includes output plus both final-state cotangents. The
projected mixer adds projection, conv4, source gates, and output projection but
excludes MLP, LM head, loss, and optimizer. Benchmarks describe this
implementation; they do not define model semantics.

## Correctness

The acceptance suite covers:

- FP64 residual prediction and exact relative-frame identities;
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

- The scalar `1+delta_t^T u_t` has no structural positive lower bound.
  Training should monitor its distribution; production does not clamp or
  silently fall back.
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

## Repository Map

- `causallsso/reference.py`: sole FP64 mathematical oracle;
- `causallsso/model.py`: projected SolveDelta mixer;
- `causallsso/modeling_solvedelta.py`: FLA/Hugging Face causal LM;
- `causallsso/ops/residual_frame/`: selected native forward and transpose;
- `docs/FROM_SCRATCH_REBUILD.md`: native implementation blueprint;
- `docs/INNOVATION_PROGRAM.md`: derivation and interpretation;
- `docs/PRIOR_ART.md`: upstream provenance and production decisions;
- `THIRD_PARTY_NOTICES.md`: adapted-source attribution.
