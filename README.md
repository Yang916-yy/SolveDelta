# SolveDelta

**Geometry-conditioned Delta memory for causal language models.**

SolveDelta lets the observed prefix change how a recurrent memory is read and
edited. Alongside the usual Delta-rule state, it maintains a compact decayed
covariance and cross-moment. Their RLS solve produces two rank-one transports
that adapt the memory before every ordinary Delta edit.

The result remains causal and recurrent, with a fixed-size continuation state:

```text
prefix geometry (m, J, D)
          |
          v
RLS history transport -> channel decay -> RLS innovation transport
          -> gated Delta edit -> query read
```

The repository includes an FP64 oracle, a BF16/FP32 CUDA training path, and a
complete FLA/Hugging Face causal LM with recurrent cache support.

## Highlights

- **Online second-order geometry.** A decayed covariance `J` and cross-moment
  `D` summarize the prefix as an online regression problem. Their RLS gain and
  innovation generate two rank-one transports that condition each Delta
  memory update.
- **Exact GDN2 containment.** At finite `gamma=0`, SolveDelta reduces exactly
  to the ordinary gated Delta edit/read while its geometry state continues to
  track the prefix. Nonzero `gamma` activates two prefix-conditioned RLS
  transports around the same edit.
- **Fixed recurrent state.** Training chunks and recurrent splits share the
  FP32 continuation state `(m,J,D,S)`.
- **Mixed-precision native path.** Tensor Core contractions use BF16 operands
  and FP32 accumulation; geometry, continuation state, CG, and sensitive
  reductions stay FP32.
- **Standard model surface.** `AutoModelForCausalLM`, fused loss, FLA
  RMSNorm/GatedMLP, hybrid attention, checkpoint save/load, and greedy
  generation are connected.
- **One source of mathematical truth.** `causallsso/reference.py` is the only
  executable FP64 operator definition.

## Quick Start

The recommended development install includes the native dependencies and
tests:

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

Use FP32 master parameters with BF16 autocast. The fused projection then emits
BF16 public operands and selects the optimized CUDA path without rounding
geometry parameters or continuation state.

For the standalone mixer:

```python
import torch

from causallsso import SolveDelta, SolveDeltaConfig

config = SolveDeltaConfig(
    hidden_size=256,
    num_heads=4,
    head_k_dim=64,
    head_v_dim=64,
)
mixer = SolveDelta(config).cuda()
hidden = torch.randn(2, 128, 256, device="cuda")

with torch.autocast("cuda", dtype=torch.bfloat16):
    output = mixer(hidden)
```

## How It Works

For normalized geometry direction `u_t`, SolveDelta updates

```text
m_t = lambda_t m_{t-1} + 1
J_t = lambda_t J_{t-1} + u_t u_t^T
D_t = lambda_t D_{t-1} + u_t h_t^T.
```

The fixed prior is `m_0=2`, `J_0=2I`, `D_0=0`. Define
the regression coordinate `C_t = solve(J_t, D_t)`. Its rank-one RLS innovation
produces a history transport and an innovation transport around the ordinary
channel-decayed Delta edit. Learned `gamma` interpolates both geometry
transports with identity, so `gamma=0` recovers the ordinary gated Delta
edit/read without stopping geometry observation.

At zero geometry-gate input, heads are initialized deterministically across
`lambda in [0.985, 0.995]`; a single head starts at `0.99`. This supplies
different initial effective horizons while keeping the SPD prior alive long
enough for early geometry acquisition. It is not a clamp: every head remains
token-dependent and learnable.

The derivation and model interpretation are in
`docs/INNOVATION_PROGRAM.md`. Exact execution, transpose ownership, and the
precision map are specified in `docs/FROM_SCRATCH_REBUILD.md`.

## Native Implementation

The dense CUDA training path specializes mature FLA/MESA normalization,
geometry, CG, generalized-Delta, WY, state/output, and transpose schedules.
The two geometry transports and ordinary Delta edit remain three private slots
attached to each token; they are not exposed as a public expanded sequence.
The exact kernel ownership, layouts, precision map, and reverse graph are in
`docs/FROM_SCRATCH_REBUILD.md`.

## Model and Cache

`SolveDeltaForCausalLM` uses an FLA-style prenorm block, GatedMLP, final
RMSNorm, LM head, and optional fused cross entropy. It registers with
`AutoConfig`, `AutoModel`, and `AutoModelForCausalLM`.

`past_key_values` stores each layer's FP32 `(m,J,D,S)` and conv4 state.
Packed training may provide `cu_seqlens`; explicit segment resets may provide
`solvedelta_reset_mask`. Dense batches without padding should omit an
all-ones `attention_mask` so they stay on the optimized dense path.

## Measured Performance

Development measurements use an RTX 5070 Ti at
`B=1,T=1024,H=8,r=V=128`, BF16 public operands, FP32 state, and CUDA Graph:

| Scope | Forward median | F+B median |
| --- | ---: | ---: |
| SolveDelta core | 0.348 ms | 1.082 ms |
| Complete projected mixer | 0.550 ms | 1.805 ms |
| Matched GDN2 core | 0.128 ms | 0.467 ms |

SolveDelta core p95 was approximately `0.364/1.133 ms`; the projected mixer
p95 was approximately `0.573/2.136 ms`. The GDN2 gap is explicit: SolveDelta
pays for an additional RLS geometry state and its strict reverse.

A separate one-layer CausalLM integration smoke at
`B=1,T=128,D=1024,H=8,r=V=128,vocab=4096` measured eager medians of
`3.837 ms` forward and `12.310 ms` F+B with a 2x GatedMLP and fused linear
cross entropy. This is a model-stack check, not a core comparison.

## Correctness

The acceptance suite covers:

- the FP64 token recurrence and fixed SPD prior;
- finite-parameter GDN2 reduction at `gamma=0`;
- masks, resets, packed sequences, and recurrent splits;
- a non-128 reference width;
- BF16 native output and FP32 continuation state;
- composed VJPs including initial/final state and symmetric `J`;
- public CausalLM loss/backward, cache, checkpoint, and native dispatch.

Run it with:

```bash
python -m pytest -q -s
```

## Current Limits

- The selected CG5 gain is a BF16-observable production approximation of the
  exact FP64 solve.
- Masks and resets currently use the reference operator path rather than the
  optimized dense native operator.
- Recurrent cache semantics are connected, but a dedicated optimized decode
  kernel is not yet claimed.
- The moving-state RLS formulation is less expressive than the archived exact
  bounded-LDU research path, but it is the only maintained operator because it
  reaches usable training latency.

The archived bounded-LDU implementation is recoverable at Git commit
`2237875`; it is not a runtime backend or compatibility target.

## Contributing

Read `AGENTS.md` before using an AI coding tool or changing operator code. It
defines the authority order, mathematical and precision contracts, provenance
requirements, test gates, and benchmark reporting rules. Material upstream
reuse is documented in `docs/PRIOR_ART.md` and `THIRD_PARTY_NOTICES.md`.

## Repository Map

- `causallsso/reference.py`: sole FP64 mathematical oracle;
- `causallsso/model.py`: projected SolveDelta mixer;
- `causallsso/modeling_solvedelta.py`: FLA/Hugging Face causal LM;
- `causallsso/ops/rls/`: selected native forward and transpose blocks;
- `docs/FROM_SCRATCH_REBUILD.md`: native implementation blueprint;
- `docs/INNOVATION_PROGRAM.md`: derivation and interpretation;
- `docs/PRIOR_ART.md`: current upstream provenance and production decisions;
- `THIRD_PARTY_NOTICES.md`: adapted-source attribution.
