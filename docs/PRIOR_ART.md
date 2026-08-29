# Prior Art and Production Provenance

This ledger records the upstream work that materially shapes the selected
Residual-Frame SolveDelta production path. It is intentionally limited to
current donors, concrete mappings, licenses, and final design decisions.
Historical experiments belong to Git history.

## Ownership boundary

SolveDelta owns:

- the residual predictor recurrence;
- the mapping from its residual write to the relative frame;
- the exact local primal/dual similarity contract;
- the composition with channel decay and the ordinary Delta edit/read;
- the public `(C,S)` state and mixed-precision contract;
- model integration, tests, and acceptance gates.

Upstream projects own much of the mature GPU execution shape: normalization,
pair formation, triangular WY action, chunk state/output ownership, strict
transpose, causal conv, and model shell.

## Flash Linear Attention

Repository: <https://github.com/fla-org/flash-linear-attention>

License: MIT. The license text is preserved in `LICENSES/MIT.txt`; adapted
production code is also listed in `THIRD_PARTY_NOTICES.md`.

### Gated Oja predictor

Principal source areas:

- `fla/ops/gated_oja_rule/chunk_kkt.py`;
- `fla/ops/gated_oja_rule/wy_fast.py`;
- `fla/ops/gated_oja_rule/chunk_h.py`;
- the matching backward owners in those files.

The Residual-Frame predictor is exactly the ungated vector-decay subset of
FLA's gated Oja recurrence under

```text
FLA key/target <- h
FLA value      <- normalized u
FLA beta       <- gamma
FLA state      <- predictor C.
```

The production specialization retains FLA's:

- source Gram/pair construction;
- chunk-local unit-lower triangular solve;
- recomputed WY source/update panels;
- FP32 chunk-boundary matrix state;
- resident state forward;
- reverse chunk-state traversal;
- WY and pair transposes.

It deletes the unrelated Oja query/output branch and does not expose FLA's
public ABI. Current code still supplies a private zero vector-decay panel to
two generic donor reverse helpers; deleting that panel is an implementation
optimization, not a mathematical change.

### Generalized DPLR memory exterior

Principal source areas:

- `fla/ops/generalized_delta_rule/dplr/chunk_h_fwd.py`;
- `fla/ops/generalized_delta_rule/dplr/chunk_h_bwd.py`;
- `fla/ops/generalized_delta_rule/dplr/chunk_o_fwd.py`;
- `fla/ops/generalized_delta_rule/dplr/chunk_o_bwd.py`;
- `fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py`;
- `fla/ops/generalized_delta_rule/dplr/wy_fast_bwd.py`.

The memory recurrence maps to generalized DPLR as

```text
q=chi, k=d, a=e, b=-d, v=z, scale=1.
```

Production reuses FLA's fast-WY, FP32 state-boundary, chunk-parallel output,
output-owned reverse, state reverse, and triangular transpose. SolveDelta
adapts the source interface to direct `e` and emits panel-native operands;
it does not retain an upstream `e=b*k` public boundary.

### L2 normalization and decay scan

Principal source areas:

- `fla/modules/l2norm.py`;
- `fla/ops/rwkv6/chunk.py`;
- `fla/ops/utils/`.

The native L2Norm keeps FLA's row ownership and strict transpose while
accepting fused-projection views with arbitrary outer strides. The exterior
uses FLA's chunk cumsum convention for channel decay and its transpose through
the direct-e pair owner. Norms, cumsums, and sensitive scalar reductions remain
FP32.

### Model shell

Principal source areas:

- `fla/models/gated_deltanet/`;
- `fla/models/mesa_net/`;
- `fla/models/hybrid.py`;
- `fla/layers/gated_deltanet.py`;
- `fla/layers/gdn2.py`.

The Hugging Face causal-LM shell follows FLA's prenorm block, recurrent cache,
hybrid-attention routing, RMSNorm, GatedMLP, fused-loss, and generation
ownership. SolveDelta replaces the mixer recurrence and cache payload with
FP32 `(C,S)`.

### Why MESA is no longer production

FLA MESA previously supplied paired covariance/cross-moment scans and
matrix-free CG for the RLS route. Residual-Frame removes `m/J/D`, covariance
inversion, CG, and their transposes from model semantics. MESA remains useful
comparative prior art, but no MESA geometry kernel is part of the selected
operator.

## TileLang

Repository: <https://github.com/tile-ai/tilelang>

License: MIT, with separately licensed bundled components as documented by the
upstream distribution.

The C16 direct-`e` pair forward and transpose use TileLang and specialize
FLA's generalized-DPLR/KDA scheduling style:

- one chunk/head owner;
- low-precision source tiles;
- Tensor Core pair GEMMs with FP32 accumulation;
- causal triangular masking in the pair epilogue;
- final source ownership in reverse.

The pair owner consumes the source owner's rectangular panels directly.
Changing the upstream ABI and tensor names is deliberate: retaining a generic
interface would reintroduce token-major copies and duplicated source work.

## GDN2 and KDA

FLA GDN2 and KDA provide the principal comparison and reverse-ownership
patterns.

Retained design influence:

- one owner for each final source cotangent;
- chunk/rank/value CTA parallelism rather than one sequence/head CTA;
- direct-`e` specialization when erase is already a covector;
- selective splitting of state and output reverse;
- gate and source epilogues fused only when their lifetimes coincide.

SolveDelta reduces exactly to the ordinary GDN2 edit/read at `gamma=0`, but
the predictor remains a SolveDelta-specific model component.

## Mamba

Repository: <https://github.com/state-spaces/mamba>

License: Apache-2.0.

No Mamba source file is vendored. Mamba informed:

- direct addressing of fused `[B,T,H,r]` projection views;
- FP32 continuation state with low-precision contraction operands;
- separate state and output ownership;
- trainer-owned CUDA Graph boundaries.

## causal-conv1d

Repository: <https://github.com/Dao-AILab/causal-conv1d>

License: BSD-3-Clause.

The frontend uses the runtime package for independent depthwise causal conv4,
SiLU, recurrent final conv state, and its complete VJP. SolveDelta does not
maintain a handwritten convolution fallback.

## PyTorch and Hugging Face Transformers

PyTorch supplies autograd composition, CUDA Graph capture, allocator
accounting, and module/runtime primitives. Hugging Face Transformers supplies
`PretrainedConfig`, auto-class registration, checkpointing, generation, and
causal-LM result types. Their runtime licenses and dependency metadata remain
owned by those packages.

## Selected implementation decisions

| Decision | Selected form | Reason |
| --- | --- | --- |
| Geometry state | FP32 residual predictor `C` | Direct solution-coordinate history; no covariance inversion |
| Frame | Token-local `F=I+u delta^T` | Exact rank-one inverse transpose and local similarity |
| Predictor schedule | FLA gated-Oja C32 specialization | Exact recurrence and mature forward/transpose |
| Memory schedule | TileLang direct-e pair + FLA DPLR C16 | Mature pair/WY/state/output ownership |
| Source ABI | Panel-native direct and paired sources | Deletes token-major `d/e/chi` copy boundary |
| Fusion | Selective | Preserves useful CTA parallelism and short lifetimes |
| Public state | `(C,S)` only | No redundant inverse or diagnostic state |
| Precision | BF16 multiplicands, FP32 accumulation/state/scalars | Matches Tensor Core and recurrence requirements |
| Dense masks/resets | Reference recurrence | Same semantics until a mature packed native owner is connected |

## Rejected or removed alternatives

| Alternative | Current reason for rejection |
| --- | --- |
| Bounded-LDU exact chart | Much stronger instantaneous chart, but its exact action/transpose did not reach usable training latency |
| RLS `m/J/D` + MESA CG | Added covariance conditioning, larger state, and reverse cost that the residual predictor does not require |
| Dense accumulated `P^-T` continuation | Duplicates a deterministic `r^2` inverse state and complicates reverse/checkpoint ownership |
| Runtime RLS/Residual selector | Creates two public contracts and prevents one authoritative oracle |
| Flat expanded token axis | Exposes private edits as a synthetic sequence and inflates checkpoints |
| Sequence/head resident mega-kernel | Measured loss of chunk/rank/value CTA parallelism |
| Public canonicalization copies | Fused projection views can be consumed by stride-aware owners |
| BF16-to-FP16 pseudo-promotion | Cannot recover discarded mantissa bits |
| Denominator clamp/fallback | Silently changes the model instead of testing frame stability |

## Provenance maintenance

When a future change copies or structurally adapts upstream code:

1. record the exact source file/revision and mathematical mapping here;
2. preserve copyright/license headers where source is copied;
3. update `THIRD_PARTY_NOTICES.md`;
4. connect both forward and strict transpose;
5. compare only after the same oracle and composed-VJP gates pass.

Performance resemblance alone is not provenance. Conversely, changing tensor
names, shapes, or a donor ABI does not erase material scheduling influence.
