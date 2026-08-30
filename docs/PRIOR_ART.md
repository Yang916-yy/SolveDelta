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

The selected runtime and source-compatibility baseline is FLA `0.6.0`.
SolveDelta imports private kernel modules, so this version is pinned until a
new release passes the full oracle, VJP, model, and CUDA Graph suite. Runtime
installation details are maintained in `docs/ENVIRONMENT.md`.

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
public ABI. The strict transpose copies FLA's chunk/head output ownership and
removes the constant-zero vector-decay loads, exponentials, cotangent, and HBM
panel. Its WY target loads accept the model's strided projection view directly.
The selected `r=128` reverse specializes that owner to 32-row tiles after an
identical-gradient A/B showed lower register/shared-memory pressure and higher
CTA parallelism than the donor's broader 64-row tile.

### Generalized DPLR memory exterior

Principal source areas:

- `fla/ops/generalized_delta_rule/dplr/chunk_A_fwd.py`;
- `fla/ops/generalized_delta_rule/dplr/chunk_A_bwd.py`;
- `fla/ops/generalized_delta_rule/dplr/chunk_h_fwd.py`;
- `fla/ops/generalized_delta_rule/dplr/chunk_h_bwd.py`;
- `fla/ops/generalized_delta_rule/dplr/chunk_o_fwd.py`;
- `fla/ops/generalized_delta_rule/dplr/chunk_o_bwd.py`;
- `fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py`;
- `fla/ops/generalized_delta_rule/dplr/wy_fast_bwd.py`.

The memory recurrence maps to generalized DPLR as

```text
q=chi, k=d, a=-exp(log_alpha)e, b=d, v=z, scale=1.
```

FLA's low-rank action consumes the pre-decay state, whereas SolveDelta erases
from `S_decay`. The current decay factor in `a` is therefore part of the exact
mapping. The direct-`e` specialization folds it into the inclusive decay
prefix instead of materializing a scaled source panel.

Production adapts FLA's exact unbounded scalar pair forward/transpose to load
the source-native rectangular panels. It forms only the two distinct pair
matrices, merges the duplicated source cotangents in-register, and closes
decay with the same vector reverse-cumsum ownership. FLA's fast-WY, FP32
state-boundary, chunk-parallel output, output-owned reverse, state reverse,
and triangular transpose remain connected. The generic token-major ABI and
its four pair matrices are not retained.

### L2 normalization, decay, and output gate

Principal source areas:

- `fla/modules/l2norm.py`;
- `fla/layers/kda.py`;
- `fla/ops/kda/gate.py`;
- `fla/modules/fused_norm_gate.py`;
- `fla/ops/rwkv6/chunk.py`;
- `fla/ops/utils/`.

The standalone native L2Norm keeps FLA's row ownership and strict transpose for
`u`. The relative source owner applies the same FP32 reduction and transpose to
strided q/key views before generating the exterior panels, eliminating two
normalized HBM panels. Its bounded frame actions have a static range proof, so
their private `d/e/chi` panels are written directly from the FP32 producer to
FP16; operands multiplied by unbounded decay factors remain BF16. The model
uses KDA's low-rank coordinate-decay
parameterization and upstream fused gate/transpose. The output readout adapts
FLA's sigmoid-gated RMSNorm row owner and strict transpose, adding a per-head
bounded radial scale while its FP32 `rstd` is resident. The exterior retains
FLA's chunk cumsum convention and its transpose
through the direct-e pair owner. Norms, cumsums, and sensitive scalar
reductions remain FP32. The source owner also specializes scaled-L2Norm
arithmetic for the bounded frame covector without writing a separate radial
panel. Since the sole `u` producer is L2Norm, production uses `||u||=1`
directly and leaves the omitted radial cotangent to the L2Norm nullspace. At
the selected width, the compact row owner uses one warp in both directions;
wider warp schedules added synchronization without useful parallel work.

The final readout further adopts FLA's norm-linear checkpoint ownership. The
normalized gated output is regenerated by the existing strict transpose for
the ordinary projection-weight GEMM instead of being saved across backward.
This is a lifetime specialization, not a Triton/CUTLASS mega-fusion. The
direct-e pair and source reverse owners retain FP32 local accumulation but use
a BF16 final-shaped handoff after the reduction is complete.

The radial modulation is mathematically related to feature-wise modulation
and channel recalibration, not copied implementation code:

- FiLM: <https://arxiv.org/abs/1709.07871>;
- Squeeze-and-Excitation: <https://arxiv.org/abs/1709.01507>.

Unlike a free FiLM scale, SolveDelta maps the per-head strength and observed
RMS through bounded coordinates, so the multiplier remains in `(1/2,3/2)`.

## Rank-one invertibility precedents

Rezende and Mohamed's planar normalizing flow identifies the determinant
condition for identity-plus-rank-one maps and reparameterizes the parallel
component to preserve invertibility. Behrmann et al.'s invertible residual
networks use a strict residual norm bound to obtain global invertibility.
SolveDelta uses the stronger norm-bounded option because a merely positive
determinant may still approach zero or permit a large shear:

- <https://proceedings.mlr.press/v37/rezende15.html>;
- <https://proceedings.mlr.press/v97/behrmann19a.html>.

No source code from either project is copied. Their mathematical conditions
motivate the static `||u|| ||phi|| < 5/8` contract; the concrete GPU arithmetic
continues to use the FLA-derived source-owner and L2Norm schedules above.

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

## GDN2 and KDA

FLA GDN2 and KDA provide the principal comparison and reverse-ownership
patterns.

Retained design influence:

- one owner for each final source cotangent;
- chunk/rank/value CTA parallelism rather than one sequence/head CTA;
- direct-`e` specialization when erase is already a covector;
- selective splitting of state and output reverse;
- gate and source epilogues fused only when their lifetimes coincide;
- independent GDN2 channel-wise erase/write gates;
- KDA low-rank coordinate decay and output-gate projections;
- KDA sigmoid-gate projection and FLA gated-RMSNorm row ownership.

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
maintain a handwritten convolution fallback. The selected compatibility
baseline is causal-conv1d `1.7.x`.

## PyTorch and Hugging Face Transformers

PyTorch supplies autograd composition, CUDA Graph capture, allocator
accounting, and module/runtime primitives. Hugging Face Transformers supplies
`PretrainedConfig`, auto-class registration, checkpointing, generation, and
causal-LM result types. Their runtime licenses and dependency metadata remain
owned by those packages.

The optional accumulation path follows the established mixed-precision
ownership used by NVIDIA Megatron-LM and Apex: FP32 optimizer masters, resident
low-precision model weights, and one refresh after an optimizer update. No
Megatron or Apex source is copied. SolveDelta implements the narrower Linear
case with PyTorch `F.linear`, `mm`, nonpersistent buffers, and an optimizer
post-step hook; it additionally rejects stale shadows before graph replay.

- Megatron-LM: <https://github.com/NVIDIA/Megatron-LM> (Apache-2.0);
- Apex: <https://github.com/NVIDIA/apex> (BSD-3-Clause).

The fused input projection uses a private 64-row physical alignment selected
by complete F+B A/B. Its logical prefix and mathematical parameters are
unchanged; unused padding-row gradients are exactly zero. PyTorch
`grouped_mm` was also tested for the decay/output-gate expansions, but its
required input/weight packing and lack of differentiable grouped bias made it
slower than two independent `F.linear` calls at the selected shape.

The fixed-shape training helper uses PyTorch's
`set_override_stale_capture_stream` behavior introduced by PR 180090:
<https://github.com/pytorch/pytorch/pull/180090>. It retains parameter gradient
edges on the caller's replay stream, captures local loss forward/backward, and
installs DDP afterward. DDP reducer hooks and NCCL collectives remain outside
the graph. This is a runtime composition decision; no PyTorch implementation
source is copied.

PyTorch PR 189914 removes eager TorchScript compilation from the Inductor
import path: <https://github.com/pytorch/pytorch/pull/189914>. Stable PyTorch
2.13.0 predates that change, so package initialization scopes only its exact
MKLDNN `script_method` deprecation while importing FLA. No warning category is
disabled globally and no PyTorch source is vendored.

## Selected implementation decisions

| Decision | Selected form | Reason |
| --- | --- | --- |
| Geometry state | FP32 residual predictor `C` | Direct solution-coordinate history; no covariance inversion |
| Frame | Token-local `F=I+u phi^T`, `||u|| ||phi||<5/8` | Exact similarity with `den>3/8` and bounded conditioning |
| Predictor schedule | FLA gated-Oja C32 specialization | Exact recurrence and mature forward/transpose |
| Memory schedule | Exact unbounded Triton direct-e pair + FLA DPLR C16 | Exact decay semantics with mature pair/WY/state/output ownership |
| Delta gates | Independent channel-wise `sigmoid(erase_raw/write_raw)` | Preserves the full GDN2 update surface and separate source cotangents |
| Decay/readout | KDA low-rank coordinate gate + bounded radial gated RMSNorm | Preserves coordinate decay/output gating and a controlled geometry-magnitude signal |
| Source ABI | Panel-native direct and paired sources | Deletes token-major `d/e/chi` copy boundary |
| Fusion | Selective | Preserves useful CTA parallelism and short lifetimes |
| Public state | `(C,S)` only | No redundant inverse or diagnostic state |
| Precision | bounded FP16 source panels, BF16 decay-scaled operands/final-shaped cotangent handoff, FP32 accumulation/state/scalars | Uses extra mantissa where a static range proof or composed-VJP evidence requires it |
| Fused projection | 64-row physical alignment, logical prefix consumers | Improves the dominant projection transpose without a canonicalization copy |
| Accumulation weights | Optimizer-bound BF16 Linear shadows | Amortizes casts across microbatches without lowering optimizer precision |
| Dense masks/resets | Reference recurrence | Same semantics until a mature packed native owner is connected |
| Distributed CUDA Graph | Local graph first, DDP reducer outside capture | Stable `AccumulateGrad` ownership without coupling graphs to NCCL buckets |

## Rejected or removed alternatives

| Alternative | Current reason for rejection |
| --- | --- |
| Bounded-LDU exact chart | Much stronger instantaneous chart, but its exact action/transpose did not reach usable training latency |
| RLS `m/J/D` + MESA CG | Added covariance conditioning, larger state, and reverse cost that the residual predictor does not require |
| Dense accumulated `P^-T` continuation | Duplicates a deterministic `r^2` inverse state and complicates reverse/checkpoint ownership |
| Runtime RLS/Residual selector | Creates two public contracts and prevents one authoritative oracle |
| Flat expanded token axis | Exposes private edits as a synthetic sequence and inflates checkpoints |
| Sequence/head resident mega-kernel | Measured loss of chunk/rank/value CTA parallelism |
| Centered Tensor-Core direct-e pair | Requires a static decay lower bound; long-run unbounded decay overflowed despite finite model inputs |
| TileLang runtime pair path | Removed with the centered owner; the selected exact-unbounded specialization uses Triton already provided by FLA |
| FLA fused DPLR `chunk_ho` at C16 | Saved state/output HBM but regressed the target shape because too few CTAs serially advanced all chunks |
| Public canonicalization copies | Fused projection views can be consumed by stride-aware owners |
| Grouped decay/output-gate GEMM | Packing and grouped-bias epilogues exceeded the two-GEMM launch saving |
| BF16-to-FP16 pseudo-promotion | Cannot recover discarded mantissa bits |
| Denominator clamp/fallback | Replaced by a smooth analytic radial parameterization with no data-dependent branch |

## Provenance maintenance

When a future change copies or structurally adapts upstream code:

1. record the exact source file/revision and mathematical mapping here;
2. preserve copyright/license headers where source is copied;
3. update `THIRD_PARTY_NOTICES.md`;
4. connect both forward and strict transpose;
5. compare only after the same oracle and composed-VJP gates pass.

Performance resemblance alone is not provenance. Conversely, changing tensor
names, shapes, or a donor ABI does not erase material scheduling influence.
