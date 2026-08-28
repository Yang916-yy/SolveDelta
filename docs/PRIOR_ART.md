# Prior Art and Production Decisions

This document records the external work that materially informs the current
SolveDelta implementation. It is a provenance and decision ledger, not a
second operator contract. Executable mathematics belongs to
`causallsso/reference.py`; native ownership belongs to
`docs/FROM_SCRATCH_REBUILD.md`.

The ledger intentionally covers only the selected RLS production route.
Detailed profiler traces, failed prototypes, and the former bounded-LDU
implementation remain available through Git history. They are not repeated
here because historical private ABIs must not constrain current work.

## Production Baseline

The maintained operator combines four established ideas:

1. a decayed covariance and cross-moment state from online regression;
2. a fixed matrix-free conjugate-gradient action for the RLS gain;
3. three ordered rank-one generalized-Delta updates per token;
4. chunked WY/state/output execution and its composed transpose.

SolveDelta owns the algebraic composition, the effective-mass state, the
mapping from RLS quantities to the three update slots, and the public
`(m,J,D,S)` continuation contract. Upstream projects own many of the numerical
and scheduling primitives used to execute that composition.

The current production selection is:

- one ordinary Delta edit per token;
- two RLS transport slots plus that edit on a private fixed `E=3` axis;
- C32 paired MESA geometry chunks;
- a fixed five-step MESA-style CG action and implicit transpose;
- C16 exterior chunks and a C48 logical WY interaction block;
- BF16 public vectors, FP32 continuation state and accumulation;
- selective state/output/source ownership rather than whole-layer fusion.

These choices describe the selected implementation. They do not redefine the
FP64 recurrence or create a second mathematical operator.

## Mathematical Sources

### Delta-rule sequence models

The recurrent edit and its chunkwise parallelization follow the Delta-rule
family:

- Schlag, Irie, and Schmidhuber, *Linear Transformers Are Secretly Fast Weight
  Programmers*, [arXiv:2102.11174](https://arxiv.org/abs/2102.11174);
- Yang et al., *Parallelizing Linear Transformers with the Delta Rule over
  Sequence Length*, [arXiv:2406.06484](https://arxiv.org/abs/2406.06484);
- Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*,
  [arXiv:2412.06464](https://arxiv.org/abs/2412.06464);
- *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*,
  [arXiv:2605.22791](https://arxiv.org/abs/2605.22791), and its
  [official implementation](https://github.com/NVlabs/GatedDeltaNet-2).

The concrete consequences for SolveDelta are:

- write and erase are allowed to use asymmetric paired directions;
- the intra-chunk interaction is represented by a causal pair matrix;
- the compact response is evaluated by a WY-style triangular action;
- state and output traversal retain chunk and value-tile parallelism;
- backward is the transpose of those block actions, including initial and
  final state cotangents.

At finite `gamma=0`, both geometry transports become identity and the memory
path reduces to the ordinary gated Delta edit/read. This reduction is a model
identity, not an approximate compatibility mode.

### Online regression and MESA

The geometry state is a decayed covariance/cross-moment pair:

```text
J_t = lambda_t J_{t-1} + u_t u_t^T
D_t = lambda_t D_{t-1} + u_t h_t^T.
```

Its natural regression coordinate is `C_t = J_t^-1 D_t`. Sherman-Morrison
rank-one update identities provide the exact RLS innovation form used by the
reference recurrence. Classical background includes Sherman and Morrison's
rank-one inverse result
([DOI](https://doi.org/10.1214/aoms/1177729893)).

The closest modern sequence-model implementation precedent is
*MesaNet: Sequence Modeling by Locally Optimal Test-Time Training*,
[arXiv:2506.05233](https://arxiv.org/abs/2506.05233), together with FLA's
[MESA implementation](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/mesa_net).

FLA MESA maintains two dense states with the same computational shape:

```text
Hkk <- decay * Hkk + K^T K2
Hkv <- decay * Hkv + K^T V.
```

SolveDelta specializes that pair under

```text
Hkk <-> J,   K <-> u,   K2 <-> u
Hkv <-> D,   V <-> h.
```

This mapping allows the paired state scan, matrix-free action, fixed CG loop,
and strict transpose to retain MESA's resident tile ownership. SolveDelta adds
the mass state and maps the resulting gain/prediction into its own memory
transports.

### Generalized Delta and compact products

FLA's generalized-DPLR and GatedDeltaProduct operators establish mature
chunk-pair, triangular/WY, state, output, and transpose schedules for ordered
rank-one updates:

- [generalized DPLR](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/generalized_delta_rule/dplr);
- [GatedDeltaProduct](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_product);
- [GDN2](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gdn2);
- [KDA](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda).

SolveDelta has two RLS transport factors and one ordinary edit. They are a
static slot axis attached to each token, not a physically expanded sequence.
The direct-`e` specialization exploits the paired direction identity so that
the generic interaction family collapses to the two representatives actually
used by the operator. Reverse first combines cotangents of shared
representatives and then applies the corresponding transpose contractions.

This is a source-level specialization of mature owners. It is not a call
through FLA's generic high-level ABI, and it does not preserve unused generic
arguments or expanded `3T` buffers.

## Reviewed Upstream Revisions

The production path was developed against FLA 0.6.0 and refreshed against
FLA main during the final ownership audits. Material source checkpoints were:

- FLA MESA and common owners at
  [`bc3b101d`](https://github.com/fla-org/flash-linear-attention/commit/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89);
- FLA generalized-DPLR TileLang owners at
  [`5e02dd3a`](https://github.com/fla-org/flash-linear-attention/commit/5e02dd3a7651f5f2797eb8b12bbec401826031e1);
- the later KDA/output-ownership audit at FLA commit `bccaf2d3`;
- Mamba-3 output/state ownership at
  [`e9594ce1`](https://github.com/state-spaces/mamba/commit/e9594ce1c732d97440f0332fdc43170a2294dbfa);
- causal-conv1d final-state handling at
  [`cd81f041`](https://github.com/Dao-AILab/causal-conv1d/commit/cd81f0413cad2fc1e6f17e785ac39f59aae690cd).

Commit identifiers document what was reviewed; package requirements remain in
`pyproject.toml`. A future dependency refresh must recheck both forward and
transpose behavior rather than assuming a matching public symbol preserves a
private schedule.

## Licenses and Attribution

### Flash Linear Attention

Flash Linear Attention is distributed under the MIT License. Production code
adapts FLA kernels and model-shell components. The retained MIT text is in
`LICENSES/MIT.txt`, and adapted-file attribution is listed in
`THIRD_PARTY_NOTICES.md`.

Principal donor areas are:

```text
fla/ops/mesa_net/
fla/ops/generalized_delta_rule/dplr/
fla/ops/common/{chunk_h,gate}.py
fla/ops/{gdn2,kda}/
fla/modules/l2norm.py
fla/ops/utils/
fla/models/{gated_deltanet,mesa_net}/
fla/models/{hybrid,utils}.py
fla/layers/{gated_deltanet,gdn2}.py
```

### TileLang

TileLang is an MIT-licensed compiler/runtime dependency. The block-E3 pair and
WY programs specialize FLA's TileLang schedules rather than defining a second
generic matrix library. TileLang may include separately licensed bundled
components; its own distribution remains authoritative for those notices.

### causal-conv1d

`causal-conv1d` is a BSD 3-Clause runtime dependency. It owns the frontend's
depthwise conv4/SiLU execution and complete final-state VJP. SolveDelta does
not reproduce that convolution in a private fallback kernel.

### Mamba

Mamba is distributed under the Apache License 2.0. Mamba-3 and Mamba-2 were
reviewed for output-owned reverse, resident value-tile state, fused projection
layout, and decay initialization. No Mamba source file is vendored here; this
is design provenance rather than copied-code attribution.

## Production Reuse Map

### Model and frontend

`causallsso/modeling_solvedelta.py` and `causallsso/config.py` specialize the
FLA GatedDeltaNet/MESA Hugging Face shell:

- prenorm residual blocks;
- FLA `RMSNorm` and `GatedMLP`;
- fused cross-entropy ownership;
- hybrid-attention configuration;
- recurrent `Cache` and generation integration;
- `AutoConfig`, `AutoModel`, and `AutoModelForCausalLM` registration.

The adaptation replaces the sequence mixer and maps each layer cache to FP32
`(m,J,D,S)` plus three conv4 continuation states. FLA's trainer-facing model
surface is reused; SolveDelta retains one owner for its parameters and state.

`causallsso/model.py` follows the GDN2/Mamba positive log-rate plus
inverse-softplus initialization pattern for associative and geometry decay.
The geometry heads' selected initial values are SolveDelta-specific and are
specified by the native blueprint.

### Vector normalization and gates

`causallsso/ops/rls/strided_l2norm.py` specializes FLA L2Norm forward and
reverse arithmetic. It loads fused-projection views with their real outer
strides, writes the packed private panel needed by downstream matrix owners,
and transposes into the original view without a public canonicalization copy.

`causallsso/ops/gates.py`, `block_e3_sources.py`, and
`block_e3_pair_reverse.py` use FLA common/GDN gate arithmetic and ownership.
Raw erase/write logits are activated inside the source owner and consumed in
registers. The reverse closes the activation in the corresponding source
epilogue, so no activated-gate HBM ABI is retained.

### Geometry and gain

`causallsso/ops/rls/mesa_specialized.py` specializes the FLA MESA paired state
and CG kernels:

- resident `Hkk/Hkv` state loops become paired `J/D` loops;
- broad BF16 contractions retain FP32 accumulation;
- continuation matrices remain FP32;
- fixed generic constants and unused model arguments are removed;
- the transpose preserves the same state and source-tile ownership.

`causallsso/ops/rls/mesa_gain.py` owns the SolveDelta-specific composition:
gain and prediction outputs, CG5 saved quantities, implicit transpose, shared
`u/h/log_lambda` cotangents, and symmetric `J` epilogue.

`causallsso/ops/rls/mass.py` uses FLA-style chunk-local affine summaries and a
small continuation scan for `m`. Its reverse is the corresponding affine
transpose. It remains separate from the dense MESA state owner because a
resident fusion lengthened that owner's critical path.

### Block-E3 pair and WY

`causallsso/ops/rls/block_e3_pair.py` and
`block_e3_pair_reverse.py` specialize FLA generalized-DPLR TileLang
`chunk_A` ownership. Their native axes are token, fixed slot, coordinate tile,
and chunk. They construct and transpose only the pair representatives needed
by direct-`e` SolveDelta.

`causallsso/ops/rls/block_e3_wy.py` specializes FLA fast-WY triangular row
updates. Three C16 token groups form one C48 logical interaction block. The
program reuses the mature triangular action but changes the private layout and
consumer boundary to match native E3 panels.

### State, output, and reverse

`causallsso/ops/rls/block_e3_state.py`, `block_e3_reverse.py`, and
`block_e3_exterior.py` adapt FLA generalized-DPLR `chunk_h/chunk_o`, GDN2/KDA
output ownership, and Mamba-3 value-tile residency.

The selected split is deliberate:

- the state owner advances FP32 chunk boundaries;
- output owners retain chunk and value-tile parallelism;
- reverse walks those boundaries in transpose order;
- each source-gradient tile has one final owner;
- pair, WY, output, and source cotangents are consumed in phases to shorten
  their live ranges.

The code does not fuse every chunk into one sequence/head CTA. It accepts
private boundaries where full fusion would reduce occupancy or retain too
many panels simultaneously.

### Action-statistics epilogues

The exterior's broad action statistics use PyTorch CUDA `bmm`/`baddbmm` with
BF16 multiplicands and FP32 accumulation. Mature GEMM epilogues write the
declared private BF16 consumer panels directly. This replaces private FP32 HBM
outputs followed by separate cast/add launches without changing the public
precision contract.

## Selected Engineering Decisions

### One compact production path

The RLS route is the sole maintained operator. The fixed `E=3` slot axis is
internal to chunk owners. Public callers see one token sequence and the
continuation state `(m,J,D,S)`; they never see a synthetic `3T` sequence,
generic DPLR metadata, or archived chart descriptors.

### Fixed CG5 approximation

The FP64 oracle uses exact linear solves. Native execution uses a fixed
five-step MESA matrix-free CG action and the matching implicit transpose. This
is the selected BF16-observable numerical approximation. It has no runtime
iteration choice, convergence branch, fallback, or alternate public backend.

### Symmetric full-matrix J

`J` is mathematically symmetric positive definite. Production currently keeps
full FP32 storage because packing changed dense tile access without removing
enough arithmetic. Its full-tensor cotangent is symmetrized once at the public
boundary. Internal code must not create two independent ambient `J` routes.

### Stride-aware public views

Fused projections may expose arbitrary batch/token/head strides with unit
innermost vector stride. First mathematical owners load those views directly
and write packed private panels. A public `permute/contiguous` canonicalization
boundary is not part of the selected path.

### Selective fusion

Fusion is accepted only when complete F+B improves under identical numerical
gates. The current graph fuses gate activation with source ownership and keeps
pair/source transposes output-owned, but separates mass, MESA geometry,
action-statistics, and state/output phases where independent CTAs are more
valuable than eliminating a small boundary.

### Precision placement

The selected mixed-precision map follows FLA/MESA practice:

- BF16 public vectors and Tensor Core multiplicands;
- FP32 Tensor Core accumulation and backward partials;
- FP32 `m/J/D/S`, log decays, normalization and CG reductions, denominators,
  and sensitive scalar divisions;
- BF16 producer rounding reproduced at deleted private boundaries when its
  value is observable by the composed VJP.

Precision is assigned from range and cancellation requirements, not from a
variable's name. Runtime precision selection and data-dependent compensation
remain forbidden.

### Saved tensors

The selected reverse keeps compact forward cache when recomputation is more
expensive than its HBM traffic. A small query-gauge panel is reconstructed;
larger pair/tail/state panels remain saved where recomputation lost complete
F+B. Unrequested final states do not materialize full zero cotangents.

## Rejected Directions

The following decisions are closed for the current production path. Their
implementations and detailed measurements remain recoverable in Git history.

| Direction | Reason it is not selected |
| --- | --- |
| Former bounded-LDU frame | More expressive, but its exact frame/chart reverse and private glue made training latency impractical. Recovery point: commit `2237875`. |
| Fixed-degree Neumann frame | Numerically admissible in the old chart, but repeated broad actions were much slower than the selected solve and are irrelevant to the RLS operator. |
| QRD, Woodbury, or explicit inverse-state RLS | Algebraically valid, but factorization, boundary solves, storage, and reverse ownership lost to the resident MESA gain path. |
| Natural-coordinate RLS as another public backend | Duplicated operator ownership and required more dense state work without a complete F+B win. |
| Flat physical `3T` exterior | Expanded output/cotangent/checkpoint storage and obscured token ownership; native E3 retains the same three real transitions. |
| Source-parallel atomic E3 reverse | Reduced output ownership and generated excessive cross-value-tile atomics. |
| Sequence/head resident mega-kernel | Collapsed chunk/value parallelism to a small number of long-lived CTAs. |
| Unconditional source-to-pair fusion | Shortened one forward boundary but lengthened reverse lifetimes and increased graph reservation. |
| Independent `bar_W` owner | Added an FP32 write/read boundary and launch without improving complete F+B. |
| Analytic local direct TRSM replacing WY inverse cache | Correct enough for the private precision envelope, but slower before transpose work was included. |
| Fusing mass into a dense MESA tile | Serial scalar recurrence extended the dense state CTA's critical path. |
| Packed symmetric `J` production storage | Saved capacity but disturbed mature dense tile access and did not improve the target graph. |
| Recomputing all E3 panels | Saved HBM capacity but repeated pair/state work cost more than reading the retained cache. |
| Lowering CG or sensitive scalar reductions to BF16 | Lost accuracy and did not improve complete F+B; some variants were slower. |
| Low-rank frontend projection as a lossless optimization | Changes model capacity unless full rank is retained; it remains a training ablation, not an implementation substitution. |
| Runtime backend or precision selectors | Create multiple contracts and data-dependent behavior; neither is accepted by the current operator. |

Reconsidering a closed direction requires new algebra or a materially different
mature upstream owner, the same composed-VJP gates, and a complete F+B A/B. A
local kernel microbenchmark alone is insufficient.

## Upstream Issues That Changed Validation

Two FLA issue classes informed the acceptance suite rather than model math:

- a reported chunk-length gradient failure
  ([FLA issue #984](https://github.com/fla-org/flash-linear-attention/issues/984))
  requires irregular lengths, cross-chunk cases, and multiple gate regimes;
- a reported shared-memory race
  ([FLA issue #889](https://github.com/fla-org/flash-linear-attention/issues/889))
  requires repeat-run gradient stability for new TileLang/Triton schedules.

FLA's MESA NaN-hardening work also informed exponent-range handling in the CG
action. SolveDelta keeps the broad MESA operand in BF16 rather than narrowing
it to FP16 merely because another bounded private panel can use FP16. This is a
static precision decision, not a runtime fallback.

## Maintenance Rule

Update this file when an external source changes production mathematics,
precision, ownership, layout, or model integration. Record:

1. the upstream project, revision, and license;
2. the concrete source block or schedule used;
3. the SolveDelta algebraic mapping and local specialization;
4. the matching transpose/reverse owner;
5. the final adoption or rejection decision.

Do not append profiler transcripts, temporary kernel names, speculative
roadmaps, or step-by-step experiment diaries. Keep current benchmark numbers
in `README.md` and the native acceptance profile in
`docs/FROM_SCRATCH_REBUILD.md`; Git history is the archive for superseded A/B
results.
