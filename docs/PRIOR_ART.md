# Prior Art and Production Provenance

SolveDelta's contribution is the operator: an online geometry solution, a
bounded relative frame, and its composition with a Delta memory. Its GPU path
builds on years of work in Flash Linear Attention and adjacent recurrent-model
libraries. This ledger makes that boundary concrete by recording the donor,
license, adapted schedule, and SolveDelta-specific change for every material
influence.

## Ownership boundary

SolveDelta contributes:

- the residual predictor recurrence;
- the mapping from its residual write to the relative frame;
- the accumulated primal and residual-local dual address contract;
- the composition with channel decay and the ordinary Delta edit/read;
- the public `(C,S)` state and mixed-precision contract;
- model integration, tests, and acceptance gates.

Upstream projects contribute the mature execution vocabulary: normalization,
pair formation, triangular WY action, chunk state/output ownership, strict
transpose, causal conv, and the model shell.

## Flash Linear Attention

Repository: <https://github.com/fla-org/flash-linear-attention>

License: MIT. The license text is preserved in `LICENSES/MIT.txt`; adapted
production code is also listed in `THIRD_PARTY_NOTICES.md`.

The selected runtime and source-compatibility baseline is FLA `0.6.0`.
SolveDelta imports private kernel modules, so this version is pinned until a
new release passes the full oracle, VJP, model, and CUDA Graph suite. Runtime
installation details are maintained in `docs/ENVIRONMENT.md`.
The September 2026 audit compared the pinned CUDA sources at
`35dceaee5408e69a555fec34cb215c93c375dabe` with upstream main
`8e84ed4a6727be082c34a3855c60623fd11411e9`. No intervening CUDA change in
gated Oja, generalized DPLR, GDN2/KDA, L2Norm, or fused norm-gate replaces a
selected owner; the newer relevant commit was Ascend-specific.

### Gated Oja predictor

Principal source areas:

- `fla/ops/gated_oja_rule/chunk_kkt.py`;
- `fla/ops/gated_oja_rule/wy_fast.py`;
- `fla/ops/gated_oja_rule/chunk_h.py`;
- `fla/ops/gated_oja_rule/chunk_o.py`;
- the matching backward owners in those files.

The Residual-Frame predictor keeps FLA's pair/WY/state/output ownership but
specializes its algebra to the pre-forgetting residual. For coordinate prefix
`G_{i,c}=sum_{l<=i} log_alpha_{l,c}`, the strict source interaction is
`gamma_i sum_c u_{i,c}u_{j,c}exp(G_{i,c}-log_alpha_{i,c}-G_{j,c})` for `j<i`.
The target branch consumes `gamma*h`; the source branch consumes the
coordinatewise exclusive prefix. Every active exponent is nonpositive, and no
reciprocal retention is formed. The strict transpose uses the same schedule and
closes a final-shaped channel cotangent before returning to public operands.

The production specialization retains FLA's:

- source Gram/pair construction;
- chunk-local unit-lower triangular solve;
- recomputed WY source/update panels;
- FP32 chunk-boundary matrix state;
- resident state forward;
- chunk-parallel accumulated-state output;
- reverse chunk-state traversal;
- output/state, WY, and pair transposes.

SolveDelta keeps the chunk mechanics and specializes their surface. The Oja
query/output branch evaluates `C_t^T k_t` for the accumulated primal; its
strict reverse is retained. The coordinate-gated pair keeps FLA's two-level
schedule: cross-16-token subchunks use centered Tensor Core contractions and
diagonal subchunks use direct bounded coordinate reductions. Its transpose is
partitioned by token subchunk and coordinate tile, with FP32 beta partials.
The public ABI remains independent of FLA's private panels.
The final reverse specialization fuses FLA's gate-branch merge with its
chunk-local suffix sum and uses one final-shaped owner for the state, WY, and
pair source cotangents. This removes launch-only epilogues without changing
the upstream pair/WY/state ownership.

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

Production adapts FLA's exact unbounded scalar pair forward/transpose to the
source-native rectangular panels. Because `k=b=d` and `A_qk=A_qb`, the state
and output owners use the common-left identity
`d z^T+d z_new^T=d(z+z_new)^T`. Forward shares one contraction; reverse makes
the final `z/z_new` cotangent the owner and deletes the separate `dV` pass.
Fast-WY, FP32 state boundaries, chunk-parallel output, state reverse, and
triangular transpose keep their donor ownership.

### L2 normalization, decay, and output gate

Principal source areas:

- `fla/modules/l2norm.py`;
- `fla/layers/kda.py`;
- `fla/ops/kda/gate.py`;
- `fla/modules/fused_norm_gate.py`;
- `fla/ops/rwkv6/chunk.py`;
- `fla/ops/utils/`.

The standalone native L2Norm keeps FLA's row ownership and strict transpose for
`u` and the edit key. The relative source owner applies the same FP32 reduction
to the strided query while generating exterior panels. Bounded dual actions
write private FP16 `e/chi` panels directly from FP32 registers; the accumulated
primal and decay-scaled operands use BF16 for range.

KDA supplies the low-rank coordinate-decay parameterization and fused gate
transpose. FLA's sigmoid-gated RMSNorm owns the final readout. The exterior
uses FLA's chunk-cumsum convention, with its transpose closed in the direct-e
pair owner. Norms, cumsums, and sensitive scalar reductions stay FP32.

The source specialization also uses `||u||=1` from its unique L2Norm producer
when radializing the frame covector. The corresponding radial cotangent lies in
the L2Norm nullspace. At the selected width, one warp wins in each direction;
wider schedules add synchronization without useful row work.

The final readout adopts FLA's norm-linear checkpoint policy: its strict
transpose regenerates the normalized gated output for the projection-weight
GEMM. The direct-e and source reverse owners likewise keep FP32 local
accumulation, then use a BF16 final-shaped handoff after reduction.

## Rank-one invertibility precedents

Rezende and Mohamed's planar normalizing flow identifies the determinant
condition for identity-plus-rank-one maps and reparameterizes the parallel
component to preserve invertibility. Behrmann et al.'s invertible residual
networks use a strict residual norm bound to obtain global invertibility.
SolveDelta uses the stronger norm-bounded option because a merely positive
determinant may still approach zero or permit a large shear:

- <https://proceedings.mlr.press/v37/rezende15.html>;
- <https://proceedings.mlr.press/v97/behrmann19a.html>.

Their mathematical conditions motivate the static
`||u|| ||phi|| < 5/8` contract. SolveDelta uses the ideas, rather than source
code, while the GPU arithmetic follows the FLA-derived owners above.

## Model shell

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

The dense MLP stores gate/up weights as one `[2I,D]` `gate_up_proj`, performs
one common-input GEMM, then splits the result before FLA's fused SwiGLU/down
owner. Packed gate-up weights are an established Transformers and tensor-
parallel layout; the local change is a parameter packing, not copied kernel
source. An FP64 output/VJP test proves equivalence after concatenating the two
ordinary projection weights.

## MESA as a research precursor

FLA MESA supplied the paired covariance/cross-moment scans and matrix-free CG
used by the earlier RLS prototype. That work helped expose the underlying
second-order problem. Residual-Frame now advances the solution directly in
`C`, so MESA remains mathematical and implementation context rather than a
runtime dependency of the geometry path.

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

Mamba contributes design patterns rather than copied source:

- direct addressing of fused `[B,T,H,r]` projection views;
- FP32 continuation state with low-precision contraction operands;
- separate state and output ownership;
- trainer-owned CUDA Graph boundaries.

## causal-conv1d

Repository: <https://github.com/Dao-AILab/causal-conv1d>

License: BSD-3-Clause.

The frontend calls the runtime package for independent depthwise causal conv4,
SiLU, recurrent final conv state, and its complete VJP. The selected
compatibility baseline is causal-conv1d `1.7.x`.

## PyTorch and Hugging Face Transformers

PyTorch supplies autograd composition, CUDA Graph capture, allocator
accounting, and module/runtime primitives. Hugging Face Transformers supplies
`PretrainedConfig`, auto-class registration, checkpointing, generation, and
causal-LM result types. Their runtime licenses and dependency metadata remain
owned by those packages.

The optional accumulation path follows the mixed-precision ownership used by
NVIDIA Megatron-LM and Apex: FP32 optimizer masters, resident low-precision
model weights, and one refresh after an optimizer update. SolveDelta implements
the narrower Linear case with PyTorch `F.linear`, `mm`, nonpersistent buffers,
and an optimizer post-step hook. The design influence is architectural; the
implementation is local.

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
the graph.

For fixed dense fused-linear loss, SolveDelta reuses FLA's chunked linear,
logsumexp, and cross-entropy kernels with the static causal-LM denominator
`B(T-1)`. This removes FLA's capture-time `.item()` and always launches the
small scalar-cotangent epilogue instead of branching on a device tensor.

PyTorch PR 189914 removes eager TorchScript compilation from the Inductor
import path: <https://github.com/pytorch/pytorch/pull/189914>. Stable PyTorch
2.13.0 predates that change, so package initialization scopes the exact MKLDNN
`script_method` deprecation raised while importing FLA.

## Selected implementation decisions

| Decision | Selected form | Reason |
| --- | --- | --- |
| Geometry state | FP32 solution `C` | Carries the online second-order fit in action-ready coordinates |
| Primal/dual geometry | Accumulated `d=(I+C)^T k`; residual-local `F^-T` for `e,chi` | Full solution history on the stable non-inverting path; bounded residual-aligned dual |
| Predictor schedule | FLA coordinate-gated Oja C32 pair/WY/state/output with split pre-decay coefficients | Exact right-coordinate leak without reciprocal retention; mature state/output transpose |
| Memory schedule | Exact unbounded Triton direct-e pair + FLA DPLR C16 | Exact decay semantics with mature pair/WY/state/output ownership |
| Delta gates | Independent channel-wise `sigmoid(erase_raw/write_raw)` | Preserves the full GDN2 update surface and separate source cotangents |
| Decay/readout | KDA low-rank coordinate gate + FLA sigmoid-gated RMSNorm | Matches the GDN2 memory surface and selected plain readout |
| Source ABI | BF16 accumulated direct plus panel-native paired sources | Keeps unbounded `C` action in BF16 and bounded dual panels in FP16 |
| Fusion | Selective | Preserves useful CTA parallelism and short lifetimes |
| Public state | `(C,S)` | Geometry solution plus Delta memory |
| Precision | bounded FP16 dual panels, BF16 primal/decay-scaled operands/final-shaped cotangent handoff, FP32 accumulation/state/scalars | Uses extra mantissa where a static range proof or composed-VJP evidence requires it |
| Fused projection | 64-row physical alignment, logical prefix consumers | Improves the dominant projection transpose and preserves strided views |
| MLP input projection | One packed `D -> 2I` gate-up GEMM | Exact common-input factorization; removes one launch and a 2 MiB Graph slot |
| Accumulation weights | Optimizer-bound BF16 Linear shadows | Amortizes casts while FP32 masters retain optimizer ownership |
| Tails and masks/resets | Neutral tail padding plus reset-free segmented native batch | Exact state identity on padding; one native owner batch without token-wise Python scheduling |
| Single-token cache | Pre-forgetting recurrent predictor + FLA recurrent DPLR | Inference-only owner avoids C16 padding while retaining FP32 `(C,S)` |
| Distributed CUDA Graph | Local graph first, DDP reducer outside capture | Gives compute and collectives independent lifecycle ownership |

## Alternatives evaluated

| Alternative | Outcome |
| --- | --- |
| Bounded-LDU exact chart | Much stronger instantaneous chart, but its exact action/transpose did not reach usable training latency |
| RLS `m/J/D` + MESA CG | Strong covariance conditioning, but larger state and reverse cost than the solution-coordinate update |
| Dense accumulated `P^-T` continuation | Duplicates a deterministic `r^2` inverse state and complicates reverse/checkpoint ownership |
| Runtime RLS/Residual selector | Split semantics and made one authoritative oracle impractical |
| Flat expanded token axis | Exposes private edits as a synthetic sequence and inflates checkpoints |
| Sequence/head resident mega-kernel | Measured loss of chunk/rank/value CTA parallelism |
| Centered Tensor-Core direct-e pair | Requires a static decay lower bound; long-run unbounded decay overflowed despite finite model inputs |
| TileLang runtime pair path | Coupled to the centered owner; the exact-unbounded path selected FLA's Triton runtime |
| FLA fused DPLR `chunk_ho` at C16 | Saved state/output HBM but regressed the target shape because too few CTAs serially advanced all chunks |
| Public canonicalization copies | Stride-aware owners made the copies unnecessary |
| Grouped decay/output-gate GEMM | Packing and grouped-bias epilogues exceeded the two-GEMM launch saving |
| Save relative-source panels for backward | Saves about 19 us core F+B but retains 6 MiB of activations per layer; local recomputation is the better deep-training trade |
| BF16 predictor pair / exterior triangular pair | Predictor had no complete-path gain; exterior coefficients are FP32 triangular loop state |
| Two-level predictor pair transpose at C32 | Improved the isolated kernel but not complete Graph F+B because the shallow cross-subchunk work extended fragment lifetime |
| BF16-to-FP16 conversion | Discarded BF16 mantissa bits cannot be recovered by a later cast |
| Denominator clamp/fallback | The smooth radial parameterization supplies an analytic lower bound |

## Provenance maintenance

When a future change copies or structurally adapts upstream code:

1. record the exact source file/revision and mathematical mapping here;
2. preserve copyright/license headers where source is copied;
3. update `THIRD_PARTY_NOTICES.md`;
4. connect both forward and strict transpose;
5. compare only after the same oracle and composed-VJP gates pass.

Provenance follows material mathematical or scheduling influence, even when
tensor names, shapes, and public ABIs change during specialization.
