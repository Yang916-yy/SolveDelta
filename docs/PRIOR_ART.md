# Prior Work and Decision Ledger

This file records primary sources that changed the single SolveDelta design. A
source is not a second project direction. Access and review date: 2026-08-22.

## Delta recurrence and parallel algebra

- Schlag, Irie, and Schmidhuber, *Linear Transformers Are Secretly Fast Weight
  Programmers*, [arXiv:2102.11174](https://arxiv.org/abs/2102.11174).
- Yang et al., *Parallelizing Linear Transformers with the Delta Rule over
  Sequence Length*, [arXiv:2406.06484](https://arxiv.org/abs/2406.06484).
- Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*,
  [arXiv:2412.06464](https://arxiv.org/abs/2412.06464).
- *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*,
  [arXiv:2605.22791](https://arxiv.org/abs/2605.22791), with the
  [official implementation](https://github.com/NVlabs/GatedDeltaNet-2).
- Official FLA implementations for
  [DeltaRule](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/delta_rule),
  [Gated Delta](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule),
  and [GDN2](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gdn2),
  together with its
  [forward/backward correctness suite](https://github.com/fla-org/flash-linear-attention/blob/main/tests/ops/test_gdn2.py).
- FLA issue reports for a
  [chunk-64 gradient failure](https://github.com/fla-org/flash-linear-attention/issues/984)
  and a
  [shared-memory race](https://github.com/fla-org/flash-linear-attention/issues/889).

Decision: SolveDelta targets exact GDN2 containment and exposes the number of
ordered prediction-error edits as a static hyperparameter `K`; all edits share
one transpose-dual solve adapter. Write
uses the primal action; erase and read use its dual. The associative phase
therefore uses generalized asymmetric WY cross terms. GDN2 supplies the
asymmetric erase/write precedent; GatedDeltaProduct supplies the ordered
multi-edit packing precedent. The canonical write direction is the normalized
key itself. A separate sigmoid write-direction gate was removed because exact
baseline recovery required its unattainable endpoint, while its extra local
capacity was not part of the solve-frame contribution.

The FLA GDN2 suite treats a pure PyTorch token recurrence as ground truth and
uses RMS error relative to reference RMS. Its dense tests allow about `0.005`
for outputs/final state, `0.01` for main vector gradients, and `0.02` for gate
gradients; packed tests use about `0.006`, `0.012`, and `0.02`. It covers
forward and backward, FP32/FP16, irregular lengths, initial/final states,
packed resets, normalization and gate fusion. SolveDelta adopts that end-to-end
low-precision envelope but compares against an FP64 oracle built from the same
quantized inputs. It adds stricter independent budgets for the Triton geometry
scan, MathDx residual/actions, dual pairing, and FP32 complete layer. This
prevents a locally inaccurate solve from consuming the full recurrent-layer
tolerance while remaining hidden by output scale. A local SM120 calibration of
the installed GDN2 chunk path measured roughly `1.0e-3--1.2e-3` FP32
output/state relative error, so SolveDelta's complete FP32 output/state ceiling is
`2e-3`; the solve-specific ceilings remain orders of magnitude tighter.

Decision: mature Delta implementations recompute chunk-local structure in
backward rather than retaining tokenwise recurrent states. SolveDelta follows
that schedule for geometry/frame state, but saves FLA's compact WY exterior
because a local target-profile benchmark found it about 28% faster for only
about 5.5 MiB extra peak memory. The two FLA issue reports changed validation,
not model math: cross-chunk lengths, multiple seeds and gate regimes, and
repeat-run gradient determinism are mandatory checks for the optimized path.

Decision: the installed FLA 0.5.2 GDN2 and GatedDeltaProduct layers establish
three independent depthwise causal `conv4`, bias-free, SiLU branches over the
projected query, packed keys, and packed values. Convolution precedes head
reshape and query/key normalization; geometry and gate branches bypass it.
SolveDelta adopts exactly that fixed frontend with one structural enable switch
and carries all three caches in its layer state. A local output/state/VJP audit
of FLA's Triton causal convolution found that its main output VJP is correct,
but a cotangent on the returned final cache omits part of
`d(final_state)/dx`. SolveDelta therefore asks FLA only for the convolution
output and constructs the exact four-token final cache by slicing
`concat(initial_cache, x)`; the CUDA parity test covers output, cache, input,
initial-cache, and weight gradients.

## Preconditioned and regression memories

- Tumma, Loo, and Rus, *Preconditioned DeltaNet: Curvature-aware Sequence
  Modeling for Linear Recurrences*,
  [arXiv:2604.21100](https://arxiv.org/abs/2604.21100).
- Zhou et al., *OSDN: Improving Delta Rule with Provable Online
  Preconditioning in Linear Attention*,
  [arXiv:2605.13473](https://arxiv.org/abs/2605.13473), with the
  [official implementation](https://github.com/Lhongpei/OSDN).
- *MesaNet: Sequence Modeling by Locally Optimal Test-Time Training*,
  [arXiv:2506.05233](https://arxiv.org/abs/2506.05233), with the
  [FLA reference](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/mesa_net).
- Le et al., *Don't Read Everything: A Curvature-Conditioned Query for Linear
  Attention*, [arXiv:2606.01294](https://arxiv.org/abs/2606.01294).
- Cutkosky and Sarlos, *Matrix-Free Preconditioning in Online Learning*,
  [arXiv:1905.12721](https://arxiv.org/abs/1905.12721).

Decision: `G_t^-1 k_t`, a learned positive preconditioner, online ridge
statistics, or covariance-conditioned reading alone are established ideas.
They cannot be the novelty claim. They are mandatory baselines. The narrower
SolveDelta claim is the driven-cross-moment-conditioned causal solve frame, its
bounded asymmetric dual factors, geometry-conditioned read, configurable
ordered edit depth, and exact Delta-family reductions.

Matrix-free and diagonal online preconditioners show that useful curvature
adaptation need not pay for a full solve. They are therefore important
quality--efficiency controls, but they do not reproduce SolveDelta's full-prefix
LSSO coordinate or its primal/dual similarity action.

## Expressive generalized Delta transitions

- Siems et al., *DeltaProduct: Improving State-Tracking in Linear RNNs via
  Householder Products*, [arXiv:2502.10297](https://arxiv.org/abs/2502.10297).
- Official FLA
  [GatedDeltaProduct layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltaproduct.py)
  and
  [chunk operator](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_product).
- Kimi Team, *Kimi Linear: An Expressive, Efficient Attention Architecture*,
  [arXiv:2510.26692](https://arxiv.org/abs/2510.26692).
- Official FLA
  [generalized Delta design note](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/generalized_delta_rule).

Decision: products of stable edits, DPLR transitions, channel-wise decay, and
separate erase/write factors are established. SolveDelta therefore does not claim
the multi-edit product itself as new. At edit depth `K` it contains
DeltaProduct-`K` as an exact identity-geometry reduction and must compare
against the corresponding baseline. The contribution is that bounded
asymmetric edits share a full-prefix LSSO transpose-dual adapter and use general dual
erase/write factors. `K=1` is the current performance default; `K>1` remains a
supported capacity setting and the exact DeltaProduct-`K` reduction remains a
required test. This changes the default compute point rather than the model
family's expressivity, and edit count is not part of the novelty claim.

Decision: GDN2 already uses asymmetric rank-one erase/write factors, so mild
non-normality is not unique to SolveDelta. A fresh algebraic audit corrected a
stronger claim: `a^T b >= 0` fixes the one nontrivial eigenvalue but does not
make the symmetric part of `ab^T` positive semidefinite unless the factors are
positively collinear. The former H+S/dissipative label was removed. The
remaining project-specific mechanism is accurately named an orthogonal erase
residual.

## LSSO provenance and inherited solve-geometry properties

- Yang, *LSSO: Solving Contextual Adaptation with Certified Global Mixing*,
  [LSSO repository](https://github.com/Yang916-yy/LSSO).
- Bauschke, Moffat, and Wang, *Firmly nonexpansive mappings and maximally
  monotone operators: correspondence and duality*,
  [arXiv:1101.4688](https://arxiv.org/abs/1101.4688).

Decision: with Rank-Rotary disabled, LSSO's soft frame admits a fixed-size
prefix representation through a normalized rank Gram and rank-core cross
moment. Causal LSSO inherits the solved-operator principle: prefix statistics
generate a compact system and its solve action defines contextual adaptation.
It does not inherit the bidirectional `I+F F^T+Omega` chart. Prefix-LSSO is a
provenance/property oracle; SolveDelta is the recurrent associative model.

Decision: the bidirectional resolvent only needs the solve direction bounded.
SolveDelta also exposes the inverse-transpose action, so both the causal system and
its inverse must be controlled. Rather than repair the old chart with a global
raw-coordinate guard, the canonical model now uses a directly parameterized
bounded LDU system. Norm-constrained operators provide general precedent for
the bi-bounded requirement:

- Miyato et al., *Spectral Normalization for Generative Adversarial Networks*,
  [arXiv:1802.05957](https://arxiv.org/abs/1802.05957).
- Cisse et al., *Parseval Networks: Improving Robustness to Adversarial
  Examples*, [PMLR 70](https://proceedings.mlr.press/v70/cisse17a.html).

The exact triangular radial formula and its primal/dual bounds are derived for
SolveDelta; they are not attributed to those architectures.

Decision: the original dynamic coordinate contains the composition
`L^-1 C W_drive`. Linearity permits the exact driven-moment recurrence
`D_t = lambda D_{t-1} + u_t (W_drive^T c_t)^T`, and the independent core
feature lets the composite be projected directly. This removes a redundant
dense matrix product while retaining a fixed-size driven prefix coordinate;
exact bidirectional chart matching is only a provenance diagnostic.

Decision: token-history dualization and associative-memory editing remain
separate. Prefix moments condition every edit through the adapter, but they are
not a lossless encoding of the ordered prefix. The Delta state stays in a fixed
ambient basis and retains ordered content; moving it into each changing prefix
frame would require dense cross-frame transport and would defeat the recurrent
and WY contracts.

## Causal solve-chart selection

- Kingma and Dhariwal, *Glow: Generative Flow with Invertible 1x1
  Convolutions*, [arXiv:1807.03039](https://arxiv.org/abs/1807.03039), with the
  [official implementation](https://github.com/openai/glow).
- Koehler, Mehta, and Risteski, *Representational Aspects of Depth and
  Conditioning in Normalizing Flows*,
  [PMLR 139](https://proceedings.mlr.press/v139/koehler21a.html).
- Meng et al., *ButterflyFlow: Building Invertible Layers with Butterfly
  Matrices*, [PMLR 162](https://proceedings.mlr.press/v162/meng22a.html).
- Dao et al., *Learning Fast Algorithms for Linear Transforms Using Butterfly
  Factorizations*, [arXiv:1903.05895](https://arxiv.org/abs/1903.05895).
- Lu and Huang, *Woodbury Transformations for Deep Generative Flows*,
  [arXiv:2002.12229](https://arxiv.org/abs/2002.12229).
- Helfrich, Willmott, and Ye, *Orthogonal Recurrent Neural Networks with Scaled
  Cayley Transform*, [PMLR 80](https://proceedings.mlr.press/v80/helfrich18a.html).
- Likhosherstov et al., *CWY Parametrization: a Solution for Parallelized
  Optimization of Orthogonal and Stiefel Matrices*,
  [PMLR 130](https://proceedings.mlr.press/v130/likhosherstov21a.html).
- Behrmann et al., *Invertible Residual Networks*,
  [PMLR 97](https://proceedings.mlr.press/v97/behrmann19a.html).

Decision: the candidate screen compared the bidirectional accretive chart,
pure SPD solve, orthogonal-scale/SVD-style frames, dense exponential and Cayley
charts, Householder/CWY, invertible butterfly, Woodbury low-rank updates,
contractive residual inverses, affine block couplings, and direct LDU.

Pure SPD has only `r(r+1)/2` local directions. A one-sided orthogonal-scale
frame has the same dimensional ceiling. Butterfly and Woodbury reduce work but
restrict the dynamic family to `O(r log r)` or rank-`p` coordinates. Dense
exponential/Cayley and residual charts retain difficult dense or iterative
inverse work. Constant-depth affine couplings are expressive: Koehler et al.
prove that at most 24 alternating linear coupling pairs represent any
orientation-preserving invertible linear map, but a small fixed number is not
universal and accumulating many factors worsens sequential depth and condition
bounds.

The selected bounded direct LDU output chart owns exactly `r^2` ambient matrix
coordinates, is smooth and full differential rank at identity with respect to
those coordinates, has analytic inverse and inverse-transpose actions, and
reduces execution to two unit-triangular solves plus diagonal scaling. This
chart-dimensionality statement is distinct from feasible-prefix reachability:
before `t=r`, the driven moment has rank at most `t` and its column space lies
in the remembered geometry span. Its two prefix coordinates are
bounded separately before their factor contributions are added: `H` supplies
sign-robust occupancy geometry and `R` supplies signed directional drive.
This separate nonlinearity prevents the exact `D+J` cross-moment collapse;
their first-order effects coincide, while their higher-order responses remain
distinguishable. The summed strict-triangular Frobenius radii and bounded log
diagonal give a global condition-number bound near 4.58. Glow supplies mature
precedent for directly owning triangular factors; the precise prefix-generated
two-coordinate bi-bounded system and transpose-dual Delta use are SolveDelta
constructions.

The alternatives remain research evidence, not model variants.

## Prefix scans and numerical solves

- Särkkä and García-Fernández, *Temporal Parallelization of Bayesian
  Smoothers*, [arXiv:1905.13002](https://arxiv.org/abs/1905.13002).
Decision: affine forgetting moments have an exact associative prefix scan. The
bounded LDU chart removes numerical factorization. DeltaNet/GDN/KDA establish
the chunk-boundary scan and backward recomputation pattern; SolveDelta reuses
that outer scan for two matrix payloads and one scalar payload. It never stores
full-sequence `T x r x r` states.

- Gohberg, Kailath, and Koltracht, *Linear Complexity Algorithms for
  Semiseparable Matrices*,
  [DOI:10.1007/BF01213791](https://doi.org/10.1007/BF01213791).
- Chandrasekaran et al., *Some Fast Algorithms for Sequentially Semiseparable
  Representations*,
  [DOI:10.1137/S0895479802405884](https://doi.org/10.1137/S0895479802405884).
- Pernet and Storjohann, *Time and Space Efficient Generators for
  Quasiseparable Matrices*,
  [DOI:10.1016/j.jsc.2017.07.010](https://doi.org/10.1016/j.jsc.2017.07.010).
- Hogg, *A Fast Dense Triangular Solve in CUDA*,
  [DOI:10.1137/12088358X](https://doi.org/10.1137/12088358X).

Decision: strict masking destroys ordinary low rank, but it retains a
sequentially semiseparable generator. For one geometry chunk, the exact affine
prefix decomposition writes each token's strict factor as one dense boundary
triangle plus masked outer products from the at most `C` local sources. A
rank-coordinate solve therefore carries two `C x C` packets of prefix inner
products instead of materializing `C x r x r` factors. Boundary work is a
panel action and local work is a generator scan, giving
`O(C r^2 + C^2 r)` per chunk and exact lower, upper, transpose, and skew
actions. These sources informed the generator interpretation and GPU solve
schedule. SolveDelta's two-axis affine coefficients, separate radial maps,
stable cancellation treatment, primal/dual pairing, and combined
`d/e/chi` packet are project-specific derivations.

The selected skew-action tile pairing is also a project-specific derivation;
no external kernel or algorithm was copied for it. The semiseparable sources
above justify retaining masked local outer products as generators, but do not
give this packet specialization. SolveDelta partitions the 128 coordinates
into eight 16-coordinate tiles. For each unordered tile pair it reads the two
directed boundary blocks once, forms both blocks related by
`Omega^T=-Omega`, and applies them to all 16 token RHS. Fixed twofold-FP32
partials carry boundary contributions into a second kernel, where they are
reduced in fixed order together with the exact local generator action. This is
algebraically the same dense `Omega=1/2(E-E^T)` action; it neither compresses
`J/D` nor changes frame update frequency. The decision it informed was to
replace the repeated-RHS boundary traversal in the C16 forward. A direct
primal-kernel fusion and a lower-workspace row-tile schedule were measured and
rejected as slower exact schedules. Applying the same identity to the skew VJP
as `grad_direct-Omega grad_omega` is valid, but its fully compensated prototype
was slower than the selected contract-passing backward action and therefore
was not adopted.

The generic quasiseparable order of a dense continuation can still grow to
`floor(r/2)`. The packet algorithm is consequently an exact chunk execution,
not evidence that the full chart has constant rank or permission to replace
`J,D` with a compressed state. A plain IEEE-FP32 Gram expansion of the radial
norm remains rejected: legal cancellation between a dense boundary and a local
outer product loses the residual and can reconstruct a negative squared norm.
The selected radial pass instead uses four independent channels
`(J_lower,D_lower,J_upper,D_upper)`. For each channel it evaluates the exact
affine-prefix quadratic from boundary norms, boundary/local contractions, and
local outer-product Grams. Large contractions use one fixed twofold FP32 path,
and the small `C x C` quadratic is composed in FP64. This retains arbitrary
asymmetric `J,D` boundaries, costs `O(C r^2 + C^2 r)`, and introduces no clamp,
fallback, or saved training statistics. Independent `2^12` cancellation probes
for both `J` and `D`, including ordinary asymmetric tails, remain finite and
match the direct FP64 matrix oracle within the bounded-chart ceiling.

- Seeger, *Low Rank Updates for the Cholesky Decomposition*,
  [EPFL technical report](https://infoscience.epfl.ch/entities/publication/00ba309a-d155-4e21-acc2-153702c4605c).
- Stewart, *Error Analysis of QR Updating with Exponential Windowing*,
  [University of Maryland report](http://hdl.handle.net/1903/557).

Decision: square-root covariance and QR updating establish stable `O(r^2)`
updates for scaled low-rank modifications. They informed the earlier Gram
factor study, but the selected direct LDU system no longer maintains or
factorizes a normalized fixed-ridge matrix. The former ridge-versus-rank-one
Cholesky conflict is therefore closed rather than approximated.

- NVIDIA, *Using cuBLASDx TRSM*,
  [official documentation](https://docs.nvidia.com/cuda/cublasdx/using_trsm.html).
- NVIDIA, *cuBLASDx Quick Installation Guide*,
  [official documentation](https://docs.nvidia.com/cuda/cublasdx/installation.html).
- NVIDIA, *cuSolverDx Triangular Matrix-Matrix Solve*,
  [official documentation](https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/trsm.html).
- NVIDIA, *cuBLAS batched TRSM*,
  [official documentation](https://docs.nvidia.com/cuda/cublas/).

Decision: exact factorwise execution is the standalone numerical oracle and a
decode candidate. At rank 128,
MathDx supports a CUDA block cooperating on one triangular solve, with input
and overwritten output staged in shared memory. MathDx 25.12 exposes this
operation through cuSolverDx, while MathDx 26.06 moves it to cuBLASDx. These
device-library paths require CUDA C++ compilation/device linking and are not
directly callable from an ordinary Triton JIT kernel. The selected engineering
boundary keeps the mature Delta-family chunk scan and Delta/WY work in Triton,
uses MathDx for the standalone exact solve oracle, and inserts a CUDA C++
operator for chunk-local prefix reconstruction, bounded factor generation,
coordinate-packet primal actions, and direct dual products. The local first target is
PyTorch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0 Update 2, and MathDx 26.06
cuBLASDx 0.7.0. An isolated official block-TRSM example has compiled, linked
through `libcublasdx.fatbin`, dispatched on SM120, and returned zero reported
L2 error; this validates the exact toolchain boundary rather than the
performance of the fused training operator. The chunk boundary avoids
materializing `T x r x r` prefix states while
retaining the established outer schedule. The selected C16 forward and its
transposed packet VJP use exact FP32 coordinate substitution. The earlier
degree-four Neumann implementation remains an independently validated
approximation only. MathDx remains the exact comparison boundary.

The backend screen also considered handwritten recursive/block TRSM in Triton,
CuTe DSL, TileLang, or CUDA tile libraries; host-level batched cuBLAS;
truncated Neumann/polynomial inverses; and replacing LDU with butterfly,
coupling, or hierarchical nilpotent-shear factors. The first
group duplicates architecture-specific solver work already owned by MathDx,
host batched TRSM forces dynamic factors through global memory, and changing
factor families would make backend convenience alter the model mathematics.
The polynomial candidate was initially rejected as an unbounded general solve,
then adopted only after the selected chart supplied a global nilpotent-factor
bound and empirical error stayed well below the composed-frame budget. MathDx
therefore owns exact factor-action validation while Triton/CUDA/FLA own the
bounded training schedule.

Decision: an SM120 Triton VJP for the isolated bounded LDU chart is retained as
a numerical validation unit, not as the production backward boundary. Its
factor-coordinate gradients agree with FP64 autograd to roughly `3e-7`
relative RMS, but a direct integration materialized full factor cotangents and
raised both complete-layer time and peak allocation. Folding moment
normalization into the same kernel recovered a small same-run speedup while
still adding about 91 MiB of peak allocation and failing to beat the retained
stable endpoint. The next production experiment must consume the low-rank
action cotangents inside the chart reduction and avoid writing full
`grad_lower/grad_upper`; optimizing the isolated chart again is not an accepted
work reduction.

Decision: a narrow-RHS audit found no free reparameterization that preserves
the current dense chart, its full `r^2` ambient differential rank, and the
shared primal/dual adapter while reducing a general action below quadratic
work. Triangular masking destroys the ordinary low-rank structure of a prefix
outer product: a generic `tril(u h^T,-1)` can already have rank `r-1` and
nilpotent depth `r`. Butterfly, Woodbury, fixed-band, or direct-inverse charts
would therefore restrict the operator or move the solve to the wider
erase/query side. Strictly equivalent execution experiments remain allowed:
right-side TRSM with lazy transpose, absorbing the diagonal into the upper
factor, zero-padding RHS to a device-library-friendly width, and bounded-tile
batched cuBLAS over exactly the same generated factors. They require
end-to-end traffic and workspace measurements before adoption and do not alter
the canonical chart.

Decision: ordinary temporal compact WY cannot be applied a second time inside
the frame. Frames are independently generated dense LDU systems rather than a
product of rank-one time transitions, and a generic strictly masked outer
product already has rank `r-1`. The admissible analogue is a rank-coordinate
blocked semiseparable packet algorithm. With coordinate tile `b=C=16`, shared
boundary panels, local generators, and all token right-hand sides form one
augmented panel product; only the `C` exact `16 x 16` diagonal-tile solves stay
serial inside a stage. This retains dense `J,D`, tokenwise frame updates, and
the fixed ambient associative state.

Decision: the complete per-token factor cotangent has masked outer-product
rank at most five for `K=1`: one primal-solve term, two erase/read terms, and
the rank-two skew cotangent. This supports a compact blocked VJP without
writing tokenwise dense cotangents. An independent multi-BMM realization was
rejected: ordinary FP32 failed the legal `2^12` cancellation probe, while a
two-sided split restored accuracy but took `2.207 ms` and added about
`336 MiB`, essentially consuming the entire `2.320 ms` baseline budget before
local, radial, or prefix reverse. Any compensated product must therefore live
inside the eventual fused blocked kernel.

Decision: linearity across those five descriptors is also the selected qbar
algorithm. The primal, two dual, and two skew masked outers are first summed in
fixed FloatFloat arithmetic for each strict entry and are then contracted once
against each dense boundary. The local semiseparable pass already sums all five
slots, so the transpose solve need not reproduce a second compensated qbar
contraction. This is an exact VJP reassociation, not a new chart or a temporal
WY approximation.

Decision: a projected radial reverse can recover each affine-prefix norm and
its scalar VJP from `<A_t,B>` and `<A_t,L_s>` projections and removes the
full action workspace. The fastest fixed-precision source reverse that passed
the driven `2^12` cancellation contract remained about 4--5% slower than the
current radial VJP, while the faster plain-FP32 form failed that contract. It
is therefore a useful derivation for a future fused algorithm but is not the
selected production path.

## Explicitly closed directions

The review above caused the repository to close, rather than maintain, the
following competing tracks: H-only preconditioning, scalar SFDD, standalone
single-edit SFDG, standalone DeltaProduct, DPLR/resolvent variants, co-decayed
or fixed-ridge geometries, the bidirectional accretive chart, pure SPD,
orthogonal-scale, butterfly, Woodbury, matrix-exponential/Cayley, residual
fixed-point and shallow coupling solve charts, piecewise-constant frames,
recursively approximated inverse states, and unpreconditioned Krylov solves.
Diagnostics may reproduce one of these ideas outside the model package, but
none is a public architecture or roadmap branch.
