# Prior Work and Decision Ledger

This file records primary sources that changed the single SolveDelta design. A
source is not a second project direction. Access and review date: 2026-08-25.

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
forward and backward, mixed dtypes, irregular lengths, initial/final states,
packed resets, normalization, and gate fusion. SolveDelta adopts one BF16/FP32
end-to-end envelope but compares against an FP64 oracle built from the same
quantized inputs. It adds stricter independent budgets for each Tensor Core
contraction, the Triton geometry scan, chart/radial construction, MathDx
residual/actions, and dual pairing. This prevents a locally inaccurate frame
from consuming the full recurrent-layer tolerance while remaining hidden by
output scale.

Decision: mature Delta implementations recompute chunk-local structure in
backward rather than retaining tokenwise recurrent states. SolveDelta follows
that schedule for geometry/frame state, but saves FLA's compact WY exterior
because a local target-profile benchmark found it about 28% faster for only
about 5.5 MiB extra peak memory. The two FLA issue reports changed validation,
not model math: cross-chunk lengths, multiple seeds and gate regimes, and
repeat-run gradient stability are mandatory checks for the optimized path.
Fixed-tree kernels remain bitwise checked; a selected FP32 atomic reduction
uses the fixed bounded-repeatability gate in `docs/VALIDATION_PLAN.md`.

Decision: the generalized DPLR exterior admits the finite equivalent mapping
`k=b=d` and `a=-e*exp(g)`, instead of `k=d`, `b=-d`, and `a=e*exp(g)`. The
installed FLA implementation and its backward were tested bitwise under both
mappings for `K=1,2`, including output and final-state cotangents. The selected
mapping aliases `b` with `k` and removes the redundant signed write-direction
allocation. The checked-in direct-`e` specialization adapts FLA 0.5.2's
MIT-licensed intra-chunk generalized-Delta staging: it generates the same BF16
`a=-e*exp(g)` bits at use and folds the backward into `e` and `g`, while FLA
continues to own the mature WY/state/output kernels. Output, final-state, input,
and state-cotangent tests validate the specialization. It removes
materialization, saving, and replay of `a`; it does not change the operator or
claim a new WY algorithm.

Decision: the direct-`e` generalized-DPLR composition above is a validated
historical checkpoint, not a frozen target ABI. The selected rewrite evaluates
both direct chunk-local SolveDelta-to-WY fusion and split mature kernels with
private `d/e/chi`, transformed panels, or upstream-native staging. Native action
panels remain `r x C`; any packed consumer dimension is an internal schedule
choice. Channel-wise decay interactions are
formed from the nonpositive log differences `exp(G_i-G_j)` and
`exp(G_C-G_i)`; the algebraically convenient inverse gauge `exp(-G_j)d_j` is
never materialized because it can overflow while the true pair interaction is
finite. The paired backward uses the C32 transpose solve and direct
interaction/frame/chart transpose actions rather than a chain of entrywise
VJPs. DeltaNet's WY derivation and FLA's GDN2/Delta implementation informed
this schedule; the exact forward, reverse, precision map, model-facing ABI, and
acceptance conditions are frozen in `docs/FROM_SCRATCH_REBUILD.md`.

Decision: a source-level audit of FLA `v0.5.2` identified smaller reusable
kernel blocks without reviving its model ABIs. The selected candidates are the
C32 16+16 unit-lower inverse and wide-RHS WY application from FLA's triangular
and generalized-Delta kernels; stable causal pair contractions and their fused
matrix reverse from KDA/GDN2/generalized-Delta; and MESA-Net's matrix-free
`((X B^T) \odot \Omega) A` chunk action, moment-tile update, and pair-score
reverse reductions. MESA's CG iteration, ridge/SPD semantics, and perturbation
constants are explicitly not candidates. `docs/UPSTREAM_REUSE.md` records the
function-level mapping, required specialization, transpose owner, provenance,
and adoption gates. This audit changes implementation priorities, not the
operator contract.

Decision update (2026-08-25): the audited FLA GDN2 path is selectively fused,
not a single-CTA pipeline. Its forward keeps gate/cumsum, intra-WY preparation,
recurrent state, and output as distinct program owners. Its backward fuses a
compatible WY/`dqkg` region but retains separate `dAv`, state reverse, intra
reverse, and cumsum reverse. SolveDelta therefore does not require removal of
an internal `d/e/chi` ABI or zero-HBM fusion. Direct frame-to-pair
generate-use-discard competes against `d/e/chi`, transformed panels, and
upstream-native private layouts; the decision is made from complete F+B latency
together with registers, shared memory, spills, barriers, active CTAs/SM, and
backward cache/recompute cost. Only model-visible outputs and continuation
states belong to the stable public contract.

Decision update (2026-08-25): `C=32`, four warps, stage count, register/shared
budgets, spill policy, and CTAs/SM are not operator contracts. The first native
`r=128,K=1` target evaluates numerically passing `C in {16,32,64}` candidates
and mature four-/eight-warp schedules, then freezes one offline winner for the
target architecture/profile from complete forward and F+B measurements. C32
timings elsewhere in this ledger remain historical evidence and a reference
profile, not a mandatory specialization. Runtime autotuning and
data-dependent schedule selection remain excluded.

Decision update (2026-08-25): FLA main was checked again at commit
`3e61322b615df248e7579222d1a68260560f7c24`. Its current MESA local action still
uses the same `tl.dot((tl.dot(P, K.T) * M), V)` schedule and the same
Gram/Hadamard transpose patterns; no newer matched forward/transpose block was
available to adopt. Local deterministic, atomic, and route-streaming
transpose schedules were therefore judged by complete SolveDelta F+B rather
than source resemblance. The tested atomic schedule's roughly `1.2e-7`
run-to-run drift was numerically acceptable, but that particular integration
did not improve complete F+B and was deleted. This is not a categorical ban on
FP32 atomic accumulation: a fused route/coordinate-block schedule remains
eligible if it passes the fixed repeatability and oracle/VJP gates and wins the
complete path.

Decision update (2026-08-25): the strict-diagonal transpose now follows the
masked block-product organization used by the audited KDA/GDN2 kernels. For
each target and 16-coordinate tile it first sums the three route descriptors
into the exact strict matrices
`A_lower=sum_a tril(l_a r_a^T,-1)` and
`A_upper=sum_a triu(l_a r_a^T,1)`, then applies `A`, `A^T`, and the companion
`h` action with fixed twofold BF16 Tensor Core products and FP32 accumulation.
This is algebraically identical to the removed coordinate-prefix reductions;
it does not form a tokenwise `r x r` matrix. On the target profile the isolated
transpose changed from about `1.865 ms` to `1.761 ms`. Alternating same-process
complete-path A/Bs changed F+B medians from `5.936/5.962 ms` to
`5.774/5.796 ms`. Splitting the 32 targets over four CTAs plus a deterministic
FP32 workspace regressed the isolated path to about `2.127 ms` and was
deleted. The retained owner is one CTA per panel/coordinate tile and remains
bitwise repeatable.

The same pass tested MESA/KDA's exact matrix-free off-diagonal action
`((X B^T) odot Omega) A` in the resident frame owner while preserving the
existing warp diagonal solve. The implementation used BF16 MMA with FP32
accumulation, materialized no score tensor in HBM, and passed the composed
forward, state, and VJP gates. It nevertheless required two resident `C x C`
score staging rounds for one primal and two transpose-dual actions. The best
version measured about `1.530/5.902 ms` forward/F+B versus roughly
`1.48/5.78 ms` for the retained scalar resident action. The shared-memory
packing and barriers cost more at `C=32` than the scalar work they removed, so
the implementation was deleted.

FLA's resident state/output reverse was then audited at the actual graph
boundary. SolveDelta's existing `chunk_state_backward` already keeps only the
FP32 `(r x d_v)` carry serial across chunks for each head/value tile; WY and
frame transpose remain chunk-panel parallel. A prototype fused the separate
WY pair reverse into the frame-transpose CTA and overwrote the dead
`grad_D_tail/grad_E_gamma/grad_Q_gamma` buffers in place, eliminating about
12 MiB of frame cotangent traffic without creating an all-chunk mega-kernel.
After fixing the required pre-overwrite tail-cotangent barrier it passed all
chunk-WY gates and compiled at 64 registers with no spills. Two 500-sample
target-profile comparisons still put backward-alone at `4.233/4.372 ms`
fused versus `4.125/4.195 ms` separated. Extending the high-shared-memory frame
CTA with pair work outweighed the saved launch and traffic; the fused ABI was
deleted and the mature separated panel-parallel reverse retained.

Decision: the first adopted block is
`causallsso/ops/paired_wy.py::_inverse_c32_blocks`, specialized from FLA
`v0.5.2`'s `merge_16x16_to_32x32_inverse_kernel`, together with the wide-RHS
application schedule from `wu_fwd_kernel` and matrix reverse pattern from
`prepare_wy_repr_bwd_kernel`. It recomputes the private 16+16 inverse from FP32
`W`, applies the native edit and value RHS
without concatenating or materializing the inverse, and fuses
`bar W=-bar B X^T` with the `write/value` pullback. The initial implementation
rejected direct BF16 inverse/RHS operands despite their passing end-to-end VJP:
their private C32 solve residual was about `7.6e-4` forward and `9.1e-5` in
transpose, above an inherited `eta <= 2e-5` gate. It used four-product twofold
BF16 solely to make that private diagnostic pass.

Decision revision (2026-08-25): the BF16-observable contract does not expose
the private inverse or FP32 pre-storage residual. Those quantities do not
justify a stricter production implementation when the stored solve action,
matrix reverse, output, state, and composed VJP pass their declared gates;
`eta <= 2e-5` remains an independent MathDx-oracle requirement. The FP32
diagnostic and twofold helper were deleted. Direct BF16 with one MMA per block
product measured `rho` about `2.3e-3` for stored forward actions,
`1.7e-3` for the transpose RHS action, and at most `3.1e-3` for the isolated
leaf/`W` reverse. A real `T=1024,H=8` SolveDelta-produced system with maximum
strict entry about `0.218` gave `2.40e-3/2.34e-3` forward-action error,
`1.68e-3` transpose-action error, and at most `2.82e-3` leaf-VJP error. At
`B=1,T=1024,H=8,r=d_v=128,C=32`, a matched current-tree A/B changed isolated
forward/reverse from `0.0416/0.0634 ms` to `0.0399/0.0613 ms`. The original
FLA-derived checkpoint measured about `0.023/0.040 ms`, versus about `0.272 ms`
for the displaced scalar reverse.
Two matched complete-operator rounds against commit `6d4e53f` reduced median
forward-plus-backward from `6.885--7.122 ms` to `6.443--6.497 ms`, about
`5.6--9.5%`; forward moved from `1.639--1.643 ms` to `1.608--1.620 ms`.
The old scalar forward solve, scalar transpose solve, `grad_Z` workspace, and
separate value-backward launch were deleted. At this checkpoint stable W/A
construction and the pair-to-frame gradient interface remained open; the later
frame-ownership change recorded below removed that interface.

Decision: the installed FLA 0.5.2 GDN2 and GatedDeltaProduct layers establish
three independent depthwise causal `conv4`, bias-free, SiLU branches over the
projected query, packed keys, and packed values. Convolution precedes head
reshape and query/key normalization; geometry and gate branches bypass it.
SolveDelta adopts exactly that fixed frontend with one structural enable switch
and carries all three minimal caches in its layer state. A local output/state/VJP
audit of FLA's Triton causal convolution found that its main output VJP is
correct, but a cotangent on the returned final cache omits part of
`d(final_state)/dx`. The later `causal-conv1d` 1.7.0 integration recorded below
supersedes that wrapper: its CUDA kernel owns the exact three-input conv4
continuation cache and complete final-state VJP.

## Mixed precision and Tensor Core arithmetic

- Kalamkar et al., *A Study of BFLOAT16 for Deep Learning Training*,
  [arXiv:1905.12322](https://arxiv.org/abs/1905.12322).
- Henry, Tang, and Heinecke, *Leveraging the bfloat16 Artificial Intelligence
  Datatype For Higher-Precision Computations*,
  [arXiv:1904.06376](https://arxiv.org/abs/1904.06376).
- Ootomo and Yokota, *Recovering Single Precision Accuracy from Tensor Cores
  While Surpassing the FP32 Theoretical Peak Performance*,
  [DOI:10.1177/10943420221090256](https://doi.org/10.1177/10943420221090256).
- Ogita, Rump, and Oishi, *Accurate Sum and Dot Product*,
  [DOI:10.1137/030601818](https://doi.org/10.1137/030601818).
- FLA 0.5.2
  [GDN2 layer](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/layers/gdn2.py)
  and
  [chunk operator](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gdn2/chunk.py).

Decision: BF16 is SolveDelta's public/raw activation dtype, not the mandatory
storage dtype for every private Tensor Core panel. Tensor Core products
accumulate in FP32; normalization, radial norms, log-decay and gate
nonlinearities, backward partials, sensitive scalars, and continuation states
`m,J,D,S` also remain FP32. An analytically bounded FP32 producer may write a
named private panel directly as FP16, and its matched reverse consumes the same
bits with FP32 accumulation. Public activation results and leaf gradients
return to BF16 only after their FP32 reductions. Rounding state at each chunk
is rejected because it makes the numerical recurrence depend on the chosen
split.

Decision: the installed GDN2 layer leaves `q,k,v,b,w` in model dtype, evaluates
its log-decay in FP32, requires an FP32 initial recurrent state, uses low-
precision `tl.dot` operands with FP32 accumulators, accumulates major backward
partials in FP32, and casts returned activation gradients to their input dtype.
Its public example uses BF16. SolveDelta adopts those public and FP32-state
boundaries. It does not copy GDN2's private operand dtype blindly: bounded
private FP16 panels are permitted by the separate MESA precedent below. A
local GDN2 BF16 audit against
the same-quantized-input FP64 recurrence measured about `3.19e-3` output,
`2.23e-3` state, and less than `4.5e-3` gradient error.

Decision: low-precision multiplication does not justify deleting deep
cancellation tests. Two BF16 significands can produce a product with a residual
well below one BF16 input ulp, and that product is exactly representable in an
FP32 accumulator over the normal range. Cancellation fixtures are therefore
quantized first and classified by their recomputed condition ratio. Fixed
twofold/high-low contractions may be used where an ordinary product fails, as
supported by the accurate-dot and Tensor Core decomposition literature, but
they must be fused, deterministic, and measured. They do not permit iterative
correction semantics, BF16 state storage, or a data-dependent fallback.

- MESA-Net FLA 0.5.2
  [`chunk.py`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/mesa_net/chunk.py).

Decision: MESA's in-kernel normalization calls `l2norm_fwd(...,
output_dtype=torch.float16)` for BF16 model inputs. The FP32 normalization
producer therefore writes bounded FP16 `q/k` panels directly, gaining three
significand bits while remaining safe because every normalized component has
magnitude at most one. Its alternate plain `q.to(float16)` path provides no
such gain: it is exact only over the formats' shared normal exponent range,
cannot restore discarded BF16 bits, and can lose values outside FP16's range.
This informed SolveDelta's
single `BF16 public/raw -> FP32 producer -> bounded FP16 private -> FP32
accumulator` contract. It does not authorize FP16 `h`, values, recurrent state,
or solve panels without an analytic magnitude certificate, and it introduces
no runtime precision branch.

Decision: a fixed high/low Tensor Core representation of an FP32 boundary is
one continuous state, not two independently differentiated parameters. Its
VJP returns the mathematical boundary adjoint once; the RHS adjoint consumes
the exact high and low bits used by the forward. Tensor Core packing occurs
inside the contraction primitive so an FP32 source receives an FP32 partial,
while a public BF16 activation leaf receives BF16 only after reduction.

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
asymmetric edits share a full-prefix LSSO transpose-dual adapter and use gated
key-derived erase covectors that become generally non-collinear with writes in
the ambient frame. `K=1` is the current performance default; `K>1` remains a
supported capacity setting and the exact DeltaProduct-`K` reduction remains a
required test. This changes the default compute point rather than the model
family's expressivity, and edit count is not part of the novelty claim.

Decision: GDN2 already uses asymmetric rank-one erase/write factors, so mild
non-normality is not unique to SolveDelta. A fresh algebraic audit corrected a
stronger claim: `a^T b >= 0` fixes the one nontrivial eigenvalue but does not
make the symmetric part of `ab^T` positive semidefinite unless the factors are
positively collinear. The former H+S/dissipative label was removed. A later
orthogonal erase-residual candidate was also deleted: no task evidence showed
that its extra skew coordinate mattered, while it required an additional chart
action in forward and a rank-two chart cotangent in backward. The canonical
erase source is now exactly the gated normalized key. This preserves the full
bounded LDU geometry and exact GDN2 reduction while making the frame contract
strictly smaller. Specifically, at fixed `(P,a)`, finite sigmoid erase gates
satisfy `a_i bar_b_i = beta_i a_i^2 >= 0`,
`supp(bar_b) = supp(a)`, and
`0 < |bar_b_i| < 2 |a_i|` on that support. The FP64 oracle also admits the
closed gate endpoints, whose zero endpoint gives the weaker
`supp(bar_b) subseteq supp(a)` condition. The deleted residual could move
`bar_b` along the chart-dependent orthogonal direction `Omega a` and leave
that cone. A different shared prefix frame
cannot in general recover this edit-specific degree because the same `P` also
acts on every write, erase, and read vector. The loss is accepted provisionally
because no task ablation supported it and it materially enlarged both frame
actions and their VJP; task evidence remains the final check, not the speed
result alone.

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

Decision: strict masking destroys ordinary low rank, but retains a
sequentially semiseparable generator. For one geometry chunk, every token's
strict factor is a dense boundary triangle plus masked outer products from at
most `C` local sources. This supports exact `O(C r^2 + C^2 r)` lower, upper,
and transpose actions without materializing `C x r x r` factors.

The deleted C16 packet and C32 panel prototypes established useful facts but
are not maintained implementations. In particular, coordinate blocking can
turn cross-block generator contractions into wide-RHS matrix products; only
the diagonal coordinate block needs a strict prefix/suffix scan. These are
algebraic inputs to the replacement chunk operator, not compatibility
requirements for its ABI.

The generic quasiseparable order of a dense continuation can still grow to
`floor(r/2)`. The generator algorithm is therefore not evidence that the full
chart has constant ordinary rank or permission to compress `J` or `D`. The
selected resident path preserves four independent radial channels. A plain
IEEE-FP32 Gram expansion may reconstruct a small signed quadratic under legal
boundary/local cancellation, so deep-cancellation acceptance gates the fixed
BF16-observable chart coordinate `A=aZ`, its action, and composed VJP rather
than the private expanded `q2` or scale. Each named FP32-state
contraction starts with one statically frozen direct-BF16 packing; high/low is
permitted only as a static promotion of the particular contraction that fails
those common gates. The `2^12` inputs remain mandatory adversarial fixtures,
not warning-only diagnostics or private bitwise-residual requirements.

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
directly callable from an ordinary Triton JIT kernel. The engineering boundary
keeps the geometry boundary scan in Triton, MathDx as an optional exact oracle,
FLA as the generalized Delta/WY owner, and one project-owned CUDA operator for
chunk-local frame forward and VJP. The local first target is
PyTorch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0 Update 2, and MathDx 26.06
cuBLASDx 0.7.0. An isolated official block-TRSM example has compiled, linked
through `libcublasdx.fatbin`, dispatched on SM120, and returned zero reported
L2 error; this validates the exact toolchain boundary rather than the
performance of the fused training operator. The chunk boundary avoids
materializing `T x r x r` prefix states while retaining the established outer
schedule. MathDx remains the exact comparison boundary and is not linked into
the production training target.

The backend screen also considered handwritten recursive/block TRSM in Triton,
CuTe DSL, TileLang, or CUDA tile libraries; host-level batched cuBLAS;
truncated Neumann/polynomial inverses; and replacing LDU with butterfly,
coupling, or hierarchical nilpotent-shear factors. Host batched TRSM forces
dynamic factors through global memory, and changing factor families would make
backend convenience alter the model mathematics. The earlier standalone
polynomial path materialized its own panels and defined a competing solve
surface; deleting it was correct.

Rejected decision (2026-08-25): a resident fixed-order Neumann lowering was
briefly selected for the first native training path. With `||N||_2 < q=1/4`,
degree `p=6` had the static single-factor truncation bound

\[
\delta_6=\frac{q^7}{1-q}=\frac1{12288}\simeq8.138\times10^{-5}.
\]

The approximation was subsequently rejected as production semantics. The
current implementation uses exact blocked coordinate-axis generalized-Delta
substitution and its exact transpose: complete coordinate blocks use pair-dot
products and each diagonal block retains ordered substitution. MESA's inline
`chunk_update_once` and resident CG loop still inform the action schedule, but
its CG recurrence, adaptive iteration count, denominator perturbation, ridge
system, and `q_star` ABI are not adopted. MathDx continues to own independent
exact factor-action validation.

An isolated SM120 chart VJP prototype matched FP64 closely but materialized
full factor cotangents and increased end-to-end time and workspace. It was
deleted. The replacement backward must consume compact action cotangents inside
the chart reduction without writing tokenwise `grad_lower/grad_upper`.

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

Decision: ordinary temporal compact WY cannot simply be applied a second time
inside the frame. Frames are independently generated dense LDU systems rather
than a product of rank-one temporal transitions, and a generic strictly masked
outer product already has rank `r-1`. The admissible analogue is blocked
semiseparable action over the finite chunk: shared boundary blocks and local
generators form broad RHS products, while exact diagonal-block substitution
retains tokenwise frame updates and dense `J,D`.

Decision: after deleting the unsupported skew residual, the complete per-token
factor cotangent has masked outer-product rank at most three for `K=1`: one
primal-action term and one term for each of the erase and read dual actions.
This supports a compact blocked VJP without writing tokenwise dense
cotangents. An independent multi-BMM realization was rejected: ordinary FP32
failed the legal `2^12` cancellation probe, while a two-sided split restored
accuracy but took `2.207 ms` and added about `336 MiB`, essentially consuming
the entire `2.320 ms` baseline budget before local, radial, or prefix reverse.
Any compensated product must therefore live inside the eventual fused blocked
kernel.

Decision: linearity across those descriptors permits one exact qbar
reassociation. The primal and dual masked outers can be summed before
contraction with each dense boundary, while the local semiseparable pass
consumes the same descriptor bundle. A first per-product high/low compensated
CUDA realization restored the `2^12` cancellation gradient but raised the
target-profile frame forward-plus-backward time to roughly `194 ms`; it was
therefore rejected. Precision compensation was not carried into the rewrite.

Decision: the selected reverse no longer preserves the earlier packet/panel or
all-FP32 replay schedules. The resident action backward applies the transpose
of the forward primal/dual block action, then contracts the rank-three
primal/erase/read descriptor bundle with local generators and dense boundaries.
BF16 descriptor and factor operands feed FP32-accumulated products; FP32
boundary products use one fixed high/low packing. Compact pair, coefficient,
and leaf primitives return FP32 partials to the composed geometry/frame
autograd owner, which joins them with the Triton affine scan adjoint before
casting BF16 activation leaves. This keeps dense `J,D`, four radial channels,
and exact LDU actions without tokenwise dense factor cotangents.

A multi-panel audit remains part of the selected path's provenance. It found a
shared-memory write-after-read race in an earlier radial reduction and an
address-mask omission in the Triton scalar adjoint's invalid tail stores.
Separate reduction/broadcast slots and an explicit valid store mask restored
bitwise repeatability for multiple batches, heads, chunks, and irregular tails;
the regressions remain mandatory even though the responsible old kernels are
gone.

Decision: a projected radial reverse can recover each affine-prefix norm and
its scalar VJP from `<A_t,B>` and `<A_t,L_s>` projections without a full action
workspace. For
`Z_t=beta_t Z_{t-1}+r_tL_t`, the exact recurrence is
`n_t=beta_t^2 n_{t-1}+2 beta_t r_t <Z_{t-1},L_t>+r_t^2<L_t,L_t>`.
Its reverse closes over the same projections and descriptor bundle. Dense
boundary work remains `O(C r^2)` and cannot be removed without restricting the
chart, but local work can use the finite `O(C^2 r)` chunk algebra.

Decision: the MESA Gram/Hadamard identity was evaluated as a replacement for
the realized radial residual. A target-profile prototype
formed `||theta B+L||^2` from boundary norm, boundary-generator pairs, and the
local generator Gram. The uncompensated form took about `0.325 ms` at P256 and
matched ordinary FP64 fixtures to roughly `1e-6`, but rounded the legal
`2^12` cancellation norm from `1.4901161193847656e-8` to zero and emitted
signed `~1e-6` noise for an exact-zero residual. A deterministic FP32
`TwoSum`/FMA-`TwoProduct` schedule, following Ogita, Rump, and Oishi's
[accurate sum and dot-product construction](https://doi.org/10.1137/030601818),
restored the `2^12` result exactly and reduced ordinary error to roughly
`1e-7`. Carrying compensation through every pair statistic still left
`[-2.4e-8, 7.7e-8]` exact-zero noise because Tensor Core dot-product
accumulation had already discarded the required low part, while latency rose
to about `1.527 ms`. Recovering that part requires a scalar compensated dot or
an equivalent exact accumulator and loses the intended Tensor Core advantage.
The later BF16-observable contract recognized that bitwise recovery of this
private quadratic is stricter than the deployed action: an identity-centered
BF16 diagonal cannot expose the requested low bits. The uncompensated direct
pair schedule is therefore the selected forward candidate, subject to the
fixed chart-coordinate, action, and reachable exact-transpose VJP gates. The
compensated variant remains rejected. Do not add a magnitude threshold, clamp, or runtime
precision branch to make the expanded quadratic appear nonnegative or exactly
zero.

Decision: the deep-cancellation audit exposed an ambiguity in validating a
shared `geometry_strength` scalar, not a missing VJP term. The parameter ties
six chart-channel contributions by the fixed linear map `g=1^T g_6`. One `J`
witness has expected/actual `1.814816e-3/1.892893e-3`, and one `D` witness has
`-6.437174e-5/-4.876703e-4`; the latter's L1-to-total ratio is about `4350`.
Accumulating only the final scalar addition in FP64 does not cure the error.
For these declared deep-cancellation fixtures, validation now measures the
tying map at its induced scale:
`rho_tie=|g_hat-1^T g_6|/(sqrt(6)||g_6||_2+1e-8) <= 2.5e-2`, with
the existing `abs <= 1e-6` alternative retained when `||g_6||_2` itself is
near zero. Ordinary fixtures still use the standard total-gradient metric.
Unlike multiplying tolerance by the observed cancellation ratio, this ceiling
is fixed by the six-to-one map's operator norm and cannot vary with the data.

On the local SM120 profile
`B=1,T=1024,H=8,r=d_v=128,K=1,C=32`, the current closed
scan--prepare--state path measures about `1.91/6.86 ms`
forward/forward-plus-backward versus `0.32/1.10 ms` for the matched GDN2
operator. Both rows exclude projections and conv4. The chunk-owned prepare
contains frame action, stable pair statistics, and the C32 WY solve; `d/e/chi`
are generated and consumed inside that ownership boundary and returned only as
a private FP16 backward cache. They are not a public frame-to-WY ABI.

The selected frame checkpoint is a paired normalized-moment schedule: each
BF16 `16x16` factor tile is generated once, then serves the primal gather solve
and the two-route dual scatter action. Its target-profile mixed frame kernels
measure about `1.01 ms` forward and `0.70 ms` adjoint, with no register spill.
This replaced four independent factor traversals and reduced the complete path
from roughly `2.25/7.7 ms` to `1.91/6.86 ms`. The reverse now also accumulates
strict, radial, and diagonal geometry cotangents into one FP32 vector panel and
one FP32 `J/D` boundary pair rather than returning separate VJP panels.

Two exact broad/local semiseparable A/Bs were then rejected at the same shape.
A direct-BF16 CuTe version with separate primal and dual traversals passed the
same full-layer FP64/VJP tests but measured about `2.25/7.68 ms`, with mixed
frame forward/adjoint near `1.08/1.02 ms`. A genuinely paired version shared
the diagonal factor and used boundary `32/64`-RHS MMA plus cumulative `C x C`
local correlations. It also passed the full-layer oracle and VJP, but its
dual-suffix precompute and `50.8 KiB` shared footprint reduced occupancy to one
CTA per SM; the result regressed to about `2.90/9.60 ms`, with mixed frame
forward/adjoint near `1.92/2.11 ms`. Both implementations were deleted rather
than retained as alternate backends.

This is rejection evidence for those schedules, not for the Section 6
algebra. The current paired scalar checkpoint remains faster at `r=128,C=32`,
but it does not complete the intended boundary-GEMM plus `O(C^2r)` production
rewrite. Closing that contract now requires a schedule that avoids both the
four-traversal duplication and the resident `C x 2C` dual-suffix correlation
state; merely translating the existing formulas into more MMA instructions is
not an accepted next step.

A later MESA-style transpose screen removed the resident suffix state in three
ways. A deterministic coordinate-block schedule used about `9 MiB` of reduced
scalar partials but took `2.820 ms`; an atomic coordinate-block form took
`2.032 ms` with repeat drift below roughly `rho=1.2e-7`; and a route-streaming
form restored the intended correlation count and took `1.981 ms` isolated.
None improved complete F+B against the current path in its controlled A/B, so
all three were deleted. This local result reinforces the ownership rule:
traffic reduction is adopted only when the composed training path wins.

That descriptor checkpoint was subsequently deleted rather than retained as a
BF16 ABI. The production reverse now asks the WY pair transpose for one FP32
primal and one two-route FP32 dual action panel, performs the upper frame
transpose, and immediately consumes the three rank-one descriptors in a
coordinate-block Triton `tl.dot` transpose. The diagonal/lower frame stage then
overwrites the same action panels and immediately runs the lower transpose.
There is no `descriptor_bundle`, `correlation`, `pair_partial`,
`projection_partial`, or `grad_d/grad_e/grad_chi` interface. The strict
transpose writes final `grad_u`, `grad_h`, `grad_weights`, `grad_theta`, and
`grad_radial_scale`; the native upper/lower stages accumulate into one shared
FP32 `grad_boundary_J/grad_boundary_D` pair.

The action panels remain short-lived global producer/consumer panels between
the native triangular action and its Triton transpose; they are not chart
descriptors or saved backward checkpoints. At most one action stage is live:
the old `upper_primal.clone()` and simultaneous four-panel return were removed.
This is the current honest fusion boundary. Moving the triangular action and
Tensor Core strict transpose into one backend remains a possible later
schedule, but it is not required to preserve the operator or to claim deletion
of the descriptor ABI.

On the SM120 target profile, changing the streamed strict kernels from eight to
four warps and splitting the diagonal transpose by action stage passed the
composed FP64/VJP gate. The warmed medians are about `1.634 ms` forward and
`6.485 ms` forward plus backward, or `4.851 ms` backward by subtraction. The
incremental allocator peaks are about `95.2 MiB` forward and `241.8 MiB` for
forward plus backward. Across 100 identical target-profile forward/VJP runs,
the maximum observed drift was `rho=1.81e-7` and `a_inf=1.79e-7`.

## MESA execution-graph and exterior-kernel audit

A warmed same-device CUDA profile at
`B=1,T=1024,H=8,r=d_v=128,C=32` was used to answer why adopting MESA's local
Gram/two-dot identities did not make the complete operators equally fast.
The sums below are attributed CUDA kernel time, not wall-clock medians; both
operators exclude projections and conv4 in this comparison.

| Profile | SolveDelta | FLA MESA |
|---|---:|---:|
| forward + backward CUDA activity | about `5.904 ms` | about `1.151 ms` |
| main resident solve work | prepare forward `0.934 ms`; prepare backward `0.807 ms` | two 30-step CG kernels `0.737 ms` total |
| outer state/output | state reverse `0.336 ms`; output forward `0.132 ms` | paired state forward kernels `0.044 ms` total |
| geometry-only work | strict diagonal/cross `1.018 ms`; geometry scan adjoint `0.343 ms`; radial transpose `0.254 ms`; strict correlations `0.238 ms` | no bounded-LDU chart or separate `J/D` radial/strict VJP |

This is not a contradiction in the shared local-action formula. MESA loads a
`BT x K` tile and its `K x K` states once, then executes all 30 fixed CG
iterations inside one resident Triton program. Its production chunk path uses
FP16 normalized query/key panels, stores ordinary chunk states in the input
dtype (`states_in_fp32=False`), and differentiates the approximate solve with
one implicit adjoint solve. With its default `C=64`, it also has half as many
chunk boundaries as SolveDelta C32.

SolveDelta has a larger exact computation graph around the shared two-dot
motif. It retains FP32 continuation states `(m,J,D,S)`, maps separate `J` and
`D` through four bounded strict/radial chart routes, applies one primal and two
transpose-dual actions, constructs and reverses the C32 WY system, and returns
the exact VJP of that declared recurrence. The current implementation splits
these stages across many CUDA/Triton programs and still performs some small
matrix products as scalar selection/reduction loops. Therefore MESA's
matrix-free action only accelerates one algebraic subexpression; it neither
removes SolveDelta's additional geometry nor gives the surrounding kernels
MESA's residency, precision, or launch schedule.

The audit found a concrete non-core bottleneck. At that checkpoint the output
kernel computes `A_qd @ residual` through 32 static `where/sum` selections and
used about `131.5 us`; FLA's Tensor-Core `chunk_gla_fwd_kernel_o` probe used
about `7.2 us`. Current state forward/reverse used about `59.2/335.9 us`, while
FLA common state forward/transpose probes used about `50.2/75.9 us`. Paired WY
itself used only about `21/36 us` forward/reverse. The next engineering target
was consequently the state/output exterior and its transpose, not another WY
micro-optimization.

FLA main was inspected at
[`3e61322b`](https://github.com/fla-org/flash-linear-attention/commit/3e61322b615df248e7579222d1a68260560f7c24).
Its MESA action remains the same resident
`dot((dot(P,K.T)*M),V)` implementation, so it supplies no newer matched strict
transpose. It does, however, now contain a more directly useful DPLR
[`fused H/O forward`](https://github.com/fla-org/flash-linear-attention/blob/3e61322b615df248e7579222d1a68260560f7c24/fla/ops/generalized_delta_rule/dplr/backends/tilelang/chunk_ho_fwd.py)
and
[`streaming reverse`](https://github.com/fla-org/flash-linear-attention/blob/3e61322b615df248e7579222d1a68260560f7c24/fla/ops/generalized_delta_rule/dplr/backends/tilelang/chunk_stream_bwd.py),
including an explicit `SM120,K=V=128,BT=32` schedule. After statically deleting
one unused DPLR route, the exact mapping is

\[
w=-Y,\quad u=U_z,\quad b_g=D_{\rm tail},\quad q_g=Q_\gamma,
\quad A_{qb}=A_{qd},
\]

which gives `R=U_z-YS`, `S'=Lambda S+D_tail^T R`, and
`O=Q_gamma S+A_qd R`. This is a code-block reuse candidate, not permission to
restore the generalized-DPLR ABI. A simpler specialized Triton rewrite remains
the first A/B; the fused TileLang schedule is the next candidate because it
can keep residual and reverse carry on chip.

The broader audit also reviewed Mamba at
[`e9594ce1`](https://github.com/state-spaces/mamba/commit/e9594ce1c732d97440f0332fdc43170a2294dbfa),
`causal-conv1d` at
[`cd81f041`](https://github.com/Dao-AILab/causal-conv1d/commit/cd81f0413cad2fc1e6f17e785ac39f59aae690cd),
and CUTLASS at
[`7107b055`](https://github.com/NVIDIA/cutlass/commit/7107b05535f8977f5ecb9d01ee203205b1fd9bc4).
The decisions are, in order: adopt original-stride addressing before retaining
panel copies; A/B Mamba's deterministic-workspace versus FP32-atomic reduction
ownership under the existing fixed gates; replace the manual conv final-cache
slice only with the latest CUDA convolution's now-complete state VJP; and keep
CUTLASS back-to-back/grouped examples as lower-priority scheduling references.
The function-level inventory and adoption gates are in
`docs/UPSTREAM_REUSE.md`.

## Exterior and frontend reuse decisions

The prioritized reuse pass then implemented and measured each item at
`B=1,T=1024,H=8,r=d_v=128,C=32`. FLA's output `tl.dot` schedule replaced the
32 static `A_qd` selections in both forward and exact transpose, moving the
complete operator from about `1.753/6.943 ms` to `1.669/6.402 ms`. In contrast,
splitting the FP32 state carry into FLA-style 64+64 blocks was rejected: one
128-rank reverse block took about `47.7 us`, while two 64-rank blocks took
about `110.0 us` and worsened complete F+B.

Mamba-style original-stride addressing was adopted next. Radial and strict
owners now derive batch, head, chunk, token, and tail masks from the original
`[B,T,H,r]` tensors; `_panelize`, `valid_count`, expanded strength, and the
production `cat/permute/contiguous` round trip were deleted. Target timing
moved from about `1.665/6.354 ms` to `1.517--1.535/5.969--6.071 ms`, while
`aten::copy_` fell from 16 launches and about `52.7 us` to five launches and
about `16.4 us`. MESA's paired `Hkk/Hkv` tile schedule then changed the J/D
summary and transpose from 32-rank to four-warp 64-rank tiles. Their isolated
times changed from about `28.3/126.1 us` to `23.2/44.4 us`; an eight-warp form
was slower and was deleted.

FLA's L2, beta-sigmoid, and GDN gate sources informed two specialized native
owners. Three BF16 `u/q/key` inputs are normalized in one forward and one
strict-transpose launch, with FP32 norm reductions and direct FP16 panel
stores. The gate owner fuses erase/write sigmoid production and specializes
scalar/vector decay forward and fixed-tree VJP. Direct use of FLA's general
gate wrappers regressed gate F+B from about `0.418 ms` to `0.583 ms` despite a
faster forward and was deleted; the native specialization measures about
`0.055/0.147 ms` versus `0.154/0.393 ms` for the original PyTorch graph. The
existing vector cumsum was retained because its dedicated forward/reverse are
already only about `7.2/9.5 us`; crossing the core ABI to fuse it was not
justified by that ceiling.

The audited `causal-conv1d` commit was built for SM120 and now owns conv4 SiLU,
initial state, final state, `dfinal_state`, and `dinitial_state`. The old FLA
Triton output plus manual `cat(...)[...,-4:]` cache graph was deleted. A width-4
causal convolution needs only the previous three raw inputs, so layer caches
are now the minimal `[B,D,3]` state accepted by upstream rather than carrying
one coordinate that is discarded before every future action. This changes no
convolution output or model expression. At `B=1,T=1024,D=1024`, a matched BF16
branch changed from about `0.012/0.285 ms` forward/F+B to `0.008/0.118 ms`, with
bitwise equal outputs and minimal final state.

Mamba's atomic/workspace policy was screened twice. The earlier strict
coordinate-block atomic had repeat drift near `rho=1.2e-7` but no complete
F+B win. A later decay-parameter atomic had at most `1.5e-7` drift and measured
about `0.149 ms`, statistically no better than the fixed tree's `0.147 ms`.
Both were deleted; production remains deterministic. CUTLASS 3.x supplies
SM120 grouped collectives, but its register-resident grouped back-to-back
example remains SM80 and schedules independent GEMMs. It cannot directly own
SolveDelta's factor recurrence, diagonal solve, and tile-dependent action.
The resident action already uses `42.6 KiB` shared with two-CTA launch bounds;
the prior `50.8 KiB` CuTe schedule fell to one CTA and regressed. A separate
grouped launch would materialize factors and pointer metadata, so no CUTLASS
backend was added.

After these retained changes, the target operator measures about
`1.35/5.80 ms` forward/F+B. The complete target layer including projections,
three conv4 branches, gates, and all returned-state VJPs measures about
`1.78/7.25 ms`. These frontend wins do not remove the remaining bounded-chart
and exact-transpose cost identified in the MESA comparison above.

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

## From-scratch execution audit

Decision update (2026-08-25): the current native implementation and all older
execution documents are no longer accepted as design constraints. Their
timings remain historical evidence, but their ABIs, kernel boundaries,
workspaces, and compatibility requirements are presumed wrong. The FP64 token
recurrence in `causallsso/reference.py` is the sole mathematical authority.
Chunked geometry plus a generalized WY exterior survives only because its
forward, final-state, and complete VJP identities were independently derived
and checked in FP64; it is not retained by provenance or compatibility.

A source audit was repeated against FLA commit `bc3b101d`, its MESA kernels,
Mamba commit `e9594ce1`, causal-conv1d commit `cd81f041`, CUTLASS commit
`7107b055`, and installed MathDx 26.06. The resulting ownership rule is to
specialize complete mature kernels at their natural global-memory boundary.
Mixing a CUTLASS collective or block-level cuBLASDx operation into a custom
frame CTA was rejected because the two owners duplicate shared staging,
barriers, and register lifetimes. CuTe copy/MMA atoms and CUB reductions are
safe device-level reuse because they are instruction mappings rather than
independent collective schedules.

The selected exterior retains private direct-FP16 `d/e/chi` panels. They cost
6 MiB at the target profile and preserve a clean boundary to FLA's complete
pair/WY/state/output forward and transpose programs. This supersedes the
earlier proposal to eliminate those panels by fusing chart and WY into one
low-occupancy CTA. Synthetic `3C` panels, descriptor bundles, entrywise VJP
workspaces, and fake DPLR gate ABIs remain forbidden. The only custom kernel
family is the bounded-LDU frame/chart orchestration; even there, copies, MMA,
reductions, and scans are sourced from CuTe, MESA, and CUB. The complete
formulas, symbol-level reuse map, SM120 microprogram, precision bounds,
forward/backward launch graph, and falsification gates are frozen in
`docs/FROM_SCRATCH_REBUILD.md`.

The same audit removed J/D state propagation from the custom-code budget.
MESA's resident `Hkk/Hkv` forward is exactly the paired `(J,D)` recurrence under
`k=u`, `k2=u`, `v=h`, and `beta=1`; FLA common state reverse supplies the
resident transpose loop. The adopted specialization keeps MESA's complete
load/dot/state/store schedule, uses FP32 state storage, gives the J and D dots
their separately declared FP16/BF16 operand bits, and replaces the CG-specific
reverse source with SolveDelta's direct chart cotangent. The scalar `m` scan
remains separate. This is specialization of mature programs, not a new J/D
kernel family.

FLA DPLR's centered `chunk_A` forward/backward programs informed the frozen
tile-local WY gauge interface: pair operands may be centered around an
arbitrary coordinate-wise reference, while the formal reference cotangent
cancels exactly between row and column operands. MESA's `Hkk/Hkv` Gram and
Hadamard schedules informed the radial factorization into complete coordinate
tiles and strict diagonal tiles. The SolveDelta gauge reverse and the H/R
Gram transposes were derived independently from the operator and checked
against FP64 automatic differentiation; no DPLR or MESA high-level model ABI
is inherited.

The same canonicalization applies to chart scalars. FLA
`modules/l2norm.py` computes `rsqrt(sum(x*x)+eps)` and its exact projection
transpose, so SolveDelta's radial map is that primitive with `eps=c^2` and an
output scale `c`. FLA fused cross entropy already uses the identical
`s*tanh(x/s)` soft-cap and `1-tanh(x/s)^2` reverse needed by the diagonal
chart. The rebuild therefore reuses those scalar programs while MESA supplies
matrix-free route norms; it never materializes a strict `r x r` chart merely
to satisfy the public L2Norm wrapper. SolveDelta owns the composition and
route timing, not new normalization or activation mathematics.

Normalizing the two geometry accumulators by mass exposes another exact
canonical form. On reachable states, `H=J/m` and `R=D/m` obey matrix-valued
delta updates with step `1/m`: H tracks a decayed online `u*u^T` observation
and R tracks `u*h^T`. A chunk merge is the corresponding weighted-mean
residual merge. This improves interpretation and confirms the generic state
primitive, but it does not justify a normalized continuation ABI: H/R without
mass is not associative, and recurrent division/rounding would change the
chunk-split numerical recurrence. The production scan therefore remains the
paired MESA `(J,D)` accumulator. FLA common `chunk_h` independently confirms
each route as `H <- decay*H + K^T*V`; the paired MESA program is selected over
two generic launches because J and D share decay and the left `u` operand.

The bounded LDU factors also canonicalize to standard primitives rather than
disappearing. `L=I+N^-` and `U=I+N^+` are residual matrix actions; their direct
transpose-dual path maps to MESA's two-dot action, while the primal path is a
unit-triangular solve. FLA `solve_tril` and the fused GDN KKT/solve kernel
provide the substitution code pattern, but not a drop-in high-level operator:
they solve a token-tile triangular factor, whereas SolveDelta has one
coordinate-triangular factor per token with nonlinear H/R route coefficients.
The reusable unit is therefore the mature action/substitution program, not the
MESA CG wrapper or the complete GDN ABI.

Decision update (2026-08-25): the paired geometry state specialization is now
connected at its natural memory boundary. It copies MESA's resident
`chunk_mesa_net_fwd_kernel_h` loop, removes only the model-specific beta/CG
interface, separates the direct FP16 J and BF16 D operand pointers, and writes
the frame consumer's `[B,H,N,r,r]` order directly. The mass owner writes the
matching cumulative and tail panels in the same order. Reverse loads the
symmetric J representative directly, retains the resident affine state loop,
uses MESA's Hkk/Hkv tiled transpose-dot pattern for `bar u/bar h`, and uses
FLA's `tl.cumsum(..., reverse=True)` schedule for the scalar affine transpose.
Two broad FP32 products remain separate because log-decay inner products are
sensitive scalars under the precision contract. At
`B=1,T=1024,H=8,r=d_v=128,K=1,C=32`, this interface specialization changed
the current complete operator from about `4.86/19.73 ms` forward/F+B to about
`4.67/18.01 ms`; it also removed the geometry permute/contiguous boundary ABI.
The same connection audit found that passing PyTorch `diagonal()` and
zero-stride `expand()` views into a contiguous-only Triton ABI made chart
values depend on storage offset. The adopted interface passes the source
strides explicitly and keeps the resident J diagonal tile in its stored
symmetric representative after every chunk. No diagonal or strength panel is
materialized merely to repair pointer arithmetic. A K=2 whole/recurrent-split
check is consequently bitwise equal for output and all continuation states.

Decision correction (2026-08-25): a same-device source and latency audit
distinguished literal upstream schedules from implementations that only reuse
an algebraic identity. The direct-e pair producer retains FLA's sub-intra
schedule and measured about `0.065 ms` for its two physical interaction
matrices, versus `0.108 ms` for the generic upstream four-matrix producer.
The fused normalization fan-out measured about `0.086/0.438 ms` forward/F+B,
versus `0.117/0.684 ms` for three independent upstream L2Norm calls. The paired
geometry resident loop appeared to measure about `0.066 ms`, versus `0.080 ms`
for the upstream MESA C32 FP32-state configuration, in that single profiling
pass. The later seven-round audit below supersedes the geometry latency claim:
it measured the state kernels themselves at `55.2/45.4 us`, current/upstream.
Direct-`e` and normalization remain genuine mature specializations; geometry
retains the upstream loop shape but has not retained its full efficiency.

The bounded frame is not. At
`B=1,T=1024,H=8,K=1,r=d_v=128,C=32`, a warmed profiler attributed about
`3.492 ms` forward and `9.673 ms` backward to `resident_frame.py`. Its eight
local representation transpose launches alone used about `5.351 ms`. The
complete upstream MESA 30-step F+B measured `1.524 ms` on the same device and
shape. This comparison does not assert algebraic equivalence between MESA CG
and bounded LDU; it established that SolveDelta's former six-step code did not
retain
MESA's resident efficiency. The cause is visible in source: the local strict
action and its VJP scan all `r` coordinates with scalar select/reduction state,
whereas upstream `chunk_update_once` expresses each action as two Tensor-Core
dots inside one resident loop. The R-route Gram transpose and composed sigma
reverse are likewise SolveDelta schedules built from MESA/FLA formulas, not
copied upstream programs.

Accordingly, provenance uses precise terms. MESA owns the adopted geometry
state loop and supplies the two-dot action primitive; FLA generalized-Delta
owns the blocked pair-dot/ordered-diagonal substitution pattern. SolveDelta
specializes the structured J/D and local u/h pair generator without adopting
either upstream ABI wholesale.

Decision revision (2026-08-26): fixed-degree Neumann is no longer authorized.
Each unit-triangular factor is evaluated by an exact coordinate-axis
generalized-Delta recurrence. The connected kernel ports FLA GDN2/KDA's
blocked-substitution split: complete coordinate blocks use pair dots while the
diagonal block preserves ordered substitution. It generates the J/D boundary
and u/h local pair tiles at their consumer, so neither a tokenwise dense factor
nor the notation-only generalized-Delta feature panel reaches HBM. Reverse
uses the corresponding exact transpose substitution and the implicit identity
`bar_N=-z y^T`.

The upstream `solve_tril` ABI was deliberately not copied. It writes a complete
strict factor and its inverse; at SolveDelta's one-factor-per-token shape that
would dominate HBM. What is reused is its diagonal solve and block-composition
schedule, with a reduced structured producer interface. Two attempts to reuse
FLA's causal `dA x feature` backward schedule without its materialized `dA`
layout were measured and deleted. A source-parallel variant regressed target
F+B from about `9.64 ms` to `24.09 ms`; grouping eight sources per CTA regressed
it to `34.82 ms`. Both repeated a causal-mask MMA per source and incurred too
many atomics. The current fastest local transpose remains a resident
prefix/suffix state pass plus ordered 32-coordinate diagonal reverse. Replacing
it requires changing the upstream representation or joining it to chart
reverse, not another local `tl.dot` rewrite.

The exact path passes the rebuilt K1/K2 FP64 output/state/composed-VJP suite,
mask/reset semantics, bitwise symmetric J, aligned recurrent splits, and exact
GDN2/DeltaProduct-K identity-geometry reductions. At the target shape,
C32/four-warps measures about `1.50/9.32 ms` forward/F+B in the direct variant
A/B, while C16 is `1.87/17.96 ms`, C64/four-warps is `5.79/24.05 ms`, and
C32/eight-warps is `1.96/12.94 ms`. The production specialization therefore
keeps C32/four-warps; the other schedules are rejected measurements rather
than runtime selectors.

Decision update (2026-08-26): mask/reset execution now uses FLA's variable-
length ownership across the complete frame and exterior. Valid tokens are
gathered once into a flat batch, valid resets define `cu_seqlens`, and
`prepare_chunk_indices` gives normalization, the MESA J/D scan, radial/chart,
and frame actions one shared global chunk order. `prepare_chunk_offsets` maps
those source-token frame panels into the independently chunked edit-expanded
direct-e/WY schedule. The transpose consumes the same indices and offsets;
there is no rectangular `[segment,max_length,...]` frame buffer or per-segment
kernel loop. FLA pair/WY/state/output kernels remain separate from the frame,
so this is metadata/layout fusion rather than a high-pressure CTA fusion.

On a measured `B=4,T=512,H=8,r=d_v=128,K=1,C=32` workload with one empty
batch and 12 unequal reset-free segments, rectangular frame scheduling needed
864 head-panels while the packed schedule needed 360. Frame-only forward fell
from `4.169` to `1.967 ms`, F+B from `26.632` to `11.892 ms`, forward allocator
peak increment from `341.4` to `150.6 MiB`, and F+B peak increment from
`1072.4` to `454.0 MiB`. The complete packed operator measured `3.936 ms`
forward and `15.630 ms` F+B; removing explicit per-segment zero initial states
reduced its forward peak increment from about `206` to `186 MiB`. The all-valid
target specialization remained separate and measured `1.770/8.777 ms`
forward/F+B in the same development pass.

### Upstream-original latency audit

A seven-round alternating A/B on the RTX 5070 Ti used
`B=1,T=1024,H=8,K=1,r=d_v=128,C=32`, CUDA events, and warmed Triton caches.
The direct-`e` exterior was compared with FLA DPLR under the exact variable
mapping `q=chi`, `k=b=d`, `a=-e*exp(g)`, `v=write*values`, and `scale=1`.
Its BF16 output/final-state differences from the generic arithmetic schedule
were `7.81e-3/8.44e-3`; these are private schedule differences rather than a
second operator contract.

| Comparison | Current | Upstream original | Result |
|---|---:|---:|---|
| direct-`e` exterior forward vs preformed generic DPLR inputs | `0.220 ms` | `0.238 ms` | current is `7%` faster |
| direct-`e` exterior F+B vs preformed generic DPLR inputs | `1.166 ms` | `1.198 ms` | current is `3%` faster |
| direct-`e` exterior forward vs generic ABI including `a/z` formation | `0.218 ms` | `0.311 ms` | current is `30%` faster |
| direct-`e` exterior F+B vs generic ABI including `a/z` formation | `1.105 ms` | `1.359 ms` | current is `19%` faster |
| J/D FP32 boundaries plus scalar mass forward | `0.145 ms` | MESA two-state FP32 `0.108 ms` | current is `34%` slower including mass |
| complete operator forward | `1.684 ms` | GDN2 C64 `0.350 ms` | current is `4.81x` slower |
| complete operator F+B | `8.798 ms` | GDN2 C64 `1.321 ms` | current is `6.66x` slower |
| complete operator allocator peak in F+B | `322.9 MiB` | GDN2 `74.1 MiB` | current is `4.36x` larger |

The direct-`e` peak was `78.1 MiB`, versus `81.1 MiB` for preformed generic
DPLR and `97.1 MiB` when the generic `a/z` ABI was composed. It is therefore a
real upstream-quality specialization and is not the remaining performance
problem.

The MESA comparison used C32, `states_in_fp32=True`, two dense `128x128`
states, and the same scalar decay. Current additionally computes and stores the
effective mass. Kernel attribution separated about `55.2 us` for the current
paired J/D state loop and `17.8 us` for mass, versus `45.4 us` for the original
MESA pair and `0.9 us` for its cumsum. Thus mass explains part of the wall-time
gap, but the copied state schedule itself remains about `21%` slower than the
original and misses the blueprint's 15-percent target. Its absolute gap is
small and is not the first optimization target.

The complete comparison is intentionally workload-matched rather than
algebraically equivalent: GDN2's upstream implementation fixes C64 and does
not own SolveDelta geometry. Subtracting forward from F+B gives about
`7.114 ms` backward for SolveDelta versus `0.970 ms` for GDN2. A three-step
profile attributed `2.266 ms` per iteration to eight
`_local_vjp_resident_kernel` launches, `1.148 ms` to four exact coordinate
solves, `0.823 ms` to five direct block-output launches, and `0.771 ms` to the
sigma reverse. The first hotspot alone is slower than the complete original
GDN2 F+B.

Source inspection narrows the meaning of "adapted". The direct coordinate
action stores two FP32 block-prefix tensors between
`_direct_prefix_states_kernel` and `_direct_block_output_kernel`. At the target
shape its paired two-RHS call writes `2 x 8 MiB` before the block consumers
reload it. The local transpose keeps no coordinate descriptor in HBM, but
still performs a scalar 128-coordinate loop with select/reduction work in each
iteration. The sigma reverse keeps several `32x128` values and a `32x32`
weight in one CTA. These are project-owned lowerings built from upstream
identities, not original MESA/GDN2 programs. Future performance claims must
distinguish this source-level provenance from measured schedule equivalence.

Decision update (2026-08-26): the first post-audit schedule replacements were
accepted or rejected only on the complete path. Retiling sigma reverse so one
panel CTA keeps the C-by-C temporal weight resident and streams 32-coordinate
tiles reduced `_sigma_backward_tiled_kernel` from roughly `0.77-0.91 ms` to
`0.050-0.052 ms`. The rebuilt composed-VJP suite remained 8/8. The geometry
resident loop now lets its `(tile_i=0,tile_j=0)` owner produce the scalar mass
boundaries, token masses, tail weights, and chunk decay from the cumulative
values already loaded by the J/D program. It also omits J loads and products
in upper-coordinate CTAs, where only unconstrained D exists. This deleted the
separate approximately `17.8 us` mass launch while changing the combined
geometry kernel from about `64.8 us` to `66.5 us` in the same profiler class.

Three direct-action no-prefix-HBM schedules were measured and deleted: a
single-CTA resident action (`11.72 ms` complete F+B), a coordinate-block
consumer that recomputed prefixes (`10.61 ms`), and a per-RHS resident split
(`14.13 ms`), versus the approximately `9 ms` selected path. The communication
choice is unavoidable in these Triton decompositions: coordinate CTAs must
share a prefix, recompute it, or serialize the coordinate traversal. Lower HBM
traffic did not compensate for lost CTA parallelism or longer live state.

Likewise, a block-parallel strict transpose that reconstructed chart scale as
`kappa_i*exp(G_i-G_j)` passed composed VJPs but regressed complete F+B to
about `13.11 ms`. A resident block variant used about `3.121 ms` for local
reverse versus `2.266 ms` for the scalar resident baseline because it extended
the `omega/prefix/suffix` live set. Upstream inspection found no mature exact
no-`dA` replacement to transplant: FLA generalized-Delta/WY reverse forms and
consumes a materialized `dA`, while MESA Hkk/Hkv backward owns the radial
Gram/Hadamard transpose rather than this coordinate-triangular factor
cotangent. A future replacement must either port a complete CUDA resident
program with its producer layout or batch multiple primal/dual cotangents in a
multi-input resident transpose without first concatenating an HBM panel.

Decision update (2026-08-26): five low-risk interface specializations were
adopted after the larger scheduling audit. First, direct-`e` backward now
replays only the pair/WY/state caches it consumes; it does not launch FLA's
final output kernel or allocate the discarded output. The complete-layer A/B
was `10.134 ms` for cache-only replay versus `10.187 ms` for the old replay.
Second, the layer uses FLA's `fused_gdn_gate` for geometry and associative
decays. Complete forward improved from about `1.976` to `1.876 ms`; complete
F+B changed from `10.351` to `10.534 ms`. The forward winner is retained. Its
transpose is mathematically connected and a formula/composed-VJP check covers
raw input, log-rate, and bias gradients, but the upstream schedule is not a
fully fused parameter transpose: it writes `[ceil(BT/32),H]` `dA` partials,
reduces them with `sum(0)`, and separately rereads `dg` to reduce `dbias`.

Passing the BF16 projection view directly to that upstream gate was tested and
rejected for the present precision contract. Raw and log-rate gradients were
unchanged or within `1.4e-7` relative error, but upstream forms `dbias` from
the input-typed `dg`; BF16 rounding raised its relative error to `2.28e-3`.
Production therefore keeps an explicit FP32 gate-input boundary until a
specialized FLA backward computes both `dA` and `dbias` partials from the FP32
register value in the same kernel. This is a static interface decision, not a
runtime precision fallback.

Third, the no-mask/no-reset branch now enters the rectangular fast path by
Python argument identity and performs no CUDA `all/any` synchronization.
Fourth, explicit mask/reset execution discovers one shared `PackedSegments`
plan and uses causal-conv1d's `seq_idx` path. Initial cache entries are prefixed
only to the first non-reset segment and final cache entries are gathered from
the last segment, so their full state VJP remains connected. On the measured
packed workload the native varlen conv path took `5.957 ms` for complete
forward versus `47.059 ms` for the deleted Python token loop. Fifth, unique
token gather/scatter uses FLA `index_first_axis`/`pad_input`, and one CPU
`cu_seqlens` mirror is passed to every FLA metadata consumer. Repeated
batch-to-segment state selection deliberately remains `index_select`: FLA's
unique-index scatter transpose would overwrite rather than accumulate those
cotangents.

The rebuilt suite is now 10/10, including packed conv output/final-cache/input,
initial-cache and weight VJPs plus the fused gate composed VJP. A final matched
complete-layer run at `B=1,T=1024,H=8,r=d_v=128,K=1` measured SolveDelta at
`1.865/10.033 ms` forward/F+B and GDN2 at `0.990/3.267 ms`; allocator peaks
were `128.3/363.0 MiB` and `38.1/102.8 MiB`. These wrapper improvements do not
change the conclusion that SolveDelta's frame transpose and gradient partial
lifetimes, rather than the mature exterior, own the remaining gap.

Decision update (2026-08-26): the mature-interface pass closed the avoidable
gate and gradient-lifetime gaps. The FLA GDN gate program now reads strided
BF16 projection views directly. Its transpose derives both log-rate and bias
partials from the same FP32 register value and reduces them in one fixed tree;
`dg` is stored only in the raw input dtype. This removes the explicit FP32 raw
panel and the upstream second `dg` scan without reproducing the rejected
BF16-rounded-bias error. Erase sigmoid is composed into normalization's
dual-source epilogue, while write sigmoid is composed into direct-e's
FLA-native `z` pack; their strict transposes remain in those same owners.

The normalization producer now directly stores bounded `u/key/paired-dual`
panels as FP16 and unbounded `h` as BF16. The chart producer writes bounded
sigma directly as FP16. H Gram forward and reverse both consume the same FP16
packed bits; the previous BF16 reverse replay was removed. FP32 state, radial,
gate, sensitive decay-scalar, and backward accumulation rules are unchanged.

Finally, the four factor reverses no longer return four complete sets of
`grad_J/grad_D/grad_u/grad_h`. Their disjoint CTA tile owners accumulate the
primal-upper, primal-lower, dual-upper, and dual-lower contributions directly
into one final FP32 buffer. No atomics or cross-chunk mega-kernel were added.
The rebuilt 10-test semantic/VJP suite passes. Three complete-layer runs at
`B=1,T=1024,H=8,r=d_v=128,K=1,C=32` measured SolveDelta forward medians
`1.776--1.808 ms` and F+B medians `9.392--9.544 ms`; allocator peaks were
`106.3 MiB` forward and `211.4 MiB` F+B. Matched GDN2 was approximately
`0.970--1.028/3.024--3.064 ms` with `38.1/102.8 MiB`. Relative to the prior
`128.3/363.0 MiB` SolveDelta peaks, the eliminated action partial ABI accounts
for the expected large backward-memory reduction and also improves complete
F+B, so the schedule is retained.

Decision update (2026-08-26): the next frame-reverse pass specialized four
additional mature ownership patterns without merging unrelated live ranges.
`BoundaryStats` now follows output-owned accumulation: one CTA owns each final
u/h tile and consumes the H-left/H-right or ordered R streams before storing.
Six vector partials, two D-matrix partials, and their combine kernel were
deleted. This reduced the target F+B allocator peak from `211.4` to
`196.8 MiB`; the measured complete F+B remained in the existing roughly
`9.3--9.8 ms` range.

The geometry decay scalar transpose now uses the same resident state-tile
epilogue organization as MESA's Hkk/Hkv reverse. It contracts FP32 boundary
state tiles directly into a compact approximately `132 KiB` partial and then
uses one fixed reduction. The former `u_panel.float()/h_panel.float()` plus two
full `torch.bmm` temporaries, about `16 MiB` at the target, are gone. The two
new launches measured about `65.6 us` and `1.9 us`; complete latency was
neutral, so the lower transient footprint is retained.

Boundary factor reverse now separates output owners as FLA `chunk_o` and KDA
`dAv` do. A matrix-tile CTA exclusively accumulates one final FP32 J/D tile;
a panel CTA exclusively accumulates the H/R coefficient rows while streaming
the same strict boundary tiles. An attempted fusion that made every matrix CTA
write coefficient partials was rejected because it introduced a `2--4 MiB`
workspace and a finalize launch. The selected split has no coefficient
workspace; the target profiler measured the four coefficient owners at about
`0.038--0.039 ms` each and matrix-owner launches at about `0.053--0.077 ms`.

Finally, R strict Gram forward and transpose now use MESA's block-prefix
schedule rather than a 128-step global-coordinate outer-product loop and an
`NB^2` source-block replay. Each 16-coordinate block forms `Ku/Kh` with
`tl.dot`; forward keeps the complete-block prefix resident and evaluates only
the within-block strict epilogue. Reverse uses ordered prefix and suffix
launches, loading each coordinate block once per direction and accumulating
into the same output-owned tile without atomics or a partial workspace. It
uses exactly one `Z+Z^T` in the broad Gram transpose; the former code
symmetrized that cotangent twice. The port also keeps H's full-width reduction
tile separate from R's 16-coordinate block. An intermediate implementation
overwrote the saved H tile width with 16 and left coordinates 16:r
uninitialized in H backward; the rebuilt composed VJP exposed and deleted
that bug.

A dedicated C16/r64 same-packed forward/VJP diagnostic is below `5e-3`
relative error. At C32/r128 the R forward and two transpose launches measured
`0.0688 ms` and `0.0540+0.0515 ms`, versus about `0.26 ms` for the former R
transpose. The current operator-only target measured `1.537 ms` forward and
`7.986 ms` F+B; the complete layer measured `1.719/9.305 ms` with
`106.3/196.8 MiB` forward/F+B allocator peaks. These are adapted upstream
schedules with SolveDelta strict masks and interfaces, not claims that the
source is byte-identical to MESA.

Decision update (2026-08-26): the H and R local factor terms were reduced to
their minimal generalized-Delta generator before further schedule work. With
token panels `U,H`, each target row is exactly

`N_local = U.T @ K_i`, where
`K_i = diag(omega_h_i) @ U + diag(omega_r_i) @ H`.

This is an algebraic specialization of the same FLA/KDA blocked generator and
transpose machinery already cited above; it introduces no new external code
source. The direct action now applies the two route weights in its prefix
producer and writes one FP32 block-prefix buffer instead of two. The strict
reverse maintains one `K_i` prefix and one `U` suffix, then splits `bar_K_i`
into final u/h and the two radial-weight cotangents in the owner epilogue. The
former separate symmetric-H and R coordinate loops are deleted, reducing the
four factor paths from eight local launches to four. Neither `K_i`, `bar_K_i`,
nor a strict matrix reaches HBM.

The real-valued generator fusion does not authorize one universal private
operand dtype. A trial that also collapsed the exact solve's bounded-H FP16
contraction and unbounded-R BF16 contraction into one BF16 dot passed factor
action checks but raised the fixed composed `values`-gradient maximum from
inside the `3e-2` gate to `3.1898e-2`. It was rejected. Production retains a
static mixed-precision contraction decomposition of the same generator in the
exact solve. Forward dual prefixes have an analytic normalization/factor bound
and use FP16 operands; reverse direct-action prefixes consume unbounded
cotangents and use BF16 operands. This choice is compile-time and has no
runtime threshold or fallback.

Three alternating warmed operator-only A/B rounds at
`B1,T1024,H8,K1,r=d_v=128,C32` measured the `6ca37ff` baseline at
`1.482--1.488 ms` forward and `7.092--7.113 ms` F+B. The single-generator path
measured `1.441--1.461 ms` and `6.480--6.690 ms`. Profiler attribution changed
local reverse from eight launches totaling `2.145--2.237 ms` to four totaling
`1.687--1.757 ms`; direct prefix/output changed from approximately
`0.064--0.067/0.772--0.800 ms` to `0.044--0.046/0.701--0.728 ms`. Exact solve
remained approximately `1.01--1.05 ms`, as expected from retaining its mixed
precision lowering. The operator F+B peak fell from `172.8` to `164.8 MiB`.

Two complete-layer A/B rounds including projection, conv4, output projection,
and all parameter gradients measured the baseline at `1.762--1.834 ms`
forward and `8.071--8.073 ms` F+B, versus `1.735--1.743 ms` and
`7.462--7.506 ms` for the selected path. F+B allocator peak fell from `196.8`
to `188.8 MiB`. The rebuilt suite is 12/12, including an independent FP64
single-generator forward/VJP identity test and the complete operator VJPs.

Decision update (2026-08-26): two remaining ownership variants were tested
after the single-generator path and rejected rather than retained as optional
ABIs. First, chart forward materialized three FP32 route-specific
`omega=kappa*Delta` tables while keeping the shared causal `Delta` table for
the analytic chart reverse. Exact solve, direct action, and local transpose
then consumed the preformed weights. The rebuilt 12-test suite passed, and the
local transpose profile changed only from about `2.08 ms` to `2.00 ms`; the
candidate added 3 MiB of saved route tables and its fresh complete-layer F+B
runs were `8.74--9.01 ms`, versus the selected shared-decay path's established
approximately `7.96 ms` run before the experiment. It did not establish a
repeatable complete-path win and was deleted. Production retains one shared
FP32 `Delta` table and forms each route weight once in the consuming CTA.

Second, the combined local transpose was split into a KDA/GDN2-style final
u/h output owner and an FLA-style scalar-weight owner. This removed the two
weight-cotangent panels from the vector kernel's live set, but both owners had
to replay the coordinate panels and suffix recurrence. The same semantic/VJP
suite passed, while the eight resulting launches totaled `2.31 ms` versus
`2.08 ms` for the four selected staged-owner launches. The split was deleted.
This falsifies mechanical owner separation for this recurrence: its vector
and scalar cotangents share enough exact prefix/suffix work that one staged
owner is cheaper. No upstream kernel was found that owns SolveDelta's distinct
per-token coordinate-triangular one-RHS solve; FLA's public triangular inverse
programs amortize one token-triangular factor over many RHS and cannot replace
this owner by an ABI rename.

Decision update (2026-08-26): the remaining exact solve and local transpose
were investigated against newer small-TRSM and staged-backward programs before
another production change. Two isolated exact-solve CUDA prototypes were
rejected. A one-CTA shared-resident owner reduced the Triton kernel's
255-register live set to 92--122 registers with no local-memory spill, but
measured `0.718--0.748 ms` per action versus `0.271--0.414 ms` for the selected
Triton path. A MAGMA-style four-block recursive solve used 37--80 registers and
seven launch waves, but measured `0.737--0.777 ms` and introduced about
`3.2e-3` relative grouped-reduction drift. Both paid factor generation,
staging, and synchronization costs that conventional TRSM avoids by assuming
an already materialized factor. The prototypes were deleted; low register
count alone is not an adoption result.

The local transpose instead adopted the resource schedule in FLA DPLR
TileLang commit `38a496e1ce58baaf1bc6613176eb2f433d0ddb90`, specifically its
256-thread fragment distribution, phase-reused shared staging, and final
output owner. SolveDelta's CUDA specialization assigns four target tokens to
each of eight warps and one source channel to each lane. It keeps the exact
coordinate prefix/suffix recurrence in registers, accumulates sixteen
coordinate outputs per phase, and reuses the 32 KiB output staging buffer for
the final row/column decay reduction. It does not copy DPLR's temporal-state
mathematics or ABI.

The C32/r128 one- and two-RHS kernels compile to 128 registers/thread, 50,176
bytes shared memory, and `LOCAL=0`, allowing two CTAs/SM on the RTX 5070 Ti.
Four real route specializations individually measured approximately
`0.19--0.27 ms`; their profiler total was about `1.00 ms`, versus the former
`1.69--2.08 ms`. The maximum same-input difference from the prior exact
transpose owner was `1.9e-6` absolute and below `2.5e-7` relative across
`bar U/bar H/bar kappa/bar G`.

Seven alternating complete-operator A/B rounds at
`B1,T1024,H8,K1,r=d_v=128,C32` measured the old Triton local owner at
`7.653 ms` F+B median and the native owner at `6.611 ms`; the respective
sample tails were `8.041/6.981 ms`. A complete layer including projections,
conv4, final state, and all parameter gradients measured `8.597/7.544 ms` in
five alternating rounds. The rebuilt suite remains 12/12.

The same pass closed two adjacent epilogues without enlarging the frame
kernel. Lower boundary coefficients now accumulate `bar kappa_H/bar G`
directly into the upper-owned final buffers, deleting the route-scalar sum.
Three small sigma owners write the lower cotangent, FP16 dual scale, FP32 dual
cotangent, and final scale cotangent without materializing `diagonal`,
`dual_sigma`, or a final two-input sum. The resulting complete operator
measured `6.160 ms` F+B and `179.0 MiB` allocator peak in a separate seven-round
run. Finally, boundary matrix/coefficient owners selected BK64/4-warps for
widths divisible by 64: isolated time was `0.136 ms` versus `0.186 ms` for
BK32, with at most `3.8e-6` reduction-order difference. BK16 and all 8-warp
candidates were rejected.

Decision update (2026-08-26): the local generator reverse now specializes the
same-direction ownership already present in the boundary reverse. One native
C32/r128/K1 owner consumes one primal route and two paired-dual routes through
independent static pointers and dtypes; it does not construct a three-route
HBM panel. Lower and upper local reverse therefore require two launches rather
than four. The compiled owner retains 128 registers/thread, 50,176 bytes
shared memory, `LOCAL=0`, and two CTAs/SM. The paired-dual lower FP16 panel is
also retained from forward and consumed by reverse, deleting a complete lower
direct-action replay. Seven alternating complete-layer rounds measured the
old split-owner plus replay path at `15.546 ms` F+B median and the selected
mixed-owner plus cache path at `13.653 ms`; all seven rounds favored the
selected path. Absolute clocks drifted during the run, so the within-round
approximately 12.2 percent result, rather than either absolute number, is the
adoption evidence. The logical saved-cache cost is 4 MiB at the target and did
not increase the measured allocator peak.

PyTorch's official
[CUDA Graph semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
and
[`make_graphed_callables`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html)
informed the graph-level experiment. A warmed fixed-shape complete SolveDelta
layer captured forward and backward successfully with bitwise-identical output
and gradients. Seven alternating rounds measured eager and graphed F+B round
medians at `11.790` and `10.515 ms`, about a 10.8 percent reduction under the
same drifting clocks. Direct `torch.compile(mode="reduce-overhead")` did not
capture the same graph because causal-conv1d's non-contiguous output contract,
Triton constexpr signatures, the lazy pybind extension loader, and FLA device
queries create independent graph breaks. The selected decision is therefore
trainer-owned static CUDA Graph capture, not a SolveDelta wrapper or a broad
Inductor-compatibility rewrite.

Decision update (2026-08-26): the final non-solve maturity pass retained three
narrow ownership changes. First, H, R-lower, and R-upper chart coefficient
routes now share one FLA-style panel owner in forward and reverse, reusing the
same decay/mass/pair loads while storing distinct route statistics. The route
forward and reverse kernels measured about `0.0065/0.0527 ms`; the complete
semantic/state/composed-VJP suite remained 12/12. Second, bounded forward
direct-action prefixes are written directly from FP32 producers to FP16 and
consumed with FP32 accumulation, halving that private prefix traffic without
BF16-to-FP16 pseudo-promotion. Reverse prefixes remain FP32. Third, geometry
matrix tile `(0,0)` now owns mass boundary reverse and the decay finalize owner
writes directly to `[B,T,H]`, deleting the standalone mass scan and panel-copy
launches. Packed/reset VJPs pass the same suite.

The surrounding audit deliberately did not fuse every remaining frame stage.
Boundary statistics already use separate KDA/FLA matrix-tile and coefficient-
row owners; merging them recreates the measured `2--4 MiB` partial workspace.
Reset-delimited packing constructs one plan and one CPU metadata mirror, then
reuses FLA gather/scatter and chunk metadata across conv, frame, and WY. Six
FP32 add kernels totaling about `0.15 ms` are autograd accumulation across
independent geometry, Gram, boundary, and action branches; removing them would
require a whole-frame custom autograd owner, not a multi-tensor rename. That
larger lifetime change was rejected for this pass. A fresh matched complete
BF16 layer measured SolveDelta median/p95 `1.648/1.910 ms` forward and
`6.445/6.790 ms` F+B with `90.3/182.5 MiB` incremental peaks. Matched FLA GDN2
measured `1.068/1.303 ms`, `3.116/3.548 ms`, and `38.1/102.8 MiB`.

Decision update (2026-08-27): the exact coordinate solve retained FLA's
16-coordinate ordered substitution but shortened the full-width solution live
range. The selected C32/r128 owner keeps only the current FP32 solved block and
the generalized-Delta prefix resident, writes that block directly to the
already required final output panel, and reloads completed blocks in BF16 for
later boundary Tensor-Core contractions. Forward bounded panels use their
declared direct FP16/BF16 stores; transpose panels remain FP32. This adds no
workspace, launch, dense factor, or recomputation and preserves the exact
coordinate recurrence.

This schedule was selected against the otherwise identical full-width
resident Triton owner. Four continuously launched exact actions totaled about
`1.078 ms` versus `1.286 ms`; three independent complete-layer runs had median
F+B about `6.123 ms` versus `6.369 ms`, a roughly 3.9 percent improvement, with
the same `182.5 MiB` peak. The four compiled specializations still report 255
registers/thread, but Triton spills fell from the previous roughly `94--106`
to `0/4/6/16`; final-panel traffic replaced compiler-generated local-memory
traffic. The full rebuilt suite remains 12/12.

Three nearby variants were rejected. Splitting rows into M16 CTAs duplicated
J/D/u/h loads and regressed complete F+B to about `7.06 ms`. Grouping prior
blocks into K32 tiles raised the FP32 lower-transpose action from about `0.286`
to `0.338 ms`. Eight warps raised the four-action total to about `1.80 ms`, and
forcing `maxnreg=168` raised it to about `1.38 ms`; neither occupancy tactic
paid for its larger live-range movement.

## Resident fixed-degree Neumann re-evaluation

FLA MESA's MIT-licensed
[`chunk_cg_solver_fwd.py`](https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/fla/ops/mesa_net/chunk_cg_solver_fwd.py)
and
[`chunk_cg_solver_bwd.py`](https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/fla/ops/mesa_net/chunk_cg_solver_bwd.py)
were re-evaluated as the owner skeleton for the previously rejected degree-six
Neumann experiment. A temporary private kernel kept `b`, the current iterate,
and the next iterate in one CTA, compiled the six updates as a fixed loop,
used the structured `N` or `N^T` action, and wrote only the final iterate. Its
backward used the same transpose owner as an implicit/phantom adjoint rather
than differentiating six polynomial nodes. No experimental selector or source
survives in the production tree.

The experiment exposed a structural mismatch with MESA's cheap action. MESA
applies an unmasked low-rank operator as two associative dots. SolveDelta
requires

\[
P_{\mathop{\rm strict}}\!\left(B_i+U^TK_i\right)x,
\]

where the strict projection is on the coordinate axes and `K_i` varies with
the target token. The projection prevents reassociating the complete local
action into one unmasked `((X U^T) \odot W)U` pair. A resident implementation
must still traverse coordinate tiles, apply strict within-tile work, and run
the broad J/D contractions once per polynomial update. The exact blocked
substitution pays the corresponding off-diagonal contractions once and keeps
only the 16-coordinate diagonal dependency ordered.

On SM120 at `P=256,C=32,r=128`, one lower factor measured `0.225 ms` for the
selected exact solve and `5.213 ms` for the temporary six-update owner. The
candidate compiled with 255 registers/thread, 16,384 bytes shared memory, and
246 spills; eight warps retained 240 spills and regressed to `6.076 ms`. C16
reduced the candidate to `2.954 ms`, still far above its `0.307 ms` exact
comparison. Independently, one selected two-kernel direct action measured
`0.134 ms`; six applications therefore carry substantially more contraction
work than the `0.295 ms` exact transpose solve even before considering the
candidate's register lifetime.

The approximation itself was numerically viable. With
`||N||_2 <= 1/4`, six updates from `x_0=b` realize degree six and retain the
static remainder bound `1/12288`. Substituting the candidate throughout the
complete layer passed the rebuilt 12/12 suite. At the target shape its BF16
layer output differed from the exact path by `1.26e-3` relative norm and its
aggregate composed gradient by `1.53e-3`. Performance nevertheless regressed
to `11.02 ms` forward and `26.27 ms` F+B, versus the current exact path's
previous matched `1.60/6.41 ms`. After deleting the candidate, a fresh exact
verification measured `1.64/5.72 ms` against matched GDN2 `1.04/3.28 ms`,
with the unchanged `90.3/182.5 MiB` allocator peaks. This rejects whole-factor
fixed-degree Neumann for the current strict chart on performance,
independently of its production-semantics status.

FLA
[`PR #1162`](https://github.com/fla-org/flash-linear-attention/pull/1162)
remains a useful but narrower precedent: its exact finite-Neumann products
invert one explicit 16x16 triangular block shared by a chunk program. A
SolveDelta frame has a different diagonal block for every target token, so
adopting that kernel literally would first materialize or separately own those
blocks and would lose the current wide-RHS owner. It does not rescue the
rejected whole-factor experiment without another measured block schedule.

## Tensor-Core prefix in the exact local transpose owner

Decision update (2026-08-27): FLA MESA's MIT-licensed
[`chunk_update_once`](https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/fla/ops/mesa_net/chunk_cg_solver_bwd.py)
and FLA DPLR's
[`chunk_dplr_bwd_kernel_intra_tensorcore`](https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/fla/ops/generalized_delta_rule/dplr/chunk_A_bwd.py)
were used to re-audit the selected C32/r128 local transpose. Both mature
schedules cast multiplicands to their low-precision consumer type, execute
the broad contraction as a dot, and retain FP32 accumulators and scalar gate
work. NVIDIA's CUDA 13 WMMA `__nv_bfloat16` fragments provide the corresponding
native CUDA primitive for the existing owner.

The old owner computed its complete-coordinate initialization
`X U^T` and `X H^T`, logically two `96x128 @ 128x32` products, through an
unrolled FP32 `fmaf` loop in every lane. The selected specialization replaces
only those two products with BF16 `m16n16k16` WMMA and FP32 accumulation. Six
warps own 16 route rows and both 16-source halves. Both result panels and the
warp-local conversion tile reuse the owner's previously dead 32 KiB output
scratch. After the result is consumed into the resident prefix, `u` is
reloaded as its original direct-FP16 bits into the same shared union and the
exact scalar prefix/suffix recurrence continues unchanged. No prefix HBM
tensor, new launch, extra shared allocation, route ABI, or solve approximation
is introduced.

The built SM120 cubin contains 256 BF16 MMA instructions across its four mixed
specializations. Each remains at 128 registers/thread, 50,176 bytes shared,
zero local allocation, and two CTAs/SM. The rebuilt semantic/state/composed-
VJP suite passes 12/12. Stable profiler attribution reduced the two production
mixed owners from `0.2853+0.2674=0.5527 ms` to
`0.2132+0.1903=0.4035 ms`, about 27 percent.

Seven same-process alternating rounds loaded the old and new cubins under the
same layer. Backward-only round medians were `4.656/4.336 ms`; complete F+B
round medians were `6.477/6.152 ms`. A final matched run measured SolveDelta
`1.662/5.990 ms` forward/F+B versus GDN2 `1.064/3.225 ms`, with SolveDelta F+B
incremental peak unchanged at about 182.5 MiB. This adopts the WMMA prefix.
The remaining strict coordinate recurrence and the 16-coordinate diagonal
substitution retain FP32 scalar arithmetic because they carry exact ordered
dependencies; radial/gate reductions and continuation states retain FP32 by
the precision contract.

## Native grouped RLS exterior and block-Woodbury A/B

Decision update (2026-08-27): FLA's MIT-licensed generalized-DPLR
[`chunk_h` and `chunk_o`](https://github.com/fla-org/flash-linear-attention/tree/5e02dd3a7651f5f2797eb8b12bbec401826031e1/fla/ops/generalized_delta_rule/dplr)
state/output ownership, together with GatedDeltaProduct's native grouping
strategy, informed an isolated `E=3` direct-`e` RLS specialization. The
selected forward keeps one `r x BV` state tile resident across both geometry
transports and the ordinary edit, joins equal-direction terms, and stores only
the final read for each token. It retains the mature direct-`e` pair/WY
boundary and does not modify production SolveDelta.

A native grouped transpose was implemented from the exact three-step reverse
and passed the experiment's output, final-state, and per-input gradient checks.
Its source-parallel closure nevertheless required roughly 80 million
cross-value-tile FP32 atomics at `B1,T1024,H8,r=V=128`, producing about
`3.4 ms` isolated exterior F+B. Saving additional state replay points did not
materially change that result. FLA's existing three-owner DPLR reverse was
therefore retained. It is faster because value tiles own their final outputs
without cross-tile atomics, despite retaining expanded logical cotangent and
zero-value panels. This is a measured ownership decision, not an ABI
compatibility constraint.

With C16/C32, isolated exterior forward measured about `0.654/0.516 ms` and
F+B `1.303/1.236 ms`; C64 was slower. C32 is selected, but the requested
`0.65--0.75 ms` F+B target was not reached. The retained reverse attribution
is about `0.226 ms` for resident state, `0.179 ms` for direct-`e` pair,
`0.166 ms` for output, plus WY and preparation. Closing the remaining gap
requires a grouped output-owner reverse rather than source-parallel atomic
closure.

Golub and Van Loan's block inverse identities and the classical Woodbury
matrix identity informed a second isolated candidate for variable-decay block
RLS. The candidate used one common boundary CG solve for `Z=J0^-1 X`, formed
the C32 Schur matrix `I+X^T Z`, and used leading-principal Cholesky solves to
recover every prefix gain. It agreed with the token FP32 route to maximum
FP16 errors of `9.8e-5` for gain, `5.4e-4` for arbitrary-RHS solve, and
`9.9e-4` for prediction. Target latency was `0.790 ms`, versus `0.364 ms` for
the matched MESA two-solve/action path: two prefix solves cost `0.514 ms`, two
boundary CG solves `0.210 ms`, and factor/cross-action work only
`0.013/0.010 ms`. It therefore misses the predeclared `0.12--0.15 ms` F+B gate
and was deleted; the mature MESA CG owner remains selected.

## TileLang direct-e and native E=3 RLS reverse

Decision update (2026-08-27): the isolated `experiments/rls` exterior now
specializes FLA main commit
[`5e02dd3a`](https://github.com/fla-org/flash-linear-attention/commit/5e02dd3a7651f5f2797eb8b12bbec401826031e1).
The concrete MIT-licensed donors are generalized-DPLR TileLang
[`chunk_A_fwd.py`](https://github.com/fla-org/flash-linear-attention/blob/5e02dd3a7651f5f2797eb8b12bbec401826031e1/fla/ops/generalized_delta_rule/dplr/backends/tilelang/chunk_A_fwd.py),
[`chunk_A_bwd.py`](https://github.com/fla-org/flash-linear-attention/blob/5e02dd3a7651f5f2797eb8b12bbec401826031e1/fla/ops/generalized_delta_rule/dplr/backends/tilelang/chunk_A_bwd.py),
[`wy_fast_fwd.py`](https://github.com/fla-org/flash-linear-attention/blob/5e02dd3a7651f5f2797eb8b12bbec401826031e1/fla/ops/generalized_delta_rule/dplr/backends/tilelang/wy_fast_fwd.py),
[`wy_fast_bwd.py`](https://github.com/fla-org/flash-linear-attention/blob/5e02dd3a7651f5f2797eb8b12bbec401826031e1/fla/ops/generalized_delta_rule/dplr/backends/tilelang/wy_fast_bwd.py),
and the low-shared-memory split ownership in
[`chunk_stream_bwd.py`](https://github.com/fla-org/flash-linear-attention/blob/5e02dd3a7651f5f2797eb8b12bbec401826031e1/fla/ops/generalized_delta_rule/dplr/backends/tilelang/chunk_stream_bwd.py).
The output-owned `(chunk, head, value tile)` precedent was also checked against
Mamba-3's Triton backward at state-spaces/mamba commit
[`e9594ce1`](https://github.com/state-spaces/mamba/commit/e9594ce1c732d97440f0332fdc43170a2294dbfa),
especially
[`mamba3_siso_bwd.py`](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py).

The algebraic specialization is exact. In the direct-`e` exterior `b=k=d`,
so the generic four interaction matrices obey

\[
A_{qk}=A_{qb}=A_{qd},\qquad A_{ak}=A_{ab}=A_{ed}.
\]

Forward therefore computes only two Tensor-Core contractions. Reverse first
adds cotangents of each shared representative and evaluates the corresponding
four transpose contractions, rather than retaining the generic eight. FLA's
fast WY keeps the strict interaction representative in FP16, uses the declared
public vector dtype for vector panels and cotangents, and accumulates its GEMMs
in FP32.

The native static `E=3` reverse does not clone FLA's complete ABI. One value-
tile owner streams chunks in reverse, reads the compact `[B,T,H,V]` output
cotangent only when `edit=2`, and owns `du`, the recurrent state cotangent, and
the initial-state cotangent. Separate coordinate-tile owners close
`dq/dd/dw` and the chunk-tail gate statistic. This preserves the useful FLA
state/output split at the target eight-head grid while deleting the physical
`3T` output cotangent, zero-value panel, and the rejected source-parallel
atomic closure. It does not fuse all chunks into one CTA per head.

Both FP16 and BF16 connected forward/VJP gates pass. For BF16 composed VJPs,
the frozen experiment gate is relative L2 below two percent plus a `0.2`
absolute corruption ceiling; observed relative L2 errors across all ten input
groups are `0.37--1.48%`. The previous expanded reverse was not better on the
most sensitive geometry-decay group (`0.0872` versus `0.0850` maximum absolute
error), so this gate reflects BF16 observability rather than an implementation-
specific exemption.

At `B1,T1024,H8,r=V=128,E3,C32,BF16`, same-process old/new microbenchmarks
measured pair forward `0.169/0.036 ms`, pair reverse `0.524/0.136 ms`, WY
forward `0.062/0.051 ms`, and WY reverse `0.088/0.058 ms`. Thirty-repeat
trainer-style CUDA Graph replay reduced complete RLS forward from the prior
`0.639` to `0.595 ms` and F+B from `1.889` to `1.440 ms`; the matched GDN2
run was `0.124/0.466 ms`. The isolated RLS graph reservation fell from about
`306` to `206 MiB`, with persistent allocation unchanged at `22 MiB` and no
replay-time allocator peak. The selected path remains experimental and does
not alter production SolveDelta's exact bounded-LDU solve.

## Natural RLS state as an exact direct-e generalized-Delta recurrence

Decision update (2026-08-27): FLA main commit
[`865a52fd`](https://github.com/fla-org/flash-linear-attention/commit/865a52fd39f378e92f1668134a03fc666fdd56b3)
was audited after upgrading the isolated environment to FLA 0.6.0, PyTorch
2.11.0+cu130, Triton 3.6.0, and TileLang 0.1.13. Its generalized-DPLR
pair/WY/state/output owners confirm that the natural RLS cross-map needs no
new recurrence kernel once the endogenous gain is known:

\[
C_t=(I-g_tu_t^T)C_{t-1}+g_th_t^T.
\]

The temporary direct-`e` A/B generalized static `E=3` to a compile-time `E`
and reused it at `E=1` with `d=g`, `e=q=u`, and `z=h`. Forward and strict
reverse shared the same FLA-derived pair/WY and split state/output ownership;
no zero-valued `[P_hat | C]` half was written. This was an ABI specialization
of MIT-licensed FLA code, not a call through its public generic DPLR interface.

PyTorch 2.11's documented CUDA
[`torch.bmm(..., out_dtype=torch.float32)`](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html)
was also tested for block-RLS broad products. FP16/BF16 operands use the
Tensor-Core-capable path while returning FP32 accumulated outputs. The gain
owner uses `torch.linalg.cholesky_ex(..., check_errors=False)` so the known-SPD
experiment can be captured by CUDA Graph without the host synchronization of
`torch.linalg.cholesky`.

The state specialization itself was successful: it reduced the unoptimized
natural route from `17.901/50.656 ms` eager forward/F+B to
`13.597/34.920 ms`, and all FP16/BF16 forward, continuation-state, and
composed-VJP checks pass, including a two-chunk non-contiguous-state fixture.
The complete route nevertheless lost decisively. Its optimistic C32 CUDA
Graph result was `2.634/7.629 ms`, versus `0.585/1.461 ms` for the retained
MESA route. C16 and C64 were no better. Profiling found 384 small batched
GEMMs, 128 FP32 TRSM calls, and 32 C32 POTRF calls in one F+B.

The limiting dependency is endogenous: chunk `n+1` cannot construct its gain
until chunk `n` has produced `P_out`. FLA can mature the known-gain `C` state
update and transpose, but it cannot turn that cross-value-tile collective into
an ordinary independent state scan. A competitive implementation would need
a dedicated cooperative block-RLS gain/factor owner and exact transpose; no
complete upstream implementation was found. The candidate remains isolated
and is rejected as a replacement for either production SolveDelta or the
selected MESA RLS experiment. Its source, tests, static `E=1` compatibility
surface, and benchmark selector were deleted after the A/B; the derivation and
measurements remain here as the rejection record.

## Block QRD-RLS gain and direct-e implicit solve

Research update (2026-08-27): the classical square-root/QRD-RLS identities
were combined with FLA's MIT-licensed MESA and direct-`e` schedules to remove
the leading-prefix solves that dominated the earlier block-Woodbury A/B.
Golub and Van Loan's Schur-complement identities, Sherman and Morrison's
rank-one inverse update
([doi:10.1214/aoms/1177729893](https://doi.org/10.1214/aoms/1177729893)),
and Sakai and Nakaoka's QRD-RLS precedent
([doi:10.1016/S0165-1684(99)00071-7](https://doi.org/10.1016/S0165-1684(99)00071-7))
informed the derivation. For the chunk gauge `x_i=exp(-G_i/2)u_i`,

\[
Z=J_0^{-1}X,\qquad I+X^TZ=LL^T,\qquad
k_i=\frac{(L^{-1}Z)_i}{L_{ii}},\qquad
g_i=e^{-G_i/2}k_i.
\]

An FP64 comparison against tokenwise RLS is at `1e-15` scale. Unlike the
rejected implementation, this formula uses one C-by-C factorization and one
C-by-r triangular solve for all prefix gains.

The arbitrary-RHS implicit reverse closes without differentiating the
factorization. Sherman-Morrison gives

\[
J_i^{-1}b_i=e^{-G_i}J_0^{-1}
\prod_{j=0}^{i}(I-x_jk_j^T)b_i.
\]

FLA generalized-Delta/direct-`e` supplies the mature compact product shape:
`d=k`, `e=x`, strict `A_ed=(-X)K^T`, causal `A_qd=BK^T`, one unit-triangular
WY solve, and one query-output contraction. C16/C32 directly transplant the
TileLang `chunk_A` Tensor-Core pair owner and fast-WY row inversion cited in
the preceding RLS section, with a reduced `(rhs,x,k)->(A_qd,A_ed)->rhs'` ABI.
The second owner keeps `W` resident and never writes it to HBM. C64 remains on
Tensor-Core-capable library `bmm(..., out_dtype=float32)` plus FP32 TRSM until
FLA's two-block C64 inverse schedule is ported. The solved cotangent then enters
MESA Hkk/Hkv's existing strict transpose. The symmetric `J` state convention is essential:
raw Cholesky and unconstrained full-matrix `solve` gradients differ, while
their `(bar_J+bar_J^T)/2` representatives agree at FP64 rounding error.

The boundary-CG donor is FLA MESA's `chunk_cg_solver_{fwd,bwd}.py`. The
specialization retains its resident C-RHS owner and Tensor-Core dense action,
but deletes the local causal pair action whose coefficient is identically zero
for a common chunk boundary solve. NVIDIA MathDx 26.06's block-level batched
POSV remains the device-level donor for the direct boundary A/B; the current
first comparison uses PyTorch's mature batched `cholesky_ex/cholesky_solve`
before any MathDx transplant is justified by timing.

Two FP64 CPU identity/VJP tests and small FP16 CUDA forward/composed-VJP tests
pass for both boundary choices. The final isolated RLS suite is 17/17,
including the connected forward/composed-VJP path. The C16 direct-`e` pair
owner was corrected to use FLA `chunk_A`'s one-warp schedule; four warps cannot
legally partition a 16x16 TileLang MMA output tile.

The idle RTX 5070 Ti A/B at
`B1,T1024,H8,r=V=128,K1,BF16,CG30,C32` used trainer-style CUDA Graph replay.
Across 100 warmed samples, selected MESA measured `0.597/0.823 ms` forward
median/p95 and `1.451/1.686 ms` F+B. Block-QRD with the specialized boundary
CG measured `0.670/0.720 ms` and `1.490/1.705 ms`; matched GDN2 measured
`0.128/0.168 ms` and `0.468/0.762 ms`. QRD's graph-resident
allocation/reservation was `86/272 MiB`, versus MESA's `22/208 MiB`. Batched
Cholesky was decisively worse at `1.033/2.573 ms` forward/F+B medians and
roughly `150.5/348 MiB` graph allocation/reservation.

The owner-level result explains why the algebraic reduction did not win.
MESA's fused gain/prediction owner is `0.1655 ms`; the QRD gain/prediction
sequence is `0.2362 ms`, including `0.1269 ms` boundary CG30 and `0.0998 ms`
C32 Gram/POTRF/TRSM. QRD's direct-`e` implicit reverse is better,
`0.1558 ms` versus MESA's `0.2018 ms`, but its roughly `0.046 ms` reverse win
does not recover the roughly `0.071 ms` forward loss. Pack/unpack owners are
about `0.010 ms` each and direct-`e` pair/WY is `0.0281 ms`; neither is the
primary blocker.

C16 and C64 complete F+B medians were respectively `1.810/1.855 ms` and
`2.519/1.954 ms` for MESA/QRD, versus `1.451/1.490 ms` at C32. The C64 QRD
path still lacks FLA's two-block C64 inversion and is only a generic reference,
not an algorithm rejection. C32 MESA remains selected because it is faster
than block-QRD/CG and uses 64 MiB less graph-resident allocated memory. The
block-QRD source remains an isolated, mathematically validated experiment; it
does not alter production SolveDelta.

## MESA constant specialization and native RLS checkpoint A/B

Research update (2026-08-27): the selected isolated RLS path was specialized
at the complete copied-program boundary, following FLA MESA's MIT-licensed
paired Hkk/Hkv, CG, and strict-transpose schedules. The experiment fixes
`beta=1` and `ridge=0`; their pointers, loads, arithmetic, and cotangents were
deleted rather than passed through the generic MESA ABI. The retained broad
products still compile to BF16 Tensor Core MMA with FP32 accumulation, while
decay, CG alpha/beta, norms, continuation states, and backward partials remain
FP32.

FLA's L2Norm forward/reverse ownership also informed a sole-consumer
specialization in the grouped RLS source owner. Raw q and edit keys now use
the same FP32 sum/rsqrt and public-dtype rounding point inside that owner; its
transpose closes there. The normalized u panel remains physical because both
MESA geometry and the exterior consume it. A stale private-ABI test that used
raw q/key as its reference was corrected to the operator expression; complete
output and composed-VJP tests already exercised the connected behavior.

FLA/Mamba state-recompute precedent informed the checkpoint comparison. The
logical exterior has three microsteps per token, so retaining one BF16 state
before every three logical chunks changes the target cache from 24 MiB to
8 MiB. The output reverse reconstructs the missing zero, one, or two boundaries
with the exact chunk update `S'=decay*S+d_tail^T z_new`; the FP32 resident
reverse state remains a per-logical-chunk temporary. Saved-tensor storage fell
by exactly 16 MiB, and trainer-style CUDA Graph reservation fell from about
216 to 192 MiB. CG5 F+B medians were 1.205 and 1.197--1.206 ms for dense and
coarse checkpoints across the comparison runs, so the coarse native-token
checkpoint is selected without a runtime switch.

A source-to-pair fusion was also implemented rather than accepted by
inspection. Its forward reproduced all former cumulative, pair, scaled-panel,
and value outputs at the declared rounding points. Isolated latency improved
only from 0.136 to 0.123 ms. With a strict recomputation-based mature pair
transpose, complete Graph F+B regressed from 1.205 to 1.253 ms and graph
reservation increased to 240 MiB despite 20 MiB less saved storage. A truly
joint reverse owner must additionally close token phases split by C32 logical
boundaries because `32 mod 3 != 0`; forcing that ownership into the forward
CTA is not a free ABI edit. The fusion and selector were deleted under the
project's complete-F+B selective-fusion rule.

The follow-up upstream audit found no missing mature owner to transplant.
FLA's current `ChunkGatedDeltaProductFunction.backward` explicitly marks its
grouped reverse as TODO, materializes zero-expanded query/output cotangent at
`T*num_householder`, and falls back to generic GatedDeltaRule backward. GDN2
and KDA provide the mature output-owner contractions used elsewhere, but not
the split-token `E=3` source closure. A complete 56 MiB seam deletion would
therefore require a new staged boundary-partial/finalize owner, not an ABI-only
specialization; it is outside this mature-code pass.
## Isolated RLS grouped-owner audit (2026-08-28)

FLA GatedDeltaProduct's grouped state/output forward and its current backward
were re-audited against the isolated RLS direct-`e` exterior. The forward state
owner can compact Householder checkpoints and its output owner loops a static
slot dimension, but the public implementation assumes a shared symmetric
direction and scalar decay. Its backward explicitly marks grouped optimization
TODO, expands query and output cotangent to `T*num_householder`, and calls the
flat GatedDeltaRule reverse. It therefore remains a schedule donor rather than
an exact replacement for independent direct-`e` factors with channel-wise
decay.

Two complete-path A/Bs were rejected. Separating the RLS state and output
owners reduced CG5 CUDA Graph forward/F+B by about `0.032/0.034 ms`, but
retaining every logical chunk boundary added 16 MiB per layer. Retaining the
MESA Hkk/Hkv boundaries instead of recomputing them saved only about
`0.014 ms` F+B for another 16 MiB. The selected coarse checkpoint and replay
remain. Profiling showed the flat RLS state owner at about `0.191 ms` for 96
logical chunks, or `1.99 us/chunk`, versus FLA GDN2 at about `0.0315 ms` for 16
chunks, or `1.97 us/chunk`. C64 reduced the state owner to about `0.153 ms` but
increased pair/WY work, leaving complete F+B unchanged near `1.158 ms`.

The decision is to default the isolated MESA experiment to CG5 and not treat
the remaining gap as a slow CG or an inferior per-chunk state primitive. A
future grouped route must algebraically own a token-level rank-three transition
and its exact transpose. A flat-layout rename, wider chunk, or forward-only
GatedDeltaProduct transplant does not meet that requirement.

## MESA Hkv-to-CG transpose seam (2026-08-28)

FLA MESA's Hkv query transpose and implicit CG transpose were specialized as
one chunk owner in the isolated RLS experiment. The owner forms

\[
\bar q_{\rm CG}=\bar q_{\rm exterior}
 + \bar y H_{kv}^T e^g
 + [((\bar y v^T)\odot\Delta)k],
\]

applies the same low-precision rounding boundary as the former materialized
right-hand side, and consumes it immediately in the fixed five-step transpose
action. The Hkv key/value transpose remains separate. This follows MESA's
staged-owner precedent: fuse a sole-consumer panel, not the entire geometry
reverse.

The connected 17-test forward/state/composed-VJP suite passes. Four target
CUDA Graph measurements gave forward medians `0.467--0.470 ms` and F+B medians
`1.156--1.171 ms`; the earlier selected graph was near `0.476/1.194 ms`.
Reservation stayed at 192 MiB. Profiling attributed about `0.042 ms` to the
new owner and found neither the old Hkv-dq kernel nor standalone CG-transpose
kernel in the graph, so the specialization is retained.

The same audit does not authorize a nominal token-native `E=3` relayout.
FLA GatedDeltaProduct's grouped forward owns shared symmetric Householder
directions and a scalar gate, while its source still marks grouped backward
optimization TODO and expands to the flat GatedDeltaRule reverse. Independent
direct-`e` factors with channel decay require a rank-three block pair/WY
transpose. Merely unrolling three flat chunks preserves their arithmetic and
cannot remove the measured logical-chunk cost.

## Symmetric Hkk private-boundary packing A/B

Research update (2026-08-28): reachable RLS `J/Hkk` is symmetric, so its
private chunk-boundary history was implemented with lower-triangular storage
without changing the full FP32 public initial/final state or the symmetric
cotangent convention. CG forward, the fused Hkv-to-CG transpose, and Hkk
reverse reconstructed logical tiles by mirrored loads. The connected forward,
state, and composed-VJP tests passed, including exact full-state symmetry.

At `B1,T1024,H8,r=128,C32`, scalar lower packing reduced the BF16 Hkk history
from 8 MiB to 4.03 MiB. Its CUDA Graph forward/F+B medians were
`0.472--0.479/1.189--1.204 ms`, versus the selected dense range
`0.467--0.470/1.156--1.171 ms`; graph reservation remained 192 MiB. Two
Tensor-Core-oriented lower-block layouts were also tested. B16 used 4.5 MiB
and measured `0.509/1.254 ms`; B32 used 5 MiB and measured
`0.708/1.248 ms`. Neither recovered dense operand locality.

The mathematical symmetry does not halve the `XJ` action. More importantly,
the resident MESA CG owner loads the dense boundary once and reuses it through
all five iterations, so packing does not remove five HBM reads. Mirrored
unpacking instead perturbs the native dense `tl.dot` operand layout in CG
forward, implicit transpose, and Hkk reverse. The packed candidates were
deleted. Full dense private Hkk remains selected; symmetry is still enforced
at the public continuation state and cotangent boundaries. A future packed
form needs a native symmetric block-action owner, not a gather adapter around
the current dense Tensor-Core action.

The next grouped-exterior study must begin from the RLS source identities, not
from FLA's generic grouped-backward TODO. Let `g=J_t^-1 u`,
`delta=1-u^T g`, `lambda` be geometry decay, and `a` be the associative
diagonal scale moved across the first transport. The connected source producer
has exactly

\[
e_0=-\frac{\lambda}{\delta a}g,\qquad d_1=\gamma g,
\qquad z_0=z_1=0.
\]

Thus `e0` and `d1` share one gain axis (when `gamma=0` both geometry updates
are structurally inactive). The two geometry microsteps compose to an
identity-plus-rank-at-most-two block with no additive RHS. They are generically
rank two, because the remaining `d0` direction is `u` and the remaining `e1`
direction is the RLS residual; the ordinary key edit adds a generally
independent third direction and the complete transition is generically rank
three. Therefore an exact simplification should target a native
"rank-two multiplicative geometry plus rank-one additive edit" block-WY and
its transpose. It must exploit the shared gain Gram and the two zero-value
slots; it must not claim a generic rank-one collapse or merely rename the flat
`3T` implementation.

## Token-block E=3 compact value RHS

The follow-up derivation uses FLA generalized-DPLR's residual equation but
changes its parallel axes. FLA TileLang `chunk_A` supplies the pair owner,
FLA's C64 WY path supplies the recursive two-block solve/merge identity,
GatedDeltaProduct supplies token-grouped state/output ownership, and Mamba-3
supplies the value-tile reverse precedent. No upstream implementation was
found that combines independent direct-`e` rank-three factors, channel decay,
one additive slot, and the strict grouped transpose.

For a C-token chunk, arrange `[E0,E1,E2,Q]` as `4C` GEMM rows and
`[D0,D1,D2]` as `3C` rows. One `4C x r @ r x 3C` contraction produces all WY
and read interactions. The unit-lower 3x3 diagonal block for each token has an
analytic FP32 inverse. Left preconditioning preserves the fixed injection
`P[3t+2,t]=1`, because the final column of every unit-lower inverse is the
third basis vector.

The two zero geometry values then give a stronger exact specialization than
a grouped layout alone. Instead of expanding compact values to `[3T,V]`, solve

\[
[Y,R]=W^{-1}[E,P].
\]

For compact `Z in R^(C x V)`, the logical residual is `RZ-YS0`. Hence

\[
O=(Q-AY)S_0+(AR)Z,
\]

\[
S_1=\Lambda S_0-D_{tail}^T(YS_0)+(D_{tail}^TR)Z.
\]

The transpose uses one solve `W^-T[bar Y,bar R]`, the standard implicit solve
cotangent `bar W=-bar B X^T`, the analytic 3x3 inverse transpose, and the two
standard pair GEMMs. FP64 explicit recurrence and VJP checks agree at
approximately `1e-15`; the local precondition transpose agrees at
approximately `1e-17`.

The initial design proposed a token-recursive block TRSM with independently
tiled RHS columns rather than padding C48 to C64. That proposal and its exact
transpose contract are recorded in `experiments/rls/BLOCK_E3_DESIGN.md`; the
measured implementation outcome and the reason it used a different C48
composition are recorded below.

## Token-block E3 implementation result

Research update (2026-08-28): the isolated RLS token-block E3 exterior was
implemented end to end to test whether compacting the two zero-value geometry
slots could remove the flat `3T` cost. FLA generalized-DPLR supplied the pair
tile ownership and implicit-WY formulas; FLA's C64 fast-WY supplied the
three-block inverse composition after TileLang SM120 rejected a monolithic
C48 fragment; Mamba-3 and FLA state/output owners supplied the resident value
tile and transpose structure. The upstream schedules were specialized to
independent direct-`e` factors, channel decay, and one compact additive slot;
their full ABIs were not copied.

The mathematical compact identities and their FP64 reverse close exactly.
The measured implementation did not use the initially proposed token-recursive
TRSM. It inverted three C16 diagonal blocks, formed the three off-diagonal
inverse blocks with Tensor Core GEMMs, stored the C48 FP32 inverse, gathered
`W^-1 P`, and used an implicit transpose solve for both RHS groups.

At the target BF16 shape, five interleaved trainer-style CUDA Graph rounds
measured flat RLS/token-block E3/GDN2 medians of
`0.489/0.388/0.129 ms` forward and `1.195/1.763/0.470 ms` F+B. Graph-resident
allocation was `23/150/18 MiB`, respectively. The compact forward win did not
survive reverse: the custom path used 101 rather than 55 launches and its
eager kernel sum was `1.819` rather than `1.220 ms`. It also left an
approximately 8% cancellation-amplified query-source VJP error despite an
FP32 residual contraction.

Decision: reject this implementation. Do not promote the token-block exterior,
weaken its VJP gate, or replace the selected flat MESA CG5 experiment. The
negative result shows that compact value algebra alone is insufficient; a
future attempt would need one mature output-owned reverse that consumes the
query residual directly, not a chain of generic BMM/copy/add cotangents.

### Output-owned reverse revisit

The required revisit was completed without changing the token-block forward.
FLA DPLR/GDN2 ownership now supplies a resident state transpose, an
output/WY owner and a pair/source owner that closes the gauge, pair GEMM
transpose, source algebra, L2 transpose, and gate suffix. The former generic
statistics/WY reverse and Python `grad_d/grad_paired` handoff are deleted. The
selected FLA-style split retains private FP32 solve/read/tail/query cotangent
panels between those two owners; they are not a public ABI and measurably
shorten live ranges.

This materially changes the earlier numerical conclusion: FP16 and BF16
composed VJPs now pass, including the query source. The full isolated suite is
`21/21`. A target 50-warmup, 200-repeat CUDA Graph measurement with
`gain_chunk=32` was `0.384/1.275 ms` forward/F+B versus matched GDN2
`0.122/0.466 ms`; graph allocation/reservation was `86/246 MiB`. Historical
eager attribution measured pair/output/state reverse at about
`0.358/0.146/0.152 ms`.

A post-recovery rewrite attempted to replace the private-panel boundary with
compact `barW/barA/direct_stats` owners. It measured `0.391/1.311 ms` under
the same Graph configuration: output plus interaction rose to about
`0.212+0.031 ms`, more than the pair reduction to `0.287 ms` recovered. That
rewrite was removed. The retained donor specialization repairs the prior
reverse and cuts its original `1.763 ms` F+B substantially, but does not beat
the retained flat MESA/FLA RLS route. The adoption decision remains unchanged
for performance and memory reasons, not numerical failure.

### FLA streaming-reverse transplant A/B

FLA's newer MIT-licensed generalized-DPLR TileLang backend was audited after
the output-owned rebuild. In particular, `chunk_stream_bwd.py` supplies a
sequence/head owner with resident FP32 state cotangent, high/mid/low shared-
memory schedules, and a 256-thread `K=V=128` specialization; `wy_fast_bwd.py`
and `chunk_A_bwd.py` supply caller-owned WY and q-side transpose stages. This
is a materially better donor than the older flat DPLR backward, but it does
not imply that sequence/head streaming is the right parallel axis for E3.

Two exact E3 specializations were implemented and removed after measurement.
The first fused the resident state scan, compact residual, output, WY RHS, and
pair transpose. It eliminated the approximately 52 MiB
`grad_state/residual/grad_residual` interface but measured about `2.562 ms`
complete Graph F+B and exceeded the intended fragment live set. The second
kept a low-resource resident state/value owner and a separate chunk-local
WY/pair finish owner. It reduced the broad fragment set substantially, but
measured `2.394--3.095 ms` complete Graph F+B in non-interleaved trials.

Eager attribution identified the structural cause. The retained owner split
measured state/output reverse at approximately `0.148/0.143 ms`; the staged
streaming replacement measured `0.788/0.082 ms`. At `B=1,H=8`, a
sequence/head stream has only eight long-lived CTAs traversing 64 token
chunks. The retained split exposes roughly 64 state CTAs and thousands of
chunk/rank output CTAs. Removing HBM traffic therefore traded away far more
GPU parallelism than it saved. The private FP32 panel boundary is not merely
legacy glue at this workload; it preserves the profitable parallel axes.

An `r=128,T=16` same-forward comparison found zero output difference and
roughly `0--0.45%` relative differences between the two low-precision VJPs,
so the rejection is scheduling-driven rather than a failed transpose
derivation. A TileLang warning also showed that the staged scalar reduction
needed a less conditional barrier schedule before it could be production
safe. The experimental source, backend selector, and tests were deleted.
Future work must retain chunk parallelism, for example through a proven
segmented reverse-scan owner; simply transplanting FLA's sequence/head stream
or deleting the private panels is now a measured rejected direction.

### Block-E3 selection over the flat exterior

The isolated RLS candidate subsequently selected block-E3 as its sole
exterior. This supersedes the earlier adoption decision but not its measured
evidence. A final target comparison with each route's best retained chunk width
measured flat C32 at `0.480/1.200 ms` forward/F+B and block-E3 C16 at
`0.390/1.278 ms`; graph allocation/reservation was `23/192 MiB` and
`86/246 MiB`, respectively. Block-E3 is therefore selected for its native
token/slot algebra and approximately 19 percent forward win, while its current
6.5 percent F+B and 63 MiB allocation regressions remain explicit limitations.

The flat `3T` source packer, expanded state/output and reverse programs, tests,
and runtime selector were deleted. The current code retains the FLA/GDN2-style
output-owned reverse and its private FP32 cotangent panels. Future work targets
a segmented reverse owner that preserves chunk/rank parallelism; it does not
restore the flat path as a compatibility fallback.

### Reverse lifetime and shared-memory specialization

The next pass audited FLA common's parallel and split state scans before
changing the retained E3 reverse. Those scans reduce independent chunk
contributions cheaply when cross-chunk propagation is diagonal or additive.
The exact E3 cotangent transition is instead

\[
\bar S_c=(\operatorname{Diag}\Lambda_c-Y_c^TD_c^{tail})\bar S_{c+1}+B_c.
\]

Composing these low-rank updates across a segment produces a dense or
growing-rank map. A segmented implementation would therefore need a dense
transition workspace, a full segment replay, or a duplicated transition pass.
At the target the retained rank/value-tiled state owner already exposes 64
CTAs and costs only about `0.15--0.18 ms`; the upstream split pattern is not a
drop-in improvement and was not imitated with a lower-quality custom scan.

The profitable specialization was inside the existing FLA-style pair/source
owner. Its first version retained complete `left` and `right` 48x128 panels
together and compiled to about 56 KiB dynamic shared memory, limiting the
target to one CTA/SM. The selected version reloads one 16-row bounded operand
tile at each pair consumer, preserves FP32 Tensor-Core accumulation and the
same output ownership, and compiles to about 40 KiB. This permits two CTAs/SM
without adding an HBM interface. Keeping one complete operand at a time used
about 53 KiB and lost the complete-path A/B, so the additional tile reloads
are intentional.

The output-to-source boundary was also reduced algebraically. The full
channelwise query-gate cotangent satisfies

\[
\bar G^{query}_{t}=q^{paired}_{t}\odot\bar q_t
+\mathbf 1_{t=C-1}\,\bar G^{tail},
\]

so only the final `r`-vector tail seed crosses the boundary; the pair/source
owner reconstructs the first term from its already consumed query panel and
query cotangent. This removes the FP32 `[panel,C,r]` transfer without another
kernel or a reduced-precision partial. Target Graph F+B stabilized at about
`1.17 ms`, versus about `1.26--1.28 ms` before this lifetime/occupancy pass,
and reservation fell from 246 to 222 MiB while forward remained about
`0.388 ms`.

### FLA precision and pending-owner audit

FLA `main` was refreshed from `88caebb` to `f4cda48` on 2026-08-28. The two
new mainline commits affect Parallax caching and model loss logits only; the
installed FLA 0.6.0 copies of MESA, GDN2, common state/output, L2Norm, and
their TileLang backends are byte-identical to refreshed main. Upgrading the
package therefore cannot change the current RLS hot path.

Open FLA PR #1152, "Prevent NaN failure in MesaNet", supplied two relevant
precision findings. Its FP32 `Hkk` history was A/B tested earlier and rejected
for this target because geometry forward rose from about `0.186` to
`0.322 ms` for only about `1e-3` relative improvement. Its FP32 `Hkv*dHkv`
scalar reduction is retained, and its BF16 `Hkv` range fix is now applied to
the diagnostic FP16 route: the unnormalized cross-moment has no valid FP16
range proof. Public BF16 execution is unchanged.

FLA PR #948 reports a Blackwell-specific failure in a different common
TileLang GDN backward and an IEEE-safe output dot. The native E3 TileLang
owners were not switched to CUDA-core IEEE dots by analogy. Three independent
`r=128,C=16` FP64-oracle seeds on the local SM120 device found BF16 forward
relative error `0.47--0.52%` and no sparse large-error class; composed input
VJPs were mostly `0.35--0.65%`, with decay/strength scalar branches reaching
about `2.0--2.1%`. Tensor-Core execution remains selected until a matching
failure is reproduced.

FLA PR #1128's KDA TileLang training owners are credible future donors for
the two remaining exterior hot blocks. Its `dAv` owner streams value tiles
while retaining one FP32 interaction accumulator; its fused `dq/kg` owner
keeps the K tile internal and closes final outputs once. Those are the same
ownership classes as E3 output/WY reverse and pair/source reverse. They are
not drop-in kernels: the promoted upstream bucket is C32/C64 with ordinary KDA
axes, while this path has a C16 token axis, a 48-row E3 logical axis, paired
direct-e gauge terms, and RLS source scalar cotangents. Adopting the schedules
requires an E3 specialization and a complete composed-VJP A/B, not a package
upgrade or ABI wrapper.

The precision audit also removed two local inconsistencies. L2Norm reverse now
consumes the exact rounded normalized q/key panels saved by forward, matching
FLA's `l2norm_bwd`; it no longer reconstructs an unrounded normalized vector
from raw operands. The C48 WY inverse is computed in FP32 inside its owner but
stored directly in the BF16/FP16 consumer dtype. Previously the FP32 HBM panel
was immediately rounded by every forward and reverse MMA consumer, so it added
traffic without changing operand bits. The BF16 target saved-tensor total is
now `77.47 MiB`; the inverse is `2.25 MiB` rather than `4.5 MiB`.

The private state-checkpoint cotangent was separately promoted to FP32 for an
A/B rather than inferred from policy. It raised complete Graph F+B from about
`1.10` to `1.13 ms` and reservation from 222 to 234 MiB, while five-seed
per-input VJP errors were effectively unchanged. The isolated RLS experiment
therefore keeps that cotangent in checkpoint dtype. This is an experimental
precision-map result, not authorization to weaken the production
FP32-backward-partial contract without an explicit contract revision.

### Scalar precision and gate-cumsum ownership

The RLS scalar audit followed FLA's actual mixed-precision execution rather
than treating every scalar as FP32 by category. FLA's chunk gate cumsums load
the raw log gate, accumulate the token axis in FP32, and keep the cumsum owner
separate from the matrix owners. The RLS specialization preserves that
schedule but accepts an additional FP32 token scalar `diagonal_log[B,T,H]` and
broadcasts it in registers before `tl.cumsum`. This replaces the former Python
ABI that expanded the scalar across rank, added it to the raw channel gate,
and materialized a 4 MiB FP32 `[B,T,H,r]` input for the same cumsum. The design
influence is FLA common's chunk-local vector cumsum and persistent autotune
cache; the operator-specific scalar input and layout are local adaptations.

Lowering scalar precision was not adopted merely to reduce the visible FP32
count. `1-u^Tg` has no static lower bound and feeds repeated divisions; the
effective-mass scan owns recurrent-split semantics; CG dot products inherit
the solve condition number. Conversely, the bounded source scalars were tested
rather than assumed to need FP32. Fixed FP16 rounding of geometry decay, mass
scale, and the diagonal factor missed the current composed-VJP gate and did
not change the generated reduction/divide/exp path into Tensor Core work.
Rounding CG `alpha/beta` to BF16 likewise slowed complete F+B by about
`20--25 us`. These A/B results distinguish precision that protects the
mathematics from FP32 storage or broadcasting that merely reflects a poor ABI.

### RLS production promotion

The MESA-CG5 plus token-native block-E3 route was promoted from an isolated
study to the sole SolveDelta production operator on 2026-08-28. The exact
bounded-LDU route remains recoverable at Git commit `2237875`; none of its
private frame, chart, descriptor, or validation ABIs remain production
dependencies.

The selected configuration is fixed to prior mass 2, C32 paired MESA geometry,
five CG iterations, and C16 exterior chunks. The choice accepts lower model
expressivity than the archived full-rank bounded chart in exchange for a fresh
contiguous-core CUDA Graph median/p95 of `0.371/0.402 ms` forward and
`1.103/1.320 ms` F+B at `B1,T1024,H8,r=V=128`. Matched GDN2 was approximately
`0.128/0.467 ms`. The full SolveDelta layer, including projections, conv4,
packed canonicalization, output projection, and parameter gradients, measured
`0.649/0.854 ms` forward and `2.329/2.624 ms` F+B.

Promotion exposed that the experiment's contiguous input assumption was not a
valid public ABI: TileLang reverse rejected the fused projection's row stride,
and forward pointer arithmetic had the same packed assumption. Production now
owns one explicit contiguous-vector boundary before MESA/E3. This is a
correctness requirement, not a claimed final schedule. Removing it requires
native stride-aware loads in every corresponding forward and transpose owner.
