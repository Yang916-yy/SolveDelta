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
