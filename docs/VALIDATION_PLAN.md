# SolveDelta Validation Plan

Validation follows the dependency graph of the single canonical model. The
FP64 token recurrence owns numerical truth; optimized kernels do not define
semantics.

## Phase 0: freeze the token contract

Implement one slow, explicit SolveDelta recurrence and freeze:

- all tensor shapes and state orientations;
- key-width resolution: consume one resolved positive per-key-head `d_k`, set
  `r=d_k`, reject nondivisible total key widths, and expose no independent rank
  override;
- projections, key normalization, and gate parameterizations;
- state initialization, one geometry update, one decay, ordered `K` edits,
  and read-after-final-write behavior;
- masking, reset, continuation, and packed-sequence semantics;
- recurrent-state dtype and gradient/detach behavior;
- system-coordinate, triangular-solve, and dual-action conventions.

Verify the geometry and associative decays against their FP32 log-space
definitions, including long chunks whose ordinary cumulative products
underflow in low precision. This is the same numerical boundary handled by
mature Gated Delta kernels, not a new model variant.

Use deterministic FP64 tests covering zero state, zero projected geometry,
query, and key vectors, repeated keys, orthogonal keys, weak and strong
geometry, short sequences, and sequences longer than `r`.
Include both direct-`head_k_dim` and total-key-width frontend resolution, with
at least one non-128 `d_k`. Perturbing future tokens must not affect earlier
outputs or states. Whole-sequence execution must equal arbitrary recurrent
splits.

Required algebraic checks include

\[
\|N_t^\pm\|_F<c,
\qquad
|\delta_{t,i}|<s_{\max},
\]

\[
M_t=(I+N_t^-)\Sigma_t(I+N_t^+),
\qquad
P_t=M_t^{-1},
\qquad
P_t^{-T}=M_t^T,
\]

\[
(P_t^{-T}b)^T(P_ta)=b^Ta.
\]

Check both moment-specific strict-triangular radial maps and both diagonal
maps:

\[
\mathcal B_{c_H}(Y)=Y+O(\|Y\|_F^3),
\qquad
\mathcal B_{c_R}(Y)=Y+O(\|Y\|_F^3),
\qquad
s_i\tanh(x/s_i)=x+O(x^3),\quad i\in\{H,R\}
\]

at the identity point, together with the declared analytic bounds on `M`, `P`,
`P^-T`, and `cond(M)`. Numerically verify that the chart Jacobian with respect
to an unconstrained ambient `X^(R)` coordinate, and hence the joint
`(X^(H),X^(R)) -> M` chart Jacobian, has rank `r^2` at zero. Do not compute
this as `dM/dR` under the structural `gamma_g=0` reduction, which is
deliberately zero. Separately, at a fixed finite nonzero `gamma_g`, measure the
reachable Jacobian from feasible `(u_1,h_1),...,(u_t,h_t)` prefixes at
`t < r`, `t = r`, and `t > r`; verify the short-prefix rank/span restriction
and use spanning versus rank-deficient `u` histories. Compare the direct
driven moment `D_t` with an unfused `C_t W_drive` construction in FP64 forward
and gradients.

Add a non-collapse witness. Construct two feasible prefixes with the same
`m` and the same `H+R` but different decompositions, and require the separate
nonlinear chart to produce different `M`. At `r=2`, one diagonal witness is

\[
(H_1,R_1)=(\operatorname{diag}(1,0),0),
\]

\[
(H_2,R_2)=(\operatorname{diag}(3/4,1/4),
            \operatorname{diag}(1/4,-1/4)).
\]

Both have `H+R=diag(1,0)` and are realizable from normalized `u` with suitable
drives, but the two radius-`1/8` diagonal maps differ. Also test the deliberate
collapsed baseline that accumulates `D+J`; it must identify this pair.

For every edit slot check

\[
e_{t,j}^Td_{t,j}
=\sum_i b_{t,j,i}k_{t,j,i}^2\in[0,2),
\]

for finite gate logits, with the upper endpoint reserved for the declared
closure. A zero normalized key must give `d=e=0`, exact pairing zero, and an
identity edit.

Check both unit-triangular solve residuals, finite forward values, and finite input/parameter
gradients over the declared training envelope. The pairing result is an
eigenvalue certificate; tests must not relabel it as a Euclidean norm bound.

## Native BF16/FP32 precision contract

The first advertised training path is mixed precision, not an FP32 kernel with
a low-precision output wrapper:

| Quantity | Required native representation |
|---|---|
| projected activations; conv4 inputs/outputs; normalized `u,q,k`; edit values and bounded erase/write operands | BF16 |
| Tensor Core matrix operands, including a factor tile when an action packs one | BF16, packed once at its declared contraction boundary |
| dot/GEMM/TRSM partials and backward partials | FP32 accumulation; reduced-precision accumulation is forbidden |
| log-decay, `exp`/`softplus`/`sigmoid`, norm and radial reductions, geometry strength, and sensitive scalar reverse | FP32 |
| continuation and chunk-boundary `m,J,D,S` | FP32, with no BF16 round trip between chunks or recurrent calls |
| public token activations, `d,e,chi`, and returned activation gradients | BF16; gradients are cast only after their FP32 reduction is complete |

The FP64 token recurrence remains the sole mathematical oracle. Generate
deterministic master tensors, round every BF16 runtime operand once, promote
those exact BF16 values to FP64, and run the oracle at that point. FP32-defined
preprocessing is evaluated from the same quantized logits or vectors on both
paths; the resulting FP32 values are promoted without another BF16 rounding.
The native path receives the same BF16 operands and FP32 scalars/states. Apply
the same rule to upstream cotangents: public activation cotangents are rounded
once to BF16 and state/scalar cotangents remain FP32. The VJP target is the
continuous FP64 operator at the quantized point, not the derivative of the
master-to-BF16 rounding operation.

Normalization is an explicit mixed boundary: reduce the squared norm in FP32
and round the normalized vector once to BF16. Gate nonlinearities and
log-decay remain FP32 even though their projected logits came from BF16 model
activations. If a Tensor Core action packs LDU factors to BF16, primal and
transpose-dual actions must consume exactly the same packed bits. A separate
same-packed-factor oracle isolates action arithmetic from factor representation
error. Rounding `J`, `D`, or `S` at a chunk boundary would instead create a
split-dependent rounded recurrence and is outside this contract.

For any reference tensor `x` and native tensor `x_hat`, report

\[
\rho(x,\widehat x)=
\frac{\operatorname{RMS}(\widehat x-x)}
{\operatorname{RMS}(x)+10^{-8}},
\qquad
a_\infty(x,\widehat x)=\|\widehat x-x\|_\infty.
\]

A FP32 internal tensor passes when `a_inf <= 1e-6` or its declared `rho`
ceiling passes. A BF16 public or packed tensor uses `a_inf <= 2e-4` instead.
The absolute branch handles exact or near-zero references; it is not added to
the relative allowance. NaN or infinity always fails. Tests must log both
metrics even when the absolute branch passes. No warning-only gradient class,
CI relaxation, sequence-length multiplier, cancellation multiplier, or
architecture-specific tolerance is allowed.

The pure FP64 reference, algebraic reductions, and equivalent unfused reference
expressions use `rtol=1e-10, atol=1e-12`. For a single contraction with the
same packed operands define

\[
\tau(A,B,\widehat C)=
\frac{\|\widehat C-AB\|_F}
{\|A\|_F\|B\|_F+10^{-12}}.
\]

The mixed implementation must pass all of these internal budgets before the
looser composed-layer budget is considered:

| Boundary | Forward ceiling | Backward ceiling | Additional gate |
|---|---:|---:|---|
| one resident BF16-operand/FP32-accumulated tile result against FP64 of the same packed operands | `tau <= 2e-4` | `tau <= 2e-4` | measure before any BF16 output store; Tensor Core and FP32 accumulator modes must be explicit where claimed |
| resident FP32 fixed-tree/twofold long triangular reduction of the same operands | `tau <= 5e-4` | `tau <= 5e-4` | deterministic order; no data-dependent path |
| FP32 chunk-boundary `m,J,D,H,R` against the quantized-input FP64 oracle | `rho <= 5e-3` | `rho <= 5e-3` | every boundary and final state; initial-state VJP additionally `rho <= 5e-4` |
| FP32 radial scalars and pre-pack chart, given the same FP32 moments | `rho <= 1e-3` | `rho <= 3e-3` | analytic chart bounds and nonnegative reconstructed norm must pass |
| BF16 packed factors against the exact FP64 chart | `rho <= 6e-3` | `rho <= 1e-2` | identity, strict masks, and unit diagonal remain exact |
| FP32-accumulated action, given the same BF16 factors and BF16 RHS | `rho <= 5e-3` | `rho <= 1e-2` | each RHS separately; every triangular `eta <= 2e-5` |
| composed BF16 `d,e,chi` from prefix inputs | `rho <= 6e-3` | use the end-to-end VJP classes below | pre-cast `pi <= 5e-4`; BF16 `pi <= 8e-3` |

If the action packs factor coordinates to BF16, round-to-nearest with
`u_b=2^-8` gives the certified packed bounds

\[
\|N_H^\pm\|_F,\|N_R^\pm\|_F
\le (1+u_b)/8=0.12548828125,
\qquad
\|N^\pm\|_F\le0.2509765625,
\]

and diagonal entries in `[0.7757586, 1.2890411]`; the resulting declared
`cond_2(M)` ceiling is `4.635`. A frame that keeps factors FP32 retains the
exact `1/8`, `1/4`, and original condition bounds. Both forms keep the identity
point exact.

The geometry row is calibrated for the actual Tensor Core scan rather than an
ideal FP32 outer product. On the local SM120 GPU, BF16 operands with FP32
boundaries gave about `0.7e-3--1.46e-3` `J` error and
`1.97e-3--2.32e-3` `D` error for `T=130,1024,8192`; BF16 leaf VJPs were about
`1.54e-3--1.67e-3`, while FP32 initial-state VJPs stayed below `8.1e-7`.
A mixed-frame proxy with FP32 chart construction, one BF16 factor pack, and
FP32 action accumulation produced factor/action error below `1.9e-3`, raw VJP
error below `3.0e-3`, and simulated BF16-leaf VJP error below `5.1e-3` at
`r=128`. These probes justified the frozen internal orders but do not
substitute for the selected resident-operator tests.

The standalone FP32 MathDx oracle retains its independent tighter budgets:

| Exact-oracle boundary, given the same FP32 factors | Forward `rho` | Backward `rho` | Additional gate |
|---|---:|---:|---|
| MathDx primal triangular/complete solve | `5e-5` | `2e-4` | normalized residual `eta <= 2e-5` for each triangular solve |
| direct dual products | `5e-5` | `2e-4` | compare each packed right-hand side separately |
| composed FP32 `d,e,chi` | `5e-4` | `1e-3` | normalized pairing drift `pi <= 5e-5` |

The standalone exact MathDx oracle validates both `K=1` and `K=2`. Its current
device-library specialization has `nrhs=2`; K1 pads the second primal RHS with
exact zeros and slices it after the solve. This is an equivalence check, not a
claim that the padded kernel has native one-RHS performance.

A local feasibility probe generated 64 bounded `r=128` LDU systems spanning
weak through saturated chart coordinates with five right-hand sides. PyTorch
FP32 factorwise execution against FP64 produced about `1.0e-7` primal and
`2.1e-7` dual `rho`, at most `4.2e-8` normalized triangular residual,
`3.4e-8` normalized pairing drift, and `1.0e-7--2.2e-7` backward `rho`. This
supports the order of the budgets but does not validate MathDx; the actual
custom operator must pass them independently.

The chunk-owned frame path must be checked before the FLA exterior. Deep
cancellation remains a production requirement because BF16 products can carry
more residual bits than one BF16 input. For example, the exactly representable
BF16 pair `(0.6953125, 0.71875)` has product
`0.499755859375 = 0.5 - 2^-12`. Combining that local product with the FP32
boundary entry `-0.5` gives a real post-quantization cancellation ratio of
`4095`; it is not a residual that disappeared when inputs were rounded.

For every cancellation fixture, first quantize the runtime operands, then
promote them and recompute

\[
\kappa=
\frac{|\alpha|\|B\|_F+\sum_s |w_s|\|L_s\|_F}
{\|\alpha B+\sum_s w_sL_s\|_F+10^{-30}}.
\]

Only then assign it to one of the fixed bins `exact zero`, `[1,16]`,
`[2^7,2^8]`, or `[2^11,2^12]`. Cover `J` and `D`, lower and upper coordinates,
and an ordinary asymmetric tail. The deepest bin uses `rho <= 2e-3` for the
radial forward and `rho <= 5e-3` for its VJP, without multiplying either bound
by `kappa`; exact zero must emit an exact zero radial component. A legal PSD
`J` boundary is required for the production witness. The former
`4096 + (-4096 + 0.01)` FP32 fixture is not a BF16 gate because `0.01` is lost
before execution.

The frame must remain finite without a norm clamp, preserve valid tokens when
affine coefficients underflow to zero, emit exact identity-chart outputs for
structurally invalid tail slots, and retain the valid diagonal auxiliaries
required by the strength VJP at identity. A negative reconstructed squared
norm, NaN, infinity, a data-dependent precision fallback, or inferring
validity from an underflowed coefficient is an unconditional failure. If a
Tensor Core tile consumes a large FP32 boundary contribution, it must keep the
sensitive residual in FP32 or use a fixed high/low representation; directly
rounding that boundary to BF16 is not allowed.

The shared geometry strength ties six chart-channel cotangents by the fixed
linear map `g=1^T g_6`. In the declared deepest-cancellation fixtures only,
report

\[
\rho_{\mathrm{tie}}
=\frac{|\widehat g-\mathbf 1^Tg_6|}
{\sqrt{6}\,\|g_6\|_2+10^{-8}}
\le 2.5\times10^{-2}.
\]

The denominator is the induced scale `||1||_2 ||g_6||_2` of the six-to-one
tying map. It is fixed by parameter sharing, not multiplied by the observed
cancellation ratio. The fixture passes when `a_inf <= 1e-6` or the displayed
`rho_tie` ceiling passes; the absolute branch is reserved for a near-zero
six-channel reference norm and is not added to the relative allowance.
All ordinary geometry-strength fixtures continue to use the standard
total-gradient metric and ceiling. An FP64 final scalar addition does not
resolve the observed discrepancy and is not a substitute for this test.

No standalone polynomial or truncated inverse is a validation boundary. The
frame forward and VJP are compared directly with the quantized-input FP64
recurrence and the optional exact MathDx factor-action oracle. A proposed
approximation remains an experiment until its algebraic error and empirical
envelope are separately accepted.

Here

\[
\eta(A,X,B)=
\frac{\|AX-B\|_F}
{\|A\|_2\|X\|_F+\|B\|_F+10^{-12}},
\]

and for every primal/dual pair

\[
\pi=
\frac{|e^Td-\bar b^Ta|}
{\|e\|_2\|d\|_2+\|\bar b\|_2\|a\|_2+10^{-12}}.
\]

The complete Triton--CUDA--FLA layer has one advertised runtime row:

| Advertised runtime path | BF16 outputs and FP32 `S` state | `q,k,v,S0,Cq0,Ck0,Cv0` gradients | `u,h,J0,D0` and gate/geometry parameter gradients |
|---|---:|---:|---:|
| native `r=128,K=1,C=32`, BF16 operands/factors and FP32 accumulation/state/scalars | `6e-3` | `1.5e-2` | `2.5e-2` |

`S0` includes the associative initial state. `J0,D0` include any exposed
geometry continuation state. `Cq0,Ck0,Cv0` are the raw projected-input conv4
caches; their final-state cotangents must reach both the initial caches and the
four contributing projected tokens. All gate logits and static gate parameters are in
the last gradient class. Geometry boundary states must additionally satisfy
their stricter internal row above; the end-to-end state allowance cannot hide
a failed scan, chart, or action. FP32 and FP16 execution may remain reference
diagnostics, but they are not alternate advertised training contracts.

The end-to-end row was calibrated against both the installed mature GDN2 BF16
path and an earlier SolveDelta staging composition on the local SM120 GPU.
For `B=1,T=128,H=2,d_k=d_v=64`, GDN2 against its quantized-input FP64 naive
recurrence produced about `3.19e-3` output and `2.23e-3` final-state `rho`, with
gradients below `4.5e-3`. The earlier SolveDelta path with an FP32 frame and WY
exterior rounded to BF16 produced about `5.34e-3` output, `4.45e-3` final-state,
`1.25e-2` main-vector VJP, and `2.25e-2` geometry-strength VJP in the frozen
cross-chunk probes. Those historical measurements justify the envelope but do
not validate the current `r=128` resident path or grant a further tolerance
increase.

Use the same random upstream cotangents for FP64 and optimized paths and compare
vector-Jacobian products, including losses on both token outputs and returned
states. The reference/unfused contract covers `K=1,2,4`, non-128 widths, masks,
resets, and arbitrary recurrent splits. The current native gate covers only
`r=128,K=1,C=32`, with identity and active geometry; weak, ordinary, and
near-boundary gates; zero and nonzero initial states; lengths straddling 32 and
multiple chunks; `t<r`, `t=r`, and `t>r`; convolution enabled and structurally
disabled; and nonzero initial convolution caches with losses on returned
caches. Native masks, resets, packed sequences, other edit counts, and other
ranks must fail explicitly until implemented and validated. Tolerances do not
grow with `T`.

The first declared long-prefix envelope is training at
`T in {1,31,32,33,1024,8192}` and no-grad recurrent decode through `T=65536`.
It covers `lambda=1`, effective memories of 32, 512, and 4096 tokens,
alternating-sign `D`, repeated and orthogonal `u`, log-decays `-110` and
`-1000`, irregular resets, and arbitrary recurrent splits. Compare every raw
FP32 boundary `m,J,D,S` and normalized `H,R`, not only the final state. The
ceilings do not grow with length.

These numbers are release ceilings, not expected errors. They may be tightened
after implementation evidence. Loosening one requires a reproduced
normal-envelope failure, localization to a named boundary, comparison with the
GDN2 baseline on the same hardware and BF16/FP32 contract, and an explicit
contract revision; a performance gain alone is insufficient.

## Phase 1: exact Delta-family reductions

Set `num_edits = 1`, `gamma_g = 0`, and verify that
`X^(H) = X^(R) = N^- = N^+ = 0`, `Sigma = M = P = I`.
Compare every output, associative intermediate/final state `S`, shared
convolution cache, shared-input gradient, and shared-parameter gradient with
the official GDN2 naive recurrence. The additional `(m,J,D)` cache continues
to update but must have zero influence on that common observable projection;
do not compare losses placed directly on states the baseline does not own.
Apply the published gate ties and repeat for KDA and GDN. Structurally disable
associative decay for ordinary DeltaNet. No sigmoid endpoint or limiting-logit
argument is allowed in an exact reduction.

At identity geometry with associative decay structurally disabled and
symmetric edits, compare `K = 1, 2, 4` with the corresponding official ungated
DeltaProduct-`K` naive recurrence, including the finite negative-eigenvalue
range. A gated DeltaProduct comparison instead retains matched decay.
These are reductions, not maintained models.

Required strictness witnesses include:

- a prefix-collision pair with identical current/local tokens but different
  old geometry and therefore different edit coefficients;
- a moment-collision pair with different ordered prefixes but identical
  `(m,J,D)`, which must produce the same solve adapter while their associative
  states may differ; label this a geometry collision, not a complete-state or
  SolveDelta-specific collision;
- a complementary fixed-state comparison showing the same general compression
  principle in the identity-geometry Delta baseline, without claiming that its
  recursive state is merely a second-order moment;
- a two-edit finite planar rotation-contraction with nonreal conjugate
  eigenvalues, unavailable to any single rank-one transition; any exact
  orthogonal-rotation or Householder check must be labeled a `beta=2` closure
  diagnostic rather than a finite exact reduction.

Also construct a legal solve-domain pair `(a,bar_b)` using nonconstant positive
`beta` with `bar_b=beta* a`. Verify the coordinatewise cone
`a_i bar_b_i>=0`, exact support inclusion, and, for `r>=2`, a negative
eigenvalue of the symmetric part despite `a^T bar_b>=0`. Then use a nontrivial
bounded frame to show that ambient `(d,e)` may have a negative coordinatewise
product while preserving `e^T d=bar_b^T a`. This prevents the pairing
certificate from being mislabeled as dissipativity and directly witnesses
strict capacity beyond identity-geometry GDN2.

## Phase 2: LSSO provenance and causal-frame property contract

Disable exponential forgetting and Rank-Rotary. For every prefix, compare the
SolveDelta moment construction with the current upstream LSSO derivation on the
same projected features where the quantities have the same meaning:

- normalized `H_t` and the unfused/fused cross drive;
- fixed-size reconstruction of the full-prefix Gram and driven moment;
- forward values and gradients for those inherited moment subcomputations.

These comparisons document the derivation; they do not force the causal
adapter to retain the original chart. Independently test the causal frame for
an exact identity point, invertibility,
inverse-transpose duality, pairing preservation, normal-envelope primal/dual
bounds, similarity amplification, and full `r^2` ambient-chart local
differential rank. Forward and gradient comparisons are made against the
SolveDelta token oracle.

Cover `t < r`, `t = r`, and `t > r`, including repeated and rank-deficient
features. For `t < r`, explicitly report that `rank(D_t), rank(R_t) <= t` and
do not expect the feasible-prefix Jacobian to equal the ambient chart rank.
For `t >= r`, include both a spanning history that exposes all ambient `R`
directions through drive perturbations and a rank-deficient counterexample.
Compare solves, not explicit inverses. This phase establishes provenance only;
Prefix-LSSO is not another architecture or a chart constraint in this
repository.

Rank-Rotary is outside the initial contract. It may enter only through a
contract revision and a proved/tested moment-transport rule.

## Phase 3: exact chunkwise execution

Validate the geometry and associative stages independently.

For the affine geometry scan, compare every token's `(m,J,D,H,R)` and all
gradients with the token oracle for arbitrary chunk sizes, padding patterns,
resets, and learned decay values. Then compare every direct-solve output for
all `K` slots of `(d,e,z)`, plus `chi`, with the token path.

Holding all slots of `(d,e,z)` plus `(chi,alpha)` fixed, compare the
asymmetric `K`-edit WY implementation in:

- every token output;
- every chunk-boundary and final associative state;
- all input and gate gradients;
- chunk sizes one, irregular sizes, and the full sequence.
- micro-step packing against the explicit `(t,1),...,(t,K)` recurrence;
- decay applied once per original token and output read only after edit `K`.

Only after both stages pass may the composition be called exact chunkwise
SolveDelta. A chunk-end adapter reused for earlier tokens or a reversed micro-edit
order fails this gate.

## Phase 4: Triton--CUDA--FLA backend

Validate the exact MathDx block-TRSM oracle, the resident chunk-owned CUDA frame
forward and compact reverse, and their Triton scan/direct-`e` FLA-WY exterior
against explicit dense `M` solves and the complete token recurrence over the
normal envelope.

The local environment gate has been exercised for PyTorch `2.13.0+cu130`,
Triton `3.7.1`, CUDA 13.0 Update 2, MathDx 26.06/cuBLASDx 0.7.0, and FLA 0.5.2
on SM120. The dense `r=128`, `K=1` scan/frame/FLA-WY composition now passes its
current forward, final-state, joint-VJP, repeatability, and tying-aware
deep-cancellation tests. This is a validated implementation checkpoint, not
the full release gate: masks, resets, packed sequences, non-128 native widths,
larger native `K`, complete frontend dispatch, the remaining stricter internal
table, and matched-baseline performance remain open. The frame still emits
explicit `d,e,chi` before WY; no full Solve-to-WY fusion is claimed.

- lower/upper residual and complete solve-action relative error;
- all `K` `(d,e)` pairs, `chi`, outputs, and recurrent states;
- input and parameter gradient error;
- BF16 factor/action finiteness with FP32 state and long-rollout drift;
- radial and diagonal saturation fractions plus their gradient magnitudes;
- realized `||M||`, `||M^-1||`, `cond(M)`, and edit transition singular values;
- pairing drift $|e^T d-\bar b^T a|$, especially at pairing zero and two;
- reuse of boundary/chart data across all local primal and dual right-hand sides;
- no full-sequence `T x r x r` geometry or factor materialization;
- native provider/architecture dispatch fails explicitly when unsupported;
- irregular masks, resets, chunk sizes, and single-token decoding;
- complete-layer wall time and workspace, not only isolated TRSM latency;
- compare against the official sample's warning signal that an isolated small
  block TRSM can lose to batched cuBLAS; accept the hybrid route only when
  fusion-level traffic savings survive end-to-end measurement.

The first native contract requires BF16 Tensor Core operands with FP32
accumulation and FP32 continuation state. Fixed high/low decomposition or
compensated reduction is allowed where the cancellation gates require it, but
it must be part of one deterministic schedule. It may not silently change the
system, add iterative correction semantics, round recurrent state to BF16, or
fall back to another chart.

## Phase 5: model evaluation

Compare the completed SolveDelta layer with matched external baselines:

- DeltaNet, GDN, KDA, and GDN2;
- DeltaProduct-2 and at least one larger-edit DeltaProduct;
- Preconditioned DeltaNet and OSDN;
- MesaNet or another online ridge-regression reference;
- the original LSSO operator where bidirectional evaluation is meaningful.

The required SolveDelta interventions are component switches inside the same
operator, not public variants:

- identity geometry (`gamma_g = 0`);
- associative decay structurally disabled for ungated reduction tests;
- `K` swept over the supported values, with matched-parameter and
  matched-compute comparisons where applicable; treat this as ordinary
  capacity/throughput selection inherited from DeltaProduct, not as a new
  architectural question;
- independent geometry feature replaced by edit key 1;
- fixed versus learned geometry forgetting;
- solve-conditioned versus identity read, clearly labeled as an ablation;
- geometry-state reset/retention interventions.

Report per-slot edit contribution, transition singular values, state norms,
and intervention loss changes. Report quality at matched
parameters, recurrent-cache bytes, training FLOPs, wall time, and decoding
latency. The main claim survives only if gains over GDN2 and DeltaProduct-2
remain after these controls.

## Release gate

Before claiming a usable implementation, require:

1. all deterministic core and reduction tests;
2. chunk/token forward, state, invariant, and gradient agreement within the
   native precision contract, with every stricter internal boundary passing;
3. repository-layout checks;
4. complete-layer benchmarks at the first native target `r = d_k = 128`, plus
   reference-contract shape tests at non-128 resolved key-head widths;
5. a written residual-limitations section covering solve cost, non-normal
   transients, the selected `num_edits` compute point, attribution, and
   recurrent-cache size.
