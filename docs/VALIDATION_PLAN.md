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

Use deterministic FP64 tests covering zero state, repeated keys, orthogonal
keys, weak and strong skew, short sequences, and sequences longer than `r`.
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
a_{t,j}^Tn_{t,j}=0,
\qquad
e_{t,j}^Td_{t,j}
=\sum_i b_{t,j,i}k_{t,j,i}^2\in[0,2].
\]

Check both unit-triangular solve residuals, finite forward values, and finite input/parameter
gradients over the declared training envelope. The pairing result is an
eigenvalue certificate; tests must not relabel it as a Euclidean norm bound.

## Native precision contract

The FP64 token recurrence is the sole numerical oracle. To compare a runtime
dtype fairly, generate deterministic master inputs, quantize every runtime
input and parameter once to that dtype, and promote those exact quantized
values to FP64 for the oracle. The native path receives the quantized values.
This isolates kernel arithmetic from input quantization. Gate formulas that the
model contract explicitly evaluates in FP32 are evaluated from the same
quantized logits in FP32 on both paths before the oracle promotes their results.

For any reference tensor `x` and native tensor `x_hat`, report

\[
\rho(x,\widehat x)=
\frac{\operatorname{RMS}(\widehat x-x)}
{\operatorname{RMS}(x)+10^{-8}},
\qquad
a_\infty(x,\widehat x)=\|\widehat x-x\|_\infty.
\]

A tensor passes when `a_inf <= 1e-6` or its declared `rho` ceiling passes.
The absolute branch handles exact or near-zero references; it is not added to
the relative allowance. NaN or infinity always fails. Tests must log both
metrics even when the absolute branch passes. No warning-only gradient class,
CI relaxation, sequence-length multiplier, or architecture-specific tolerance
is allowed.

The pure FP64 reference, its algebraic reductions, and equivalent unfused
reference expressions use `rtol=1e-10, atol=1e-12`. The optimized FP32
geometry adapter has the following stricter internal budgets:

| Boundary, always against FP64 | Forward `rho` | Backward `rho` | Additional gate |
|---|---:|---:|---|
| Triton chunk-boundary `m,J,D,H,R` | `2e-4` | `5e-4` | every boundary, not only the final state |
| bounded chart coordinates and generated LDU factors | `2e-4` | `1e-3` | analytic bounds must also pass |
| MathDx primal triangular/complete solve, given the same FP32 factors | `5e-5` | `2e-4` | normalized residual `eta <= 2e-5` for each triangular solve |
| direct dual products, given the same FP32 factors | `5e-5` | `2e-4` | compare each packed right-hand side separately |
| composed `d,e,chi` from prefix inputs | `5e-4` | `1e-3` | normalized pairing drift `pi <= 5e-5` |

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

The selected dense fixed-length `r=128,K=1` forward uses exact FP32 C16
coordinate-packet substitution for the two unit-triangular solves. Its legal
`2^12` cancellation probe currently gives approximately
`1.24e-5--1.37e-5` composed `d,e,chi` relative RMS, `1.67e-5` maximum
normalized solve residual, and `6.52e-6` maximum pairing drift. Random
asymmetric cases are around `4.3e-8--5.8e-8` relative RMS.

The radial parameter pass is also checked before factor actions. Its fixed
four-channel twofold implementation must remain finite without a norm clamp for
independent `2^12` boundary/local cancellation in both `J` and `D`, including an
ordinary asymmetric tail. It must preserve valid tokens when affine
coefficients underflow to zero, emit exact `(coefficient=0, diagonal=1)` for
structurally invalid tail slots, and keep the unscaled valid diagonal
auxiliaries at the identity-strength point for the strength VJP. Random and
cancellation chart coordinates use the `2e-4` forward ceiling in the table;
NaN, a negative reconstructed norm, or inferring validity from `alpha` is an
unconditional failure.

The independently retained `cuda_chunk_solve_frame128` validation forward uses
a fourth-order Neumann action. This is not algebraically exact: for `||N||<1`,
truncation after degree four has the operator remainder `N^5(I+N)^-1`. The
fixed chart bound `||N||_2 <= ||N||_F < 1/4` gives the conservative operator
bound `1/768`; actual composed `d,e,chi` must still pass the stricter `5e-4`
empirical row above. The deterministic Neumann kernel, including FP16
`J`/factor storage and BF16 `D` storage with FP32 accumulation, measures
approximately `3.5e-5--3.7e-5`. Production packet backward instead transposes
the exact coordinate-packet actions and contracts the five masked-outer
descriptors directly. Its merged qbar contraction, complete frame VJP, and all
component gradients are independently compared with the FP64 recurrence and
must satisfy the `1e-3` internal backward ceiling, including the legal `2^12`
cancellation case.

Here

\[
\eta(A,X,B)=
\frac{\|AX-B\|_F}
{\|A\|_2\|X\|_F+\|B\|_F+10^{-12}},
\]

and for every primal/dual pair

\[
\pi=
\frac{|e^Td-\widetilde b^Ta|}
{\|e\|_2\|d\|_2+\|\widetilde b\|_2\|a\|_2+10^{-12}}.
\]

The complete Triton--CUDA--FLA layer then uses these end-to-end ceilings:

| Advertised runtime path | outputs, every chunk state, final state | `q,k,v,S0,Cq0,Ck0,Cv0` gradients | `u,h,J0,D0` and gate/geometry parameter gradients |
|---|---:|---:|---:|
| FP32 | `2e-3` | `2e-3` | `5e-3` |
| FP16 outer, dense/fixed length | `5e-3` | `1e-2` | `2e-2` |
| BF16 outer, dense/fixed length | `6e-3` | `1.5e-2` | `2.5e-2` |
| FP16 or BF16 outer, packed/varlen | `6e-3` | `1.5e-2` | `2.5e-2` |

`S0` includes the associative initial state. `J0,D0` include any exposed
geometry continuation state. `Cq0,Ck0,Cv0` are the raw projected-input conv4
caches; their final-state cotangents must reach both the initial caches and the
four contributing projected tokens. All gate logits and static gate parameters are in
the last gradient class. Geometry boundary states must additionally satisfy
their stricter internal row above; the end-to-end state allowance cannot hide
a failed scan or solve.

The FP32 layer ceiling was calibrated against the installed mature GDN2 outer
kernel on the local SM120 GPU. For `B=1,T=128,H=2,d_k=d_v=64`, its chunk path
against the same quantized inputs promoted through its FP64 naive recurrence
produced about `1.03e-3` output and `1.14e-3`
final-state `rho`; main vector gradients were about `0.70e-3--1.18e-3` and the
largest erase-gate gradient was about `2.47e-3`. The corresponding BF16 probe
gave about `3.19e-3` output and `2.23e-3` state `rho`, with gradients below
`4.5e-3`. The `2e-3` FP32 state/output ceiling therefore admits the mature
Delta exterior on this hardware while remaining tighter than GDN2's general
`5e-3` ceiling. It is not evidence that SolveDelta itself passes.

Use the same random upstream cotangents for FP64 and native paths and compare
vector-Jacobian products, including losses on both token outputs and returned
states. Required cases include `K=1,2,4`; identity and active geometry; weak,
ordinary, and near-boundary gates; zero and nonzero initial states; chunk sizes
one, 64, irregular tails, and full sequence; packed resets; `t<r`, `t=r`, and
`t>r`; convolution enabled and structurally disabled; nonzero initial conv
caches with losses on returned caches; and the longest declared training/decode envelope. Tolerances do not
grow with `T`.

These numbers are release ceilings, not expected errors. They may be tightened
after implementation evidence. Loosening one requires a reproduced normal-
envelope failure, localization to a named boundary, comparison with the GDN2
baseline on the same hardware/dtype, and an explicit contract revision; a
performance gain alone is insufficient.

The BF16 row is the one explicit revision made during backward bring-up. On a
deterministic cross-chunk `T=65,r=128` VJP, BF16 produced approximately
`5.34e-3` output and `4.45e-3` final-state `rho`, with the largest input
main-vector gradient near `1.25e-2`; a second nonzero-initial-state case placed
the scalar geometry-strength gradient at `2.25e-2`. FP16 produced about
`8.49e-4`, `6.78e-4`, and `3.20e-3`, respectively. FP16 is therefore the
selected outer dtype. BF16 is retained under its separately stated ceiling
rather than consuming the FP16 budget.

## Phase 1: exact Delta-family reductions

Set `num_edits = 1`, `gamma_g = 0`, disable skew, and verify that
`X^(H) = X^(R) = N^- = N^+ = 0`, `Sigma = M = P = I`.
Compare every output, intermediate state, final state, input gradient, and
parameter gradient with the official GDN2 naive recurrence. Apply the
published gate ties and repeat for KDA, GDN, and DeltaNet. No sigmoid endpoint
or limiting-logit argument is allowed in an exact reduction.

At identity geometry with symmetric edits, compare `K = 1, 2, 4` with the
corresponding official DeltaProduct-`K` naive recurrence, including the
negative-eigenvalue range.
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
- a legal prefix-conditioned orthogonal-residual erase unavailable to one
  identity-geometry GDN2 edit;
- a two-edit planar rotation unavailable to any single rank-one transition.

Also construct a non-collinear legal `(a,b)` pair and verify that the symmetric
part of `ab^T` has a negative eigenvalue despite `a^T b >= 0`. This prevents the
pairing certificate from being mislabeled as dissipativity.

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

## Phase 4: hybrid Triton--CUDA--FLA backend

Validate the exact MathDx block-TRSM oracle, the bounded polynomial CUDA
training candidate, and their Triton scan/FLA-WY exterior
against both explicit dense `M` solves and the complete token recurrence over
the normal envelope:

The local environment gate is already satisfied for PyTorch `2.13.0+cu130`,
Triton `3.7.1`, CUDA 13.0 Update 2, and MathDx 26.06/cuBLASDx 0.7.0: CUDA and
Triton smoke kernels pass, FLA GatedDeltaNet2 and BF16 GatedDeltaProduct pass
forward/backward, a PyTorch CUDA C++ extension compiles and runs for SM120, and
the official cuBLASDx block-TRSM sample passes on SM120. The dense fixed-length
`r=128,K=1` fused forward, state, VJP, initial-state, and model-parameter gates
now pass. The list below remains the full release checklist; masks/resets,
packed sequences, generic native shapes, long training drift, and architecture
dispatch are still open.

- lower/upper residual and complete solve-action relative error;
- all `K` `(d,e)` pairs, `chi`, outputs, and recurrent states;
- input and parameter gradient error;
- FP32 factor/action finiteness and long-rollout drift;
- radial and diagonal saturation fractions plus their gradient magnitudes;
- realized `||M||`, `||M^-1||`, `cond(M)`, and edit transition singular values;
- pairing drift `|e^T d-b_tilde^T a|`, especially at pairing zero and two;
- reuse of one factor load for all packed primal and dual right-hand sides;
- no full-sequence `T x r x r` geometry or factor materialization;
- MathDx provider/architecture dispatch fails explicitly when unsupported and
  never silently selects a handwritten alternative solve chart;
- irregular masks, resets, chunk sizes, and single-token decoding;
- complete-layer wall time and workspace, not only isolated TRSM latency;
- compare against the official sample's warning signal that an isolated small
  block TRSM can lose to batched cuBLAS; accept the hybrid route only when
  fusion-level traffic savings survive end-to-end measurement.

The first native contract remains FP32 for factors and solve accumulation.
The isolated solve-frame tests also cover BF16 and FP16 direct-dual operands;
they pass the `5e-3` forward envelope, but are slower than FP32 in the current
unfused wrapper because of conversion traffic. They therefore remain explicit
experiments and may not silently change the system, add iterative correction
semantics, or fall back to a dense chart.

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
- zero skew versus the full bounded LDU chart;
- orthogonal erase residual forced to zero;
- `K` swept over the supported values, with matched-parameter and
  matched-compute comparisons where applicable; treat this as ordinary
  capacity/throughput selection inherited from DeltaProduct, not as a new
  architectural question;
- independent geometry feature replaced by edit key 1;
- fixed versus learned geometry forgetting;
- solve-conditioned versus identity read, clearly labeled as an ablation;
- geometry-state reset/retention interventions.

Report the learned orthogonal-residual utilization, per-slot edit contribution, transition singular
values, state norms, and intervention loss changes. Report quality at matched
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
