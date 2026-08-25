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

## Acceptance hierarchy

Validation has three tiers. The tier is determined by mathematical ownership
and observability, not by which tensor is easiest for a kernel test to expose.

1. **Hard semantic gates.** The FP64 token recurrence, causality, masking,
   reset and recurrent-split behavior, FP32 `(m,J,D,S)` continuation semantics,
   arbitrary supplied initial state, structural identity and zero behavior,
   finite exact reductions, and the chart's pairing, invertibility, bounds, and
   Delta-family reductions define the operator. A failure changes the model or
   its declared state semantics and is never hidden by a numerical allowance.
2. **Production-observable numerical gates.** BF16 token outputs, every returned
   and chunk-boundary continuation state, and composed input/state/parameter
   VJPs are compared with the quantized-input FP64 oracle using the fixed public
   ceilings below. Finiteness and stability over the declared training envelope
   are part of this tier. These gates decide whether a native path may ship.
3. **Private diagnostics.** Same-packed contractions, private `q2` and scale,
   packed strict coordinates, strict-coordinate cotangents, pairing drift before
   a public store, reduction order, repeatability, and source-code structure are
   localization tools. They may have a broad corruption guard, but a tighter
   historical checkpoint must not veto a candidate that passes tiers 1 and 2.
   No private test may require `tl.dot`, forbid `tl.cumsum`, require twofold
   products, or otherwise turn the current instruction schedule into semantics.

The geometry scan is a special case only because its FP32 `(m,J,D)` boundaries
are continuation state and therefore observable; its state and state-VJP gates
remain production gates. MathDx also retains its independent exact-solve row
because that standalone oracle explicitly claims FP32 solve accuracy. Neither
exception promotes unrelated frame or WY intermediates into public state.

## Native BF16/FP16/FP32 precision contract

The first advertised training path is mixed precision, not an FP32 kernel with
a low-precision output wrapper:

| Quantity | Required native representation |
|---|---|
| projected/raw activations; conv4 inputs/outputs; unnormalized `h`; edit values; erase/write gates | BF16 at the public/raw boundary; no later BF16-to-FP16 pseudo-promotion |
| normalized `u,q,k` | reduce and normalize in FP32, then write the private panel directly as FP16; `l2` norm and every component are at most one |
| erase source `b=beta odot k` | form in FP32 from `0 <= beta <= 2` and normalized `k`, then write directly as FP16; `||b||_2 <= 2` |
| bounded strict chart coordinates and certified frame-action/decayed panels | compute in FP32 and write directly as FP16 using the analytic bounds below |
| Tensor Core matrix operands | BF16 raw or analytically bounded FP16 private values chosen statically per specialization |
| dot/GEMM/TRSM partials and backward partials | FP32 accumulation; reduced-precision accumulation is forbidden |
| log-decay, `exp`/`expm1`/`softplus`/`sigmoid`, norm and radial reductions, geometry strength, diagonal residual action, and sensitive scalar reverse | FP32 |
| continuation and chunk-boundary `m,J,D,S` | FP32, with no BF16 round trip between chunks or recurrent calls |
| public token outputs and returned activation gradients | BF16; gradients are cast only after their FP32 reduction is complete |
| `W` and continuation/state-derived panels | FP32 |
| unbounded `Y/U_z` solve outputs | write BF16 only after FP32 accumulation; FP16 is forbidden |
| write-value product `z` | form in FP32 from BF16 gate/value operands at use; do not materialize a low-precision panel |

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

A same-packed diagnostic may execute a named FP32 producer, round its result
directly to FP16, and promote those exact bits to FP64 to isolate one action's
arithmetic. That comparison is not the operator oracle. It neither requires a
production kernel to materialize that panel nor freezes its tiling, packing,
reduction tree, or forward/reverse instruction schedule.

Normalization is an explicit mixed boundary: load the BF16 raw vector, reduce
the squared norm and normalize in FP32, then write the normalized private panel
directly as FP16. A later `BF16 -> FP16` cast is exact only in the shared normal
exponent range, may lose small or large values outside it, and can never
recover the three significand bits already discarded by BF16; it does not
satisfy this contract. Gate nonlinearities and log-decay
remain FP32 even though their projected logits came from BF16 model
activations. Primal and transpose-dual kernels must implement the same
mathematical factor actions and pass the pairing and composed-VJP gates, but
may use different private layouts or reduction orders. Rounding `J`, `D`, or
`S` at a chunk boundary would instead create a split-dependent rounded
recurrence and is outside this contract.

Private FP16 is a representation privilege, not another backend. It is legal
only for the statically named panels whose FP32 producer has a dtype-global
analytic bound inside FP16 range, whose consumer accumulates in FP32, and whose
producer writes FP16 directly. An implementation may generate, consume, or
recompute such a panel in any algebraically equivalent static schedule. The
initial certified set is

\[
\|u\|_2,\|q\|_2,\|k\|_2\le1,
\qquad \|\beta\odot k\|_2\le2,
\]

the strict chart coordinates with per-channel Frobenius radius `1/8` and
combined radius `1/4`, and frame actions certified by

\[
B_P=\frac{e^{1/4}}{(1-1/4)^2}\approx2.283,
\qquad B_D=(1+1/4)^2e^{1/4}\approx2.007,
\]

\[
\|d\|_2\le B_P,
\qquad \|\chi\|_2\le B_D,
\qquad \|e\|_2\le2B_D<4.013.
\]

Including the declared FP16 strict-factor rounding gives conservative bounds
`||d||_2 < 2.284`, `||chi||_2 < 2.007`, and `||e||_2 < 4.014`.
Nonamplifying tile-local decays of these frame panels inherit the same bound.
Unnormalized `h`, values, write-value products, recurrent states, and WY solve
intermediates such as `Y/U_z` do not have this certificate and cannot be stored
as FP16 under the first contract. Adding another FP16 panel requires an
analytic certificate and a contract change, never runtime range inspection,
clamping, or fallback. The upper bounds exclude FP16 overflow; possible
underflow of tiny components is governed by the unchanged action and VJP gates,
not a BF16 fallback.

The diagonal is identity-centered. Compute its bounded log coordinate
`delta` and `expm1(delta)` in FP32 and apply

\[
\exp(\delta)x=x+\operatorname{expm1}(\delta)x.
\]

Do not require a materialized low-precision `1 + delta` to preserve a
perturbation below one BF16 ulp at one. Zero-centered strict coordinates are
packed directly from FP32 to FP16 because their exponent follows their
magnitude. This representation rule
preserves a cheap FP32 diagonal residual without imposing FP32 dense actions.

For any reference tensor `x` and native tensor `x_hat`, report

\[
\rho(x,\widehat x)=
\frac{\operatorname{RMS}(\widehat x-x)}
{\operatorname{RMS}(x)+10^{-8}},
\qquad
a_\infty(x,\widehat x)=\|\widehat x-x\|_\infty.
\]

A FP32 production-observable tensor passes when `a_inf <= 1e-6` or its
declared `rho` ceiling passes. A BF16 public tensor uses `a_inf <= 2e-4`
instead. Private diagnostics may use the same metric and an explicitly named
broad safety envelope, but their historical target values are not release
ceilings. The absolute branch handles exact or near-zero references; it is not
added to the relative allowance. NaN or infinity in a public result, state, or
reachable VJP always fails. Tests must log both metrics when they make a
numerical acceptance decision. No CI relaxation, sequence-length multiplier,
cancellation multiplier, or architecture-specific public tolerance is allowed.

The pure FP64 reference, algebraic reductions, and equivalent unfused reference
expressions use `rtol=1e-10, atol=1e-12`. For a single contraction with the
same packed operands define

\[
\tau(A,B,\widehat C)=
\frac{\|\widehat C-AB\|_F}
{\|A\|_F\|B\|_F+10^{-12}}.
\]

The directly observable FP32 scan boundary retains this dedicated production
row. The complete output/state/VJP row appears below.

| Observable boundary | Forward ceiling | Backward ceiling | Additional gate |
|---|---:|---:|---|
| every FP32 chunk-boundary and final `m,J,D` against the quantized-input FP64 recurrence | `rho <= 5e-3` | `rho <= 5e-3` | arbitrary supplied initial state; initial-state VJP `rho <= 5e-4` |

The following are private diagnostics, not prerequisites that must all pass
before the composed layer is considered:

- same-packed `tau` measurements for individual Tensor Core tiles;
- private radial `q2`/scale and pre-pack strict coordinates;
- isolated strict-chart reverse and transpose inner products;
- private `d/e/chi`, `Y/U_z`, `bar W`, pairing drift, and triangular residuals;
- use of a fixed tree, twofold representation, atomics, or a particular Triton,
  CUDA, or library instruction sequence.

Normal-envelope private tests may retain named broad corruption guards: `rho <=
1e-3` for the isolated strict-chart transpose, `rho <= 1e-2` for radial/chart
statistics and packed frame actions, and `rho <= 1e-2` for isolated C16/C32/C64
WY forward/transpose actions. These are deliberately not per-contraction accuracy
claims and do not imply an `eta` gate. They may be revised from reproduced
end-to-end evidence without changing operator semantics or the public ceilings.
Deep nonstructural cancellation is judged at the realized chart action and
reachable composed VJP, as specified below, rather than at private `q2`.

FP32 atomic accumulation across independent route or coordinate-block partials
is an admissible implementation schedule; atomic execution is not an operator
approximation. Every run must pass the public oracle and VJP ceilings. Over at
least 100 identical launches, the maximum run-to-run `rho` must be at most
`1e-6`; deterministic schedules are held to the same bounded-repeatability
criterion, not universal bitwise equality. A candidate is selected only when
its warmed complete forward-plus-backward median and p95 latency improve over
the current schedule. An isolated-kernel win or lower workspace alone is not an
adoption result.

If the action packs factor coordinates to FP16, round-to-nearest with
`u_h=2^-11` gives the certified packed bounds

\[
\|N_H^\pm\|_F,\|N_R^\pm\|_F
\le (1+u_h)/8=0.12506103515625,
\qquad
\|N^\pm\|_F\le0.2501220703125,
\]

and FP32 diagonal entries in `[0.7788008, 1.2840255]`; using the rounded strict
bounds gives a declared `cond_2(M)` ceiling below `4.59`. A frame that keeps
factors FP32 retains the exact `1/8`, `1/4`, and original condition bounds.
Both forms keep the identity point exact.

The geometry row is calibrated for the actual Tensor Core scan rather than an
ideal FP32 outer product. On the local SM120 GPU, BF16 operands with FP32
boundaries gave about `0.7e-3--1.46e-3` `J` error and
`1.97e-3--2.32e-3` `D` error for `T=130,1024,8192`; BF16 leaf VJPs were about
`1.54e-3--1.67e-3`, while FP32 initial-state VJPs stayed below `8.1e-7`.
A mixed-frame proxy with FP32 chart construction, one BF16 factor pack, and
FP32 action accumulation produced factor/action error below `1.9e-3`, raw VJP
error below `3.0e-3`, and simulated BF16-leaf VJP error below `5.1e-3` at
`r=128`. These probes calibrate useful diagnostics but do not freeze an
instruction order or substitute for the production-observable tests.

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

The chunk-owned frame path should be checked independently to localize errors,
but its private checkpoints are not a prerequisite for evaluating the composed
FLA exterior. Deep cancellation remains an adversarial BF16-observable test,
not a requirement to recover every low bit of a private expanded quadratic. For example, the
exactly representable BF16 pair `(0.6953125, 0.71875)` has product
`0.499755859375 = 0.5 - 2^-12`. Combining that local product with the FP32
boundary entry `-0.5` gives a real post-quantization cancellation ratio of
`4095`. In the frozen `m=2` fixture the normalized residual is `2^-13`, only
`2^-10` of the radial scale `c=2^-3`. The FP64 oracle retains that mathematical
residual. Record the realized zero-centered chart coordinate `A=aZ` and its
action to localize failures, but production acceptance is decided by finite
chart construction plus the composed output, returned-state, and VJP gates. The
expanded private quadratic and radial scale are not standalone model
observables.

For every cancellation fixture, first quantize the runtime operands, then
promote them and recompute

\[
\kappa=
\frac{|\alpha|\|B\|_F+\sum_s |w_s|\|L_s\|_F}
{\|\alpha B+\sum_s w_sL_s\|_F+10^{-30}}.
\]

Only then assign it to one of the fixed bins `nonstructural exact
cancellation`, `[1,16]`, `[2^7,2^8]`, or `[2^11,2^12]`. Cover `J` and `D`,
lower and upper coordinates, and an ordinary asymmetric tail. In the deepest
nonzero bin, record the chart-coordinate and action diagnostics without
multiplying any bound by `kappa`, and require the ordinary per-dtype composed
output/state/VJP ceilings. It does not separately gate the private reconstructed
norm, `q2`, or scale. A legal PSD `J` boundary is required for the production
witness. The former
`4096 + (-4096 + 0.01)` FP32 fixture is not a BF16 gate because `0.01` is lost
before execution.

To distinguish structural identity from numerical cancellation, write one
strict route as

\[
Z=\alpha B+\sum_s w_sL_s,qquad
n=\|Z\|_F^2,qquad
Q=c^2+g^2n,qquad
a=gcQ^{-1/2}.
\]

The FP64 operator always has `n >= 0`. A pair implementation may instead form
an FP32 signed estimate `n_hat` from separately rounded boundary norm, pair,
and Gram contractions. For a nonstructural reference `n=0`, require

\[
\widehat Q>0,\qquad
\widehat Q,\widehat a\ \text{finite},\qquad
a_\infty(\widehat A,A)\le2\times10^{-4}
\ \text{or}\ \rho(\widehat A,A)\le6\times10^{-3}.
\]

Here `A=aZ` is the zero-centered strict chart coordinate; an implementation is
not required to materialize it at a particular storage boundary. The private
`n_hat`, `Q`, and `a` have no independent accuracy or sign requirement as long
as they are finite, `Q` is positive, and the composed observables and VJPs pass
their ordinary rows. A valid
cotangent at the private scale node must be reachable from the composed chart:
`bar a=<bar A,Z>`. Tests must not inject an arbitrary `O(1)` `bar a` when `Z`
is zero or deeply cancelled. Structural identity geometry, `g=0`, invalid tail
slots, and explicit component-disable reductions still require bitwise
identity and zero geometry VJPs where the FP64 graph has them.

The frame must remain finite without a norm clamp, preserve valid tokens when
affine coefficients underflow to zero, emit exact identity-chart outputs for
structurally invalid tail slots, and retain the valid diagonal auxiliaries
required by the strength VJP at identity. A nonpositive `Q`, a chart coordinate
or composed VJP outside the fixed observable envelope, NaN, infinity, a
data-dependent precision fallback, or inferring validity from an underflowed
coefficient is an unconditional failure. If a Tensor Core tile consumes a
large FP32 boundary contribution, it must keep an FP32 accumulator. Its operand
lowering is chosen statically for a specialization and may be revised between
implementations; direct BF16, static high/low packing, or an FP32 path is
selected by the common production gates, never by a runtime cancellation test.

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

The first native frame inverse is the compile-time fixed degree-six Neumann
action

\[
P_6(N)=\sum_{j=0}^{6}(-N)^j,
\qquad x^{(0)}=b,\quad x^{(j+1)}=b-Nx^{(j)},\ j=0,\ldots,5.
\]

This is a production numerical approximation to the exact bounded-LDU oracle,
not a separate operator. With `q=1/4`, its exact-arithmetic single-factor error
is

\[
\delta_6=\frac{q^7}{1-q}=\frac1{12288}\simeq8.138\times10^{-5},
\]

and its complete primal LDU action error is bounded by

\[
2e^{1/4}\frac43\delta_6\simeq2.79\times10^{-4}.
\]

For a one-factor implicit solve VJP, the factor-cotangent approximation obeys

\[
\|\widehat{\bar N}-\bar N\|_F
\le\frac{2q^7}{(1-q)^2}\|\bar y\|_2\|b\|_2
\simeq2.17\times10^{-4}\|\bar y\|_2\|b\|_2,
\]

before declared operand rounding. Forward and transpose reverse use the same
packed factor bits and exactly six updates. Runtime stopping, degree selection,
precision fallback, clipping, iterative refinement, and per-iteration HBM
panels are forbidden. Structural `N=0` must return the source exactly at every
iteration.

The displayed symbolic bounds use the exact chart ceiling `q=1/4`. The
same-packed oracle instead uses the certified rounded-factor ceiling
`q_h=0.2501220703125`, which gives `delta_6,h <= 8.168e-5`; this small increase
is reported with packing/contraction error and does not alter a public gate.

Validation compares lower and upper `P_6` actions against the quantized-input
FP64 exact solve, compares `P_6(N)^T` and the streamed implicit `-bar_b y^T`
factor cotangent against the exact VJP, records pairing drift, and then applies
the unchanged composed output/state/VJP ceilings below. Low-precision
contraction error is measured separately from the analytic truncation bound.
The `eta <= 2e-5` residual remains exclusive to the standalone exact MathDx
oracle; it is not a production Neumann gate.

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
| native `r=128,K=1,C=C_*`, `C_*` selected offline from `{16,32,64}`, BF16 public/raw operands, bounded private FP16 panels, and FP32 accumulation/state/scalars | `6e-3` | `1.5e-2` | `2.5e-2` |

`S0` includes the associative initial state. `J0` must be exactly symmetric and
its VJP is compared using the frozen full-storage `(G+G^T)/2` representative;
`D0` remains dense and asymmetric. `Cq0,Ck0,Cv0` are the raw projected-input conv4
caches; their final-state cotangents must reach both the initial caches and the
four contributing projected tokens. All gate logits and static gate parameters are in
the last gradient class. Returned geometry boundary states additionally satisfy
their dedicated observable scan row above; the end-to-end `S` allowance cannot
hide a failed FP32 continuation recurrence. Private chart, action, and WY
diagnostics do not add release gates. FP32 and FP16 execution may remain
reference diagnostics, but they are not alternate advertised training
contracts.

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
resets, and arbitrary recurrent splits. The current checkpoint native gate
covers only `r=128,K=1,C=32`. Before offline selection, each candidate
`C in {16,32,64}` must pass the same identity and active-geometry; weak,
ordinary, and near-boundary; zero/nonzero initial-state; irregular-tail; and
multiple-chunk cases, including lengths straddling its own boundary. The gate
also covers `t<r`, `t=r`, and `t>r`; convolution enabled and structurally
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
GDN2 baseline on the same hardware and BF16/FP16/FP32 contract, and an explicit
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
resets, and learned decay values. Use any exposed direct-solve output for all
`K` slots of `(d,e,z)`, plus `chi`, as a private localization diagnostic. A
fused implementation need not materialize or expose those panels.

Holding all slots of `(d,e,z)` plus `(chi,alpha)` fixed, compare the
asymmetric `K`-edit WY implementation in:

- every token output;
- every chunk-boundary and final associative state;
- all input and gate gradients;
- chunk sizes one, irregular sizes, and the full sequence.
- micro-step packing against the explicit `(t,1),...,(t,K)` recurrence;
- decay applied once per original token and output read only after edit `K`.

The composition is exact chunkwise SolveDelta when its hard sequence semantics
and production-observable forward/state/VJP rows pass. A chunk-end adapter
reused for earlier tokens or a reversed micro-edit order changes the recurrence
and fails the hard semantic gate, regardless of private checkpoints.

## Phase 4: Triton--CUDA--FLA backend

Validate the exact MathDx block-TRSM oracle, the chunk-owned CUDA frame forward
and compact reverse, and their Triton scan/FLA-derived chunk-WY blocks against
explicit dense `M` solves and the complete token recurrence over the normal
envelope.

The local environment gate has been exercised for PyTorch `2.13.0+cu130`,
Triton `3.7.1`, CUDA 13.0 Update 2, MathDx 26.06/cuBLASDx 0.7.0, and FLA 0.5.2
on SM120. The pre-Neumann dense `r=128`, `K=1` scan/frame/C32-WY checkpoint
passes its current forward, final-state, joint-VJP, repeatability, tying-aware
deep-cancellation, irregular-C32, and C32 transpose-action tests. It is a
historical C32 replacement baseline, not evidence that the selected resident
Neumann path or the C16/C64 candidates have passed.

The earlier `eta <= 2e-5` production gate was deleted because the inverse and
pre-storage solve residual are private implementation details; that tighter
residual remains mandatory only for the independent FP32 MathDx oracle. This is a
validated implementation checkpoint, not the full release gate: masks,
resets, packed sequences, non-128 native widths, larger native `K`, complete
frontend dispatch, remaining production-observable envelope rows, and
matched-baseline performance against external models remain open. The frame still emits private
`d,e,chi` caches before WY; no full Solve-to-WY fusion is claimed.

- lower/upper residual and complete solve-action relative error;
- all `K` `(d,e)` pairs, `chi`, outputs, and recurrent states;
- input and parameter gradient error;
- declared FP16 factor/action finiteness with FP32 state and long-rollout drift;
- radial and diagonal saturation fractions plus their gradient magnitudes;
- realized `||M||`, `||M^-1||`, `cond(M)`, and edit transition singular values;
- pairing drift $|e^T d-\bar b^T a|$, especially at pairing zero and two;
- reuse of boundary/chart data across all local primal and dual right-hand sides;
- no full-sequence `T x r x r` geometry or factor materialization;
- native provider/architecture dispatch fails explicitly when unsupported;
- irregular masks, resets, chunk sizes, and single-token decoding;
- fixed `p=6` in both forward and transpose reverse, with exact structural
  identity and no runtime convergence or correction branch;
- no powers, dense tokenwise factors, or per-iteration Neumann panels in HBM;
- complete-layer wall time and workspace, not only isolated TRSM latency;
- for every proposed frame/pair/WY fusion, an A/B against compact private
  staging that records registers, shared bytes, spills, barriers, active
  CTAs/SM, launch count, HBM traffic, and backward cache/recompute cost;
- offline C16/C32/C64 and mature four-/eight-warp comparisons after numerical
  acceptance, reporting complete forward/backward/F+B median and p95, resource
  use, and allocator peak; retain one static target winner;
- compare against the official sample's warning signal that an isolated small
  block TRSM can lose to batched cuBLAS; accept the hybrid route only when
  fusion-level traffic savings survive end-to-end measurement.

The resident Neumann path may replace the blocked-substitution checkpoint only
when its warmed complete forward-plus-backward median and p95 improve at the
target shape while all public gates pass. An isolated action win is
insufficient.

No validation gate requires one CTA, zero frame-to-pair HBM traffic, or a
particular internal frame/pair ABI. Internal `d/e/chi`, transformed panels, and
upstream-compatible packed layouts are all valid when their schedule wins
complete F+B; they do not become returned model state. Fusion is rejected when
reduced traffic is outweighed by longer lifetimes, synchronization, spills,
lower CTA concurrency, or backward recomputation.

Likewise, no gate requires `C=32`, four warps, a fixed stage count, a fixed
shared-memory ceiling below the device limit, two CTAs/SM, or zero spills.
`C in {16,32,64}` and mature four-/eight-warp launch shapes are numerically
equivalent candidates. Selection is offline and target-specific, based on the
complete path; it is never data-dependent and does not alter tolerances.

The first native contract requires declared BF16/FP16 Tensor Core operands
with FP32 accumulation and FP32 continuation state. Direct low-precision
operands, fixed high/low decomposition, compensated reduction, FP32 scalar
work, atomics, and different forward/transpose tile schedules are all eligible
private lowerings. A specialization chooses its lowering statically; it may not
silently change the system, add iterative correction semantics, round recurrent
state to BF16, or select precision from runtime magnitudes or cancellation.
Candidates are judged after the complete forward and transpose paths are
connected to the token oracle. An isolated private checkpoint does not justify
adding compensation, and an old checkpoint does not forbid removing it when
the hard and production-observable gates improve or remain satisfied.

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

1. all hard semantic and production-observable core tests, exact structural
   identities, and the bounded-repeatability gate for ordinary numerical paths;
2. chunk/token forward, state, invariant, and gradient agreement within the
   native production precision contract; private diagnostics are reported but
   do not create additional release ceilings;
3. repository-layout checks;
4. complete-layer benchmarks at the first native target `r = d_k = 128`, plus
   reference-contract shape tests at non-128 resolved key-head widths;
5. a written residual-limitations section covering solve cost, non-normal
   transients, the selected `num_edits` compute point, attribution, and
   recurrent-cache size.
