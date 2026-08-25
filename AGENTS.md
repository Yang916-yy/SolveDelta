# AI-Assisted Contribution Contract

This repository has one research operator: **Causal LSSO: SolveDelta**.
Prefix-LSSO is a provenance and property oracle,
not an exact parameterization constraint or a second architecture. DeltaNet,
Gated DeltaNet, GDN2, KDA,
DeltaProduct, PDN, and related models are reductions or comparison baselines,
not parallel development tracks.

`Causal LSSO` is named for solved contextual adaptation: a prefix constructs a
compact system whose solve action defines the current memory frame. The name
does not require the bidirectional LSSO accretive chart. Never treat
`I + F F^T + Omega`, its coordinate extraction, or its factorization as the
identity of the model.

The human contributor remains responsible for scope, correctness, provenance,
testing, and reviewability. AI-generated mathematics, code, experiments, and
citations must satisfy the same standards as human-authored work. Material use
of external research or upstream code must be identified with the source and
the decision it informed.

## Design principles

- Keep one current operator contract. Do not preserve abandoned candidates,
  legacy variants, compatibility aliases, or speculative branches.
- Geometry width is derived, not independently configured: `r := d_k`, where
  `d_k` is the resolved per-key-head width supplied by the containing model.
  Frontends may provide `head_k_dim` directly or derive total key width from
  model width and key expansion, but must resolve and validate it before
  constructing SolveDelta. `r=128` is the first native specialization and benchmark
  profile, not a mathematical default. A smaller independent rank is a later
  compression study, not part of the base model.
- Prefer the simplest engineering form that is algebraically equivalent to the
  reference and measurably better for the intended workload.
- The first native training contract is BF16-observable with BF16 public and
  raw operands, analytically bounded private FP16 panels, and FP32
  accumulation. Geometry and associative continuation states, log-decay and
  gate evaluation, normalization and radial reductions, sensitive scalars,
  and backward partials remain FP32. FP64 remains the mathematical oracle; it
  is not the production execution dtype.
- A private panel may be FP16 only when its FP32 producer has a static analytic
  magnitude bound safely inside the FP16 finite range, writes the FP16 panel
  directly, and every consumer accumulates in FP32. Forward and reverse may use
  different algebraically equivalent private layouts and reduction schedules;
  neither schedule is part of the operator contract. Casting an already-rounded
  BF16 tensor to FP16 recovers no information, may lose exponent range, and is
  forbidden as pseudo-promotion. There is no runtime dtype selection,
  magnitude threshold, or precision fallback.
- Production numerical acceptance is defined at BF16 outputs, FP32
  continuation states, and composed VJPs. Structural identity interventions
  remain bitwise exact, but private chart coordinates and expanded radial
  quadratics are not required to reproduce nonstructural cancellation below
  BF16 observable resolution.
- Stable means finite, correct, and trainable throughout the declared dtype,
  rank, sequence-length, gate, and optimizer envelope.
- Report residual limitations plainly. Do not hide them behind clipping,
  fallbacks, thresholds, or untested configuration switches.

## Mathematical ownership

- `causallsso/reference.py` is the only owner of operator mathematics and the
  FP64 numerical oracle. `docs/INNOVATION_PROGRAM.md` explains that contract
  but must not define a competing executable recurrence.
- Parameters and `nn.Module` behavior must have one owner. Public variants and
  validation must have one owner. Do not duplicate model classes or reference
  implementations.
- Every state must declare shape, orientation, initialization, update order,
  dtype, padding/reset behavior, and gradient contract.
- Native continuation states `(m,J,D,S)` are FP32 and are never rounded to BF16
  at chunk boundaries. Such rounding would make recurrent splits change the
  numerical recurrence rather than merely its implementation.
- `J` is a symmetric continuation state. The production API accepts only an
  exactly symmetric `J0`, retains full FP32 matrix storage for the first native
  implementation, and represents a full-tensor cotangent by
  `(bar_J + bar_J^T) / 2`. `D` remains an unconstrained dense state.
- The current token updates geometry once, constructs one shared
  transpose-dual solve adapter, performs `num_edits = K` ordered associative
  micro-edits, and is read after edit `K`. `K` is a positive, static model
  hyperparameter; it does not create a second operator contract. The default is
  `K = 1`; larger values remain ordinary capacity/compute settings.
- The model frontend applies one fixed GDN2-style depthwise causal `conv4`
  with SiLU independently to the projected query, packed edit keys, and packed
  edit values before head reshape and key/query normalization. It is enabled by
  default and may be structurally disabled; geometry and gate branches are not
  convolved.
- `r = d_k`; geometry features, keys, queries, and solve-domain coordinates live
  in the same key-side space unless the contract is explicitly revised.
- The normalized edit key is the canonical local write direction. Do not add a
  sigmoid write-direction gate whose endpoint must be forced to recover a
  baseline. Exact reductions must hold at finite shared parameters or through
  an explicit structural component switch, never only in a parameter limit.
- The erase solve-domain covector is the elementwise gated normalized key.
  There is no skew logit, orthogonal erase residual, or skew specialization in
  the current operator.
- The associative state always remains in one fixed ambient basis. Do not move
  it into a time-varying solve domain. Local write vectors use the primal action;
  erase and read covectors use its exact dual action.
- Channel-wise associative decay is enabled by default. It may be structurally
  disabled to test exact ungated DeltaNet and DeltaProduct reductions; that
  intervention is not a second operator variant.
- Full-prefix conditioning means dependence on every token represented in
  `(m,J,D)`, not lossless recovery of the original ordered prefix. State the
  moment-collision boundary explicitly. Do not present history collision as
  unique to SolveDelta: Delta-family models also compress unbounded histories into
  fixed `S`. A geometry collision means equal `(m,J,D)` and therefore equal
  solve adapters; it is a complete-state collision only when `S` also agrees.
- `J` and `D` are separate canonical capacity states: `J` owns sign-robust
  occupancy geometry and `D` owns directional driven geometry. Their packed
  fixed cache is an accepted model cost. They must enter separate nonlinear
  bounded maps; do not sum them before the chart, which collapses them to the
  single cross moment `D + J`. Do not merge or remove one as a work reduction
  without matched utilization and task evidence.
- Prefix moments directly generate the bounded LDU system in the canonical
  recurrence. Do not reintroduce token-local Cholesky/LU factorization, the
  bidirectional accretive chart, or full-matrix whitening as hidden lineage
  requirements.
- Canonical numerical constants are `c_H = c_R = 1/8` for the two separately
  mapped strict-triangular coordinates, total `c = 1/4`, and
  `s_H = s_R = 1/8` with total diagonal log-scale bound `s_max = 1/4`.
  They are fixed implementation details, not public model variants.
- Do not call a general asymmetric rank-one factor dissipative or positive
  semidefinite merely because `e^T d >= 0`. The contract guarantees pairing and
  eigenvalue location, not a positive-semidefinite Hermitian part or Euclidean
  contraction.

## Equivalent implementations

- The FP64 token recurrence is the numerical oracle.
- Prefix-LSSO with Rank-Rotary disabled is a provenance diagnostic. Matching
  its normalized Gram and driven cross-moment construction is useful derivation
  evidence, not an acceptance gate for the causal system chart.
- The bounded LDU chart must retain its exact identity point, invertible primal
  and inverse-transpose dual actions, exact pairing, certified primal/dual and
  similarity bounds, and full `r^2` local differential rank supplied by the
  unconstrained ambient `X^(R)` chart coordinate while `X^(H)` is symmetric.
  Do not test this as
  `dM/dR` under the structural `gamma_g=0` reduction, and do not conflate the
  chart property with feasible-prefix reachability: `rank(D_t), rank(R_t) <= t`,
  and arbitrary local `R` directions at nonzero geometry strength require a
  remembered geometry span of rank `r`.
- Numerical acceptance has three tiers. Hard semantic gates cover the FP64
  token recurrence, causality and sequence semantics, FP32 continuation states,
  structural identities and reductions, and the chart's mathematical
  invariants. Production-observable gates cover BF16 outputs, every returned
  or chunk-boundary state, and composed VJPs under the fixed ceilings in
  `docs/VALIDATION_PLAN.md`. Private diagnostics cover intermediate reduction
  order, packed-panel comparisons, strict-coordinate cotangents, repeatability,
  and source-level implementation shape. A private diagnostic may localize a
  regression or impose a broad corruption guard, but it must not reject an
  implementation that passes the hard and production-observable gates merely
  because it uses a different arithmetic schedule.
- The Triton geometry scan's FP32 `(m,J,D)` boundaries are production-observable
  continuation state and retain their dedicated state gates. The standalone
  MathDx path retains its independent exact-solve budgets because it explicitly
  claims FP32 exact-oracle behavior. Other strict/radial/WY intermediates do not
  acquire production status merely because they are convenient checkpoints.
  Do not add warning-only failures, architecture-specific public tolerances, or
  sequence-length-scaled public allowances.
- Runtime-oracle comparisons quantize public/raw activation operands once to
  BF16, promote those exact values to FP64, and compare against that oracle.
  A declared FP16 private panel is instead produced by the matching FP32
  operation and rounded directly to FP16. A same-packed oracle may use those
  bits to diagnose one contraction, but it is not the operator oracle and does
  not prescribe the production reduction tree. Tensor Core contractions use
  declared BF16 or analytically bounded FP16 multiplicands and FP32
  accumulators; reduced-precision accumulation is outside the contract. Primal
  and transpose-dual implementations must realize the same mathematical factor
  actions and pass pairing and composed-VJP gates; they need not share a private
  packing or instruction schedule.
- Keep structural zero distinct from algebraic cancellation. Explicit identity
  geometry, zero strength, invalid padding, and disabled components must remain
  exact. A nonstructural zero produced by cancellation is accepted through the
  fixed per-dtype `q2`, chart-action, and VJP gates in
  `docs/VALIDATION_PLAN.md`; do not add a data-dependent threshold or fallback.
- An optimized solve must state its algebraic equivalence before adoption.
  Approximate solves are experiments until forward, state, and gradient error
  are bounded over the normal training envelope.
- Benchmarks measure implementations; they never define model semantics.

## Investigation before invention

- Reproduce and localize a failure before adding a numerical mechanism.
- For numerical linear algebra, optimization, CUDA, PyTorch behavior, or known
  architectures, search primary papers, official documentation, upstream
  source, and issue trackers before inventing a replacement.
- Exploratory scripts, downloaded papers, generated data, and benchmark output
  stay outside Git.
- Record research that materially changes the design in `docs/PRIOR_ART.md`.

## Scope discipline

- Do not reintroduce H-only, SFDD, single-edit SFDG, standalone DeltaProduct,
  DPLR geometry, co-decayed-ridge geometry, piecewise-constant geometry, or
  inverse-state approximation as competing model tracks.
- A diagnostic ablation may disable a component of SolveDelta, but it must not
  create another maintained architecture.
- Ordinary Delta/GDN2 identity-geometry reductions are mandatory tests.
- PDN and other preconditioned Delta rules are external baselines; a bare
  `G^-1 k` write is not the contribution.

## Implementation order

1. freeze the SolveDelta FP64 token recurrence in `causallsso/reference.py`;
2. prove and test GDN2 and DeltaProduct-`K` reductions;
3. establish Prefix-LSSO provenance diagnostics and independently test the
   causal chart property contract;
4. add masks, resets, packed sequences, and recurrent decoding semantics;
5. derive exact geometry-prefix and associative chunk algebra;
6. compare every optimized forward and gradient with the token oracle;
7. benchmark the complete BF16/FP16/FP32 layer at the resolved `r = d_k`, including
   the first native `r = 128` target profile and at least one non-128 reference
   width;
8. integrate the selected Triton--CUDA--FLA path: Triton owns prefix
   boundaries, the chunk-owned CUDA operator owns bounded frame actions, FLA
   owns the mature scan/WY exterior, and MathDx remains the exact triangular
   oracle and decode candidate without changing the chart.

Optimized candidates may be written and benchmarked while later envelope tests
are still being added. They may replace the current path only after the hard
semantic and production-observable gates relevant to their supported surface
pass. Private diagnostics guide that iteration; they do not freeze an older
kernel's arithmetic structure.
