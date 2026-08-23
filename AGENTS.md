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
- The associative state always remains in one fixed ambient basis. Do not move
  it into a time-varying solve domain. Local write vectors use the primal action;
  erase and read covectors use its exact dual action.
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
  `kappa_max = 1` bounds the orthogonal erase residual. They are fixed
  implementation details, not public model variants.
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
  similarity bounds, and full `r^2` local differential rank with respect to
  unconstrained ambient `(X^(H),X^(R))` chart coordinates. Do not test this as
  `dM/dR` under the structural `gamma_g=0` reduction, and do not conflate the
  chart property with feasible-prefix reachability: `rank(D_t), rank(R_t) <= t`,
  and arbitrary local `R` directions at nonzero geometry strength require a
  remembered geometry span of rank `r`.
- A chunkwise implementation must match the SolveDelta token oracle in outputs,
  every returned/chunk-boundary state, invariants, and gradients under the
  metric and per-dtype ceilings frozen in `docs/VALIDATION_PLAN.md`. Internal
  Triton scan and MathDx solve budgets are mandatory even when the composed
  layer passes its looser end-to-end ceiling. Do not add warning-only failures,
  CI relaxations, architecture-specific tolerances, or sequence-length-scaled
  allowances.
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
7. benchmark the complete layer at the resolved `r = d_k`, including the first
   native `r = 128` target profile and at least one non-128 reference width;
8. only then integrate the selected Triton--CUDA--FLA path: Triton owns prefix
   boundaries, the chunk-owned CUDA operator owns bounded frame actions, FLA
   owns the mature scan/WY exterior, and MathDx remains the exact triangular
   oracle and decode candidate without changing the chart.

No optimized kernel should precede these gates.
