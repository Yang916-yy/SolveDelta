# SolveDelta Research Program

This document explains the current relative Residual-Frame operator. It does
not define a second recurrence; executable mathematics belongs to
`causallsso/reference.py`.

## Residual geometry

SolveDelta observes a normalized direction `u_t` and target `h_t`. Its
geometry state is an online linear predictor:

```text
r_t = h_t - C_{t-1} u_t
delta_t = gamma_t r_t
C_t = C_{t-1} + delta_t u_t^T.
```

This is normalized LMS, or equivalently an ungated Oja-style residual update
under a change of variable names. Since `||u_t||=1`, the residual on the
observed direction immediately becomes `(1-gamma_t)r_t`. A large residual
writes quickly, a small residual writes little, and directions orthogonal to
`u_t` are unaffected.

Unlike the former RLS route, there is no covariance that is globally
multiplied by a forgetting factor and later inverted. The model therefore has
no persistent-excitation requirement, SPD prior, covariance windup, or CG
accuracy contract. History is stored directly in the predictor's solution
coordinates.

The predictor still has full `r^2` state capacity. Each token contributes one
rank-one update; products and sums across tokens can populate the full matrix.
This is a statement about reachable state rank, not a claim that one token has
`r^2` instantaneous degrees of freedom.

## Relative frame

The current operator does not use `P_t=I+X_t` as a dense accumulated frame.
Instead, the current residual generates one relative factor:

```text
rho = 5/8
phi_t = rho delta_t / sqrt(rho^2 + ||u_t||^2 ||delta_t||^2)
F_t = I + u_t phi_t^T.
```

The predictor still writes the unmodified `delta_t`; the smooth radial map is
only the frame parameterization. It is identity to first order, preserves the
residual direction, and bounds the full rank-one perturbation rather than only
repairing its parallel component.

Its inverse transpose is exact by Sherman-Morrison:

```text
F_t^-T x =
    x - phi_t (u_t^T x) / (1 + phi_t^T u_t).
```

Because `||u_t|| ||phi_t|| < 5/8`, the denominator lies strictly between
`3/8` and `13/8`; moreover `||F_t^-1||_2 <= 8/3` and
`kappa_2(F_t) <= 13/3`. These are model-level analytic bounds, not observed
precision tolerances.

The ordinary edit key, erase covector, and query are mapped as

```text
d_t   = F_t k_t
e_t   = F_t^-T (erase_t * k_t)
chi_t = F_t^-T q_t.
```

This gives the exact local identities

```text
e_t^T d_t = (erase_t*k_t)^T k_t
I-d_t e_t^T = F_t (I-k_t(erase_t*k_t)^T) F_t^-1.
```

The residual predictor can learn gradually while every realized edit remains
an exact similarity transform. No inverse frame is carried across chunks.

## Memory rule

After channel decay, the memory uses the transformed Delta edit:

```text
S'_t = Diag(exp(log_alpha_t)) S_{t-1}
z_t = write_t * v_t
S_t = S'_t + d_t (z_t - S'_t^T e_t)^T
o_t = S_t^T chi_t.
```

At `gamma_t=0`, `delta_t=phi_t=0` and `F_t=I`. If the whole layer has
`gamma=0`, `C` remains fixed and the memory path is exactly GDN2's independent
channel-wise erase/write Delta edit/read. This is a finite-parameter structural
reduction, not a limit argument.

The dynamic coordinate decay and output gate use GDN2/KDA's low-rank
projections. Decay retains a distinct value for every key coordinate, while a
head-wise positive rate supplies its static scale. A plain RMSNorm readout
removes the component of the core gradient parallel to its output. SolveDelta
instead retains a bounded radial coordinate:

```text
r = sqrt(mean(o^2)+eps),  r0 = 1/sqrt(V)
z = (r-r0)/(r+r0)
alpha_h = sigmoid(2*a_h)-1/2
y = (1+alpha_h*z) RMSNorm(o) sigmoid(gate).
```

This is still scale-controlled because the learned multiplier lies in
`(1/2,3/2)`, but it does not make the geometry-dependent output magnitude
completely unobservable. The radial path has an exact composed transpose and
adds only one scalar parameter per head. SolveDelta's principal added model
content remains the residual predictor and relative frame.

## Relationship to GDN2

SolveDelta deliberately keeps GDN2's ordinary channel-wise Delta edit, KDA
coordinate decay, and gated normalized readout. Setting the entire geometry
rate to finite `gamma=0` leaves `C` unchanged, makes `F=I`, and recovers that
memory path exactly. Geometry therefore adds one capability rather than
replacing the baseline memory rule:

```text
GDN2:       prefix memory -> ordinary Delta edit/read
SolveDelta: prefix predictor -> relative coordinates -> same Delta edit/read
```

The added state is a full `r x r` map learned through rank-one residual writes.
It can make two otherwise similar edits act differently when their preceding
`u -> h` relationships differ. This is most plausibly useful when retrieval or
editing depends on history-conditioned coordinates: repeated or conflicting
keys, contextual ambiguity, instruction replay, and structured long-range
associations. These are hypotheses for targeted evaluation, not claims that
average language-model loss must improve.

The implementation cost is equally concrete: extra `u/h/gamma` projections,
the predictor forward/transpose, and three relative frame actions. Current
same-shape CUDA Graph measurements put the complete projected mixer about
`18%` slower forward and `12%` slower F+B than FLA GDN2. See
`docs/RESULTS.md` for the scoped numbers and exploratory training evidence.

## Why it maps to mature kernels

The predictor is FLA gated Oja with renamed operands and no vector decay:

```text
target <- h
source <- u
beta   <- gamma
state  <- C.
```

Within a chunk it is pair GEMM, unit-lower WY solve, and state GEMM. The
relative frame is identity plus rank one, its radial map is scaled L2Norm, and
the memory update is a generalized-Delta/DPLR transition. Its forward and
strict transpose therefore reuse the same norm, pair, WY, state, output, and
output-owner reverse families used by FLA. Decay reuses KDA's low-rank gate.
Output gating specializes FLA's gated RMSNorm owner in place, reusing its
resident `rstd` and strict transpose rather than introducing an elementwise
HBM chain or a tokenwise dense solve.

This equivalence concerns concrete computation blocks. SolveDelta still owns
the operand mapping, local similarity contract, composition, public state, and
precision boundaries.

## Expressivity

Residual-Frame and RLS can write the same one-token predictor update direction
when their gains coincide, but they select that direction differently:

- RLS rotates and scales the observation by an inverse covariance;
- Residual-Frame writes the observed normalized direction directly with a
  learned token-local rate.

Residual-Frame may need repeated correlated observations where an ideal RLS
gain would take a larger one-step move. In exchange, it does not lose a prior
or unobserved subspace through repeated decay, and it avoids making
invertibility of a decayed covariance part of model semantics.

The archived bounded-LDU operator had full-rank instantaneous chart
derivatives. Residual-Frame does not reproduce that operator. Its hypothesis is
that full predictor state accumulated through fast rank-one residual writes,
plus exact local primal/dual actions, is a better quality/latency trade.

## Open limitations

- The bounded frame attenuates unusually large predictor updates while leaving
  the predictor state write unchanged. Whether `rho=5/8` is the best
  quality/conditioning budget remains an empirical model question.
- The optimized dense path currently requires C16-aligned sequence lengths.
  Masks, resets, and irregular segments use the reference path.
- The predictor uses one rank-one write per token. Whether a second residual
  slot improves quality enough to justify its cost is an empirical model
  question, not an implementation equivalence.
- Compared with GDN2, SolveDelta pays for the predictor and relative source
  actions. Their value must be established by matched training, not by kernel
  novelty or state-rank arguments alone.

These limitations are research questions. They are not invitations to restore
RLS, bounded-LDU, inverse state, or runtime backend selectors.
