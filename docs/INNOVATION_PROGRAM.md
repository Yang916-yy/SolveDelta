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
F_t = I + u_t delta_t^T.
```

Its inverse transpose is exact by Sherman-Morrison:

```text
F_t^-T x =
    x - delta_t (u_t^T x) / (1 + delta_t^T u_t).
```

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

At `gamma_t=0`, `delta_t=0` and `F_t=I`. If the whole layer has
`gamma=0`, `C` remains fixed and the memory path is exactly the ordinary
gated Delta edit/read. This is a finite-parameter structural reduction, not a
limit argument.

## Why it maps to mature kernels

The predictor is FLA gated Oja with renamed operands and no vector decay:

```text
target <- h
source <- u
beta   <- gamma
state  <- C.
```

Within a chunk it is pair GEMM, unit-lower WY solve, and state GEMM. The
relative frame is identity plus rank one, and the memory update is a
generalized-Delta/DPLR transition. Its forward and strict transpose therefore
reuse the same pair, WY, state, output, and output-owner reverse families used
by FLA rather than introducing a tokenwise dense solve.

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

- The scalar `1+delta_t^T u_t` has no structural positive lower bound.
  Training must monitor its distribution and the realized relative-frame
  condition number. Production must not hide failure with a clamp or fallback.
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
