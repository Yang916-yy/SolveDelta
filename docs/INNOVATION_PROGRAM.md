# Why SolveDelta

SolveDelta is built around a simple idea: a recurrent memory should be able to
learn the coordinate system in which it writes. The current operator treats
that geometry as an online least-squares problem, stores its evolving solution
in a full matrix. The accumulated solution directs the primal write, while the
latest solve residual controls the dual erase and read.

The executable definition is `causallsso/reference.py`. This document explains
the model behind that recurrence and the choices that make it useful.

## Prefix geometry as a linear solve

At token `t`, the geometry branch observes a normalized direction `u_t` and a
target `h_t`. A prefix of these observations conceptually defines

```text
J_t = J_{t-1} + u_t u_t^T
G_t = G_{t-1} + h_t u_t^T.
```

The least-squares objective

```text
L_t(C) = 1/2 sum_{s<=t} ||h_s - C u_s||^2
```

has normal equation

```text
C J_t = G_t.
```

`J_t` is a second moment and `G_t` is a cross moment. They explain what the
geometry branch is solving, but they are not recurrent state. SolveDelta keeps
the current solution estimate `C` and applies one leaky normalized-LMS step per
observation:

```text
D_t     = Diag(exp(log_alpha_t))
r_t     = h_t - C_{t-1} u_t
delta_t = gamma_t r_t
C_t     = C_{t-1} D_t + delta_t u_t^T.
```

The residual is evaluated against the complete old solution. Forgetting is
then applied only to historical `C`, while `delta_t` is written without
attenuation. The per-head bias initializes `gamma` at `0.9`, so useful geometry
is exposed early while the token projection can learn a different relaxation
rate. `D_t` is the complete existing DeltaRule decay and introduces no second
forgetting controller, mean reduction, or time-scale parameter. It acts on
the source/address axis, so the accumulated primal history obeys
`(C D)^T k = D C^T k`. Coordinatewise turnover is a recurrent regularizer on
the carried solution, not a claim that production explicitly carries or
exactly solves a decayed moment system. Moving the decay before residual
evaluation would alter the instantaneous fitting error; the two forms are not
equivalent over long sequences.

The update is deliberately recurrent. `C_t` is an online solution shaped by
observation order, learned relaxation, and memory-controlled turnover rather
than the closed-form batch solution recomputed at every prefix. That temporal
asymmetry is meaningful for language: a recent correction can change how a
repeated key is interpreted while stale geometry gradually leaves the state.

Although each observation contributes rank one, `C` has `r^2` state capacity.
A sequence of diverse directions can populate the full matrix. The model thus
combines a full-matrix history with token work that maps naturally to chunked
pair, WY, and state contractions.

## Accumulated primal and residual-local dual

The predictor update and the two address paths have different jobs. The
unattenuated update is accumulated in the solution matrix and defines

```text
P_t = I + C_t
d_t = P_t^T k_t.
```

Thus every earlier residual can affect a later primal write. No inverse of
`P_t` or `C_t` is needed. For the dual path, the current update is mapped to a
bounded covector:

```text
rho   = 5/8
phi_t = rho delta_t / sqrt(rho^2 + ||u_t||^2 ||delta_t||^2)
F_t   = I + u_t phi_t^T.
```

The radial map is smooth and identity-like near zero. Its static bound
`||u_t|| ||phi_t|| < rho` gives

```text
3/8 < 1 + phi_t^T u_t < 13/8
||F_t||_2 <= 13/8
||F_t^-1||_2 <= 8/3
kappa_2(F_t) <= 13/3.
```

The inverse-transpose action follows directly from Sherman-Morrison:

```text
F_t^-T x = x - phi_t (u_t^T x) / (1 + phi_t^T u_t).
```

This factor is relative to the current token. The persistent history lives in
the primal solution `C`; no second dense inverse state is needed at a chunk
boundary.

## Asymmetric Delta addresses

Let `k_t` be the normalized edit key, `b_t` the independently gated erase
covector, and `q_t` the normalized query. SolveDelta uses accumulated history
for the primal direction and the current innovation for dual directions:

```text
d_t   = (I + C_t)^T k_t
e_t   = F_t^-T b_t
chi_t = F_t^-T q_t.
```

This is intentionally not a local similarity pair. Requiring both paths to use
the same residual-local factor would discard most of the accumulated solution;
requiring the dual to use `(I+C_t)^-1` would reintroduce the conditioning and
inverse-state problem that the residual formulation removed. The selected
asymmetry gives the stable full-history matrix to the non-inverting primal path
and keeps the inverse-transpose action rank one and residual aligned.

## Memory and structural reduction

The value memory uses coordinate-wise decay, independent write/erase gates,
and the transported sources:

```text
S'_t = Diag(exp(log_alpha_t)) S_{t-1}
z_t  = write_t * v_t
S_t  = S'_t + d_t (z_t - S'_t^T e_t)^T
o_t  = S_t^T chi_t.
```

Decay and output gating follow the KDA/GDN2 low-rank parameterization. The
readout is standard sigmoid-gated RMSNorm, selected by matched training at the
current high-relaxation initialization.

From the zero continuation state, a zero geometry rate gives `C=0`,
`delta=phi=0`, `P=F=I`; the memory recurrence is then the ordinary GDN2
edit/read at finite parameters. This gives a clean experimental interpretation:
SolveDelta adds a prefix-conditioned coordinate solver to a familiar Delta
memory.

## What the added state can represent

GDN2 can distinguish writes through their current keys, values, gates, and
memory. SolveDelta can additionally distinguish them through preceding
`u -> h` relationships. Two identical edit keys may therefore act differently
after different geometric histories.

The most relevant workloads are those where the meaning of an edit depends on
context accumulated before it:

- repeated or conflicting keys;
- contextual and lexical ambiguity;
- instruction replay and correction;
- structured long-range associations;
- retrieval in a moving latent basis.

These are targeted hypotheses. Current small-scale language-model evidence
shows competitive average loss and a positive LAMBADA NLL signal, while longer
and multi-seed evaluation remains open. The measured results are reported in
`docs/RESULTS.md`.

## Why the implementation is practical

The mathematics lands on established GPU building blocks:

| SolveDelta block | Mature execution form |
| --- | --- |
| leaky normalized-LMS predictor | gated-Oja pair/WY/state |
| bounded `phi` | scaled L2 reduction |
| accumulated primal action | gated-Oja chunk output owner |
| residual-local `F^-T` actions | rank-one dual source owner |
| decayed Delta memory | generalized-DPLR pair/WY/state/output |
| coordinate decay | KDA low-rank gate |
| readout | FLA sigmoid-gated RMSNorm |

The production path specializes FLA's ownership and transpose schedules around
SolveDelta's operands. It keeps matrix contractions on Tensor Cores, sensitive
reductions and continuation states in FP32, and kernel boundaries where they
preserve useful chunk/rank/value parallelism.

At the documented `B=1,T=1024,H=8,r=V=128` profile, the complete projected
mixer costs about `18%` more forward latency and `12%` more F+B latency than
the same-shape FLA GDN2 mixer. Those historical numbers predate the accumulated
primal change and must be remeasured before they are used for comparison.

## Open questions

- `rho=5/8` sets the current balance between frame strength and conditioning;
  larger-scale training may favor another static budget.
- One rank-one solver step per token may be enough for language modeling, or a
  second structured observation may justify its cost on targeted tasks.
- C16-aligned dense training remains the fastest surface. Irregular tails and
  reset-free packed segments are now native but still need broader shape and
  reset-density performance characterization.
- The model's strongest case is history-conditioned editing. Evaluation should
  increasingly emphasize tasks that isolate that capability.

These questions define the next experiments without changing the current
operator contract.
