# Why SolveDelta

SolveDelta is built around a simple idea: a recurrent memory should be able to
learn the coordinate system in which it writes. The current operator treats
that geometry as an online least-squares problem, stores its evolving solution
in a full matrix, and uses the latest solve residual to transport an ordinary
Delta edit.

The executable definition is `causallsso/reference.py`. This document explains
the model behind that recurrence and the choices that make it useful.

## Prefix geometry as a linear solve

At token `t`, the geometry branch observes a normalized direction `u_t` and a
target `h_t`. Over a prefix, these observations define

```text
J_t = sum_{s<=t} u_s u_s^T
G_t = sum_{s<=t} h_s u_s^T.
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
the current solution estimate `C` and applies one normalized-LMS step per
observation:

```text
r_t     = h_t - C_{t-1} u_t
delta_t = gamma_t r_t
C_t     = C_{t-1} + delta_t u_t^T.
```

This is the negative instantaneous gradient of the least-squares objective.
With `||u_t||=1`, the residual on the direction just observed becomes
`(1-gamma_t) r_t`; directions orthogonal to `u_t` keep their previous value.
The per-head bias initializes `gamma` at `0.9`, so useful geometry is exposed
early while the token projection can learn a different relaxation rate.

The update is deliberately recurrent. `C_t` is an online solution shaped by
observation order and learned relaxation, rather than the closed-form batch
solution recomputed at every prefix. That temporal asymmetry is meaningful for
language: a recent correction can change how a repeated key is interpreted
without erasing unrelated directions.

Although each observation contributes rank one, `C` has `r^2` state capacity.
A sequence of diverse directions can populate the full matrix. The model thus
combines a full-matrix history with token work that maps naturally to chunked
pair, WY, and state contractions.

## Turning a solve residual into coordinates

The predictor update and the coordinate transform have different jobs.
`delta_t` is written into `C` without attenuation. For the token-local frame,
the same update is mapped to a bounded covector:

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
`C`; no second dense inverse state is needed at a chunk boundary.

## Exact transport of a Delta edit

Let `k_t` be the normalized edit key, `b_t` the independently gated erase
covector, and `q_t` the normalized query. SolveDelta applies the frame in
primal and dual directions:

```text
d_t   = F_t k_t
e_t   = F_t^-T b_t
chi_t = F_t^-T q_t.
```

The pairing is preserved exactly:

```text
e_t^T d_t = b_t^T k_t.
```

More strongly, the erase operator is a similarity transform:

```text
I - d_t e_t^T = F_t (I - k_t b_t^T) F_t^-1.
```

The geometry solver may advance gradually, but every frame it realizes acts
exactly on the current edit. This separation is central to the design: online
learning determines the coordinates; primal/dual algebra preserves the Delta
rule inside those coordinates.

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

When the geometry rate is zero, `delta=phi=0`, `F=I`, and `C` remains fixed.
The memory recurrence then becomes the ordinary GDN2 edit/read at finite
parameters. This gives a clean experimental interpretation: SolveDelta adds a
prefix-conditioned coordinate solver to a familiar Delta memory.

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
| normalized-LMS predictor | gated-Oja pair/WY/state |
| bounded `phi` | scaled L2 reduction |
| `F` and `F^-T` actions | rank-one primal/dual source owner |
| decayed Delta memory | generalized-DPLR pair/WY/state/output |
| coordinate decay | KDA low-rank gate |
| readout | FLA sigmoid-gated RMSNorm |

The production path specializes FLA's ownership and transpose schedules around
SolveDelta's operands. It keeps matrix contractions on Tensor Cores, sensitive
reductions and continuation states in FP32, and kernel boundaries where they
preserve useful chunk/rank/value parallelism.

At the documented `B=1,T=1024,H=8,r=V=128` profile, the complete projected
mixer costs about `18%` more forward latency and `12%` more F+B latency than
the same-shape FLA GDN2 mixer. That gap is the current price of the full matrix
geometry solution and three relative-frame actions.

## Open questions

- `rho=5/8` sets the current balance between frame strength and conditioning;
  larger-scale training may favor another static budget.
- One rank-one solver step per token may be enough for language modeling, or a
  second structured observation may justify its cost on targeted tasks.
- Dense native execution is optimized for C16-aligned sequences; irregular
  segments currently use the reference recurrence.
- The model's strongest case is history-conditioned editing. Evaluation should
  increasingly emphasize tasks that isolate that capability.

These questions define the next experiments without changing the current
operator contract.
