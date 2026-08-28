# SolveDelta Research Program

This document explains the current RLS SolveDelta operator. It does not define
a second recurrence; executable mathematics belongs to
`causallsso/reference.py`.

## Geometry as online regression

The states

```text
J_t = lambda_t J_{t-1} + u_t u_t^T
D_t = lambda_t D_{t-1} + u_t h_t^T
```

are a decayed covariance and cross moment. Their natural coordinate is

```text
C_t = J_t^-1 D_t.
```

With `g_t=J_t^-1u_t`, Sherman-Morrison gives the exact RLS update

```text
C_t = C_{t-1} + g_t (h_t - C_{t-1}^T u_t)^T.
```

This converts a dense geometry change into a rank-one innovation. The
effective mass

```text
m_t = lambda_t m_{t-1} + 1
```

tracks the normalization of the covariance state and supplies the matching
history transport.

## Moving-state Delta rule

The memory is transported by two identity-plus-rank-one factors before the
ordinary Delta edit:

```text
history transport -> channel decay -> RLS innovation transport -> Delta edit
```

Each factor is a generalized-Delta update. Therefore a token is represented by
three fixed internal slots, allowing FLA's DPLR pair/WY/state/output schedules
to be specialized without exposing a synthetic `3T` sequence.

This is the current model, not an algebraically exact implementation of the
archived bounded-LDU chart. It gives up that chart's full local differential
rank in exchange for rank-one transport structure and practical latency. The
expressivity tradeoff must be evaluated by training, not hidden behind the
older model's claims.

## Exact reductions

At finite `gamma=0`, both geometry transports become identity. The memory path
is then exactly the ordinary gated Delta edit/read, including normalized key,
paired erase covector, channel decay, and post-edit query. This is the required
GDN2 reduction.

The fixed RLS prior keeps `J` SPD and the gain defined from the first token.
Masks preserve state; resets restore that prior. Recurrent splits pass
`(m,J,D,S)` without rounding in native execution.

## Engineering thesis

The implementation should contain as little operator-specific GPU machinery as
possible:

- covariance/cross-moment recurrence is MESA state scan;
- gain is MESA matrix-free CG and implicit transpose;
- effective mass is a scalar affine scan;
- three rank-one updates are generalized-DPLR pair/WY/state/output;
- normalization and gates use FLA primitives;
- conv4 and SiLU use causal-conv1d.

The SolveDelta-specific work is the algebraic source mapping, native E3 slot
ownership, and the exact composition of the transposes. Performance work must
retain mature parallel axes. Full fusion is not a goal when it reduces chunk,
rank, or value-tile concurrency.

## Open limitations

- CG5 approximates the exact FP64 gain action and must remain inside frozen
  BF16 output/state/VJP gates.
- The public fused projection currently needs packed-vector canonicalization
  before MESA/E3 kernels.
- The selected block-E3 reverse keeps substantial forward cache and FP32
  partial storage.
- The core remains materially slower and larger than matched GDN2 because it
  pays for two real geometry transports plus covariance/cross-moment state.
- Masks and resets have current reference semantics but no optimized packed
  RLS native schedule.

These are current facts, not invitations to restore archived paths or add
runtime backend selectors.
