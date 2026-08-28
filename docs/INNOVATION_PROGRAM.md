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

The model does not initialize every head at one identical history scale. At
zero projected gate input, multiple heads are deterministically spread from
`lambda=0.985` to `0.995`; a single head uses `0.99`. Their initial EMA-style
effective horizons therefore span roughly `67` to `200` tokens instead of
collapsing to one horizon near `100`. This distribution keeps the SPD prior
alive while the rank-128 state acquires geometry, but it is not a bound: the
ordinary learned gate remains token-dependent.

## Moving-state Delta rule

The memory is transported by two identity-plus-rank-one factors before the
ordinary Delta edit:

```text
history transport -> channel decay -> RLS innovation transport -> Delta edit
```

Each factor is a generalized-Delta update. Therefore a token is represented by
the ordered composition of two geometry transports and one ordinary edit.
This composition does not change the model's public token axis.

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
Masks preserve state, resets restore that prior, and recurrent splits carry the
same continuation state into the next segment.

## Open limitations

- The native CG5 action approximates the exact FP64 gain solve. Its usefulness
  depends on remaining inside the declared BF16-observable output, state, and
  composed-VJP envelope.
- Two rank-one RLS transports have less local differential rank than the
  archived bounded-LDU chart. Whether the cheaper structure is sufficient is
  an empirical training question.
- Compared with ordinary GDN2, SolveDelta necessarily pays for covariance,
  cross-moment, effective-mass, and two geometry transports. Their value must
  be justified by model quality, not kernel novelty alone.
- Masks, resets, and recurrent cache semantics are defined, but optimized
  packed training and decode remain open implementation work.

These are current facts, not invitations to restore archived paths or add
runtime backend selectors.
