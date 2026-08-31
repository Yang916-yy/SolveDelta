# SolveDelta Native Execution Blueprint

This document maps the Residual-Frame recurrence in
`causallsso/reference.py` onto its production forward and reverse owners. It is
an implementation guide: tensor contracts, schedules, precision, and
lifetimes live here; the executable mathematics remains in the FP64 oracle.

## 1. Public surface

For batch `B`, length `T`, heads `H`, key width `r`, and value width
`V`:

```text
u,h,q                 [B,T,H,r]
k,v                   [B,T,H,1,r] / [B,T,H,1,V]
log_alpha             [B,T,H,r]
erase_raw,write_raw   shapes of k,v
gamma                 [B,T,H]
C0,S0                 [B,H,r,r], [B,H,r,V]
output                [B,T,H,V]
```

The native vector operands and output are BF16. `log_alpha`, `gamma`, and
both continuation states are FP32. `K=1`; the selected schedules use C32 for
the predictor and C16 for the memory exterior. Chunk size and private packing
remain internal tuning choices.

The geometry solution is stored in the orientation

```text
prediction = C u.
```

`C` is the online solution of the prefix fitting problem. The zero
continuation is `C0=0,S0=0`.

## 2. Token recurrence

Normalize `u`, `q`, and `k` in the last dimension. Activate GDN2's independent
channel-wise gates as

```text
b = sigmoid(erase_raw) * k
z = sigmoid(write_raw) * v.
```

The residual predictor update is

```text
r = h - C_prev u
delta = gamma r
C = C_prev + delta u^T.
```

Because `u` is normalized, the just-observed residual becomes
`(1-gamma)r`, while orthogonal source directions keep their previous value.
The recurrence advances the least-squares solution directly in `C`.
The per-head geometry bias initializes to `logit(0.9)=log(9)`, so the
predictor begins as a high-relaxation normalized-LMS solver while the
token-local projection remains free to learn smaller or larger rates.

The token-local relative frame and its exact inverse-transpose action are

```text
rho = 5/8
phi = rho delta / sqrt(rho^2 + ||u||^2 ||delta||^2)
F = I + u phi^T
den = 1 + phi^T u
F^-T x = x - phi (u^T x) / den.
```

The predictor update `delta` is unchanged. Only the token-local frame covector
is radialized. Since `||u|| ||phi|| < rho`, the model has the static bounds

```text
3/8 < den < 13/8
||F||_2 <= 13/8
||F^-1||_2 <= 8/3
kappa_2(F) <= 13/3.
```

These bounds follow from the smooth model parameterization itself.

The three public source actions are

```text
d   = F k
e   = F^-T b
chi = F^-T q.
```

Therefore

```text
e^T d = b^T k
I - d e^T = F (I - k b^T) F^-1.
```

This is an exact local similarity transform. `F_t` is the relative factor for
the current solve residual; prefix history remains in `C_t`.

The memory recurrence is

```text
S_decay = Diag(exp(log_alpha)) S_prev
S = S_decay + d (z - S_decay^T e)^T
o = S^T chi.
```

At finite `gamma=0`, `delta=phi=0`, `F=I`, `C` is unchanged, and the memory
path is exactly the ordinary gated Delta edit/read.

## 3. Chunk predictor

For one chunk, stack source rows in `U`, target rows in `H`, and write rates
in diagonal `Gamma`. Define `D_gamma=Gamma U`. With entry predictor `C0`,

```text
W = I + tril(U D_gamma^T, -1)
R = W^-1 (H - U C0^T)
C_out = C0 + (D_gamma^T R)^T.
```

The stored implementation uses `C=X^T`, so equivalent transposes may appear
in code. The production owner is a specialization of FLA gated Oja:

```text
FLA key    <- h
FLA value  <- u
FLA beta   <- gamma
FLA state  <- C.
```

Its pair GEMM, unit-lower triangular WY solve, state GEMM, chunk boundaries,
and reverse schedule are retained. Forward emits the token-local `delta` panel
and final FP32 `C`; reverse applies `W^-T` and returns final-shaped cotangents
for `h,u,gamma,C0`. The specialization drops FLA's constant-zero vector-gate
panel and reads strided target views directly. At `r=128`, reverse owns 32
predictor rows per CTA, doubling CTA parallelism relative to the 64-row donor
schedule and lowering register/shared-memory pressure.

## 4. Relative source owner

The source owner consumes normalized `u`, raw strided `q/k`, predictor update
`delta`, raw erase/write logits, and value. It computes q/key L2 norms and
generates and discards `phi` in the same CTA:

```text
phi = rho delta / sqrt(rho^2 + ||delta||^2)  # ||u||=1
den = 1 + phi^T u
d = k + u (phi^T k)
e = b - phi (u^T b) / den
chi = q - phi (u^T q) / den
z = sigmoid(write_raw) * v.
```

The upstream L2Norm owner establishes `||u||=1`, making the shortened radial
formula algebraically exact in the composed operator. Its purely radial
`bar_u` term lies in the null direction of the same owner's strict transpose.
This identity avoids a repeated BF16-rounded norm and slightly improves the
production-observable oracle/VJP error.

Q/key norm reductions, frame actions, and erase/write sigmoid arithmetic are
evaluated in FP32. The bounded frame gives a static range proof for the private
`d/e/chi` panels, so the source owner writes FP16 directly from FP32 registers.
Products with the unbounded decay exponent are materialized later in BF16 to
retain exponent range:

```text
d           [panels,1,C16,r] FP16
paired e,q  [panels,2,C16,r] FP16
z           [B,T,H,V]        BF16.
```

The panel layout is produced in consumer order, eliminating a token-major
`cat/permute/contiguous` handoff. The source transpose consumes the matching
panel cotangents, closes the shared denominator once, applies the
scaled-L2Norm transpose for `phi`, and returns independent erase/write
cotangents. Radial and gate scalars stay resident.

The model frontend uses KDA's low-rank coordinate-decay parameterization:

```text
decay_hidden = W_decay_in x                 [d_gate]
decay_raw = W_decay_out decay_hidden        [H,r]
log_alpha = -exp(A_log[h]) softplus(decay_raw + dt_bias[h,r]).
```

`d_gate` is the resolved value-head width. The first low-rank projection is a
slice of the main fused input projection; only the narrow expansion is a
separate GEMM. The core output uses the matching KDA readout owner:

```text
gate_hidden = W_gate_in x                   [d_gate]
gate_raw = W_gate_out gate_hidden           [H,V]
y = RMSNorm(o) sigmoid(gate_raw).
```

CUDA uses FLA `fused_kda_gate` and FLA's fused sigmoid RMSNorm-gate row owner
with its strict transpose. The CPU/FP64 path evaluates the same formula
directly. FLA's norm-linear lifetime policy regenerates the normalized output
during reverse for the ordinary Tensor Core weight-gradient GEMM, saving a
2 MiB checkpoint panel. The row owner and GEMM remain separate to preserve
their natural parallelism.

## 5. Memory exterior

The exterior maps directly to FLA generalized DPLR with

```text
q = chi,  k = d,  a = -exp(log_alpha) e,  b = d,  v = z,  scale = 1.
```

FLA applies its low-rank term to the pre-decay state. Multiplying `e` by
`exp(log_alpha)` aligns the erase with `S_decay`. The pair owner forms this
scaled source from `-e` and the inclusive decay prefix while the tile is
resident.

Production ownership is:

1. an FP32 chunk cumsum for channel decay;
2. a C16 exact-unbounded Triton specialization of FLA's scalar direct-`e`
   pair owner for strict edit/read interactions;
3. FLA fast-WY preparation and unit-lower action;
4. FLA chunk boundary state and chunk-parallel output owners;
5. the matching output/state, WY, pair, and decay transposes.

The state owner advances FP32 chunk-boundary `S`; eligible predictor-pair and
DPLR WY/state/output contractions use low-precision multiplicands with FP32
accumulation. The direct-`e` pair evaluates its unbounded causal exponent
products and reductions in FP32. Output is written BF16 and a requested final
`S` is FP32.

Fusion follows ownership. Source generation and panelization share one CTA and
remove a full HBM boundary. Predictor, pair/WY, state, and output retain
separate kernels where chunk/rank/value CTA parallelism is worth more than a
small handoff.

## 6. Reverse graph

Backward is the strict transpose of the composed forward blocks:

1. output/state reverse walks chunk boundaries backward and consumes
   `bar_o,bar_S_final`;
2. fast-WY reverse applies the transpose triangular action;
3. the direct-`e` pair transpose owns final source tiles and closes decay;
4. the source transpose returns `bar_u,bar_delta,bar_q,bar_k,bar_v` and the
   independent raw erase/write cotangents;
5. the gated-Oja transpose combines `bar_delta` with `bar_C_final` and
   returns `bar_h,bar_u,bar_gamma,bar_C0`;
6. the source transpose applies q/key L2Norm transposes, while the standalone
   u L2Norm transpose returns the remaining gradient to its strided view.

Each reverse owner mirrors one forward block and emits final-shaped
cotangents. Contributions to shared `u` close after the source and predictor
owners have both completed.

The pair and source transposes accumulate locally in FP32. Their
single-consumer, final-shaped BF16 handoff reduces this backward interface from
12 MiB to 6 MiB at the optimized profile.

## 7. Precision map

```text
BF16: public vector operands/output; decay-scaled DPLR, WY, and state multiplicands;
      final-shaped pair-to-source cotangent handoff
FP16: statically bounded source-native d/e/chi panels, written directly from FP32
FP32: C,S,erase/write gates,gamma,log_alpha,norm/radial and denominator reductions, divisions,
      Tensor Core accumulators, backward partials, continuation boundaries
FP64: token oracle only
```

The relative denominator uses its analytic `3/8` lower bound and an FP32 norm
reduction. Private FP16 panels are admitted with a static range proof and a
direct FP32 producer.

## 8. Layout and lifecycle

Public fused-projection vector views may have arbitrary outer strides and
require unit innermost vector stride. Native owners pack normalized or WY
panels privately. The fused input projection pads its physical output row to a
multiple of 64 while exposing the logical prefix. At the optimized profile,
`7432 -> 7488` improves the Tensor Core projection transpose and satisfies
causal-conv1d's multiple-of-eight stride requirement.

Fixed-shape Graph training may bind an optimizer-step BF16 shadow for every
FP32 Linear master. The shadows are nonpersistent buffers and their strict
Linear transpose returns FP32 master gradients. An optimizer post-step hook
refreshes all shadows once, outside capture; replay rejects a stale shadow.
This is selected only for gradient accumulation because factor-one refresh
cost exceeds the saved casts.

The recurrent operator state is FP32 `(C,S)`. Forward caches follow a simple
rule: keep a panel when complete F+B beats local recomputation under the same
VJP gate.

Dense native currently requires lengths aligned to C16. Masks and resets run
the same recurrence through the model reference path.

## 9. Acceptance and benchmark

Hard gates:

- FP64 predictor recurrence and exact relative-frame action;
- adversarial verification of the `3/8` denominator lower bound;
- exact local similarity and finite `gamma=0` GDN2 reduction;
- independent erase/write activation and KDA gated-output formula;
- masks, resets, recurrent splits, `K=1`, and a non-128 reference width;
- complete initial/final `(C,S)` composed VJP.

Production-observable gates:

- BF16 output and FP32 final `(C,S)` against quantized-input FP64 oracle;
- complete VJP for every public input and state endpoint;
- dense projected layer forward/backward from real strided projection views;
- fixed-shape CausalLM CUDA Graph loss and gradients against eager.

At `B1,T1024,H8,r=V=128`, BF16 public operands and FP32 state, the
panel-native production composition measured:

```text
eager forward median/p95  0.412 / 0.538 ms
eager F+B median/p95      1.476 / 1.663 ms
Graph forward median/p95  0.173 / 0.178 ms
Graph F+B median/p95      0.610 / 0.785 ms
Graph allocated           58.1 MiB
```

These are core-operator measurements on the development RTX 5070 Ti, not
complete mixer or causal-LM numbers. Benchmark reports must continue to state
shape, dtype, device, scope, and execution mode.

The complete projected mixer at the same shape, using FP32 master parameters
and BF16 autocast, measured Graph forward `0.428/0.614 ms` median/p95 and F+B
`1.437/1.632 ms`, with `245.8 MiB` Graph allocation. The synchronized p95
samples in this run include the same roughly `0.17 ms` P-state tail observed
for the matched GDN2 run. The mixer has `8.99M` parameters versus `9.46M`
before low-rank decay and output gating.

The matched FLA GDN2 comparison and exploratory model evidence are maintained
in `docs/RESULTS.md`; they do not alter this implementation contract.

Model-level capture uses `SolveDeltaGraphedTrainingStep` around a fixed-shape
loss-only CausalLM wrapper. The trainer owns optimizer steps, clipping,
gradient accumulation, distributed reduction, and recapture policy.
