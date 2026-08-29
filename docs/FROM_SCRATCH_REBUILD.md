# SolveDelta Native Blueprint

This document defines the sole current native implementation. Executable
mathematics belongs to `causallsso/reference.py`. The production operator is
relative Residual-Frame SolveDelta; the former bounded-LDU and RLS operators
are Git history, not fallbacks or compatibility targets.

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
both continuation states are FP32. `K=1`; predictor chunks use C32 and the
memory exterior uses C16. These chunk sizes are selected schedules, not model
semantics or public layout.

The predictor is stored in the orientation

```text
prediction = C u.
```

It is neither a covariance nor an inverse state. The zero continuation is
`C0=0,S0=0`.

## 2. Token recurrence

Normalize `u`, `q`, and `k` in the last dimension. Activate raw gates as

```text
b = 2 sigmoid(erase_raw) * k
z = 2 sigmoid(write_raw) * v.
```

The residual predictor update is

```text
r = h - C_prev u
delta = gamma r
C = C_prev + delta u^T.
```

Because `u` is normalized, the just-observed residual becomes
`(1-gamma)r`. Orthogonal source directions are unchanged. No full-state
forgetting factor, covariance, ridge, or linear solve is part of this model.

The token-local relative frame and its exact inverse-transpose action are

```text
F = I + u delta^T
den = 1 + delta^T u
F^-T x = x - delta (u^T x) / den.
```

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

This is an exact local similarity transform. `F_t` is a relative factor
generated from the current residual; it is not the accumulated absolute frame
`I+X_t`.

The memory recurrence is

```text
S_decay = Diag(exp(log_alpha)) S_prev
S = S_decay + d (z - S_decay^T e)^T
o = S^T chi.
```

At finite `gamma=0`, `delta=0`, `F=I`, `C` is unchanged, and the memory
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
and reverse schedule are retained. The unrelated Oja query/output branch is
absent. Forward emits the token-local `delta` panel and final FP32 `C`.
Reverse applies `W^-T` and returns final-shaped cotangents for
`h,u,gamma,C0`; it does not differentiate through a Python token loop.

## 4. Relative source owner

The source owner consumes normalized `u,q,k`, predictor update `delta`,
raw erase/write logits, and value. Per token it evaluates

```text
den = 1 + delta^T u
d = k + u (delta^T k)
e = b - delta (u^T b) / den
chi = q - delta (u^T q) / den
z = 2 sigmoid(write_raw) * v.
```

Erase/write sigmoid arithmetic is evaluated in FP32 and rounded at the public
BF16 gate boundary before multiplication. The owner writes exterior-native
panels directly:

```text
d           [panels,1,C16,r]
paired e,q  [panels,2,C16,r]
z           [B,T,H,V].
```

There is no token-major `d/e/chi` HBM ABI followed by
`cat/permute/contiguous`. The source transpose consumes the same panel
cotangents, closes the shared denominator once, and writes one gradient for
each public source.

## 5. Memory exterior

The exterior maps directly to FLA generalized DPLR with

```text
q = chi,  k = d,  a = e,  b = -d,  v = z,  scale = 1.
```

Production ownership is:

1. an FP32 chunk cumsum for channel decay;
2. a C16 TileLang direct-`e` pair owner for strict edit/read interactions;
3. FLA fast-WY preparation and unit-lower action;
4. FLA chunk boundary state and chunk-parallel output owners;
5. the matching output/state, WY, pair, and decay transposes.

The state owner advances FP32 chunk-boundary `S`; Tensor Core pair, WY, and
state contractions use BF16 multiplicands with FP32 accumulation. Output is
written BF16 and requested final `S` is FP32.

This is selective fusion. Source generation and panelization are fused because
they share ownership and delete a real HBM boundary. Predictor, pair/WY, state,
and output remain separate where chunk/rank/value CTA parallelism is more
valuable than eliminating a small handoff. Do not replace them with a
sequence/head mega-kernel.

## 6. Reverse graph

Backward is the strict transpose of the composed forward blocks:

1. output/state reverse walks chunk boundaries backward and consumes
   `bar_o,bar_S_final`;
2. fast-WY reverse applies the transpose triangular action;
3. the direct-`e` pair transpose owns final source tiles and closes decay;
4. the source transpose returns `bar_u,bar_delta,bar_q,bar_k,bar_v` and raw
   gate cotangents;
5. the gated-Oja transpose combines `bar_delta` with `bar_C_final` and
   returns `bar_h,bar_u,bar_gamma,bar_C0`;
6. L2Norm transposes return gradients to the original strided public views.

No descriptor bundle, coordinate VJP, covariance replay, CG implicit
transpose, expanded sequence, or dense inverse-state cotangent exists.
Contributions to shared `u` close only after their independent owners have
produced final-shaped gradients.

## 7. Precision map

```text
BF16: public vector operands/output; pair, WY, and state multiplicands
FP32: C,S,gamma,log_alpha,norm and denominator reductions, divisions,
      Tensor Core accumulators, backward partials, continuation boundaries
FP64: token oracle only
```

The relative denominator has no artificial clamp, threshold, fallback, or
data-dependent compensation. Its behavior is a model-state acceptance risk and
must be measured directly. Private FP16 panels still require a static range
proof and a direct FP32 producer; BF16-to-FP16 casting is not promotion.

## 8. Layout and lifecycle

Public fused-projection vector views may have arbitrary outer strides and
require unit innermost vector stride. Native owners may pack private normalized
or WY panels. Those layouts are not public ABI.

The only recurrent operator state is FP32 `(C,S)`. No `P^-T`, `J/D/m`,
CG cache, or per-token dense matrix is saved. Forward caches are retained only
when a complete F+B A/B beats local recomputation under the same VJP gate.

Dense native currently requires lengths aligned to C16. Masks and resets use
the same FP64/PyTorch recurrence through the model reference path. This is a
performance limit, not a second operator.

## 9. Acceptance and benchmark

Hard gates:

- FP64 predictor recurrence and exact relative-frame action;
- exact local similarity and finite `gamma=0` GDN2 reduction;
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
eager forward median/p95  0.498 / 0.644 ms
eager F+B median/p95      1.861 / 2.056 ms
Graph forward median/p95  0.187 / 0.193 ms
Graph F+B median/p95      0.717 / 0.918 ms
eager allocator increment 105.1 MiB
```

These are core-operator measurements on the development RTX 5070 Ti, not
complete mixer or causal-LM numbers. Benchmark reports must continue to state
shape, dtype, device, scope, and execution mode.

The complete projected mixer at the same shape measured Graph forward
`0.406/0.518 ms` median/p95 and F+B `1.428/1.639 ms`, with about
`120.0 MiB` capture-incremental training allocation.

Model-level capture uses `SolveDeltaGraphedTrainingStep` around a fixed-shape
loss-only CausalLM wrapper. The trainer owns optimizer steps, clipping,
gradient accumulation, distributed reduction, and recapture policy.
