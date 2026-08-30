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
`(1-gamma)r`. Orthogonal source directions are unchanged. No full-state
forgetting factor, covariance, ridge, or linear solve is part of this model.

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

This is a smooth model parameterization, not clipping or a numerical fallback.

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
and reverse schedule are retained. The unrelated Oja query/output branch is
absent. Forward emits the token-local `delta` panel and final FP32 `C`.
Reverse applies `W^-T` and returns final-shaped cotangents for
`h,u,gamma,C0`; it does not differentiate through a Python token loop. The
ungated specialization removes FLA's otherwise mandatory zero vector-gate
panel and reads strided target views directly. At the optimized `r=128`
profile, reverse owns 32 predictor rows per CTA. Compared with the former
64-row tile, this preserves identical arithmetic while doubling CTA
parallelism and reducing the compiled register/shared-memory footprint.

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

The shortened radial formula is not an approximation: the unique upstream
L2Norm owner establishes `||u||=1` in the composed operator. Its omitted
purely radial `bar_u` term lies in the null direction of that owner's strict
transpose. This removes a repeated BF16-rounded norm and slightly improves the
production-observable oracle/VJP error.

Q/key norm reductions, frame actions, and erase/write sigmoid arithmetic are
evaluated in FP32. The bounded frame gives a static range proof for the private
`d/e/chi` panels, so the source owner writes them directly from FP32 registers
to FP16. This is not BF16-to-FP16 pseudo-promotion. Products with the unbounded
decay exponent are materialized later in BF16 to retain exponent range:

```text
d           [panels,1,C16,r] FP16
paired e,q  [panels,2,C16,r] FP16
z           [B,T,H,V]        BF16.
```

There is no token-major `d/e/chi` HBM ABI followed by
`cat/permute/contiguous`. The source transpose consumes the same panel
cotangents, closes the shared denominator once, applies the scaled-L2Norm
transpose for `phi`, and returns independent erase/write cotangents. No radial
panel, gate panel, or scale is written to HBM.

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
r = sqrt(mean(o^2) + eps),  r0 = 1/sqrt(V)
z = (r-r0)/(r+r0)
alpha_h = sigmoid(2*a_h)-1/2
y = (1+alpha_h*z) RMSNorm(o) sigmoid(gate_raw).
```

`a_h` is one per-head FP32 parameter initialized to `1`. The scale is strictly
inside `(1/2,3/2)`: it preserves a bounded component of output magnitude while
retaining RMSNorm's stable coordinate normalization. CUDA uses FLA
`fused_kda_gate` and a specialization of FLA's fused RMSNorm-gate row owner.
The owner reuses resident FP32 `rstd`; its strict transpose adds the exact
radial `grad_o` and `grad_a` before the final output store. The CPU/FP64 model
path evaluates the same formulas explicitly. The output projection follows
FLA's norm-linear lifetime ownership: forward does not checkpoint the 2 MiB
normalized output panel, and the existing norm transpose regenerates it for
the ordinary Tensor Core weight-gradient GEMM. The GEMM is not fused into the
Triton row owner. No raw-output or norm panel is added to the persistent HBM
ABI.

## 5. Memory exterior

The exterior maps directly to FLA generalized DPLR with

```text
q = chi,  k = d,  a = -exp(log_alpha) e,  b = d,  v = z,  scale = 1.
```

FLA applies its low-rank term to the pre-decay state. The current
`exp(log_alpha)` in `a` is therefore required to erase from `S_decay`; omitting
it changes the recurrence. Production does not materialize this scaled source:
the pair owner obtains it from `-e` and the inclusive decay prefix.

Production ownership is:

1. an FP32 chunk cumsum for channel decay;
2. a C16 exact-unbounded Triton specialization of FLA's scalar direct-`e`
   pair owner for strict edit/read interactions;
3. FLA fast-WY preparation and unit-lower action;
4. FLA chunk boundary state and chunk-parallel output owners;
5. the matching output/state, WY, pair, and decay transposes.

The state owner advances FP32 chunk-boundary `S`; eligible predictor-pair and
DPLR WY/state/output contractions use low-precision multiplicands with FP32
accumulation. The exact-unbounded direct-`e` pair keeps its causal exponent
products and reductions in FP32 rather than licensing a centered low-precision
factorization without a static decay bound. Output is written BF16 and
requested final `S` is FP32.

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
4. the source transpose returns `bar_u,bar_delta,bar_q,bar_k,bar_v` and the
   independent raw erase/write cotangents;
5. the gated-Oja transpose combines `bar_delta` with `bar_C_final` and
   returns `bar_h,bar_u,bar_gamma,bar_C0`;
6. the source transpose applies q/key L2Norm transposes, while the standalone
   u L2Norm transpose returns the remaining gradient to its strided view.

No descriptor bundle, coordinate VJP, covariance replay, CG implicit
transpose, expanded sequence, or dense inverse-state cotangent exists.
Contributions to shared `u` close only after their independent owners have
produced final-shaped gradients.

The pair transpose and source transpose each accumulate in FP32. Their
single-consumer, final-shaped cotangent handoff is BF16, reducing this backward
interface from 12 MiB to 6 MiB at the optimized profile. It is not a
cross-owner partial reduction.

## 7. Precision map

```text
BF16: public vector operands/output; decay-scaled DPLR, WY, and state multiplicands;
      final-shaped pair-to-source cotangent handoff
FP16: statically bounded source-native d/e/chi panels, written directly from FP32
FP32: C,S,erase/write gates,gamma,log_alpha,norm/radial and denominator reductions, divisions,
      Tensor Core accumulators, backward partials, continuation boundaries
FP64: token oracle only
```

The relative denominator has no artificial clamp, threshold, fallback, or
data-dependent compensation. Its `3/8` lower bound follows from the static
radial model parameterization and FP32 norm reduction. Private FP16 panels
still require a static range proof and a direct FP32 producer; BF16-to-FP16
casting is not promotion.

## 8. Layout and lifecycle

Public fused-projection vector views may have arbitrary outer strides and
require unit innermost vector stride. Native owners may pack private normalized
or WY panels. The fused input projection pads its physical output row to a
multiple of 64 and exposes only the logical prefix. At the optimized profile,
`7432 -> 7488` improves the Tensor Core projection transpose while also
satisfying causal-conv1d's multiple-of-eight stride requirement. Those layouts
and padding rows are not public operator ABI.

Fixed-shape Graph training may bind an optimizer-step BF16 shadow for every
FP32 Linear master. The shadows are nonpersistent buffers and their strict
Linear transpose returns FP32 master gradients. An optimizer post-step hook
refreshes all shadows once, outside capture; replay rejects a stale shadow.
This is selected only for gradient accumulation because factor-one refresh
cost exceeds the saved casts.

The only recurrent operator state is FP32 `(C,S)`. No `P^-T`, `J/D/m`,
CG cache, or per-token dense matrix is saved. Forward caches are retained only
when a complete F+B A/B beats local recomputation under the same VJP gate.

Dense native currently requires lengths aligned to C16. Masks and resets use
the same FP64/PyTorch recurrence through the model reference path. This is a
performance limit, not a second operator.

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
Graph forward median/p95  0.172 / 0.178 ms
Graph F+B median/p95      0.611 / 0.793 ms
Graph allocated           58.1 MiB
```

These are core-operator measurements on the development RTX 5070 Ti, not
complete mixer or causal-LM numbers. Benchmark reports must continue to state
shape, dtype, device, scope, and execution mode.

The complete projected mixer at the same shape, using FP32 master parameters
and BF16 autocast, measured Graph forward `0.430/0.603 ms` median/p95 and F+B
`1.435/1.623 ms`, with `245.8 MiB` Graph allocation. The synchronized p95
samples in this run include the same roughly `0.17 ms` P-state tail observed
for the matched GDN2 run. The mixer has `8.99M` parameters versus `9.46M`
before low-rank decay and output gating.

The matched FLA GDN2 comparison and exploratory model evidence are maintained
in `docs/RESULTS.md`; they do not alter this implementation contract.

Model-level capture uses `SolveDeltaGraphedTrainingStep` around a fixed-shape
loss-only CausalLM wrapper. The trainer owns optimizer steps, clipping,
gradient accumulation, distributed reduction, and recapture policy.
