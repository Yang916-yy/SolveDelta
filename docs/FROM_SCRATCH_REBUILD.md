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
D = Diag(exp(log_alpha))
r = h - C_prev u
delta = gamma r
C = C_prev D + delta u^T.
```

The residual is evaluated before forgetting. Thus `delta` retains the intended
innovation direction and magnitude, while only historical `C_prev` is turned
over. `D` is exactly the existing DeltaRule channel retention and has no
separate projection, parameter, scalar mean, or learned exponent. It acts on
the right/source axis of `C`, which is also the accumulated primal address axis:
`(C D)^T k = D C^T k`. The recurrence advances a coordinate-leaky online
least-squares solution directly in `C`; the former notation's `rho_h` is fixed
to one.
The per-head geometry bias initializes to `logit(0.9)=log(9)`, so the
predictor begins as a high-relaxation normalized-LMS solver while the
token-local projection remains free to learn smaller or larger rates.

The accumulated primal frame and token-local inverse-transpose action are

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
P   = I + C
d   = P^T k
e   = F^-T b
chi = F^-T q.
```

The primal consumes the complete ordered prefix in `C_t`; the dual and query
remain aligned with the current solve residual through the exact rank-one
inverse-transpose. This intentionally does not form a local similarity pair and
never requires an inverse of accumulated `C_t`.

The memory recurrence is

```text
S_decay = Diag(exp(log_alpha)) S_prev
S = S_decay + d (z - S_decay^T e)^T
o = S^T chi.
```

From zero continuation, finite `gamma=0` gives `C=0`, `delta=phi=0`, and
`P=F=I`, so the memory path is exactly the ordinary gated Delta edit/read.

## 3. Chunk predictor

The production owner specializes FLA coordinate-gated Oja to the
pre-forgetting residual order. Let
`G_{i,c}=sum_{l<=i} log_alpha_{l,c}` within a chunk. The source interaction for
`j<i` is

```text
L_ij = gamma_i sum_c u_{i,c} u_{j,c}
       exp(G_{i,c} - log_alpha_{i,c} - G_{j,c}).
```

Every active exponent is nonpositive. Cross-16-token subchunks retain FLA's
centered Tensor Core factorization; diagonal subchunks evaluate the bounded
coordinate exponent directly. The target branch consumes `gamma_i h_i`, while
the pair/WY branch uses the coordinatewise exclusive prefix. Neither a
decay-compensated target nor `gamma/a` is materialized. The strict transpose
differentiates the same form and returns one cotangent per channel retention.
The resulting state update is exactly `C_prev D + delta u^T`, not a
decay-before-residual approximation.

Its pair GEMM, unit-lower triangular WY solve, state GEMM, chunk-parallel output
owner, chunk boundaries, and reverse schedule are retained. Forward emits
`C_t^T k_t`, the token-local `delta` panel, and final FP32 `C`; reverse uses
FLA's matching output/state owners before applying `W^-T`, returning
final-shaped cotangents for `h,u,gamma,log_alpha,k,C0`.

## 4. Relative source owner

The source owner consumes normalized `u/k`, raw strided `q`, predictor update
`delta`, raw erase/write logits, and value. It computes q L2Norm and generates
and discards `phi` in the same CTA:

```text
phi = rho delta / sqrt(rho^2 + ||delta||^2)  # ||u||=1
den = 1 + phi^T u
e = b - phi (u^T b) / den
chi = q - phi (u^T q) / den
z = sigmoid(write_raw) * v.
```

The primal is produced separately by the Oja output owner:

```text
d = k + C^T k.
```

The upstream L2Norm owner establishes `||u||=1`, making the shortened radial
formula algebraically exact in the composed operator. Its purely radial
`bar_u` term lies in the null direction of the same owner's strict transpose.
This identity avoids a repeated BF16-rounded norm and slightly improves the
production-observable oracle/VJP error.

Q norm reductions, dual actions, and erase/write sigmoid arithmetic are
evaluated in FP32. The bounded local factor gives a static range proof for the
private `e/chi` panels, so the source owner writes FP16 directly from FP32
registers. The accumulated primal is BF16 because `C` has no static range
bound.
Products with the unbounded decay exponent are materialized later in BF16 to
retain exponent range:

```text
d           [panels,1,C16,r] BF16
paired e,q  [panels,2,C16,r] FP16
z           [B,T,H,V]        BF16.
```

The dual panel layout is produced in consumer order. The source transpose
consumes the matching panel cotangents, closes the shared denominator once,
applies the scaled-L2Norm transpose for `phi`, and returns independent
erase/write cotangents. The primal cotangent is owned by the Oja output
transpose and closes through the shared key L2Norm. Radial and gate scalars
stay resident.

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
4. the dual source transpose returns `bar_u,bar_delta,bar_q,bar_k,bar_v` and
   the independent raw erase/write cotangents;
5. the gated-Oja output/state transpose combines `bar_d,bar_delta` with
   `bar_C_final` and returns `bar_h,bar_u,bar_gamma,bar_log_a,bar_k,bar_C0`;
6. the source transpose applies q L2Norm, while shared key and u L2Norm owners
   return their accumulated gradients to strided public views.

Each reverse owner mirrors one forward block and emits final-shaped
cotangents. Contributions to shared `u` close after the source and predictor
owners have both completed. The predictor epilogue merges its final-shaped
source cotangents directly into BF16 and combines the output, WY, and boundary
gate terms with the chunk-local suffix sum in one FP32 owner.

The pair and source transposes accumulate locally in FP32. Their
single-consumer, final-shaped BF16 handoff reduces this backward interface from
12 MiB to 6 MiB at the optimized profile.

## 7. Precision map

```text
BF16: public vector operands/output; accumulated d panel; decay-scaled DPLR, WY, and state multiplicands;
      final-shaped pair-to-source cotangent handoff
FP16: statically bounded source-native e/chi panels, written directly from FP32
FP32: C,S,erase/write gates,gamma,frame/channel log_alpha,norm/radial and denominator reductions, divisions,
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

Non-C16 tails are padded with neutral private tokens whose state transpose is
the identity. Masks and valid resets are compacted into independent segments,
executed as one neutral-padded native batch, and scattered back; the first
segment may inherit continuation while reset segments start from zero. No-grad
`T=1` cache inference uses a residual-before-forgetting recurrent predictor
owner followed by FLA's inference-only DPLR recurrent memory owner.

## 9. Acceptance and benchmark

Hard gates:

- FP64 pre-decay residual, right coordinate frame forgetting, accumulated
  primal action, and exact local inverse-transpose action;
- adversarial verification of the `3/8` denominator lower bound;
- finite `gamma=0` GDN2 reduction from zero continuation;
- independent erase/write activation and KDA gated-output formula;
- masks, resets, recurrent splits, `K=1`, and a non-128 reference width;
- complete initial/final `(C,S)` composed VJP.

Production-observable gates:

- BF16 output and FP32 final `(C,S)` against quantized-input FP64 oracle;
- complete VJP for every public input and state endpoint;
- dense projected layer forward/backward from real strided projection views;
- fixed-shape CausalLM CUDA Graph loss and gradients against eager.

At `B=1,T=1024,H=8,r=V=128` on the development RTX 5070 Ti, CUDA Graph replay
measures `0.262/0.279 ms` forward median/p95 and `0.908/1.068 ms` F+B for the
core with random FP32 initial `(C,S)` and final continuation output disabled.
The complete projected mixer measures `0.514/0.682 ms` forward and
`1.753/1.954 ms` F+B. Live Graph allocations are `63.0 MiB` and `208.1 MiB`
respectively. Reports must continue to state shape, dtype, device, scope, and
execution mode.

The matched FLA GDN2 comparison and exploratory model evidence are maintained
in `docs/RESULTS.md`; they do not alter this implementation contract.

Model-level capture uses `SolveDeltaGraphedTrainingStep` around a fixed-shape
loss-only CausalLM wrapper. The trainer owns optimizer steps, clipping,
gradient accumulation, distributed reduction, and recapture policy.
