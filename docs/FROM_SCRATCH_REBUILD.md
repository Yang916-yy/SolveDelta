# SolveDelta Native Blueprint

This document defines the sole current native implementation. The production
operator is the RLS moving-state recurrence in `causallsso/reference.py`. The
bounded-LDU implementation was archived at Git commit `2237875` and is not a
fallback, ABI, or acceptance source.

## 1. Public surface

For batch `B`, length `T`, heads `H`, key width `r`, and value width `V`:

```text
u,h,q                 [B,T,H,r]
k,v                   [B,T,H,1,r] / [B,T,H,1,V]
log_lambda            [B,T,H]
log_alpha             [B,T,H,r]
erase_raw,write_raw   shapes of k,v
gamma                 [H]
m0,J0,D0,S0           [B,H], [B,H,r,r], [B,H,r,r], [B,H,r,V]
output                 [B,T,H,V]
```

The native vector operands and output are BF16. Log gates, `gamma`, and all
continuation states are FP32. `K=1`, prior mass is `2`, geometry chunk size is
32, exterior token chunk size is 16, and the gain action uses five CG steps.
These constants are one selected production implementation, not runtime
backend choices.

`J0` is symmetric positive definite. Its stored representation is full FP32.
The cotangent returned for the symmetric state is

```text
bar_J0 <- (bar_J0 + bar_J0^T) / 2.
```

## 2. Token recurrence

Normalize

```text
u <- u / ||u||_2
q <- q / ||q||_2
k <- k / ||k||_2
b <- 2 sigmoid(erase_raw) elementwise-multiplied by k
z <- 2 sigmoid(write_raw) elementwise-multiplied by v.
```

The geometry state updates once:

```text
lambda = exp(log_lambda_t)
m_t = lambda m_prev + 1
J_t = lambda J_prev + u u^T
D_t = lambda D_prev + u h^T.
```

Define the RLS quantities

```text
g = solve(J_t, u)
p = solve(J_prev, u)
C_prev = solve(J_prev, D_prev)
r_h = h - C_prev^T u
rho = m_prev / m_t.
```

The two exact rank-one transport factors are

```text
F_H = rho (lambda I + u p^T)
F_C = I + g r_h^T.
```

Geometry strength interpolates each factor with identity:

```text
F_H_gamma = I + gamma (F_H - I)
F_C_gamma = I + gamma (F_C - I).
```

Memory update and read are

```text
S_a = F_H_gamma S_prev
S_b = Diag(exp(log_alpha_t)) S_a
S_c = F_C_gamma S_b
prediction = S_c^T b
S_t = S_c + k (z - prediction)^T
o_t = S_t^T q.
```

At `gamma=0`, both geometry transports are identity and this is the ordinary
gated Delta edit/read. Geometry state still advances so a later nonzero
`gamma` observes the prefix.

## 3. RLS identities used by native code

The native source owner avoids separately solving `J_prev p=u` and
`J_prev C_prev=D_prev`. From Sherman-Morrison identities, with

```text
den = 1 - u^T g,
```

the required previous-state quantities are

```text
p = lambda g / den
r_h = (h - updated_prediction) / den,
```

where the MESA paired owner supplies `g` and the matching prediction
`updated_prediction = h - den (h - C_prev^T u)`. The implementation must
preserve the exact mapping in `reference.py`; these identities only remove
redundant dense solves.

The first transport is diagonal plus rank one:

```text
F_H_gamma = a I + d0 e0^T
a  = 1 + gamma (rho lambda - 1)
d0 = exp(log_alpha) * gamma rho/a * u
e0 = -p / exp(log_alpha).
```

After absorbing `a` into the channel gate, the second transport and ordinary
edit give three ordered generalized-Delta slots per token:

```text
slot 0: (d0, e0)              geometry H transport
slot 1: (gamma g, -r_h)       geometry C transport
slot 2: (k, erase*k)          ordinary edit
```

The slot convention uses the generalized update `S <- S - d e^T S`; signs in
the source panel above follow that convention. The ordinary slot additionally
injects `k z^T`. This fixed `E=3` axis is kept inside token/chunk owners and is
never exposed as a public `3T` sequence.

## 4. Forward ownership

1. The stride-aware FLA L2Norm specialization loads the fused-projection `u`
   view and writes a packed normalized panel. The E3 source owner directly
   loads strided `q/k` and saves their FP32 reciprocal norms for transpose.
2. The MESA paired state owner advances FP32 `J/D` chunk boundaries and emits
   the matrix-free gain and updated prediction. The dense states are not built
   per token in HBM.
3. The FP32 mass affine scan emits previous/current mass and the final mass.
4. The source owner directly loads strided `h/q/k/v` and raw erase/write
   logits, evaluates `2 sigmoid` in FP32, rounds the gate to the declared BF16
   public boundary in registers, and emits BF16 `[token,3,r]` direct and
   `[token,4,r]` paired panels plus the write value. It computes scalar
   denominators and diagonal logs in FP32. No activated gate panel is written.
5. The C16 TileLang pair owner constructs strict `W`, read interaction `A`,
   globally gauged direct/dual panels, and tail panels using 16x16 MMA tiles
   with FP32 accumulators.
6. The C48 WY owner solves the unit-lower interaction system and forms its
   compact response. The inverse is stored only in the consumer dtype.
7. The three action statistics use mature BF16 `bmm`/`baddbmm` epilogues:
   Tensor Cores accumulate in FP32 and write the declared private BF16 panels
   directly. No FP32 action-statistics HBM temporary or follow-up cast/add
   kernel is part of the execution graph.
8. Separate action-statistics and state/output owners preserve chunk/rank/value
   CTA parallelism. They advance FP32 chunk boundary `S`, emit BF16 outputs,
   and return FP32 final `S`.

This boundary is selective fusion. Combining the state and output traversal
into one sequence/head CTA was measured slower because it reduced the target
from chunk/rank parallelism to eight long-lived CTAs.

## 5. Reverse ownership

Backward is the transpose of the selected blocks, not differentiation through
an expanded sequence:

1. The state/output reverse walks chunk boundaries in reverse, consumes
   `bar_o` and `bar_S_final`, and produces final-shaped cotangents for WY,
   pair statistics, write values, and `S0`.
2. WY reverse applies the transpose unit-triangular action and closes the
   compact interaction cotangent.
3. Pair reverse owns each source output tile, streams all interaction
   contributions, and accumulates Tensor Core products in FP32. It does not
   materialize per-source full gradients or a `3T` checkpoint.
4. The fused source transpose returns cotangents for normalized vectors, raw
   gate logits, decay, mass, gain, prediction, and `gamma`. Its gate epilogues
   reproduce the deleted BF16 intermediate rounding in registers. L2Norm VJPs
   consume the exact rounded normalized panels saved by forward and write
   compact final-shaped gradients for the original strided views.
5. The MESA implicit transpose solves the adjoint gain action with the same
   fixed CG5 schedule, reverses `Hkk/Hkv`, and returns final-shaped
   `bar_u,bar_h,bar_log_lambda,bar_J0,bar_D0`.
6. The mass transpose is a chunk affine reverse scan and returns
   `bar_log_lambda,bar_m0`.
7. Contributions to shared inputs and `log_lambda` are added only after each
   owner has closed its private reduction. The returned `bar_J0` is
   symmetrized exactly once.

The native CG reverse is the implicit transpose action for the exact solve
equations evaluated with the selected five-step numerical action. It does not
backpropagate through five stored polynomial nodes.

## 6. Precision map

```text
BF16: public vector operands/output, E3/WY multiplicands and checkpoints
FP32: m,J,D,S, log gates, gamma, norm and CG reductions, denominators,
      mass scan, MMA accumulators, backward partials, continuation boundaries
FP64: token oracle only
```

No BF16-to-FP16 pseudo-promotion, runtime dtype branch, threshold, fallback,
or data-dependent compensation is permitted. Low-level E3 helpers may retain
FP16 templates for isolated diagnostics, but the public native path is BF16.

## 7. Layout and lifecycle

Public `u/h/q/k/v` and raw gate views retain the fused projection's row stride
and require only unit innermost vector stride. The stride-aware L2Norm, MESA
cross-moment/CG owners, E3 source owner, and their transposes load these views
directly. Producer outputs (`u_normalized`, gain/prediction, E3 panels) remain
packed private tensors for their matrix owners. Gate activation is
generate-use-discard and never becomes an HBM ABI.

The retained saved-tensor footprint is dominated by compact forward cache and
FP32 reverse partials. Do not lower those tensors merely to reduce allocator
metrics. A save-versus-recompute change needs a complete F+B A/B and identical
production-observable gates. The selected training path reconstructs the
2 MiB BF16 query-gauge panel from the saved paired source and FP32 cumulative
gate immediately before state reverse; all larger panel-recompute candidates
lost their complete F+B A/B. Unrequested final-state outputs do not materialize
zero cotangents, and their public `J` symmetrization is skipped when the caller
does not request the state.

## 8. Acceptance and benchmark

Hard gates:

- FP64 recurrence, SPD prior, mask/reset, recurrent split, and `K=1` surface;
- finite `gamma=0` GDN2 reduction;
- symmetric `J` continuation and cotangent convention.

Production-observable gates:

- BF16 output and FP32 final `(m,J,D,S)` against quantized-input FP64 oracle;
- complete VJP for every public input and initial/final state;
- dense public layer forward/backward from real strided projection views.

Report the core operator and complete projected layer separately. At
`B1,T1024,H8,r=V=128`, three 50-warmup/200-replay CUDA Graph runs measured the
selected contiguous core at median `0.348--0.349 ms` forward and
`1.081--1.085 ms` F+B. Their p95 ranges were `0.363--0.365 ms` and
`1.124--1.297 ms`. The full projected layer measured median
`0.545--0.551 ms` forward and `1.805 ms` F+B, with p95 ranges
`0.568--0.726 ms` and `2.110--2.182 ms`. Capture-incremental allocation was
`173.456 MiB` for inference and `120.046 MiB` for training. Unique core
forward-saved storage was `75.47 MiB`; the complete eager training peak was
`248.04 MiB`.
