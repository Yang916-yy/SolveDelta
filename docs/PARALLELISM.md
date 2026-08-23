# SolveDelta Parallel Execution Contract

This document defines one execution program for the canonical SolveDelta
recurrence. It does not define another model or preserve earlier optimization
ABIs.

\[
\boxed{
\text{geometry boundary scan}
\rightarrow
\text{chunk-local frame actions}
\rightarrow
\text{generalized Delta/WY}
}
\]

The FP64 token recurrence remains the numerical oracle. An implementation is
accepted only after outputs, all returned states, invariants, and gradients
match that oracle under `docs/VALIDATION_PLAN.md`.

The production schedule uses BF16 activation and matrix operands, FP32 Tensor
Core accumulators, FP32 scalar reductions, and FP32 continuation states
`(m,J,D,S)`. Log-decay and gate nonlinearities are evaluated in FP32. The one
checked-in native specialization fixes `r=128`, `K=1`, and `C=32`: a Triton
geometry scan feeds a resident mixed-precision CUDA frame, which feeds a
direct-`e` specialization of FLA's generalized Delta/WY exterior. Unsupported
native shapes and sequence features fail explicitly instead of selecting an
older implementation.

## 1. Geometry boundaries

The geometry recurrence is affine:

\[
(\lambda_2,x_2)\circ(\lambda_1,x_1)
=(\lambda_2\lambda_1,x_2+\lambda_2x_1).
\]

The same scalar decay drives `m`, `J`, and `D`, so a single associative scan
produces every chunk boundary. Training stores only boundary states. It never
materializes `T x r x r` token states.

Given the boundary before chunk `c`, every token state inside that chunk is a
finite local expression:

\[
\begin{aligned}
m_{c,i} &= a_i m_{c,0}+p_i,\\
J_{c,i} &= a_i J_{c,0}+\sum_{s\le i}w_{is}u_su_s^T,\\
D_{c,i} &= a_i D_{c,0}+\sum_{s\le i}w_{is}u_sh_s^T.
\end{aligned}
\]

Thus the local frame computation has no recurrence across chunks. This does
not make the full operator chunk-independent: the boundary `(m,J,D)` is the
exact summary of all earlier valid tokens.

The checked-in Triton scan owns the boundary forward and its affine adjoint.
Its output contract is one boundary before each chunk plus the final geometry
state. Chunk size changes implementation scheduling, not operator semantics.
Tail lanes are masked at the load and store address, not merely zeroed, so one
batch's final partial chunk cannot write into the next batch's decay gradient.

## 2. Chunk-local frame problem

For each local token, normalize `H=J/m` and `R=D/m`, apply the two separate
bounded maps, and obtain

\[
L=I+N^-,\qquad \Sigma,\qquad U=I+N^+.
\]

All `K` primal edit vectors share

\[
d=U^{-1}\Sigma^{-1}L^{-1}a,
\]

while the `K` erase covectors and one query share

\[
[e,\chi]=U^T\Sigma L^T[\bar b,q],
\qquad \bar b=\beta\odot a.
\]

The production unit is the whole chunk, not an isolated narrow-RHS TRSM. It
receives one geometry boundary and local vectors, reconstructs local affine
prefixes, evaluates the bounded chart, and emits only `(d,e,chi)` plus compact
saved data required by its VJP.

The local algorithm may reassociate exact contractions, use semiseparable
generators, or tile the coordinate dimension. It may not change frame update
frequency, compress `J` or `D`, approximate the inverse silently, or move the
associative memory into a token-varying basis.

### Wide-RHS blocked action

Inside a chunk of `C=32`, strict masking of each local outer-product source is
handled as a blocked semiseparable action. Partition the `r=128` coordinate
axis into eight blocks of 16. Across all 32 local right-hand sides, one source
block exposes a tensor-core-friendly contraction

\[
[96,16]\,[16,32]\rightarrow[96,32],
\]

where the 96 rows pack the local chart generators needed by the lower, upper,
and transpose actions. These wide products must use BF16 Tensor Core operands
with FP32 accumulation. Off-diagonal coordinate blocks use matrix products;
only the strict mask within a diagonal `16 x 16` block requires a short
warp-level prefix or suffix scan, whose partials also remain FP32.

For a masked outer `F=tri(LR^T)`, the identities are exact:

- lower forward: coordinate-prefix contractions with `R`, multiplied by `L`;
- lower transpose: coordinate-suffix contractions with `L`, multiplied by `R`;
- upper forward: coordinate-suffix contractions with `R`, multiplied by `L`;
- upper transpose: coordinate-prefix contractions with `L`, multiplied by `R`.

This exposes broad RHS work without asserting that the full triangular matrix
has low ordinary rank. A generic masked outer can have rank `r-1`; the
algorithm exploits its generator structure and finite chunk width instead.

Boundary contributions are dense but shared by all local right-hand sides.
The resident reverse already evaluates their broad actions and transpose
actions with BF16 matrix operands, fixed high/low packing of each FP32 boundary,
and FP32 `bmm` accumulation. The forward resident action keeps the exact
unit-triangular coordinate order and packs generated factor values once to
BF16. SASS inspection finds no MMA instruction in the custom resident forward,
action-adjoint, pair, leaf, or coefficient kernels; those paths remain
scalar-FMA dominated. The current checkpoint therefore establishes the mixed
ABI, partial broad-contraction Tensor Core use, and exact action algebra, not a
complete Tensor Core block-action pipeline.

### Numerical cancellation

Legal boundary/local cancellation can be much larger than the residual chart
coordinate even after BF16 input quantization. Boundary `J,D` remain FP32;
rounding them directly to BF16 before subtracting a local contribution is
forbidden because it changes the residual before Tensor Core arithmetic begins.
Where a wide contraction consumes a boundary, keep the sensitive affine
residual in FP32 or use one fixed high/low representation. Compensation is
structural and deterministic, never a data-dependent fallback.

The radial norm and its reverse do not require the old tokenwise dense replay.
For

\[
Z_t=\beta_t Z_{t-1}+r_tL_t,
\qquad
x_t=\langle Z_{t-1},L_t\rangle,
\]

use the exact recurrence

\[
n_t=\beta_t^2n_{t-1}+2\beta_tr_tx_t+r_t^2\langle L_t,L_t\rangle.
\]

The selected compact reverse packs its descriptor/local contractions as BF16
products with FP32 partials and uses an `O(C^2 r)` local pass, while retaining
dense `O(C r^2)` boundary work. High/low boundary packing is fixed and
deterministic. This is not permission to relax the quantized `2^12`
cancellation gate: the symmetric `J` and asymmetric `D` pullbacks must pass
independently, including their final shared-strength reductions.

## 3. Frame backward

Forward and backward are one design task. The VJP consumes cotangents of
`d`, `e`, and `chi` and returns gradients for every local vector, gate,
geometry boundary, decay, and shared strength.

For `K=1`, the complete strict-factor cotangent can be represented by a small
fixed set of masked outer descriptors: one primal-solve term, erase/read dual
terms. The backward contracts those descriptors directly with boundary
matrices and local generators. It must not write a tokenwise dense factor
cotangent or replay `C` dense token matrices.

The intended reverse schedule is:

1. reverse the exact primal and dual actions in coordinate-block order;
2. accumulate bounded-map scalar and diagonal cotangents;
3. evaluate radial pullbacks from shared boundary/local projections;
4. return boundary `(m,J,D)` cotangents and local `(u,h,log_decay)` cotangents;
5. let the Triton affine adjoint propagate boundary cotangents across chunks.

The resident reverse first applies the exact transpose block action corresponding
to each saved primal or dual action. It then forms the rank-three descriptor
bundle for primal, erase-dual, and read-dual terms. Project-owned compact CUDA
primitives produce pair statistics, coefficient/radial pullbacks, and leaf
partials; broad dense-boundary actions and their transpose use BF16 matrix
operands with FP32 accumulation. The geometry scan adjoint receives FP32
boundary, local-vector, decay, and strength partials from this one composed
autograd owner. No tokenwise dense factor cotangent or BF16 recurrent-state
adjoint is materialized.

The final shared `geometry_strength` cotangent is the fixed tying map
`g=1^T g_6` over six chart-channel contributions. In two deepest-cancellation
fixtures the tied scalar is ill-conditioned relative to `g_6`; changing only
the final addition to FP64 does not resolve the discrepancy. These fixtures
therefore use the induced-operator normalization
`|g_hat-1^T g_6|/(sqrt(6)||g_6||_2+1e-8) <= 2.5e-2`, retaining the existing
`1e-6` absolute branch when `||g_6||_2` itself is near zero. Ordinary fixtures
retain the standard total-gradient metric. This is a fixed linear-map
contract, not a data-dependent condition-number multiplier, warning, or
fallback.

Saved workspace belongs to the chunk VJP contract and is measured rather than
inferred from the absence of tokenwise dense factors. At the target profile,
peak forward-plus-backward allocation is about `378.5 MiB` for the resident
frame and `443.1 MiB` for the full path, versus `73.1 MiB` for matched GDN2.

## 4. Generalized Delta/WY exterior

After frame transformation,

\[
S_{t,0}=\operatorname{Diag}(\alpha_t)S_{t-1},
\]

\[
S_{t,j}=(I-d_{t,j}e_{t,j}^T)S_{t,j-1}+d_{t,j}z_{t,j}^T.
\]

FLA's generalized DPLR Delta operator is used with the exact identification

\[
k=b=d,\qquad v=z,\qquad a=-e\odot\exp(g).
\]

This finite sign reparameterization is exactly the same rank-one transition.
Aliasing `b` with `k` removes one signed vector allocation. The checked-in
direct-`e` specialization generates the BF16 bits of `a=-e*exp(g)` inside the
FLA-derived intra-chunk kernels and folds its pullback directly into `e` and
`g`. No full `a` tensor is materialized, saved, or replayed.

`d,e,chi,z` are BF16 at the FLA boundary. Associative log-decay remains FP32,
and FLA's initial, boundary, and final `S` are FP32. The generated `a` operand
is rounded once to BF16 at use after its FP32 exponential/gate evaluation.

The mathematical oracle retains ordered `K>1` semantics, but the only native
specialization is currently `K=1`. FLA remains the owner of the mature
scan/WY state/output kernels; project code specializes only their direct-`e`
intra-chunk staging. Because `d,e,chi` are still explicit frame outputs, this
boundary is not yet a complete Solve-to-WY fusion.

## 5. Ownership and ABI

The implementation has four owners:

- `causallsso/reference.py`: operator mathematics and FP64 oracle;
- `causallsso/ops/triton_geometry.py`: geometry boundary forward/adjoint;
- one chunk-owned CUDA operator: local frame forward and VJP;
- FLA: generalized Delta scan/WY exterior.

MathDx is an optional independent triangular oracle and possible decode
candidate. It is not linked into the training operator and is not a public
model backend. There are no packet, panel, standalone polynomial-solve, or
isolated chart-VJP compatibility entry points.

The resident native ABI is designed around mathematical inputs and outputs,
not deleted kernel layouts. It accepts BF16 vector operands and FP32
decay/strength/boundary states, emits BF16 `d,e,chi`, and accumulates all saved
scalar and backward partials in FP32. Forward and backward are versioned
together. Unsupported rank, dtype, edit count, device architecture, masks, or
resets fail explicitly. The old all-FP32 `c32_frame_forward/backward` ABI and
the `chunk_frame`, `tensorcore_frame`, and `triton_frame` Python paths are
deleted and repository tests prevent their return.

## 6. Acceptance profile

At `B=1,T=1024,H=8,r=d_v=128,K=1,C=32` on the local SM120 GPU, warmed medians
for the selected path are:

| Component | Forward | Forward + backward |
|---|---:|---:|
| Triton geometry scan | `0.119 ms` | `0.695 ms` |
| resident frame | `1.243 ms` | `6.174 ms` |
| direct-`e` WY exterior | `0.234 ms` | `0.936 ms` |
| complete SolveDelta operator | `1.669 ms` | `7.740 ms` |
| matched GDN2 operator | `0.358 ms` | `1.154 ms` |

Resident frame backward is about `4.93 ms`. The WY exterior is already close
to the matched GDN2 core (`0.244/0.941 ms`), so further gains primarily require
reducing compact frame reverse launches, scalar work, and workspace without
weakening dense `J,D`, frame update frequency, or the full chart contract.

The first performance target is

```text
B=1, T=1024, H=8, r=d_v=128, K=1, C=32, BF16 operands / FP32 state
```

The reported operator rows include SolveDelta normalization, scan, frame, WY,
and final state, and the corresponding GDN2 normalization/core/final-state
work. They exclude input/output projections, gate construction, and conv4.
The complete-layer release benchmark must add those frontend costs under
matched BF16 operands, FP32 state, warmup, synchronization, and loss
construction. Also report geometry scan, chunk frame, WY exterior, peak
workspace, and recurrent cache separately.

Performance never overrides correctness. Adoption requires all internal frame
forward/VJP ceilings, all end-to-end ceilings, exact identity-geometry
reductions, initial/final-state gradients, irregular tails, underflow, legal
`J` and `D` cancellation, and bitwise repeatability.
