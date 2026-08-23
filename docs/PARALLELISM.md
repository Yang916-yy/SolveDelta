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
[e,\chi]=U^T\Sigma L^T[\widetilde b,q].
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
and transpose actions. Off-diagonal coordinate blocks use matrix products;
only the strict mask within a diagonal `16 x 16` block requires a short
warp-level prefix or suffix scan.

For a masked outer `F=tri(LR^T)`, the identities are exact:

- lower forward: coordinate-prefix contractions with `R`, multiplied by `L`;
- lower transpose: coordinate-suffix contractions with `L`, multiplied by `R`;
- upper forward: coordinate-suffix contractions with `R`, multiplied by `L`;
- upper transpose: coordinate-prefix contractions with `L`, multiplied by `R`.

This exposes broad RHS work without asserting that the full triangular matrix
has low ordinary rank. A generic masked outer can have rank `r-1`; the
algorithm exploits its generator structure and finite chunk width instead.

Boundary contributions are dense but shared by all local right-hand sides.
They are evaluated as tiled wide-RHS triangular products rather than repeated
narrow actions. The diagonal block retains the exact unit-triangular order.

### Numerical cancellation

Legal boundary/local cancellation can be much larger than the residual chart
coordinate. Fixed compensation is required wherever ordinary FP32 fails the
frozen cancellation cases. It must be structural and deterministic, not a
data-dependent fallback. The symmetric `J` radial pullback and asymmetric `D`
lower/upper pullbacks should use their derived shared contractions, but each
must still pass its independent error ceiling.

## 3. Frame backward

Forward and backward are one design task. The VJP consumes cotangents of
`d`, `e`, and `chi` and returns gradients for every local vector, gate,
geometry boundary, decay, and shared strength.

For `K=1`, the complete strict-factor cotangent can be represented by a small
fixed set of masked outer descriptors: one primal-solve term, erase/read dual
terms, and the skew terms. The backward contracts those descriptors directly
with boundary matrices and local generators. It must not write a tokenwise
dense factor cotangent or replay `C` dense token matrices.

The intended reverse schedule is:

1. reverse the exact primal and dual actions in coordinate-block order;
2. accumulate bounded-map scalar and diagonal cotangents;
3. evaluate radial pullbacks from shared boundary/local projections;
4. return boundary `(m,J,D)` cotangents and local `(u,h,log_decay)` cotangents;
5. let the Triton affine adjoint propagate boundary cotangents across chunks.

Saved workspace belongs to the chunk VJP contract and must be reported. The
first target is approximately 8--10 MiB at
`B=1,T=1024,H=8,r=128,C=32,K=1`, not the previous tokenwise dense replay.

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
k=d,\qquad v=z,\qquad a=e\odot\exp(g),\qquad b=-d.
\]

For `K>1`, micro-time is flattened in token-major, edit-minor order. Decay is
applied only on the first edit and the query is read only after the final edit.
`K=1` bypasses packing. FLA owns the mature scan/WY forward and backward;
SolveDelta does not reimplement that exterior.

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

The new native ABI must be designed around the mathematical inputs and outputs,
not around deleted kernel layouts. Forward and backward are versioned together.
Unsupported rank, dtype, edit count, device architecture, masks, or resets must
fail explicitly until implemented.

## 6. Acceptance profile

The first performance target is

```text
B=1, T=1024, H=8, r=d_v=128, K=1, C=32, conv4, FP16 outer
```

Benchmark complete forward and forward-plus-backward against the installed
GDN2 layer under matched projection, convolution, dtype, warmup, synchronization,
and loss construction. Also report geometry scan, chunk frame, WY exterior,
peak workspace, and recurrent cache separately.

Performance never overrides correctness. Adoption requires all internal frame
forward/VJP ceilings, all end-to-end ceilings, exact identity-geometry
reductions, initial/final-state gradients, irregular tails, underflow, legal
`J` and `D` cancellation, and bitwise repeatability.
