# SolveDelta Parallel Execution Contract

This document defines one execution program for the canonical SolveDelta
recurrence with static `num_edits = K`. It does not define alternative models.

The dependency graph is one-way:

\[
\text{geometry prefix scan}
\longrightarrow
\text{token-local bounded LDU solves}
\longrightarrow
\text{K-edit generalized Delta/WY scan}.
\]

Exact recurrent decoding, exact parallel training, and optimized arithmetic are
separate claims. Every optimized implementation must match the FP64 token
oracle. The reference accepts every positive `K`; specialized kernels may
support a documented finite set such as `K in {1,2,4}` and must reject other
values explicitly.

## 1. Exact geometry-prefix scan

The geometry states have affine recurrences

\[
m_t=\lambda_t^{(g)}m_{t-1}+1,
\]

\[
J_t=\lambda_t^{(g)}J_{t-1}+u_tu_t^T,
\qquad
D_t=\lambda_t^{(g)}D_{t-1}+u_th_t^T.
\]

Represent an update by `(lambda, x)` and compose temporal updates as

\[
(\lambda_2,x_2)\circ(\lambda_1,x_1)
=(\lambda_2\lambda_1,\;x_2+\lambda_2x_1).
\]

Composition is associative. The same scalar decay supports one shared exact
prefix scan for `m`, `J`, and `D`; invalid tokens contribute `(1,0)`. Normalize
pointwise after the scan:

\[
H_t=J_t/m_t,\qquad R_t=D_t/m_t.
\]

`D` is already the driven cross moment. Projecting
`h_t=W_drive^T c_t` before accumulation is exactly equivalent to forming
`C_t=sum u_t c_t^T` and multiplying `C_t W_drive` afterward, while removing a
dense matrix product. Token projections are packed before normalization and
gate activations.

Training uses a two-level scan: scan one affine summary per chunk, then scan
local contributions from each chunk boundary. Backward stores chunk boundaries
and reconstructs local prefixes. It must not materialize full-sequence
`T x r x r` copies of `J_t` and `D_t`. This is the established
DeltaNet/GDN/KDA chunkwise pattern with a larger matrix payload, not a new
scan problem. Associativity removes temporal depth but not matrix bandwidth,
so the adapted fused scan remains part of complete-layer benchmarks.

## 2. Exact token-local causal solve

Once `(H_t,R_t)` is known, every token/head system is independent until the
associative update. Form two coordinates

\[
X_t^{(H)}=\gamma_g(H_t-I/r),
\qquad X_t^{(R)}=\gamma_gR_t.
\]

Apply separate fixed radii `c_H=c_R=1/8` before adding their strict lower and
upper factors. Apply separate diagonal log-scale radii `s_H=s_R=1/8` before
adding, giving total bounds `c=s_max=1/4`:

\[
N_t^\pm=
\mathcal B_{c_H}(\operatorname{tri}_\pm(X_t^{(H)}))+
\mathcal B_{c_R}(\operatorname{tri}_\pm(X_t^{(R)})),
\qquad
\Sigma_t=\operatorname{Diag}\!\left(
\exp\left[
s_H\tanh(\operatorname{diag}(X_t^{(H)})/s_H)+
s_R\tanh(\operatorname{diag}(X_t^{(R)})/s_R)
\right]
\right).
\]

The separate maps are semantic, not merely an implementation choice. Adding
the coordinates first would make `J` and `D` exactly collapsible to the single
cross moment `D+J`; their independent post-accumulation nonlinear response
cannot in general be recovered from that sum.

The system and frame are

\[
M_t=(I+N_t^-)\Sigma_t(I+N_t^+),
\qquad
P_t=M_t^{-1},
\qquad
P_t^{-T}=M_t^T.
\]

No dense matrix is factorized. Pack all `K` primal edit vectors into one
right-hand-side matrix and execute:

1. unit-lower TRSM with `I+N_t^-`;
2. reciprocal diagonal scaling;
3. unit-upper TRSM with `I+N_t^+`.

Pack all `K` erase covectors and the query for the direct dual products:

1. multiply by `(I+N_t^-)^T`;
2. diagonal scale by `Sigma_t`;
3. multiply by `(I+N_t^+)^T`.

The factors, primal right-hand sides, dual right-hand sides, and orthogonal
residual action should be fused per token/head tile. `K` changes projection,
right-hand-side, and Delta/WY work, but not geometry state or system generation.

### Work audit at r=128

The rejected bidirectional-derived chart spent a leading `4r^3`, or about
8.39 million FLOPs/head/token at `r=128`, on one Cholesky factorization, one
full-right-hand-side triangular solve, `F F^T`, and one nonsymmetric
factorization. The canonical LDU chart has no cubic term. System generation is
`O(r^2)` and a bundle with `K` primal vectors plus `K+1` dual/read vectors costs
`O((2K+1)r^2)` triangular actions, before the associative edits.

An isolated PyTorch 2.11/RTX 5070 Ti probe at `r=128` measured the former
complete chart action at about 2.81 ms and the two-coordinate direct LDU at
about 0.54 ms for 32 independent systems with five right-hand sides, a `5.2x`
ratio. At 256 systems the measurements were about 3.50 ms and 0.95 ms, a
`3.7x` ratio. These are selection diagnostics, not complete-layer throughput
claims.

The earlier single-coordinate factorwise probe matched an explicit dense solve
in FP64 with approximately `2.2e-16` relative forward error and `4.8e-16`
parameter-gradient error. The required acceptance test repeats this for the
selected two-coordinate chart. Its summed bounds remain `c=s_max=1/4`, so the
analytic condition-number bound remains 4.58; realized conditioning must be
remeasured for the selected chart.

## 3. Exact associative-memory phase

After geometry solving, decay is applied once and every token contributes `K`
ordered generalized Delta micro-steps:

\[
S_{t,0}=\operatorname{Diag}(\alpha_t)S_{t-1},
\]

\[
S_{t,j}=(I-d_{t,j}e_{t,j}^T)S_{t,j-1}+d_{t,j}z_{t,j}^T,
\qquad j=1,\ldots,K.
\]

Flatten micro-time in the fixed order
`(t,1)<...<(t,K)<(t+1,1)`. The compact WY system must use each actual
cross-pairing `e_{t,j}^T d_{s,l}`. Replacing transformed factors with original
keys or changing edit order changes the model.

Channel-wise decay follows the GDN2/KDA cumulative-decay algebra. For a
cumulative channel factor `g_t`, rescale

\[
\bar d_t=g_t^{-1}\odot d_t,
\qquad
\bar e_t=g_t\odot e_t.
\]

Decay cancels in same-coordinate pairings. Apply it to the first micro-step of
each token and read after the last. Mature GatedDeltaProduct packing supplies
the ordered multi-edit precedent; GDN2 supplies asymmetric erase/write WY.
The composition still requires token-oracle forward and gradient comparison.

## 4. Training and decoding schedule

The canonical training schedule is:

1. scan chunk-boundary affine geometry elements;
2. assign one cooperative CUDA block to a chunk, load its boundary
   `(m,J,D)`, and reconstruct causal token prefixes inside that block;
3. generate each token's lower and upper LDU factors in reusable shared
   memory, execute packed primal solves and dual products, and emit only
   transformed vector factors;
4. pack `K` micro-edits per token and run asymmetric generalized Delta WY.

Chunk-boundary geometry and associative states are the operator caches. The
complete layer additionally carries the three four-token Q/K/V frontend caches
declared in the operator contract. Decoding executes the same equations for one
token.

At `r=d_k=d_v=128`, packed FP32 state per head is approximately:

- symmetric lower triangle of `J`: 8,256 values or 32.25 KiB;
- dense `D`: 64 KiB;
- dense `S`: 64 KiB;
- scalar `m`.

The operator-state total is approximately 160.25 KiB/head. Conv4 adds
`4(r + Kr + Kd_v)` activation values per head: at `K=1,r=d_v=128`, 1,536
values, or 3 KiB in BF16/FP16 and 6 KiB in FP32. Token-local LDU factors and
reconstructed prefixes are workspace, not recurrent cache.

## 5. Selected hybrid GPU backend

The first implementation uses FP32 system coordinates, factors, triangular
solves, and dual products. There is no iterative refinement because there is no
numerically discovered LU factorization to repair. BF16 and FP16 have passed
the isolated direct-dual action envelope, but remain non-default until their
casts are fused and the complete forward/state/gradient envelope passes.

The selected implementation boundary is hybrid rather than Triton-only:

1. reuse/adapt the mature Triton chunk-boundary scan and generalized
   Delta/WY outer kernels from the Delta family;
2. insert one CUDA C++ custom operator compiled with the available MathDx
   device-TRSM provider between them;
3. let that operator reconstruct chunk-local geometry, generate factors,
   execute TRSM, and write only `O(TKr)` transformed edit/query vectors.

This is the only selected native route. Do not replace the bounded LDU chart
with butterfly, coupling, hierarchical shear, polynomial inverse, or another
factor family merely to avoid TRSM. MathDx owns architecture-specific
triangular-solve optimization; project code owns only the surrounding
geometry fusion, tensor contracts, and Delta integration.

This boundary is required by two independent facts. First, an isolated
per-token TRSM operator would need full `T x r x r` `H/R` inputs and would
reintroduce the forbidden prefix-state traffic. A chunk-owned operator instead
loads one boundary state and consumes local token contributions immediately.
Second, MathDx device TRSM uses a prebuilt device library, relocatable device
code, and device LTO. In MathDx 25.12 the operation is exposed through
cuSolverDx; in MathDx 26.06 it is exposed through cuBLASDx and linked from
`libcublasdx.fatbin`. A normal Triton JIT kernel cannot directly call either
C++ collective without a custom compiler/linker integration. Triton therefore
remains the outer orchestration and Delta implementation, not a literal wrapper
inside the same device kernel.

The selected local compatibility path is PyTorch `2.13.0+cu130`, Triton
`3.7.1`, CUDA 13.0 Update 2, and MathDx 26.06 cuBLASDx 0.7.0. It uses an
isolated project environment; the earlier LSSO environment is not an ABI
dependency. An official cuBLASDx block-TRSM sample has completed the
CUDA compilation, device-link/LTO, fatbin, SM120 dispatch, and numerical-check
path on the local RTX 5070 Ti. Its standalone `M=64, N=4` result reported zero
L2 error and about 0.356 ms versus about 0.103 ms for its cuBLAS reference.
These sample timings are not a rejection: the selected operator must earn its
advantage by fusing prefix reconstruction, factor generation, all right-hand
sides, and Delta-facing output traffic. Do not claim a speedup from the
toolchain probe or compile the PyTorch-facing operator with a mismatched CUDA
major.

The checked-in first build is intentionally an SM120-only specialization and
checks the device capability before launch. MathDx, rather than project-owned
TRSM code, remains the route to additional architectures, but each additional
SM target must be compiled and tested explicitly; the current binary does not
claim cross-architecture coverage.

Radial parameter generation first evaluates the exact four-channel affine
quadratic in a dedicated native pass. Boundary norms and boundary/local
contractions use fixed twofold FP32 accumulation, local outer-product Grams use
`O(C^2 r)` prefix contractions, and the small quadratic composition uses FP64.
The four channels remain independent for arbitrary asymmetric `J,D`; the
implementation does not infer padding from an underflowed affine coefficient
and does not clamp a reconstructed norm. The statistics are forward
temporaries, not saved-state additions for backward.

Inside the action operator, one CUDA block owns one `(batch, head, chunk)`.
The candidate starting layout keeps the running packed `J` and dense `D`
distributed across registers, uses one shared-memory factor buffer repeatedly,
and processes tokens in causal order. For each token it:

1. consumes the four bounded radial coefficients and mapped diagonal generated
   by the exact affine invariant pass;
2. materializes `I+N^-` in shared memory, applies one block-level unit-lower
   MathDx block TRSM to all `K` primal right-hand sides, and applies the matching
   direct lower-transpose products to the `K+1` dual/read vectors;
3. applies reciprocal/direct diagonal scaling;
4. overwrites the same shared buffer with `I+N^+`, then repeats the
   unit-upper solve and upper-transpose products;
5. writes transformed vectors and advances the local geometry recurrence.

At `r=128`, one full FP32 triangular factor occupies 64 KiB and five FP32
right-hand sides occupy 2.5 KiB. The local RTX 5070 Ti exposes 99 KiB opt-in
shared memory per block, so one reusable factor plus vector workspace fits,
although it implies one such block per SM and rules out two independent
128-wide systems in the same block. `BatchesPerBlock=1` is therefore the
starting point; batch, head, and chunk axes supply grid parallelism. Packed
`J+D` contains 24,640 FP32 scalars, or about 96 scalars per thread at 256
threads before MathDx temporaries. `BlockDim` and register pressure must be
tuned together; any local-memory spill of the moments is an acceptance failure
for this layout, not a cost to hide in the benchmark.

Backward is a second custom CUDA/MathDx operator. It recomputes the bounded
factors from chunk boundaries, uses transpose triangular solves for gradients
through the primal action, direct products for gradients through the dual
action, and then differentiates the radial/diagonal maps and affine prefix
scan. It must not save full-sequence factors. Exact reverse reconstruction of
an affine moment is algebraically available while `lambda_g > 0`,

\[
X_{t-1}=(X_t-c_t)/\lambda_t^{(g)},
\]

but division by a very small realized decay may be numerically unsafe. The
implementation must compare reverse reconstruction with sparse subchunk
checkpoints and select the lowest-traffic method that matches the FP32/FP64
gradient oracle over the measured gate envelope. This is a backend question,
not permission to add an unvalidated clipping threshold.

Scalar triangular dependence along rank is owned by the MathDx collective. It
is bounded at fixed `r=128`, distributed across many chunks, and no longer
compounded by token-local factorization. A plain Triton triangular solve may
be used only as a temporary diagnostic oracle; it is not a maintained native
backend or permission to redesign the model chart.

Rejected chart families are recorded in `PRIOR_ART.md`. In particular, pure
SPD, orthogonal-scale, butterfly, Woodbury, dense exponential/Cayley, residual
fixed-point, and bidirectional accretive charts are not retained as runtime
variants.

### Implemented staging boundary

The checked-in bring-up path deliberately exposes the three selected stages
separately so each has an FP64 oracle:

- `triton_geometry_chunk_scan` emits only `O((T/C)r^2)` chunk boundaries and
  final `(m,J,D)`; FP32 IEEE, FP32 TF32, and BF16 forward cases pass;
- `mathdx_solve_frame128` is the independent K2 validation interface and packs
  two primal directions into one
  `nrhs=2` lower/upper MathDx sequence and batches the two erase covectors plus
  query into one direct-dual chain. Its FP32 forward is one native block launch
  covering both TRSMs, diagonal scaling, and the dual; forward and backward
  pass for FP32, BF16-dual, and FP16-dual modes;
- `fla_dplr_delta_outer` uses the exact DPLR identification
  `k=d, v=z, a=e*exp(g), b=-d`, with decay only on the first micro-edit and a
  read after the last; `K in {1,2,4}` and backward pass.
- `packet_frame128` is the selected dense fixed-length `r=128,K=1,C=16`
  forward. Triton owns packing, affine-prefix coefficients, stable radial
  replay, and the packed transpose-dual exterior. CUDA owns exact
  coordinate-packet lower/upper substitution, the compensated boundary/local
  skew action, and dual-right-hand-side construction. No tokenwise factor is
  written.
- `cuda_chunk_solve_frame128` reconstructs eight-token local prefixes from
  each Triton boundary and fuses chart construction plus all primal/dual frame
  actions. It stores `J` and the off-diagonal factor in FP16, `D` in BF16,
  accumulates in FP32, and uses a fourth-order Neumann action under the
  certified `||N^\pm||_F < 1/4` bound. It remains a validation interface; the
  production packet path uses exact coordinate substitution and its dedicated
  packet-native VJP.

The standalone operators remain validation interfaces. In
particular, the low-level standalone TRSM validation operator still packs
row-major PyTorch factors into its column-major ABI, but the default fused
chunk-frame entry constructs packed factors directly and does not create
tokenwise matrix intermediates. A fused Triton pack now writes the
token-major/edit-minor DPLR tensors directly and its backward scatters
`dchi,dd,de,dz,dg` without the former layout copies, zero temporaries, or
separate gate kernels. The former AOTAutograd frame backward is now only the
local derivative oracle. The training path uses the dedicated native packet
action, rank-five descriptor, chart, radial, prefix, and moment VJPs and does
not construct a nested autograd graph.

Historical K2 bring-up on the local SM120 GPU measured isolated forward medians
were approximately: geometry `B=1,T=1024,H=8,r=128`, 0.0864 ms IEEE and
0.0856 ms TF32; 64 independent solve frames before fusion, 0.2274 ms FP32
dual, 0.2943 ms BF16 dual, and 0.2651 ms FP16 dual; BF16 DPLR outer
`B=1,T=512,H=8,r=d_v=128,K=2`, 0.3391 ms. Explicit low-precision conversion
made the unfused dual slower, and TF32 saved about one percent, so FP32/IEEE is
the staging default. The single-kernel FP32 Solve-Frame reduced the same
64-system measurement to about 0.1540 ms with the selected 256-thread block,
roughly 32% below the staged FP32 path. Exact MathDx factor actions for 8192
token systems took about `20.3 ms`, which rejected per-token MathDx as the
training schedule. The fused polynomial frame took about `8.43 ms` for
`B=1,T=1024,H=8,r=128,K=2`; the complete FP16 forward took about `9.23 ms`, so
frame construction/action remains roughly 93% of runtime. The same complete
path took about `0.78 ms` at `B=1,T=64,H=1`; the unspecialized single-token
composition was `0.57 ms` at `B=1,H=8`. A 256-thread dual-residency
attempt reduced one large case slightly but spilled hundreds of bytes per
thread and badly regressed the small case; the retained 512-thread/eight-token
schedule uses 128 registers, 49120 bytes shared memory, and only 4-byte
load/store spills.

In that historical K2 backward, saving the mature FLA WY cache measured about 28% faster than
recomputing it for only about 5.5 MiB extra memory. The fused DPLR pack reduced
its isolated forward-plus-backward from about `1.336 ms` and `239.5 MiB` to
`1.151 ms` and `231.5 MiB`. The first full-chunk VJP took about `73.26 ms` and
`2269 MiB` for the target `T=1024,H=8` profile. Reverse-scanning eight-token
checkpoints across all chunks reduced that to about `62.99 ms` and `1530 MiB`.
A 16-token checkpoint was only about `0.4 ms` faster but raised peak memory to
`2621 MiB`, so eight is selected. Enabling TF32 for the compiled FP32 GEMMs
changed the total by less than 0.3% and was rejected to avoid a global precision
side effect. These warm measurements are local SM120 engineering evidence, not
portable performance guarantees.

The preceding Neumann K1 endpoint was remeasured after replacing generated
frame backward with the explicit adjoint. At
`B=1,T=1024,H=8,r=d_v=128,hidden_size=1024,K=1`, the complete BF16 conv4 layer
takes about `7.529 ms` forward and `44.860 ms` forward-plus-backward, with
`858 MiB` peak allocated memory. The geometry+frame portion alone takes
`41.111 ms` forward-plus-backward and `761 MiB`. Relative to the immediately
preceding K1 generated-VJP layer result, the explicit VJP is `9.0%` faster and
reduces allocation by `37.0%`. The matched installed FLA GDN2 conv4 layer takes
`1.104 ms` forward, `3.312 ms` forward-plus-backward, and `182 MiB`, leaving a
measured SolveDelta gap of about `6.8x` forward and `13.5x` training. The K1
FLA outer independently confirms that retaining its WY cache is `12.2%` faster
than recomputation for `2.75 MiB` extra allocation.

The selected skew action uses an exact coordinate-tile identity. For output
tile `a`, input tile `b`, and `a>b`, its dense-boundary block is

\[
B_{ab,t}=\frac{\alpha_t}{2}\left(
h^-_tJ_{ab}+r^-_tD_{ab}
-h^+_tJ_{ba}^T-r^+_tD_{ba}^T\right),
\qquad B_{ba,t}=-B_{ab,t}^T.
\]

The diagonal tile uses the corresponding strict masks. One CUDA block owns an
unordered tile pair, reads each boundary `J/D` element once, and applies both
skew-related blocks to all 16 token RHS. It emits fixed twofold-FP32 partials;
a second kernel reduces the eight contributor tiles in fixed order, adds the
exact packet-local semiseparable generator action in the same arithmetic, and
constructs the two dual RHS. At the target `P=512,r=128,C=16` profile this
workspace is 64 MiB. It is forward workspace, not recurrent state or a
compressed chart. Full compensation of both `J` and `D` is required: retaining
twofold arithmetic only for `D` reduced isolated time by about 6%, but a
`2^12` symmetric-`J`/asymmetric-tail probe had `rho=2.2e-2` and was rejected.

The coordinate-pair schedule was selected over two exact alternatives. A
row-tile schedule reduced workspace to 8 MiB but took about `1.02 ms` including
the dual-RHS epilogue, versus `0.89 ms` for tile pairs. Fusing the same action
into the primal substitution retained bitwise-identical primal results and
about `2.5e-8` skew-action `rho`, used 72 registers without spills and 42.4 KiB
shared memory, but serialized extra compensated work inside every coordinate
barrier and took `2.32 ms`, versus `2.01 ms` for the then-current two-kernel
composition. This fusion is therefore a scheduling no-go, not an algebraic
failure. The same tile action can evaluate the skew VJP as
`grad_direct - Omega grad_omega` because `Omega^T=-Omega`; its fully
compensated prototype took about `0.92 ms`, slower than the selected
contract-passing `0.53 ms` backward chain, so it is a precision endpoint rather
than a production backward replacement.

The complete endpoint was remeasured in one process on the same target after
the exact four-channel radial invariant, packet-native compact VJP, and tile-pair
skew action were integrated. At
`B=1,T=1024,H=8,r=d_v=128,C=16,K=1`, matched medians are:

| path | forward | forward + backward | peak allocated |
|---|---:|---:|---:|
| packet frame | `2.616 ms` | `8.904 ms` | `619.0 MiB` during F+B |
| packet recurrence | `3.222 ms` | `10.760 ms` | `701.2 MiB` during F+B |
| complete SolveDelta BF16 conv4 | `3.770 ms` | `12.315 ms` | `813.5 MiB` |
| installed GDN2 BF16 conv4 | `0.992 ms` | `3.068 ms` | `331.0 MiB` |

The native primal plus skew/RHS section is `1.49--1.54 ms`; its skew portion
fell from about `1.29 ms` to `0.84 ms`. The remaining complete-layer gap is
about `3.80x` forward and `4.01x` training: materially smaller, but not
performance parity.

An exact scalar-coordinate packet VJP prototype was also rejected. At
`P=512,C=16,r=128`, action adjoints took `3.864 ms` and factor contraction
took `3.669 ms`, for `7.568 ms` before radial, skew, prefix, or moment reverse.
The factor cotangent itself is nevertheless compact: primal solve, two direct
dual actions, and skew contribute at most five masked outer products per token.
The selected VJP keeps this rank-five descriptor, contracts its strict
lower/upper packets directly with dense boundaries, and evaluates local source
actions through rank-coordinate prefix scans. Fixed lazy twofold arithmetic is
used where the driven cancellation requires it. This is a blocked packet
execution, not a second temporal WY and not a low-rank chart.

The final qbar path uses the descriptor linearity directly. For each triangular
entry it first combines the five `[primal, dual0, dual1, skew0, skew1]` masked
outer products in fixed FloatFloat arithmetic, then contracts that total once
with each `J/D` boundary. The local generator traversal already consumes the
same five-slot panel. This removes duplicate qbar work from the transpose solve
without changing the chart derivative. The legal cancellation probe gives
`5.49e-8` qbar relative RMS, `7.60e-15` boundary-contraction relative RMS, and
`7.25e-5` worst complete VJP relative RMS. Same-process AB/BA measurements at
the target profile reduced backward by `0.534 ms` and forward-plus-backward by
`0.637 ms` versus the preceding checkpoint; all 88 repository tests pass.

A projected radial VJP was also derived but not adopted. Saving
`<A_t,B>` and `<A_t,L_s>` projections removes a 32 MiB action workspace and
gives exact scalar derivatives. The fastest fixed compensated source reverse
that passed the `1e-3` cancellation ceiling took `1.252 ms`, about 4--5% slower
than the retained `1.19--1.20 ms` production-like path. Its uncompensated
`0.934--0.943 ms` form missed the driven cancellation ceiling at about `5e-3`.
The result is retained as research evidence, not a production switch.

Plain compact BMM is not an accepted endpoint. It reduced the isolated dense
boundary contraction from `2.320 ms` to about `0.755 ms` on ordinary inputs,
but violated the legal `2^12` cancellation case. A fixed two-sided 12-bit
split restored cancellation to about `4.35e-7` relative RMS, yet the unfused
multi-BMM composition took `2.207 ms` and added about `336 MiB`. Compensation
must be fused into the blocked kernel's accumulation or epilogue; a
data-dependent precision fallback remains forbidden.

The preceding explicit-VJP `T=256,H=8` CUDA profile attributed 73.2% of
self-device time to frame reverse: 34.7% to chart reverse and reductions, 20.2%
to action VJP, 9.5% to local replay, and 5.8% to the moment reverse. Native
forward frame work is 20.9%, while geometry scans and the FLA outer are only
3.3% and 1.0%. Former layout conversion and zero-fill kernels no longer appear
among the dominant operations.

A standalone SM120 Triton chart VJP was retained as a validation interface. It
matches FP64 autograd at maximum relative RMS below `3e-7`, uses 118 registers
per thread with no spill, and takes about 32/47/149 microseconds for 8/64/256
independent systems. Two production integrations were rejected. Emitting full
`dH,dR` raised the complete layer from `44.860 ms / 858 MiB` to
`46.887 ms / 1077 MiB`. Folding moment normalization into the kernel removed
most of that workspace and was 3.6% faster than its same-run compiled-PyTorch
control, but still raised peak allocation by about 91 MiB and did not beat the
retained stable endpoint. Further material speedup therefore requires fusing
action and chart work in one CTA without materializing factor cotangents, or a
new algebraically equivalent action; another isolated chart kernel is not
sufficient.

## 6. Performance acceptance

Benchmark the complete layer over:

- the `r=128` native target and at least one non-128 reference width;
- representative batch, head, chunk, and sequence sizes;
- declared public activation dtype with FP32 geometry/solve accumulation;
- forward, backward, prefill, and single-token decoding;
- peak workspace and recurrent-cache bytes;
- reference PyTorch triangular solves and exact MathDx oracle versus the hybrid
  Triton--CUDA--FLA path;
- register count, opt-in shared memory, blocks per SM, chunk residency, and
  whether transformed-vector traffic dominates after factor fusion.
- reverse-reconstruction error versus checkpoint interval and geometry-decay
  distribution in forward and backward.

Adoption of the native path requires an end-to-end speedup and forward,
final-state, and gradient agreement with the validation plan. The remaining
measured costs are additional prefix-moment bandwidth and triangular actions,
not an unresolved prefix algorithm, cubic factorization, or approximate
inverse quality.
