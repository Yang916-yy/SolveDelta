# SolveDelta Upstream Code-Block Reuse Checklist

This inventory controls source-level reuse for the one production SolveDelta
path. It does not define operator mathematics; `causallsso/reference.py`
remains the sole executable oracle, and `docs/PARALLELISM.md` remains the
execution contract. A checked source-review item means that the upstream block
has been inspected, not that it has been adopted.

The reviewed baseline is Flash Linear Attention (FLA) `v0.5.2`, installed in
the repository's `causallsso` virtual environment. FLA is MIT licensed. Any
adapted source must retain its copyright header, identify the exact upstream
function, and update `THIRD_PARTY_NOTICES.md` before merge.

## Reuse Rules

- [x] Reuse the smallest algebraically matching kernel block, tile schedule, or
  transpose formula. Do not import an entire model or operator ABI.
- [x] Keep the native specialization fixed at `C=32`, `r=128`, `K=1`, BF16
  public/raw operands, analytically bounded private FP16 panels, FP32
  accumulation, and FP32 continuation states.
- [x] Preserve SolveDelta-owned layouts. Do not create generic `qg/kg/ag`,
  public `d/e/chi`, synthetic `3C=96`, or DPLR compatibility tensors.
- [x] Keep stable decay gauges tile-local. Never materialize
  `exp(-G_i) * d_i` in HBM.
- [x] Adopt a forward block only with its strict transpose reverse in the same
  change. Entrywise VJP chains are forbidden.
- [x] Remove upstream varlen, C64, multi-backend, fallback, and unused autotune
  branches unless they exactly serve the frozen specialization.
- [x] Benchmark the complete SolveDelta forward and forward-plus-backward path.
  A faster isolated upstream primitive is not sufficient evidence.
- [x] Delete a failed adaptation instead of retaining another backend or
  runtime switch.

## P0: Pair Statistics and C32 WY

### C32 unit-lower inverse

Upstream:

- FLA [`solve_tril_16x16_kernel`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/utils/solve_tril.py)
- FLA [`merge_16x16_to_32x32_inverse_kernel`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/utils/solve_tril.py)
- Generalized-Delta
  [`prepare_wy_repr_fwd_kernel_chunk32`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py)

Reuse target:

\[
R=W^{-1},\qquad
W=\begin{bmatrix}W_{00}&0\\W_{10}&W_{11}\end{bmatrix},
\]

\[
R_{10}=-R_{11}W_{10}R_{00}.
\]

- [x] The 16-by-16 diagonal inverse and 16+16 block merge are algebraically
  compatible with SolveDelta's unit-lower C32 matrix.
- [x] Specialize the block to the native panel layout; do not call FLA's
  allocation-owning public wrapper.
- [x] Keep diagonal substitution in FP32 and separately validate the selected
  operand precision for the two off-diagonal products.
- [x] Handle irregular final chunks with identity-padded rows and columns.
- [x] Compare `R @ B` and the transpose solve against the FP64 C32 WY oracle
  without materializing `R`. MathDx remains the separate `r=128` frame-action
  oracle.

### Wide-RHS application

Upstream:

- Generalized-Delta
  [`wu_fwd_kernel`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py)
- GDN2
  [`recompute_w_u_fwd_gdn2_kernel`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gdn2/wy_fast.py)

Reuse target:

\[
[Y,U_z]=R[E,Z],
\]

with one native `C x (r+d_v)` RHS rather than independent scalar solves.

- [x] The upstream `C x C` by `C x width` `tl.dot` loop matches the native
  contraction.
- [x] Batch `E` and the value RHS internally without exposing a combined ABI.
- [x] Avoid a separately materialized concatenated RHS when two source layouts
  can be loaded directly by one specialized kernel.
- [x] Save only the inverse or solve factors actually required by transpose
  reverse; do not save both `W` and `R` without an A/B result.

Adopted result: `causallsso/ops/paired_wy.py` recomputes the private inverse
from saved FP32 `W`, applies the edit and value RHS in one launch, and never
writes `R` or a concatenated RHS. The accepted kernel rounds the private
inverse blocks and FP32-formed RHS directly to BF16, uses one Tensor Core
product per block action with FP32 accumulation, and stores unbounded `Y/U_z`
once to BF16. Forward and transpose are governed by action, matrix-reverse, and
composed VJP gates; the private inverse and pre-storage residual are not a
production interface.

### Stable W/A construction

Upstream:

- KDA
  [`chunk_kda_fwd_kernel_inter_solve_fused`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/kda/chunk_intra.py)
- Gated Delta
  [`chunk_gated_delta_rule_fwd_kkt_solve_kernel`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk_fwd.py)
- Generalized-Delta DPLR WY preparation in
  [`wy_fast_fwd.py`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py)

SolveDelta mapping, written with a mathematical gauge only, is

\[
\bar D_j=d_j\odot e^{-G_j},\qquad
E_i=e_i\odot e^{G_i},\qquad
Q_i=\chi_i\odot e^{G_i},
\]

\[
W=I+\operatorname{tril}(E\bar D^T,-1),\qquad
A_{qd}=\operatorname{tril}(Q\bar D^T,0).
\]

- [x] The upstream tile-local reference gauge and causal masked `tl.dot`
  schedule are structurally compatible.
- [ ] Replace upstream `q/k/beta/g` leaf semantics with native tile producers
  for `E`, `Q`, and `D`; do not create compatibility tensors.
- [ ] Center every gauge on a tile reference so both exponent arguments are
  nonpositive. The displayed inverse gauge must never exist as a full tensor.
- [ ] Accumulate W and A in FP32 and freeze their BF16/FP32 storage decision by
  the cancellation fixtures.
- [ ] Measure a separate Triton consumer A/B before attempting a CUDA/Triton
  fusion. A temporary workspace is internal to the prepare owner and must be
  deleted if it does not improve complete F+B latency.

### WY and pair transpose reverse

Upstream:

- Generalized-Delta
  [`prepare_wy_repr_bwd_kernel`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/generalized_delta_rule/dplr/wy_fast_bwd.py)
- GDN2
  [`chunk_gdn2_bwd_kernel_wy_dqkg_fused`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gdn2/chunk_bwd.py)

For `X=R B`, the selected reverse must implement

\[
\bar B=R^T\bar X,\qquad \bar W=-\bar B X^T.
\]

With

\[
L_W=\operatorname{tril}(\bar W,-1),\qquad
L_A=\operatorname{tril}(\bar A_{qd},0),
\]

the pair reverse is

\[
\bar E\mathrel{+}=L_W\bar D,\qquad
\bar Q\mathrel{+}=L_A\bar D,
\]

\[
\overline{\bar D}\mathrel{+}=L_W^TE+L_A^TQ.
\]

- [x] FLA contains the required matrix VJP pattern and channel-wise decay
  reductions.
- [x] Fuse the two pair cotangents before entering frame transpose.
- [x] Do not allocate `grad_d`, `grad_e`, or `grad_chi` as cross-kernel
  interfaces.
- [x] Accumulate exponent/gauge cotangents in FP32 and reduce them once into the
  associative log-decay gradient.
- [x] Verify output and final-state cotangents together; neither may be omitted
  from the solve reverse.

The adopted matrix reverse fuses the transpose solve,
`-barB X^T`, and the `write/value` pullback. The displaced scalar CUDA solve,
its `grad_Z` workspace, and its separate value-backward launch were deleted.
The native pair reverse now writes directly into the one primal and two-route
dual workspaces owned by the frame adjoint. The frame action consumes and
overwrites those workspaces in traversal order, so no `grad_d/grad_e/grad_chi`
allocation or cross-kernel ABI remains. The unbounded rank-three descriptor
cotangent bundle is produced directly as BF16, the exact operand format used
by the strict Tensor Core transpose, rather than making an FP32 HBM round trip.
On the target profile, seven replicated measurements changed median backward
from about `5.172 ms` to `5.024 ms` and complete F+B from about `6.762 ms` to
`6.704 ms`; the workspace ownership rewrite itself was performance-neutral.

## P1: SolveDelta Frame and Radial Blocks

### Bounded private FP16 producer boundary

Upstream:

- MESA-Net
  [`chunk.py`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/mesa_net/chunk.py), specifically the
  `l2norm_fwd(..., output_dtype=torch.float16)` calls in its chunk autograd
  owner.

MESA demonstrates the useful boundary, not an FP16 backend: BF16 model inputs
are reduced and normalized in FP32 and the producer writes the normalized
private panel directly as FP16. A plain BF16-to-FP16 cast cannot restore bits
and may lose exponent range.

- [x] Adopt the producer boundary for normalized `u/q/k`, whose norm and
  components are analytically bounded by one.
- [x] Extend it to the FP32-formed erase source `b=beta odot k`, bounded by two.
- [x] Permit FP32-produced strict chart coordinates and `d/e/chi`-family frame
  panels using the existing `1/4`, `B_P~=2.283`, and rounded-factor
  `2B_D<4.014` certificates.
- [x] Require every consumer and every backward partial to accumulate in FP32,
  with forward/reverse consuming the same frozen FP16 bits.
- [x] Forbid FP16 storage for unnormalized `h`, values/write-value products,
  and recurrent states. Keep `W` FP32 as the canonical chunk system, and keep
  unbounded `Y/U_z` in BF16 rather than FP16.
- [x] Forbid runtime dtype selection, range tests, clamps, fallbacks, and
  BF16-to-FP16 pseudo-promotion.

This is the selected rewrite contract. The production normalization and frame
producers now write the certified vector and strict-factor panels directly from
FP32 to FP16. The paired frame reverse consumes those same stored FP16 bits.
Unbounded descriptor cotangents are formed in FP32 registers and written once
as BF16 for their Tensor Core consumer; descriptor contractions and
geometry-scan composition still accumulate in FP32. The unbounded paired-WY
solve outputs likewise cross their storage boundary once to BF16.

### Matrix-free local outer action

Upstream:

- MESA-Net
  [`chunk_update_once`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/mesa_net/chunk_cg_solver_fwd.py)

MESA evaluates the matrix-free action

\[
((P K^T)\odot M)V.
\]

For SolveDelta generators `a_s b_s^T`, the corresponding chunk action is

\[
\boxed{O=((X B^T)\odot\Omega)A}.
\]

Use `(A,B)=(u,u)` for J generators and `(A,B)=(u,h)` for D generators.

- [x] The two-`tl.dot` schedule is an exact match for every off-diagonal
  coordinate block of the local semiseparable action.
- [x] Retain a warp strict-triangular solve only for each 16-by-16 diagonal
  coordinate block.
- [ ] Stream tile-local `C x C` interactions; do not restore the rejected
  resident `C x 2C` dual-suffix state.
- [x] Share one generated factor/action tile between primal gather and the two
  dual routes without creating a synthetic `3C` dimension.
- [ ] Derive and implement the exact transpose of this same two-dot action.
- [ ] Record shared memory, registers, spills, barriers, and CTAs per SM. A
  correct kernel that falls to one CTA per SM is not accepted without a
  complete-path win.

Three workspace-free MESA two-dot schedules were evaluated and deleted. A
deterministic coordinate-block schedule reduced the old `48 MiB` correlation
and `32 MiB` pair partials to `9 MiB`, but took about `2.820 ms` for the strict
transpose. Its atomic variant removed the remaining partials and took about
`2.032 ms`; repeat drift was at most roughly `rho=1.2e-7`, but complete F+B did
not improve. A route-streaming variant restored the exact `O(C^2 r)`
correlation count and reached about `1.981 ms` isolated, yet regressed the same
complete path. The current paired scalar action and existing strict transpose
remain production until one matched forward/transpose schedule wins end to
end; none of the failed implementations remains in source. This rejects those
three schedules, not cross-route or coordinate-block FP32 atomics as a class.
A later fused atomic schedule remains admissible under the fixed repeated-run
precision and complete-path performance gates in `VALIDATION_PLAN.md`.

### J/D moment tile update

Upstream:

- MESA-Net
  [`chunk_mesa_net_fwd_kernel_h`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/mesa_net/chunk_h_fwd.py)

MESA's paired states

\[
H_{kk}\leftarrow\lambda H_{kk}+K^TK,\qquad
H_{kv}\leftarrow\lambda H_{kv}+K^TV
\]

map at the tile-schedule level to

\[
J\leftarrow\lambda J+uu^T,\qquad
D\leftarrow\lambda D+uh^T.
\]

- [x] The FP32 resident tile plus low-precision `safe_dot` update is structurally
  compatible with SolveDelta moment generation.
- [ ] Reuse it only inside frame/radial tile production. Do not replace the
  already fast exact affine boundary scan without a measured reason.
- [ ] Preserve separate J and D states and their separate nonlinear chart maps.
- [ ] Preserve mass normalization and update-before-read token order.

### Radial Gram and pair reductions

Upstream:

- MESA-Net
  [`chunk_mesa_net_h_kk_bwd_intra_kernel`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/mesa_net/chunk_h_kk_intra_bwd.py)
- MESA-Net
  [`chunk_mesa_net_h_kv_bwd_intra_kernel`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/mesa_net/chunk_h_kv_intra_bwd.py)

The reusable identity for a full generator tile is

\[
\langle a_sb_s^T,a_tb_t^T\rangle
=(a_s^Ta_t)(b_s^Tb_t).
\]

- [x] MESA's pair-score, Hadamard weight, row/column reduction, and decay VJP
  schedule supplies the needed building blocks.
- [x] Use two C32 Gram contractions plus an elementwise product for each full
  off-diagonal coordinate tile.
- [x] Handle diagonal coordinate tiles with the exact strict mask rather than
  a full-outer approximation.
- [x] Keep H lower/upper distinct at the public full-ambient `J0` boundary;
  reached symmetric states may be optimized only inside an ownership boundary
  that restores the full ambient cotangent. Keep R lower/upper independent.
- [x] Keep radial norm, `q2`, mass, tanh, and sensitive reductions in FP32.
- [x] Apply radial transpose once to the combined chart cotangent; do not emit
  per-descriptor VJP panels.
- [x] Save the forward-reduced `G[4,C,C]`, boundary pair `c[4,C]`, and
  `||B||^2[4]` as a private FP32 training cache. Backward consumes these final
  statistics directly and must not replay the eight coordinate-block partials
  or instantiate their `8 x G` workspace.

At `P=256,C=32`, the reduced statistics occupy `4.128906 MiB`; including the
existing radial norm gives `4.253906 MiB`. This replaces a `32 MiB` backward
Gram workspace and four route launches of the pair-statistics producer. On the
local SM120 profile, the standalone radial reverse changed from about
`1.439 ms` to `1.126 ms`, while the training forward that writes the cache is
about `0.340 ms` versus `0.318 ms` without saved tensors.

The direct Gram expansion is the production forward under the BF16-observable
numerical contract, paired with its explicit row/column and strict-diagonal
transpose. A static compensated experiment recovered
the `2^12` private norm but could not make an independently rounded Tensor Core
quadratic bitwise zero and cost about `1.527 ms`; it is rejected. Reuse the
uncompensated MESA pair and row/column transpose schedule is therefore the
single production pair. Deep cancellation gates its realized chart coordinate,
action, and reachable composed VJP rather than private `q2` or scale. Do not
add a private-norm threshold, runtime precision branch, or local compensation
before the complete path is evaluated.

## P2: State and Output Exterior

### Factorized state scan

Upstream:

- FLA common Delta
  [`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/chunk_delta_h.py)
- Its paired reverse
  [`chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/chunk_delta_h.py)

- [x] The persistent FP32 state and declared BF16/FP16 contraction schedule is relevant to
  SolveDelta's factorized state recurrence.
- [ ] Reuse only after confirming native inputs match `(Y,U_z,D_tail,G_last)`
  without a DPLR staging adapter.
- [ ] Preserve FP32 state at every chunk boundary and both initial/final-state
  cotangents.
- [ ] Benchmark against the current dedicated `chunk_state.py`; do not assume
  the generic FLA kernel is faster at `r=d_v=128,C=32`.

### Output contraction and reverse

Upstream:

- Generalized-Delta
  [`chunk_dplr_fwd_kernel_o`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/generalized_delta_rule/dplr/chunk_o_fwd.py)
- Generalized-Delta output reverse kernels in
  [`chunk_o_bwd.py`](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/generalized_delta_rule/dplr/chunk_o_bwd.py)

- [x] The `C x r` state term and `C x C` residual term already use the desired
  `tl.dot` structure upstream.
- [ ] Adapt the contraction body to native `Q_gamma`, `A_qd`, state boundary,
  and residual tensors; do not retain DPLR argument names or unused terms.
- [ ] A/B a separate output kernel against state-plus-output fusion. The latter
  loses chunk-parallel CTAs and is not presumed faster.
- [ ] Implement output reverse with the same operand rounding and causal mask
  as forward.

## Explicit Non-Reuse

- [x] Do not reuse MESA-Net's conjugate-gradient loop, ridge parameter, SPD
  solve semantics, `1e-5` denominator perturbations, or `q_star` ABI.
- [x] Do not replace the exact bounded LDU chart with MESA's Gram/ridge system.
- [x] Do not call complete FLA GDN2, KDA, MESA, generalized-Delta, or DPLR
  operators from the SolveDelta production path.
- [x] Do not copy runtime architecture fallbacks, warning-only checks, or
  sequence-length-scaled numerical tolerances.
- [x] Do not use MathDx as the training backend; it remains an exact triangular
  oracle and possible decode candidate.
- [x] Do not reintroduce a failed CuTe or resident semiseparable backend merely
  because an upstream block resembles part of it.

## Adoption Gate for Every Block

- [x] Add the upstream URL, tag/commit, original symbol, local symbol, and
  adaptation summary to `docs/PRIOR_ART.md`.
- [x] Preserve the MIT copyright header in every substantially adapted source
  file and update `THIRD_PARTY_NOTICES.md` to name the actual local file.
- [x] State algebraic equivalence next to the implementation contract.
- [x] Pass the isolated FP64 forward, state, transpose, gradient, irregular-tail,
  cancellation, identity, and repeatability fixtures.
- [ ] Pass the mandatory internal Triton and MathDx budgets from
  `docs/VALIDATION_PLAN.md`.
- [ ] Report warmed forward, backward-alone, F+B, peak memory, registers,
  shared memory, spills, and occupancy at the target profile.
- [x] Test at least one non-128 reference width at the Python/oracle layer even
  when the native block remains specialized to 128.
- [x] Remove the replaced scalar loop, obsolete cache fields, old tests, and
  unused ABI in the same change.
- [x] Keep only the winner after a complete-path A/B.
