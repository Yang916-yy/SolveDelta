# AI-Assisted Contribution Guide

This guide applies to humans and coding agents contributing to Causal LSSO:
SolveDelta. AI tools are welcome, but they do not lower the standard for
mathematics, evidence, licensing, tests, or reviewability. The human submitting
a change remains responsible for every generated formula, source line,
benchmark, and citation.

## Start Here

Read these files before changing the operator:

1. `causallsso/reference.py` is the only executable mathematical oracle.
2. `docs/FROM_SCRATCH_REBUILD.md` is the sole native implementation blueprint.
3. `docs/INNOVATION_PROGRAM.md` explains the model and its intended reductions.
4. `docs/PRIOR_ART.md` records current upstream provenance and production
   decisions.
5. `THIRD_PARTY_NOTICES.md` records adapted code and runtime dependencies.

Use `docs/ENVIRONMENT.md` for installation, supported hardware, validated
package versions, and native-path constraints. Do not infer a supported
environment merely from one contributor's local setup.

Tests are evidence for this contract; they do not replace it. Do not change the
reference recurrence to make an optimized implementation pass.

Before editing, inspect `git status` and `git diff`. Preserve unrelated work in
the tree. Do not create a compatibility path merely because an older test,
checkpoint, ABI, or private layout expects one.

## Using AI Responsibly

- State the requested scope and keep generated changes inside it.
- Verify generated mathematics independently against the FP64 oracle.
- Reproduce and localize failures before proposing numerical mechanisms.
- For CUDA, numerical linear algebra, PyTorch behavior, or known architectures,
  consult primary papers, official documentation, upstream source, and issue
  trackers before inventing a replacement.
- Identify material upstream influence in `docs/PRIOR_ART.md`. If code or a
  schedule is adapted, update `THIRD_PARTY_NOTICES.md` and preserve its license.
- Never present an AI-generated citation, benchmark, or profiler attribution as
  fact until it has been checked.
- Do not use subagents for repository work. One agent should retain the full
  mathematical and worktree context.
- Do not commit, push, publish, or modify remote state unless the contributor
  explicitly requests it.
- Keep exploratory scripts, downloaded papers, generated data, and benchmark
  output outside Git. `experiments/` is not production package content.

## Contribution Workflow

1. Write the algebraic equivalence or intended semantic change first.
2. Add or update the FP64/reference test that distinguishes the behavior.
3. Search for a mature primitive and its strict transpose before writing a new
   kernel.
4. Connect forward and backward together. A forward-only optimization is not a
   production candidate.
5. Test the complete composed VJP, including initial and final states.
6. Benchmark only candidates passing the same semantic and numerical gates.
7. Update design, provenance, and user documentation in the same change.

Prefer one owner for each output and cotangent. Selective fusion is required:
fuse when it removes a real boundary without extending lifetimes or reducing
occupancy, and split when independent CTAs preserve more useful parallelism.
Do not build a low-parallelism mega-kernel to remove a small HBM handoff.

## Current Operator Contract

This repository maintains one operator and one production path. The current
operator is the relative Residual-Frame recurrence in
`causallsso/reference.py`. The older bounded-LDU and RLS operators are Git
history, not variants or compatibility targets.

The geometry width is the resolved key-head width `r := d_k`. `r=128` is the
first optimized profile, not a mathematical default. There is exactly one
ordinary Delta edit per token (`K=1`). The zero state is

```text
C_0 = 0,  S_0 = 0.
```

The FP32 continuation state is `(C,S)`. Prefix observations conceptually
define `J=sum(u u^T)`, `G=sum(h u^T)`, and the normal equation `C J = G`.
Production carries the online solution `C` in the orientation
`prediction = C u` and advances it directly with normalized LMS. For
normalized geometry direction `u_t`, the update is

```text
r_t = h_t - C_{t-1} u_t
delta_t = gamma_t r_t
C_t = C_{t-1} + delta_t u_t^T.
```

The token-local relative frame is

```text
rho = 5/8
phi_t = rho delta_t / sqrt(rho^2 + ||u_t||^2 ||delta_t||^2)
F_t = I + u_t phi_t^T
den_t = 1 + phi_t^T u_t
d_t = F_t k_t
e_t = F_t^-T (erase_t * k_t)
chi_t = F_t^-T q_t.
```

The implementation uses the exact rank-one inverse-transpose action

```text
F_t^-T x = x - phi_t (u_t^T x) / den_t.
```

This preserves `e_t^T d_t = (erase_t*k_t)^T k_t` and makes the erase transition
an exact local similarity transform. The static radial parameterization gives
`3/8 < den_t < 13/8`, `||F_t^-1||_2 <= 8/3`, and
`kappa_2(F_t) <= 13/3`; it is part of the model definition, not a runtime
clamp. It does not claim that `F_t` is the accumulated absolute frame `I+X_t`.

The memory is channel-decayed, updated by the conjugated Delta edit, and read
through the same relative frame:

```text
S'_t = Diag(exp(log_alpha_t)) S_{t-1}
z_t = write_t * v_t
S_t = S'_t + d_t (z_t - S'_t^T e_t)^T
o_t = S_t^T chi_t.
```

Raw erase and write logits use `sigmoid(x)` independently on the key and value
axes, matching GDN2. Query, geometry direction, and edit key are L2-normalized.
The frontend applies independent depthwise causal conv4 plus SiLU to query,
edit key, and edit value by default. The token-local geometry rate is
`gamma_t = sigmoid(geometry_raw_t + geometry_write_bias)`. The core output uses
standard sigmoid-gated RMSNorm before output projection:

```text
y = RMSNorm(x) sigmoid(output_gate).
```

Its strict RMSNorm/gate transpose is part of the model contract; stop-gradient
or a surrogate VJP is not permitted.

At `gamma=0`, the memory path must reduce exactly at finite parameters to the
ordinary gated Delta edit/read and `C` remains unchanged. The geometry write
bias initializes to `logit(0.9)=log(9)` for every head. This high-relaxation
initialization is not a clamp, threshold, or runtime fallback.

Masks leave `(C,S)` unchanged and return zero operator output. A valid reset
restores `(0,0)` before consuming that token. Recurrent splitting must preserve
the same continuation semantics.

## Precision and Native Ownership

- Public/raw vector operands and native outputs are BF16.
- `(C,S)`, erase/write gates, `gamma`, log-decays, normalization/radial
  reductions, relative-frame denominators, sensitive divisions, and backward
  partials are FP32.
- A final-shaped owner-to-owner source cotangent may use BF16 after its owner
  has completed FP32 accumulation and the composed VJP gate has passed. Such a
  handoff is not a reduction partial; partials shared by multiple owners remain
  FP32.
- Eligible predictor-pair and DPLR WY/state/output contractions use BF16
  multiplicands and FP32 accumulation. The exact-unbounded direct-`e` pair
  owner uses FP32 scalar accumulation because its exponential factors cannot
  be centered into low-precision Tensor Core operands without a static range
  bound.
- A private FP16 panel requires a static range proof and a direct FP32-to-FP16
  producer. Casting an already-rounded BF16 value to FP16 is not promotion.
- Runtime dtype selection, magnitude thresholds, clipping, precision fallbacks,
  and data-dependent compensation are forbidden.
- Fixed-shape gradient accumulation may use nonpersistent BF16 Linear shadows
  with FP32 master Parameters. They must refresh exactly once from the bound
  optimizer's post-step hook; a stale shadow must raise rather than replay.
- FP64 defines the mathematics. Production acceptance is observed at BF16
  outputs, FP32 continuation states, and composed VJPs; private reduction order
  and panel bits are diagnostics only.

The selected dense CUDA path is:

1. stride-aware FLA L2 normalization for `u`;
2. a C32 FLA gated-Oja pair/WY/state specialization for the residual predictor
   and its ungated strict transpose;
3. source-owned q/key L2 normalization and fused generation of `d`, paired
   `(e,chi)`, and `z` directly in C16 exterior panels, using the exact
   normalized-`u` radial specialization;
4. an exact unbounded Triton direct-`e` specialization of FLA generalized-DPLR
   pair formation plus FLA WY/state/output;
5. matching output-owned reverse, pair transpose, source transpose, and
   predictor transpose;
6. FLA KDA low-rank coordinate decay and FLA's sigmoid-gated RMSNorm owner,
   with norm-gate/linear lifetime ownership and its strict transpose.

Public fused-projection views may have arbitrary outer strides but require unit
innermost vector stride. The physical fused projection row is padded to a
multiple of 64 for its Tensor Core GEMM; consumers expose only the logical
prefix. Packed private panels and padding rows are not public operator ABI.

Dense CUDA BF16 is the optimized training surface. Masks and resets currently
use the same Residual-Frame semantics through the model reference path; they
must never fall back to an archived operator.

## Scope Boundaries

Do not restore bounded-LDU, RLS, covariance state, CG, QRD, Neumann, flat
`3T`, multiple-edit, inverse-frame state, backend selectors, or abandoned
private ABIs as maintained alternatives. An ablation may disable a component
for a controlled experiment, but it must not create a second public model
contract.

Parameters, `nn.Module` behavior, reference mathematics, and public variants
must each have one owner. Avoid duplicate model classes or Python glue that
unpacks a chain of private VJPs.

## Acceptance and Reporting

The minimum suite covers:

- FP64 residual predictor and relative-frame recurrence;
- the analytic relative-frame denominator and condition bounds;
- masks, resets, recurrent splits, and a non-128 width;
- the finite-parameter GDN2 reduction at `gamma=0`;
- the exact local similarity and inverse-transpose identities;
- independent raw erase/write activation and gated-output formula;
- native BF16 outputs and FP32 `(C,S)` against the FP64 oracle;
- composed VJPs, including initial and final `(C,S)`;
- the public model's dense native, mask/reset, cache, and loss paths;
- fixed-shape CUDA Graph loss forward/backward against eager model gradients;
- single-rank NCCL DDP graph replay against eager parameter gradients, with no
  `AccumulateGrad` stream-mismatch warning.

Report performance as forward, backward, and F+B median/p95, with allocator
peak and the exact shape/dtype/device. State whether a number covers the core
operator, projected mixer, model block, or complete causal LM. Compare only
under the same execution mode, including CUDA Graph use. Benchmarks measure an
implementation; they never define model semantics.
