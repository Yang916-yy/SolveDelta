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
operator is the RLS moving-state recurrence in `causallsso/reference.py`. The
older bounded-LDU generalized-Delta operator is available only as history at
commit `2237875`; it is not a variant or compatibility target.

The geometry width is the resolved key-head width `r := d_k`. `r=128` is the
first optimized profile, not a mathematical default. There is exactly one
ordinary Delta edit per token (`K=1`). The fixed prior is

```text
m_0 = 2,  J_0 = 2I,  D_0 = 0,  S_0 = 0.
```

The FP32 continuation state is `(m,J,D,S)`. `J` is symmetric positive definite
and currently stored as a full matrix. A supplied `J0` must be exactly
symmetric; its full-tensor cotangent is represented by
`(bar_J + bar_J^T) / 2`.

For normalized geometry direction `u_t`, the geometry update is

```text
m_t = lambda_t m_{t-1} + 1
J_t = lambda_t J_{t-1} + u_t u_t^T
D_t = lambda_t D_{t-1} + u_t h_t^T
g_t = solve(J_t, u_t)
p_t = solve(J_{t-1}, u_t)
C_{t-1} = solve(J_{t-1}, D_{t-1})
r_t = h_t - C_{t-1}^T u_t
rho_t = m_{t-1} / m_t
F_H,t = rho_t (lambda_t I + u_t p_t^T)
F_C,t = I + g_t r_t^T.
```

Learned geometry strength interpolates both transports with identity:

```text
F_H,t(gamma) = I + gamma (F_H,t - I)
F_C,t(gamma) = I + gamma (F_C,t - I).
```

The memory is transported by `F_H`, channel-decayed, transported by `F_C`,
updated by one gated Delta edit, and read by the normalized query. Raw erase
and write logits use `2 sigmoid(x)`. Query, geometry direction, and edit key
are L2-normalized. The frontend applies independent depthwise causal conv4 plus
SiLU to query, edit key, and edit value by default.

At `gamma=0`, the memory path must reduce exactly at finite parameters to the
ordinary gated Delta edit/read while `(m,J,D)` continues to update.

At zero projected geometry input, model initialization deterministically
spreads heads across `lambda in [0.985, 0.995]`; a single head uses `0.99`.
This is an initialization, not a clamp or a runtime fallback. Every head remains
learnable through the ordinary geometry gate.

Masks leave state unchanged and return zero operator output. A valid reset
restores the fixed prior before consuming that token. Recurrent splitting must
preserve the same continuation semantics.

## Precision and Native Ownership

- Public/raw vector operands and native outputs are BF16.
- `(m,J,D,S)`, log-decays, effective mass, normalization/CG reductions,
  sensitive divisions, and backward partials are FP32.
- Tensor Core pair/WY/state contractions use BF16 multiplicands and FP32
  accumulation.
- A private FP16 panel requires a static range proof and a direct FP32-to-FP16
  producer. Casting an already-rounded BF16 value to FP16 is not promotion.
- Runtime dtype selection, magnitude thresholds, clipping, precision fallbacks,
  and data-dependent compensation are forbidden.
- FP64 defines the mathematics. Production acceptance is observed at BF16
  outputs, FP32 continuation states, and composed VJPs; private reduction order
  and panel bits are diagnostics only.

The selected dense CUDA path is:

1. stride-aware FLA L2 normalization;
2. paired MESA `Hkk/Hkv` scans for `J/D` and their strict transpose;
3. fixed five-step matrix-free CG gain and implicit transpose;
4. a scalar FP32 affine scan for effective mass;
5. token-block `E=3` direct-`e` pair/WY/state/output owners with C32 geometry
   chunks and C16 exterior chunks;
6. matching output-owned reverse and source transpose.

The three micro-edits are private fixed slots, not a public `3T` sequence.
Public fused-projection views may have arbitrary outer strides but require unit
innermost vector stride. Do not restore public canonicalization copies.

Dense CUDA BF16 is the optimized training surface. Masks and resets currently
use the same RLS semantics through the model reference path; they must never
fall back to the archived bounded-LDU operator.

## Scope Boundaries

Do not restore bounded-LDU, QRD, Neumann, flat `3T`, multiple-edit, inverse
state, backend selectors, or abandoned private ABIs as maintained alternatives.
An ablation may disable a component for a controlled experiment, but it must
not create a second public model contract.

Parameters, `nn.Module` behavior, reference mathematics, and public variants
must each have one owner. Avoid duplicate model classes or Python glue that
unpacks a chain of private VJPs.

## Acceptance and Reporting

The minimum suite covers:

- FP64 recurrence and the fixed SPD prior;
- masks, resets, recurrent splits, and a non-128 width;
- the finite-parameter GDN2 reduction at `gamma=0`;
- raw gate activation;
- native BF16 outputs and FP32 `(m,J,D,S)` against the FP64 oracle;
- composed VJPs, including initial/final state and symmetric `J` convention;
- the public model's dense native, mask/reset, cache, and loss paths.

Report performance as forward, backward, and F+B median/p95, with allocator
peak and the exact shape/dtype/device. State whether a number covers the core
operator, projected mixer, model block, or complete causal LM. Compare only
under the same execution mode, including CUDA Graph use. Benchmarks measure an
implementation; they never define model semantics.
