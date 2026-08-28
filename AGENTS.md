# AI-Assisted Contribution Contract

This repository has one research operator: **Causal LSSO: SolveDelta**. The
current operator is the RLS moving-state recurrence owned by
`causallsso/reference.py`. The older bounded-LDU generalized-Delta operator is
not a production variant or compatibility target; Git commit `2237875` is its
recovery point.

The human contributor remains responsible for scope, correctness, provenance,
testing, and reviewability. Material use of external research or upstream code
must identify the source and the decision it informed.

## Current Contract

- Keep one operator, one reference recurrence, and one production path.
- Geometry width is the resolved key-head width `r := d_k`; `r=128` is the
  target specialization, not a mathematical default.
- The current contract has exactly one ordinary Delta edit per token (`K=1`).
- The geometry prior is fixed to `m_0=2`, `J_0=2I`, `D_0=0`; `S_0=0`.
- The continuation state is `(m,J,D,S)`. All four tensors are FP32 in native
  execution. `J` is symmetric positive definite, stored as a full matrix.
- A supplied `J0` must be exactly symmetric and belong to the SPD state domain.
  Its full-tensor cotangent is represented by `(bar_J + bar_J^T)/2`.
- Raw erase and write logits are activated as `2 sigmoid(x)`. Query, geometry
  direction, and edit key are L2-normalized.
- The frontend applies independent depthwise causal conv4 plus SiLU to query,
  edit key, and edit value by default. It can be structurally disabled.
- Channel-wise associative decay is enabled by default and can be structurally
  disabled for the GDN2/Delta reduction.

For normalized geometry direction `u_t`, define

```text
m_t = lambda_t m_{t-1} + 1
J_t = lambda_t J_{t-1} + u_t u_t^T
D_t = lambda_t D_{t-1} + u_t h_t^T
g_t = J_t^-1 u_t
p_t = J_{t-1}^-1 u_t
C_{t-1} = J_{t-1}^-1 D_{t-1}
r_t = h_t - C_{t-1}^T u_t
rho_t = m_{t-1} / m_t
F_H,t = rho_t (lambda_t I + u_t p_t^T)
F_C,t = I + g_t r_t^T
```

With learned geometry strength `gamma`:

```text
F_H,t(gamma) = I + gamma (F_H,t - I)
F_C,t(gamma) = I + gamma (F_C,t - I)
```

The memory transition is

```text
S = F_H,t(gamma) S_{t-1}
S = Diag(exp(a_t)) S
S = F_C,t(gamma) S
k = normalize(k_raw)
b = 2 sigmoid(b_raw) elementwise-multiplied by k
z = 2 sigmoid(z_raw) elementwise-multiplied by v
S_t = S + k (z - S^T b)^T
o_t = S_t^T normalize(q_t)
```

`gamma=0` must reduce exactly, at finite parameters, to the ordinary gated
Delta edit/read while geometry continuation state still updates.

## Mathematical Ownership

- `causallsso/reference.py` is the only executable mathematical oracle.
- `docs/FROM_SCRATCH_REBUILD.md` is the sole native implementation blueprint.
- `docs/INNOVATION_PROGRAM.md` explains the current recurrence but cannot
  define a second executable operator.
- Parameters and `nn.Module` behavior have one owner. Do not duplicate model
  classes, reference implementations, or backend selectors.
- Masks leave state unchanged and return zero output. A valid reset restores
  the fixed prior before consuming the token. Recurrent splitting must preserve
  the same continuation semantics.
- The exact reference uses `torch.linalg.solve`. Native gain evaluation uses
  the selected fixed five-step MESA conjugate-gradient action. This is an
  explicitly authorized BF16-observable production approximation, not a new
  mathematical recurrence and not permission to add runtime fallbacks.

## Precision Contract

- Public/raw vector operands and outputs are BF16 in native execution.
- `(m,J,D,S)`, geometry and associative log-decays, effective-mass recurrence,
  CG reductions, normalization reductions, sensitive scalar divisions, and
  backward partials are FP32.
- Tensor Core pair/WY/state contractions use BF16 multiplicands and FP32
  accumulation.
- A private FP16 panel is allowed only with a static analytic range proof and a
  direct FP32-to-FP16 producer. BF16-to-FP16 pseudo-promotion is forbidden.
- There is no runtime dtype selection, magnitude threshold, precision fallback,
  clipping, or data-dependent compensation.
- FP64 defines operator mathematics. Production acceptance is evaluated at
  BF16 outputs, FP32 continuation states, and composed VJPs. Private arithmetic
  order and private panel bits are diagnostic only.

## Native Ownership

The selected dense BF16 CUDA path is fixed as:

1. FLA L2Norm for public vector normalization;
2. MESA paired `Hkk/Hkv` state scans for `J/D` and their strict transpose;
3. fixed CG5 matrix-free gain and implicit transpose;
4. a scalar FP32 affine scan for effective mass;
5. native token-block `E=3` direct-`e` pair/WY/state/output owners, with C32
   geometry chunks and C16 exterior chunks;
6. the matching output-owned reverse and source transpose.

The three micro-edits are the two RLS transport factors and one ordinary Delta
edit. They are a fixed internal slot axis, not an expanded public `3T` ABI.
Use GDN2-style selective fusion: preserve chunk and output parallelism, and
accept a private boundary when fusion would lengthen lifetimes or reduce CTA
occupancy.

The native kernels require packed operand layouts. The public operator owns the
single canonicalization boundary from strided fused-projection views. Removing
those copies requires adding and testing native stride support in every
consumer; silently treating a strided view as packed is incorrect.

Dense CUDA BF16 is the optimized training surface. Masks and resets currently
use the same RLS reference semantics through the model fallback; they must not
fall back to the archived bounded-LDU operator.

## Investigation and Scope

- Reproduce and localize failures before adding numerical mechanisms.
- Search primary papers, official documentation, upstream source, and issue
  trackers before inventing numerical linear algebra or GPU schedules.
- Record material upstream use in `docs/PRIOR_ART.md` and
  `THIRD_PARTY_NOTICES.md`.
- Do not restore bounded-LDU, QRD, Neumann, flat `3T`, multiple-edit, backend
  selector, or exact-path compatibility branches.
- Do not use subagents for repository work.
- Exploratory scripts, downloaded data, generated artifacts, and benchmark
  output stay outside Git. `experiments/` is not production package content.

## Acceptance

The minimum suite covers:

- FP64 token recurrence and fixed SPD prior;
- geometry state, masks, resets, recurrent splits, and non-128 widths;
- finite-parameter GDN2 reduction at `gamma=0`;
- raw gate activation;
- native BF16 output and FP32 `(m,J,D,S)` against the FP64 oracle;
- composed VJPs including initial and final state and symmetric `J` convention;
- the public model's dense native and mask/reset paths.

Benchmark candidate changes only after the same semantic and VJP gates pass.
Report forward, backward, F+B, allocator peak, and whether a number measures the
core operator or the complete projected layer. Benchmarks measure an
implementation; they never define model semantics.
