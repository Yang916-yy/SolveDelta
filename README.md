# SolveDelta

> **Status:** unfinished research repository. The operator contract, optimized
> kernels, and validation program are still under active development.

Causal LSSO studies one causal sequence operator: **SolveDelta**. It maintains
normalized prefix geometry and an
associative fast-weight memory. One LSSO-derived primal/dual solve adapter
conditions `K` ordered bounded asymmetric Delta edits at every token.

The name **Causal LSSO** refers to solved contextual adaptation: every prefix
generates a compact geometry system, and the action of solving that system
defines the frame in which causal memory is edited and read. It does not mean
applying a causal mask to the bidirectional LSSO operator or retaining its
specific `I + F F^T + Omega` chart. The solve principle is the inheritance;
the causal system and its hardware form are part of the new model design.

This repository is independent of the upstream
[`LSSO`](https://github.com/Yang916-yy/LSSO) project. That repository is a
source of mathematical properties and derivation evidence; its exact operator
chart is not an architectural constraint, and implementation code is not
imported.

## Quick start

The FP64-friendly token reference and CPU layer require Python 3.10+ and
PyTorch. Install the package and tests in an isolated environment:

```bash
python -m pip install -e ".[test]"
python -m pytest -q -s
```

Minimal layer use without the optional optimized convolution/backend:

```python
import torch

from causallsso import SolveDelta, SolveDeltaConfig

config = SolveDeltaConfig(
    hidden_size=256,
    num_heads=4,
    head_k_dim=64,
    head_v_dim=64,
    use_short_conv=False,
)
layer = SolveDelta(config)
hidden = torch.randn(2, 32, 256)
output = layer(hidden)
```

The optimized CUDA path additionally requires Flash Linear Attention. The
checked-in native specialization targets CUDA 13, MathDx 26.06/cuBLASDx 0.7,
and SM120; see [`docs/PARALLELISM.md`](docs/PARALLELISM.md) for its deliberately
narrow support boundary.

The Python wheel contains the reference and Python-managed kernels, not the
native MathDx library. Configure that optional extension separately from a
source checkout by pointing CMake at PyTorch and a MathDx installation:

```bash
CUDACXX=/path/to/nvcc cmake -S native -B build/native \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DMATHDX_ROOT=/path/to/mathdx
cmake --build build/native --parallel
```

Third-party attribution for the adapted NVIDIA TRSM sample is recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). No repository-wide license
has been selected yet.

## One current model

- **Operator:** SolveDelta, defined in
  [`docs/INNOVATION_PROGRAM.md`](docs/INNOVATION_PROGRAM.md).
- **Geometry width:** `r := d_k`, where `d_k` is the resolved per-key-head
  width supplied by the containing model. There is no separate rank setting;
  `r=128` is the first native specialization and main benchmark profile.
- **Edit width:** `num_edits = K`, a positive static hyperparameter; `K = 1` is
  the recommended performance default. Larger values retain ordered
  DeltaProduct-style capacity, and all edits share one solve adapter.
- **Short convolution:** three independent depthwise causal `conv4` + SiLU
  branches process projected query, packed keys, and packed values before head
  reshape and normalization. The fixed frontend is enabled by default; geometry
  and gate projections bypass it.
- **Reference:** [`causallsso/reference.py`](causallsso/reference.py), the one
  slow token recurrence and sole executable owner of operator mathematics.
- **LSSO inheritance:** preserve prefix-conditioned solve geometry, fixed-size
  sufficient statistics, global mixing, and a controlled invertible frame.
  Matching the no-forgetting Prefix-LSSO moment derivation is a provenance
  diagnostic, not a requirement to retain its exact chart.
- **Backend:** exact generalized asymmetric Delta/WY, informed by mature GDN2
  and GatedDeltaProduct implementations.

GDN2, DeltaNet, KDA, and DeltaProduct-`K` are exact reductions. Preconditioned
Delta rules are matched baselines. None is a parallel route.

## Geometry and memory

Per head, the operator state is

\[
m_t\in\mathbb R,
\qquad J_t,D_t\in\mathbb R^{r\times r},
\qquad S_t\in\mathbb R^{r\times d_v}.
\]

The complete layer state also carries raw projected-input conv4 caches
`C_q:[B,Hr,4]`, `C_k:[B,HKr,4]`, and `C_v:[B,HKd_v,4]`. Their last axis runs
from oldest to newest. They use the projection activation dtype, start at zero,
hold on invalid tokens, reset immediately before a valid reset token, and shift
in that token before depthwise convolution and SiLU. Gradients include both the
convolution outputs and any returned final caches. With the structural
`use_short_conv=False` switch, the three projections receive SiLU directly and
the corresponding cache fields are `None`.

An independent geometry feature `u_t` and already-driven core feature `h_t`
update

\[
m_t=\lambda_t^{(g)}m_{t-1}+1,
\]

\[
J_t=\lambda_t^{(g)}J_{t-1}+u_tu_t^T,
\qquad
D_t=\lambda_t^{(g)}D_{t-1}+u_th_t^T.
\]

With L2-normalized `u_t`, form

\[
H_t=J_t/m_t,
\qquad R_t=D_t/m_t.
\]

These moments remain separate until after their bounded nonlinear maps and
then generate one dense, directly factored contextual system,

\[
X_t^{(H)}=\gamma_g(H_t-I/r),
\qquad X_t^{(R)}=\gamma_gR_t.
\]

\[
N_t^\pm=
\mathcal B_{1/8}(\operatorname{tri}_\pm X_t^{(H)})+
\mathcal B_{1/8}(\operatorname{tri}_\pm X_t^{(R)}),
\]

with the diagonal log scale formed by the analogous sum of two radius-`1/8`
`tanh` maps, followed by

\[
M_t=(I+N_t^-)\Sigma_t(I+N_t^+),
\qquad P_t=M_t^{-1}.
\]

`N^-` and `N^+` are smoothly bounded strict-triangular coordinates and
`Sigma` has bounded positive diagonal. As a map of ambient `(X^(H),X^(R))`
coordinates, the chart has all `r^2` local matrix directions at identity, a certified
condition-number bound of about `4.58`, and requires no token-local dense
factorization. This does not mean every short prefix reaches all those
directions: `rank(D_t), rank(R_t) <= t`, and full local `R` reachability
at nonzero geometry strength requires a remembered `u` span of rank `r`. Its exact dual is
`P_t^-T=M_t^T`. The separate nonlinear maps are essential: summing `H` and
`R` first makes `D+J=sum u(h+u)^T` a single cross moment and removes the
independent occupancy statistic.

Local token projections define solve-domain edits; the prefix adapter maps
write vectors through `P_t` and erase/read covectors through `P_t^-T`. The
associative state remains in one fixed basis, is decayed once, edited `K` times
in order, and then read. Geometry never reads or transports that state.

## Expressivity contract

The one operator must contain, by exact algebraic restrictions:

\[
\text{DeltaNet/GDN/KDA/GDN2}
\subset
\text{SolveDelta},
\]

and it must contain DeltaProduct-`K` at identity geometry with symmetric edit
factors. The canonical write direction is the normalized key itself; there is
no auxiliary sigmoid write-direction gate whose unattainable endpoint is
needed for these reductions. One rank-one edit fixes an `r-1` dimensional subspace; `K` ordered
edits can produce an identity-plus-rank-at-most-`K` transition and, from
`K >= 2`, realize transformations such as planar rotations that one edit
cannot. Prefix geometry additionally makes the edit coefficients depend on
every token represented in the fixed-size prefix moments rather than only the
current token. This is lossy full-prefix conditioning: histories with identical
`(m,J,D)` remain indistinguishable to the solve adapter, while the ordered
associative state can still distinguish them. Fixed-state collisions already
exist in Delta-family recurrences through compression into `S`; SolveDelta adds a
separate, explicitly second-order geometry equivalence relation. A collision
of `(m,J,D)` is not a collision of the complete recurrent state unless `S`
also agrees.

These are fixed-width, single-layer operator claims. They do not imply that
SolveDelta contains arbitrary dense transitions or every deeper/wider Delta model.

## Execution status

The implementation program has three one-way stages:

\[
\text{exact affine geometry scan}
\rightarrow
\text{one token-local primal/dual solve adapter}
\rightarrow
\text{K-edit asymmetric Delta/WY}.
\]

The last stage has mature engineering precedents at `r = 128`: GDN2 supplies
asymmetric erase/write WY and GatedDeltaProduct supplies ordered multi-edit
packing. The geometry scan is exactly associative. The causal solve is two
unit-triangular solves plus diagonal scaling. Training uses a hybrid backend:
Triton owns the chunk-boundary scan and Delta/WY exterior, while a
CUDA C++/MathDx operator reconstructs each chunk locally and applies the
block-level TRSM without materializing `T x r x r` prefix states.

Prior local probes established:

- the former H+S/dissipativity interpretation of a general asymmetric
  rank-one edit was false and has been removed;
- directly accumulating the driven cross moment is FP64-equivalent to the
  former `C W_drive` path to about `1e-15` in local probes and removes one
  dense matrix product;
- the former `G+A` chart exposed an unbounded transpose direction and required
  two token-local dense factorizations, so it was rejected for the causal
  operator rather than repaired further;
- the selected bounded LDU chart is full-dimensional in its ambient coordinates
  at identity, while SPD, one-sided orthogonal-scale, butterfly, and Woodbury
  candidates restrict the local matrix family; feasible early-prefix moments
  retain their separate rank/span reachability constraint;
- at `r=128`, the two-coordinate direct LDU action was about `3.7--5.2x`
  faster than the former chart in an isolated RTX 5070 Ti PyTorch probe,
  depending on token/head batch size;
- in FP64, factorwise triangular execution matched an explicit dense solve to
  `2.2e-16` forward relative error and `4.8e-16` gradient relative error;
- saturated random-coordinate probes realized maximum condition numbers below
  `1.76`, versus the analytic global bound of about `4.58`.

These are selection probes, not complete-layer results. The standalone native
oracle uses algebraically exact, non-iterative FP32 MathDx block TRSM; "exact"
does not mean bitwise equality with FP64. The selected fixed-length forward now
uses exact coordinate-packet triangular substitution. The earlier bounded
Neumann implementation remains an independent validation interface and
derivative oracle, not the production packet backward.

Native correctness is measured against the FP64 token recurrence using RMS
relative error, with inputs first quantized to the advertised runtime dtype and
then promoted unchanged for the oracle. The contract follows GDN2's
forward/state/gradient separation for complete BF16/FP16 paths, while imposing
stricter independent ceilings on FP32 Triton geometry scans, MathDx solve
residuals and actions, dual pairing, and the complete FP32 layer. Exact values
and required shape/sequence coverage live in
[`docs/VALIDATION_PLAN.md`](docs/VALIDATION_PLAN.md); passing only the final
output tolerance is insufficient.

The implementation was validated in an isolated environment with PyTorch
`2.13.0+cu130`, Triton `3.7.1`, CUDA 13.0 Update 2, MathDx 26.06 /
cuBLASDx 0.7.0, and FLA `0.5.2`. CUDA, Triton, representative Delta-family
forward/backward passes, a PyTorch CUDA C++ extension, and the official MathDx
block-TRSM SM120 path have been smoke-tested. The repository now contains and
tests an autograd-capable FP32 Triton chunk-boundary geometry scan, an exact
FP32 `r=128,K=1,C=16` packet frame, and the FLA generalized-DPLR chunk/WY outer
mapped exactly to ordered SolveDelta edits. `packet_frame128` reconstructs
affine local prefixes, evaluates four stable radial invariants without
tokenwise dense replay, and carries two `C x C` coordinate packets through
lower, upper, transpose-dual, and skew actions. It does not materialize
`T*r*r` factors. The standalone
exact MathDx frame remains the factor-action oracle and decode candidate.
`cuda_chunk_solve_frame128` retains the bounded fourth-order Neumann path only
as a validation interface. The packet-native VJP keeps each token factor
cotangent as at most five masked outer products and contracts those descriptors
directly against dense boundaries and local semiseparable generators.

`solvedelta_fused` composes the Triton scan, exact packet frame, and FLA DPLR
recurrence. Its packet-native backward differentiates the packed primal and
dual actions, applies the compact rank-five chart contraction, evaluates the
radial and prefix VJPs, and reverse-scans chunk-boundary moment cotangents. It
saves no tokenwise `r x r` chart or factor. A fused Triton pack handles
`K>1`; `K=1` bypasses micro-time packing. The FLA WY cache remains selected:
on the current K1 target it makes outer forward-plus-backward about `12.2%`
faster for `2.75 MiB` additional allocation.

On the local RTX 5070 Ti / SM120 target, the matched
`B=1,T=1024,H=8,r=d_v=128,hidden_size=1024,K=1` endpoint measures about
`3.770 ms` forward and `12.315 ms` forward-plus-backward for the complete BF16
conv4 layer. The installed FLA GDN2 layer measures about `0.992 ms` and
`3.068 ms`, leaving roughly a `3.8x` forward and `4.0x` training gap. The
packet-native VJP is implemented and validated; a later five-descriptor qbar
contraction reduced its same-process packet backward by `0.534 ms` and packet
forward-plus-backward by `0.637 ms` relative to the immediately preceding
checkpoint. This is a useful exact contraction reduction, not performance
parity. Masks/resets, generic native `K`/rank, packed variable lengths, and
non-SM120 specializations remain release work.

## Audit attribution

| Issue | Already present in Delta family? | SolveDelta decision |
|---|---|---|
| rank-`K` transition ceiling and multi-edit packing | Yes, DeltaProduct | expose `K`; reuse ordered packing |
| asymmetric erase non-normality | Yes, GDN2 | do not claim Euclidean contraction; measure singular values |
| decay prefix products and WY scaling | Yes, GDN/KDA/GDN2 | reuse log-space gates and mature scan algebra |
| two dense prefix moments | No | accept as deliberate capacity: `J` stores occupancy geometry and `D` stores directional drive; map them separately, fuse their scan, and pack symmetric `J` |
| unbounded inverse-transpose adapter | No; original LSSO uses only the bounded solve | replace the bidirectional chart with a bi-bounded LDU causal system |
| redundant `C W_drive` matrix product | No | accumulate the exactly equivalent driven moment `D` directly |
| token-local dense factorizations | No | eliminate them; apply two owned unit-triangular factors |
| fixed-state history collisions | Yes; every fixed-size Delta recurrence compresses history into `S` | `(m,J,D)` adds an explicit second-order geometry collision class; distinguish solve-adapter collision from complete-state collision |
| positive pairing mistaken for dissipativity | Not a Delta claim | remove H+S label and retain only the valid eigenvalue certificate |

## Recorded open problems

The current route has three explicit unresolved research or release boundaries:

1. find an algebraically equivalent frame and frame-VJP algorithm that changes
   the cost of the dense per-token actions, rather than only removing constant
   overhead from the implemented rank-five packet VJP;
2. verify on training runs that the LDU bound keeps non-normal transient amplification and
   primal/dual magnitude balance inside the declared envelope;
3. demonstrate that the orthogonal erase residual, independent geometry
   feature, `J`, and `D` are used under matched parameter and compute budgets.

`num_edits = K` remains an ordinary static DeltaProduct capacity/compute
hyperparameter. The default `K = 1` and any supported larger values must be
benchmarked and reduction-tested, but their rank--cost tradeoff is not a
Causal LSSO research blocker.

They are acceptance gates, not alternative architecture branches.

## Repository map

- [`AGENTS.md`](AGENTS.md): contribution and review contract.
- [`docs/INNOVATION_PROGRAM.md`](docs/INNOVATION_PROGRAM.md): canonical model
  mathematics, containment, and limitations.
- [`docs/PARALLELISM.md`](docs/PARALLELISM.md): exact and optimized execution.
- [`docs/VALIDATION_PLAN.md`](docs/VALIDATION_PLAN.md): acceptance gates.
- [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md): primary sources and decisions.
