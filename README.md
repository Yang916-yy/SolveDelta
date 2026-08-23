# SolveDelta

> **Status:** unfinished research repository. The FP64 operator contract and
> first BF16-operands/FP32-accumulation numerical contract are frozen. The one
> native training path is implemented for `r=128`, `K=1`, `C=32` on SM120, but
> its resident frame reverse is still the dominant performance and workspace
> cost. Deep-cancellation shared-strength gradients use a separately derived
> tying-map error metric; they are not hidden by a condition-number multiplier.

Causal LSSO studies one causal sequence operator: **SolveDelta**. A decayed
prefix constructs a compact bounded system, whose primal and exact
transpose-dual actions condition ordered Delta edits in a fixed ambient memory
basis.

The name describes solved contextual adaptation. It does not require the
bidirectional LSSO chart, a particular matrix factorization lineage, or an
additional architecture. DeltaNet, GDN, KDA, GDN2, and DeltaProduct are exact
reductions or comparison baselines.

## Current contract

- `causallsso/reference.py` is the sole executable owner of operator
  mathematics and the FP64 numerical oracle.
- Geometry width is `r := d_k`, the resolved per-key-head width. `r=128` is the
  first native specialization, not a mathematical default.
- `num_edits = K` is a positive static hyperparameter. `K=1` is the default.
- Query, packed edit keys, and packed edit values each receive an independent
  GDN2-style depthwise causal `conv4` followed by SiLU. It is enabled by
  default and may be structurally disabled.
- The geometry states `J` and `D` remain separate through separate bounded
  nonlinear maps. They are not compressed or summed before the chart.
- The associative memory stays in one fixed basis. Writes use the primal
  action; erase and read covectors use the exact dual action.
- Channel-wise associative decay is enabled by default and has one structural
  off intervention for exact ungated Delta-family reductions.
- Native activations and large contraction operands are BF16. Tensor Core
  contractions, normalization/radial reductions, gate and decay evaluation,
  backward partials, and recurrent `m,J,D,S` states accumulate or reside in
  FP32. State is never rounded at chunk boundaries.

Per head, the recurrent operator state is

\[
m_t\in\mathbb R,\qquad
J_t,D_t\in\mathbb R^{r\times r},\qquad
S_t\in\mathbb R^{r\times d_v}.
\]

For normalized geometry features `u_t`,

\[
\begin{aligned}
m_t &= \lambda_t^{(g)}m_{t-1}+1,\\
J_t &= \lambda_t^{(g)}J_{t-1}+u_tu_t^T,\\
D_t &= \lambda_t^{(g)}D_{t-1}+u_th_t^T,\\
H_t &= J_t/m_t,\qquad R_t=D_t/m_t.
\end{aligned}
\]

The two moments separately generate bounded strict-lower, diagonal, and
strict-upper coordinates. They define

\[
M_t=(I+N_t^-)\Sigma_t(I+N_t^+),\qquad P_t=M_t^{-1}.
\]

At every token, one shared frame transforms all edit vectors:

\[
d_{t,j}=P_ta_{t,j},\qquad
e_{t,j}=P_t^{-T}b_{t,j},\quad b_{t,j}=\beta_{t,j}\odot a_{t,j},\qquad
\chi_t=P_t^{-T}q_t.
\]

After one channel-wise memory decay, the `K` edits execute in order and the
result is read after edit `K`:

\[
S_{t,j}=(I-d_{t,j}e_{t,j}^T)S_{t,j-1}+d_{t,j}z_{t,j}^T,
\qquad y_t=S_{t,K}^T\chi_t.
\]

The normalized edit key is the local write direction, and its elementwise
erase gate defines the solve-domain erase covector. There is no skew branch.

## Implementation status

The intended training path has one direction:

\[
\text{Triton geometry boundaries}
\longrightarrow
\text{chunk-owned frame forward/backward}
\longrightarrow
\text{FLA generalized Delta/WY}.
\]

Its required execution form is BF16 Tensor Core operands with FP32
accumulation and continuation state. The FP64 oracle is evaluated after the
runtime operands have been rounded once to BF16, so validation separates input
representation from kernel arithmetic.

The Triton affine scan and adjoint consume BF16 geometry activations while
keeping chunk boundaries and final `(m,J,D)` states in FP32. The resident CUDA
frame consumes BF16 vectors and packed factors, accumulates actions and reverse
partials in FP32, emits BF16 `d,e,chi`, and saves compact FP32 chart data for one
joint backward. The composed autograd owner joins the frame cotangents with the
FP32 scan adjoint before any BF16 leaf cast. Tensor Core instructions are used
by the Triton scan/WY contractions and the resident reverse's broad BF16
`bmm`s; the custom resident forward and compact CUDA primitives remain
scalar-FMA kernels. Full block-action MMA utilization is still open.

The WY exterior specializes FLA 0.5.2's mature generalized-Delta kernels to
generate `a=-e*exp(g)` at use. It neither materializes nor saves `a`, and its
backward folds the pullback directly into `e` and `g`. The current frame/WY
boundary still explicitly materializes `d,e,chi`; this is not a fully fused
Solve-to-WY kernel. The former packet, panel, `chunk_frame`,
`tensorcore_frame`, `triton_frame`, standalone polynomial solve, isolated
chart-VJP, and all-FP32 C32 ABIs have been removed rather than retained as
compatibility paths.

On the local SM120 target profile
`B=1,T=1024,H=8,r=d_v=128,K=1,C=32`, warmed medians are:

| Component | Forward | Forward + backward |
|---|---:|---:|
| Triton geometry scan | `0.119 ms` | `0.695 ms` |
| resident frame | `1.243 ms` | `6.174 ms` |
| direct-`e` WY exterior | `0.234 ms` | `0.936 ms` |
| complete SolveDelta operator | `1.669 ms` | `7.740 ms` |
| matched GDN2 operator | `0.358 ms` | `1.154 ms` |

Resident frame backward is therefore about `4.93 ms` and remains the primary
optimization target; the direct-`e` WY exterior is already close to the
matched GDN2 core. The current implementation passes ordinary forward,
state, VJP, irregular-tail, identity, and repeatability checks. In two deepest
`2^12` cancellation fixtures the six shared-strength channels nearly cancel in
the final tied scalar, making relative-to-total error ill-conditioned. Those
fixtures are judged by the operator-scaled error of the fixed six-to-one tying
map, with a `2.5e-2` ceiling and the existing `1e-6` absolute branch when the
six-channel reference scale itself is near zero. Ordinary strength fixtures
retain the standard total-gradient metric; no condition-number multiplier or
warning-only exception is used.

These are operator measurements: SolveDelta includes normalization, geometry
scan, resident frame, WY, and final state; GDN2 includes its corresponding
normalization/core/final-state work. Neither row includes input/output
projections or conv4.

MathDx is retained only as an optional exact `r=128` triangular validation
oracle and possible decode candidate. It is not imported by model dispatch and
is not a default build dependency. FLA owns the mature generalized Delta/WY
exterior; project code owns the geometry scan and local frame action.

See [docs/PARALLELISM.md](docs/PARALLELISM.md) for the execution design and
[docs/VALIDATION_PLAN.md](docs/VALIDATION_PLAN.md) for acceptance gates.

## Quick start

The reference layer requires Python 3.10+ and PyTorch:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

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
layer = SolveDelta(config).double()
hidden = torch.randn(2, 32, 256, dtype=torch.float64)
output = layer(hidden)
```

The SM120 native frame library and optional MathDx oracle are built with CMake.
The dense composition additionally requires FLA:

```bash
python -m pip install -e ".[native,test]"
CUDACXX=/path/to/nvcc cmake -S native -B build/native \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build/native --parallel
```

Enable the MathDx oracle separately when its SDK is installed:

```bash
CUDACXX=/path/to/nvcc cmake -S native -B build/native \
  -DCAUSALLSSO_BUILD_MATHDX_ORACLE=ON \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DMATHDX_ROOT=/path/to/mathdx
cmake --build build/native --parallel
```

Third-party attribution for the adapted NVIDIA TRSM and FLA DPLR/WY sources is
recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No repository-wide
license has been selected yet.

## Repository map

- [AGENTS.md](AGENTS.md): contribution and operator contract.
- [causallsso/reference.py](causallsso/reference.py): FP64 token oracle.
- [docs/INNOVATION_PROGRAM.md](docs/INNOVATION_PROGRAM.md): model mathematics,
  reductions, and limitations.
- [docs/PARALLELISM.md](docs/PARALLELISM.md): exact chunk/WY execution program.
- [docs/VALIDATION_PLAN.md](docs/VALIDATION_PLAN.md): numerical and performance
  gates.
- [docs/PRIOR_ART.md](docs/PRIOR_ART.md): primary sources and decisions.
