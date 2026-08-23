# SolveDelta

> **Status:** unfinished research repository. The FP64 operator contract is
> frozen; the single `chunk + WY` training backend is implemented as a dense
> `r=128`, `K=1`, SM120 prototype with a complete forward and backward. It has
> passed the current dense composition tests, but not the full model-dispatch,
> mask/reset, numerical-release, or matched-GDN2 performance gates.

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

The Triton affine geometry boundary scan and its adjoint are checked in. The
former packet, panel, standalone polynomial solve, isolated chart-VJP, and old
multi-VJP frame paths were removed. One replacement `r=128`, `K=1`, `C=32`
CUDA operator now owns frame forward and backward under a single ABI. Its
forward, per-action and joint backward, irregular-tail, identity,
zero-boundary-mass, cancellation, determinism, and complete dense
scan/frame/FLA-WY composition tests pass against the FP64 token oracle.

The frame reverse assigns the eight `16 x 16` coordinate tiles through one
diagonal phase and seven deterministic perfect-matching phases. Every dense
moment entry is replayed once, and each phase has disjoint output ownership, so
the implementation needs neither atomics nor a full-sequence `T x r x r`
workspace. On the local SM120 target profile
`B=1,T=1024,H=8,r=d_v=128,K=1`, the dense prototype measured about `1.918 ms`
forward and `14.622 ms` forward-plus-backward. This replaces the previous
roughly `118 ms` frame-backward path, but is still not close enough to the
matched GDN2 target to enable model dispatch. The uncompensated local
`boundary_m` cancellation diagnostic also remains a declared numerical
limitation; no compensation or tolerance fallback is hidden in the backend.

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

Third-party attribution for the adapted NVIDIA TRSM sample is recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No repository-wide license
has been selected yet.

## Repository map

- [AGENTS.md](AGENTS.md): contribution and operator contract.
- [causallsso/reference.py](causallsso/reference.py): FP64 token oracle.
- [docs/INNOVATION_PROGRAM.md](docs/INNOVATION_PROGRAM.md): model mathematics,
  reductions, and limitations.
- [docs/PARALLELISM.md](docs/PARALLELISM.md): exact chunk/WY execution program.
- [docs/VALIDATION_PLAN.md](docs/VALIDATION_PLAN.md): numerical and performance
  gates.
- [docs/PRIOR_ART.md](docs/PRIOR_ART.md): primary sources and decisions.
