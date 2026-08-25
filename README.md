# SolveDelta

> **Status:** unfinished research repository. The FP64 operator contract and
> first BF16-observable, bounded-private-FP16, FP32-accumulation numerical
> contract are frozen. The one
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
- Public and raw native activations remain BF16. Analytically bounded private
  panels are produced in FP32 and stored directly as FP16. Tensor Core
  contractions and backward partials accumulate in FP32; normalization/radial
  reductions, gate and decay evaluation, and recurrent `m,J,D,S` states also
  compute or reside in FP32. State is never rounded at chunk boundaries, and
  BF16-to-FP16 casts are not treated as precision promotion.

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
\text{chunk-owned frame and pair statistics}
\longrightarrow
\text{FLA-derived C32 WY blocks and project-owned state scan}.
\]

The native precision contract uses BF16 public/raw operands, FP32-produced
private FP16 panels where the chart supplies an analytic range certificate, and FP32
Tensor Core accumulation and continuation state. The FP64 oracle is evaluated
after public runtime operands have been rounded once to BF16. A same-packed
internal oracle may consume exact FP16 panel bits to diagnose one contraction;
it does not define the production packing or reduction schedule.

In the selected rewrite, Triton normalization loads raw BF16 geometry, query,
and key activations, computes in FP32, and writes private FP16 `u/q/k` panels;
the affine scan consumes `u` while keeping chunk boundaries and final
`(m,J,D)` states in FP32. Bounded frame producers likewise write erase-source,
strict-coordinate, and certified frame-action panels directly from FP32 to
FP16. Paired actions and their transpose accumulate in FP32 and may use
different algebraically equivalent private layouts. Radial, strict-diagonal,
and sensitive scalar reductions remain FP32, as do all state and scalar
cotangents. Nonstructural deep cancellation is accepted at the composed output,
state, and VJP, not by requiring a private expanded norm or chart coordinate to
be bitwise exact. Structural
identity remains exact. Analytically bounded private action panels are written
directly to FP16 by their FP32 producers and consumed with FP32 accumulation;
public BF16 values are never cast to FP16 as pseudo-promotion. Remaining
pair-to-frame fusion is rewrite work, not a completed claim.

The C32 WY owner specializes FLA 0.5.2's 16+16 unit-lower inverse, wide-RHS
application, and matrix reverse to SolveDelta's native panels. It keeps the
inverse private, never materializes a concatenated RHS, applies each solve
block with one direct BF16 Tensor Core product and FP32 accumulation, and fuses
`-barB X^T` with the `write/value` pullback. The current frame/WY boundary
still explicitly materializes private `d,e,chi` caches; this is not a fully
fused Solve-to-WY kernel. The former packet, panel, `chunk_frame`,
`tensorcore_frame`, `triton_frame`, standalone polynomial solve, isolated
chart-VJP, and all-FP32 C32 ABIs have been removed rather than retained as
compatibility paths.

On the local SM120 target profile
`B=1,T=1024,H=8,r=d_v=128,K=1,C=32`, warmed medians are:

| Complete-operator measurement | Scalar C32 at `6d4e53f` | Initial FLA C32 block | Current symmetric streamed reverse |
|---|---:|---:|---:|
| forward | `1.639--1.643 ms` | `1.608--1.620 ms` | about `1.634 ms` |
| forward + backward | `6.885--7.122 ms` | `6.443--6.497 ms` | about `6.485 ms` |
| isolated paired WY forward/reverse | scalar reverse about `0.272 ms` | `0.023 / 0.040 ms` | unchanged |

The initial FLA C32 replacement improved complete forward-plus-backward by
about `5.6--9.5%`. Original-stride frame addressing, `tl.dot` output, paired
geometry tiles, fused normalization, symmetric-state ownership, and streamed
strict transpose then changed the current core. Current backward is about
`4.851 ms` by forward subtraction; the remaining
frame/radial/strict/state/scan aggregate,
not the roughly `0.040 ms` paired-WY reverse, is now the primary optimization
target. The separately matched GDN2 operator remains much faster at about
`0.358 ms` forward and `1.154 ms` forward-plus-backward. The current
implementation passes ordinary forward, state, VJP, irregular-tail, identity,
and repeatability checks. In two deepest
`2^12` cancellation fixtures the six shared-strength channels nearly cancel in
the final tied scalar, making relative-to-total error ill-conditioned. Those
fixtures are judged by the operator-scaled error of the fixed six-to-one tying
map, with a `2.5e-2` ceiling and the existing `1e-6` absolute branch when the
six-channel reference scale itself is near zero. Ordinary strength fixtures
retain the standard total-gradient metric; no condition-number multiplier or
warning-only exception is used.

These are operator measurements: SolveDelta includes normalization, geometry
scan, resident frame, WY, and final state; GDN2 includes its corresponding
normalization/core/final-state work. Neither operator timing includes
input/output projections or conv4.

The current complete target layer, including projections, three CUDA conv4
branches, fused gates, and all returned-state VJPs, measures about
`1.78/7.25 ms` forward/F+B on the same device.

MathDx is retained only as an optional exact `r=128` triangular validation
oracle and possible decode candidate. It is not imported by model dispatch and
is not a default build dependency. FLA supplies the attributed C32 inverse,
wide-RHS application, and matrix-reverse blocks; project code owns the
specialized composition, geometry scan, local frame action, pair reverse, and
state exterior.

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
