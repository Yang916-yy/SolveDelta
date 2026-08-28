# Third-Party Notices

## Flash Linear Attention and MESA

The selected native path adapts and specializes portions of Flash Linear
Attention (FLA), including MESA and generalized Delta-rule kernels. FLA is
available under the MIT License.

Adapted production files:

- `causallsso/ops/rls/mesa_specialized.py`: paired MESA `Hkk/Hkv` state,
  matrix-free CG, covariance/cross-moment reverse, and implicit transpose;
- `causallsso/ops/rls/mesa_gain.py`: SolveDelta ownership and composed VJP for
  those MESA blocks;
- `causallsso/ops/rls/block_e3_pair.py` and
  `block_e3_pair_reverse.py`: FLA generalized-DPLR TileLang pair ownership
  specialized to native token/slot `E=3` and direct-`e` gauge terms;
- `causallsso/ops/rls/block_e3_wy.py`: FLA fast-WY triangular action
  specialized to a C48 logical interaction block;
- `causallsso/ops/rls/block_e3_state.py`, `block_e3_reverse.py`, and
  `block_e3_exterior.py`: FLA/GDN2 state/output and transpose ownership adapted
  to the RLS source mapping;
- `causallsso/ops/rls/gate.py`: FLA common beta-sigmoid specialized to BF16
  output;
- `causallsso/ops/rls/mass.py`: scalar affine scan and transpose following FLA
  chunk-state ownership;
- `causallsso/ops/gates.py`: FLA GDN decay gate specialized to the model's
  strided projection views.

Principal upstream source areas:

- `fla/ops/mesa_net/`;
- `fla/ops/generalized_delta_rule/dplr/`;
- `fla/ops/common/{chunk_h,gate}.py`;
- `fla/ops/{gdn2,kda}/`;
- `fla/modules/l2norm.py`;
- `fla/ops/utils/`.

The reviewed upstream revisions and measured adoption decisions are recorded
in `docs/PRIOR_ART.md`.

> MIT License
>
> Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

The MIT license text is included at `LICENSES/MIT.txt`. Upstream repository:
<https://github.com/fla-org/flash-linear-attention>.

## TileLang

The block-E3 pair/WY owners use TileLang as a runtime/compiler dependency and
specialize FLA's TileLang scheduling patterns. TileLang 0.1.13 is distributed
under the MIT License and includes separately licensed bundled components.
Upstream repository:
<https://github.com/tile-ai/tilelang>.

## causal-conv1d

The frontend uses `causal-conv1d` for depthwise conv4, SiLU, and final-state
VJP. It is an external runtime dependency distributed under the BSD 3-Clause
License. Upstream repository:
<https://github.com/Dao-AILab/causal-conv1d>.

No repository-wide license has been selected for code not covered by these
notices.
