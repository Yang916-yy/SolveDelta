# Third-Party Notices

## Flash Linear Attention and MESA

The selected native path adapts and specializes portions of Flash Linear
Attention (FLA), including MESA and generalized Delta-rule kernels. FLA is
available under the MIT License.

Adapted production files:

- `causallsso/modeling_solvedelta.py` and `causallsso/config.py`: FLA's
  GatedDeltaNet/MESA Hugging Face model, hybrid-attention, recurrent Cache,
  RMSNorm, GatedMLP, and fused-loss ownership specialized around the
  SolveDelta mixer and `(m,J,D,S)` state;
- `causallsso/model.py`: FLA GDN2/Mamba-style log-rate and inverse-softplus
  decay initialization, with the geometry heads spread across the RLS
  CG5-safe initial decay interval;
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
- `causallsso/ops/rls/block_e3_sources.py` and
  `block_e3_pair_reverse.py`: FLA common beta-sigmoid fused into the strided
  source forward/transpose without an activated-gate HBM panel;
- `causallsso/ops/rls/strided_l2norm.py`: FLA L2Norm arithmetic and row
  ownership specialized to fused-projection source strides;
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
- `fla/ops/utils/`;
- `fla/models/{gated_deltanet,mesa_net}/`, `fla/models/hybrid.py`, and
  `fla/models/utils.py`;
- `fla/layers/{gated_deltanet,gdn2}.py`.

The reviewed upstream revisions, concrete mappings, and production decisions
are recorded in `docs/PRIOR_ART.md`.

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

## Mamba

Mamba-2 and Mamba-3 informed fused-projection layout, resident value-tile
state, output-owned reverse, and decay initialization. No Mamba source file is
vendored in this repository. Mamba is distributed under the Apache License
2.0. Upstream repository:
<https://github.com/state-spaces/mamba>.

## causal-conv1d

The frontend uses `causal-conv1d` for depthwise conv4, SiLU, and final-state
VJP. It is an external runtime dependency distributed under the BSD 3-Clause
License. Upstream repository:
<https://github.com/Dao-AILab/causal-conv1d>.

No repository-wide license has been selected for code not covered by these
notices.
