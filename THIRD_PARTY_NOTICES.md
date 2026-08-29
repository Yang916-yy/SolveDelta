# Third-Party Notices

## Flash Linear Attention

The selected native path adapts and specializes portions of Flash Linear
Attention (FLA), including gated Oja and generalized Delta-rule kernels. FLA
is available under the MIT License.

Adapted production files:

- `causallsso/modeling_solvedelta.py` and `causallsso/config.py`: FLA's
  GatedDeltaNet/MESA Hugging Face model, hybrid-attention, recurrent Cache,
  RMSNorm, GatedMLP, and fused-loss ownership specialized around the
  SolveDelta mixer and `(C,S)` state;
- `causallsso/model.py`: FLA GDN2/Mamba-style associative-decay
  parameterization and fused-projection ownership;
- `causallsso/ops/residual_frame/predictor.py`: FLA gated-Oja pair,
  triangular-WY, state, and strict-transpose owners specialized to normalized
  LMS residual prediction without the unrelated query/output branch;
- `causallsso/ops/residual_frame/exterior.py`: FLA generalized-DPLR fast-WY,
  state/output forward, and matching reverse specialized to direct relative
  frame sources;
- `causallsso/ops/residual_frame/pair.py`: FLA/KDA generalized-DPLR TileLang
  direct-`e` pair ownership specialized to one token-local edit;
- `causallsso/ops/residual_frame/sources.py`: FLA/GDN2 output-owner style
  source generation and transpose, including raw erase/write gate epilogues;
- `causallsso/ops/residual_frame/l2norm.py`: FLA L2Norm arithmetic and row
  ownership specialized to fused-projection source strides;
- `causallsso/ops/gates.py`: FLA GDN decay gate specialized to the model's
  strided projection views.

Principal upstream source areas:

- `fla/ops/gated_oja_rule/`;
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

The Residual-Frame direct-`e` pair owner uses TileLang as a runtime/compiler
dependency and specializes FLA's TileLang scheduling patterns. TileLang 0.1.13
is distributed under the MIT License and includes separately licensed bundled
components. Upstream repository:
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
