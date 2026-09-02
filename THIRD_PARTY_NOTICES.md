# Third-Party Notices

## Flash Linear Attention

The selected native path adapts and specializes portions of Flash Linear
Attention (FLA), including gated Oja and generalized Delta-rule kernels. FLA
is available under the MIT License.

Adapted production files:

- `causallsso/modeling_solvedelta.py` and `causallsso/config.py`: FLA's
  GatedDeltaNet/MESA Hugging Face model, hybrid-attention, recurrent Cache,
  RMSNorm, GatedMLP activation/down owner, and fused-loss ownership specialized
  around the SolveDelta mixer, packed common-input gate/up projection, and
  `(C,S)` state;
- `causallsso/model.py`: FLA GDN2/KDA-style low-rank coordinate decay,
  independent erase/write gates, and fused-projection ownership;
- `causallsso/ops/norm_gate.py`: FLA fused sigmoid RMSNorm-gate row ownership,
  FP32 reductions, strict transpose, and norm-linear checkpoint ownership;
- `causallsso/ops/residual_frame/predictor.py`: FLA gated-Oja pair,
  triangular-WY, state/output, and strict-transpose owners specialized to
  shared coordinate-decay normalized LMS residual prediction and accumulated
  primal action;
- `causallsso/ops/residual_frame/leaky_wy.py`: FLA gated-Oja WY recompute and
  transpose schedules specialized to separate target-write and source-erase
  coefficients, fused vector-gate suffix closure, and
  final-shaped source cotangent ownership;
- `causallsso/ops/residual_frame/vector_pair.py`: FLA gated-Oja coordinate-gate
  pair schedule and transpose specialized to the residual-before-decay
  exclusive prefix;
- `causallsso/ops/residual_frame/exterior.py`: FLA generalized-DPLR fast-WY,
  state/output forward, and matching reverse specialized to direct relative
  frame sources;
- `causallsso/ops/residual_frame/common_left_{h,o_fwd,o_bwd}.py`: FLA DPLR
  state/output owners specialized to the exact `k=b=d`, `A_qk=A_qb`
  common-left case and its output-owned transpose;
- `causallsso/ops/residual_frame/pair.py`: FLA generalized-DPLR exact-unbounded
  Triton pair forward/transpose specialized to one token-local direct-`e`
  edit and the source-native panel layout;
- `causallsso/ops/residual_frame/sources.py`: FLA/GDN2 output-owner style
  dual source generation and transpose, including strided q L2Norm,
  scaled-L2Norm frame parameterization, and erase/write gate epilogues;
- `causallsso/ops/residual_frame/l2norm.py`: FLA L2Norm arithmetic and row
  ownership specialized to fused-projection source strides.
- `causallsso/ops/residual_frame/recurrent.py`: FLA fused-recurrent Oja state
  ownership specialized to pre-forgetting residual order, composed with FLA's
  inference-only generalized-DPLR recurrent memory owner;
- `causallsso/graph_linear_cross_entropy.py`: FLA fused-linear cross-entropy
  chunking and Triton kernels specialized to a static dense CUDA Graph
  denominator and graph-safe cotangent epilogue.

Principal upstream source areas:

- `fla/ops/gated_oja_rule/`;
- `fla/ops/generalized_delta_rule/dplr/`;
- `fla/ops/common/{chunk_h,gate}.py`;
- `fla/ops/{gdn2,kda}/`;
- `fla/modules/fused_norm_gate.py`;
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

## Runtime Frameworks

PyTorch provides the tensor runtime, autograd, Triton dependency, CUDA Graphs,
and allocator. PyTorch is distributed under its BSD-style license:
<https://github.com/pytorch/pytorch>.

Hugging Face Transformers provides configuration, auto classes, checkpointing,
and causal-LM interfaces. Transformers is distributed under Apache-2.0:
<https://github.com/huggingface/transformers>.

Triton compiles the custom GPU kernels and is installed in a version matched to
PyTorch. Triton is distributed under the MIT License:
<https://github.com/triton-lang/triton>.

These packages are runtime dependencies and are not vendored. Supported and
validated versions are listed in `docs/ENVIRONMENT.md`.

No repository-wide license has been selected for code not covered by these
notices.
