# Third-Party Notices

## NVIDIA cuBLASDx TRSM block example

[`native/mathdx_trsm.cu`](native/mathdx_trsm.cu) contains adapted and modified
portions of NVIDIA's cuBLASDx `trsm_block.cu` example.

> SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
> All rights reserved.
>
> SPDX-License-Identifier: Apache-2.0

The adapted portions are provided under the Apache License 2.0; see
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt). The upstream example ships
with NVIDIA MathDx 26.06 and is documented at
<https://docs.nvidia.com/cuda/cublasdx/using_trsm.html>.

This software contains source code provided by NVIDIA Corporation.

This notice applies only to the identified third-party-derived file. No
repository-wide license has been selected for the remaining SolveDelta code.

## Flash Linear Attention generalized Delta/WY kernels

[`causallsso/ops/paired_wy.py`](causallsso/ops/paired_wy.py) contains adapted
and modified portions of Flash Linear Attention 0.5.2's C32 unit-lower inverse
and generalized Delta Rule DPLR/WY wide-RHS application and matrix reverse.
The local specialization uses SolveDelta's contiguous panel layout, keeps the
inverse private to one kernel, applies it to the native edit/value RHS panels,
and fuses the `write * value` pullback without importing FLA's model ABI.

> MIT License
>
> Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

The adapted portions are provided under the MIT License; see
[`LICENSES/MIT.txt`](LICENSES/MIT.txt). The reviewed upstream functions are
`merge_16x16_to_32x32_inverse_kernel` in
<https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/utils/solve_tril.py>
and `wu_fwd_kernel` in
<https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py>
and `prepare_wy_repr_bwd_kernel` in
<https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/generalized_delta_rule/dplr/wy_fast_bwd.py>.

This notice applies only to the identified third-party-derived portions. No
repository-wide license has been selected for the remaining SolveDelta code.
