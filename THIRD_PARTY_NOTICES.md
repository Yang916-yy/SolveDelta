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

[`causallsso/ops/wy.py`](causallsso/ops/wy.py) contains adapted and modified
portions of Flash Linear Attention 0.5.2's generalized Delta Rule DPLR/WY
kernels. The local specialization consumes SolveDelta's erase direction
directly and generates `a = -e * exp(g)` at use instead of materializing it.

> MIT License
>
> Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

The adapted portions are provided under the MIT License; see
[`LICENSES/MIT.txt`](LICENSES/MIT.txt). The reviewed upstream source is at
<https://github.com/fla-org/flash-linear-attention/tree/v0.5.2/fla/ops/generalized_delta_rule/dplr>.

This notice applies only to the identified third-party-derived portions. No
repository-wide license has been selected for the remaining SolveDelta code.
