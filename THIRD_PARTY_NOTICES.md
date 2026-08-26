# Third-Party Notices

## Flash Linear Attention and MESA kernels

The following files contain adapted and modified portions of Flash Linear
Attention commit `bc3b101dcb713ddc5bd8924b66754eb68b5ccf89`:

- `causallsso/ops/direct_e.py`: generalized Delta DPLR pair, WY, state, output,
  transpose, and variable-length chunk schedules specialized to SolveDelta's
  direct-`e` interface;
- `causallsso/ops/normalization.py`: L2 normalization forward and transpose
  reduction schedule specialized to the producer/consumer layouts;
- `causallsso/ops/gates.py`: GDN decay forward and transpose tiles specialized
  to strided projection views and a shared FP32 parameter-gradient reduction;
- `causallsso/ops/geometry_scan.py`: MESA paired resident `Hkk/Hkv` state loop,
  transition transpose-dot schedule, and FLA reverse-cumsum primitive;
- `causallsso/ops/radial.py`: MESA Gram/Hadamard forward and transpose
  organization specialized to SolveDelta's three strict routes;
- `causallsso/ops/resident_frame.py`: FLA GDN2/KDA blocked substitution and
  ordered 16-coordinate diagonal pattern, combined with MESA's two-dot action
  identity and specialized to generated J/D/u/h pair tiles.
- `causallsso/ops/csrc/local_transpose.cu`: FLA DPLR's 256-thread,
  phase-staged output ownership specialized to the exact coordinate-axis
  generalized-Delta transpose and its H/R generator epilogue.

> MIT License
>
> Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

The adapted portions are provided under the MIT License; see
`LICENSES/MIT.txt`. The principal reviewed upstream files are:

- `fla/ops/generalized_delta_rule/dplr/{chunk_A_fwd,chunk_A_bwd,wy_fast_fwd,wy_fast_bwd,chunk_h_fwd,chunk_h_bwd,chunk_o_fwd,chunk_o_bwd}.py`;
- `fla/modules/l2norm.py`;
- `fla/ops/mesa_net/{chunk_h_fwd,chunk_h_kk_intra_bwd,chunk_h_kv_intra_bwd}.py`;
- `fla/ops/{gdn2,kda}/chunk_intra.py` and `fla/ops/utils/solve_tril.py`;
- `fla/ops/mesa_net/{chunk_cg_solver_fwd,chunk_cg_solver_bwd}.py`;
- `fla/ops/gated_delta_rule/gate.py`;
- `fla/layers/utils.py` and `fla/ops/utils/{cumsum,index}.py` variable-length
  gather/scatter, chunk indices, offsets, and cumulative schedules.

Upstream source:
<https://github.com/fla-org/flash-linear-attention/tree/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89>.

The staged local-transpose schedule additionally follows
`fla/ops/generalized_delta_rule/dplr/backends/tilelang/chunk_stream_bwd.py`
and `schedules.py` at commit
`38a496e1ce58baaf1bc6613176eb2f433d0ddb90`:
<https://github.com/fla-org/flash-linear-attention/tree/38a496e1ce58baaf1bc6613176eb2f433d0ddb90>.

SolveDelta changes pointer layouts, operand dtypes, masks, route composition,
and private interfaces. It does not import the upstream model ABI. This notice
applies only to the identified third-party-derived portions. No
repository-wide license has been selected for the remaining SolveDelta code.

## Runtime dependencies

SolveDelta uses Flash Linear Attention under the MIT License and
`causal-conv1d` under the BSD 3-Clause License as external Python/CUDA runtime
dependencies. Their source is not vendored in this repository.
