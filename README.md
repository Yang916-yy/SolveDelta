# SolveDelta

SolveDelta is a causal sequence operator that uses a decayed RLS geometry state
to transport a gated Delta memory before each edit. The current implementation
selects a lower-expressivity moving-state formulation because it reaches usable
training latency. The previous exact bounded-LDU implementation is retained in
Git history at commit `2237875`, not as a runtime path.

## Recurrence

For normalized geometry feature `u_t`:

```text
m_t = lambda_t m_{t-1} + 1
J_t = lambda_t J_{t-1} + u_t u_t^T
D_t = lambda_t D_{t-1} + u_t h_t^T
```

The fixed prior is `m_0=2`, `J_0=2I`, `D_0=0`. Define

```text
g_t = J_t^-1 u_t
p_t = J_{t-1}^-1 u_t
C_{t-1} = J_{t-1}^-1 D_{t-1}
r_t = h_t - C_{t-1}^T u_t
rho_t = m_{t-1}/m_t
```

and two rank-one geometry transports

```text
F_H = rho_t (lambda_t I + u_t p_t^T)
F_C = I + g_t r_t^T.
```

Learned strength `gamma` interpolates each factor with identity. The memory is
transported by `F_H`, channel-decayed, transported by `F_C`, updated by one
ordinary gated Delta edit, and then read by the normalized query. At
`gamma=0`, the memory path reduces exactly to GDN2 while geometry state still
tracks the prefix.

`causallsso/reference.py` is the sole FP64 mathematical oracle. The full
formula, implementation mapping, transpose ownership, and precision map are in
`docs/FROM_SCRATCH_REBUILD.md`.

## Production path

The dense CUDA path composes mature FLA/MESA primitives with a token-native
block-E3 exterior:

- MESA paired covariance/cross-moment state scans for FP32 `J/D`;
- fixed five-step matrix-free CG gain and implicit transpose;
- FP32 effective-mass affine scan;
- BF16 Tensor Core C16 pair/WY/state/output owners with FP32 accumulation;
- output-owned reverse and a fused source transpose.

The three internal slots are the two geometry transports plus one Delta edit.
They remain a local slot axis and are not expanded into a public `3T` sequence.
The production surface has `K=1`, BF16 public vectors/output, and FP32
continuation states `(m,J,D,S)`.

The selected contiguous core at `B=1,T=1024,H=8,r=V=128` measured median/p95
`0.371/0.402 ms` forward and `1.103/1.320 ms` F+B under CUDA Graph on the
development RTX 5070 Ti, with `86.3 MiB` resident Graph allocation. The full
projected layer including conv4, packed-view canonicalization, output
projection, and all parameter gradients measured `0.649/0.854 ms` forward and
`2.329/2.624 ms` F+B, with `152.0 MiB` Graph allocation. Matched GDN2 core was
about `0.128/0.467 ms`; the remaining latency and memory gap is explicit.

## Install

The reference requires PyTorch. The native path additionally requires FLA,
TileLang, and causal-conv1d:

```bash
python -m pip install -e ".[native,test]"
```

```python
import torch

from causallsso import SolveDelta, SolveDeltaConfig

config = SolveDeltaConfig(
    hidden_size=256,
    num_heads=4,
    head_k_dim=64,
    head_v_dim=64,
)
layer = SolveDelta(config).cuda().to(torch.bfloat16)
hidden = torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16)
output = layer(hidden)
```

CUDA BF16 dense inputs select the optimized path. Masks/resets use the same RLS
semantics through the model reference path. Non-CUDA and non-BF16 inputs use
the PyTorch recurrence.

## Repository map

- `AGENTS.md`: current mathematical, precision, and acceptance contract;
- `causallsso/reference.py`: FP64 token oracle;
- `causallsso/ops/rls/`: selected native forward and transpose blocks;
- `docs/FROM_SCRATCH_REBUILD.md`: sole native implementation blueprint;
- `docs/INNOVATION_PROGRAM.md`: derivation and model interpretation;
- `docs/PRIOR_ART.md`: upstream provenance and measured design decisions;
- `THIRD_PARTY_NOTICES.md`: adapted-source attribution.
