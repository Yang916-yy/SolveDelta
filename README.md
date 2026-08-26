# SolveDelta

> **Status:** unfinished research repository. The FP64 operator is frozen; the
> replacement BF16/FP16/FP32 production path is connected end to end and its
> minimal semantic/composed-VJP acceptance suite is rebuilt. The final hardware
> resource report and further backward optimization remain incomplete.

SolveDelta is one causal sequence operator. A decayed prefix constructs a
bounded linear system; its primal action conditions ordered Delta writes and
its transpose-dual action conditions erase and read covectors. DeltaNet,
Gated DeltaNet, GDN2, KDA, and DeltaProduct are reductions or comparison
baselines rather than maintained alternative architectures.

## Contract

- `causallsso/reference.py` is the sole executable mathematical owner and FP64
  oracle.
- Geometry width is the resolved key-head width, `r = d_k`; `r=128` is the
  first benchmark specialization, not a mathematical default.
- `K` ordered edits share one token frame. `K=1` is the default.
- The frontend applies independent depthwise causal conv4 plus SiLU to query,
  packed edit keys, and packed edit values by default.
- The symmetric occupancy state `J`, unconstrained driven state `D`, scalar
  mass `m`, and associative memory `S` remain FP32 continuation states.
- Public activations and outputs are BF16. Statically bounded private panels
  may be produced directly as FP16. Tensor Core contractions and backward
  partials accumulate in FP32.

For normalized geometry feature `u_t`, the geometry recurrence is

\[
\begin{aligned}
m_t &= \lambda_t m_{t-1}+1,\\
J_t &= \lambda_t J_{t-1}+u_tu_t^T,\\
D_t &= \lambda_t D_{t-1}+u_th_t^T.
\end{aligned}
\]

The separately bounded maps of `H_t=J_t/m_t` and `R_t=D_t/m_t` define

\[
M_t=(I+N_t^-)\operatorname{diag}(\sigma_t)(I+N_t^+).
\]

With normalized edit key `a`, erase source `b=erase\odot a`, query `q`, and
write target `z=write\odot v`,

\[
d=M^{-1}a,\qquad e=M^Tb,\qquad \chi=M^Tq.
\]

The fixed-basis associative state performs the ordered rank-one Delta edits and
is read after edit `K`.

## Production graph

The current all-valid CUDA path is a selective composition:

1. fused normalization writes each geometry/frame consumer layout directly
   and consumes raw erase logits in its dual-source epilogue;
2. a MESA-derived paired resident loop produces FP32 `J/D` boundaries while a
   scalar affine owner produces `m`;
3. MESA-style Gram/radial blocks and exact coordinate-axis generalized-Delta
   solve/direct actions produce primal and paired-dual frame panels;
4. a direct-`e` specialization activates raw write logits while packing `z`
   and feeds FLA's mature pair/WY/state/output kernels;
5. backward follows the corresponding transpose blocks and accumulates the
   four factor routes into output-owned FP32 tiles, without descriptor bundles,
   coordinate-entry VJP chains, or four-way full-matrix partials.

The frame-to-WY panel boundary is private and deliberately split: combining
owners is allowed only when complete forward and F+B measurements beat the
mature multi-kernel schedule. Masks and resets use one compact valid-token
buffer plus FLA `cu_seqlens`, chunk indices, and chunk offsets across frame,
pair/WY, state, output, and their transposes. No segment is padded to the
longest segment; one gather and one scatter remain at the model boundary.

On the local RTX 5070 Ti target at
`B=1,T=1024,H=8,K=1,r=d_v=128,C=32`, the exact operator without returned states
measures about `1.77 ms` forward and `8.78 ms` forward plus backward. The
three latest matched complete BF16 layer runs, including conv4 and projections,
measured about `1.78/9.39--9.54 ms` versus FLA GDN2's roughly
`0.97--1.03/3.02--3.06 ms`; allocator peaks were `106/211 MiB` versus
`38/103 MiB`. These are development measurements. The
remaining dominant hotspot is the local strict transpose, and the formal
register/shared/spill/occupancy report is still outstanding.

## Install

The FP64 reference requires PyTorch. The CUDA production path additionally
uses Flash Linear Attention and causal-conv1d:

```bash
python -m pip install -e ".[native,test]"
```

Reference example:

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

The native path is selected for CUDA BF16 inputs. It requires continuation
states in FP32 and derives `r` from the configured key-head width.

## Repository map

- `AGENTS.md`: contribution, mathematical, precision, and acceptance contract;
- `causallsso/reference.py`: FP64 token oracle;
- `docs/FROM_SCRATCH_REBUILD.md`: sole native implementation blueprint;
- `docs/INNOVATION_PROGRAM.md`: operator derivation and reductions;
- `docs/PRIOR_ART.md`: source provenance and design decisions;
- `THIRD_PARTY_NOTICES.md`: adapted-source attribution.
