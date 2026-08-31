# Evaluation and Performance

This page answers three practical questions about the current Residual-Frame
operator: what the geometry solver costs, which implementation choices earned
their place, and what the first language-model runs suggest.

The short version:

- at the projected-mixer boundary, SolveDelta adds about `18%` forward and
  `12%` F+B latency over same-shape FLA GDN2;
- paired 50M-token runs selected `gamma=0.90` with plain gated RMSNorm;
- an earlier 200M-token run was competitive on average NLL and produced a
  positive paired LAMBADA NLL signal.

## CUDA Graph operator and mixer

Environment: RTX 5070 Ti, Python 3.12, PyTorch 2.13.0+cu130, Triton 3.7.1,
FLA 0.6.0, BF16 public operands/autocast, FP32 master parameters and recurrent
state. Shape: `B=1,T=1024,H=8,r=V=128`. Timings are milliseconds and report
median/p95 after warmup. Final continuation output is disabled for both core
paths.

| Scope | Path | Forward | F+B | Graph allocated |
| --- | --- | ---: | ---: | ---: |
| Core operator | SolveDelta | 0.173 / 0.178 | 0.610 / 0.785 | 58.1 MiB |
| Core operator | FLA GDN2 | 0.107 / 0.112 | 0.467 / 0.643 | 46.0 MiB |
| Projected mixer | SolveDelta | 0.428 / 0.614 | 1.437 / 1.632 | 245.8 MiB |
| Projected mixer | FLA GDN2 | 0.364 / 0.534 | 1.276 / 1.468 | 233.2 MiB |

The core rows isolate SolveDelta's predictor, relative-source actions, and
initial `(C,S)` cotangents. The projected mixer adds the common projection,
conv4, normalization, gates, and output projection; it is the primary
hidden-to-hidden comparison. At that boundary the gap is about `18%` forward
and `12%` F+B. MLP, LM head, optimizer, clipping, and distributed communication
sit outside the measurement.

Graph allocated bytes were stable and are reported above. Reserved bytes vary
with capture pools and allocator buckets, so they are omitted. The projected
mixers contain `8.99M` SolveDelta parameters and `6.83M` GDN2 parameters at
this width.

The synchronized p95 samples include a roughly `0.17 ms` device P-state tail
shared by both paths. The table reports the observed tail; medians and minima
were stable.

## Projection and accumulation A/B

At the same projected-mixer shape, padding the fused input projection's
physical row from `7432` to `7488` reduced Graph F+B from `1.477` to
`1.442 ms` in an interleaved A/B. Output bits and all logical parameter
gradients were identical; the upstream hidden gradient changed by `0.373%`
relative from the low-precision GEMM reduction order. The 56 unused FP32 rows
cost `0.219 MiB` per layer before optimizer state.

Grouping the two `[1024,128] -> [1024,1024]` decay and output-gate projections
measured `210.0 us` F+B versus `172.9 us` for two independent `F.linear`
calls. Strided-input/weight packing and a separate bias epilogue outweighed the
saved launch, so the independent GEMMs remain selected.

For a one-layer `D=1024,H=8,T=1024,vocab=4096` CausalLM Graph, an
optimizer-bound BF16 Linear shadow produced the following complete
microbatch-accumulation step times. Each shadow row includes one refresh:

| Accumulation | FP32 master autocast | BF16 shadow | Change |
| ---: | ---: | ---: | ---: |
| 1 | 2.939 ms | 2.971 ms | +1.1% |
| 2 | 6.199 ms | 6.100 ms | -1.6% |
| 4 | 13.048 ms | 12.631 ms | -3.2% |
| 8 | 26.668 ms | 25.574 ms | -4.1% |

Loss was bitwise identical and the maximum parameter-gradient relative error
was `7.4e-8`. The shadow added `41.63 MiB` of buffers, about `43.63 MiB` after
capture, and `27.5 MiB` to measured capture peak. It is useful when two or more
microbatches amortize each refresh; the ordinary FP32-master path remains the
single-microbatch default.

## Geometry initialization and readout A/B

One paired 50M-token run selected the current high-relaxation/plain-readout
defaults. Both three-layer models started from identical weights and saw the
same FineWeb-Edu batches under seed `20260830`; the candidate changed the
geometry bias from `-2` (`gamma about 0.119`) to `logit(0.9)` and removed the
bounded output-radius modulation. Shape was `B=8,T=512,hidden=512,H=4,r=128`.

Final fixed validation NLL was `3.91459` for the selected candidate and
`3.92034` for the former defaults. On 256 held-out windows per corpus, grouped
into 32 paired samples with 50,000 bootstrap draws:

| Corpus | Former NLL | Selected NLL | Selected advantage | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| FineWeb-Edu | 4.32489 | 4.32245 | 0.00244 | [-0.00214, 0.00677] |
| WikiText-103 | 5.67065 | 5.65833 | 0.01233 | [0.00427, 0.02087] |
| PG19 | 5.15784 | 5.14661 | 0.01123 | [0.00244, 0.01904] |

A separate matched high-gamma run isolated the readout. Plain gated RMSNorm
improved NLL by `0.01624` on WikiText-103 and `0.01672` on PG19, with both
confidence intervals above zero. Together these small-scale runs support the
current high-relaxation, plain-readout default.

A follow-up paired 50M-token run compared plain gated RMSNorm at `gamma=0.90`
and `gamma=0.95`. The fixed validation slice favored `0.90` by `0.00056` NLL.
The effect was corpus-dependent: `0.95` improved FineWeb-Edu by
`0.00305` (95% CI `[0.00006,0.00623]`) and WikiText-103 by `0.03674`
(`[0.03040,0.04333]`), but regressed PG19 by `0.01550`
(`[-0.01892,-0.01196]`). The reversal on PG19 and the fixed-validation result
favor `0.90` as the general default.

## Exploratory 200M-token comparison

This run predates the high-relaxation initialization and plain gated RMSNorm
readout. It measures an earlier Residual-Frame configuration and is retained
as the longest paired training evidence available.

One paired run trained three-layer causal LMs from fresh initialization on
FineWeb-Edu `sample-10BT`:

```text
seed                 20260828
shape                B=8, T=512, hidden=512, H=4, r=128, layers=3
tokens/model         200,003,584
optimizer            AdamW, betas=(0.9,0.95), weight decay=0.1
learning rate        3e-4 peak, 4% warmup, cosine decay to zero
precision            FP32 parameters, BF16 autocast
parameters           SolveDelta 43,422,628; GDN2 41,837,452
```

Final fixed validation NLL was `3.30745` for SolveDelta and `3.31320` for
GDN2, a `0.00575` SolveDelta advantage. Independent FineWeb-Edu, WikiText-103,
and PG19 estimates were statistical ties.

On 5,153 paired LAMBADA test items, item NLL was `6.73632` versus `6.77104`,
a SolveDelta advantage of `0.03471` with 95% confidence interval
`[0.01722,0.05249]`. Accuracy was tied within uncertainty. In a 4096-token
continuation evaluation WikiText-103 favored GDN2; both models had trained at
length 512.

The evidence supports a narrow conclusion: competitive small-scale language
modeling and one positive LAMBADA NLL signal. Scaling, long-context behavior,
and multi-seed significance remain open. Raw datasets, checkpoints, and
benchmark artifacts live outside Git as required by `AGENTS.md`.

## Reporting a comparable result

A comparable report states:

- exact shape, dtype, device, package versions, and execution mode;
- whether the scope is core operator, projected mixer, model block, or complete
  causal LM;
- forward, backward, and F+B median/p95 where separately available;
- allocator peak or stable Graph allocation;
- the same oracle and composed-VJP gates for every compared candidate.

CUDA Graph and eager measurements use separate comparison rows.
