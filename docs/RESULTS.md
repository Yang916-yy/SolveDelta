# Evaluation and Performance

The current measurement covers the accumulated-primal operator defined by the
oracle. Older mixer and training results are retained below with their model
contract stated explicitly.

## Current right-coordinate forgetting path

Environment: RTX 5070 Ti, Python 3.12, PyTorch 2.13.0+cu130, Triton 3.7.1,
FLA 0.6.0, BF16 public operands and FP32 `(C,S)`. Shape:
`B=1,T=1024,H=8,r=V=128`. CUDA Graph timings report median/p95 after warmup:
800 samples for the core and 400 for each model surface. Final continuation
output is disabled.

| Scope | Forward | F+B | Graph allocated |
| --- | ---: | ---: | ---: |
| Core operator | 0.262 / 0.279 ms | 0.908 / 1.068 ms | 63.0 MiB |
| Projected mixer | 0.514 / 0.682 ms | 1.753 / 1.954 ms | 208.1 MiB |
| Full model block | 0.754 / 0.781 ms | 2.822 / 3.002 ms | 278.1 MiB |

The core row uses random FP32 initial `(C,S)` and omits final continuation
output. The projected mixer and block include their trainable projections; the
block also includes prenorm and MLP. The coordinate-gated pair retains FLA's
centered Tensor Core cross-subchunk schedule, while diagonal subchunks and the
strict transpose evaluate only nonpositive exclusive-prefix exponents.

The preceding scalar-retention contract measured `0.246 ms` forward and
`0.796 ms` F+B at the core, `1.592 ms` F+B at the projected mixer, and
`2.709 ms` F+B at the block. The right-coordinate contract measures `0.262`,
`0.908`, `1.753`, and `2.822 ms` respectively. Live allocation is unchanged at
each scope. These are different model semantics, so the comparison quantifies
the price of retaining the complete DeltaRule forgetting field rather than an
implementation-only regression.

A matched private ablation under the preceding scalar-retention contract
replaced the accumulated action `C^T k` by `C k`
using transposed reads of the same chunk-boundary state and a strict FLA-style
reverse. Across interleaved Graph measurements, `C^T k` took `0.831-0.834 ms`
F+B while `C k` took `0.852-0.856 ms`; forward was tied at `0.232-0.234 ms`
and allocation was identical. The `C k` owner was removed after the ablation,
and no long training comparison was started because it did not pass the
latency prerequisite.

## Archived residual-local comparison

The remaining operator/mixer comparison was collected for commit `93735fad`,
whose primal used the token-local residual frame. It remains useful historical
evidence for the shared predictor, dual, exterior, and model shell, but it is
not a current accumulated-primal comparison.

For that archived execution graph:

- at the projected-mixer boundary, SolveDelta adds about `18%` forward and
  `12%` F+B latency over same-shape FLA GDN2;
- paired 50M-token runs selected `gamma=0.90` with plain gated RMSNorm;
- an earlier 200M-token run was competitive on average NLL and produced a
  positive paired LAMBADA NLL signal.

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

## Independent source/target versus shared geometry A/B

A paired 300M-token run tested whether the independent geometry projections
`u=W_u x` and `h=W_h x` could be replaced by one projection `g=W_g x`, used
as normalized source `u=normalize(g)` and unnormalized target `h=g`. The
candidate is the variance-matched folded form of `(u_raw+h_raw)/sqrt(2)` and
removes one complete `d_model x H*r` projection. The operator, accumulated
`C^T k` primal, dual, memory recurrence, optimizer, token order, and all other
initial weights were matched.

```text
seed                 20260831
shape                B=48, T=512, hidden=768, H=6, r=128, layers=7
tokens/model         300,023,808
parameters           independent 101,881,556; shared 97,752,788
optimizer            AdamW, betas=(0.9,0.95), weight decay=0.1
learning rate        3e-4 peak, 4% warmup, cosine decay to zero
precision            FP32 parameters, BF16 autocast
```

The shared candidate learned faster early and retained a small final
FineWeb-Edu validation advantage: `3.44060` independent versus `3.43228`
shared. Frozen paired evaluation gave a mixed result:

| Corpus | Independent NLL | Shared NLL | Shared advantage | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| FineWeb-Edu | 3.61293 | 3.60552 | 0.00742 | [0.00303, 0.01180] |
| WikiText-103 | 4.46044 | 4.47461 | -0.01416 | [-0.02379, -0.00485] |
| PG19 | 4.26036 | 4.24425 | 0.01611 | [0.00977, 0.02245] |
| LAMBADA item | 5.72098 | 5.70820 | 0.01278 | [-0.00619, 0.03172] |

The shared parameterization did not pass the structure-sensitive gate. Across
lengths 128, 256, and 512, its target-versus-best-wrong margins were lower by
about `0.81` on recall, `0.73` on overwrite, `0.49` on disambiguation, and
`0.21` on instruction override. Overwrite accuracy at lengths 256 and 512
fell from `20.3/21.9%` to `9.4/10.9%`; both paired intervals excluded zero.
These low absolute accuracies come from an exploratory three-token-per-
parameter run, but the consistent paired margin loss directly localizes the
missing arbitrary source-to-target direction.

At `B=1,T=1024,H=8,r=V=128` on the RTX 5070 Ti, the shared candidate reduced
projected-mixer CUDA Graph forward from `0.4873` to `0.4666 ms` and F+B from
`1.6598` to `1.5857 ms`, while stable Graph allocation fell by `10 MiB`.
That engineering gain does not compensate for the structure-sensitive
regression. Independent `u/h` remains selected; the shared form is not a
maintained model variant.

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
