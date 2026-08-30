# Evaluation and Performance

This page records scoped development evidence for the current Residual-Frame
operator. It is not a leaderboard claim. Benchmarks measure one implementation
on one device; the language-model comparison uses one exploratory seed.

## CUDA Graph operator and mixer

Environment: RTX 5070 Ti, Python 3.12, PyTorch 2.13.0+cu130, Triton 3.7.1,
FLA 0.6.0, BF16 public operands/autocast, FP32 master parameters and recurrent
state. Shape: `B=1,T=1024,H=8,r=V=128`. Timings are milliseconds and report
median/p95 after warmup. Final continuation output is disabled for both core
paths.

| Scope | Path | Forward | F+B | Graph allocated |
| --- | --- | ---: | ---: | ---: |
| Core operator | SolveDelta | 0.175 / 0.182 | 0.635 / 0.645 | 61.1 MiB |
| Core operator | FLA GDN2 | 0.108 / 0.124 | 0.464 / 0.476 | 46.0 MiB |
| Projected mixer | SolveDelta | 0.425 / 0.434 | 1.493 / 1.861 | 247.7 MiB |
| Projected mixer | FLA GDN2 | 0.366 / 0.380 | 1.272 / 1.450 | 233.2 MiB |

The core comparison exposes all predictor and relative-source work, so its
percentage gap is larger. Common projection, conv4, normalization, gating, and
output work reduce the projected-mixer gap to about `16%` forward and `17%`
F+B. The mixer boundary excludes the MLP, LM head, optimizer, gradient clipping,
and distributed communication.

Core is a cost decomposition rather than identical semantics: SolveDelta owns
the predictor and initial `(C,S)` cotangents, while GDN2 has no predictor. The
projected-mixer rows are the primary hidden-to-hidden comparison.

Graph reserved memory is intentionally omitted: independent capture pools and
allocator buckets changed it across otherwise identical runs. Allocated bytes
were stable. The projected mixers contain `8.94M` SolveDelta parameters and
`6.83M` GDN2 parameters at this width.

## Exploratory 200M-token comparison

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
with a paired SolveDelta advantage of `0.03471` and 95% confidence interval
`[0.01722,0.05249]`. Accuracy differences included zero. A 4096-token
continuation evaluation was not uniformly better: WikiText-103 favored GDN2,
and both models had trained only at length 512.

The responsible conclusion is narrow: this run establishes competitive small-
scale language modeling and one positive LAMBADA NLL signal. It does not
establish universal quality, long-context superiority, scaling behavior, or
multi-seed significance. Raw datasets, checkpoints, and benchmark artifacts
remain outside Git as required by `AGENTS.md`.

## Reproducing reports

Every future report must state:

- exact shape, dtype, device, package versions, and execution mode;
- whether the scope is core operator, projected mixer, model block, or complete
  causal LM;
- forward, backward, and F+B median/p95 where separately available;
- allocator peak or stable Graph allocation;
- the same oracle and composed-VJP gates for every compared candidate.

CUDA Graph and eager measurements are different execution modes and must not be
mixed in one speedup ratio.
