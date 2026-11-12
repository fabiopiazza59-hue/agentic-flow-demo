# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (33 sessions, 989 name-forecasts; mean daily APE gain -0.003%, CS [-0.015%, +0.010%]).

**Pending (2026-11-12):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 5 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.003% | [-0.015%, +0.010%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.000% | [-0.010%, +0.009%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.041 | [-0.288, +0.207] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0012 | [-0.003, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.079% | [-0.067%, +0.226%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.082% | [-0.059%, +0.223%] | ≈ not distinguishable |

MAPE — LLM 1.59%, free baseline 1.59%, prior close 1.67%; LLM 80% interval coverage 79.38%; names moved per session 1.0. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 28 | -0.010% | [-0.043%, +0.023%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 27 | -0.003% | [-0.037%, +0.030%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 33 | 1.22% | 1.21% | 1 |
| ADBE | 33 | 1.82% | 1.82% | 0 |
| AMD | 33 | 3.65% | 3.66% | 2 |
| AMZN | 33 | 1.84% | 1.84% | 0 |
| AVGO | 33 | 2.39% | 2.38% | 4 |
| BAC | 33 | 1.06% | 1.07% | 1 |
| CAT | 33 | 2.38% | 2.36% | 1 |
| COST | 33 | 0.96% | 0.96% | 1 |
| CRM | 33 | 1.55% | 1.55% | 0 |
| CVX | 33 | 1.20% | 1.21% | 1 |
| DIS | 33 | 1.05% | 1.05% | 1 |
| GOOGL | 33 | 1.25% | 1.26% | 1 |
| HD | 33 | 1.48% | 1.49% | 1 |
| JNJ | 33 | 1.53% | 1.52% | 1 |
| JPM | 33 | 1.02% | 1.02% | 1 |
| KO | 33 | 0.89% | 0.87% | 1 |
| LLY | 33 | 1.48% | 1.50% | 3 |
| MA | 33 | 1.21% | 1.22% | 1 |
| META | 32 | 1.98% | 1.98% | 0 |
| MSFT | 33 | 1.51% | 1.50% | 2 |
| NFLX | 33 | 1.56% | 1.56% | 0 |
| NVDA | 33 | 1.99% | 1.96% | 3 |
| ORCL | 33 | 3.38% | 3.33% | 2 |
| PEP | 33 | 1.15% | 1.15% | 0 |
| PG | 33 | 1.18% | 1.19% | 1 |
| TSLA | 33 | 2.01% | 2.01% | 0 |
| UNH | 33 | 1.44% | 1.43% | 1 |
| V | 33 | 1.14% | 1.14% | 0 |
| WMT | 33 | 1.24% | 1.24% | 1 |
| XOM | 33 | 1.26% | 1.27% | 1 |

_Research experiment, not financial advice._
