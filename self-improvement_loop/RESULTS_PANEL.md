# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (25 sessions, 749 name-forecasts; mean daily APE gain -0.003%, CS [-0.017%, +0.011%]).

**Pending (2026-11-02):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.003% | [-0.017%, +0.011%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.001% | [-0.011%, +0.009%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.048 | [-0.346, +0.250] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0015 | [-0.004, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.085% | [-0.099%, +0.268%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.088% | [-0.090%, +0.266%] | ≈ not distinguishable |

MAPE — LLM 1.40%, free baseline 1.39%, prior close 1.48%; LLM 80% interval coverage 78.65%; names moved per session 1.2. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 20 | -0.012% | [-0.053%, +0.030%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 19 | -0.007% | [-0.053%, +0.040%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 25 | 0.98% | 0.98% | 0 |
| ADBE | 25 | 2.02% | 2.02% | 0 |
| AMD | 25 | 2.43% | 2.43% | 0 |
| AMZN | 25 | 1.67% | 1.67% | 0 |
| AVGO | 25 | 1.93% | 1.91% | 4 |
| BAC | 25 | 0.89% | 0.91% | 1 |
| CAT | 25 | 2.10% | 2.07% | 1 |
| COST | 25 | 0.83% | 0.82% | 1 |
| CRM | 25 | 1.54% | 1.54% | 0 |
| CVX | 25 | 1.02% | 1.04% | 1 |
| DIS | 25 | 0.92% | 0.93% | 1 |
| GOOGL | 25 | 1.18% | 1.20% | 1 |
| HD | 25 | 1.23% | 1.24% | 1 |
| JNJ | 25 | 1.51% | 1.49% | 1 |
| JPM | 25 | 0.82% | 0.80% | 1 |
| KO | 25 | 0.90% | 0.88% | 1 |
| LLY | 25 | 1.49% | 1.52% | 3 |
| MA | 25 | 1.09% | 1.10% | 1 |
| META | 24 | 1.93% | 1.93% | 0 |
| MSFT | 25 | 1.05% | 1.03% | 2 |
| NFLX | 25 | 1.50% | 1.50% | 0 |
| NVDA | 25 | 1.35% | 1.31% | 3 |
| ORCL | 25 | 3.08% | 3.02% | 2 |
| PEP | 25 | 1.07% | 1.07% | 0 |
| PG | 25 | 1.19% | 1.21% | 1 |
| TSLA | 25 | 1.60% | 1.60% | 0 |
| UNH | 25 | 1.38% | 1.37% | 1 |
| V | 25 | 1.01% | 1.01% | 0 |
| WMT | 25 | 1.13% | 1.13% | 1 |
| XOM | 25 | 1.05% | 1.07% | 1 |

_Research experiment, not financial advice._
