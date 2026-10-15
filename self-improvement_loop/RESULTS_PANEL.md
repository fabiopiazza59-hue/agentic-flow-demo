# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (13 sessions, 389 name-forecasts; mean daily APE gain -0.004%, CS [-0.019%, +0.012%]).

**Pending (2026-10-15):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 2 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.004% | [-0.019%, +0.012%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.002% | [-0.017%, +0.012%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.109 | n=5 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | -0.0009 | [-0.006, +0.004] | ≈ not distinguishable |
| APE gain: LLM vs prior close | -0.026% | [-0.101%, +0.049%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | -0.023% | [-0.094%, +0.049%] | ≈ not distinguishable |

MAPE — LLM 1.33%, free baseline 1.33%, prior close 1.31%; LLM 80% interval coverage 76.37%; names moved per session 1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 8 | +0.002% | — | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 7 | +0.005% | — | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 13 | 0.95% | 0.95% | 0 |
| ADBE | 13 | 1.48% | 1.48% | 0 |
| AMD | 13 | 2.21% | 2.21% | 0 |
| AMZN | 13 | 1.18% | 1.18% | 0 |
| AVGO | 13 | 2.11% | 2.07% | 2 |
| BAC | 13 | 1.01% | 1.04% | 1 |
| CAT | 13 | 1.89% | 1.89% | 0 |
| COST | 13 | 0.85% | 0.84% | 1 |
| CRM | 13 | 1.54% | 1.54% | 0 |
| CVX | 13 | 1.01% | 1.01% | 0 |
| DIS | 13 | 0.91% | 0.92% | 1 |
| GOOGL | 13 | 0.93% | 0.93% | 0 |
| HD | 13 | 1.06% | 1.07% | 1 |
| JNJ | 13 | 1.71% | 1.68% | 1 |
| JPM | 13 | 0.74% | 0.72% | 1 |
| KO | 13 | 0.87% | 0.87% | 0 |
| LLY | 13 | 1.24% | 1.24% | 0 |
| MA | 13 | 1.10% | 1.10% | 0 |
| META | 12 | 1.88% | 1.88% | 0 |
| MSFT | 13 | 0.94% | 0.93% | 1 |
| NFLX | 13 | 1.44% | 1.44% | 0 |
| NVDA | 13 | 1.36% | 1.39% | 1 |
| ORCL | 13 | 3.25% | 3.18% | 1 |
| PEP | 13 | 0.97% | 0.97% | 0 |
| PG | 13 | 1.19% | 1.19% | 0 |
| TSLA | 13 | 1.50% | 1.50% | 0 |
| UNH | 13 | 1.39% | 1.37% | 1 |
| V | 13 | 1.01% | 1.01% | 0 |
| WMT | 13 | 1.45% | 1.46% | 1 |
| XOM | 13 | 0.95% | 0.95% | 0 |

_Research experiment, not financial advice._
