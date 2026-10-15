# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (14 sessions, 419 name-forecasts; mean daily APE gain -0.004%, CS [-0.018%, +0.010%]).

**Pending (2026-10-16):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.004% | [-0.018%, +0.010%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.002% | [-0.015%, +0.011%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.112 | n=6 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | -0.0011 | [-0.006, +0.004] | ≈ not distinguishable |
| APE gain: LLM vs prior close | -0.039% | [-0.120%, +0.043%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | -0.035% | [-0.114%, +0.045%] | ≈ not distinguishable |

MAPE — LLM 1.36%, free baseline 1.36%, prior close 1.33%; LLM 80% interval coverage 76.63%; names moved per session 1.1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 9 | +0.003% | — | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 8 | +0.001% | — | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 14 | 0.94% | 0.94% | 0 |
| ADBE | 14 | 1.73% | 1.73% | 0 |
| AMD | 14 | 2.14% | 2.14% | 0 |
| AMZN | 14 | 1.22% | 1.22% | 0 |
| AVGO | 14 | 1.97% | 1.98% | 3 |
| BAC | 14 | 0.96% | 0.99% | 1 |
| CAT | 14 | 1.80% | 1.80% | 0 |
| COST | 14 | 0.82% | 0.81% | 1 |
| CRM | 14 | 1.80% | 1.80% | 0 |
| CVX | 14 | 1.09% | 1.09% | 0 |
| DIS | 14 | 0.95% | 0.96% | 1 |
| GOOGL | 14 | 0.96% | 0.96% | 0 |
| HD | 14 | 1.28% | 1.30% | 1 |
| JNJ | 14 | 1.66% | 1.64% | 1 |
| JPM | 14 | 0.70% | 0.67% | 1 |
| KO | 14 | 0.82% | 0.82% | 0 |
| LLY | 14 | 1.37% | 1.37% | 0 |
| MA | 14 | 1.17% | 1.17% | 0 |
| META | 13 | 1.80% | 1.80% | 0 |
| MSFT | 14 | 0.99% | 0.97% | 1 |
| NFLX | 14 | 1.54% | 1.54% | 0 |
| NVDA | 14 | 1.34% | 1.31% | 2 |
| ORCL | 14 | 3.39% | 3.33% | 1 |
| PEP | 14 | 0.98% | 0.98% | 0 |
| PG | 14 | 1.12% | 1.12% | 0 |
| TSLA | 14 | 1.42% | 1.42% | 0 |
| UNH | 14 | 1.49% | 1.46% | 1 |
| V | 14 | 1.08% | 1.08% | 0 |
| WMT | 14 | 1.41% | 1.42% | 1 |
| XOM | 14 | 1.03% | 1.03% | 0 |

_Research experiment, not financial advice._
