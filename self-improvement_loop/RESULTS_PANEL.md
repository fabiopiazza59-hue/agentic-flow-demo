# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (21 sessions, 629 name-forecasts; mean daily APE gain -0.006%, CS [-0.020%, +0.008%]).

**Pending (2026-10-27):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 1 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.006% | [-0.020%, +0.008%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.003% | [-0.013%, +0.007%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.108 | n=9 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | -0.0015 | [-0.005, +0.002] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.011% | [-0.096%, +0.119%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.018% | [-0.087%, +0.122%] | ≈ not distinguishable |

MAPE — LLM 1.39%, free baseline 1.39%, prior close 1.41%; LLM 80% interval coverage 78.23%; names moved per session 1.0. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 16 | -0.001% | [-0.040%, +0.039%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 15 | +0.002% | [-0.035%, +0.038%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 21 | 0.93% | 0.93% | 0 |
| ADBE | 21 | 2.01% | 2.01% | 0 |
| AMD | 21 | 2.34% | 2.34% | 0 |
| AMZN | 21 | 1.42% | 1.42% | 0 |
| AVGO | 21 | 1.91% | 1.88% | 4 |
| BAC | 21 | 0.91% | 0.93% | 1 |
| CAT | 21 | 1.66% | 1.66% | 0 |
| COST | 21 | 0.85% | 0.84% | 1 |
| CRM | 21 | 1.60% | 1.60% | 0 |
| CVX | 21 | 1.08% | 1.08% | 0 |
| DIS | 21 | 0.97% | 0.97% | 1 |
| GOOGL | 21 | 1.21% | 1.21% | 0 |
| HD | 21 | 1.28% | 1.29% | 1 |
| JNJ | 21 | 1.59% | 1.57% | 1 |
| JPM | 21 | 0.75% | 0.74% | 1 |
| KO | 21 | 0.95% | 0.95% | 0 |
| LLY | 21 | 1.49% | 1.46% | 1 |
| MA | 21 | 1.09% | 1.09% | 0 |
| META | 20 | 2.03% | 2.03% | 0 |
| MSFT | 21 | 1.00% | 1.00% | 1 |
| NFLX | 21 | 1.60% | 1.60% | 0 |
| NVDA | 21 | 1.39% | 1.34% | 3 |
| ORCL | 21 | 2.89% | 2.81% | 2 |
| PEP | 21 | 1.14% | 1.14% | 0 |
| PG | 21 | 1.31% | 1.33% | 1 |
| TSLA | 21 | 1.65% | 1.65% | 0 |
| UNH | 21 | 1.53% | 1.51% | 1 |
| V | 21 | 1.03% | 1.03% | 0 |
| WMT | 21 | 1.25% | 1.26% | 1 |
| XOM | 21 | 1.02% | 1.02% | 0 |

_Research experiment, not financial advice._
