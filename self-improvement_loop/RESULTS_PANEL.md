# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (40 sessions, 1198 name-forecasts; mean daily APE gain -0.002%, CS [-0.012%, +0.009%]).

**Pending (2026-11-23):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.002% | [-0.012%, +0.009%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.001% | [-0.009%, +0.007%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.019 | [-0.239, +0.200] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0011 | [-0.003, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.083% | [-0.052%, +0.218%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.085% | [-0.046%, +0.216%] | ≈ not distinguishable |

MAPE — LLM 1.67%, free baseline 1.67%, prior close 1.76%; LLM 80% interval coverage 78.13%; names moved per session 0.9. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 35 | -0.010% | [-0.038%, +0.018%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 34 | -0.003% | [-0.030%, +0.024%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 40 | 1.16% | 1.16% | 1 |
| ADBE | 40 | 1.89% | 1.89% | 0 |
| AMD | 40 | 3.91% | 3.93% | 3 |
| AMZN | 40 | 1.92% | 1.92% | 0 |
| AVGO | 40 | 2.73% | 2.73% | 5 |
| BAC | 40 | 1.11% | 1.12% | 1 |
| CAT | 40 | 2.50% | 2.49% | 1 |
| COST | 40 | 0.92% | 0.92% | 1 |
| CRM | 40 | 1.73% | 1.73% | 0 |
| CVX | 40 | 1.25% | 1.26% | 1 |
| DIS | 40 | 1.12% | 1.11% | 2 |
| GOOGL | 40 | 1.43% | 1.44% | 2 |
| HD | 40 | 1.31% | 1.31% | 1 |
| JNJ | 40 | 1.49% | 1.48% | 1 |
| JPM | 40 | 1.16% | 1.15% | 1 |
| KO | 40 | 0.86% | 0.85% | 1 |
| LLY | 40 | 1.41% | 1.43% | 3 |
| MA | 40 | 1.20% | 1.21% | 1 |
| META | 38 | 2.00% | 2.00% | 0 |
| MSFT | 40 | 1.48% | 1.47% | 2 |
| NFLX | 40 | 1.63% | 1.63% | 0 |
| NVDA | 40 | 2.62% | 2.57% | 5 |
| ORCL | 40 | 3.47% | 3.43% | 2 |
| PEP | 40 | 1.18% | 1.18% | 0 |
| PG | 40 | 1.13% | 1.13% | 1 |
| TSLA | 40 | 2.13% | 2.13% | 0 |
| UNH | 40 | 1.87% | 1.86% | 1 |
| V | 40 | 1.14% | 1.14% | 0 |
| WMT | 40 | 1.16% | 1.16% | 1 |
| XOM | 40 | 1.30% | 1.31% | 1 |

_Research experiment, not financial advice._
