# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (12 sessions, 359 name-forecasts; mean daily APE gain -0.005%, CS [-0.022%, +0.011%]).

**Pending (2026-10-14):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 1 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.005% | [-0.022%, +0.011%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.004% | [-0.017%, +0.008%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.182 | n=4 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | -0.001 | [-0.007, +0.005] | ≈ not distinguishable |
| APE gain: LLM vs prior close | -0.034% | [-0.111%, +0.042%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | -0.029% | [-0.103%, +0.046%] | ≈ not distinguishable |

MAPE — LLM 1.25%, free baseline 1.24%, prior close 1.21%; LLM 80% interval coverage 79.13%; names moved per session 1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 7 | +0.005% | — | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 6 | +0.006% | — | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 12 | 0.97% | 0.97% | 0 |
| ADBE | 12 | 1.44% | 1.44% | 0 |
| AMD | 12 | 1.85% | 1.85% | 0 |
| AMZN | 12 | 1.16% | 1.16% | 0 |
| AVGO | 12 | 2.02% | 1.98% | 2 |
| BAC | 12 | 0.91% | 0.91% | 0 |
| CAT | 12 | 1.78% | 1.78% | 0 |
| COST | 12 | 0.68% | 0.67% | 1 |
| CRM | 12 | 1.64% | 1.64% | 0 |
| CVX | 12 | 0.94% | 0.94% | 0 |
| DIS | 12 | 0.93% | 0.95% | 1 |
| GOOGL | 12 | 0.85% | 0.85% | 0 |
| HD | 12 | 1.11% | 1.13% | 1 |
| JNJ | 12 | 1.69% | 1.66% | 1 |
| JPM | 12 | 0.72% | 0.70% | 1 |
| KO | 12 | 0.79% | 0.79% | 0 |
| LLY | 12 | 1.06% | 1.06% | 0 |
| MA | 12 | 1.18% | 1.18% | 0 |
| META | 11 | 1.74% | 1.74% | 0 |
| MSFT | 12 | 0.90% | 0.89% | 1 |
| NFLX | 12 | 1.51% | 1.51% | 0 |
| NVDA | 12 | 1.19% | 1.21% | 1 |
| ORCL | 12 | 2.72% | 2.64% | 1 |
| PEP | 12 | 0.91% | 0.91% | 0 |
| PG | 12 | 1.09% | 1.09% | 0 |
| TSLA | 12 | 1.45% | 1.45% | 0 |
| UNH | 12 | 1.14% | 1.11% | 1 |
| V | 12 | 0.99% | 0.99% | 0 |
| WMT | 12 | 1.26% | 1.28% | 1 |
| XOM | 12 | 0.89% | 0.88% | 0 |

_Research experiment, not financial advice._
