# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (43 sessions, 1288 name-forecasts; mean daily APE gain -0.002%, CS [-0.011%, +0.008%]).

**Pending (2026-11-27):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.002% | [-0.011%, +0.008%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.001% | [-0.009%, +0.007%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.019 | [-0.239, +0.200] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.001 | [-0.003, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.107% | [-0.043%, +0.258%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.109% | [-0.039%, +0.256%] | ≈ not distinguishable |

MAPE — LLM 1.69%, free baseline 1.69%, prior close 1.80%; LLM 80% interval coverage 78.72%; names moved per session 0.9. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 38 | -0.010% | [-0.035%, +0.016%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 37 | -0.003% | [-0.027%, +0.022%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 43 | 1.16% | 1.16% | 1 |
| ADBE | 43 | 1.95% | 1.95% | 0 |
| AMD | 43 | 3.92% | 3.94% | 3 |
| AMZN | 43 | 1.84% | 1.84% | 0 |
| AVGO | 43 | 2.66% | 2.66% | 5 |
| BAC | 43 | 1.10% | 1.11% | 1 |
| CAT | 43 | 2.57% | 2.55% | 1 |
| COST | 43 | 0.91% | 0.91% | 1 |
| CRM | 43 | 1.85% | 1.85% | 0 |
| CVX | 43 | 1.25% | 1.26% | 1 |
| DIS | 43 | 1.17% | 1.17% | 2 |
| GOOGL | 43 | 1.45% | 1.46% | 2 |
| HD | 43 | 1.37% | 1.37% | 1 |
| JNJ | 43 | 1.48% | 1.47% | 1 |
| JPM | 43 | 1.12% | 1.12% | 1 |
| KO | 43 | 0.88% | 0.87% | 1 |
| LLY | 43 | 1.42% | 1.44% | 3 |
| MA | 43 | 1.22% | 1.23% | 1 |
| META | 41 | 2.11% | 2.11% | 0 |
| MSFT | 43 | 1.44% | 1.43% | 2 |
| NFLX | 43 | 1.63% | 1.63% | 0 |
| NVDA | 43 | 2.97% | 2.92% | 5 |
| ORCL | 43 | 3.47% | 3.44% | 2 |
| PEP | 43 | 1.12% | 1.12% | 0 |
| PG | 43 | 1.08% | 1.09% | 1 |
| TSLA | 43 | 2.26% | 2.26% | 0 |
| UNH | 43 | 1.82% | 1.81% | 1 |
| V | 43 | 1.15% | 1.15% | 0 |
| WMT | 43 | 1.16% | 1.17% | 1 |
| XOM | 43 | 1.28% | 1.29% | 1 |

_Research experiment, not financial advice._
