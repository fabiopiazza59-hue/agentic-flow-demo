# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
⏳ **Not enough clean sessions yet** to judge the LLM against the free baseline (9 sessions, 269 name-forecasts; mean daily APE gain 0.002%, CS n=9 sessions (too few)).

**Pending (2026-10-09):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 4 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | 0.002% | n=9 sessions (too few) | ⏳ too few sessions |
| CRPS gain: LLM vs free baseline | 0.001% | n=9 sessions (too few) | ⏳ too few sessions |
| Rank IC (adjustment vs realized move off the anchor) | 0.075 | n=1 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | 0.0008 | n=9 sessions (too few) | ⏳ too few sessions |
| APE gain: LLM vs prior close | -0.025% | n=9 sessions (too few) | ⏳ too few sessions |
| APE gain: anchor vs prior close (the free information itself) | -0.027% | n=9 sessions (too few) | ⏳ too few sessions |

MAPE — LLM 1.23%, free baseline 1.23%, prior close 1.20%; LLM 80% interval coverage 76.99%; names moved per session 0.4. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 4 | -0.009% | — | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 3 | -0.015% | — | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 9 | 0.99% | 0.99% | 0 |
| ADBE | 9 | 1.57% | 1.57% | 0 |
| AMD | 9 | 1.73% | 1.73% | 0 |
| AMZN | 9 | 1.05% | 1.05% | 0 |
| AVGO | 9 | 2.25% | 2.25% | 0 |
| BAC | 9 | 0.89% | 0.89% | 0 |
| CAT | 9 | 1.92% | 1.92% | 0 |
| COST | 9 | 0.71% | 0.69% | 1 |
| CRM | 9 | 1.67% | 1.67% | 0 |
| CVX | 9 | 0.94% | 0.94% | 0 |
| DIS | 9 | 1.08% | 1.10% | 1 |
| GOOGL | 9 | 0.98% | 0.98% | 0 |
| HD | 9 | 1.16% | 1.18% | 1 |
| JNJ | 9 | 1.19% | 1.19% | 0 |
| JPM | 9 | 0.75% | 0.75% | 0 |
| KO | 9 | 0.77% | 0.77% | 0 |
| LLY | 9 | 1.09% | 1.09% | 0 |
| MA | 9 | 1.03% | 1.03% | 0 |
| META | 8 | 1.61% | 1.61% | 0 |
| MSFT | 9 | 0.75% | 0.75% | 0 |
| NFLX | 9 | 1.71% | 1.71% | 0 |
| NVDA | 9 | 1.20% | 1.20% | 0 |
| ORCL | 9 | 2.27% | 2.27% | 0 |
| PEP | 9 | 0.97% | 0.97% | 0 |
| PG | 9 | 1.22% | 1.22% | 0 |
| TSLA | 9 | 1.53% | 1.53% | 0 |
| UNH | 9 | 1.00% | 1.00% | 0 |
| V | 9 | 0.77% | 0.77% | 0 |
| WMT | 9 | 1.35% | 1.37% | 1 |
| XOM | 9 | 0.81% | 0.81% | 0 |

_Research experiment, not financial advice._
