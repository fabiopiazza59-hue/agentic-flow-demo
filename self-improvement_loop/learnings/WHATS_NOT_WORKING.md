# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core problem.** 17 of 23 fails (74%) are directional misses, not magnitude. The model calls the sign wrong and loses regardless of tight error bars.
- **We rarely beat baseline on fails.** Most failing days the ensemble lands worse than or barely at the naive baseline — we're adding noise, not signal.
- **News is the crutch.** It gets the top weight_hint (0.23) and highest hit rate (0.31), but that's still a coin-flip that loses. When news is the winning strategy, outcomes are split at best.
- **Every strategy is sub-random on direction.** Best is news at 31%; technical/contrarian/macro all sit at ~14-17% hit rate. That's structurally broken sign prediction, not one bad regime.
- **Tail blowups.** 2026-07-31 (13.2% APE) and multiple 3-5% days (06-22, 06-25, 07-15, 07-23, 07-30) cluster on macro/news wins during large moves.

## Unreliable under these conditions
- **High-volatility / large-move days:** APE spikes to 3-13% cluster around macro- and news-led calls; the model cannot size big moves.
- **Macro-led days:** hit rate 14%, worst MAPE (0.020). Macro as winner correlates with the biggest misses (07-15, 07-30, 07-31).
- **Contrarian and technical as winners:** lowest hit rates; contrarian especially fails on trend continuation.
- **Confidence is uninformative, not overconfident.** 0 overconfident misses, but confidence is flat/low (0.4 dominant) on both wins and fails — it carries no signal. Some 0.55 calls fail (06-12, 06-30, 07-13, 07-20); some 0.1 calls pass. Calibration is dead.
- **Empty-weights days** (06-30, 07-22, 07-28, 08-03) show config gaps slipping through.

## Fixes to try next
- **Attack the sign problem directly:** train/evaluate a dedicated direction classifier; stop grading magnitude when sign is wrong.
- **Down-weight macro, technical, contrarian** toward zero on high-vol days; gate ensemble to news+momentum when realized vol is elevated.
- **Add a volatility regime detector** and widen/scale predictions on large-move days to avoid systematic under-sizing.
- **Rebuild confidence calibration** — current scores are noise; suppress trades when calibrated confidence is low rather than emitting flat 0.4.
- **Fix empty-weights path**; no prediction should ship without a resolved weight vector.