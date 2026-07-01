# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the killer.** 8 of 9 fails are directional misses (only 1 pure magnitude). We are systematically calling the wrong sign, then the error size follows. Fixing APE won't help until we fix the up/down call.
- **We lose to the naive baseline on nearly every fail.** 8 of 9 failing days had ape ≥ baseline_ape. On misses the model isn't just wrong, it's *worse than doing nothing*.
- **No overconfidence flagged, but confidence is uniformly flat (~0.4–0.58).** Confidence carries almost no signal — it doesn't separate the 4 passes from the 9 fails. That's miscalibration by uselessness, not by arrogance.
- **Every strategy has a losing hit rate.** Best is news at 0.31; macro is a disaster at 0.08. No strategy clears 50% — the ensemble is averaging garbage.

## Unreliable under these conditions
- **Late-June clustering (6/22–6/30): 6 straight fails**, mostly 3–5% APE. Something regime-shifted (volatility/trend break) and the model never adapted — it kept fading the real move.
- **Momentum as winning_strategy → fails hard** (6/12, 6/25 both big directional misses, ~2–3.5% APE). Momentum is being trusted right when trend is reversing.
- **Macro-heavy weighting fails** (6/12 macro 0.34 → miss; macro hit rate 0.08). Macro tilt is actively harmful.
- **News wins nominally but still misses direction on big days** (6/15, 6/22, 6/29) — news captures small moves, breaks on large ones.
- **Contrarian is the worst reliability (0.15 hit rate)** yet keeps getting meaningful weight.

## Fixes to try next
- **Retrain/gate the direction classifier separately from magnitude**; current blend can't call sign.
- **Cut macro and contrarian weight toward zero** until hit rate clears 0.4; they're dead weight.
- **Add a regime/volatility filter** — the 6/22+ streak shows the ensemble doesn't detect trend breaks; widen intervals or abstain when volatility spikes.
- **Recalibrate confidence** so it actually predicts pass/fail; flat 0.4 confidence is a non-signal. Suppress trades below a calibrated threshold.
- **Add a beat-baseline guardrail:** if the blend can't beat naive persistence in backtest for the current regime, default to baseline.