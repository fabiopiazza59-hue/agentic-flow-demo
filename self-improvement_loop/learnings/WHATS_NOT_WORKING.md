# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate**: 17 of 24 failures are wrong-direction, not just magnitude. This is a *sign* problem, not a calibration-of-size problem. The model can't call which way AMZN closes.
- **Barely beats baseline**: on most fails APE ≈ baseline_ape (e.g. 06-17 .0373 vs .0358, 07-15 .0298 vs .0293). We're adding no edge over a naive predictor on hard days.
- **Fat-tail blowups**: 07-31 APE 13.2%, 06-22 5.1%, 07-30 3.9% — large one-directional gaps swamp the mean_fail_ape (2.83%).
- **All strategies are weak**: best is news at 29.7% hit rate; technical/contrarian/macro sit at 13-16%. No strategy is reliably right — this is a coin-flip ensemble.

## Unreliable under these conditions
- **High-volatility / large-move days**: whenever the true move is big (>3%), the model consistently under-shoots and gets direction wrong (06-15, 06-22, 06-25, 07-15, 07-23, 07-31).
- **macro-led and news-led calls on volatile days**: macro wins only 13.5% and is the pick on several worst blowups (07-30, 07-31 region, 06-08 the lone macro win is luck). news is "best" but still misses direction on big days.
- **Confidence is NOT overconfident — it's uninformative**: 0 overconfident misses, but confidence clusters at 0.4 regardless of outcome. Passes and fails share the same ~0.4 confidence. It carries zero discriminative signal.
- **Empty-weights days** (06-30, 07-22, 07-24, 07-28) are mixed-to-bad — fallback path is unmanaged.

## Fixes to try next
- Treat this as a **directional classification problem first**; stop optimizing magnitude when sign is wrong 46% of the time.
- **Down-weight technical, contrarian, macro** (hit rates ≤16%); lean on news but cap it — it's still <30%.
- Build a **volatility regime gate**: on high-vol days widen intervals or defer to baseline instead of committing to a direction.
- **Recalibrate confidence** — current 0.4-flat output is meaningless; force separation so low-confidence days are actually predictive of misses.
- Audit the empty-`weights` fallback; it should never silently fire.