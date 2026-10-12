# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate**: 28 of 47 failures (60%) are wrong-way calls, not just magnitude. The system can't reliably call sign, which is the whole game.
- **Baseline-beaten on failures**: many fails have ape ≈ or above baseline_ape (e.g. 06-22, 06-25, 07-13, 08-19), so the ensemble adds no edge exactly when it misses — it's not just losing, it's underperforming a naive guess.
- **Big-move blindness**: the worst APEs cluster on high-volatility days (07-31 ape 13.2%, 06-22 5.1%, 07-23 4.2%, 07-30 3.9%). Model systematically undershoots magnitude on gap/shock days.
- **Macro is the weakest link**: hit_rate 11.8%, highest MAPE (0.0165). Nearly every day macro "wins" the ensemble it's a fail (06-12, 07-13, 07-15, 07-30, 08-19, 08-25). Technical nearly as bad (12.9%).

## Unreliable under these conditions
- **High-volatility / large true-move days**: magnitude consistently too small; errors are directional + large.
- **Macro-led and technical-led days**: lowest hit rates; these strategies should rarely be the decider.
- **Rising-confidence regime (recent, conf 0.52–0.66)**: confidence climbed into the 0.6s but fails continue (10-06 through 10-09 all fail at conf ~0.6). **Miscalibrated** — confidence rose without accuracy rising. The 7 overconfident misses (08-19 conf .62, 08-31 .72, 09-09 .72) confirm high-conf calls still miss, often on direction.
- **Early-period low-confidence (0.1–0.3)**: appropriately low but still fails — model knew it was lost but had no fallback.

## Fixes to try next
- **Cut macro and technical weight** on decider role; cap their standalone influence. Lean on news (31.8% hit, lowest MAPE) and contrarian.
- **Recalibrate confidence**: current 0.6s are not earning their calibration. Regress confidence against realized hit rate and shrink toward base rate (~40%).
- **Add a volatility regime detector**: widen magnitude / fatten predicted move on high-ATR or post-earnings days to kill the undershoot pattern.
- **Build a directional sign model separate from magnitude** — 60% of failures are sign errors; magnitude tuning won't fix those.
- **Fallback to baseline** when ensemble disagreement is high or macro/technical lead; stop losing to the naive benchmark.