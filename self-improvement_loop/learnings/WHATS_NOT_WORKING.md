# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core problem.** 21 of 30 fails (70%) are directional misses; APE is often only marginally worse than baseline. The model gets the price roughly right but the sign wrong — it's essentially tracking the prior close, not predicting the move.
- **The system barely beats a naive baseline.** Many fails have ape ≈ baseline_ape (e.g. 06-15, 06-22, 07-15, 08-20), meaning no edge is being added on hard days.
- **Every strategy has a losing directional hit rate.** Best is news at 26%, worst is macro at 12% — all below a coin flip. This isn't a strategy-selection problem; the underlying signals lack directional information.
- **Big-magnitude tail blowups.** 07-31 (13% APE) and repeated ~3-5% misses drag MAPE up; these cluster on high-volatility days.

## Unreliable under these conditions
- **`macro` as winning strategy is a red flag.** 12% hit rate, worst MAPE (0.0182); nearly every macro-led day fails (06-12 setup, 07-13, 07-15, 07-30, 08-19). Stop trusting macro-driven calls.
- **High-confidence misses on volatile days.** Confidence ≥0.5 fails: 06-12, 07-13, 07-20, 08-07, 08-19, 08-20. Late-Aug cluster (08-19/20 at conf 0.62/0.54) shows confidence rising precisely when accuracy collapses — mild but real miscalibration despite the "1 overconfident" flag.
- **News-heavy weightings (news ≥0.34) underdeliver** on directional turns (06-17, 07-13, 07-24, 08-11) — heavy news tilt correlates with directional whiffs.
- **Large-move days (baseline_ape >0.03)** are almost never caught correctly regardless of strategy.

## Fixes to try next
- Attack direction directly: add a dedicated sign/regime classifier and score/reward directional hit, not just APE.
- Cut or floor `macro` weight; it's the weakest on both hit rate and MAPE.
- Recalibrate confidence — cap confidence on high-volatility (high baseline_ape) days; current confidence trends up when it should trend down.
- Cap/dampen news weight above ~0.30; heavy news tilts precede directional misses.
- Add a volatility guard: on expected large-move days, widen intervals or abstain rather than emit a confident point estimate.