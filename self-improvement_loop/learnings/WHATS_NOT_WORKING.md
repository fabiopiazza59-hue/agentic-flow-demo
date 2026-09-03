# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- Direction is the primary failure mode: 22 of 33 fails are directional misses (67%). We aren't magnitude-off, we're getting the sign of the move wrong.
- Overall directional hit rate is coin-flip-or-worse across every strategy (best is news at 27.6%, macro at 12%). The ensemble adds little edge over baseline — many fails have ape ≈ baseline_ape (e.g., 07-15, 08-20, 08-24).
- Big-move days are where damage concentrates: fails cluster at 3–5% APE (06-22, 06-25, 07-15, 07-23, 07-30) and one 13% blowup (07-31). The model reverts to small moves and gets run over on volatility.
- macro is the worst engine (12% hit, highest MAPE .0172) yet keeps getting picked as winning_strategy on losing days (06-08 aside, see 07-13, 07-15, 07-30, 08-19).

## Unreliable under these conditions
- High-volatility / gap days: whenever the true move is large, direction flips and APE spikes. The system has no regime awareness.
- macro-led and contrarian-led predictions: both low hit rate; contrarian fails repeatedly late-Aug (08-11, 08-20, 08-24) even at moderate confidence.
- Late-period confidence inflation: confidence crept up (0.5–0.72 in late Aug) while still missing — 08-19 (conf .62, fail), 08-20 (.54, fail), 08-31 (.72, fail). Calibration is drifting the wrong way despite only 2 flagged "overconfident" misses (threshold too loose).
- Days with empty weights blob still get scored — several fails (06-30, 07-22, 07-24) suggest a config/plumbing gap.

## Fixes to try next
- Add a volatility regime filter; widen predicted move (stop mean-reverting) or abstain when expected range is high.
- Cut macro weight hard and down-weight contrarian; it should not win on high-vol days.
- Recalibrate confidence: recent high-confidence misses mean the mapping is stale — refit on last ~20 days and tighten the overconfidence flag threshold.
- Treat this as a direction-first problem: build/evaluate a dedicated sign classifier separate from magnitude; current blend barely beats baseline.
- Investigate empty-weights predictions — likely silent fallback producing bad calls.