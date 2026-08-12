# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core problem, not magnitude.** 19 of 27 fails are directional misses (70%). Mean fail APE of 2.7% means when we're wrong, we're wrong on the sign, and the size hurts. Getting the level roughly right doesn't save us when we call the wrong way.
- **We barely beat the naive baseline.** On failed days APE is routinely at/above baseline_ape (e.g. 06-12, 06-15, 06-17, 06-22, 06-25, 07-13, 07-15, 07-20, 07-21, 07-24, 08-10). We're adding noise, not signal, on hard days.
- **Big misses cluster around specific dates** (07-31 at 13.2% APE, 06-22/06-25/07-15/07-30 near 3–5%) — likely earnings/gap events the model doesn't anticipate.
- **All strategies have losing directional hit-rates.** Best is news at 28.6%; that's worse than a coin flip across the board. This is a systemic edge problem, not a rotation problem.

## Unreliable under these conditions
- **Macro and technical are dead weight**: macro hit_rate 11.9%, technical 14.3%, both with the highest MAPE. When macro "wins" the blend (06-12, 07-13, 07-15, 07-30) it loses directionally and misses big.
- **High-volatility / gap days** (APE >3%): the model consistently misses direction and lands at baseline — no regime awareness for large moves.
- **News-heavy weight days** underperform despite news being the "best" strategy — heavy news tilts (07-13, 07-23, 07-24 at 0.40+ news) still failed.
- **Confidence is uninformative, not overconfident**: 0 overconfident misses, but low-confidence days (0.1–0.24) pass and fail almost at random (07-27 pass@0.1, 07-30 fail@0.1). Confidence carries no predictive information either way.

## Fixes to try next
- Attack direction directly: train/evaluate a separate sign classifier and gate magnitude on it; stop optimizing APE alone.
- Cut or heavily down-weight macro and technical; they lose more than they add.
- Build a gap/volatility regime flag (earnings calendar, prior-day range) and widen or abstain on those days rather than defaulting to baseline.
- Recalibrate confidence against realized directional hits — current scores are flat noise; if it can't separate, drop it.