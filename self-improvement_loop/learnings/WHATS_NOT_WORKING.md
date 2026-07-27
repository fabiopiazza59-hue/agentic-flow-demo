# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the killer**: 16 of 20 fails are directional misses (80%). The model gets the size roughly right but calls the wrong way. Magnitude-only fails are just 4.
- Overall we fail 2 of every 3 days (20/30), and mean fail APE is 2.35% — large moves get missed, not just noise.
- **We lose to the naive baseline on most fails**: on the big-error days (June 22, 25, 26, 29; July 15, 23) we are barely at or worse than baseline — no edge when it matters.
- No overconfidence flag firing (0), but that's misleading: confidence sits in a dead 0.40–0.55 band regardless of outcome, so it carries zero discriminating signal. It's uncalibrated by being flat, not by being too high.
- **News is the most-trusted strategy (highest weight ~0.22) yet only hits 33%** — it wins the ensemble often and is still wrong 2/3 of the time. It's leading us off cliffs.

## Unreliable under these conditions
- **High-volatility / large-move days** (baseline APE >3%): near-total directional failure — June 15, 17, 22, 25, 26, 29; July 15, 23 all missed direction. The ensemble mean-reverts into trending/gapping tapes.
- **macro-led days**: hit rate 0.10 (3/30) — essentially a coin flip that loses. macro should not be a winning strategy.
- **momentum-led calls in choppy periods** (June 12, 16, 25; July 24): flips direction repeatedly.
- Late-July low-confidence days (0.24) still logged as active predictions and still failing — near-abstention isn't stopping bad trades.

## Fixes to try next
- Attack direction explicitly: build/score a separate sign classifier; stop optimizing APE alone.
- Cut macro to near-zero weight; cap news influence — its high weight + low hit rate is a net drag.
- Add a volatility regime gate: when expected move >~2.5%, widen/abstain rather than mean-revert.
- Rebuild confidence so it spans a real range and correlates with hit rate; auto-abstain below a threshold instead of logging weak predictions.