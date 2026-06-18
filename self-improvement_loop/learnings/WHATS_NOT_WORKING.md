# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **The failures are 100% directional, not magnitude.** All 3 misses (and even the lucky pass on 06-16) blew the direction call; dir_misses=3, mag_misses=0. The model gets the size roughly right but the sign wrong — it's a turning-point/regime problem, not a calibration-of-magnitude problem.
- **Losing to baseline on every fail.** All 3 failures had ape > baseline_ape (06-12, 06-15, 06-17). On miss days a naive baseline beats us — we add negative value precisely when it matters.
- **The one clean win (06-08) came from a flat equal-weight 0.2 ensemble.** As soon as weights tilted toward momentum/news/macro, results degraded. Concentration hurt.
- Confidence is low across the board (0.38–0.58), so no overconfidence flags fire — but it's also uselessly flat. Confidence isn't discriminating wins from losses.

## Unreliable under these conditions
- **Contrarian is dead weight: 0/5 hit rate, highest-ish MAPE (0.020).** It never wins yet still carries ~0.14–0.22 weight on fail days.
- **Technical and macro are weak (0.2 hit rate, MAPE ~0.020–0.022).** Technical "won" 06-17 and still missed direction badly (ape 0.037).
- **Consecutive-day clustering:** 06-15→06-16→06-17 all directional misses — looks like a sustained trend/regime the ensemble kept fading. Errors compound in trending or news-driven stretches.
- "Winning strategy" label is misleading: on 06-12, 06-15, 06-17 the named winner still got direction wrong — the selector is picking the least-bad loser.

## Fixes to try next
- **Add a directional gate / regime filter.** Since magnitude is fine but sign isn't, predict direction separately and only commit when strategies agree; otherwise default to baseline.
- **Cut contrarian weight to ~0 until it shows a positive hit rate;** trim technical/macro.
- **Penalize disagreement:** when no strategy clears a confidence threshold for direction, fall back to baseline (would have avoided sub-baseline losses).
- **Rebuild confidence to track directional hit-rate**, not magnitude — current scores don't separate wins from misses.
- Sample size is tiny (n=5); treat all of this as provisional and re-evaluate after ~20 scored days.