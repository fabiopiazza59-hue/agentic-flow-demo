# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- Directional errors dominate: 24 of 36 fails (67%) are direction misses, not just magnitude. The model can't consistently call up vs down — it's a coin flip that's actually worse than a coin flip on fail days.
- Overall hit rates are dismal across the board (12.5%–28%). Even the "best" strategy (news, 28%) loses direction 7 times in 10.
- Big-move days are catastrophic. On high-APE days (2.5%+ moves: 6/15, 6/22, 6/25, 6/29, 7/15, 7/30, 7/31, 8/19) the model repeatedly misses and barely beats or trails baseline. 7/31 at 13.2% APE is a total blowup.
- Failures cluster in time (mid-June, late-June, early Aug) suggesting regime-driven breakdowns, not random noise.

## Unreliable under these conditions
- **macro** strategy is the worst performer (12.5% hit, highest MAPE 0.0166) yet keeps winning selection on losing days (7/13, 7/15, 7/30, 8/19, 8/25) — it's being trusted exactly when it shouldn't.
- **momentum** and **contrarian** both crater on high-volatility / reversal days — they chase or fade wrong at turning points (7/8, 8/11, 8/20, 9/3).
- Confidence miscalibration in the RECENT high-confidence era (late Aug–Sep): conf 0.62–0.72 on 8/19 (fail), 8/31 (fail), 9/3 (fail), 9/9 (fail), 9/11 (fail). Rising confidence, still missing — the 4 flagged overconfident misses understate the problem.
- Baseline is hard to beat: on most fails the model trails a naive baseline, meaning the ensemble adds negative value on bad days.

## Fixes to try next
- Cap or drop **macro** as a winning strategy; its low hit rate doesn't justify its selection frequency. Reweight toward **news**.
- Recalibrate confidence: current 0.6+ predictions are not more accurate than 0.4 ones. Fit confidence to realized hit rate; suppress high-confidence output on high-implied-vol days.
- Add a volatility regime gate: when expected move >2%, defer to baseline or widen intervals rather than committing to a direction.
- Since errors are directional, build a separate direction classifier and only trust magnitude blend when direction confidence clears a threshold.