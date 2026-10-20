# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- Overall 51/90 fail (57%). Directional misses (30) outnumber magnitude misses (21) — the core problem is **getting the sign wrong**, not just the size.
- Big-move days blow up: APE spikes to 3-13% (2026-06-22, 07-31, 08-19, 10-19). The model systematically under-reacts and lags on large daily swings.
- **Macro** is the worst strategy: 12% hit rate, highest MAPE (0.0167). Every day it wins outright (06-12, 07-13, 07-15, 08-19, 07-30) it tends to miss. It's a loser whenever it dominates the blend.
- **Technical** also weak (13% hit rate) yet carries the second-highest weight hint (0.207). Weighting is misaligned with realized skill.
- Recent October stretch is ugly: 10-06 through 10-19 is almost all fails despite rising confidence — persistent regime the model hasn't adapted to.
- Note: many fails still "beat baseline" — meaning the baseline is also missing. These are hard regime days, not pure model error, but the model isn't beating the bar often enough.

## Unreliable under these conditions
- High-volatility / large-gap days (>2% true move): consistent magnitude under-shoot and frequent sign error.
- When macro or technical is the winning strategy, especially with low macro sub-weight (~0.07-0.12) — the blend leans on a strategy that's already failing.
- **Confidence is miscalibrated upward over time.** Recent conf 0.55-0.66 days (10-06, 10-08, 10-19, 09-03, 08-19) are misses. 7 overconfident misses confirm high conf ≠ accuracy; calibration has drifted since the 0.1-0.4 early period.
- Choppy sideways runs where contrarian wins but still misses direction (10-08, 10-15, 10-19).

## Fixes to try next
- Cut macro and technical weights toward their realized hit rates; raise news (0.30 hit, lowest MAPE) and contrarian.
- Add a volatility regime filter: widen intervals / lower confidence on predicted large-move days instead of under-reacting.
- Recalibrate confidence against recent realized pass rate — current 0.55-0.66 band is unjustified; shrink it.
- Build a directional-accuracy gate separate from magnitude; 30 sign errors is the biggest lever.
- Investigate the October regime break specifically — retrain or down-weight stale signals.