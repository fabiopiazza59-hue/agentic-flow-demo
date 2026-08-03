# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional errors dominate:** 17 of 23 misses (74%) are wrong on *direction*, not just magnitude. This is a sign problem, not a calibration/tuning problem — the model can't tell up from down on most fail days.
- **We barely beat a naive baseline.** On most fails baseline_ape ≈ our ape (e.g. 06-15, 06-17, 06-22, 07-15, 07-29). We're adding no edge on hard days.
- **Big-move days are catastrophic.** APE spikes on 06-22 (5.1%), 07-23 (4.2%), 07-30 (3.9%), 07-31 (13.2%) — the model has no handle on volatility regime shifts / gap days.
- **macro as winning_strategy is a red flag:** when macro "wins" it's usually on high-APE fails (06-12, 07-13, 07-15, 07-30). It gets picked in the absence of a better signal.

## Unreliable under these conditions
- **High-volatility / large daily moves:** every APE >3% is a fail; magnitude is systematically underestimated.
- **contrarian and macro are the worst strategies** (hit rate 14% each, MAPE ~2.0%). technical isn't much better (17%). None should carry ~19–20% weight.
- **Confidence is uninformative, not overconfident.** overconfident_misses=0, but confidence is compressed (0.1–0.58) and clusters at 0.40 on both wins and fails — no discrimination. Low-conf days (0.10, 0.24) fail just as often. Calibration is dead, not miscalibrated.
- **Empty-weights days (06-30, 07-22, 07-28) still get scored** — pipeline/fallback bug producing predictions with no strategy blend.

## Fixes to try next
- **Attack the direction problem first:** add a regime/volatility filter; when expected move is large, widen intervals and down-weight point-direction bets.
- **Reweight by realized hit rate:** boost news (31%, best MAPE), cut contrarian and macro toward zero; stop letting macro "win" by default.
- **Rebuild confidence:** recalibrate so it actually separates hits from misses, or drop it until it earns predictive value.
- **Fix the empty-`weights` fallback path** and audit those days.