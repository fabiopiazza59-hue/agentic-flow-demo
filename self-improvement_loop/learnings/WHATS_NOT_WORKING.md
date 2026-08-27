# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core problem.** 22 of 31 fails are directional misses (71%). The model gets the size roughly right but bets the wrong way. Mean fail APE 2.62% — these aren't blowups, they're coin-flip sign errors.
- **Overall directional hit rate is barely a coin flip.** Passes cluster on small-move days; whenever the true move is large (APE >3%), the sign is almost always wrong (see 06-15, 06-17, 06-22, 06-25, 06-29, 07-15, 07-30, 08-19).
- **"Beats baseline" tracks "directional hit" almost 1:1.** When we miss direction we lose to the naive baseline; we add nothing on the days that matter.
- **macro is the weakest winning strategy** — 13.2% hit rate, highest MAPE (0.0175), yet still gets picked as winner and carries a ~0.19 weight. Every macro-led day in the log (06-08 aside) is a fail: 07-13, 07-15, 07-30, 08-19.
- **No systematic overconfidence problem** (only 1 flagged), BUT confidence is uninformative: 08-19/08-20 misses ran conf 0.62/0.54; a clean pass (08-26) ran 0.54 too. Confidence doesn't separate hits from misses — it's noise, not miscalibration in the loud sense.

## Unreliable under these conditions
- **Large-move / high-volatility days:** every fail with APE >2.5% is a directional miss. The system has no edge in trending or gap regimes.
- **macro- and contrarian-led selections:** macro 13% hit, contrarian 21%; both underperform news/momentum (24.5%) and both dominate the fail list late July–August.
- **Blank-weights days** (07-30, 08-05, 08-11 window, 07-22/24 low-conf): correlate with poor outcomes — likely fallback/default pathing when inputs are thin.
- **Consecutive-miss streaks** (06-25→06-30, 07-20→07-24, 08-19→08-24) suggest regime persistence the model fails to detect and keeps fighting.

## Fixes to try next
- Add a **regime/volatility filter**: when expected move is large, either widen or abstain — stop taking sign bets we lose 70%+ of the time.
- **Demote or bench macro** (and reduce contrarian weight) until it clears baseline hit rate; reallocate to news/momentum.
- Build a **direction-specific model/gate** separate from magnitude, since magnitude is already acceptable.
- **Recalibrate confidence** against realized hits — current scores don't predict accuracy; consider suppressing trades below a calibrated threshold.
- **Flag blank-weights fallbacks** and treat them as low-conviction/abstain rather than scoring them as live calls.