# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional failure dominates**: 4 of 5 misses are wrong-direction, not magnitude. The model can't call which way AMZN moves — only 1 pure magnitude miss. This is a signal problem, not a calibration-of-size problem.
- **Errors are escalating**: APE walks up over time (2.0% → 3.4% → 3.7% → 5.1% on 6/22). The model is drifting further from price, not converging.
- **We lose to baseline on the misses**: every failed day except 6/18 also failed `beats_baseline`. A naive baseline would have beaten us on 4 of 7 days. That's the headline indictment.
- **No genuine overconfidence** (0 overconfident misses), but only because confidence is uniformly low (0.38–0.58). The system isn't calibrated so much as it's perpetually unsure — it never earns a high-confidence bet, so it can't be "wrong with conviction."

## Unreliable under these conditions
- **News-led days are a trap**: `news` wins the ensemble on both 6/15 and 6/22 — the two worst APE days (3.4%, 5.1%), both directional misses. Yet news carries the **highest weight_hint (0.243)**. We're over-trusting the worst regime driver.
- **macro-led** is the only winning day (6/08), but macro has the **worst MAPE (0.0278)** and lowest hit rate over the window — its one good call is masking systemic weakness.
- **technical** is the weakest pillar: 14% hit rate, won only on 6/17 (a 3.7% miss). It should not be earning a 0.20 weight.
- All five strategies cluster at **14–29% hit rate** — the whole ensemble is below coin-flip directionally. This is a regime the model fundamentally misreads (likely a trending/gapping period it keeps fading).

## Fixes to try next
- **Cut news weight** from 0.24 toward ~0.15 until it stops winning the worst days; it's a contrarian indicator of failure right now.
- **Add a baseline-fallback gate**: when ensemble disagreement is high or no strategy clears a hit-rate threshold, default to last-close/random-walk — it would have saved 4 days.
- **Diagnose the directional signal directly** — build a sign-only classifier and audit it separately from magnitude; the size logic is roughly fine, the direction logic is broken.
- **Suppress technical and macro** weights; promote whichever strategy actually called direction on trending days (none have yet — flag this regime as out-of-distribution).
- Sample size is tiny (n=7); treat all of this as directional, re-run after 20+ scored days.