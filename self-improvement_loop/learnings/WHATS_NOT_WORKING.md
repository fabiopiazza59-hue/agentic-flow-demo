# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core failure.** 22 of 31 misses are directional (71%). The model gets the size roughly right but bets the wrong way. This is a sign/regime problem, not a scaling problem.
- **We barely beat a naive baseline.** Many fails have ape ≈ baseline_ape (e.g. 06-15, 06-17, 06-22, 07-15, 07-30), meaning on hard days the model just tracks yesterday's price and adds no edge.
- **News-weighted days are a recurring trap.** News is the winning strategy on a disproportionate share of the big misses (06-15, 06-22, 06-25→06-29 cluster, 07-20, 08-10) despite only a 25% hit rate.
- **Macro is the weakest link:** 12.7% hit rate, highest MAPE (0.0171), and it "wins" on several large-ape blowups (07-15, 07-30, 08-19). It's chosen precisely when it shouldn't be.

## Unreliable under these conditions
- **High-volatility / gap days** (ape 3–5%: 06-22, 06-25, 07-15, 07-23, 07-30, 08-19). Directional accuracy collapses here — the ensemble smooths through the move.
- **When macro or news carries the top weight in a choppy tape.** These two account for most oversized, wrong-direction misses.
- **Confidence is flat and uninformative, not overconfident.** Only 1 overconfident miss, but confidence (0.1–0.64) shows no relationship to outcome: 0.62 conf → miss (08-19), 0.1 conf → pass (07-27). Calibration is essentially noise.
- **Late-July regime break (07-29→07-31)** shows a sustained miss streak — the model lagged a trend change for multiple days.

## Fixes to try next
- **Add a volatility gate:** when expected daily range is high, widen tolerance and down-weight macro/news; don't force a directional call.
- **Cut macro weight** (0.187 → near floor) or gate it to genuine macro-event days only; it's a net drag.
- **Rebuild the directional layer** — magnitude is fine, sign is broken. Add a regime/trend filter so we stop fading real moves.
- **Recalibrate confidence** against realized hit rate; current scores carry no signal and should not be trusted for sizing.
- **Track "beats_baseline" as a gating metric** — on days we can't beat naive persistence, flag low-conviction rather than committing.