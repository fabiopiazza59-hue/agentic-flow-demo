# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core failure.** 22 of 31 misses (71%) are directional. Most fails have APE within ~0.1% of baseline (e.g. 07-15, 08-20, 08-24) — the model is basically tracking the naive baseline and losing the sign, not blowing out on size.
- **We rarely beat baseline on fails.** Nearly all failing days also fail `beats_baseline`, meaning on hard days the ensemble adds nothing over "yesterday's close."
- **Macro is the weakest winning strategy** (13% hit rate, highest MAPE 0.0173) yet still gets picked as the winner on several big-miss days (06-08 good, but 07-13, 07-15, 07-30, 08-19, 08-25 all fails/near-fails).
- **Technical wins the ensemble often but has the worst standalone hit rate (16.7%)** — it's being over-trusted as a tiebreak strategy.
- **Big tail miss 07-31 (APE 13.2%)** — likely a split/earnings/data event the model didn't flag; directional hit but magnitude catastrophic.

## Unreliable under these conditions
- **High-volatility / large-move days** (baseline_ape >2.5%): the model consistently misses direction — 06-15, 06-22, 06-25, 06-29, 07-15, 07-30, 08-19. On big-move days it's essentially a coin flip that loses.
- **Macro-led and news-led regimes:** macro and news win-days cluster in the fail list. Event-driven days (earnings/macro prints) overwhelm the technical/momentum signal.
- **Confidence is flat and uninformative, not overconfident.** Only 1 overconfident miss flagged, but confidence hovers 0.4–0.5 on both wins and losses (miss 08-19 @0.62, miss 08-20 @0.54; win 08-18 @0.6). Confidence carries almost no discriminating signal — it's noise, not calibration.
- **Low-confidence days aren't safer either** (0.1 conf still both passes and fails), so the score isn't measuring anything real.

## Fixes to try next
- **Attack the sign problem directly:** add a directional-classifier head and gate predictions on it; stop optimizing APE alone since we're already at baseline magnitude.
- **Cut or floor macro weight** on non-event days; only elevate macro when an actual scheduled macro/earnings event is present.
- **Down-weight technical as ensemble winner** given its 16.7% standalone hit rate; require corroboration before it wins.
- **Add an event/volatility flag** (earnings, macro prints, gap detection) to abstain or widen intervals on high-baseline-ape days where we currently coin-flip.
- **Rebuild confidence calibration** — current scores don't separate wins from losses; retrain confidence against realized directional hits.