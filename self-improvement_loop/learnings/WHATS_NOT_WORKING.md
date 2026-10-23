# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate** (30 of 52 fails, 58%). The system isn't just imprecise on magnitude — it's frequently picking the wrong sign. That's the core defect; a coin flip on direction would do comparably.
- **Fails cluster on high-volatility days.** Every large-APE fail (2.5%+) — 06-22, 06-25, 07-15, 07-30, 07-31 (13%!), 08-19, 10-19 (5.5%), 10-22 (4.5%) — happens when the move is large. On quiet days APE is tiny; the model only breaks when it matters.
- **Barely beats baseline on fails.** mean_fail_ape 2.37% vs comparable baseline; on many misses baseline_ape ≈ or beats our ape. The ensemble adds little edge over naive persistence.
- **macro and technical are dead weight.** macro hit_rate 11.8%, technical 12.9% — both worse than contrarian (25.8%) and news (31%), yet they still win selection and carry ~0.18–0.21 weight.

## Unreliable under these conditions
- **macro-led and technical-led predictions on volatile days** are the worst offenders (07-15, 07-30, 08-19, 06-17 all macro/technical winners with big APE and wrong direction).
- **Confidence is miscalibrated upward over time.** Newer predictions carry rising confidence (0.55–0.66) but still fail frequently (10-06, 10-07, 10-08, 10-09, 10-19 all conf >0.59 and fail). Recent overconfident misses (08-19 0.62, 08-20 0.54, 09-03 0.62) confirm high confidence ≠ accuracy.
- **Large-gap/event days** (earnings-type moves like 07-31) produce catastrophic APE with no regime awareness.

## Fixes to try next
- **Down-weight or bench macro and technical** as lead strategies; shift weight toward news/contrarian. Require a volatility-regime gate before trusting macro/technical.
- **Add a directional accuracy loss/selector** — current selection optimizes magnitude but keeps missing sign. Prioritize strategies by recent directional hit_rate, not just APE.
- **Recalibrate confidence** against realized hit rate; current high-confidence band (0.55–0.66) is not earning its score. Shrink confidence on high-volatility days.
- **Flag event/high-vol days** (widen intervals, defer to baseline) instead of forcing a point call where the system demonstrably collapses.