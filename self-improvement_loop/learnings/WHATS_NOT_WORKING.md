# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- 55% fail rate with directional misses (30) slightly outnumbering magnitude misses (25) — the model is wrong on *direction* more than on size, meaning the core signal, not calibration of magnitude, is the deeper problem.
- Big-move days blow up: APEs of 13.2%, 5.5%, 5.2%, 5.1%, 4.5%, 4.2% all cluster on volatile/gap sessions. The model systematically under-reacts to large moves (mean fail APE 2.42% vs typical ~0.5% on wins).
- On many fails the model loses to the naive baseline too (2026-06-12/15/17/25/26, 10-15, 10-19) — it isn't just hard days, the model actively adds error.
- `macro` is the recurring winning strategy on the worst misses (07-30 3.9%, 08-19 2.9%, 10-30 5.2%) despite being the weakest overall.

## Unreliable under these conditions
- **macro** (hit rate 12%, worst MAPE 1.70%) and **technical** (12%) are near coin-flip-bad; they "win" the ensemble mostly on high-volatility days and then miss badly.
- **momentum** (17%) fails in choppy/mean-reverting regimes — see the June cluster of consecutive misses where momentum/news were selected.
- Large single-day gaps (earnings/macro-print days) — every APE >3% is a regime the ensemble cannot handle.
- Confidence is miscalibrated upward over time: later predictions carry 0.5–0.72 confidence yet still fail (08-19 conf .62, 08-31 .72, 09-09 .72, 10-06 .66, 10-08 .62). 7 overconfident misses are concentrated in the high-confidence band.

## Fixes to try next
- Cut or cap `macro` and `technical` weight; they drag MAPE and win rarely. Lean on `news`/`contrarian` (the only strategies >25% hit rate).
- Add a volatility/gap detector; on high-vol days widen predicted magnitude and distrust momentum.
- Recalibrate confidence — current scores above 0.5 don't predict passes; shrink confidence toward empirical ~0.45 base rate and penalize overconfident regimes.
- Add a baseline-guard: when the ensemble diverges far from naive persistence, fall back toward baseline to stop sub-baseline losses.