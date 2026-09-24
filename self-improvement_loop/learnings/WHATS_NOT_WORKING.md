# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core problem.** 26 of 41 fails are directional misses (63%). The model gets the size roughly right but is on the wrong side of the tape — a sign it's lagging turning points, not mis-scaling.
- **Failures cluster on high-volatility / large-move days.** Nearly every large-APE fail (2.5–5%+, and the 13% blowup on 2026-07-31) coincides with baseline_ape also being large. On calm days it passes; on big-move days it whiffs, meaning it can't handle the days that actually matter.
- **We rarely beat the baseline on fails.** Most misses have ape ≥ baseline_ape — we add error rather than signal precisely when conditions are hard.
- **Macro is a net liability.** hit_rate 0.125, worst MAPE (0.0171). When macro "wins" the selection it frequently fails (07-13, 07-15, 07-30, 08-19). It should not be trusted as a lead strategy.

## Unreliable under these conditions
- **Macro-led days:** lowest hit rate, highest error — the least reliable strategy, especially into large moves.
- **Technical-led and contrarian-led days:** hit rates 0.15 / 0.18, both below coin-flip; contrarian repeatedly gets direction wrong in trending markets (08-11, 08-20, 08-24, 09-22).
- **Confidence is mildly miscalibrated on the high end.** Only 4 flagged overconfident misses, but note real high-confidence fails: 0.72 (08-31), 0.72 (09-09), 0.62 (08-19, 09-03), 0.62 (09-17-adjacent). Confidence ≥0.6 is NOT delivering reliably better outcomes — calibration is flat, not monotonic.
- **Empty-weights days** (07-22, 09-17, 09-22, etc.) skew toward fails — degraded/fallback ensemble state is unreliable.

## Fixes to try next
- **Cut macro to a confirmation-only role**; stop letting it be the winning strategy. Redistribute weight toward news (best hit rate 0.32, lowest MAPE).
- **Add a volatility regime gate:** on high-expected-move days, widen intervals and lower confidence; the current model has no edge there.
- **Attack the directional bias directly** — add a trend/turning-point filter so contrarian isn't fired mid-trend.
- **Recalibrate confidence** against realized hit rate; current ≥0.6 bucket is unjustified. Force high confidence only when multiple strategies agree.
- **Fix the empty-weights fallback path** — those days fail disproportionately.