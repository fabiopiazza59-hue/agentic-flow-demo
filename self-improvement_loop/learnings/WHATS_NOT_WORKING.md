# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate**: 28 of 45 failures (62%) are wrong-way calls, not just magnitude slips. The system can't reliably call direction — it's near a coin flip on down days.
- **Baseline-beat rate is poor**: failures routinely have ape > baseline_ape (e.g., 06-12, 06-25, 06-26, 09-22), meaning the ensemble actively adds error over a naive persistence baseline on hard days.
- **Big-move days blow up**: the worst APEs (03-31 at 13.2%, 06-22 at 5.1%, 06-17/06-15 at ~3.7%) cluster on high-volatility sessions. The model systematically under-reacts to large gap moves.
- **macro and technical are dead weight**: macro hit_rate 10.8%, technical 13.3% — both below random and both have the highest MAPE. They're dragging the ensemble.

## Unreliable under these conditions
- **High-volatility / large-gap days**: magnitude is consistently too small and direction flips. All of the >3% APE fails are large-move sessions.
- **macro as winning_strategy**: 06-12 region, 07-13, 07-15, 08-19, 08-25 — macro "wins" mostly on losing days. It surfaces when nothing else agrees, and it's usually wrong.
- **Rising-confidence regime (late Sept–Oct)**: confidence climbed to 0.58–0.66 but 10-06/10-07/10-08 still failed. Confidence is trending up while pass rate isn't — **miscalibration is worsening recently**.
- **Overconfident misses (7)**: 08-19 (0.62), 08-20 (0.54), 09-03 (0.62), 09-09 (0.72), 09-11 (0.54), 08-31 (0.72) — high conf on wrong-direction calls, several news/macro/contrarian-led.

## Fixes to try next
- **Cut macro and technical weights hard** (toward 0.05–0.10); reallocate to news (best hit rate 0.33, lowest MAPE). They're the only strategies beating the field.
- **Add a volatility regime gate**: when expected range is large, widen magnitude and down-weight mean-reverting (contrarian) calls that fight the move.
- **Recalibrate confidence**: current high-confidence buckets (>0.6) are not outperforming — refit confidence against realized hit rate; penalize macro/news-led high-conf signals.
- **Track direction separately from magnitude**: build a dedicated direction classifier; 62% of failures are directional, so fixing sign beats shrinking APE.