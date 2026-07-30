# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional errors dominate: 16 of 21 fails are wrong-direction, not just magnitude.** The model can size a move but repeatedly picks the wrong sign. This is the core problem, not calibration.
- Overall directional hit rate is coin-flip-or-worse. Every strategy's standalone hit rate is below 50% — best is news at 30%, worst is macro at 12%.
- The worst APE fails cluster on high-volatility days (2026-06-22 5.1%, 2026-07-23 4.2%, 2026-06-17 3.7%, 2026-06-25 3.6%, 2026-06-15/26/29 ~3.2%). On big-move days the model both misses direction and undershoots magnitude.
- The system rarely beats a naive baseline on fails — most fails have ape ≥ baseline_ape, so the ensemble is actively adding error, not just failing to help.

## Unreliable under these conditions
- **Large true moves (>2.5%):** near-universally missed, usually wrong direction. The model is anchored to small-move regimes.
- **`macro` as winning strategy is a red flag:** 3 of its appearances (06-12? no — 06-08 pass, but 07-13, 07-15 fails) skew bad; 12% hit rate overall. When macro wins the blend, expect a miss.
- **News-weighted days after gaps:** news carries the highest weight (~0.42 on several fails: 07-23, 07-24, 06-17) yet still misses direction — heavy news tilt does not rescue big days.
- Confidence is NOT the problem: overconfident_misses = 0, and fails cluster at low confidence (0.4). If anything the model is under-confident on its rare wins (07-27 conf 0.10 passed).

## Fixes to try next
- Build an explicit **volatility regime gate**: when expected move >2%, widen the band and de-weight the mean-reversion/contrarian and macro strategies.
- **Add a directional meta-model** — the magnitude is roughly OK; a dedicated sign classifier could fix the 16 directional misses.
- Cut or floor `macro` and `contrarian` weights (both <16% hit); redistribute toward news/momentum, and only trust news when a same-day catalyst is confirmed.
- Recalibrate confidence upward on clean setups — current confidence carries no signal (fails and passes both ~0.4).