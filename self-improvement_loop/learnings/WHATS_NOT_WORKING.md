# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional errors dominate**: 25 of 38 fails (66%) miss direction, not magnitude. The system is guessing the sign of the day, and losing more than a coin flip on fail days.
- **Overall hit rate is weak**: even the best strategy (news) only picks the direction ~30% of the time as winner; macro and technical are near-random-to-worse (13–16%).
- **Fails cluster in high-move regimes**: nearly every large-APE fail (2.5–13%) is a day AMZN moved sharply and the model faded/missed the move (e.g., 06-22, 06-25, 07-15, 07-31, 08-19). The model systematically under-reacts to big days.
- **Baseline barely beaten on fails**: mean fail APE 2.48% vs baseline ~similar; on many fails we don't even beat naive carry-forward, meaning the ensemble adds noise, not signal.

## Unreliable under these conditions
- **macro as winning_strategy = red flag**: 06-12, 07-13, 07-15, 07-30, 08-19 all fails, several the worst APEs. macro hit_rate 0.13 — routinely wrong and overconfident when it wins.
- **High-volatility / gap days**: any day with baseline_ape >2% is a near-certain miss regardless of strategy.
- **Rising confidence, not rising accuracy (late-Aug/Sep)**: confidence climbed to 0.52–0.72 while fails persisted (08-19 conf .62, 08-31 .72, 09-09 .72, 09-11 .54, 09-17 .54 — all fails). Confidence is drifting up decoupled from outcomes.
- **contrarian on trending days**: fades continuation and eats direction misses (07-21, 08-11, 08-20, 08-24, 09-03).

## Fixes to try next
- **Down-weight or gate macro**: cut macro to near-zero unless a scheduled macro event is present; it's the worst contributor.
- **Add a volatility regime filter**: on predicted high-move days, widen intervals and lean on news/momentum continuation instead of contrarian.
- **Recalibrate confidence**: current high-confidence band (>0.55) has multiple misses (08-19, 08-31, 09-09) — refit confidence against realized hit rate; cap until calibration proven.
- **Directional model separate from magnitude**: since 66% of fails are sign errors, train/評価 a dedicated up/down classifier rather than trusting ensemble point estimate's sign.
- **Promote news, demote technical/contrarian** in weight_hint given news leads on hit rate and lowest MAPE.