# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core problem.** 27 of 42 fails are directional misses (64%). Mean fail APE is only 2.39% — the model is close on size but picks the wrong sign repeatedly. We're guessing the daily turn poorly.
- **The ensemble barely beats baseline.** On fail days APE ≈ baseline APE almost every time; many "passes" also lose to baseline (e.g., 06-16, 07-14, 08-06, 08-27/28, 10-01, 10-05). On flat days we add noise, not signal.
- **Macro is the weakest winner.** When macro is the winning strategy it fails hard (06-12 region, 07-13, 07-15, 07-30, 08-19, 08-25) — macro hit_rate 0.11, worst MAPE 0.0163. Yet it still gets ~18% weight.
- **Big-miss days are regime breaks, not model errors.** 07-31 (APE 13%!), 06-22/25/26, 08-19/20 cluster around large moves the model never sized — it stays near prior close while price gaps.

## Unreliable under these conditions
- **High-volatility / large-move days:** any day the true move >3% gets both direction and magnitude wrong; the model is anchored to prior close.
- **Macro- and technical-led days:** technical hit_rate 0.14, macro 0.11 — these two "win" the ensemble but lose the prediction. They are dragging the blend.
- **Confidence is weakly calibrated but not wildly overconfident:** only 4 overconfident misses. However the recent high-confidence era (0.52–0.63, late Sep–Oct) still throws directional misses (10-01, 10-05, 09-22, 09-11) — rising confidence is NOT tracking rising accuracy. Confidence has drifted up with no accuracy gain.

## Fixes to try next
- **Cut macro and technical weight** toward their hit_rate (0.11/0.14); lean on news (0.325) and contrarian (0.225). Current weight_hints over-reward losers.
- **Add a direction-focused sub-model / sign gate** — most error value is in the sign, not the size. Score and optimize directional hit explicitly.
- **Detect high-vol regime** (ATR / gap signal) and widen predicted move or defer to baseline; stop anchoring to prior close on breakout days.
- **Recalibrate confidence** against realized hit rate; the late-sample confidence inflation (0.5→0.63) is unearned.
- **Add a "don't beat baseline → abstain" rule** for low-signal flat days where we currently just add noise.