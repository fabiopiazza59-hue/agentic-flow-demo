# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core problem**: 27 of 44 fails are directional misses. The model gets the move size roughly right but points the wrong way too often. On quiet days APE is tiny; the fails cluster on real moves.
- **Big-move days are routinely blown**: the worst APEs (0.13 on 2026-07-31, 0.05 on 06-22, 0.038–0.042 on 06-17/07-30/07-23) all come on large real moves. The system is essentially a low-vol mean reverter that cannot size shocks.
- **Beating baseline is coin-flip at best**: many fails (06-12, 06-15, 06-17, 06-25, 06-26, 07-13, 09-22) have ape > baseline_ape, meaning the ensemble actively *hurts* vs. a naive last-price carry.
- **macro and technical are dead weight**: macro hit_rate 0.11, technical 0.13 — both below random. Yet they still carry ~0.18–0.21 weight. Every day macro "wins" the blend (06-12, 07-13, 07-15, 08-19, 08-30) it misses.

## Unreliable under these conditions
- **Macro-led days**: near-universal failure (06-12, 07-13, 07-15, 08-19, 07-30). macro is the worst strategy and should rarely drive the call.
- **High-volatility / gap days**: errors blow out 5–20x normal; directional sign flips.
- **Late-period overconfidence**: confidence has crept to 0.6–0.72 while fails persist (08-19 conf 0.62 fail, 08-31 conf 0.72 fail, 09-09 conf 0.72 fail, 09-11 conf 0.54 fail, 10-06 conf 0.66 fail). 6 overconfident misses — calibration is drifting upward without accuracy support.
- **Momentum in reversals**: momentum "wins" then eats the turn (06-12, 06-25, 07-08, 08-07).

## Fixes to try next
- **Cut macro and technical weight toward ~0.05–0.10**; reallocate to news (best: 0.33 hit, lowest MAPE).
- **Add a volatility/regime gate**: widen intervals and lower confidence on high-vol or earnings/gap days instead of committing to a point direction.
- **Recalibrate confidence** — recent 0.6–0.72 confidences are not earned; cap confidence until directional hit_rate on that regime exceeds ~0.5.
- **Build a direction-specific ensemble** separate from magnitude; the magnitude model is fine, the sign vote is broken.