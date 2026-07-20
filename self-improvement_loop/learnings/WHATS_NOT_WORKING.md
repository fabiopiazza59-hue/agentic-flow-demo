# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core problem.** 12 of 15 fails are directional misses; only 3 are pure magnitude. The model gets the size roughly right but points the wrong way — a sign-prediction failure, not a calibration-of-move failure.
- **Overall dir hit rate ~48%** (a coin flip). We are not adding directional edge over chance.
- **Fails cluster on high-volatility days.** Every large-APE miss (2.6%–5.1%: 06-15, 06-17, 06-22, 06-25, 06-26, 06-29, 07-15) is a directional miss AND fails to beat baseline. On big-move days we're both wrong-way and worse than naive.
- **No single strategy is reliable.** Best is `news` at 28% hit rate; `macro` (12%) and `contrarian` (16%) are actively harmful. `macro` won 3 of 25 yet still carries ~18% weight.
- **Winning-strategy label is misleading:** days "won" by momentum/news still miss directionally, meaning the ensemble blend picks a strategy that was least-wrong, not right.

## Unreliable under these conditions
- **Large daily moves (APE >2.5%):** near-100% failure, always wrong direction. Model has no regime-shift/breakout detection.
- **`macro`- and `contrarian`-led days:** consistently the worst; macro-led 07-13 and 07-15 both large misses.
- **Confidence is flat and uninformative** (mostly 0.38–0.58). Zero overconfident misses only because confidence never rises — it's not calibrated, it's just muted. It gives no signal to trust or discount a call.
- Small quiet days (<1% moves) pass, but often by tying/losing to baseline — low value-add.

## Fixes to try next
- Add a **volatility/regime filter**: on expected-high-move days, widen intervals or abstain rather than commit a direction.
- **Cut `macro` and `contrarian` weight toward zero**; they underperform baseline and drag the blend.
- **Rebuild directional signal separately from magnitude** — treat sign as its own classifier; current blend optimizes size while flipping sign.
- **Make confidence earn its range**: tie it to strategy agreement + recent regime; a persistently ~0.4 confidence is a dead feature.
- Benchmark hard against naive baseline daily; on ~half of fails we lose to it — flag and suppress those setups.