# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- Failure is overwhelmingly **directional**: 13 of 16 fails are wrong-direction, only 3 are magnitude-only. The model can't call which way AMZN moves, not just by how much.
- On failures the model rarely beats baseline: most fails have ape ≥ baseline_ape. It's not just missing, it's underperforming a naive guess when it misses.
- No single strategy is reliable — best (news) hits only 31%, worst (macro) 12%. All five are below coin-flip on direction. This is an ensemble of weak directional signals.
- The biggest-error days cluster (2026-06-15/17/22/25/26/29, 07-15) at ape 3–5%, well above the 2.43% mean fail — these are large moves the model consistently fades or lags.

## Unreliable under these conditions
- **Macro-led** predictions: 3/26 hit rate, highest MAPE (1.80%), and it "won" on outright fails (06-08 aside, 07-13, 07-15). Macro strategy is actively harmful.
- **Contrarian-led** days: 15% hit and it repeatedly loses when a move continues (06-26, 06-18) — the contrarian call is timing tops/bottoms wrong.
- **Large-move regimes** (>3% daily): the model is nearly always wrong-direction, suggesting it defaults to mean-reversion/small-move priors and gets run over by trend/gap days.
- Confidence is flat and low (0.38–0.58) — **no useful calibration signal**. 0 overconfident misses only because confidence is never high; higher-confidence days (0.55 on 06-30, 07-13) still failed.

## Fixes to try next
- Cut macro weight toward zero; it's the worst directional contributor. Redistribute to news/technical.
- Add a **volatility/large-move regime detector**; suppress contrarian and mean-reversion logic when a trend or gap is in force.
- Fix directional logic first — magnitude is fine; build a dedicated up/down classifier separate from the price-level regression.
- Make confidence meaningful: currently it's noise. Widen the range and backtest so high confidence actually predicts hits before using it to size or gate.