# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core failure**: 10 of 11 misses are directional, only 1 is magnitude. The model gets the *size* of moves roughly right but calls the *sign* wrong. This is not a calibration-of-magnitude problem, it's a sign problem.
- **We rarely beat baseline on misses**: nearly every failing day also had `beats_baseline=false`. On big-error days (APE 3–5%) we're worse than a naive baseline, meaning we add noise exactly when volatility is high.
- **All strategies are barely-better-than-coin-flip on direction**: best hit rate is news at 33%, everything else 5–28%. Aggregate directional accuracy is dismal. The ensemble has no real directional edge.
- **Confidence is flat and uninformative**: confidence sits in a narrow 0.38–0.58 band regardless of outcome. `overconfident_misses=0` only because confidence never gets high — it's not calibrated, it's just muted. It carries no signal to gate trades.

## Unreliable under these conditions
- **Large-move / high-volatility days** (APE > 3%, e.g. 06-15, 06-17, 06-22, 06-25, 06-26, 06-29): consistent directional misses AND worse than baseline. The model mean-reverts / lags into trend days.
- **Contrarian strategy is the worst offender** (11% hit rate, 1 win as picker) — it's actively harmful and should not be trusted, especially in trending regimes.
- **Macro as a winning strategy** (5.5% hit) — near-zero reliability; only "won" on the single balanced-weight day.
- **News-weighted days that miss** (06-22, 06-25, 06-29, 07-08): heavy news weighting doesn't rescue directional calls on volatile days.
- Passing days cluster at low APE (<0.5%), i.e. quiet, small-range sessions — the model only "works" when nothing happens.

## Fixes to try next
- **Down-weight or drop contrarian and macro**; they drag directional accuracy. Reallocate to news/technical.
- **Add a volatility regime gate**: when expected range is large, either widen intervals or abstain rather than fighting the trend.
- **Rebuild confidence to be outcome-correlated** and use it to size/skip — current confidence is decorative.
- **Directly target sign**: add a trend-following/momentum-persistence feature and validate directional hit-rate as the primary metric, not APE.
- **Baseline-guardrail**: if the ensemble can't beat baseline in backtest for a regime, default to baseline there.