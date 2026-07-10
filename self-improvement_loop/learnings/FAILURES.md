# Failure Log — AMZN Close Predictor

Every missed prediction (>1% error), newest at the bottom. Reread before each shot.

## 2026-06-12 — FAIL (APE 2.01%)
- Predicted 243.35 vs actual 238.55 (prior 241.51); dir hit: False; beat baseline: False; closest analyst: momentum.
- **What went wrong:** wrong direction; final blend 243.35 missed by 2.01%.
- **Likely culprit:** analyst `contrarian` was furthest from actual and pulled the blend.
- **Try next:** reduce weight on `contrarian` under today's conditions and lean on `momentum`.

## 2026-06-15 — FAIL (APE 3.40%)
- Predicted 237.65 vs actual 246.02 (prior 238.55); dir hit: False; beat baseline: False; closest analyst: news.
**What went wrong:** We blended toward continuation-down (momentum/macro at 0.60 combined) when every structural signal screamed exhaustion: RSI 26.76, ~7% below SMA20, -10.7% over 20d. Oversold extremes don't continue linearly — they snap back, and a U.S.-Iran peace deal lifting Nasdaq futures ~2% was the exact catalyst news flagged. **Root cause:** the meta-weights rewarded trend-following analysts in a regime where trend-following is structurally wrong (deep oversold + exogenous positive catalyst), and news — despite the best track record (MAPE 1.06%, weight_hint 0.31) — was under-weighted at 0.12. We trusted recency-of-trend over signal-of-reversal. **One change:** add a regime gate — when RSI < 30 and a fresh market-moving catalyst is detected, cut momentum+macro weight in half and floor news+contrarian at the blend majority. Had we done so here, the blend lands near 242-244, halving the error and catching direction.

## 2026-06-17 — FAIL (APE 3.73%)
- Predicted 246.35 vs actual 237.5 (prior 246.0); dir hit: False; beat baseline: False; closest analyst: technical.
- **What went wrong:** wrong direction; final blend 246.35 missed by 3.73%.
- **Likely culprit:** analyst `contrarian` was furthest from actual and pulled the blend.
- **Try next:** reduce weight on `contrarian` under today's conditions and lean on `technical`.

## 2026-06-18 — FAIL (APE 2.66%)
- Predicted 237.9 vs actual 244.39 (prior 237.5); dir hit: True; beat baseline: True; closest analyst: contrarian.
- **What went wrong:** right direction, magnitude off; final blend 237.9 missed by 2.66%.
- **Likely culprit:** analyst `macro` was furthest from actual and pulled the blend.
- **Try next:** reduce weight on `macro` under today's conditions and lean on `contrarian`.

## 2026-06-22 — FAIL (APE 5.08%)
- Predicted 244.62 vs actual 232.79 (prior 244.39); dir hit: False; beat baseline: False; closest analyst: news.
**What went wrong:** The blend predicted flat (+0.09%) against a -4.75% drop — wrong direction and far too small in magnitude. Contrarian, macro, and momentum all bet on an RSI-31 oversold bounce and collectively held the blend near prior close. **Root cause:** mistaking a deep-downtrend extension for a mean-reversion setup; 'oversold' was treated as a buy signal when the stock was in sustained decline below both SMA20 and SMA50 amid an active Nasdaq selloff. The only analyst reading the tape correctly (news, down) was under-weighted relative to its strong 0.0198 MAPE track record. **One change to try:** Add a regime gate — when price is >3% below SMA20 and a sector-selloff catalyst is live, cap the combined weight of contrarian+macro and shift it to the news analyst, whose lower MAPE and directional reads have been the desk's best signal.

## 2026-06-25 — FAIL (APE 3.56%)
- Predicted 235.1 vs actual 227.01 (prior 234.27); dir hit: False; beat baseline: False; closest analyst: momentum.
**What went wrong:** We predicted +0.35% into a -3.1% drop — wrong direction and magnitude, 8 points off. **Root cause:** News (0.34 weight) sold an unrealized Nasdaq-futures beta bounce while dismissing its own hot-PCE warning, and contrarian+macro piled on a mean-reversion bounce that never came against a strongly confirmed downtrend; the two correct down-callers (momentum, technical) were collectively underweighted at 0.38. The blend bought a one-day catalyst narrative over weeks of price evidence. **The one change:** In confirmed downtrends (ret_20d < -8%, below all SMAs), require news-driven bullish reversals to clear a higher confidence bar and structurally cap their combined weight below trend-following analysts — don't let a futures-pop story override a falling knife.

## 2026-06-26 — FAIL (APE 3.18%)
- Predicted 225.3 vs actual 232.69 (prior 227.01); dir hit: False; beat baseline: False; closest analyst: contrarian.
## What Went Wrong

We predicted 225.3 vs actual 232.69 (3.18% APE, missed baseline too). The miss was both directional and magnitude: four of five analysts called 'down' into a deeply oversold tape, and the meta-judge rewarded that consensus by piling weight onto news (0.32) and momentum (0.24). The contrarian analyst nailed it at 231.5 but carried only 0.10 weight.

**Root cause**: regime misclassification. With RSI at 29.55 and price ~7.75% below SMA20, the setup was a classic exhaustion/mean-reversion zone, but our weighting treated it as trend-continuation. The bearish analysts kept citing the SAME inputs (RSI oversold, no gap, -16.5% over 20d) and reached opposite conclusions — a sign of narrative anchoring rather than independent signal.

**One change to try**: implement a hard regime gate — when RSI < 30, force minimum 0.30 weight on the contrarian analyst and cap combined momentum+news weight at 0.40. Backtest whether this flips the directional hit rate on oversold days.

## 2026-06-29 — FAIL (APE 3.26%)
- Predicted 232.3 vs actual 240.14 (prior 232.69); dir hit: False; beat baseline: False; closest analyst: news.
**What went wrong:** Four of five analysts anchored on a downtrend (RSI 39, price below SMA20/50) and clustered at 230–234, but those signals were backward-looking and already reflected in price; AMZN gapped up and closed +3.2%. The blend missed both direction and magnitude by ~7.8 points.

**Root cause:** The news analyst was *reporting realized intraday tape* (+3.8% live) — effectively ground truth — yet was capped at 0.20 weight while technical/momentum/macro opinion dominated at 0.59. We systematically discount the only analyst observing actual same-day price action, despite it leading every scorecard metric.

**One change to try:** When news cites a confirmed gap + intraday percentage move, anchor the prediction to its estimate (weight floor 0.40) and demote pure-technical bearishness; back-test whether this would have flipped this and similar gap-day misses.

## 2026-06-30 — FAIL (APE 1.05%)
- Predicted 240.85 vs actual 238.34 (prior 240.14); dir hit: False; beat baseline: False; closest analyst: technical.
- **What went wrong:** wrong direction; final blend 240.85 missed by 1.05%.
- **Likely culprit:** analyst `news` was furthest from actual and pulled the blend.
- **Try next:** reduce weight on `news` under today's conditions and lean on `technical`.

## 2026-07-01 — FAIL (APE 1.47%)
- Predicted 238.15 vs actual 241.7 (prior 238.34); dir hit: False; beat baseline: False; closest analyst: news.
- **What went wrong:** wrong direction; final blend 238.15 missed by 1.47%.
- **Likely culprit:** analyst `macro` was furthest from actual and pulled the blend.
- **Try next:** reduce weight on `macro` under today's conditions and lean on `news`.

## 2026-07-08 — FAIL (APE 1.04%)
- Predicted 246.15 vs actual 243.62 (prior 245.98); dir hit: False; beat baseline: False; closest analyst: news.
**What went wrong:** Directional miss — we predicted a modest up-drift into a broad risk-off day. Momentum and macro both parroted the same 'positive 5d trend, RSI ~50, continuation' rationale and together held 0.42 weight, anchoring the blend near flat. **Root cause:** The meta-judge weighted on backward-looking scorecard MAPE and momentum's raw hit count, ignoring that news had the highest hit_rate (0.33) AND a specific, dated catalyst the others literally could not see (they were re-reading yesterday's SMAs). Four of five analysts recycled identical technical priors, creating false consensus. **One change:** Add a catalyst-override rule — if the news analyst cites a quantified same-session macro shock (futures/oil move) with conf ≥0.6, dynamically boost its weight to ≥0.35 and haircut pure trend-continuation analysts by half; that single reallocation moves the blend from 246.15 toward ~244.3, flipping direction and beating baseline.

## 2026-07-09 — FAIL (APE 1.28%)
- Predicted 243.87 vs actual 247.04 (prior 243.62); dir hit: True; beat baseline: True; closest analyst: news.
- **What went wrong:** right direction, magnitude off; final blend 243.87 missed by 1.28%.
- **Likely culprit:** analyst `contrarian` was furthest from actual and pulled the blend.
- **Try next:** reduce weight on `contrarian` under today's conditions and lean on `news`.
