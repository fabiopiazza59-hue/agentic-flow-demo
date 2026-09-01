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

## 2026-07-13 — FAIL (APE 1.40%)
- Predicted 243.85 vs actual 247.31 (prior 245.34); dir hit: False; beat baseline: False; closest analyst: macro.
**What went wrong:** Consensus-down blend (4/5 analysts) missed a +0.8% rally by 1.4%; baseline (prior close) beat us. **Root cause:** News, at 40% weight, anchored on an Iran-Hormuz 'futures down 1.2%' narrative that was either unconfirmed or already discounted — no analyst verified the premarket gap actually persisted, yet everyone repeated 'no gap cushion' while still predicting decline. The one contrarian-to-the-group voice (macro, up) was buried at 7% despite being the only correct direction. **The one change:** Require news-driven directional calls to be corroborated by an actual confirmed premarket gap in AMZN itself; if the cited move is index-level and unconfirmed, halve news weight and let momentum's positive 5d/20d trend set the directional lean.

## 2026-07-15 — FAIL (APE 2.98%)
- Predicted 247.35 vs actual 254.96 (prior 247.49); dir hit: False; beat baseline: False; closest analyst: macro.
**What went wrong:** The blend was directionally wrong AND magnitude-timid — all five analysts predicted within 245.8–248.75 while price closed at 254.96, a >3% up move nobody sized. News and technical (jointly 48% weight) both said 'down' off an RSI-overbought / below-SMA50 thesis. **Root cause:** groupthink on the SMA-50 resistance level (~253) as a hard ceiling; the desk repeatedly read RSI 66.89 as mean-reversion pressure when momentum (positive 5d/20d, above SMA5/20) was actually confirming a breakout. The macro analyst was the only one directionally right yet carried minimum weight. **One change to try:** when the bullish/directionally-correct analyst is momentum or macro but sits at min weight, and price is pressing a well-flagged resistance with positive multi-day returns, add an explicit breakout scenario that lets the prediction exceed the resistance level rather than pinning to it.

## 2026-07-16 — FAIL (APE 1.60%)
- Predicted 253.9 vs actual 249.89 (prior 254.96); dir hit: True; beat baseline: True; closest analyst: contrarian.
**What went wrong:** Directionally correct but magnitude-short by 4pts — the reversion was ~2% and the whole panel underestimated it, with momentum/macro (both 255.80, 0.32 combined weight) pinning the blend near prior close. **Root cause:** Two analysts anchored to 'strong close yesterday = continuation' despite every technical signal (RSI 72, +5.6% over SMA20, red Nasdaq futures) screaming stretched-and-selling; the meta-weights gave momentum+macro too much voice in a textbook mean-reversion regime, and even contrarian under-sized the move. **One change to try:** In confirmed overbought-reversion setups (RSI>70 + extended + negative futures), down-weight momentum/macro to near-zero and shift the blend toward min(analyst estimates) minus a reversion-overshoot buffer of ~0.5-1%.

## 2026-07-20 — FAIL (APE 1.24%)
- Predicted 246.9 vs actual 249.99 (prior 247.23); dir hit: False; beat baseline: False; closest analyst: news.
**What went wrong:** The blend was directionally wrong (predicted −0.13%, actual +1.12%), and it lost to the naive baseline. **Root cause:** correlated-analyst herding — technical, momentum, macro, and contrarian all recycled the same 'overbought fade below SMAs' story and held 72% of weight, so the ensemble had no real diversity. The single independent view (news, citing +1% Nasdaq futures on Iran diplomacy and pre-earnings drift) nailed both direction and level but was structurally outvoted. The desk mistook consensus for confidence when it was actually redundancy. **The one change:** detect thesis-collinearity across analysts (e.g., cluster by rationale keywords/direction) and down-weight clusters so any single-source cluster caps at ~40% combined — this would have lifted news' effective weight and pulled the prediction toward 248+, likely beating baseline and hitting direction.

## 2026-07-21 — FAIL (APE 1.45%)
- Predicted 251.15 vs actual 247.55 (prior 249.99); dir hit: False; beat baseline: False; closest analyst: contrarian.
- **What went wrong:** wrong direction; final blend 251.15 missed by 1.45%.
- **Likely culprit:** analyst `news` was furthest from actual and pulled the blend.
- **Try next:** reduce weight on `news` under today's conditions and lean on `contrarian`.

## 2026-07-22 — FAIL (APE 1.12%)
- Predicted 247.59 vs actual 244.85 (prior 247.55); dir hit: False; beat baseline: False; closest analyst: news.
## What went wrong
Three analysts (technical 248.47, macro 248.80, momentum 248.10) all leaned on the same stale 'consolidation, mild drift up' read and collectively pulled the blend to a near-flat 247.59, while the actual close fell 1.09% to 244.85. News (245.60) and contrarian (246.80) both nailed the down direction and were far closer, but their votes were diluted.

**Root cause**: the blend treated a live macro catalyst (Nasdaq futures -0.6%, oil surge, rate-hike odds) as equal to three variations of the same 'range-bound' technical thesis — effectively triple-counting a correlated bullish view and burying the one analyst reading the actual news tape.

**One change to try**: when news flags a concrete same-day risk-off catalyst and disagrees with a cluster of correlated technical/momentum/macro bulls, cap the correlated-bull contribution and shift weight toward news (its scorecard already leads on hit_rate and mape). Test: would news-weighted blend have passed here? Yes (~0.30% ape).

## 2026-07-23 — FAIL (APE 4.21%)
- Predicted 243.5 vs actual 233.66 (prior 244.85); dir hit: True; beat baseline: True; closest analyst: news.
**What went wrong:** Directional call was correct but the desk anchored to prior-close support (241-244) and got the magnitude wrong by ~$8. All five analysts clustered tightly, giving false consensus confidence; the news analyst had the right story (stacked bearish catalysts, live down print) but still under-shot because it treated its mid-session ~$241.62 as a floor rather than a midpoint of an accelerating decline. **Root cause:** systematic support-anchoring — analysts assumed the 242-244 zone would hold, but in a multi-catalyst risk-off tape support gives way and selling accelerates into the close. **The one change:** when news flags 2+ compounding negative catalysts on a confirmed-down session, apply a downside-skew multiplier that projects continued intraday velocity past nearby support instead of parking the estimate at it — test by extrapolating the last-hour rate of change rather than clamping to the prior close.

## 2026-07-24 — FAIL (APE 1.20%)
- Predicted 234.9 vs actual 232.11 (prior 233.66); dir hit: False; beat baseline: False; closest analyst: momentum.
**What went wrong:** Meta-judge over-weighted news (0.42) and technical (0.28) on a mean-reversion 'bounce off 233 support' story, producing a +1.24 up-call while the stock closed -1.55 down — a clean directional miss that also lost to baseline. **Root cause:** The news analyst leaned on a soft premarket '+0.5%' signal to override a strong, confirmed downtrend (1d -4.57%, 5d -6.5%, below all SMAs, distribution volume). Two analysts (momentum, macro) correctly read continuation but held only 15% combined weight. The desk systematically underweights momentum despite its 231.5 call being closest to the 232.11 actual. **The one change:** In confirmed downtrends (price below 5/20/50 SMA + negative 5d return), demote premarket-gap-based bull cases to tiebreaker status and floor momentum weight at 0.25 — the trend was the signal, and we faded it.

## 2026-07-29 — FAIL (APE 1.72%)
- Predicted 230.55 vs actual 226.65 (prior 230.86); dir hit: True; beat baseline: True; closest analyst: macro.
**What went wrong:** Directionally right but magnitude-short — we predicted a −0.13% drift when AMZN fell −1.82%. The blend was pulled up by news (+0.85%) and contrarian (+0.71%) betting on a pre-earnings oversold bounce that failed to materialize; together they held 40% weight against a unanimous bearish trend. **Root cause:** Over-weighting mean-reversion (news/contrarian, both hit-rates ~15-30% and MAPE ~1.5-1.6%) on the eve of an earnings print, when pre-earnings positioning drifted the stock DOWN not up, and under-weighting macro (228.50) which best captured the ongoing 4-day decline. RSI 35 was treated as a bounce trigger but a stock can stay oversold and keep falling into a catalyst. **One change:** On earnings-eve sessions with a confirmed downtrend, demote the contrarian/news bounce thesis and let the trend cluster (macro/technical/momentum) set the magnitude — target closer to the most-bearish analyst, not the prior close.

## 2026-07-30 — FAIL (APE 3.89%)
- Predicted 226.35 vs actual 235.5 (prior 226.65); dir hit: False; beat baseline: False; closest analyst: macro.
## Failure reflection
**What went wrong:** We shipped a near-flat 226.35 into a +3.9% rally, missing both direction and magnitude. Momentum (0.34 weight) was the only bearish voice and it dominated the blend, overriding four analysts who correctly called 'up' off a 27.4 RSI and -7.4% 5-day slide.
**Root cause:** The meta-judge over-trusted trailing-trend momentum at a statistical extreme — exactly where trend-continuation is weakest. Momentum's scorecard (23% hit, 0.017 MAPE) didn't justify its 0.34 weight, and the oversold mean-reversion signal was structurally underweighted. Even the bulls capped upside at +1.2%, so the desk never had a path to +3.9%.
**One change to try:** Add a regime override — when RSI < 30 and price > 6% below SMA20, force momentum weight down to ≤0.15 and raise contrarian/macro; test whether this recovers directional hits on oversold-bounce days without hurting trending days.

## 2026-07-31 — FAIL (APE 13.21%)
- Predicted 235.7 vs actual 271.5799865722656 (prior 235.5); dir hit: True; beat baseline: True; closest analyst: news.
## What Went Wrong

**Direction correct, magnitude catastrophically short.** We predicted +0.2 into a +36 move. The single analyst who correctly read the day — news, citing an ~11% earnings gap to ~$263 with 0.78 confidence — was buried at 0.12 weight, while momentum (0.28) and technical (0.27) dominated the blend with pre-gap chart levels (SMA5=231, RSI=40) that the earnings report had already invalidated.

**Root cause:** The meta-judge applied normal-regime weighting to an earnings-day regime. It rewarded low-MAPE chart analysts whose signals are structurally useless the morning after a blowout print, and discounted the one analyst pricing the actual overnight news.

**The one change:** Add an earnings-gap regime detector — when news flags a >5% premarket gap at high confidence, hard-floor news weight to 0.6+ and cap momentum/technical to 0.1 each. This one fix likely turns 13.2% APE into ~3%.

## 2026-08-04 — FAIL (APE 2.26%)
- Predicted 283.69 vs actual 277.42 (prior 284.02); dir hit: True; beat baseline: True; closest analyst: contrarian.
## What went wrong
The desk got direction right (down) but massively under-shot magnitude: predicted -0.12%, actual -2.32%. The blend was diluted by momentum's contrarian +284.5 up-call carrying 0.24 weight — the second-highest — despite four analysts unanimously flagging exhaustion at a 252-week high after a +22% surge with a $4.07B Bezos insider-sale overhang.

**Root cause:** the ensemble averaged toward the prior close instead of respecting a strong consensus overextension signal; momentum's low-conviction (0.45) up-call should never have offset four down-calls, two of which (news, contrarian) targeted 279-280 — much closer to the 277.42 outcome.

**One change to try:** when analysts show ≥80% directional agreement backed by a concrete supply catalyst, drop the dissenting analyst's weight to near-zero and anchor the blend to the median of the agreeing down-targets (~280.1), not the confidence-weighted mean.

## 2026-08-05 — FAIL (APE 1.34%)
- Predicted 276.3 vs actual 272.65 (prior 277.42); dir hit: True; beat baseline: True; closest analyst: contrarian.
- **What went wrong:** right direction, magnitude off; final blend 276.3 missed by 1.34%.
- **Likely culprit:** analyst `news` was furthest from actual and pulled the blend.
- **Try next:** reduce weight on `news` under today's conditions and lean on `contrarian`.

## 2026-08-10 — FAIL (APE 1.37%)
- Predicted 274.29 vs actual 278.09 (prior 274.48); dir hit: False; beat baseline: False; closest analyst: news.
**What went wrong:** Directional AND magnitude fail — blend said flat-to-down (274.29) vs actual +1.31% (278.09), missing baseline too. The bearish cluster (contrarian/macro/technical/momentum all ≤273.8) dragged the blend below prior close, when the correct read was a bull-flag breakout above the 272-278 chop. **Root cause:** every analyst anchored to the same 'RSI 62.93 + extended above SMA20' mean-reversion thesis; news correctly saw upside catalysts (Zoox launch, positive futures) but capped itself at SMA-5 276.17 despite noting price already tagged 278.31. **One change:** when analyst directions are ≥80% one-sided on pure technical-extension logic (no fresh catalyst), cap the crowd's weight and let the single catalyst-driven (news) analyst set the directional sign, then extend its target past the noted intraday high rather than anchoring to SMA-5.

## 2026-08-11 — FAIL (APE 2.16%)
- Predicted 278.15 vs actual 272.27 (prior 278.09); dir hit: False; beat baseline: False; closest analyst: contrarian.
**What went wrong:** The blend was flat (278.15) on a day AMZN fell to 272.27. News (0.34) and momentum (0.26) both called 'up' on trend/post-earnings momentum and dominated, drowning out contrarian and technical which correctly called 'down' from an overextended RSI 66 / +10% above SMA20 setup. **Root cause:** Meta-judge over-trusted news despite its rationale being anchored to a stale intraday +1.6% quote ('trading at 278.85') that had no forward information — a lookahead-flavored anchor, not a catalyst. Momentum reward for a strongly-stretched name ignored that stretch is precisely the mean-reversion risk the contrarian flagged. **The one change:** When RSI>65 AND price >8% above SMA20, down-weight momentum and news by half and let contrarian/technical drive; test this gate on the next 10 extended-stretch days.

## 2026-08-12 — FAIL (APE 1.77%)
- Predicted 272.01 vs actual 267.28 (prior 272.27); dir hit: True; beat baseline: True; closest analyst: contrarian.
**What went wrong:** Unanimous down-call was right on sign but the blend (-0.10%) captured a fraction of the actual -1.83% drop. The panel clustered in a 2-point band anchored to a '270-271 support' that failed to hold, and the meta-blend averaged toward prior-close.

**Root cause:** Consensus support-level anchoring in a 4%-vol regime — analysts named the same floor as a target rather than a level that could break, so magnitude was systematically compressed toward zero.

**One change to try:** When direction is unanimous, override the blend to at least 0.5x the trailing realized-vol move in that direction rather than snapping to the nearest cited support; test whether magnitude-scaling on high-conviction consensus days reduces MAPE without hurting the (already correct) hit rate.

## 2026-08-19 — FAIL (APE 2.89%)
- Predicted 258.15 vs actual 265.84 (prior 259.45); dir hit: False; beat baseline: False; closest analyst: macro.
**What went wrong:** The blend was directionally and magnitudinally wrong — predicted -0.5%, actual +2.46%. Bearish herding (4/5 analysts within 258±0.7) drowned out the lone correct up-call from macro, which had the right thesis (oversold snapback) but was penalized to 7% weight for its poor historical record.

**Root cause:** A -4.7% 5-day decline was treated as trend-continuation evidence by momentum/technical/contrarian/news alike, when it was actually setting up an oversold bounce that overshot. The desk had no mechanism to detect that unanimous bearishness itself was the contrarian tell.

**One change to try:** Add a 'consensus-crowding' override — when analyst predictions cluster within 1% AND the setup is a multi-day oversold decline, cap the consensus direction's weight and boost the dissenting mean-reversion analyst, testing whether crowded bearish agreement systematically precedes reversals.

## 2026-08-20 — FAIL (APE 2.23%)
- Predicted 265.91 vs actual 260.11 (prior 265.84); dir hit: False; beat baseline: False; closest analyst: contrarian.
**What went wrong:** We predicted +0.07pt (flat) into a -2.16% drop — both direction and magnitude wrong. The blend was dragged bullish by news+macro+momentum (62% weight, all up), while the two down-calls (contrarian, technical) that correctly read mean-reversion were under-weighted at 38%.

**Root cause:** The +6.39pt prior-session bounce was mistaken for durable momentum when it was a corrective bounce inside a downtrend (5d return negative, extended vs SMAs, RSI 66.6). News catalysts (Rosenblatt, drone) were priced as fresh upside but were already stale/discounted. Even contrarian's 263.5 undershot 260.11 — nobody sized the reversal aggressively enough.

**One change:** When a >2% single-day bounce occurs after a multi-day selloff with RSI>65 and price >5% above SMA50, boost contrarian+technical combined weight above 0.50 and discount same-day news initiations as non-catalytic. That single tilt would have pulled the blend toward 263.5 and cut the error nearly in half.

## 2026-08-24 — FAIL (APE 1.37%)
- Predicted 258.49 vs actual 262.07 (prior 258.63); dir hit: False; beat baseline: False; closest analyst: contrarian.
**What went wrong:** The blend predicted a flat-to-down close (258.49) into a +1.33% up day. Contrarian was the only analyst with the right thesis and magnitude (262.5 vs 262.07) but its 0.18 weight was swamped by news (0.26) and momentum (0.24), both of which extrapolated the recent downtrend and ignored the RSI-24 exhaustion signal. **Root cause:** the meta-weighting rewards long-run MAPE (news/momentum score well on average) but is blind to regime — deeply oversold conditions are exactly when trend-following breaks and mean-reversion pays, yet the desk gave the mean-reversion voice minority weight AND ran low confidence (0.36) instead of leaning in. Also note 4-of-5 analysts said 'up' but the blend printed down — a majority-direction override was absent. **One change:** add a regime gate — when RSI<25 and analyst-direction majority is 'up', dynamically boost contrarian weight to at least 0.30 and demote momentum, rather than using static MAPE-based weights.

## 2026-08-31 — FAIL (APE 2.35%)
- Predicted 265.87 vs actual 259.77 (prior 266.43); dir hit: True; beat baseline: True; closest analyst: news.
**What went wrong:** Directionally correct (all down/flat) but magnitude collapsed — predicted -0.2%, actual -2.5%. **Root cause:** Four of five analysts anchored to prior_close and SMA20, framing a genuine breakdown as harmless 'profit-taking'; the RSI=38.71 'oversold, room to bounce' narrative created a false floor and momentum even voted up. Only news read the live tape (~$261 mid-session drag) and it was still too conservative. The blend averaged real information (news) against four anchored guesses. **The one change:** On days where the news analyst is already quoting a live intraday price materially below prior_close, treat that as a hard prior — set the floor near news's estimate and cut mean-reversion analysts' weight in half rather than averaging them in.
