# Living Strategy — AMZN Close Predictor

Append-only insights. Newest at the bottom.

- (2026-06-08) On low-realized-volatility days (small expected |move|), shrink momentum's magnitude toward prior_close and lean on macro, since momentum tends to overshoot when the actual move is muted.
- (2026-06-12) 2026-06-12: FAIL, did not beat baseline; closest momentum, worst contrarian.
- (2026-06-15) When RSI < 30 AND a same-day risk-on news catalyst is present, the news/contrarian analysts should override momentum/macro continuation calls — test inverting the weight stack in deep-oversold + catalyst regimes.
- (2026-06-16) On flat-gap, oversold-but-not-extreme days (RSI 33-38, no premarket gap), default toward prior_close rather than directional tilt — test whether |predicted - prior_close| < 0.2% beats baseline more often than directional bets.
- (2026-06-17) 2026-06-17: FAIL, did not beat baseline; closest technical, worst contrarian.
- (2026-06-18) 2026-06-18: FAIL, beat baseline; closest contrarian, worst macro.
- (2026-06-22) When 3+ analysts cite 'oversold RSI' as a bounce thesis while price is already >3% below SMA20 AND news flags an active sector selloff, suppress the mean-reversion analysts and lean toward the news/momentum-down read — oversold can stay oversold in a downtrend.
- (2026-06-23) When a real-time intraday reference print (e.g. Morningstar) sits above prior close AND RSI is oversold, weight the news anchor more heavily than momentum — the bounce tends to exceed the blended estimate.
- (2026-06-24) When analyst directions are split (no >60% agreement) and RSI is 30-40, the realized move is typically within ±0.3% of prior close — test whether a flat-baseline anchor outperforms the blended forecast in these regimes.
- (2026-06-25) When ret_20d < -8% and price is below all SMAs, down-calling momentum+technical should outweigh news/mean-reversion analysts; test capping bullish-blend weight below trend-following weight in confirmed downtrends.
- (2026-06-26) When RSI < 30 AND price > 7% below SMA20 with no confirming premarket gap down, up-weight the contrarian/mean-reversion analyst above all trend-followers — this is a testable regime switch.
- (2026-06-29) When the news analyst reports a confirmed same-day gap and intraday direction (e.g. '+3.8% intraday'), treat that as a realized-price anchor and floor its weight at 0.40, overriding RSI/SMA-distance bearishness which is already priced in.
- (2026-06-30) 2026-06-30: FAIL, did not beat baseline; closest technical, worst news.
- (2026-07-01) 2026-07-01: FAIL, did not beat baseline; closest news, worst macro.
- (2026-07-02) In a confirmed short-term uptrend (1d>0, 5d>0, price>SMA20, RSI 50-60), down-side fades from news/contrarian should be down-weighted ~25% relative to their default, testable via directional-hit rate on similar regime days.
- (2026-07-06) When 4 of 5 analysts agree on direction and prior-day momentum exceeds +5% over 5 days, tilt the blended estimate toward the higher-confidence news/macro cluster rather than the ensemble mean — testable by comparing APE of mean-blend vs. momentum-weighted-blend on days with >80% directional consensus.
- (2026-07-07) On clean-uptrend days (price above SMA5/SMA20, RSI 45-60, zero gap), overweight the technical analyst and discount macro/news bear calls that rest only on index futures with no AMZN-specific catalyst.
- (2026-07-08) When the news analyst flags a same-day macro catalyst (index futures gap >1%, commodity shock) with confidence >0.6, its weight should be floored at 0.35 regardless of scorecard MAPE, because trend-continuation analysts are structurally blind to overnight regime shifts.
- (2026-07-09) 2026-07-09: FAIL, beat baseline; closest news, worst contrarian.
- (2026-07-10) When 3+ analysts agree on a down/mean-reversion call after a >2% multi-day run with RSI in 55-60 and flat premarket, widen the fade magnitude — historically the pullback exceeds the timid consensus estimate.
- (2026-07-13) When a news analyst cites a specific premarket futures move (e.g. 'Nasdaq -1.2%') as its core thesis, cap its weight unless the gap is confirmed at the AMZN open — stale/unconfirmed macro headlines with high confidence are the desk's biggest directional trap.
- (2026-07-14) 2026-07-14: PASS, did not beat baseline; closest momentum, worst contrarian.
- (2026-07-15) When 4+ analysts cluster within a 3-point band below prior close citing the same SMA-50 resistance level, treat that shared anchor as a crowding risk and widen the upside tail — SMA-50 broken on positive 5d/20d momentum is a breakout signal, not a reversion cap.
- (2026-07-16) When RSI>70 AND price >5% above SMA20 AND broad-tech futures are red, the overbought unwind tends to overshoot analyst estimates — bias the blend toward the most-bearish analyst rather than the weighted mean.
- (2026-07-17) When RSI>75 coincides with a same-day bearish reversal candle, weight the down-consensus more heavily and discount any lone bullish (macro) call, targeting the nearest defined support zone rather than the raw blend.
- (2026-07-20) When 4+ analysts share an identical technical thesis (RSI/SMA mean-reversion), treat them as one correlated vote and cap their combined weight so a lone fundamentally-driven dissenter (news/macro catalyst) isn't drowned out.
- (2026-07-21) 2026-07-21: FAIL, did not beat baseline; closest contrarian, worst news.
- (2026-07-22) When the news analyst flags a specific same-day risk-off catalyst (index futures down + macro shock) while technicals only cite range-bound consolidation, overweight news toward its solo call rather than blending — the catalyst dominates on the day.
- (2026-07-23) When multiple independent catalysts stack (regulatory probe + layoffs + macro yield spike) on a confirmed down day, widen the downside target beyond nearby 'support' zones — support levels fail during multi-catalyst selloffs, so extrapolate intraday velocity rather than anchoring to prior close.