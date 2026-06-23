# Living Strategy — AMZN Close Predictor

Append-only insights. Newest at the bottom.

- (2026-06-08) On low-realized-volatility days (small expected |move|), shrink momentum's magnitude toward prior_close and lean on macro, since momentum tends to overshoot when the actual move is muted.
- (2026-06-12) 2026-06-12: FAIL, did not beat baseline; closest momentum, worst contrarian.
- (2026-06-15) When RSI < 30 AND a same-day risk-on news catalyst is present, the news/contrarian analysts should override momentum/macro continuation calls — test inverting the weight stack in deep-oversold + catalyst regimes.
- (2026-06-16) On flat-gap, oversold-but-not-extreme days (RSI 33-38, no premarket gap), default toward prior_close rather than directional tilt — test whether |predicted - prior_close| < 0.2% beats baseline more often than directional bets.
- (2026-06-17) 2026-06-17: FAIL, did not beat baseline; closest technical, worst contrarian.
- (2026-06-18) 2026-06-18: FAIL, beat baseline; closest contrarian, worst macro.
- (2026-06-22) When 3+ analysts cite 'oversold RSI' as a bounce thesis while price is already >3% below SMA20 AND news flags an active sector selloff, suppress the mean-reversion analysts and lean toward the news/momentum-down read — oversold can stay oversold in a downtrend.