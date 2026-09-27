"""Anytime-valid inference — verdicts that stay honest when the dashboard is read every day.

RESULTS.md re-renders its verdict after every scored day. A fixed-sample test (bootstrap CI, sign
test) read that way is a p-hacking machine: under the null, "significant at 5%" is eventually
crossed by chance alone (canon 15 of arXiv:2609.05663 shows a live p-value wandering 0.0067 ->
0.19 -> 0.0277). A confidence sequence (CS) is a sequence of intervals that covers the true mean
at *every* sample size simultaneously with probability >= 1 - alpha, so a verdict read off it
daily keeps its error rate, and there is no horizon to pre-register.

We use the asymptotic CS of Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas, "Time-uniform central
limit theory and asymptotic confidence sequences" (arXiv:2103.06476): Robbins' normal-mixture
boundary with the running sample standard deviation plugged in,

    mean_n ± sd_n · sqrt( 2(nρ² + 1) / (n²ρ²) · ln( sqrt(nρ² + 1) / alpha ) ),

with ρ² = (−2 ln alpha + ln(−2 ln alpha + 1)) / t* making the boundary tightest near t*
observations. Because the variance is estimated it is asymptotic, so nothing is claimed below
`min_n` observations. The target is the running average of the per-day conditional means, which
can drift when the market regime changes — so the interval reported is the current one, not a
running intersection (intersecting is only valid for a constant target).
"""

from __future__ import annotations

import math
import statistics as st

DEFAULT_ALPHA = 0.05
DEFAULT_T_STAR = 120      # ~6 months of sessions: where the boundary is tuned to be tightest
DEFAULT_MIN_N = 10


def rho_squared(t_star: int = DEFAULT_T_STAR, alpha: float = DEFAULT_ALPHA) -> float:
    a = -2.0 * math.log(alpha)
    return (a + math.log(a + 1.0)) / float(t_star)


def cs_radius(n: int, sd: float, alpha: float = DEFAULT_ALPHA,
              t_star: int = DEFAULT_T_STAR) -> float:
    """Half-width of the two-sided (1 - alpha) asymptotic CS after n observations."""
    if n <= 0:
        return math.inf
    r2 = rho_squared(t_star, alpha)
    return sd * math.sqrt(2.0 * (n * r2 + 1.0) / (n * n * r2)
                          * math.log(math.sqrt(n * r2 + 1.0) / alpha))


def confidence_sequence(xs: list[float], alpha: float = DEFAULT_ALPHA,
                        t_star: int = DEFAULT_T_STAR, min_n: int = DEFAULT_MIN_N) -> dict:
    """Anytime-valid interval for the mean of `xs` (in arrival order).

    Returns {n, mean, lo, hi, alpha, decision} with decision one of
    "positive" (lo > 0), "negative" (hi < 0), "undecided", or "insufficient" (n < min_n).
    """
    xs = [float(x) for x in xs]
    n = len(xs)
    out = {"n": n, "mean": round(st.mean(xs), 8) if xs else None, "lo": None, "hi": None,
           "alpha": alpha, "t_star": t_star, "decision": "insufficient"}
    if n < max(min_n, 2):
        return out
    m, sd = st.mean(xs), st.stdev(xs)
    r = cs_radius(n, max(sd, 1e-12), alpha, t_star)
    lo, hi = m - r, m + r
    out["lo"], out["hi"] = round(lo, 8), round(hi, 8)
    out["decision"] = "positive" if lo > 0 else "negative" if hi < 0 else "undecided"
    return out
