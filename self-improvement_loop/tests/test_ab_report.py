"""A/B comparison: paired-day matching, sign test, verdict gating."""

from src.config import settings
from src.report_ab import build_comparison, paired_days, sign_test_p


def _scored(date, ape, seed=False, dir_hit=True):
    row = {"date": date, "status": "scored", "ape": ape, "baseline_ape": 0.01,
           "pass": ape <= 0.01, "directional_hit": dir_hit,
           "predicted_close": 200.0, "actual_close": 200.0, "prior_close": 199.0,
           "brier": 0.25}
    if seed:
        row["seed"] = True
    return row


def test_paired_days_matching_and_seed_exclusion():
    rows_a = [_scored("2026-07-01", 0.02), _scored("2026-07-02", 0.01),
              _scored("2026-07-03", 0.01, seed=True)]
    rows_b = [_scored("2026-07-02", 0.005), _scored("2026-07-03", 0.005),
              _scored("2026-07-04", 0.005)]
    pairs = paired_days(rows_a, rows_b)
    assert [p["date"] for p in pairs] == ["2026-07-02"]  # only real overlap; A's seed excluded
    assert pairs[0]["b_wins"] is True
    assert pairs[0]["delta"] == 0.005


def test_sign_test():
    assert sign_test_p(0, 0) is None
    assert sign_test_p(5, 10) == 1.0
    assert sign_test_p(10, 10) < 0.01  # 2/1024 ≈ 0.002


def test_no_verdict_below_min_paired_days():
    rows_a = [_scored(f"2026-07-{d:02d}", 0.02) for d in range(1, 4)]
    rows_b = [_scored(f"2026-07-{d:02d}", 0.01) for d in range(1, 4)]
    cmp = build_comparison(rows_a, rows_b)
    assert cmp["n_paired"] == 3
    assert "no verdict" in cmp["verdict"].lower() or "⏳" in cmp["verdict"]


def test_verdict_with_enough_days():
    n = settings.AB_MIN_PAIRED_DAYS + 2
    rows_a = [_scored(f"2026-06-{d:02d}", 0.02) for d in range(1, n + 1)]
    rows_b = [_scored(f"2026-06-{d:02d}", 0.005) for d in range(1, n + 1)]
    cmp = build_comparison(rows_a, rows_b)
    assert cmp["b_wins"] == n
    assert cmp["mean_delta_a_minus_b"] > 0
    assert "Arm B leads" in cmp["verdict"]
    assert cmp["sign_test_p"] < 0.05
