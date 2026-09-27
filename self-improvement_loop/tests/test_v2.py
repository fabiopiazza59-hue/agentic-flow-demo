"""v2 pipeline: session timing, anchors, confidence sequences, probabilistic scoring, the lab."""

import math
from datetime import date, datetime, timezone

import pytest

from src.config import settings
from src.data import market_calendar as mc
from src.evals import integrity, lab, probabilistic as pb
from src.evals.gates import apply_gates
from src.evals.sequential import confidence_sequence, cs_radius, rho_squared
from src.features import build_features
from src.loop import session


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


# ------------------------------------------------------------------------ calendar & timing

def test_session_times_are_dst_and_early_close_aware():
    summer = mc.session_times("2026-09-25")
    assert summer == (utc(2026, 9, 25, 13, 30), utc(2026, 9, 25, 20, 0))
    winter = mc.session_times("2026-12-01")
    assert winter[0] == utc(2026, 12, 1, 14, 30)
    assert mc.session_times("2026-11-27")[1] == utc(2026, 11, 27, 18, 0)   # day after Thanksgiving
    assert mc.session_times("2026-09-26") is None                         # Saturday


@pytest.mark.parametrize("now,expected", [
    (utc(2026, 9, 25, 15, 0), None),                  # Friday session in progress
    (utc(2026, 9, 25, 20, 10), None),                 # just closed, still settling
    (utc(2026, 9, 25, 20, 35), date(2026, 9, 28)),    # Friday evening -> Monday
    (utc(2026, 9, 27, 12, 0), date(2026, 9, 28)),     # weekend -> Monday
    (utc(2026, 9, 28, 13, 25), date(2026, 9, 28)),    # Monday pre-open
    (utc(2026, 9, 28, 13, 31), None),                 # Monday after the open
    (utc(2026, 11, 2, 14, 0), date(2026, 11, 2)),     # winter: open is 14:30 UTC
    (utc(2026, 11, 26, 12, 0), date(2026, 11, 27)),   # Thanksgiving -> Friday
])
def test_forecast_target_windows(now, expected):
    assert mc.forecast_target(now, settle_minutes=30) == expected


def test_integrity_uses_the_calendar_open_and_margins():
    assert integrity.minutes_after_open("2026-12-01T14:00:00+00:00", "2026-12-01") == -30
    assert integrity.is_pre_open_now("2026-09-28", utc(2026, 9, 28, 13, 15), margin_minutes=10)
    assert not integrity.is_pre_open_now("2026-09-28", utc(2026, 9, 28, 13, 21), margin_minutes=10)
    assert not integrity.session_has_closed("2026-09-25", utc(2026, 9, 25, 20, 10), 30)
    assert integrity.session_has_closed("2026-09-25", utc(2026, 9, 25, 20, 31), 30)


def test_a_refresh_after_the_open_makes_the_row_late():
    row = {"date": "2026-09-28", "created_at": "2026-09-25T21:45:00+00:00",
           "anchored_at": "2026-09-28T13:40:00+00:00"}
    assert integrity.last_write(row) == "2026-09-28T13:40:00+00:00"
    assert integrity.is_late(row)
    row["anchored_at"] = "2026-09-28T13:10:00+00:00"
    assert not integrity.is_late(row)


# ------------------------------------------------------------------- confidence sequences

def test_cs_radius_matches_the_normal_mixture_formula():
    n, sd, alpha, t_star = 34, 0.0025, 0.05, 120
    r2 = rho_squared(t_star, alpha)
    expected = sd * math.sqrt(2 * (n * r2 + 1) / (n * n * r2) * math.log(math.sqrt(n * r2 + 1) / alpha))
    assert cs_radius(n, sd, alpha, t_star) == pytest.approx(expected)
    # wider than a fixed-n 95% interval — the price of reading it every day
    assert cs_radius(n, sd) > 1.96 * sd / math.sqrt(n)


def test_confidence_sequence_decisions():
    assert confidence_sequence([0.01] * 5)["decision"] == "insufficient"
    assert confidence_sequence([0.01] * 30)["decision"] == "positive"
    assert confidence_sequence([-0.01] * 30)["decision"] == "negative"
    noise = [0.01 if i % 2 else -0.01 for i in range(40)]
    out = confidence_sequence(noise)
    assert out["decision"] == "undecided" and out["lo"] < out["mean"] < out["hi"]


# ------------------------------------------------------------------ probabilistic scoring

def test_student_t_quantiles_and_symmetry():
    q = pb._student_t_unit_quantiles()
    scale = math.sqrt(3 / 5)
    assert q[-1] / scale == pytest.approx(2.015048, abs=1e-5)       # t5 0.95 quantile
    assert q[9] == pytest.approx(0.0, abs=1e-9)
    assert all(a == pytest.approx(-b) for a, b in zip(q, reversed(q)))


def test_crps_of_a_point_mass_is_the_absolute_percent_error():
    point = [100.0] * len(pb.LEVELS)
    assert pb.crps_from_quantiles(point, 101.0) == pytest.approx(1.0 / 101.0, rel=1e-9)


def test_conformal_quantiles_are_symmetric_widened_order_statistics():
    residuals = [(-1) ** i * k for i, k in enumerate(range(1, 41))]    # |z| = 1..40, n = 40
    zq, method = pb.z_quantiles(residuals)
    assert method == "conformal_n40"
    assert zq[-1] == 37                         # 90% central: rank ceil(41 * 0.90) = 37
    assert zq[0] == -37
    assert zq[9] == 0.0                         # the center stays the point forecast
    assert pb.z_quantiles(residuals[:10])[1] == "student_t5"
    # a one-sided (drifting) sample still yields a symmetric distribution
    drift = pb.z_quantiles([-(k % 7) - 0.5 for k in range(40)])[0]
    assert all(a == pytest.approx(-b) for a, b in zip(drift, reversed(drift)))


def test_prob_above_and_prob_pass():
    q = pb.predictive_quantiles(100.0, 0.02, list(pb._student_t_unit_quantiles()))
    assert pb.prob_above(q, 100.0) == pytest.approx(0.5, abs=1e-6)
    assert pb.prob_above(q, 90.0) > 0.95 and pb.prob_above(q, 110.0) < 0.05
    p_pass = pb.prob_pass(q, 100.0)
    assert 0.3 < p_pass < 0.6                   # ±1% of a 2%-σ distribution


def _scored(d, predicted, actual, prior, **extra):
    r = {"date": d, "status": "scored", "predicted_close": predicted, "actual_close": actual,
         "prior_close": prior, "ape": abs(predicted - actual) / actual,
         "baseline_ape": abs(prior - actual) / actual, "created_at": f"{d}T12:00:00+00:00"}
    r.update(extra)
    return r


def test_probabilistic_evaluation_has_no_lookahead():
    rows, price = [], 100.0
    for i in range(45):
        d = f"2026-0{6 + i // 28}-{1 + i % 28:02d}"
        nxt = price * (1.01 if i % 3 else 0.985)
        rows.append(_scored(d, price, nxt, price))
        price = nxt
    rets = pb._returns_by_date(rows)
    full = pb.evaluate(rows, rets)
    # Changing the last day's outcome must not change any earlier day's score.
    altered = [dict(r) for r in rows]
    altered[-1]["actual_close"] *= 1.2
    part = pb.evaluate(altered, pb._returns_by_date(altered))
    for a, b in zip(full[:-1], part[:-1]):
        assert a["model"]["crps"] == pytest.approx(b["model"]["crps"])
    s = pb.summarize(full)
    assert s["n"] == len(full) and s["model"]["crps"] > 0


# ------------------------------------------------------------------------------------ lab

def _lab_row(i, analysts, actual, prior=100.0):
    d = f"2026-07-{i + 1:02d}"
    mean = sum(analysts.values()) / len(analysts)
    r = _scored(d, mean, actual, prior)
    r["analyst_predictions"] = {k: {"predicted_close": v} for k, v in analysts.items()}
    return r


def test_aggregators_and_no_lookahead_in_replay():
    c = {"a": 100.0, "b": 102.0, "c": 110.0, "d": 101.0}
    assert lab.agg_mean(c, [], 100.0) == pytest.approx(103.25)
    assert lab.agg_median(c, [], 100.0) == pytest.approx(101.5)
    assert lab.agg_trimmed_mean(c, [], 100.0) == pytest.approx(101.5)
    assert lab.agg_shrink_half(c, [], 100.0) == pytest.approx(101.625)
    rows = [_lab_row(i, {"a": 101.0, "b": 99.0, "c": 104.0}, 101.0) for i in range(12)]
    days = lab.replay(rows)
    # inverse_mse on day k may only use days < k: the first 5 days fall back to the mean
    assert days[0]["apes"]["inverse_mse"] == pytest.approx(days[0]["apes"]["mean"])


def test_lab_promotes_a_clearly_better_rule(monkeypatch):
    # analyst "a" is always exact, "c" is always far off: the median beats the mean every day
    rows = [_lab_row(i, {"a": 101.0, "b": 101.2, "c": 110.0}, 101.0) for i in range(25)]
    out = lab.evaluate(rows)
    assert out["challengers"]["median"]["decision"] == "positive"
    assert out["champion"] != lab.REFERENCE
    assert out["state"]["champion"] == out["champion"]


def test_lab_switches_off_a_harmful_judge():
    rows = []
    for i in range(20):
        r = _lab_row(i, {"a": 101.0, "b": 101.0, "c": 101.0}, 101.0)
        r.update(predicted_close_blend=101.0, predicted_close_pre_gates=102.0,
                 predicted_close_raw=102.0, predicted_close=102.0, ape=1 / 101)
        rows.append(r)
    out = lab.evaluate(rows)
    assert out["judge_effect"]["decision"] == "negative"
    assert out["state"]["judge_enabled"] is False
    assert out["state"]["gates_enabled"] is True     # gates changed nothing


def test_lab_state_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "LEARNINGS_DIR", tmp_path, raising=False)
    assert lab.load_state() == {"champion": "mean", "judge_enabled": True, "gates_enabled": True}
    lab.save_state({"state": {"champion": "median", "judge_enabled": False, "gates_enabled": True},
                    "n_days": 3, "challengers": {"median": {"decision": "positive"}},
                    "champion": "median", "judge_effect": {"decision": "negative"},
                    "gates_effect": {"decision": "undecided"}})
    assert lab.load_state()["champion"] == "median"
    assert lab.load_state()["judge_enabled"] is False


# ------------------------------------------------------------------------ anchors & refresh

def _pending_row(**extra):
    r = {"date": "2026-09-28", "status": "pending", "prior_close": 250.0, "anchor": 250.0,
         "anchor_live": False, "anchor_time": "2026-09-25T20:00:00+00:00",
         "predicted_close": 252.5, "predicted_close_raw": 252.5, "predicted_close_blend": 251.0,
         "quantiles": [240.0 + i for i in range(19)], "anchor_updates": 0}
    r.update(extra)
    return r


def test_reanchor_scales_every_price_level():
    anchor = {"price": 255.0, "time": "2026-09-28T12:30:00+00:00", "source": "yfinance_ext",
              "live": True}
    out = session.reanchor_row(_pending_row(), anchor, "2026-09-28T12:31:00+00:00")
    k = 255.0 / 250.0
    assert out["predicted_close"] == round(252.5 * k, 2)
    assert out["predicted_close_blend"] == round(251.0 * k, 2)
    assert out["quantiles"][0] == round(240.0 * k, 4)
    assert out["anchor"] == 255.0 and out["anchor_live"] and out["anchor_updates"] == 1
    assert out["predicted_direction"] == "up"
    assert 0 <= out["p_up"] <= 1 and 0 <= out["confidence"] <= 1


def test_reanchor_refuses_stale_or_non_live_anchors():
    live_old = {"price": 251.0, "time": "2026-09-28T11:00:00+00:00", "live": True}
    row = _pending_row(anchor_live=True, anchor_time="2026-09-28T12:00:00+00:00")
    assert session.reanchor_row(row, live_old, "now") is None               # older trade
    assert session.reanchor_row(_pending_row(), {"price": 251.0, "live": False}, "now") is None
    assert session.reanchor_row(_pending_row(status="scored"), live_old, "now") is None


def test_features_never_fabricate_a_flat_gap(monkeypatch):
    import pandas as pd
    hist = pd.DataFrame({"open": [1.0] * 30, "high": [1.0] * 30, "low": [1.0] * 30,
                         "close": [100.0 + i for i in range(30)], "volume": [1] * 30},
                        index=list(pd.bdate_range("2026-01-01", periods=30).date))
    stale = build_features(hist, anchor={"price": 129.0, "live": False, "source": "prior_close"})
    assert stale["premarket_last"] is None and stale["premarket_gap_pct"] is None
    live = build_features(hist, anchor={"price": 130.29, "live": True, "source": "yfinance_ext",
                                        "time": "t"})
    assert live["premarket_gap_pct"] == pytest.approx(0.01)
    legacy = build_features(hist, quote={"last": 129.0, "source": "alphavantage", "stale": True})
    assert legacy["premarket_gap_pct"] is None


def test_get_anchor_falls_back_and_rejects_bad_ticks(monkeypatch):
    import src.data.providers as pv
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    since = utc(2026, 9, 25, 20, 0)

    def boom(*a, **k):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(pv, "yfinance_extended_last", boom)
    a = pv.get_anchor("AMZN", 250.0, since, utc(2026, 9, 28, 12, 0))
    assert a["source"] == "prior_close" and a["live"] is False and a["price"] == 250.0
    monkeypatch.setattr(pv, "yfinance_extended_last",
                        lambda *a, **k: {"price": 500.0, "time": "t", "source": "yfinance_ext"})
    assert pv.get_anchor("AMZN", 250.0, since)["live"] is False              # +100%: bad tick
    monkeypatch.setattr(pv, "yfinance_extended_last",
                        lambda *a, **k: {"price": 290.0, "time": "t", "source": "yfinance_ext"})
    assert pv.get_anchor("AMZN", 250.0, since)["price"] == 290.0             # +16%: earnings gap


def test_gates_shrink_toward_the_live_anchor():
    final = {"predicted_close": 104.0, "direction": "up", "confidence": 0.5, "rationale": ""}
    analysts = {f"a{i}": {"predicted_close": 101.0, "direction": d}
                for i, d in enumerate(["up", "down", "down", "down", "down"])}
    gated, gates = apply_gates(final, analysts, [], {"prev_close": 100.0, "premarket_last": 102.0})
    assert gates == ["low_consensus_shrink"]
    assert gated["predicted_close"] == 103.0          # halfway from 104 back to the 102 anchor


# ----------------------------------------------------------------- run flow (research/refresh)

def test_forecast_both_researches_then_refreshes_then_stops(monkeypatch):
    import src.loop.run_ab as ab

    calls = {"a": 0, "b": 0}

    def fake_predict(rows, target, client, features=None, allow_late=False):
        calls["a"] += 1
        row = _pending_row(date=target.isoformat())
        return rows + [row], row

    def fake_predict_b(rows_b, target, client, features, history):
        calls["b"] += 1
        return rows_b + [_pending_row(date=target.isoformat())], None

    snaps = iter([
        {"features": {"prev_close": 250.0}, "history": None,
         "anchor": {"price": 250.0, "live": False, "source": "prior_close", "time": None}},
        {"features": {}, "history": None,
         "anchor": {"price": 255.0, "live": True, "source": "yfinance_ext",
                    "time": "2026-09-28T12:30:00+00:00"}},
    ])
    monkeypatch.setattr(ab, "do_predict", fake_predict)
    monkeypatch.setattr(ab, "do_predict_b", fake_predict_b)
    monkeypatch.setattr(ab.session, "take_snapshot", lambda target: next(snaps))
    monkeypatch.setattr("src.loop.run_daily.integrity.is_pre_open_now", lambda *a, **k: True)
    monkeypatch.setattr(ab, "iso_today", lambda: "2026-09-25")
    target = date(2026, 9, 28)

    rows_a, rows_b = ab.forecast_both([], [], target, None)                 # evening: research
    assert calls == {"a": 1, "b": 1} and len(rows_a) == 1 and len(rows_b) == 1
    rows_a, rows_b = ab.forecast_both(rows_a, rows_b, target, None)         # morning: re-anchor
    assert calls == {"a": 1, "b": 1}
    assert rows_a[0]["anchor"] == 255.0 and rows_b[0]["anchor"] == 255.0

    monkeypatch.setattr("src.loop.run_daily.integrity.is_pre_open_now", lambda *a, **k: False)
    before = [dict(r) for r in rows_a]
    rows_a, rows_b = ab.forecast_both(rows_a, rows_b, target, None)         # after the open
    assert rows_a == before and calls == {"a": 1, "b": 1}


def test_scoring_waits_for_the_close_to_settle(monkeypatch, tmp_path):
    import src.loop.run_daily as rd

    monkeypatch.setattr(rd.integrity, "session_has_closed", lambda *a, **k: False)
    monkeypatch.setattr(rd, "get_actual_close", lambda *a: pytest.fail("must not fetch"))
    rows = [_pending_row(date="2026-09-28")]
    out, last = rd.do_score(rows, None)
    assert last is None and out[0]["status"] == "pending"


def test_judge_adjustment_is_clamped():
    from src.agents.meta_judge import adjust, clamp_adjustment
    cap = settings.A_JUDGE_MAX_SIGMA
    assert clamp_adjustment(3.0) == cap and clamp_adjustment(-3.0) == -cap
    assert clamp_adjustment(0.2) == 0.2
    assert adjust(100.0, "mean", 0.02, {"a": {}}, {}, "", [], {}, client=None)[0] == 0.0
