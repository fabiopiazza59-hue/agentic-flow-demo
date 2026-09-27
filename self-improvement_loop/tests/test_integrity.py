"""Pre-open integrity: which rows count as forecasts, and what the gates are worth."""

from datetime import datetime, timezone

import pytest

from src.config import settings
from src.evals import integrity
from src.evals.gates import apply_gates
from src.evals.metrics import aggregate
from src.evals.paired import paired_effect, sign_test_p
from src.report import gate_effect


def row(date, created_hour, ape=0.01, base=0.012, **extra):
    r = {
        "date": date,
        "created_at": f"{date}T{created_hour}:00:00+00:00",
        "status": "scored",
        "prior_close": 100.0,
        "predicted_close": 101.0,
        "actual_close": 100.5,
        "ape": ape,
        "baseline_ape": base,
        "pass": ape <= 0.01,
        "directional_hit": True,
        "brier": 0.25,
    }
    r.update(extra)
    return r


def test_minutes_after_open_signs():
    assert integrity.minutes_after_open("2026-09-04T12:15:00+00:00", "2026-09-04") == -75
    assert integrity.minutes_after_open("2026-09-04T13:30:00+00:00", "2026-09-04") == 0
    assert integrity.minutes_after_open("2026-09-04T21:08:00+00:00", "2026-09-04") == 458
    assert integrity.minutes_after_open(None, "2026-09-04") is None
    assert integrity.minutes_after_open("2026-09-04T12:15:00+00:00", None) is None


def test_is_late_boundary_and_seeds():
    assert integrity.is_late(row("2026-09-04", "13")) is False        # 13:00, before the open
    assert integrity.is_late(row("2026-09-04", "14")) is True         # after the open
    # A row stamped between 13:30 and 14:00 is late — an hour-granularity check would miss it.
    assert integrity.is_late({"date": "2026-09-04",
                              "created_at": "2026-09-04T13:45:00+00:00"}) is True
    # Backfill seeds carry the timestamp of the backfill run, not of a forecast.
    assert integrity.is_late(row("2026-09-04", "21", seed=True)) is False


def test_annotate_stamps_late_minutes_and_skips_seeds():
    r = integrity.annotate(row("2026-09-04", "15"))
    assert r["late_minutes"] == 90
    assert "late_minutes" not in integrity.annotate(row("2026-09-04", "15", seed=True))


def test_aggregate_pre_open_only_excludes_late_rows():
    rows = [row("2026-09-01", "12"), row("2026-09-02", "12"), row("2026-09-03", "16")]
    assert aggregate(rows)["n"] == 3
    assert aggregate(rows, pre_open_only=True)["n"] == 2


def test_is_pre_open_now_guards_the_write():
    assert integrity.is_pre_open_now("2026-09-04", datetime(2026, 9, 4, 13, 29, tzinfo=timezone.utc))
    assert not integrity.is_pre_open_now("2026-09-04",
                                         datetime(2026, 9, 4, 13, 30, tzinfo=timezone.utc))


def test_apply_gates_records_the_ungated_counterfactual():
    final = {"predicted_close": 110.0, "direction": "up", "confidence": 0.5, "rationale": ""}
    analysts = {n: {"predicted_close": 110.0, "direction": "up", "confidence": 0.5}
                for n in ("a", "b", "c")}
    gated, _ = apply_gates(final, analysts, [], {"prev_close": 100.0})
    assert gated["predicted_close_raw"] == 110.0
    assert final.get("predicted_close_raw") is None  # input is not mutated


def test_gate_effect_is_paired_over_days():
    # Gated prediction is nearer the actual than the ungated one on both days => positive effect.
    rows = [
        row("2026-09-01", "12", ape=0.005, predicted_close_raw=110.0, actual_close=100.5),
        row("2026-09-02", "12", ape=0.005, predicted_close_raw=110.0, actual_close=100.5),
    ]
    effect = gate_effect(rows)
    assert effect["n"] == 2
    assert effect["mean"] > 0
    assert effect["wins"] == 2


def test_gate_effect_absent_without_counterfactuals():
    assert gate_effect([row("2026-09-01", "12")])["n"] == 0


@pytest.mark.parametrize("wins,n,expected", [(5, 5, 0.0625), (0, 0, None), (3, 6, 1.0)])
def test_sign_test_p(wins, n, expected):
    assert sign_test_p(wins, n) == expected


def test_paired_effect_flags_significance_only_when_both_agree():
    weak = paired_effect([0.001, -0.002, 0.003])
    assert weak["significant"] is False
    strong = paired_effect([0.01] * 12)
    assert strong["ci_low"] > 0 and strong["significant"] is True


def test_predict_guard_blocks_today_after_open_but_allows_replay(monkeypatch, tmp_path):
    """The guard protects the live path only; a past-dated replay still runs."""
    from datetime import date

    from src.loop import run_daily as rd

    monkeypatch.setattr(rd.settings, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(rd, "iso_today", lambda: "2026-09-04")
    monkeypatch.setattr(rd.integrity, "is_pre_open_now", lambda *a, **k: False)
    called = []
    monkeypatch.setattr(rd, "run_analysts", lambda *a, **k: called.append(1) or {})

    rows, row = rd.do_predict([], date(2026, 9, 4), None, features={"prev_close": 100.0})
    assert row is None and rows == [] and not called   # today, post-open -> refused

    rows, row = rd.do_predict([], date(2026, 8, 3), None, features={"prev_close": 100.0})
    assert row is not None and called                  # past-dated replay -> allowed


def test_verdict_refuses_to_claim_an_edge_on_noise():
    """A nominally positive edge that fails the paired test must not read as 'Edge confirmed'."""
    from src.report import build_metrics, render_results_md

    # Mean edge is positive but the per-day sign flips: noise, not skill.
    rows = []
    for i in range(20):
        better = i % 2 == 0
        rows.append(row(f"2026-09-{i + 1:02d}", "12",
                        ape=0.0100 if better else 0.0135,
                        base=0.0140 if better else 0.0100))
    metrics = build_metrics(rows, {})
    assert metrics["rolling_pre_open"]["edge"] > 0          # nominally ahead
    assert metrics["baseline_effect"]["significant"] is False
    md = render_results_md(metrics, rows)
    assert "No edge yet" in md and "not distinguishable from noise" in md
    assert "Edge confirmed" not in md


def test_verdict_confirms_an_edge_that_clears_the_firm_bar():
    from src.report import build_metrics, render_results_md

    rows = [row(f"2026-09-{i + 1:02d}", "12", ape=0.004, base=0.012) for i in range(20)]
    metrics = build_metrics(rows, {})
    assert metrics["baseline_effect"]["significant"] is True
    assert "Edge confirmed" in render_results_md(metrics, rows)
