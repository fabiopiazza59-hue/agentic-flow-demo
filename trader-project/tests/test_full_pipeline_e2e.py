#!/usr/bin/env python3
"""End-to-end test that exercises the exact same code path as the Streamlit app.

Simulates: load fixture → Stage 3 → Stage 4a → Stage 4b → Stage 5 → Stage 6.
This is the same sequence of calls the Streamlit button handler makes.
"""

import os
import sys
import uuid
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import Settings, AccountConfig, PipelineConfig
from src.models.schemas import (
    AccountState, Decision, EnrichedScenario, Stage4Output, Verdict,
)
from src.stages.stage3_triage import run_stage3, load_stage2_from_yaml
from src.stages.stage4a_risk_math import compute_risk_math, parse_levels_from_scenario
from src.stages.stage4b_bear_challenger import run_stage4b_batch
from src.stages.stage5_synthesis import run_stage5
from src.stages.stage6_action_table import run_stage6, render_markdown
from src.journal import write_journal


def main():
    # ── Config (same as Streamlit sidebar defaults) ──
    capital = 2000
    settings = Settings(
        account=AccountConfig(
            total_capital_usd=capital,
            max_loss_per_trade_pct=0.02,
            max_portfolio_heat_pct=0.06,
        ),
        pipeline=PipelineConfig(model="claude-sonnet-4-20250514"),
    )
    account_state = AccountState(
        total_capital_usd=capital,
        available_capital_usd=capital,
        current_portfolio_heat_pct=0.0,
    )
    prices = {
        "MU": 268.50, "AMAT": 195.00, "KRE": 48.20, "CCJ": 55.00,
        "OKLO": 28.00, "NVDA": 135.00, "GLD": 230.00,
    }
    run_id = f"ui-test-{uuid.uuid4().hex[:6]}"

    # ── Load fixture (same as Streamlit "Use sample fixture") ──
    print("Loading fixture...")
    stage2_output = load_stage2_from_yaml("fixtures/sample_stage2_output.yaml")
    print(f"  {len(stage2_output.scenarios)} scenarios loaded")

    # ── Stage 3 ──
    print("\n[Stage 3] Triage gate...")
    stage3_result = run_stage3(stage2_output, settings, run_id=run_id)
    write_journal(3, run_id, stage3_result)

    summary = stage3_result.batch_summary
    print(f"  In={summary.scenarios_in}  LIVE={summary.verdicts.get('LIVE',0)}  "
          f"WATCH={summary.verdicts.get('WATCH',0)}  KILL={summary.verdicts.get('KILL',0)}")
    for v in stage3_result.verdicts:
        icon = {"LIVE": "✅", "WATCH": "⏸️", "KILL": "❌"}[v.verdict.value]
        print(f"  {icon} {v.scenario_id} → {v.verdict.value}: {v.one_line_reason}")
    print(f"  Desk note: {summary.desk_note}")

    # ── Filter survivors (same logic as app.py lines 260-271) ──
    surviving_ids = {
        v.scenario_id for v in stage3_result.verdicts
        if v.verdict in (Verdict.LIVE, Verdict.WATCH)
    }
    surviving_scenarios = [
        s for s in stage2_output.scenarios if s.scenario_id in surviving_ids
    ]
    if not surviving_scenarios:
        print("\n  All scenarios killed. No trades this week.")
        return

    # ── Stage 4a ──
    print(f"\n[Stage 4a] Risk math — {len(surviving_scenarios)} scenarios...")
    enriched = []
    risk_maths = {}

    for scenario in surviving_scenarios:
        price = prices.get(scenario.instrument, 0)
        if price == 0:
            print(f"  WARNING: No price for {scenario.instrument}")
            continue

        entry, stop, target = parse_levels_from_scenario(
            scenario.proposed_kill, scenario.proposed_catalyst,
            price, scenario.direction,
        )
        rm = compute_risk_math(
            entry, stop, target, scenario.direction, price,
            scenario.proposed_horizon, settings.account,
            account_state.open_positions,
        )
        risk_maths[scenario.scenario_id] = rm

        stage3_verdict = Verdict.LIVE
        for v in stage3_result.verdicts:
            if v.scenario_id == scenario.scenario_id:
                stage3_verdict = v.verdict
                break

        enriched.append(EnrichedScenario(
            scenario_id=scenario.scenario_id,
            stage3_verdict=stage3_verdict,
            current_price=price,
            risk_math=rm,
            bear_case=None,
        ))

        floor_str = "PASS" if rm.passes_risk_floor else f"FAIL ({', '.join(rm.risk_floor_failures)})"
        print(f"  {scenario.instrument}: E=${entry:.2f} S=${stop:.2f} T=${target:.2f} "
              f"R={rm.r_multiple:.1f} Size=${rm.position_size_usd:.0f} Floor={floor_str}")

    # ── Stage 4b ──
    print(f"\n[Stage 4b] Bear challenger — {len(surviving_scenarios)} scenarios...")
    bear_cases = run_stage4b_batch(surviving_scenarios, risk_maths, settings)
    for e in enriched:
        if e.scenario_id in bear_cases:
            e.bear_case = bear_cases[e.scenario_id]
            print(f"  {e.scenario_id}: {e.bear_case.bear_verdict.value} "
                  f"(haircut: {e.bear_case.recommended_size_haircut:.0%})")

    stage4_output = Stage4Output(run_id=run_id, enriched_scenarios=enriched)
    write_journal(4, run_id, stage4_output)

    # ── Stage 5 ──
    print(f"\n[Stage 5] Synthesis — {len(enriched)} scenarios...")
    stage5_result = run_stage5(
        stage4_output, account_state,
        settings=settings, run_id=run_id,
    )
    write_journal(5, run_id, stage5_result)

    takes = [d for d in stage5_result.decisions if d.decision in (Decision.take, Decision.reduce_size)]
    passes = [d for d in stage5_result.decisions if d.decision == Decision.pass_]
    print(f"  Take/reduce: {len(takes)} | Pass: {len(passes)}")
    for d in stage5_result.decisions:
        s = next((x for x in stage2_output.scenarios if x.scenario_id == d.scenario_id), None)
        ticker = s.instrument if s else d.scenario_id
        print(f"  {'🟢' if d.decision in (Decision.take, Decision.reduce_size) else '🔴'} "
              f"{ticker}: {d.decision.value} ${d.final_size_usd:.0f}")
        print(f"    Rationale: {d.rationale[:120]}...")
    print(f"  IC Note: {stage5_result.ic_note}")

    # ── Stage 6 ──
    print(f"\n[Stage 6] Action table...")
    stage6_result = run_stage6(stage5_result)

    # Enrich with risk math (same as app.py lines 417-425)
    rm_lookup = {e.scenario_id: e.risk_math for e in enriched}
    for row in stage6_result.new_trades:
        rm = rm_lookup.get(row.scenario_id)
        if rm:
            row.entry_price = rm.proposed_entry
            row.stop_loss = rm.proposed_stop
            row.target_price = rm.proposed_target
            row.r_multiple = rm.r_multiple

    write_journal(6, run_id, stage6_result)

    md = render_markdown(stage6_result)
    print(f"\n{'='*60}")
    print(md)
    print(f"{'='*60}")
    print(f"\nPipeline complete. Run ID: {run_id}")
    print("All stages passed — Streamlit UI code path is valid.")


if __name__ == "__main__":
    main()
