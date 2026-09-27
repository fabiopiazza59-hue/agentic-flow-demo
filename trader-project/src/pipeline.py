"""Pipeline orchestrator — runs the six-stage Seamaster funnel.

Stages run sequentially. Each stage writes to the journal before the next
stage reads its output. Information flows one way.

Usage:
    from src.pipeline import run_pipeline
    result = run_pipeline()  # full run
    result = run_pipeline(start_stage=3, stage2_input="path/to/stage2.yaml")  # partial
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import yaml

from src.config.settings import Settings
from src.journal import write_journal
from src.models.schemas import (
    AccountState,
    Direction,
    EnrichedScenario,
    OpenPosition,
    Scenario,
    Stage1Output,
    Stage2Output,
    Stage3Output,
    Stage4Output,
    Stage5Output,
    Stage6Output,
    Verdict,
)
from src.stages.stage1_market_scan import run_stage1
from src.stages.stage2_ticker_discovery import run_stage2
from src.stages.stage3_triage import run_stage3, load_stage2_from_yaml
from src.stages.stage4a_risk_math import compute_risk_math, parse_levels_from_scenario
from src.stages.stage4b_bear_challenger import run_stage4b_batch
from src.stages.stage5_synthesis import run_stage5
from src.stages.stage6_action_table import run_stage6, render_markdown, render_yaml


@dataclass
class PipelineResult:
    """Holds all intermediate outputs from a pipeline run."""
    run_id: str
    stage1: Optional[Stage1Output] = None
    stage2: Optional[Stage2Output] = None
    stage3: Optional[Stage3Output] = None
    stage4: Optional[Stage4Output] = None
    stage5: Optional[Stage5Output] = None
    stage6: Optional[Stage6Output] = None
    journal_paths: dict[int, Path] = field(default_factory=dict)


def run_pipeline(
    settings: Settings | None = None,
    account_state: AccountState | None = None,
    start_stage: int = 1,
    stage2_input: str | None = None,
    current_prices: dict[str, float] | None = None,
    run_id: str | None = None,
) -> PipelineResult:
    """Run the full or partial pipeline.

    Args:
        settings: Pipeline and account settings.
        account_state: Current account state (for Stage 5). Defaults to empty.
        start_stage: Stage to start from (1-6). Requires appropriate input.
        stage2_input: Path to Stage 2 YAML file (required if start_stage >= 3).
        current_prices: Dict of ticker -> current price (for Stage 4a).
        run_id: Override run ID.

    Returns:
        PipelineResult with all outputs and journal paths.
    """
    if settings is None:
        settings = Settings()
    if account_state is None:
        account_state = AccountState(
            total_capital_usd=settings.account.total_capital_usd,
            available_capital_usd=settings.account.total_capital_usd,
            current_portfolio_heat_pct=0.0,
        )
    if current_prices is None:
        current_prices = {}

    rid = run_id or f"run-{date.today().isoformat()}-{uuid.uuid4().hex[:6]}"
    result = PipelineResult(run_id=rid)

    print(f"\n{'='*60}")
    print(f"Seamaster Pipeline — Run {rid}")
    print(f"{'='*60}")

    # ── Stage 1: Market Scan ──
    if start_stage <= 1:
        print("\n[Stage 1] Market scan — running journalist + macro lenses...")
        result.stage1 = run_stage1(settings, run_id=rid)
        result.journal_paths[1] = write_journal(1, rid, result.stage1)
        print(f"  Themes: {len(result.stage1.themes)}")
        print(f"  Desk note: {result.stage1.desk_note}")
        print(f"  Journal: {result.journal_paths[1]}")

    # ── Stage 2: Ticker Discovery ──
    if start_stage <= 2:
        if result.stage1 is None:
            raise ValueError("Stage 2 requires Stage 1 output")
        print("\n[Stage 2] Ticker discovery — translating themes to scenarios...")
        result.stage2 = run_stage2(result.stage1, settings, run_id=rid)
        result.journal_paths[2] = write_journal(2, rid, result.stage2)
        print(f"  Scenarios: {len(result.stage2.scenarios)}")
        print(f"  Journal: {result.journal_paths[2]}")

    # Load Stage 2 output from file if starting mid-pipeline
    if start_stage >= 3 and result.stage2 is None:
        if stage2_input is None:
            raise ValueError("Starting at Stage 3+ requires stage2_input path")
        result.stage2 = load_stage2_from_yaml(stage2_input)

    # ── Stage 3: Triage Gate ──
    if start_stage <= 3:
        if result.stage2 is None:
            raise ValueError("Stage 3 requires Stage 2 output")
        print(f"\n[Stage 3] Triage gate — {len(result.stage2.scenarios)} scenarios...")
        result.stage3 = run_stage3(result.stage2, settings, run_id=rid)
        result.journal_paths[3] = write_journal(3, rid, result.stage3)
        s = result.stage3.batch_summary
        print(f"  LIVE={s.verdicts.get('LIVE', 0)} | "
              f"WATCH={s.verdicts.get('WATCH', 0)} | "
              f"KILL={s.verdicts.get('KILL', 0)}")
        print(f"  Journal: {result.journal_paths[3]}")

    # ── Stage 4: Deterministic Checks (4a + 4b) ──
    if start_stage <= 4:
        if result.stage3 is None:
            raise ValueError("Stage 4 requires Stage 3 output")
        if result.stage2 is None:
            raise ValueError("Stage 4 requires Stage 2 scenarios")

        # Filter to LIVE + WATCH only
        surviving_ids = {
            v.scenario_id
            for v in result.stage3.verdicts
            if v.verdict in (Verdict.LIVE, Verdict.WATCH)
        }
        surviving_scenarios = [
            s for s in result.stage2.scenarios
            if s.scenario_id in surviving_ids
        ]

        print(f"\n[Stage 4a] Risk math — {len(surviving_scenarios)} scenarios...")
        enriched = []
        risk_maths = {}
        for scenario in surviving_scenarios:
            # Get current price (from provided dict or use a placeholder)
            price = current_prices.get(scenario.instrument, 0)
            if price == 0:
                print(f"  WARNING: No current price for {scenario.instrument}, skipping 4a")
                continue

            entry, stop, target = parse_levels_from_scenario(
                scenario.proposed_kill,
                scenario.proposed_catalyst,
                price,
                scenario.direction,
            )

            rm = compute_risk_math(
                entry=entry,
                stop=stop,
                target=target,
                direction=scenario.direction,
                current_price=price,
                proposed_horizon=scenario.proposed_horizon,
                account=settings.account,
                open_positions=account_state.open_positions,
            )
            risk_maths[scenario.scenario_id] = rm

            # Find the Stage 3 verdict for this scenario
            stage3_verdict = Verdict.LIVE
            for v in result.stage3.verdicts:
                if v.scenario_id == scenario.scenario_id:
                    stage3_verdict = v.verdict
                    break

            enriched.append(EnrichedScenario(
                scenario_id=scenario.scenario_id,
                stage3_verdict=stage3_verdict,
                current_price=price,
                risk_math=rm,
                bear_case=None,  # filled by 4b
            ))

            floor_status = "PASS" if rm.passes_risk_floor else f"FAIL ({', '.join(rm.risk_floor_failures)})"
            print(f"  {scenario.instrument}: R={rm.r_multiple} | "
                  f"Size=${rm.position_size_usd:.0f} | Floor: {floor_status}")

        # Stage 4b: Bear Challenger
        print(f"\n[Stage 4b] Bear challenger — {len(surviving_scenarios)} scenarios...")
        bear_cases = run_stage4b_batch(surviving_scenarios, risk_maths, settings)
        for e in enriched:
            if e.scenario_id in bear_cases:
                e.bear_case = bear_cases[e.scenario_id]
                print(f"  {e.scenario_id}: {e.bear_case.bear_verdict.value} "
                      f"(haircut: {e.bear_case.recommended_size_haircut})")

        result.stage4 = Stage4Output(run_id=rid, enriched_scenarios=enriched)
        result.journal_paths[4] = write_journal(4, rid, result.stage4)
        print(f"  Journal: {result.journal_paths[4]}")

    # ── Stage 5: Synthesis Decision ──
    if start_stage <= 5:
        if result.stage4 is None:
            raise ValueError("Stage 5 requires Stage 4 output")
        print(f"\n[Stage 5] Synthesis — PM deciding on {len(result.stage4.enriched_scenarios)} scenarios...")
        result.stage5 = run_stage5(
            result.stage4,
            account_state,
            settings=settings,
            run_id=rid,
        )
        result.journal_paths[5] = write_journal(5, rid, result.stage5)
        takes = sum(1 for d in result.stage5.decisions if d.decision.value in ("take", "reduce_size"))
        passes = sum(1 for d in result.stage5.decisions if d.decision.value == "pass")
        print(f"  Take/reduce: {takes} | Pass: {passes}")
        print(f"  IC note: {result.stage5.ic_note[:100]}...")
        print(f"  Journal: {result.journal_paths[5]}")

    # ── Stage 6: Action Table ──
    if start_stage <= 6:
        if result.stage5 is None:
            raise ValueError("Stage 6 requires Stage 5 output")
        print(f"\n[Stage 6] Formatting action table...")

        # Build open position details for the formatter
        open_details = {}
        for p in account_state.open_positions:
            open_details[p.ticker] = {
                "held_since": date.today(),  # simplified
                "current_pnl_usd": p.unrealized_pnl_usd,
            }

        result.stage6 = run_stage6(result.stage5, open_details)

        # Enrich Stage 6 rows with risk math numbers from Stage 4
        if result.stage4:
            rm_lookup = {e.scenario_id: e.risk_math for e in result.stage4.enriched_scenarios}
            for row in result.stage6.new_trades:
                rm = rm_lookup.get(row.scenario_id)
                if rm:
                    row.entry_price = rm.proposed_entry
                    row.stop_loss = rm.proposed_stop
                    row.target_price = rm.proposed_target
                    row.r_multiple = rm.r_multiple

        result.journal_paths[6] = write_journal(6, rid, result.stage6)

        # Write the human-readable markdown
        md_content = render_markdown(result.stage6)
        md_path = result.journal_paths[6].parent / f"{date.today().isoformat()}_{rid}.md"
        md_path.write_text(md_content)

        # Write latest.md symlink equivalent
        latest_path = result.journal_paths[6].parent / "latest.md"
        latest_path.write_text(md_content)

        print(f"  New trades: {len(result.stage6.new_trades)}")
        print(f"  Position mgmt: {len(result.stage6.position_management)}")
        print(f"  Journal: {result.journal_paths[6]}")
        print(f"  Action table: {md_path}")

    print(f"\n{'='*60}")
    print(f"Pipeline complete. Run ID: {rid}")
    print(f"{'='*60}\n")

    return result
