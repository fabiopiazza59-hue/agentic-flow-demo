#!/usr/bin/env python3
"""Streamlit UI for the Seamaster Trading Advisor Pipeline."""

from __future__ import annotations

import os
import sys
import uuid
from datetime import date, datetime
from pathlib import Path

import streamlit as st
import yaml

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.settings import Settings, AccountConfig, PipelineConfig
from src.models.schemas import (
    AccountState,
    Decision,
    Direction,
    EnrichedScenario,
    OpenPosition,
    Stage2Output,
    Stage4Output,
    Verdict,
)
from src.stages.stage1_market_scan import run_stage1
from src.stages.stage2_ticker_discovery import run_stage2
from src.stages.stage3_triage import run_stage3, load_stage2_from_yaml
from src.stages.stage4a_risk_math import compute_risk_math, parse_levels_from_scenario
from src.stages.stage4b_bear_challenger import run_stage4b_batch
from src.stages.stage5_synthesis import run_stage5
from src.stages.stage6_action_table import run_stage6, render_markdown
from src.journal import write_journal

# ── Page config ──

st.set_page_config(
    page_title="Seamaster — Trading Advisor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──

st.markdown("""
<style>
    .stage-header {
        border-left: 4px solid #6366f1;
        padding-left: 12px;
        margin: 20px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──

with st.sidebar:
    st.title("Seamaster")
    st.caption("Six-stage trading advisor pipeline")

    st.divider()

    # API Key
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Your Anthropic API key (sk-ant-...)",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()

    # Account config
    st.subheader("Account")
    capital = st.number_input("Capital (USD)", value=2000, step=500, min_value=100)
    max_risk_pct = st.slider("Max risk per trade (%)", 1.0, 5.0, 2.0, 0.5) / 100
    max_heat_pct = st.slider("Max portfolio heat (%)", 2.0, 10.0, 6.0, 1.0) / 100

    st.divider()

    # Pipeline config
    st.subheader("Pipeline")
    model = st.selectbox("Model", [
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
    ], index=0)

    start_mode = st.radio(
        "Pipeline mode",
        ["Stages 3-6 (from scenarios)", "Stages 1-6 (full pipeline)"],
        index=0,
        help="Stage 3-6 uses fixture or uploaded scenarios. Full pipeline runs Stage 1+2 first (web search, slower).",
    )

    st.divider()
    st.caption("v0.2 | Spec-driven pipeline")


# ── Build settings ──

def get_settings() -> Settings:
    return Settings(
        account=AccountConfig(
            total_capital_usd=capital,
            max_loss_per_trade_pct=max_risk_pct,
            max_portfolio_heat_pct=max_heat_pct,
        ),
        pipeline=PipelineConfig(model=model),
    )


# ── Main area ──

st.header("Seamaster — Trading Advisor Pipeline")

# Tab layout
tab_run, tab_journal = st.tabs(["Run Pipeline", "Journal"])

# ── Run Pipeline tab ──

with tab_run:
    # ── Scenario input section ──
    # Always visible so stage2_output is always defined before the button

    stage2_output: Stage2Output | None = None
    prices: dict[str, float] = {}

    if start_mode == "Stages 3-6 (from scenarios)":
        st.subheader("Stage 2 Input")
        input_mode = st.radio(
            "Scenario source",
            ["Use sample fixture", "Upload YAML"],
            horizontal=True,
        )

        if input_mode == "Upload YAML":
            uploaded = st.file_uploader("Upload Stage 2 output YAML", type=["yaml", "yml"])
            if uploaded:
                stage2_data = yaml.safe_load(uploaded.read())
                stage2_output = Stage2Output(**stage2_data)
                st.success(f"Loaded {len(stage2_output.scenarios)} scenarios")
        else:
            stage2_output = load_stage2_from_yaml("fixtures/sample_stage2_output.yaml")
            st.info(f"Using sample fixture: {len(stage2_output.scenarios)} scenarios")
    else:
        st.info(
            "**Full pipeline mode** — Stage 1 will scan the web for themes, "
            "Stage 2 will generate scenarios. This takes 2-5 minutes and uses web search."
        )

    # ── Show scenario preview + price inputs when we have scenarios ──

    if stage2_output is not None:
        with st.expander(f"Preview scenarios ({len(stage2_output.scenarios)})", expanded=False):
            for s in stage2_output.scenarios:
                direction_icon = {"long": "📈", "short": "📉", "pair": "🔄", "avoid": "⛔"}.get(s.direction.value, "")
                st.markdown(f"**{s.scenario_id}** — {direction_icon} {s.direction.value.upper()} {s.instrument} ({s.instrument_class.value})")
                st.caption(s.thesis_summary)
                st.caption(f"Catalyst: {s.proposed_catalyst} | Kill: {s.proposed_kill}")
                st.divider()

        st.subheader("Current Prices")
        st.caption("Required for Stage 4a risk math. Enter current market prices.")

        tickers = sorted({s.instrument for s in stage2_output.scenarios})
        default_prices = {
            "MU": 268.50, "AMAT": 195.00, "KRE": 48.20, "CCJ": 55.00,
            "OKLO": 28.00, "NVDA": 135.00, "GLD": 230.00,
        }

        cols = st.columns(min(len(tickers), 4))
        for i, ticker in enumerate(tickers):
            with cols[i % len(cols)]:
                prices[ticker] = st.number_input(
                    f"{ticker}",
                    value=default_prices.get(ticker, 100.0),
                    step=0.50,
                    format="%.2f",
                    key=f"price_{ticker}",
                )

    # ── Run button ──

    st.divider()

    if st.button("Run Pipeline", type="primary", use_container_width=True):

        # ── Validation ──
        if not api_key:
            st.error("Set your Anthropic API key in the sidebar.")
            st.stop()

        settings = get_settings()
        run_id = f"run-{date.today().isoformat()}-{uuid.uuid4().hex[:6]}"

        account_state = AccountState(
            total_capital_usd=capital,
            available_capital_usd=capital,
            current_portfolio_heat_pct=0.0,
        )

        if "results" not in st.session_state:
            st.session_state.results = {}

        # ── Stages 1+2 (full pipeline mode only) ──

        if start_mode == "Stages 1-6 (full pipeline)":
            st.markdown('<div class="stage-header"><h3>Stage 1 — Market Scan</h3></div>', unsafe_allow_html=True)
            with st.spinner("Journalist + macro lenses scanning the market..."):
                try:
                    stage1_result = run_stage1(settings, run_id=run_id)
                    write_journal(1, run_id, stage1_result)
                    st.session_state.results["stage1"] = stage1_result
                except Exception as e:
                    st.error(f"Stage 1 failed: {e}")
                    st.stop()

            st.success(f"Stage 1 complete: {len(stage1_result.themes)} themes")
            st.info(f"**Desk note:** {stage1_result.desk_note}")

            with st.expander("Themes", expanded=False):
                for t in stage1_result.themes:
                    st.markdown(f"**{t.theme_id}** [{t.lens}] — {t.title}")
                    st.caption(t.summary)
                    st.divider()

            st.markdown('<div class="stage-header"><h3>Stage 2 — Ticker Discovery</h3></div>', unsafe_allow_html=True)
            with st.spinner("Generating tradeable scenarios from themes..."):
                try:
                    stage2_output = run_stage2(stage1_result, settings, run_id=run_id)
                    write_journal(2, run_id, stage2_output)
                    st.session_state.results["stage2"] = stage2_output
                except Exception as e:
                    st.error(f"Stage 2 failed: {e}")
                    st.stop()

            st.success(f"Stage 2 complete: {len(stage2_output.scenarios)} scenarios")

            # Collect prices — for full pipeline, use placeholder prices
            # (in production, these would be fetched live)
            for s in stage2_output.scenarios:
                if s.instrument not in prices:
                    prices[s.instrument] = 100.0  # placeholder
            st.warning(
                "Full pipeline generated new tickers. Price inputs above may not "
                "include them — using $100 placeholder for unknown tickers. "
                "Re-run in Stage 3-6 mode with correct prices for accurate risk math."
            )

        # ── Validate stage2_output exists before proceeding ──

        if stage2_output is None:
            st.error("No scenarios loaded. Select a fixture or upload a YAML file.")
            st.stop()

        # ── Stage 3 — Triage Gate ──

        st.markdown('<div class="stage-header"><h3>Stage 3 — Triage Gate</h3></div>', unsafe_allow_html=True)
        with st.spinner("Skeptical PM reviewing scenarios..."):
            try:
                stage3_result = run_stage3(stage2_output, settings, run_id=run_id)
                write_journal(3, run_id, stage3_result)
                st.session_state.results["stage3"] = stage3_result
            except Exception as e:
                st.error(f"Stage 3 failed: {e}")
                st.stop()

        # Display verdicts
        summary = stage3_result.batch_summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Scenarios In", summary.scenarios_in)
        col2.metric("LIVE", summary.verdicts.get("LIVE", 0))
        col3.metric("WATCH", summary.verdicts.get("WATCH", 0))
        col4.metric("KILL", summary.verdicts.get("KILL", 0))

        for v in stage3_result.verdicts:
            with st.expander(
                f"{'✅' if v.verdict == Verdict.LIVE else '⏸️' if v.verdict == Verdict.WATCH else '❌'} "
                f"{v.scenario_id} → {v.verdict.value} — {v.one_line_reason}",
                expanded=(v.verdict == Verdict.LIVE),
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Trade:** {v.answers.trade_in_one_sentence}")
                    st.markdown(f"**Variant perception:** {v.answers.variant_perception}")
                    st.markdown(f"**Asymmetry:** {v.answers.asymmetry}")
                    st.markdown(f"**Catalyst:** {v.answers.catalyst_and_date}")
                with col_b:
                    st.markdown(f"**Crowding:** `{v.answers.crowding.value}`")
                    st.markdown(f"**Edge source:** `{v.answers.edge_source.value}`")
                    st.markdown(f"**Kill scenario:** {v.answers.kill_scenario}")
                    st.markdown(f"**Conviction:** `{v.answers.conviction.value}`")
                if v.answers.not_shorting_because:
                    st.markdown(f"**Not shorting because:** {v.answers.not_shorting_because}")

        st.info(f"**Desk note:** {summary.desk_note}")

        # ── Filter survivors ──

        surviving_ids = {
            v.scenario_id for v in stage3_result.verdicts
            if v.verdict in (Verdict.LIVE, Verdict.WATCH)
        }
        surviving_scenarios = [
            s for s in stage2_output.scenarios if s.scenario_id in surviving_ids
        ]

        if not surviving_scenarios:
            st.warning("All scenarios killed. Pipeline stops here — no trades this week.")
            st.stop()

        # ── Stage 4a — Risk Math ──

        st.markdown('<div class="stage-header"><h3>Stage 4a — Risk Math</h3></div>', unsafe_allow_html=True)

        enriched = []
        risk_maths = {}

        for scenario in surviving_scenarios:
            price = prices.get(scenario.instrument, 0)
            if price == 0:
                st.warning(f"No price for {scenario.instrument} — skipping")
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

        # Display risk math table
        risk_data = []
        for e in enriched:
            s = next((x for x in stage2_output.scenarios if x.scenario_id == e.scenario_id), None)
            ticker = s.instrument if s else e.scenario_id
            rm = e.risk_math
            risk_data.append({
                "Ticker": ticker,
                "Entry": f"${rm.proposed_entry:.2f}",
                "Stop": f"${rm.proposed_stop:.2f}",
                "Target": f"${rm.proposed_target:.2f}",
                "R": f"{rm.r_multiple:.1f}",
                "Size": f"${rm.position_size_usd:.0f}",
                "Max Loss": f"${rm.max_loss_usd:.0f}",
                "Heat After": f"{rm.portfolio_heat_after:.1%}",
                "CFD Break-even": f"{rm.cfd_breakeven_days}d",
                "Floor": "✅" if rm.passes_risk_floor else f"❌ {', '.join(rm.risk_floor_failures)}",
            })
        st.dataframe(risk_data, use_container_width=True, hide_index=True)

        # ── Stage 4b — Bear Challenger ──

        st.markdown('<div class="stage-header"><h3>Stage 4b — Bear Challenger</h3></div>', unsafe_allow_html=True)
        with st.spinner("Adversarial analyst building bear cases..."):
            try:
                bear_cases = run_stage4b_batch(surviving_scenarios, risk_maths, settings)
                for e in enriched:
                    if e.scenario_id in bear_cases:
                        e.bear_case = bear_cases[e.scenario_id]
            except Exception as ex:
                st.error(f"Stage 4b failed: {ex}")
                st.stop()

        stage4_output = Stage4Output(run_id=run_id, enriched_scenarios=enriched)
        write_journal(4, run_id, stage4_output)
        st.session_state.results["stage4"] = stage4_output

        for e in enriched:
            s = next((x for x in stage2_output.scenarios if x.scenario_id == e.scenario_id), None)
            ticker = s.instrument if s else e.scenario_id
            if e.bear_case:
                bc = e.bear_case
                verdict_icon = {"dangerous": "🔴", "acceptable_risk": "🟢", "strong_objection": "🟡"}
                with st.expander(
                    f"{verdict_icon.get(bc.bear_verdict.value, '⚪')} {ticker} — "
                    f"{bc.bear_verdict.value.replace('_', ' ').title()} "
                    f"(haircut: {bc.recommended_size_haircut:.0%})",
                    expanded=True,
                ):
                    st.markdown(f"**Thesis against:** {bc.thesis_against}")
                    for risk in bc.specific_risks:
                        st.markdown(
                            f"- **{risk.risk}** ({risk.probability_assessment.value}) — "
                            f"{risk.evidence}"
                        )
                    st.markdown(f"**Historical precedent:** {bc.historical_precedent}")
                    st.markdown(f"**What market knows:** {bc.what_market_knows_that_we_dont}")

        # ── Stage 5 — Synthesis Decision ──

        st.markdown('<div class="stage-header"><h3>Stage 5 — Synthesis Decision</h3></div>', unsafe_allow_html=True)
        with st.spinner("Senior PM making final decisions..."):
            try:
                stage5_result = run_stage5(
                    stage4_output, account_state,
                    settings=settings, run_id=run_id,
                )
                write_journal(5, run_id, stage5_result)
                st.session_state.results["stage5"] = stage5_result
            except Exception as ex:
                st.error(f"Stage 5 failed: {ex}")
                st.stop()

        # Decision summary
        takes = [d for d in stage5_result.decisions if d.decision in (Decision.take, Decision.reduce_size)]
        passes = [d for d in stage5_result.decisions if d.decision == Decision.pass_]

        col1, col2, col3 = st.columns(3)
        col1.metric("Take / Reduce", len(takes))
        col2.metric("Pass", len(passes))
        col3.metric("Projected Heat", f"{stage5_result.portfolio_after.projected_heat_pct:.1f}%")

        for d in stage5_result.decisions:
            s = next((x for x in stage2_output.scenarios if x.scenario_id == d.scenario_id), None)
            ticker = s.instrument if s else d.scenario_id
            decision_icon = {
                "take": "🟢", "reduce_size": "🟡", "pass": "🔴", "defer": "⏳"
            }.get(d.decision.value, "⚪")

            with st.expander(
                f"{decision_icon} {ticker} — {d.decision.value.upper()} "
                f"{'$' + str(int(d.final_size_usd)) if d.final_size_usd else ''}",
                expanded=(d.decision in (Decision.take, Decision.reduce_size)),
            ):
                st.markdown(f"**Rationale:** {d.rationale}")
                if d.risks_acknowledged:
                    st.markdown(f"**Risks acknowledged:** {d.risks_acknowledged}")
                if d.interaction_with_book:
                    st.markdown(f"**Book interaction:** {d.interaction_with_book}")
                if d.defer_until:
                    st.markdown(f"**Defer until:** {d.defer_until}")

        st.info(f"**IC Note:** {stage5_result.ic_note}")

        # ── Stage 6 — Action Table ──

        st.markdown('<div class="stage-header"><h3>Stage 6 — Action Table</h3></div>', unsafe_allow_html=True)

        stage6_result = run_stage6(stage5_result)

        # Enrich with risk math
        rm_lookup = {e.scenario_id: e.risk_math for e in enriched}
        for row in stage6_result.new_trades:
            rm = rm_lookup.get(row.scenario_id)
            if rm:
                row.entry_price = rm.proposed_entry
                row.stop_loss = rm.proposed_stop
                row.target_price = rm.proposed_target
                row.r_multiple = rm.r_multiple

        write_journal(6, run_id, stage6_result)
        st.session_state.results["stage6"] = stage6_result

        # Render the markdown action table
        md = render_markdown(stage6_result)
        st.markdown(md)

        # Write latest.md
        journal_dir = Path("journal/stage6")
        journal_dir.mkdir(parents=True, exist_ok=True)
        (journal_dir / "latest.md").write_text(md)
        (journal_dir / f"{date.today().isoformat()}_{run_id}.md").write_text(md)

        st.success(f"Pipeline complete. Run ID: `{run_id}`")

# ── Journal tab ──

with tab_journal:
    st.subheader("Audit Trail")

    journal_root = Path("journal")
    if not journal_root.exists():
        st.info("No journal entries yet. Run the pipeline first.")
    else:
        for stage_num in range(1, 7):
            stage_dir = journal_root / f"stage{stage_num}"
            if stage_dir.exists():
                files = sorted(stage_dir.glob("*.yaml"), reverse=True)
                if files:
                    with st.expander(f"Stage {stage_num} — {len(files)} entries", expanded=False):
                        for f in files[:10]:
                            st.caption(f.name)
                            with open(f) as fh:
                                content = fh.read()
                            st.code(content[:3000], language="yaml")
                            if len(content) > 3000:
                                st.caption("... (truncated)")

        # Show latest action table
        latest = journal_root / "stage6" / "latest.md"
        if latest.exists():
            st.divider()
            st.subheader("Latest Action Table")
            st.markdown(latest.read_text())
