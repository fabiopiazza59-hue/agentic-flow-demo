"""Stage 6 — Action Table Formatter.

Pure formatter. No LLM, no judgment. Takes Stage 5 decisions and produces
two artifacts: actions.md (human read) and actions.yaml (machine read).
"""

from __future__ import annotations

from datetime import datetime

import yaml

from src.models.schemas import (
    Decision,
    Direction,
    NewTradeRow,
    PositionCheck,
    PositionManagementRow,
    Stage5Output,
    Stage6Output,
    TradeAction,
    TradeDecision,
)


def run_stage6(
    stage5_output: Stage5Output,
    open_positions_details: dict | None = None,
) -> Stage6Output:
    """Format Stage 5 decisions into the action table.

    Args:
        stage5_output: Parsed Stage 5 output.
        open_positions_details: Optional dict of ticker -> {held_since, current_pnl_usd}
            for position management rows.

    Returns:
        Stage6Output with new_trades and position_management.
    """
    if open_positions_details is None:
        open_positions_details = {}

    new_trades = _build_new_trades(stage5_output.decisions)
    position_mgmt = _build_position_management(
        stage5_output.position_checks, open_positions_details
    )

    return Stage6Output(
        run_id=stage5_output.run_id,
        generated_at=datetime.now(),
        new_trades=new_trades,
        position_management=position_mgmt,
        portfolio_after=stage5_output.portfolio_after,
        ic_note=stage5_output.ic_note,
    )


def _build_new_trades(decisions: list[TradeDecision]) -> list[NewTradeRow]:
    """Convert take/reduce_size decisions into action table rows.

    Sorted by conviction descending (approximated by size — larger = higher conviction).
    Max 4 rows per spec.
    """
    actionable = [
        d for d in decisions
        if d.decision in (Decision.take, Decision.reduce_size)
    ]

    rows = []
    for d in actionable:
        action = (
            TradeAction.BUY
            if d.final_direction in (Direction.long, Direction.pair)
            else TradeAction.SELL_SHORT
        )

        # Reason: "<catalyst>; bear: <bear point>"
        # Truncate to 150 chars
        reason = d.rationale[:100]
        if d.risks_acknowledged:
            bear_part = f"; bear: {d.risks_acknowledged}"
            reason = reason[: 150 - len(bear_part)] + bear_part
        reason = reason[:150]

        rows.append(NewTradeRow(
            ticker=_extract_ticker(d),
            action=action,
            entry_price=0,  # filled by caller from Stage 4a
            stop_loss=0,
            target_price=0,
            size_usd=d.final_size_usd,
            r_multiple=0,
            reason=reason,
            scenario_id=d.scenario_id,
        ))

    # Sort by size descending (proxy for conviction)
    rows.sort(key=lambda r: r.size_usd, reverse=True)
    return rows[:4]  # max 4 per spec


def _build_position_management(
    checks: list[PositionCheck],
    details: dict,
) -> list[PositionManagementRow]:
    """Convert position checks into management rows. Max 6."""
    rows = []
    for check in checks:
        info = details.get(check.ticker, {})
        rows.append(PositionManagementRow(
            ticker=check.ticker,
            held_since=info.get("held_since", datetime.now().date()),
            current_pnl_usd=info.get("current_pnl_usd", 0),
            action=check.action,
            action_size_pct=check.action_size_pct,
            reason=check.reason[:100],
        ))

    # Sort by held_since ascending (oldest first)
    rows.sort(key=lambda r: r.held_since)
    return rows[:6]


def _extract_ticker(decision: TradeDecision) -> str:
    """Extract ticker from scenario_id or decision context."""
    return decision.scenario_id


def render_markdown(output: Stage6Output) -> str:
    """Render the action table as markdown (actions.md)."""
    lines = [f"# Weekly Action Table — {output.generated_at.strftime('%Y-%m-%d')}"]
    lines.append("")

    if output.new_trades:
        lines.append("## New trades")
        lines.append("")
        lines.append("| Ticker | Action | Entry | Stop | Target | Size $ | R | Reason |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for t in output.new_trades:
            lines.append(
                f"| {t.ticker} | {t.action.value} | ${t.entry_price:.2f} | "
                f"${t.stop_loss:.2f} | ${t.target_price:.2f} | "
                f"${t.size_usd:.0f} | {t.r_multiple:.1f} | {t.reason} |"
            )
    else:
        lines.append("**No new trades this week.**")

    lines.append("")

    if output.position_management:
        lines.append("## Position management")
        lines.append("")
        lines.append("| Ticker | Held since | Current P&L | Action | Reason |")
        lines.append("|---|---|---|---|---|")
        for p in output.position_management:
            pnl_str = f"+${p.current_pnl_usd:.0f}" if p.current_pnl_usd >= 0 else f"-${abs(p.current_pnl_usd):.0f}"
            action_str = p.action.value
            if p.action_size_pct is not None:
                action_str += f" ({p.action_size_pct:.0f}%)"
            lines.append(
                f"| {p.ticker} | {p.held_since} | {pnl_str} | {action_str} | {p.reason} |"
            )
    else:
        lines.append("All open positions: HOLD")

    lines.append("")
    lines.append("## Portfolio after")
    lines.append("")
    lines.append(f"- Heat: {output.portfolio_after.projected_heat_pct:.1f}% of capital")
    lines.append(f"- New positions: {output.portfolio_after.new_positions_count}")
    lines.append(f"- Net direction: {output.portfolio_after.net_direction.value}")

    lines.append("")
    lines.append("## IC note")
    lines.append("")
    lines.append(output.ic_note)

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Run ID: {output.run_id} | Pipeline version: {output.pipeline_version} "
        f"| Generated {output.generated_at.isoformat()}*"
    )

    return "\n".join(lines)


def render_yaml(output: Stage6Output) -> str:
    """Render the action table as YAML (actions.yaml)."""
    data = output.model_dump(mode="json")
    return yaml.dump(data, default_flow_style=False, sort_keys=False)
