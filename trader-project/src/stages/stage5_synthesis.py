"""Stage 5 — Senior PM Synthesis & Decision.

The decision-maker. Receives Stage 4 enriched scenarios and makes take/pass/
reduce_size/defer calls. Sees account state and open positions. Default: pass.

Different persona from Stage 3 — this is the PM who commits capital, not the
skeptic who kills ideas.
"""

from __future__ import annotations

import json
from datetime import date

import anthropic
import yaml

from src.config.settings import Settings
from src.models.schemas import (
    AccountState,
    Decision,
    EnrichedScenario,
    PortfolioAfter,
    PositionCheck,
    Stage4Output,
    Stage5Output,
    TradeDecision,
)

SYSTEM_PROMPT = """\
You are a senior portfolio manager with a real book who has lost real money. You've earned the right to take risk because you've calibrated your judgment over cycles. You are decisive but not impulsive.

## Your job

For each enriched scenario (thesis + risk math + bear case), make a take-or-pass decision. You also review open positions.

## Your disposition

- **Default action: pass.** Inertia is the right default. A trade gets taken only when evidence overcomes inertia.
- **Conviction over completeness.** Better to take 2 trades with high conviction than 5 with medium.
- **Honor the bear case.** The Stage 4b bear case is read carefully. If bear_verdict is "dangerous", your default is pass — override only when you can explicitly articulate why the bear is wrong (reference the bear's evidence, not just dismiss it).
- **Honor the risk math.** If passes_risk_floor is false, that's a near-veto. You can override only by writing "risk floor breach acknowledged because [specific reason]" and reducing size.
- **Portfolio thinking.** Evaluate each trade against the existing book. A third semis long when the book is already concentrated in AI is sized down or passed.
- **Calibrated conviction.** Talk in R multiples and probability-adjusted returns. No "this could be a 5x" lottery-ticket framing.

## Decision taxonomy

- **take** — full size as per Stage 4a. Enters the action table.
- **pass** — no action. Logged with reason.
- **reduce_size** — take, but smaller than Stage 4a recommended. Bear case heeded.
- **defer** — concept right, timing wrong. Must specify a specific event in defer_until. Vague defer ("when conditions improve") → convert to pass.

## Behavioral rules

- Every decision rationale must reference: (a) original thesis, (b) bear case (even if dismissed), (c) risk math fit with the book.
- Never override quantitative risk floors silently. If risk math fails → pass or reduce_size unless explicit override written.
- If account is in drawdown >5% from peak, reduce all sizes by 30% (multiply Stage 4a recommendation by 0.7).
- If bear_verdict is "dangerous", any "take" requires explicit reasoning against the bear's evidence.
- reduce_size should be 20-30% of decisions, not the modal answer. Don't be a coward.
- Approval rate target: 30-50% of input scenarios.

## Open position review

For each open position in the account state:
- **HOLD** — thesis intact, no action.
- **TRIM** — reduce size, partial profit-take or de-risk. Specify action_size_pct.
- **EXIT** — close. Reason required.
- **ADD** — only if a new scenario for the same ticker justifies adding. Rare.

## Output format

Respond with valid JSON:

{
  "decisions": [
    {
      "scenario_id": "S-001",
      "decision": "take" | "pass" | "reduce_size" | "defer",
      "final_direction": "long" | "short",
      "final_size_usd": <number>,
      "rationale": "3-5 sentences — why take, why now, why this size",
      "risks_acknowledged": "1-2 sentences naming the strongest bear point",
      "defer_until": "YYYY-MM-DD" | null,
      "interaction_with_book": "1 sentence — how this fits with open positions"
    }
  ],
  "position_checks": [
    {
      "ticker": "CCJ",
      "action": "HOLD" | "TRIM" | "EXIT" | "ADD",
      "action_size_pct": <number> | null,
      "reason": "≤100 chars"
    }
  ],
  "portfolio_after": {
    "projected_heat_pct": <number>,
    "new_positions_count": <int>,
    "net_direction": "long_skew" | "short_skew" | "balanced"
  },
  "ic_note": "3-5 sentences — overall decision-making logic for the week"
}

Return ONLY valid JSON.
"""


def _format_enriched_scenarios(
    enriched: list[EnrichedScenario],
    account_state: AccountState,
) -> str:
    """Format enriched scenarios + account state for the PM."""
    account_data = {
        "total_capital_usd": account_state.total_capital_usd,
        "available_capital_usd": account_state.available_capital_usd,
        "current_portfolio_heat_pct": account_state.current_portfolio_heat_pct,
        "open_positions": [
            {
                "ticker": p.ticker,
                "direction": p.direction.value,
                "entry": p.entry,
                "current_price": p.current_price,
                "stop": p.stop,
                "unrealized_pnl_usd": p.unrealized_pnl_usd,
                "days_held": p.days_held,
            }
            for p in account_state.open_positions
        ],
    }

    scenarios_data = []
    for e in enriched:
        record = {
            "scenario_id": e.scenario_id,
            "stage3_verdict": e.stage3_verdict.value,
            "current_price": e.current_price,
            "risk_math": {
                "entry": e.risk_math.proposed_entry,
                "stop": e.risk_math.proposed_stop,
                "target": e.risk_math.proposed_target,
                "r_multiple": e.risk_math.r_multiple,
                "position_size_usd": e.risk_math.position_size_usd,
                "max_loss_usd": e.risk_math.max_loss_usd,
                "max_loss_pct_account": e.risk_math.max_loss_pct_account,
                "portfolio_heat_after": e.risk_math.portfolio_heat_after,
                "passes_risk_floor": e.risk_math.passes_risk_floor,
                "risk_floor_failures": e.risk_math.risk_floor_failures,
                "cfd_breakeven_days": e.risk_math.cfd_breakeven_days,
            },
        }
        if e.bear_case:
            record["bear_case"] = {
                "thesis_against": e.bear_case.thesis_against,
                "specific_risks": [
                    {
                        "risk": r.risk,
                        "evidence": r.evidence,
                        "probability_assessment": r.probability_assessment.value,
                    }
                    for r in e.bear_case.specific_risks
                ],
                "historical_precedent": e.bear_case.historical_precedent,
                "what_market_knows_that_we_dont": e.bear_case.what_market_knows_that_we_dont,
                "recommended_size_haircut": e.bear_case.recommended_size_haircut,
                "bear_verdict": e.bear_case.bear_verdict.value,
            }
        scenarios_data.append(record)

    return (
        "## Account state\n\n"
        f"{yaml.dump(account_data, default_flow_style=False, sort_keys=False)}\n"
        "## Enriched scenarios for decision\n\n"
        f"{yaml.dump(scenarios_data, default_flow_style=False, sort_keys=False)}"
    )


def run_stage5(
    stage4_output: Stage4Output,
    account_state: AccountState,
    original_scenarios: list | None = None,
    settings: Settings | None = None,
    run_id: str | None = None,
) -> Stage5Output:
    """Run Stage 5 synthesis decision.

    Args:
        stage4_output: Enriched scenarios from Stage 4.
        account_state: Current account with open positions.
        original_scenarios: Original Stage 2 scenarios (for context).
        settings: Pipeline settings.
        run_id: Override run ID.

    Returns:
        Stage5Output with decisions and position checks.
    """
    if settings is None:
        settings = Settings()

    rid = run_id or stage4_output.run_id
    enriched = stage4_output.enriched_scenarios

    # Apply drawdown multiplier if needed
    drawdown_note = ""
    if account_state.total_capital_usd > 0:
        # Simplified: check if current heat indicates drawdown
        # In production, compare against peak equity
        pass

    user_message = (
        f"Run ID: {rid}\n"
        f"Date: {date.today().isoformat()}\n"
        f"Enriched scenarios for decision ({len(enriched)} total):\n\n"
        f"{_format_enriched_scenarios(enriched, account_state)}"
    )

    client = anthropic.Anthropic()

    response = client.messages.create(
        model=settings.pipeline.model,
        max_tokens=8192,
        temperature=settings.pipeline.stage5_temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text
    parsed = json.loads(raw_text)

    decisions = [TradeDecision(**d) for d in parsed["decisions"]]
    position_checks = [
        PositionCheck(**p) for p in parsed.get("position_checks", [])
    ]
    portfolio_after = PortfolioAfter(**parsed["portfolio_after"])

    return Stage5Output(
        run_id=rid,
        decisions=decisions,
        position_checks=position_checks,
        portfolio_after=portfolio_after,
        ic_note=parsed["ic_note"],
    )
