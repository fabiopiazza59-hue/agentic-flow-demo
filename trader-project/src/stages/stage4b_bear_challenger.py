"""Stage 4b — Bear Challenger.

Adversarial LLM agent. Receives the original scenario + Stage 4a risk math numbers.
Does NOT see Stage 3's reasoning — only knows the scenario survived.
Web search mandatory (≥3 sources). Historical precedent required.
"""

from __future__ import annotations

import json
from datetime import date

import anthropic
import yaml

from src.config.settings import Settings
from src.utils import extract_json
from src.models.schemas import (
    BearCase,
    EnrichedScenario,
    RiskMath,
    Scenario,
    Verdict,
)

SYSTEM_PROMPT = """\
You are a short-selling research analyst tasked with making the strongest possible case against this trade. Your reputation is built on calling out bad longs (and bad shorts when you're given a short to attack). Do not be balanced. Do not hedge. Make the strongest argument the other side has.

## Your job

For each scenario you receive, produce a bear case that includes:
1. A 3-5 sentence thesis against the trade
2. 3+ specific risks with cited evidence
3. A historical precedent — the last time this kind of trade failed, with date
4. What the market knows that the pitch is missing
5. A recommended size haircut (0.0 = full size OK, 1.0 = don't take the trade)
6. A verdict: dangerous, acceptable_risk, or strong_objection

## Rules

- **Web search is mandatory.** Cite ≥3 distinct sources for your bear case. Pure-reasoning without evidence is unacceptable.
- **Historical precedent required.** "Last time this kind of trade failed" — must be a real, dated, citable instance. If you truly can't find one, say so explicitly: "no recent precedent — this may be a novel setup or I missed it." Never fabricate.
- **No diplomatic hedging.** "There are some concerns…" → unacceptable. Be direct: "This trade is wrong because…"
- **No generic risks.** "Valuation could compress" on every long is lazy. Each risk must be specific to this scenario.
- **Calibrated aggression.** You must distinguish between "this trade has real risks" and "every trade has risks." acceptable_risk should be your modal verdict — if you mark everything dangerous, you're over-tuned.
- **You do NOT know Stage 3's reasoning.** You see the original pitch and the risk math numbers. That's it.

## Verdicts

- **dangerous** — bear case so strong the trade should probably be killed entirely.
- **acceptable_risk** — bear case is real but known/priced; trade can proceed at normal size.
- **strong_objection** — significant but not killer; recommend sizing down.

## Size haircut

If your verdict is strong_objection, recommend a haircut:
- 0.0 = no haircut needed (only for acceptable_risk)
- 0.3 = take 70% of proposed size
- 0.5 = take half
- 1.0 = don't take (only for dangerous)

## Output format

Respond with valid JSON:

{
  "thesis_against": "3-5 sentences, strongest case against",
  "specific_risks": [
    {
      "risk": "description",
      "evidence": "cited source with detail",
      "probability_assessment": "low" | "medium" | "high"
    }
  ],
  "historical_precedent": "real dated instance or explicit 'no recent precedent found'",
  "what_market_knows_that_we_dont": "1-2 sentences",
  "recommended_size_haircut": 0.0-1.0,
  "bear_verdict": "dangerous" | "acceptable_risk" | "strong_objection"
}

Return ONLY valid JSON.
"""


def _format_scenario_for_prompt(
    scenario: Scenario,
    risk_math: RiskMath,
) -> str:
    """Format a single scenario + risk math for the bear challenger."""
    scenario_data = {
        "scenario_id": scenario.scenario_id,
        "direction": scenario.direction.value,
        "instrument": scenario.instrument,
        "thesis_summary": scenario.thesis_summary,
        "proposed_horizon": scenario.proposed_horizon.value,
        "proposed_catalyst": scenario.proposed_catalyst,
        "proposed_kill": scenario.proposed_kill,
        "key_data_points": [
            {"claim": dp.claim, "source": dp.source} for dp in scenario.key_data_points
        ],
    }
    risk_data = {
        "entry": risk_math.proposed_entry,
        "stop": risk_math.proposed_stop,
        "target": risk_math.proposed_target,
        "r_multiple": risk_math.r_multiple,
        "position_size_usd": risk_math.position_size_usd,
        "max_loss_usd": risk_math.max_loss_usd,
        "passes_risk_floor": risk_math.passes_risk_floor,
        "risk_floor_failures": risk_math.risk_floor_failures,
    }
    return (
        "## Scenario to challenge\n\n"
        f"{yaml.dump(scenario_data, default_flow_style=False, sort_keys=False)}\n"
        "## Risk math (from Stage 4a)\n\n"
        f"{yaml.dump(risk_data, default_flow_style=False, sort_keys=False)}"
    )


def run_stage4b_single(
    scenario: Scenario,
    risk_math: RiskMath,
    settings: Settings | None = None,
) -> BearCase:
    """Run the bear challenger on a single scenario.

    Args:
        scenario: Original Stage 2 scenario (not Stage 3 reasoning).
        risk_math: Stage 4a output for this scenario.
        settings: Pipeline settings.

    Returns:
        BearCase with adversarial analysis.
    """
    if settings is None:
        settings = Settings()

    user_message = _format_scenario_for_prompt(scenario, risk_math)

    client = anthropic.Anthropic()

    messages = [{"role": "user", "content": user_message}]
    response = client.messages.create(
        model=settings.pipeline.model,
        max_tokens=4096,
        temperature=settings.pipeline.stage4b_temperature,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 8,
        }],
    )

    # Handle multi-turn tool use
    while response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Search completed.",
                })
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=settings.pipeline.model,
            max_tokens=4096,
            temperature=settings.pipeline.stage4b_temperature,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 8,
            }],
        )

    text_content = ""
    for block in response.content:
        if block.type == "text":
            text_content = block.text

    parsed = extract_json(text_content)
    return BearCase(**parsed)


def run_stage4b_batch(
    scenarios: list[Scenario],
    risk_maths: dict[str, RiskMath],
    settings: Settings | None = None,
) -> dict[str, BearCase]:
    """Run bear challenger on all scenarios sequentially.

    Args:
        scenarios: Original Stage 2 scenarios (only those that survived Stage 3).
        risk_maths: Dict of scenario_id -> RiskMath from Stage 4a.
        settings: Pipeline settings.

    Returns:
        Dict of scenario_id -> BearCase.
    """
    if settings is None:
        settings = Settings()

    results = {}
    for scenario in scenarios:
        if scenario.scenario_id in risk_maths:
            bear_case = run_stage4b_single(
                scenario,
                risk_maths[scenario.scenario_id],
                settings,
            )
            results[scenario.scenario_id] = bear_case
    return results
