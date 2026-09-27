"""Stage 2 — Ticker Discovery.

Translates Stage 1 themes into 3-5 tradeable scenarios each. Creative step:
given a theme, propose the best instrument expressions with reasoning.
LLM-based with web search for catalyst verification and liquidity checks.
"""

from __future__ import annotations

import json
from datetime import date

import anthropic
import yaml

from src.config.settings import Settings
from src.utils import extract_json
from src.models.schemas import (
    Stage1Output,
    Stage2Output,
    Scenario,
    Theme,
)

SYSTEM_PROMPT = """\
You are a trade idea generator on a multi-strategy desk. Your job is to translate macro themes into specific tradeable instruments. You are creative but disciplined.

## Your job

For each theme you receive, propose 1-3 ticker scenarios. Pick the best expression as the primary scenario. List alternatives in alternative_expressions.

## Rules

- **Direction is theme-derived.** If the theme is long_bias, scenarios should be long. If short_bias, short. If mixed, you can produce both. Don't invent direction.
- **Catalyst and kill are mandatory.** Every scenario needs:
  - A named catalyst with an approximate date. "Eventually" or "later this year" → drop the scenario.
  - A specific kill condition (price level or event). Vague kill → drop.
- **Liquidity floor.** Each instrument must have median daily volume ≥ $5M USD. Microcaps are dropped. Verify via web search if unsure.
- **CFD-tradeable universe.** Limit to: liquid US/EU equities, major ETFs, occasional commodity proxies (e.g., URA). No options, no futures, no individual bonds.
- **No theme without ≥1 scenario.** If a theme produces zero viable scenarios, log it in a dropped_themes field with reason.
- **Symmetric direction continues.** If a theme is short_bias, produce short scenarios. No quiet conversion to longs.
- **Max 3 scenarios per theme** as primary records. Others go in alternative_expressions.

## The "why this expression" field

This is the most valuable field. It must answer: "of the alternative tickers, why is this one the cleanest play?" Acceptable reasons:
- Pure-play exposure (highest revenue concentration)
- Liquidity advantage
- Asymmetric setup (cycle positioning)
- Balance sheet quality

Unacceptable (drop the scenario):
- "It's the most well-known" (lazy)
- "It's trending" (recency bias)
- "Best performing recently" (chasing)

## Anti-patterns to avoid

1. Ticker dump — proposing 5+ scenarios per theme. Cap at 3 primary.
2. Re-using the same ticker across themes without flagging it.
3. Inventing catalysts — if web search doesn't find a real dated catalyst, drop it.
4. ETF cop-out — don't default to broad sector ETFs when a clean single-name exists.
5. Liquidity violations — pure + untradeable = useless on a $2k CFD account.

## Output format

Respond with valid JSON:

{
  "scenarios": [
    {
      "scenario_id": "S-001",
      "parent_theme_id": "T-001",
      "direction": "long" | "short" | "pair" | "avoid",
      "instrument": "TICKER",
      "instrument_class": "equity" | "etf" | "commodity_etf",
      "thesis_summary": "2-4 sentences",
      "proposed_horizon": "days" | "weeks" | "months",
      "proposed_catalyst": "named event + approximate date",
      "proposed_kill": "specific price level or event",
      "key_data_points": [
        {"claim": "...", "source": "...", "url": "..."}
      ],
      "why_this_expression": "1-2 sentences",
      "alternative_expressions": ["TICKER2", "TICKER3"]
    }
  ],
  "dropped_themes": [
    {"theme_id": "T-XXX", "reason": "..."}
  ]
}

Number scenario_ids sequentially: S-001, S-002, etc. Return ONLY valid JSON.
"""


def _format_themes_for_prompt(themes: list[Theme]) -> str:
    """Format themes as YAML for the LLM."""
    records = []
    for t in themes:
        records.append({
            "theme_id": t.theme_id,
            "lens": t.lens,
            "title": t.title,
            "summary": t.summary,
            "direction_implication": t.direction_implication.value,
            "horizon": t.horizon.value,
            "evidence": [
                {"claim": e.claim, "source": e.source}
                for e in t.evidence
            ],
            "contradicting_evidence": t.contradicting_evidence,
            "confidence": t.confidence.value,
        })
    return yaml.dump(records, default_flow_style=False, sort_keys=False)


def run_stage2(
    stage1_output: Stage1Output,
    settings: Settings | None = None,
    run_id: str | None = None,
) -> Stage2Output:
    """Run Stage 2 ticker discovery on Stage 1 themes.

    Args:
        stage1_output: Parsed Stage 1 output with themes.
        settings: Pipeline settings.
        run_id: Override run ID.

    Returns:
        Stage2Output with scenarios.
    """
    if settings is None:
        settings = Settings()

    rid = run_id or stage1_output.run_id
    themes = stage1_output.themes

    user_message = (
        f"Run ID: {rid}\n"
        f"Date: {date.today().isoformat()}\n"
        f"Themes to translate into tradeable scenarios ({len(themes)} total):\n\n"
        f"{_format_themes_for_prompt(themes)}"
    )

    client = anthropic.Anthropic()

    messages = [{"role": "user", "content": user_message}]
    response = client.messages.create(
        model=settings.pipeline.model,
        max_tokens=8192,
        temperature=0.4,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 15,
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
            max_tokens=8192,
            temperature=0.4,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 15,
            }],
        )

    # Extract final text
    text_content = ""
    for block in response.content:
        if block.type == "text":
            text_content = block.text

    parsed = extract_json(text_content)
    scenarios = [Scenario(**s) for s in parsed["scenarios"]]

    return Stage2Output(run_id=rid, scenarios=scenarios)
