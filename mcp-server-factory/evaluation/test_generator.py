"""
Synthetic Test Case Generator for MCP Tools.

Generates test cases based on tool descriptions.
"""

import os
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from registry.schemas import ToolDefinition


GENERATOR_PROMPT = """Generate {num_cases} realistic test cases for this MCP tool.

TOOL: {tool_name}
DESCRIPTION: {tool_description}
PARAMETERS: {parameters}

For each test case, create a natural user query that would use this tool,
and a realistic mock response the tool would return.

Format your response as JSON:
{{
    "test_cases": [
        {{
            "query": "natural language query from user",
            "expected_params": {{"param1": "value1"}},
            "response": {{"key": "mock response data"}}
        }}
    ]
}}

Make the queries diverse and realistic. Include edge cases like:
- Simple, direct requests
- Requests with multiple parameters
- Requests with typos or informal language
- Requests that push parameter boundaries

Return ONLY valid JSON, no other text."""


class TestCaseGenerator:
    """Generate synthetic test cases for tool evaluation."""

    def __init__(self, model: str = "claude-3-5-haiku-latest"):
        """
        Initialize the generator.

        Args:
            model: LLM model to use for generation
        """
        self.model = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.llm = ChatAnthropic(
                model=model,
                api_key=api_key,
                max_tokens=2000,
            )
        else:
            self.llm = None

    async def generate_for_tool(
        self,
        tool: ToolDefinition,
        num_cases: int = 5,
    ) -> list[dict]:
        """
        Generate test cases for a tool.

        Args:
            tool: Tool definition to generate tests for
            num_cases: Number of test cases to generate

        Returns:
            List of test case dicts with query/response pairs
        """
        if not self.llm:
            return self._generate_fallback_cases(tool, num_cases)

        # Build parameters string
        params_str = ", ".join(
            f"{p.name}: {p.type} ({'required' if p.required else 'optional'})"
            for p in tool.parameters
        )

        prompt = GENERATOR_PROMPT.format(
            num_cases=num_cases,
            tool_name=tool.name,
            tool_description=tool.description,
            parameters=params_str or "None",
        )

        try:
            result = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = result.content

            # Parse JSON response
            import json

            # Find JSON in response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                cases = data.get("test_cases", [])

                # Add tool_id to each case
                for case in cases:
                    case["expected_tool"] = tool.id

                return cases

        except Exception as e:
            print(f"[TestGenerator] Failed to generate cases: {e}")

        return self._generate_fallback_cases(tool, num_cases)

    def _generate_fallback_cases(
        self, tool: ToolDefinition, num_cases: int
    ) -> list[dict]:
        """Generate simple fallback test cases without LLM."""
        cases = []

        # Build a base query from the tool description
        base_query = f"Use {tool.name}"
        if tool.parameters:
            param_examples = []
            for p in tool.parameters:
                if p.type == "string":
                    param_examples.append(f"{p.name}='example'")
                elif p.type in ("number", "integer"):
                    param_examples.append(f"{p.name}=10")
                elif p.type == "boolean":
                    param_examples.append(f"{p.name}=true")
            if param_examples:
                base_query += f" with {', '.join(param_examples)}"

        for i in range(num_cases):
            cases.append({
                "query": f"{base_query} (test case {i + 1})",
                "expected_tool": tool.id,
                "expected_params": {
                    p.name: f"test_value_{i}" for p in tool.parameters
                },
                "response": {
                    "status": "mock",
                    "data": f"Mock response for test case {i + 1}",
                },
            })

        return cases

    async def generate_for_server(
        self,
        tools: list[ToolDefinition],
        cases_per_tool: int = 3,
    ) -> list[dict]:
        """
        Generate test cases for all tools in a server.

        Args:
            tools: List of tool definitions
            cases_per_tool: Number of cases per tool

        Returns:
            Combined list of test cases
        """
        all_cases = []
        for tool in tools:
            cases = await self.generate_for_tool(tool, cases_per_tool)
            all_cases.extend(cases)
        return all_cases
