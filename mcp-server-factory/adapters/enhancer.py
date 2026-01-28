"""
LLM-Enhanced Description Generator.

Improves tool descriptions for better LLM understanding.
"""

import os
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from registry.schemas import ToolDefinition


ENHANCE_PROMPT = """Improve this MCP tool description for better LLM tool selection.

Current tool:
- Name: {name}
- Description: {description}
- Parameters: {parameters}

Write an enhanced description that:
1. Clearly states what the tool does in the first sentence
2. Explains when an LLM should use this tool
3. Lists any important constraints or limitations
4. Is between 50-150 words

Return ONLY the enhanced description, no other text."""


class DescriptionEnhancer:
    """Enhance tool descriptions using LLM."""

    def __init__(self, model: str = "claude-3-5-haiku-latest"):
        """Initialize the enhancer."""
        self.model = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.llm = ChatAnthropic(
                model=model,
                api_key=api_key,
                max_tokens=300,
            )
        else:
            self.llm = None

    async def enhance_description(
        self,
        name: str,
        current_description: str,
        parameters: list[dict],
    ) -> str:
        """
        Enhance a tool description.

        Args:
            name: Tool name
            current_description: Current description text
            parameters: List of parameter dicts

        Returns:
            Enhanced description
        """
        if not self.llm:
            return current_description

        params_str = ", ".join(
            f"{p.get('name', 'unknown')}: {p.get('type', 'string')}"
            for p in parameters
        )

        prompt = ENHANCE_PROMPT.format(
            name=name,
            description=current_description,
            parameters=params_str or "None",
        )

        try:
            result = await self.llm.ainvoke([HumanMessage(content=prompt)])
            enhanced = result.content.strip()

            # Ensure minimum length
            if len(enhanced) >= 10:
                return enhanced

        except Exception as e:
            print(f"[Enhancer] Failed to enhance description: {e}")

        return current_description

    async def enhance_tool(self, tool: ToolDefinition) -> ToolDefinition:
        """
        Enhance a tool definition.

        Args:
            tool: Tool to enhance

        Returns:
            Enhanced tool definition
        """
        enhanced_desc = await self.enhance_description(
            name=tool.name,
            current_description=tool.description,
            parameters=[p.model_dump() for p in tool.parameters],
        )

        # Return new tool with enhanced description
        return ToolDefinition(
            id=tool.id,
            name=tool.name,
            description=enhanced_desc,
            parameters=tool.parameters,
            implementation=tool.implementation,
        )

    async def enhance_all_tools(
        self, tools: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        """Enhance all tools in a list."""
        enhanced = []
        for tool in tools:
            enhanced_tool = await self.enhance_tool(tool)
            enhanced.append(enhanced_tool)
        return enhanced
