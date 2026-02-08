"""
MCP Gateway Agent Runtime

Implements a ReAct (Reasoning + Acting) loop that:
1. Receives a user query
2. Loads assigned skills from the registry
3. Injects skill content into the LLM system prompt
4. Calls Claude API in a think-act-observe loop
5. Uses authorized tools via the MCP Gateway
6. Returns the final answer with trace of reasoning steps
"""

import os
import re
import json
import httpx
import asyncio
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field
from anthropic import Anthropic

from .config import AgentRuntimeConfig, load_config, AGENT_PRESETS


@dataclass
class ReActStep:
    """A single step in the ReAct loop."""
    iteration: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AgentResult:
    """Result of agent execution."""
    query: str
    answer: str
    steps: list[ReActStep]
    total_iterations: int
    tokens_used: int
    success: bool
    error: Optional[str] = None


class MCPAgent:
    """ReAct agent that uses skills and tools via the MCP Gateway."""

    def __init__(self, config: AgentRuntimeConfig):
        self.config = config
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.skills_cache: dict[str, str] = {}
        self.tools_cache: list[dict] = []

    async def _fetch_skills(self) -> dict[str, str]:
        """Fetch skill content from the gateway."""
        skills = {}
        async with httpx.AsyncClient() as client:
            for skill_id in self.config.assigned_skills:
                try:
                    response = await client.get(
                        f"{self.config.gateway_url}/skills/{skill_id}",
                        headers={"X-API-Key": self.config.api_key}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        skills[skill_id] = data.get("content", "")
                except Exception as e:
                    print(f"[Warning] Failed to fetch skill {skill_id}: {e}")
        return skills

    async def _fetch_tools(self) -> list[dict]:
        """Fetch available tools from the gateway."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.config.gateway_url}/mcp/tools",
                    headers={"X-API-Key": self.config.api_key}
                )
                if response.status_code == 200:
                    all_tools = response.json().get("tools", [])
                    # Filter by authorized patterns
                    return self._filter_tools(all_tools)
            except Exception as e:
                print(f"[Warning] Failed to fetch tools: {e}")
        return []

    def _filter_tools(self, tools: list[dict]) -> list[dict]:
        """Filter tools based on authorized patterns."""
        filtered = []
        for tool in tools:
            full_name = tool.get("full_name", "")
            for pattern in self.config.authorized_tools:
                # Simple pattern matching (e.g., "pathology.*" matches "pathology.run_inference")
                if pattern.endswith(".*"):
                    prefix = pattern[:-2]
                    if full_name.startswith(prefix):
                        filtered.append(tool)
                        break
                elif pattern == full_name:
                    filtered.append(tool)
                    break
        return filtered

    def _build_system_prompt(self, skills: dict[str, str], tools: list[dict]) -> str:
        """Build the system prompt with skills and tools."""
        prompt_parts = [
            "You are an AI assistant for MCP Gateway, a life sciences platform.",
            "You help users analyze pathology data, evaluate models, and generate reports.",
            "",
            "## Your Skills",
            ""
        ]

        for skill_id, content in skills.items():
            prompt_parts.append(f"### {skill_id}")
            prompt_parts.append(content)
            prompt_parts.append("")

        prompt_parts.append("## Available Tools")
        prompt_parts.append("")
        for tool in tools:
            prompt_parts.append(f"- **{tool['full_name']}**: {tool['description']}")
            if tool.get('parameters'):
                prompt_parts.append(f"  Parameters: {json.dumps(tool['parameters'])}")
        prompt_parts.append("")

        prompt_parts.extend([
            "## How to Respond",
            "",
            "Use the ReAct framework:",
            "1. **Thought**: Analyze what you need to do",
            "2. **Action**: Call a tool if needed (format: ACTION: tool_name)",
            "3. **Action Input**: Provide parameters as JSON",
            "4. **Observation**: I will provide the tool result",
            "5. Repeat until you have enough information",
            "6. **Final Answer**: Provide your response to the user",
            "",
            "Format your responses exactly like this:",
            "```",
            "Thought: [your reasoning]",
            "Action: [tool_name or 'none' if ready to answer]",
            "Action Input: {\"param\": \"value\"}",
            "```",
            "",
            "Or if ready to answer:",
            "```",
            "Thought: [final reasoning]",
            "Final Answer: [your complete response to the user]",
            "```"
        ])

        return "\n".join(prompt_parts)

    async def _call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool via the MCP Gateway."""
        # Parse tool name (format: server_id.tool_name)
        parts = tool_name.split(".")
        if len(parts) != 2:
            return f"Error: Invalid tool name format '{tool_name}'. Expected 'server.tool'"

        server_id, tool = parts

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.config.gateway_url}/mcp/call",
                    headers={"X-API-Key": self.config.api_key},
                    json={
                        "server_id": server_id,
                        "tool_name": tool,
                        "arguments": arguments
                    }
                )
                data = response.json()
                if data.get("success"):
                    return json.dumps(data.get("result", {}), indent=2)
                else:
                    return f"Error: {data.get('error', 'Unknown error')}"
            except Exception as e:
                return f"Error calling tool: {str(e)}"

    def _parse_response(self, text: str) -> tuple[Optional[str], Optional[str], Optional[dict], Optional[str]]:
        """Parse the LLM response to extract thought, action, input, and final answer."""
        thought = None
        action = None
        action_input = None
        final_answer = None

        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Check for final answer
        final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if final_match:
            final_answer = final_match.group(1).strip()
            return thought, None, None, final_answer

        # Extract action
        action_match = re.search(r"Action:\s*(\S+)", text)
        if action_match:
            action = action_match.group(1).strip()
            if action.lower() == "none":
                action = None

        # Extract action input
        input_match = re.search(r"Action Input:\s*(\{.+?\})", text, re.DOTALL)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                action_input = {}

        return thought, action, action_input, final_answer

    async def run(self, query: str) -> AgentResult:
        """Run the agent on a query."""
        steps = []
        tokens_used = 0

        # Fetch skills and tools
        if not self.skills_cache:
            self.skills_cache = await self._fetch_skills()
        if not self.tools_cache:
            self.tools_cache = await self._fetch_tools()

        # Build system prompt
        system_prompt = self._build_system_prompt(self.skills_cache, self.tools_cache)

        # Initialize conversation
        messages = [{"role": "user", "content": query}]

        for iteration in range(1, self.config.max_iterations + 1):
            if self.config.verbose:
                print(f"\n{'='*50}")
                print(f"Iteration {iteration}/{self.config.max_iterations}")
                print('='*50)

            # Call Claude
            try:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.token_budget // self.config.max_iterations,
                    system=system_prompt,
                    messages=messages,
                    temperature=self.config.temperature
                )

                assistant_text = response.content[0].text
                tokens_used += response.usage.input_tokens + response.usage.output_tokens

                if self.config.verbose:
                    print(f"\n[Assistant]\n{assistant_text}")

            except Exception as e:
                return AgentResult(
                    query=query,
                    answer="",
                    steps=steps,
                    total_iterations=iteration,
                    tokens_used=tokens_used,
                    success=False,
                    error=str(e)
                )

            # Parse response
            thought, action, action_input, final_answer = self._parse_response(assistant_text)

            step = ReActStep(
                iteration=iteration,
                thought=thought or "",
                action=action,
                action_input=action_input
            )

            # Check for final answer
            if final_answer:
                step.observation = "Final answer provided"
                steps.append(step)
                return AgentResult(
                    query=query,
                    answer=final_answer,
                    steps=steps,
                    total_iterations=iteration,
                    tokens_used=tokens_used,
                    success=True
                )

            # Execute action if provided
            if action and action_input is not None:
                if self.config.verbose:
                    print(f"\n[Calling Tool] {action}")
                    print(f"[Input] {json.dumps(action_input, indent=2)}")

                observation = await self._call_tool(action, action_input)
                step.observation = observation

                if self.config.verbose:
                    print(f"\n[Observation]\n{observation[:500]}...")

                # Add to conversation
                messages.append({"role": "assistant", "content": assistant_text})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                # No action, might be thinking
                messages.append({"role": "assistant", "content": assistant_text})
                messages.append({"role": "user", "content": "Please continue with an Action or provide your Final Answer."})

            steps.append(step)

        # Max iterations reached
        return AgentResult(
            query=query,
            answer="I was unable to complete the task within the allowed iterations.",
            steps=steps,
            total_iterations=self.config.max_iterations,
            tokens_used=tokens_used,
            success=False,
            error="Max iterations reached"
        )


async def main():
    """CLI entry point for testing the agent."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m agent.runtime \"Your question here\"")
        print("\nExample: python -m agent.runtime \"What is the tumor classification for slide S-2024-001?\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print("\n" + "="*60)
    print("  MCP GATEWAY AGENT")
    print("="*60)
    print(f"\nQuery: {query}\n")

    # Load config (use pathology preset by default)
    config = AGENT_PRESETS["full-access"]
    config.verbose = True

    # Run agent
    agent = MCPAgent(config)
    result = await agent.run(query)

    # Print result
    print("\n" + "="*60)
    print("  RESULT")
    print("="*60)
    print(f"\nSuccess: {result.success}")
    print(f"Iterations: {result.total_iterations}")
    print(f"Tokens Used: {result.tokens_used}")

    print("\n--- ANSWER ---")
    print(result.answer)

    print("\n--- REASONING TRACE ---")
    for step in result.steps:
        print(f"\n[Step {step.iteration}]")
        print(f"Thought: {step.thought[:200]}..." if len(step.thought) > 200 else f"Thought: {step.thought}")
        if step.action:
            print(f"Action: {step.action}")
            print(f"Input: {step.action_input}")
        if step.observation:
            obs = step.observation[:200] + "..." if len(step.observation) > 200 else step.observation
            print(f"Observation: {obs}")


class MCPAgentStreaming(MCPAgent):
    """Streaming version of MCPAgent that yields events as they happen."""

    async def run_streaming(self, query: str):
        """Run the agent and yield events as they happen."""
        # Fetch skills and tools
        if not self.skills_cache:
            self.skills_cache = await self._fetch_skills()
        if not self.tools_cache:
            self.tools_cache = await self._fetch_tools()

        # Build system prompt
        system_prompt = self._build_system_prompt(self.skills_cache, self.tools_cache)

        # Initialize conversation
        messages = [{"role": "user", "content": query}]

        yield {"type": "start", "query": query}

        for iteration in range(1, self.config.max_iterations + 1):
            yield {"type": "iteration_start", "iteration": iteration}

            # Call Claude
            try:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.token_budget // self.config.max_iterations,
                    system=system_prompt,
                    messages=messages,
                    temperature=self.config.temperature
                )

                assistant_text = response.content[0].text

                yield {"type": "thinking", "content": assistant_text[:500]}

            except Exception as e:
                yield {"type": "error", "error": str(e)}
                return

            # Parse response
            thought, action, action_input, final_answer = self._parse_response(assistant_text)

            yield {"type": "thought", "thought": thought}

            # Check for final answer
            if final_answer:
                yield {"type": "final_answer", "answer": final_answer}
                return

            # Execute action if provided
            if action and action_input is not None:
                yield {"type": "action", "action": action, "input": action_input}

                observation = await self._call_tool(action, action_input)

                yield {"type": "observation", "observation": observation[:500]}

                # Add to conversation
                messages.append({"role": "assistant", "content": assistant_text})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                messages.append({"role": "assistant", "content": assistant_text})
                messages.append({"role": "user", "content": "Please continue with an Action or provide your Final Answer."})

        yield {"type": "max_iterations", "message": "Max iterations reached"}


if __name__ == "__main__":
    asyncio.run(main())
