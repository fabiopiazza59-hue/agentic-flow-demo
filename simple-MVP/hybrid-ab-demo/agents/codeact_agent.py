"""
CodeAct Agent for Complex Analysis

Code-generation agent that writes and executes Python for complex tasks.
Routes here when complexity_score >= 4.

Handles:
- Monte Carlo simulations
- Portfolio optimization
- Statistical analysis
- Data visualization descriptions
"""

import os
import json
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from code_interpreter.sandbox import execute_code, CodeExecutionResult

load_dotenv()


# ============================================================
# SYSTEM PROMPT FOR CODE GENERATION
# ============================================================

CODEACT_SYSTEM_PROMPT = """You are a Quantitative Financial Analyst AI that writes Python code to solve complex financial problems.

## AVAILABLE LIBRARIES (pre-imported, no import statements needed)
- `np` (numpy): Arrays, random sampling, statistics
- `pd` (pandas): DataFrames, time series
- `math`: Mathematical functions
- `statistics`: Statistical calculations
- `random`: Random number generation

## RULES
1. Write clean, working Python code
2. Do NOT use import statements - libraries are pre-loaded
3. Store your final answer in a variable called `result` (as a dict)
4. Use `print()` to show intermediate steps
5. Handle edge cases (empty data, division by zero)
6. Keep code concise but readable

## OUTPUT FORMAT
Return a JSON object with:
{
    "explanation": "Brief explanation of the approach (1-2 sentences)",
    "code": "```python\\n<your code here>\\n```"
}

## EXAMPLE: Monte Carlo Retirement Simulation

User: "Run a Monte Carlo simulation for retirement with $100k savings, $12k annual contribution, over 30 years"

Your response:
{
    "explanation": "Running 1000 simulations with 7% mean return and 15% standard deviation to estimate retirement outcomes.",
    "code": "```python
# Monte Carlo retirement simulation
initial = 100000
annual_contrib = 12000
years = 30
n_sims = 1000

final_values = []
for _ in range(n_sims):
    balance = initial
    for y in range(years):
        ret = np.random.normal(0.07, 0.15)
        balance = balance * (1 + ret) + annual_contrib
    final_values.append(balance)

values = np.array(final_values)
result = {
    'median': float(np.median(values)),
    'p10': float(np.percentile(values, 10)),
    'p90': float(np.percentile(values, 90)),
    'prob_1m': float(np.mean(values > 1000000) * 100)
}
print(f\"Median: ${result['median']:,.0f}\")
print(f\"10th percentile: ${result['p10']:,.0f}\")
print(f\"90th percentile: ${result['p90']:,.0f}\")
print(f\"Probability of $1M+: {result['prob_1m']:.1f}%\")
```"
}

## EXAMPLE: Portfolio Optimization

User: "Optimize a portfolio of AAPL, GOOGL, MSFT for maximum Sharpe ratio"

Your response:
{
    "explanation": "Using mean-variance optimization to find weights that maximize risk-adjusted returns.",
    "code": "```python
# Simulated returns data (in production, use real data)
returns = {
    'AAPL': np.random.normal(0.12, 0.25, 252),
    'GOOGL': np.random.normal(0.10, 0.22, 252),
    'MSFT': np.random.normal(0.11, 0.20, 252)
}
df = pd.DataFrame(returns)

# Calculate expected returns and covariance
exp_ret = df.mean() * 252
cov_matrix = df.cov() * 252

# Find optimal weights (simplified grid search)
best_sharpe = -999
best_weights = None
rf = 0.04  # risk-free rate

for w1 in np.linspace(0, 1, 21):
    for w2 in np.linspace(0, 1-w1, 21):
        w3 = 1 - w1 - w2
        weights = np.array([w1, w2, w3])
        port_ret = np.dot(weights, exp_ret)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_ret - rf) / port_vol
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights

result = {
    'AAPL': float(best_weights[0] * 100),
    'GOOGL': float(best_weights[1] * 100),
    'MSFT': float(best_weights[2] * 100),
    'sharpe_ratio': float(best_sharpe)
}
print(f\"Optimal Allocation:\")
print(f\"  AAPL: {result['AAPL']:.1f}%\")
print(f\"  GOOGL: {result['GOOGL']:.1f}%\")
print(f\"  MSFT: {result['MSFT']:.1f}%\")
print(f\"Expected Sharpe Ratio: {result['sharpe_ratio']:.2f}\")
```"
}

Remember: Output ONLY the JSON object, no other text."""


# ============================================================
# CODEACT AGENT CLASS
# ============================================================

class CodeActAgent:
    """Agent that generates and executes Python code for complex analysis."""

    def __init__(self):
        self.model = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=2048,
        )

    async def run(self, query: str, context: dict = None) -> dict:
        """
        Process a complex query by generating and executing code.

        Args:
            query: User's analysis request
            context: Optional context data (portfolio, preferences)

        Returns:
            Dict with explanation, code, output, and result
        """
        # Step 1: Generate code
        code_response = await self._generate_code(query)

        if "error" in code_response:
            return {
                "success": False,
                "error": code_response["error"],
                "path": "codeact"
            }

        explanation = code_response.get("explanation", "")
        code = code_response.get("code", "")

        # Extract code from markdown block if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        # Step 2: Execute code in sandbox
        exec_result = execute_code(code, context=context)

        return {
            "success": exec_result.success,
            "explanation": explanation,
            "code": code,
            "output": exec_result.output,
            "result": exec_result.result,
            "error": exec_result.error,
            "execution_time_ms": exec_result.execution_time_ms,
            "path": "codeact"
        }

    async def _generate_code(self, query: str) -> dict:
        """Generate Python code for the given query."""
        try:
            messages = [
                SystemMessage(content=CODEACT_SYSTEM_PROMPT),
                HumanMessage(content=query)
            ]

            response = await self.model.ainvoke(messages)
            content = response.content.strip()

            # Parse JSON response
            # Handle case where response might have markdown code block
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            return json.loads(content)

        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract code directly
            return {
                "explanation": "Generated analysis code",
                "code": content,
            }
        except Exception as e:
            return {"error": f"Code generation failed: {str(e)}"}


# Singleton pattern
_codeact_agent = None


def get_codeact_agent() -> CodeActAgent:
    """Get or create the CodeAct agent singleton."""
    global _codeact_agent
    if _codeact_agent is None:
        _codeact_agent = CodeActAgent()
    return _codeact_agent


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        agent = get_codeact_agent()

        # Test Monte Carlo
        print("Testing Monte Carlo simulation...")
        result = await agent.run(
            "Run a Monte Carlo simulation for retirement with $50,000 initial savings, "
            "$6,000 annual contribution, over 25 years. What's the probability of reaching $500k?"
        )

        print(f"\nSuccess: {result['success']}")
        print(f"Explanation: {result.get('explanation', 'N/A')}")
        print(f"\nCode:\n{result.get('code', 'N/A')[:500]}...")
        print(f"\nOutput:\n{result.get('output', 'N/A')}")
        print(f"\nResult: {result.get('result', 'N/A')}")
        if result.get('error'):
            print(f"Error: {result['error']}")

    asyncio.run(test())
