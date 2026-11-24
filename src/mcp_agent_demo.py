"""
One CrewAI agent that gets its tools OVER MCP (course criterion 2).

The agent has no local tool objects: it connects to src/mcp_server.py through
crewai_tools.MCPServerAdapter, discovers the tools, and calls them to answer a
small question. Uses ~3 LLM calls, so it costs very little Groq quota.

    python -m src.mcp_agent_demo            # full demo (needs GROQ_API_KEY)
    python -m src.mcp_agent_demo --dry      # load tools over MCP only, no LLM
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Crew, Process, Task  # noqa: E402
from crewai_tools import MCPServerAdapter  # noqa: E402
from mcp import StdioServerParameters  # noqa: E402

SERVER = StdioServerParameters(
    command=sys.executable, args=["-m", "src.mcp_server"], env=dict(os.environ)
)


def main(dry: bool = False):
    with MCPServerAdapter(SERVER) as mcp_tools:
        names = [t.name for t in mcp_tools]
        print(f"CrewAI loaded {len(names)} tools over MCP: {names}")
        if dry:
            return
        import src.litellm_patch  # noqa: F401  (Groq compatibility patch)

        agent = Agent(
            role="MCP Cost Analyst",
            goal="Answer supply-chain questions using only the MCP tools provided.",
            backstory="You verify every number by calling a tool; you never guess.",
            tools=list(mcp_tools),
            llm="groq/openai/gpt-oss-120b",
            max_iter=5,
            verbose=True,
        )
        task = Task(
            description=(
                "Use forecast_demand for product 'Widget-A' over 10 days, then use "
                "calculate_cost_impact for 5000 units of Widget-A with option "
                "'switch to Vertex' and cost_premium_pct -12. Report the forecast "
                "total and the total cost in two sentences."
            ),
            expected_output="Two sentences citing the tool results.",
            agent=agent,
        )
        result = Crew(agents=[agent], tasks=[task], process=Process.sequential).kickoff()
        print("\n=== RESULT (tools were called over MCP) ===\n", result)


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
