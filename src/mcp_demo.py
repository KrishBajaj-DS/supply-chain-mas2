"""
Verifies the MCP integration end to end with a real MCP client:
spawns src/mcp_server.py over stdio, discovers its tools, and calls them.
No LLM and no API quota are used.

    python -m src.mcp_demo
"""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            listed = await session.list_tools()
            print(f"MCP server exposes {len(listed.tools)} tools:")
            for t in listed.tools:
                print(f"  - {t.name}: {t.description}")

            calls = [
                ("check_shipment_status", {"supplier_id": "SUP-201"}),
                ("forecast_demand", {"product": "Widget-A", "days_ahead": 10}),
                ("check_inventory", {"product": "Widget-A", "forecasted_units_needed": 2912, "delay_days": 10}),
                ("calculate_cost_impact", {"product": "Widget-A", "units": 5000,
                                           "option": "switch to Vertex", "cost_premium_pct": -12}),
            ]
            print("\nCalling tools through the MCP client:")
            for name, args in calls:
                res = await session.call_tool(name, args)
                text = "".join(getattr(c, "text", "") for c in res.content)
                print(f"\n> {name}({args})\n  {text}")


if __name__ == "__main__":
    asyncio.run(main())
