"""
MCP server exposing the supply-chain tools over the Model Context Protocol
(course criterion 2: tool integration using MCP).

Exposes four deterministic tools (no LLM, no API quota):
  check_shipment_status, forecast_demand, check_inventory, calculate_cost_impact
They are the same functions the CrewAI agents use, including the Pydantic
tool-output guardrail, so MCP clients get validated outputs too.
(The RAG supplier search is not exposed here because it needs the Gemini key.)

Run standalone (stdio transport):   python -m src.mcp_server
Verify with a real MCP client:      python -m src.mcp_demo
"""

import os

# tools.py imports the RAG retriever, which builds a Gemini client at import
# time. None of the tools exposed here call it, so a placeholder is fine.
os.environ.setdefault("GEMINI_API_KEY", "unused-by-mcp-tools")

try:
    from mcp.server.mcpserver import MCPServer
except (ImportError, ModuleNotFoundError):
    try:
        from mcp.server import MCPServer
    except (ImportError, ModuleNotFoundError):
        from mcp.server.fastmcp import FastMCP as MCPServer

from src import tools as _t  # noqa: E402

mcp = MCPServer("supply-chain-tools")


@mcp.tool()
def check_shipment_status(supplier_id: str) -> str:
    """Check the current shipment status for a supplier ID (e.g. 'SUP-201')."""
    return _t.check_shipment_status.func(supplier_id)


@mcp.tool()
def forecast_demand(product: str, days_ahead: int = 10) -> str:
    """Linear-trend demand forecast (units) for a product over the next N days."""
    return _t.forecast_demand.func(product, days_ahead)


@mcp.tool()
def check_inventory(product: str, forecasted_units_needed: int, delay_days: int) -> str:
    """Project whether stock covers demand during a shipment delay."""
    return _t.check_inventory.func(product, forecasted_units_needed, delay_days)


@mcp.tool()
def calculate_cost_impact(product: str, units: int, option: str,
                          cost_premium_pct: float = 0.0) -> str:
    """Estimate total cost of a sourcing option vs the market baseline."""
    return _t.calculate_cost_impact.func(product, units, option, cost_premium_pct)


if __name__ == "__main__":
    mcp.run()  # stdio transport
