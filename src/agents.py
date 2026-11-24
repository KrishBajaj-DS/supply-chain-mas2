"""
Agent definitions for the Supply Chain Disruption Management MAS.

Six agents:
1. Supplier Monitoring Agent - tracks shipment/supplier status
2. Demand Forecasting Agent  - forecasts near-term demand
3. Inventory Agent           - checks stock levels & projected shortage
4. Alternative Supplier Agent- finds backup suppliers (uses RAG)
5. Cost Agent                - estimates cost impact of each option
6. Manager Agent             - coordinates, uses ReAct-style reasoning, proposes final action
"""

import os
from dotenv import load_dotenv
from crewai import Agent, LLM
from src.tools import (
    check_shipment_status,
    forecast_demand,
    check_inventory,
    search_alternative_suppliers,
    calculate_cost_impact,
    read_blackboard,
    post_to_blackboard,
)

load_dotenv()

# Tool override registry. Empty by default (agents use the local tools). Set
# by crew builders to swap in tools served over MCP (see src/mcp_server.py).
TOOL_OVERRIDES: dict = {}


def _tool(name, default):
    return TOOL_OVERRIDES.get(name, default)

llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)


def get_supplier_monitoring_agent() -> Agent:
    return Agent(
        role="Supplier Monitoring Agent",
        goal="Track and report the real-time status of supplier shipments, "
             "flagging delays, quantities affected, and reasons where known.",
        backstory=(
            "You are a logistics specialist embedded in the supply chain team. "
            "Your job is to watch incoming shipment data and immediately surface "
            "any disruption signal - a delay, a partial shipment, or a cancellation - "
            "with precise numbers so downstream agents can react fast."
        ),
        llm=llm,
        tools=[
            _tool("check_shipment_status", check_shipment_status),
            _tool("post_to_blackboard", post_to_blackboard),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )


def get_demand_forecasting_agent() -> Agent:
    return Agent(
        role="Demand Forecasting Agent",
        goal="Forecast expected demand for the affected product(s) over the "
             "disruption window using historical sales trends.",
        backstory=(
            "You are a data-driven demand planner. You use historical sales "
            "patterns and seasonality to estimate how much product will be "
            "needed in the coming days, so the team knows the real exposure "
            "of any supply delay."
        ),
        llm=llm,
        tools=[
            _tool("forecast_demand", forecast_demand),
            _tool("read_blackboard", read_blackboard),
            _tool("post_to_blackboard", post_to_blackboard),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )


def get_inventory_agent() -> Agent:
    return Agent(
        role="Inventory Agent",
        goal="Calculate current on-hand inventory and project whether a "
             "shortage will occur before the delayed shipment arrives.",
        backstory=(
            "You are the inventory controller. Given current stock levels, "
            "incoming forecasted demand, and the delay reported by the "
            "Supplier Monitoring Agent, you compute whether and when a "
            "stockout will occur."
        ),
        llm=llm,
        tools=[
            _tool("check_inventory", check_inventory),
            _tool("read_blackboard", read_blackboard),
            _tool("post_to_blackboard", post_to_blackboard),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=8,
    )


def get_alternative_supplier_agent() -> Agent:
    return Agent(
        role="Alternative Supplier Agent",
        goal="Identify viable alternative suppliers who can cover the "
             "shortfall, using the supplier knowledge base and any live "
             "web data available.",
        backstory=(
            "You are a sourcing specialist with deep knowledge of the "
            "vendor network. You search a retrieval-augmented knowledge base "
            "of past supplier performance and capacity to recommend the best "
            "backup options, citing why each one is credible."
        ),
        llm=llm,
        tools=[
            _tool("search_alternative_suppliers", search_alternative_suppliers),
            _tool("read_blackboard", read_blackboard),
            _tool("post_to_blackboard", post_to_blackboard),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=8,
    )


def get_cost_agent() -> Agent:
    return Agent(
        role="Cost Agent",
        goal="Calculate the true financial impact by calling calculate_cost_impact for wait, switch, and expedite options and reporting the exact tool numbers.",
        backstory=(
            "You are a supply chain financial analyst. For every option the "
            "team considers - waiting out the delay, switching supplier, or "
            "expediting shipment - you ALWAYS invoke calculate_cost_impact to calculate the "
            "true cost and delta so the Manager can weigh trade-offs numerically. You never "
            "guess or fabricate estimates; you strictly report the tool's numbers."
        ),
        llm=llm,
        tools=[
            _tool("calculate_cost_impact", calculate_cost_impact),
            _tool("read_blackboard", read_blackboard),
            _tool("post_to_blackboard", post_to_blackboard),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=8,
    )


def get_manager_agent() -> Agent:
    return Agent(
        role="Manager Agent",
        goal="Synthesize all agent findings using explicit step-by-step reasoning "
             "(ReAct, CoT, or direct synthesis) and propose one final, strictly justified action "
             "following shortfall arithmetic and cost-for-quantity rules.",
        backstory=(
            "You are the supply chain operations manager. You think through "
            "problems systematically: you compute exact inventory shortfall "
            "(shortfall = max(0, forecast - stock)), default to 'wait' when shortfall is zero, "
            "quantify risk when proposing alternative vendors or expedite logistics, "
            "calculate costs for actual quantities, and incorporate compliance audit findings."
        ),
        llm=llm,
        tools=[
            _tool("read_blackboard", read_blackboard),
            _tool("post_to_blackboard", post_to_blackboard),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=8,
    )