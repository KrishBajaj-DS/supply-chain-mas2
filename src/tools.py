"""
Tools for the Supply Chain Disruption Management MAS.

One primary tool per agent:
- check_shipment_status      -> Supplier Monitoring Agent
- forecast_demand            -> Demand Forecasting Agent (simple trend-based ML)
- check_inventory            -> Inventory Agent
- search_alternative_suppliers -> Alternative Supplier Agent (RAG-backed)
- calculate_cost_impact      -> Cost Agent
"""

try:
    from crewai.tools import tool
except ImportError:
    def tool(name=None):
        def decorator(f):
            f.name = name or getattr(f, "__name__", "tool")
            f.func = f
            return f
        return decorator
from src.guardrails import guarded
from src.mock_data import SHIPMENTS, INVENTORY, SALES_HISTORY, MARKET_UNIT_COST
from src.rag_retriever import get_retriever
import numpy as np


@tool("check_shipment_status")
@guarded("check_shipment_status")
def check_shipment_status(supplier_id: str) -> str:
    """Check the current shipment status for a given supplier ID.
    Returns delay info, affected product, and quantity impacted.
    Pass the supplier_id specified in the disruption scenario (e.g. 'SUP-201', 'SUP-305', 'SUP-410')."""
    data = SHIPMENTS.get(supplier_id)
    if not data:
        return f"No shipment record found for supplier_id={supplier_id}."
    return (
        f"Supplier: {data['supplier_name']} | Product: {data['product']} | "
        f"Ordered: {data['ordered_units']} units | "
        f"Delay: {data['delay_days']} days | "
        f"Reason: {data['reason']} | Status: {data['status']}"
    )


@tool("forecast_demand")
@guarded("forecast_demand")
def forecast_demand(product: str, days_ahead: int = 10) -> str:
    """Forecast expected demand (units) for a product over the next N days
    using a simple linear trend fit on the last 14 days of sales history.
    Pass the product affected by the disruption (e.g. 'Widget-A', 'Sensor-B', 'Cable-Harness')."""
    history = SALES_HISTORY.get(product)
    if not history:
        return f"No sales history found for product={product}."

    x = np.arange(len(history))
    y = np.array(history)
    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(len(history), len(history) + days_ahead)
    forecast = slope * future_x + intercept
    forecast = np.clip(forecast, a_min=0, a_max=None)
    total_forecast = int(round(forecast.sum()))
    avg_daily = round(float(forecast.mean()), 1)

    trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    return (
        f"Demand forecast for {product} over next {days_ahead} days: "
        f"~{total_forecast} units total (avg {avg_daily}/day). "
        f"Trend: {trend} (slope={round(float(slope), 2)} units/day)."
    )


@tool("check_inventory")
@guarded("check_inventory")
def check_inventory(product: str, forecasted_units_needed: int, delay_days: int) -> str:
    """Check current inventory for a product and project whether a shortage
    will occur before a delayed shipment arrives. Provide the forecasted
    units needed over the delay window and the delay length in days."""
    inv = INVENTORY.get(product)
    if not inv:
        return f"No inventory record found for product={product}."

    current_stock = inv["current_stock"]
    daily_usage = inv["daily_avg_usage"]
    projected_usage = daily_usage * delay_days
    projected_ending_stock = current_stock - max(projected_usage, forecasted_units_needed)

    if projected_ending_stock < 0:
        shortfall = abs(projected_ending_stock)
        return (
            f"SHORTAGE PROJECTED for {product}: current stock {current_stock} units, "
            f"projected usage over {delay_days}-day delay ~{projected_usage} units "
            f"(forecast says {forecasted_units_needed}). "
            f"Projected shortfall: {shortfall} units before delayed shipment arrives."
        )
    else:
        return (
            f"No shortage projected for {product}: current stock {current_stock} units "
            f"is sufficient to cover ~{max(projected_usage, forecasted_units_needed)} units "
            f"of usage over the {delay_days}-day delay. "
            f"Projected ending stock: {projected_ending_stock} units."
        )


@tool("search_alternative_suppliers")
@guarded("search_alternative_suppliers")
def search_alternative_suppliers(query: str, top_k: int = 3) -> str:
    """Search the supplier knowledge base (RAG) for alternative suppliers
    matching a query, e.g. 'urgent low lead time supplier' or
    'cheapest bulk supplier'. Returns top matching supplier profiles."""
    retriever = get_retriever()
    results = retriever.retrieve(query, top_k=top_k)

    if not results:
        return "No matching suppliers found."

    formatted = []
    for r in results:
        formatted.append(
            f"[{r['id']}] {r['name']} (relevance={r['score']}): {r['text']}"
        )
    return "\n\n".join(formatted)


@tool("calculate_cost_impact")
@guarded("calculate_cost_impact")
def calculate_cost_impact(product: str, units: int, option: str, cost_premium_pct: float = 0.0) -> str:
    """Calculate the estimated cost impact of a sourcing option.
    option should describe the choice (e.g. 'wait for delayed shipment',
    'switch to alternative supplier', 'expedite shipment').
    cost_premium_pct is the % above/below market rate for that option
    (e.g. 25 for +25%, -12 for -12%), from the supplier's profile if known."""
    base_cost = MARKET_UNIT_COST.get(product)
    if base_cost is None:
        return f"No baseline cost data found for product={product}."

    adjusted_unit_cost = base_cost * (1 + cost_premium_pct / 100)
    total_cost = round(adjusted_unit_cost * units, 2)
    baseline_total = round(base_cost * units, 2)
    delta = round(total_cost - baseline_total, 2)

    return (
        f"Cost estimate for option '{option}' on {units} units of {product}: "
        f"unit cost {round(adjusted_unit_cost, 2)} (baseline {base_cost}), "
        f"total cost {total_cost} vs baseline total {baseline_total} "
        f"(delta: {'+' if delta >= 0 else ''}{delta})."
    )


@tool("read_blackboard")
def read_blackboard(section: str = "all") -> str:
    """Read shared state from the Blackboard.
    Provide a specific section name (e.g. 'supplier_status', 'demand_forecast',
    'inventory_projection', 'alternative_suppliers', 'cost_analysis') or 'all' to get the full state."""
    from src.blackboard import get_blackboard
    bb = get_blackboard()
    if not section or section.lower() in ("all", "*", "full"):
        return bb.format_blackboard_context()
    val = bb.read(section)
    if val is None:
        return f"Section '{section}' is not yet populated on the Blackboard."
    return str(val)


@tool("post_to_blackboard")
def post_to_blackboard(section: str, content: str) -> str:
    """Publish specialist findings or state update to a section of the shared Blackboard."""
    from src.blackboard import get_blackboard
    bb = get_blackboard()
    bb.post(section, "SpecialistAgent", content)
    return f"Successfully posted to Blackboard section '{section}'."


def get_all_tools():
    """Convenience accessor returning every tool, useful for testing."""
    return [
        check_shipment_status,
        forecast_demand,
        check_inventory,
        search_alternative_suppliers,
        calculate_cost_impact,
        read_blackboard,
        post_to_blackboard,
    ]
