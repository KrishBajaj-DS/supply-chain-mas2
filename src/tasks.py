"""
Task definitions for the Supply Chain Disruption Management MAS.

Each task is bound to one agent. Tasks are passed in dependency order via
`context=` so each agent's output is available to the next - this is what
CrewAI uses as its default (Blackboard-style) messaging mechanism: every
task writes its result to a shared context that later tasks can read.

Req 3 (Pydantic Structured Output): the Manager task uses
`output_pydantic=SupplyChainDecision` so CrewAI validates the final
decision against the schema defined in src/output_models.py before
returning it.
"""

from crewai import Task
from src.agents import (
    get_supplier_monitoring_agent,
    get_demand_forecasting_agent,
    get_inventory_agent,
    get_alternative_supplier_agent,
    get_cost_agent,
    get_manager_agent,
)
from src.output_models import SupplyChainDecision

DISRUPTION_SCENARIO = (
    "Supplier X has delayed shipment of 5,000 units of Widget-A by 10 days "
    "due to port congestion (supplier_id='SUP-201', product='Widget-A')."
)


def get_manager_prompt(prompt_style: str = "react", structural_variant: str = "json_first", persona: str = None) -> str:
    """Build manager task prompt enforcing Phase 1 decision logic and Phase 4 prompt variance."""
    rules = (
        "DECISION RULES (MANDATORY):\n"
        "1. SHORTFALL ARITHMETIC: Compute shortfall = max(0, forecasted_demand_over_delay - current_stock) using numbers from the Blackboard. Never call forecasted demand 'shortfall'.\n"
        "2. DEFAULT TO WAIT: If projected_shortfall_units == 0, the default recommended_action MUST be 'wait' (monitor inventory and keep existing order). "
        "A switch or expedite is ONLY permitted if you state a specific quantified operational risk (e.g., buffer under 2 days of usage, or supplier reliability < 85%) and quantify it.\n"
        "3. COST FOR QUANTITY: Cost for an option must be computed for the quantity actually ordered/expedited (units x unit_price x premium), not the full order size by default.\n"
        "4. SHIPMENT DISPOSITION: State what happens to the original delayed shipment (cancelled, reduced, or kept) and address overstock risk.\n"
        "5. COMPLIANCE AUDIT: Inspect the 'compliance_audit' section posted by the Google ADK Risk & Compliance agent and cite its risk score.\n"
        "6. PERSONA DIRECTIVES:\n"
        "   - Operations Manager: Minimise stockout risk and downtime; accepts premium cost only when an actual shortfall exists.\n"
        "   - CFO: Choose the lowest-cost option that still closes the gap; prefer 'wait' when shortfall == 0.\n"
        "   - Supply Chain Planner: Prefer dual-sourcing / supplier reliability, but still default to 'wait' with an active monitoring plan when shortfall == 0.\n"
    )

    if prompt_style == "cot":
        style_instruction = "Reason step-by-step through a Chain-of-Thought (CoT): (1) Calculate shortfall, (2) Screen compliance risk, (3) Evaluate option costs, (4) Decide based on persona."
    elif prompt_style == "direct":
        style_instruction = "Synthesize findings directly and crisply: state the bottom-line action first, then provide numerical justification and trade-offs."
    else:  # default 'react'
        style_instruction = "Use an explicit ReAct (Reason -> Act -> Observe) trace to inspect Blackboard sections and justify the decision."

    struct_instruction = (
        "Output MUST strictly conform to the SupplyChainDecision schema."
        if structural_variant == "json_first" else
        "Provide a concise narrative analysis followed by the SupplyChainDecision structured schema."
    )

    persona_prompt = f"TARGET PERSONA CONTEXT: {persona}\n" if persona else ""

    return (
        f"You are the Manager Agent. Review the Blackboard shared state using read_blackboard.\n"
        f"{persona_prompt}"
        f"{style_instruction}\n"
        f"{struct_instruction}\n\n"
        f"{rules}\n"
        f"CRITICAL: Keep recommended_action and justification crisp, numerical, and strictly 2-3 lines.\n"
        f"Your output MUST strictly conform to the SupplyChainDecision schema."
    )


def build_tasks(scenario_text: str = DISRUPTION_SCENARIO, parallel: bool = False,
                prompt_style: str = "react", structural_variant: str = "json_first",
                persona: str = None):
    """parallel=False: strict chain (sequential pattern).
    parallel=True: Demand Forecasting and Alternative Supplier both depend only
    on the shipment report, so they run concurrently (fan-out), and the
    Inventory task waits for both (fan-in). Everything after depends on
    earlier outputs and stays serial.
    Supports prompt_style ('react', 'cot', 'direct') and structural_variant ('json_first', 'narrative_first')."""
    supplier_agent = get_supplier_monitoring_agent()
    forecast_agent = get_demand_forecasting_agent()
    inventory_agent = get_inventory_agent()
    alt_supplier_agent = get_alternative_supplier_agent()
    cost_agent = get_cost_agent()
    manager_agent = get_manager_agent()

    task_supplier = Task(
        description=(
            f"Scenario: {scenario_text}\n"
            "Use check_shipment_status to verify the shipment status, delay duration, and units. "
            "CRITICAL: Keep your final response strictly to 2-3 concise lines: "
            "(1) Supplier & product affected, (2) Delay length & reason, (3) Ordered quantity. "
            "Publish these 2-3 lines to Blackboard section 'supplier_status' using post_to_blackboard."
        ),
        expected_output="A concise 2-3 line factual status report with supplier, delay days, and units.",
        agent=supplier_agent,
    )

    task_forecast = Task(
        description=(
            "Use read_blackboard to inspect 'supplier_status', then run forecast_demand for the product over the delay days. "
            "CRITICAL: Keep your final response strictly to 2-3 concise lines: "
            "(1) Total forecasted demand units, (2) Daily average usage, (3) Trend direction. "
            "Publish these 2-3 lines to Blackboard section 'demand_forecast' using post_to_blackboard."
        ),
        expected_output="A concise 2-3 line demand forecast with total units, daily average, and trend.",
        agent=forecast_agent,
        context=[task_supplier],
        async_execution=parallel,
    )

    task_inventory = Task(
        description=(
            "Use read_blackboard to inspect 'supplier_status' and 'demand_forecast', then call check_inventory. "
            "CRITICAL: Keep your final response strictly to 2-3 concise lines: "
            "(1) Current stock vs forecasted need, (2) Projected buffer and runway days, (3) Shortage risk statement. "
            "Publish these 2-3 lines to Blackboard section 'inventory_projection' using post_to_blackboard."
        ),
        expected_output="A concise 2-3 line inventory projection with buffer units, runway days, and shortage status.",
        agent=inventory_agent,
        context=[task_supplier, task_forecast],
    )

    alt_description_chain = (
        "Use read_blackboard to inspect 'inventory_projection' and 'supplier_status'. "
        "Use search_alternative_suppliers to find backup suppliers. "
        "CRITICAL: Keep your final response strictly to 2-3 concise lines: "
        "(1) Top 2 backup suppliers with lead time, (2) Reliability and cost delta, (3) Recommended backup option. "
        "Publish these 2-3 lines to Blackboard section 'alternative_suppliers' using post_to_blackboard."
    )
    alt_description_parallel = (
        "Use read_blackboard to inspect delay duration in 'supplier_status'. "
        "Use search_alternative_suppliers to find backup suppliers. "
        "CRITICAL: Keep your final response strictly to 2-3 concise lines: "
        "(1) Top 2 backup suppliers with lead time, (2) Reliability and cost delta, (3) Recommended backup option. "
        "Publish these 2-3 lines to Blackboard section 'alternative_suppliers' using post_to_blackboard."
    )
    task_alt_supplier = Task(
        description=alt_description_parallel if parallel else alt_description_chain,
        expected_output="A concise 2-3 line shortlist of top alternative suppliers with lead times and cost deltas.",
        agent=alt_supplier_agent,
        context=[task_supplier] if parallel else [task_inventory],
        async_execution=parallel,
    )

    task_cost = Task(
        description=(
            "Use read_blackboard to retrieve 'supplier_status' and 'alternative_suppliers' to obtain the product, affected units, and alternative supplier details. "
            "You MUST call calculate_cost_impact 3 separate times for each option: "
            "1. calculate_cost_impact for option='wait for delayed shipment' with cost_premium_pct=0.0 "
            "2. calculate_cost_impact for option='switch to alternative supplier' with the alternative supplier's premium % (or 0.0 if not specified) "
            "3. calculate_cost_impact for option='expedite shipment' with cost_premium_pct=25.0 "
            "You MUST use the exact cost and delta figures returned by calculate_cost_impact tool (do NOT invent or estimate your own figures). "
            "CRITICAL: Keep your final response strictly to 3 concise lines citing the exact tool results: "
            "(1) Option A (Wait): cost delta $0.00 "
            "(2) Option B (Switch): exact cost and delta from tool "
            "(3) Option C (Expedite): exact cost and delta from tool. "
            "Publish these 3 lines to Blackboard section 'cost_analysis' using post_to_blackboard."
        ),
        expected_output="A concise 3-line financial impact breakdown citing the exact delta values returned by calculate_cost_impact.",
        agent=cost_agent,
        context=[task_supplier, task_alt_supplier],
    )

    task_manager = Task(
        description=get_manager_prompt(prompt_style=prompt_style, structural_variant=structural_variant, persona=persona),
        expected_output=(
            "A valid SupplyChainDecision JSON object with concise 2-3 line recommended_action, "
            "justification, options_evaluated, react_trace, and persona_context."
        ),
        agent=manager_agent,
        context=[task_supplier, task_forecast, task_inventory, task_alt_supplier, task_cost],
        output_pydantic=SupplyChainDecision,
    )

    agents = [
        supplier_agent, forecast_agent, inventory_agent,
        alt_supplier_agent, cost_agent, manager_agent,
    ]
    if parallel:
        # Async tasks must be adjacent and followed by a synchronous task
        # (Inventory) that consumes both outputs.
        tasks = [
            task_supplier, task_forecast, task_alt_supplier,
            task_inventory, task_cost, task_manager,
        ]
    else:
        tasks = [
            task_supplier, task_forecast, task_inventory,
            task_alt_supplier, task_cost, task_manager,
        ]
    return agents, tasks
