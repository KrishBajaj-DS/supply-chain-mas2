"""
Hierarchical architecture variant of the Supply Chain Disruption Management MAS.

Unlike the sequential variant (fixed task order), here the Manager Agent
acts as the delegator: CrewAI's Process.hierarchical uses a manager LLM
(or the designated manager agent) to decide which agent handles what and
in what order, then reviews/re-delegates as needed.

Run with:  python -m src.crew_hierarchical
"""

import time
import src.litellm_patch  # noqa: F401  (patches litellm before any LLM calls)
from crewai import Crew, Process, Task
from src.agents import (
    get_supplier_monitoring_agent,
    get_demand_forecasting_agent,
    get_inventory_agent,
    get_alternative_supplier_agent,
    get_cost_agent,
    llm,
)
from src.tasks import DISRUPTION_SCENARIO
from src.guardrails import enforce_guardrails
from src.blackboard import reset_blackboard, get_blackboard
from src.output_models import SupplyChainDecision
from src.a2a_crew_bridge import perform_a2a_compliance_step
from src.observability import make_step_callback


def build_hierarchical_crew(scenario_text: str = None, agent_timing_sink: list = None,
                            tool_overrides: dict = None, run_id: str = None,
                            prompt_style: str = "react", structural_variant: str = "json_first",
                            persona: str = None) -> Crew:
    scenario_text = scenario_text or DISRUPTION_SCENARIO
    enforce_guardrails(scenario_text)
    bb = reset_blackboard(initial_scenario=scenario_text, run_id=run_id)

    # --- REQUIREMENT 6 & 7: A2A Protocol step to Google ADK Compliance Agent ---
    try:
        perform_a2a_compliance_step(architecture="hierarchical", scenario_text=scenario_text, run_id=run_id)
    except Exception:
        pass

    from src import agents as _agents
    _agents.TOOL_OVERRIDES.clear()
    if tool_overrides:
        _agents.TOOL_OVERRIDES.update(tool_overrides)
    try:
        supplier_agent = get_supplier_monitoring_agent()
        forecast_agent = get_demand_forecasting_agent()
        inventory_agent = get_inventory_agent()
        alt_supplier_agent = get_alternative_supplier_agent()
        cost_agent = get_cost_agent()
    finally:
        _agents.TOOL_OVERRIDES.clear()

    persona_prefix = f"TARGET PERSONA: {persona}\n" if persona else ""

    rules = (
        "\nDECISION RULES (MANDATORY):\n"
        "1. Compute shortfall = max(0, forecasted_demand_over_delay - current_stock). Never call forecasted demand 'shortfall'.\n"
        "2. If projected_shortfall_units == 0, default recommended action MUST be 'wait'. "
        "A switch or expedite is ONLY permitted if you state a specific quantified operational risk (e.g., buffer under 2 days of usage, or supplier reliability) and quantify it.\n"
        "3. Option costs must be computed for quantity actually ordered/expedited (units x unit_price x premium).\n"
        "4. State what happens to original delayed shipment (cancelled, reduced, or kept) and address overstock risk.\n"
        "5. Review 'compliance_audit' section from Google ADK Risk & Compliance agent and cite its risk score.\n"
    )

    overall_task = Task(
        description=(
            f"Scenario: {scenario_text}\n"
            f"{persona_prefix}"
            "Coordinate your specialist team to resolve this disruption. Delegate to specialists:\n"
            "(1) Supplier Monitoring: confirm shipment/delay facts (2-3 lines),\n"
            "(2) Demand Forecasting: forecast demand units and trend (2-3 lines),\n"
            "(3) Inventory Specialist: check stock buffer and shortage status (2-3 lines),\n"
            "(4) Alternative Sourcing: identify top backup suppliers (2-3 lines),\n"
            "(5) Cost Analyst: evaluate cost delta across options (2-3 lines).\n"
            f"{rules}"
            "Finally, synthesize into ONE concise, structured SupplyChainDecision. Keep justification and recommendation strictly 2-3 lines with clear numbers."
        ),
        expected_output=(
            "A SupplyChainDecision object with recommended_action, justification, "
            "projected_shortfall_units, delay_days, forecasted_demand_units, "
            "options_evaluated, react_trace, risks_and_tradeoffs, and persona_context."
        ),
        output_pydantic=SupplyChainDecision,
    )

    if agent_timing_sink is not None:
        clock = {"last": time.time()}

        def _on_task_done(task_output):
            now = time.time()
            output_text = getattr(task_output, "raw", None) or str(task_output)
            bb.publish_agent_result(agent_role="HierarchicalManager", section="final_decision", content=output_text)
            agent_timing_sink.append({
                "agent": "HierarchicalManager",
                "latency_seconds": round(now - clock["last"], 3),
                "task_description": overall_task.description,
                "output_text": output_text,
            })
            clock["last"] = now

        overall_task.callback = _on_task_done

    worker_agents = [
        supplier_agent, forecast_agent, inventory_agent,
        alt_supplier_agent, cost_agent,
    ]

    return Crew(
        agents=worker_agents,
        tasks=[overall_task],
        process=Process.hierarchical,
        manager_llm=_agents.llm,
        max_rpm=8,          # throttle proactively instead of only retrying reactively
        verbose=True,
        step_callback=make_step_callback(run_id=run_id),
    )


if __name__ == "__main__":
    import re
    crew = build_hierarchical_crew()
    result = crew.kickoff()
    print("\n\n===== FINAL MANAGER DECISION (HIERARCHICAL) =====\n")
    decision = None
    if hasattr(result, "pydantic") and isinstance(result.pydantic, SupplyChainDecision):
        decision = result.pydantic
    else:
        raw_text = getattr(result, "raw", str(result))
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if json_match:
            decision = SupplyChainDecision.model_validate_json(json_match.group())
        else:
            raise ValueError(f"Hierarchical output failed Pydantic schema validation: {raw_text[:200]}")

    try:
        print(decision.to_markdown())
    except UnicodeEncodeError:
        print(decision.to_markdown().encode("ascii", errors="replace").decode("ascii"))