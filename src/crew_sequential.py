"""
Sequential architecture for the Supply Chain Disruption Management MAS.

One of three supported agentic architectures in this project (Requirement 4):
1. Sequential (src/crew_sequential.py) - pipeline with Blackboard callbacks & step tracking.
2. Parallel fan-out/fan-in (src/crew_parallel.py) - concurrent forecasting & backup sourcing.
3. Hierarchical (src/crew_hierarchical.py) - dynamic manager delegation with manager_llm.

Unified execution interface available in src/run_architecture.py.

Req 8 (Guardrails): every scenario text is validated through
src/guardrails.enforce_guardrails() before any agent is created or any
LLM call is made. Violations raise GuardrailViolation and are logged to
logs/guardrail_events.jsonl for auditability.

Run with:  python -m src.crew_sequential
"""

import time
import src.litellm_patch  # noqa: F401  (patches litellm before any LLM calls)
from crewai import Crew, Process
from src.tasks import build_tasks, DISRUPTION_SCENARIO
from src.guardrails import enforce_guardrails, guardrail_summary
from src.a2a_crew_bridge import perform_a2a_compliance_step
from src.observability import make_step_callback


def build_sequential_crew(scenario_text: str = None, agent_timing_sink: list = None,
                          tool_overrides: dict = None, run_id: str = None,
                          prompt_style: str = "react", structural_variant: str = "json_first",
                          persona: str = None) -> Crew:
    """
    Build the sequential crew with run-scoped state.
    Integrates A2A protocol audit with Google ADK and agent step_callback.
    """
    scenario_text = scenario_text or DISRUPTION_SCENARIO
    # --- GUARDRAIL: validate input before anything else ---
    enforce_guardrails(scenario_text)

    # --- REQUIREMENT 1: Reset Blackboard for this run to avoid contamination ---
    from src.blackboard import reset_blackboard, get_blackboard
    bb = reset_blackboard(initial_scenario=scenario_text, run_id=run_id)

    # --- REQUIREMENT 6 & 7: A2A Protocol step to Google ADK Compliance Agent ---
    try:
        perform_a2a_compliance_step(architecture="sequential", scenario_text=scenario_text, run_id=run_id)
    except Exception:
        pass

    from src import agents as _agents
    _agents.TOOL_OVERRIDES.clear()
    if tool_overrides:
        _agents.TOOL_OVERRIDES.update(tool_overrides)
    try:
        agents, tasks = build_tasks(
            scenario_text=scenario_text,
            parallel=False,
            prompt_style=prompt_style,
            structural_variant=structural_variant,
            persona=persona,
        )
    finally:
        _agents.TOOL_OVERRIDES.clear()

    SECTION_MAP = {
        "Supplier Monitoring Agent": "supplier_status",
        "Demand Forecasting Agent": "demand_forecast",
        "Inventory Agent": "inventory_projection",
        "Alternative Supplier Agent": "alternative_suppliers",
        "Cost Agent": "cost_analysis",
        "Manager Agent": "final_decision",
    }

    clock = {"last": time.time()}

    def make_callback(agent_role: str, task_description: str):
        def _on_task_done(task_output):
            now = time.time()
            output_text = getattr(task_output, "raw", None) or str(task_output)
            # 1. Publish to Blackboard shared state
            sec = SECTION_MAP.get(agent_role, agent_role.lower().replace(" ", "_"))
            bb.publish_agent_result(agent_role=agent_role, section=sec, content=output_text)

            # 2. Record agent timing if sink provided
            if agent_timing_sink is not None:
                agent_timing_sink.append({
                    "agent": agent_role,
                    "latency_seconds": round(now - clock["last"], 3),
                    "task_description": task_description,
                    "output_text": output_text,
                })
            clock["last"] = now
            # Throttle between agent steps to keep under 8,000 TPM limit
            import os
            throttle_s = int(os.environ.get("THROTTLE_SECONDS", "9"))
            if throttle_s > 0 and agent_role != "Manager Agent":
                print(f"[throttle] Pacing pipeline: sleeping {throttle_s}s after {agent_role} to stay under 8,000 TPM...")
                time.sleep(throttle_s)
        return _on_task_done

    for task in tasks:
        task.callback = make_callback(task.agent.role, task.description)

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        step_callback=make_step_callback(run_id=run_id),
    )


if __name__ == "__main__":
    import json, re
    from src.output_models import SupplyChainDecision
    from src.blackboard import get_blackboard

    timings = []
    crew = build_sequential_crew(agent_timing_sink=timings)
    result = crew.kickoff()

    print("\n\n===== FINAL MANAGER DECISION (Pydantic structured) =====")
    decision = None
    # 1. Check if CrewAI returned validated pydantic object directly
    if hasattr(result, "pydantic") and isinstance(result.pydantic, SupplyChainDecision):
        decision = result.pydantic
    else:
        # Fallback to inspecting manager task output pydantic or validating raw JSON
        raw_text = getattr(result, "raw", str(result))
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if json_match:
            try:
                decision = SupplyChainDecision.model_validate_json(json_match.group())
            except Exception as e:
                raise ValueError(f"Manager output violated SupplyChainDecision Pydantic schema: {e}")
        else:
            raise ValueError(f"Manager output failed structured Pydantic output requirement: {raw_text[:200]}")

    try:
        print(decision.to_markdown())
    except UnicodeEncodeError:
        print(decision.to_markdown().encode("ascii", errors="replace").decode("ascii"))

    print("\n===== BLACKBOARD SHARED STATE AUDIT =====")
    bb = get_blackboard()
    print(f"Total Blackboard Sections Populated: {len(bb.get_all_state())}")
    for sec in bb.get_all_state():
        print(f"  * [{sec}]")

    print("\n===== PER-AGENT LATENCY =====")
    for t in timings:
        print(f"  {t['agent']}: {t['latency_seconds']}s")
    print("\n===== GUARDRAIL SUMMARY =====")
    print(guardrail_summary())

