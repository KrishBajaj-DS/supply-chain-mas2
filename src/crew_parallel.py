"""
Parallel (fan-out / fan-in) architecture for the Supply Chain MAS.

Same six agents and tools as the sequential crew; the only change is the task
graph (see build_tasks(parallel=True)): Demand Forecasting and Alternative
Supplier run concurrently, Inventory waits for both, then Cost and Manager
run in order. Only one pair can overlap because every later step consumes
earlier outputs, which limits the possible speed-up (Amdahl's law).

Per-agent timing in a parallel crew cannot be a simple "time since previous
task" figure, so each agent records its FINISH OFFSET (seconds since the crew
started) with kind='finish_offset'. Compare architectures on total crew
latency, judge score and failure rate, not on per-agent latency.

Run with:  python -m src.crew_parallel
"""

import time
import src.litellm_patch  # noqa: F401  (patches litellm before any LLM calls)
from crewai import Crew, Process
from src.tasks import build_tasks, DISRUPTION_SCENARIO
from src.guardrails import enforce_guardrails
from src.a2a_crew_bridge import perform_a2a_compliance_step
from src.observability import make_step_callback


def build_parallel_crew(scenario_text: str = None, agent_timing_sink: list = None,
                        tool_overrides: dict = None, run_id: str = None,
                        prompt_style: str = "react", structural_variant: str = "json_first",
                        persona: str = None) -> Crew:
    scenario_text = scenario_text or DISRUPTION_SCENARIO
    # --- GUARDRAIL: validate input before any agent or LLM call (Req 8) ---
    enforce_guardrails(scenario_text)

    # --- REQUIREMENT 1: Reset Blackboard for this run ---
    from src.blackboard import reset_blackboard
    bb = reset_blackboard(initial_scenario=scenario_text, run_id=run_id)

    # --- REQUIREMENT 6 & 7: A2A Protocol step to Google ADK Compliance Agent ---
    try:
        perform_a2a_compliance_step(architecture="parallel", scenario_text=scenario_text, run_id=run_id)
    except Exception:
        pass

    from src import agents as _agents
    _agents.TOOL_OVERRIDES.clear()
    if tool_overrides:
        _agents.TOOL_OVERRIDES.update(tool_overrides)
    try:
        agents, tasks = build_tasks(
            scenario_text=scenario_text,
            parallel=True,
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

    t0 = time.time()

    def make_callback(agent_role: str, task_description: str):
        def _on_task_done(task_output):
            output_text = getattr(task_output, "raw", None) or str(task_output)
            sec = SECTION_MAP.get(agent_role, agent_role.lower().replace(" ", "_"))
            bb.publish_agent_result(agent_role=agent_role, section=sec, content=output_text)

            if agent_timing_sink is not None:
                agent_timing_sink.append({
                    "agent": agent_role,
                    "latency_seconds": round(time.time() - t0, 3),
                    "kind": "finish_offset",
                    "task_description": task_description,
                    "output_text": output_text,
                })
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
    import re
    from src.output_models import SupplyChainDecision
    from src.blackboard import get_blackboard

    timings = []
    crew = build_parallel_crew(agent_timing_sink=timings)
    start = time.time()
    result = crew.kickoff()
    print("\n===== FINAL MANAGER DECISION (PARALLEL) =====")
    decision = None
    if hasattr(result, "pydantic") and isinstance(result.pydantic, SupplyChainDecision):
        decision = result.pydantic
    else:
        raw_text = getattr(result, "raw", str(result))
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if json_match:
            decision = SupplyChainDecision.model_validate_json(json_match.group())
        else:
            raise ValueError(f"Manager output failed structured Pydantic output requirement: {raw_text[:200]}")

    try:
        print(decision.to_markdown())
    except UnicodeEncodeError:
        print(decision.to_markdown().encode("ascii", errors="replace").decode("ascii"))

    print(f"\nTotal crew time: {time.time() - start:.1f}s")
    for t in sorted(timings, key=lambda x: x["latency_seconds"]):
        print(f"  finished at +{t['latency_seconds']}s: {t['agent']}")

    print("\n===== BLACKBOARD SHARED STATE AUDIT =====")
    bb = get_blackboard()
    print(f"Total Blackboard Sections Populated: {len(bb.get_all_state())}")
