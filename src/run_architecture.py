"""
Common Execution Interface for Supply Chain MAS Architectures.

Requirement 4: Make all three architectures executable
- Sequential (fixed chain)
- Parallel (fan-out / fan-in)
- Hierarchical (manager delegation)

Provides a single CLI interface to independently execute, verify, and test
any of the three agentic design patterns.

Usage:
    python -m src.run_architecture --architecture sequential
    python -m src.run_architecture --architecture parallel
    python -m src.run_architecture --architecture hierarchical
    python -m src.run_architecture --architecture sequential --dry-run
    python -m src.run_architecture --architecture parallel --tools_via mcp
"""

import argparse
import json
import os
import sys
import time
from contextlib import ExitStack

from src.tasks import DISRUPTION_SCENARIO
from src.crew_sequential import build_sequential_crew
from src.crew_parallel import build_parallel_crew
from src.crew_hierarchical import build_hierarchical_crew
from src.output_models import SupplyChainDecision
from src.blackboard import get_blackboard
from src.guardrails import enforce_guardrails, guardrail_summary
from src.mock_data import set_runtime_overrides, clear_runtime_overrides

ARCHITECTURES = {
    "sequential": build_sequential_crew,
    "parallel": build_parallel_crew,
    "hierarchical": build_hierarchical_crew,
}


def run_architecture(architecture: str, scenario_text: str = None,
                     tools_via: str = "local", dry_run: bool = False,
                     prompt_style: str = "react", persona: str = None,
                     custom_overrides: dict = None):
    """
    Execute or dry-run a specific multi-agent architecture pattern.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{architecture}'. Options: {list(ARCHITECTURES.keys())}")

    scenario = scenario_text or DISRUPTION_SCENARIO
    print("=" * 80)
    print(f"RUNNING ARCHITECTURE PATTERN: {architecture.upper()}")
    print("=" * 80)
    print(f"Scenario: {scenario[:90]}...")
    print(f"Tools Transport: {tools_via} | Prompt Style: {prompt_style} | Dry Run Mode: {dry_run}\n")

    # Guardrails check (allows sanitising angry executive language when supply-chain context is valid)
    scenario = enforce_guardrails(scenario, allow_sanitisation=True)
    print("[1/4] Guardrails validation: PASSED")

    prev_env_overrides = os.environ.get("MAS_CUSTOM_OVERRIDES")
    if custom_overrides:
        set_runtime_overrides(custom_overrides)
        os.environ["MAS_CUSTOM_OVERRIDES"] = json.dumps(custom_overrides)

    builder = ARCHITECTURES[architecture]
    timings = []

    try:
        with ExitStack() as stack:
            tool_overrides = None
            if tools_via == "mcp":
                from src.run_experiments import _open_mcp_tools
                print("[MCP] Spawning MCP server and discovering tools...")
                tool_overrides = _open_mcp_tools(stack)
                print(f"[MCP] Loaded {len(tool_overrides)} tools over MCP.")

            # Build crew
            crew = builder(
                scenario,
                agent_timing_sink=timings,
                tool_overrides=tool_overrides,
                prompt_style=prompt_style,
                persona=persona,
            )
            print(f"[2/4] Architecture Assembly: PASSED (Loaded {len(crew.agents)} agents, {len(crew.tasks)} tasks)")

            if dry_run:
                print("[3/4] Dry Run: Crew constructed successfully without making LLM calls.")
                bb = get_blackboard()
                print(f"[4/4] Blackboard initialized: {list(bb.get_all_state().keys())}")
                print(f"\nArchitecture '{architecture}' verified successfully in offline mode.")
                return None

            # Live Execution
            print("[3/4] Executing Multi-Agent Workflow...")
            start_time = time.time()
            result = crew.kickoff()
            total_time = round(time.time() - start_time, 2)
            print(f"[3/4] Multi-Agent Workflow Completed in {total_time}s")
    finally:
        if custom_overrides:
            clear_runtime_overrides()
            if prev_env_overrides is None:
                os.environ.pop("MAS_CUSTOM_OVERRIDES", None)
            else:
                os.environ["MAS_CUSTOM_OVERRIDES"] = prev_env_overrides

    # Structured Output Verification
    print("\n[4/4] Verifying Pydantic Structured Output...")
    decision = None
    if hasattr(result, "pydantic") and isinstance(result.pydantic, SupplyChainDecision):
        decision = result.pydantic
    else:
        import re
        raw_text = getattr(result, "raw", str(result))
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if json_match:
            decision = SupplyChainDecision.model_validate_json(json_match.group())
        else:
            raise ValueError(f"Output failed structured Pydantic schema validation: {raw_text[:200]}")

    print("\n" + "=" * 80)
    print(f"FINAL MANAGER DECISION REPORT ({architecture.upper()})")
    print("=" * 80)
    try:
        print(decision.to_markdown())
    except UnicodeEncodeError:
        print(decision.to_markdown().encode("ascii", errors="replace").decode("ascii"))

    print("\n" + "-" * 80)
    print("BLACKBOARD STATE SNAPSHOT:")
    bb = get_blackboard()
    for sec in bb.get_all_state():
        print(f"  * Section: [{sec}]")

    print("\nAGENT TIMING BREAKDOWN:")
    for t in timings:
        label = "finished at +" if t.get("kind") == "finish_offset" else "duration: "
        print(f"  * {t['agent']}: {label}{t['latency_seconds']}s")

    print(f"\n{guardrail_summary()}")
    print("=" * 80)
    return decision


def main():
    parser = argparse.ArgumentParser(description="Run any Supply Chain MAS architecture pattern.")
    parser.add_argument(
        "--architecture",
        choices=["sequential", "parallel", "hierarchical"],
        default="sequential",
        help="Architecture pattern to execute",
    )
    parser.add_argument("--scenario", type=str, default=None, help="Scenario description")
    parser.add_argument("--tools_via", choices=["local", "mcp"], default="local", help="Tool execution transport")
    parser.add_argument("--prompt_style", choices=["react", "cot", "direct"], default="react", help="Manager reasoning prompt style")
    parser.add_argument("--persona", type=str, default=None, help="Target persona directive")
    parser.add_argument("--overrides-json", type=str, default=None, help="JSON string of custom numeric overrides")
    parser.add_argument("--dry-run", action="store_true", help="Assemble and verify without calling LLMs")

    args = parser.parse_args()
    custom_overrides = json.loads(args.overrides_json) if args.overrides_json else None
    run_architecture(
        architecture=args.architecture,
        scenario_text=args.scenario,
        tools_via=args.tools_via,
        dry_run=args.dry_run,
        prompt_style=args.prompt_style,
        persona=args.persona,
        custom_overrides=custom_overrides,
    )


if __name__ == "__main__":
    main()
