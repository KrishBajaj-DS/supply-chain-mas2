"""
Experiment runner: executes the MAS across scenario/persona combinations,
logging per-run observability data (per-agent latency, per-agent
LLM-as-Judge accuracy, token usage, and cost) plus a whole-run judge
score and Pydantic structured output validation.

Usage:
    python -m src.run_experiments --architecture sequential --tools_via mcp --prompt_style react
    python -m src.run_experiments --fill-missing --target-per-scenario 5 --max-runs 6
    python -m src.run_experiments --architecture all --tools_via mcp

Flags:
  --architecture {sequential,parallel,hierarchical,all}  task-graph pattern (default: all)
  --tools_via {mcp,local}        mcp = tools via src/mcp_server.py over stdio (default: mcp); local = in-process
  --prompt_style {react,cot,direct}  Manager reasoning prompt style (default: react)
  --structural_variant {json_first,table_first} output format variant
  --runs_per_combo N             runs per scenario when not using --fill-missing (default: 5)
  --fill-missing                 inspect logs/runs.jsonl and run only missing combos up to target
  --target-per-scenario N        target completions per scenario in --fill-missing (default: 5)
  --max-runs K                   max runs to execute in this batch (quota-safe cap)
  --timeout SECONDS              hard timeout per run in seconds (default: 600)
  --judge_agents                 LLM-judge every agent's intermediate output
  --structured                   also produce a Pydantic-validated DecisionReport
"""

import argparse
import os
import sys
import time
import json
import re
import traceback
from contextlib import ExitStack
from collections import defaultdict
from typing import Dict, List, Optional, Any
import concurrent.futures

from src.scenarios import get_all_scenarios
from src.crew_sequential import build_sequential_crew
from src.crew_parallel import build_parallel_crew
from src.crew_hierarchical import build_hierarchical_crew
from src.observability import RunTracer, load_all_runs
from src.judge import judge_decision, judge_agent_output
from src.guardrails import validate_user_input, GuardrailViolation
from src.cost import usage_from_result, reset_token_counter
from src.output_models import SupplyChainDecision

BUILDERS = {
    "sequential": build_sequential_crew,
    "parallel": build_parallel_crew,
    "hierarchical": build_hierarchical_crew,
}


def _open_mcp_tools(stack: ExitStack) -> dict:
    """Start src/mcp_server.py over stdio and return {tool_name: CrewAI tool}."""
    from crewai_tools import MCPServerAdapter
    from mcp import StdioServerParameters
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server"],
        env=dict(os.environ),
    )
    adapter = stack.enter_context(MCPServerAdapter(params))
    tools_dict = {}
    for t in adapter:
        orig_run = getattr(t, "_run", None)
        if orig_run is not None:
            def _make_wrapper(tool_obj, original_run):
                def _wrapped(*args, **kwargs):
                    from src.observability import log_tool_event
                    res = original_run(*args, **kwargs)
                    try:
                        arguments = kwargs if kwargs else ({"args": [str(a) for a in args]} if args else {})
                        log_tool_event(agent=tool_obj.name, tool_name=tool_obj.name, arguments=arguments, result=str(res), status="ok")
                    except Exception:
                        pass
                    return res
                return _wrapped
            t._run = _make_wrapper(t, orig_run)
        tools_dict[t.name] = t
    return tools_dict


def run_once(scenario: dict, judge_agents: bool = False, architecture: str = "sequential",
             tools_via: str = "mcp", structured: bool = False,
             prompt_style: str = "react", structural_variant: str = "json_first",
             timeout_seconds: int = 600) -> Optional[dict]:
    """Execute a single end-to-end multi-agent run with safety timeout and tracing."""
    reset_token_counter()
    with RunTracer(architecture=architecture, persona=scenario["persona"],
                   scenario=scenario["id"]) as tracer:
        tracer.set_extra(
            prompt_style=prompt_style,
            structural_variant=structural_variant,
            tools_via=tools_via,
            persona=scenario["persona"],
            scenario=scenario["id"],
        )
        try:
            # 1. Input Guardrail: Screen input before execution
            g = validate_user_input(scenario["text"], log=False, allow_sanitisation=True, run_id=tracer.run_id)
            if not g.allowed:
                raise GuardrailViolation(f"{g.category}: {g.reason}", violation_type=g.category)
            cleaned_scenario_text = g.sanitized_text or scenario["text"]

            agent_timings = []
            result = None
            latency = 0.0

            def _execute_crew():
                nonlocal result, latency
                with ExitStack() as stack:
                    overrides = _open_mcp_tools(stack) if tools_via == "mcp" else None
                    builder_fn = BUILDERS[architecture]
                    crew = builder_fn(
                        scenario_text=cleaned_scenario_text,
                        agent_timing_sink=agent_timings,
                        tool_overrides=overrides,
                        run_id=tracer.run_id,
                        prompt_style=prompt_style,
                        structural_variant=structural_variant,
                        persona=scenario["persona"],
                    )
                    start = time.time()
                    res = crew.kickoff()
                    latency = time.time() - start
                    return res

            # Run with hard timeout
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_execute_crew)
            try:
                result = future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                executor.shutdown(wait=False, cancel_futures=True)
                tracer.set_extra(error_category="timeout", error_message=f"Run exceeded {timeout_seconds}s limit", tools_via=tools_via)
                print(f"[TIMEOUT] {architecture} / {scenario['id']} exceeded {timeout_seconds}s limit.")
                return None
            finally:
                executor.shutdown(wait=False)

            decision_text = str(result)
            tracer.set_final_decision(decision_text)

            # Record Blackboard shared state provenance
            from src.blackboard import get_blackboard
            bb = get_blackboard()
            tracer.set_provenance(bb.get_all_state())

            # Validate structured output explicitly
            validated_decision = getattr(result, "pydantic", None)
            validation_details = "Direct CrewAI output_pydantic"
            if not (validated_decision and isinstance(validated_decision, SupplyChainDecision)):
                json_match = re.search(r'\{[\s\S]*\}', decision_text)
                if json_match:
                    try:
                        validated_decision = SupplyChainDecision.model_validate_json(json_match.group())
                        validation_details = "JSON regex extraction validated"
                    except Exception as ve:
                        validation_details = f"Extraction failed: {ve}"
                else:
                    validation_details = "No JSON object discovered in decision output"

            if validated_decision and isinstance(validated_decision, SupplyChainDecision):
                status = getattr(validated_decision, "validation_status", "VALID")
                tracer.set_validation_outcome(passed=(status == "VALID"), schema="SupplyChainDecision", details=validation_details)
            else:
                tracer.set_validation_outcome(passed=False, schema="SupplyChainDecision", details=validation_details)

            extra = {
                "crew_latency_seconds": round(latency, 2),
                "tools_via": tools_via,
                "prompt_style": prompt_style,
                "structural_variant": structural_variant,
            }
            if validated_decision:
                extra["structured_decision"] = validated_decision.model_dump()
            usage = usage_from_result(result)
            if usage:
                extra.update(usage)
            tracer.set_extra(**extra)

            for t in agent_timings:
                agent_judge = None
                if judge_agents:
                    agent_judge = judge_agent_output(
                        agent_role=t["agent"],
                        task_description=t.get("task_description", ""),
                        output_text=t.get("output_text", ""),
                    )
                tracer.log_agent_step(t["agent"], t["latency_seconds"],
                                      judge_score=agent_judge, kind=t.get("kind"))

            if structured:
                from src.structured_output import structure_decision
                report, method = structure_decision(decision_text)
                tracer.set_extra(structuring_method=method,
                                 structured_decision=report.model_dump() if report else None)

            # Offline or live evaluation scoring
            val_stat = getattr(validated_decision, "validation_status", "VALID") if validated_decision else "INVALID_LOGIC"
            if decision_text and decision_text.strip():
                judged = judge_decision(scenario["text"], decision_text, validation_status=val_stat)
                tracer.set_judge_score(judged.get("score") if judged else None, judged.get("rationale") if judged else "")
            else:
                judged = None
                tracer.set_judge_score(None, "")

            tok = f" tokens={usage['total_tokens']} est_cost=${usage['estimated_cost_usd']}" if usage else ""
            score_disp = judged.get('score') if judged else 'N/A'
            print(f"[OK] {architecture} / {scenario['id']} [{prompt_style}] -> score={score_disp} "
                  f"latency={round(latency, 2)}s{tok}")
            for t in agent_timings:
                label = "finished at +" if t.get("kind") == "finish_offset" else ""
                print(f"    {t['agent']}: {label}{t['latency_seconds']}s")
            return tracer.to_dict()

        except Exception as e:
            err_str = str(e)
            err_lower = err_str.lower()
            if any(k in err_lower for k in ["tpd", "tokens per day", "daily_limit", "quota", "exceeded your current quota"]):
                cat = "rate_limit_quota"
            elif any(k in err_lower for k in ["ratelimit", "rate_limit", "tpm", "tokens per minute", "too many requests"]):
                cat = "rate_limit_tpm"
            else:
                cat = "execution_error"
            print(f"[FAIL] {architecture} / {scenario['id']}: {e}")
            tracer.set_extra(error=err_str, error_category=cat, tools_via=tools_via, prompt_style=prompt_style)
            traceback.print_exc()
            raise


def get_existing_run_counts() -> Dict[str, Dict[str, int]]:
    """Tally existing runs from logs/runs.jsonl per architecture and scenario."""
    counts = defaultdict(lambda: defaultdict(int))
    runs = load_all_runs()
    for r in runs:
        arch = r.get("architecture")
        scen = r.get("scenario")
        if arch and scen:
            counts[arch][scen] += 1
    return counts


def main(runs_per_combo: int, judge_agents: bool, architecture: str, tools_via: str,
         structured: bool, prompt_style: str, structural_variant: str,
         fill_missing: bool, target_per_scenario: int, max_runs: Optional[int],
         timeout: int, scenario_id: Optional[str] = None):
    scenarios = get_all_scenarios()
    if scenario_id:
        scenarios = [s for s in scenarios if s["id"] == scenario_id]
        if not scenarios:
            print(f"Error: Scenario '{scenario_id}' not found.")
            return
    scenarios_by_id = {s["id"]: s for s in scenarios}
    architectures = list(BUILDERS.keys()) if architecture == "all" else [architecture]

    # Plan execution queue
    queue: List[tuple] = []  # list of (arch, scenario_dict)

    if fill_missing:
        existing_counts = get_existing_run_counts()
        print("=== FILL-MISSING MODE: Current Log Inventory ===")
        for arch in architectures:
            print(f"Architecture [{arch}]:")
            for s in scenarios:
                sid = s["id"]
                done = existing_counts[arch][sid]
                deficit = max(0, target_per_scenario - done)
                print(f"  - {sid:25s}: {done}/{target_per_scenario} (Deficit: {deficit})")
                for _ in range(deficit):
                    queue.append((arch, s))
        print(f"\nTotal runs needed across {len(architectures)} architecture(s): {len(queue)}")
    else:
        for i in range(runs_per_combo):
            for arch in architectures:
                for s in scenarios:
                    queue.append((arch, s))

    if max_runs is not None and max_runs < len(queue):
        print(f"Capping execution queue from {len(queue)} to {max_runs} runs via --max-runs.")
        queue = queue[:max_runs]

    print(f"\nPlanned execution queue size: {len(queue)} runs | tools_via={tools_via} | prompt_style={prompt_style}")
    completed = 0
    start_time = time.time()

    for idx, (arch, scenario) in enumerate(queue):
        elapsed = time.time() - start_time
        avg_time = elapsed / completed if completed > 0 else 60.0
        rem_runs = len(queue) - idx
        est_rem_min = (rem_runs * avg_time) / 60.0
        print(f"\n[{idx+1}/{len(queue)}] Running {arch} / {scenario['id']} (Est. {est_rem_min:.1f} min remaining)...")

        try:
            run_once(
                scenario=scenario,
                judge_agents=judge_agents,
                architecture=arch,
                tools_via=tools_via,
                structured=structured,
                prompt_style=prompt_style,
                structural_variant=structural_variant,
                timeout_seconds=timeout,
            )
            completed += 1
        except Exception as e:
            err_msg = str(e).lower()
            if "tokens per day" in err_msg or "tpd" in err_msg or "429" in err_msg or "ratelimit" in err_msg or "rate limit" in err_msg:
                print("\n[QUOTA] LLM API rate limit reached (429) - safely stopping experiment batch immediately.")
                print(f"Runs completed this session: {completed}. Total in logs/runs.jsonl: {len(load_all_runs())}")
                return
            print(f"Continuing queue despite error in {arch} / {scenario['id']}.")

    print(f"\nAll scheduled runs completed. Total runs now logged in logs/runs.jsonl: {len(load_all_runs())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Supply Chain Experiment Runner")
    parser.add_argument("--runs_per_combo", type=int, default=5, help="Runs per scenario/persona combo")
    parser.add_argument("--judge_agents", action="store_true", help="Run LLM-as-judge on each agent step")
    parser.add_argument("--architecture", choices=["sequential", "parallel", "hierarchical", "all"], default="all")
    parser.add_argument("--tools_via", choices=["mcp", "local"], default="mcp", help="Tools transport")
    parser.add_argument("--prompt_style", choices=["react", "cot", "direct"], default="react")
    parser.add_argument("--structural_variant", choices=["json_first", "table_first"], default="json_first")
    parser.add_argument("--structured", action="store_true", help="Post-process decision into Pydantic model")
    parser.add_argument("--fill-missing", action="store_true", help="Fill missing runs up to target")
    parser.add_argument("--target-per-scenario", type=int, default=5, help="Target runs per scenario in fill-missing")
    parser.add_argument("--max-runs", type=int, default=None, help="Cap maximum runs to execute")
    parser.add_argument("--timeout", type=int, default=600, help="Per-run timeout in seconds")
    parser.add_argument("--scenario", type=str, default=None, help="Filter to specific scenario ID (e.g. cost_conscious_cfo)")
    a = parser.parse_args()

    main(
        runs_per_combo=a.runs_per_combo,
        judge_agents=a.judge_agents,
        architecture=a.architecture,
        tools_via=a.tools_via,
        structured=a.structured,
        prompt_style=a.prompt_style,
        structural_variant=a.structural_variant,
        fill_missing=a.fill_missing,
        target_per_scenario=a.target_per_scenario,
        max_runs=a.max_runs,
        timeout=a.timeout,
        scenario_id=a.scenario,
    )
