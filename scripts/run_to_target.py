"""
Resumable driver script to reach 42 VALID runs in logs/runs.jsonl:
- Balanced as: sequential 14, parallel 14, hierarchical 14
- Spread across 3 personas (Operations Manager, CFO, Supply Chain Planner)
- Spread across 3 disruptions (Widget-A / Port Congestion, Sensor-B / Factory Fire, Cable-Harness / Customs Hold)
- Spread across prompt styles: react, cot, direct
- API key rotation across GROQ_API_KEY_1 ... GROQ_API_KEY_N
- Immediate flush to runs.jsonl
- Retries INVALID_LOGIC once per combination
- Automated backfill, prompt ablation, and report regeneration upon completion
"""

import os
import sys
import json
import time
import re
import subprocess
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from src.run_experiments import run_once
from src.scenarios import get_all_scenarios
from src.output_models import SupplyChainDecision

RUNS_FILE = os.path.join(PROJECT_ROOT, "logs", "runs.jsonl")
PROMPT_ABLATION_FILE = os.path.join(PROJECT_ROOT, "logs", "prompt_ablation.jsonl")

TARGET_PER_ARCH = 14
TOTAL_TARGET = 42
ARCHITECTURES = ["sequential", "parallel", "hierarchical"]
PROMPT_STYLES = ["react", "cot", "direct"]

# 9 core scenarios covering 3 disruptions x 3 personas
CORE_SCENARIO_IDS = [
    # Disruption 1: Port Congestion (Widget-A)
    "urgent_ops_manager",
    "cost_conscious_cfo",
    "risk_averse_planner",
    # Disruption 2: Factory Fire (Sensor-B)
    "critical_ops_sensor",
    "cost_cfo_sensor",
    "risk_planner_sensor",
    # Disruption 3: Customs Regulatory Hold (Cable-Harness)
    "audit_ops_cable",
    "cost_cfo_cable",
    "risk_planner_cable",
]


class KeyManager:
    """Manages Groq API keys with rotation on daily quota (TPD) exhaustion."""

    def __init__(self):
        load_dotenv()
        self.keys = []
        i = 1
        while True:
            k = os.getenv(f"GROQ_API_KEY_{i}")
            if not k:
                break
            self.keys.append(k.strip())
            i += 1
        if not self.keys:
            single = os.getenv("GROQ_API_KEY")
            if single:
                self.keys.append(single.strip())

        self.current_index = 0
        self.exhausted_indices = set()
        if self.keys:
            self._apply_key(self.keys[0])

    def _apply_key(self, key: str):
        os.environ["GROQ_API_KEY"] = key
        try:
            import litellm
            litellm.api_key = key
        except Exception:
            pass
        try:
            import src.agents as _agents
            from crewai import LLM
            _agents.llm = LLM(
                model="groq/openai/gpt-oss-120b",
                api_key=key,
                temperature=0.3,
            )
        except Exception:
            pass

    def rotate_on_tpd(self) -> bool:
        """Mark current key as exhausted and rotate to next key. Return False if all exhausted."""
        self.exhausted_indices.add(self.current_index + 1)  # 1-based index
        for idx in range(len(self.keys)):
            if (idx + 1) not in self.exhausted_indices:
                self.current_index = idx
                self._apply_key(self.keys[idx])
                return True
        return False


def is_tpd_error(err_str: str) -> bool:
    s = err_str.lower()
    return any(k in s for k in [
        "tokens per day", "tpd", "daily_limit", "quota",
        "exceeded your current quota", "limit 200000"
    ])


def is_tpm_error(err_str: str) -> bool:
    s = err_str.lower()
    return any(k in s for k in [
        "tokens per minute", "tpm", "too many requests",
        "ratelimit", "rate limit"
    ]) and not is_tpd_error(err_str)


def extract_wait_seconds(err_str: str) -> float:
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", err_str, re.IGNORECASE)
    if m:
        mins = float(m.group(1)) if m.group(1) else 0.0
        secs = float(m.group(2))
        return max(5.0, mins * 60.0 + secs + 2.0)
    m = re.search(r"wait(?:ing)? (?:(\d+)m)?([\d.]+)s", err_str, re.IGNORECASE)
    if m:
        mins = float(m.group(1)) if m.group(1) else 0.0
        secs = float(m.group(2))
        return max(5.0, mins * 60.0 + secs + 2.0)
    return 25.0


def is_run_valid(r: dict) -> bool:
    """Determine if a logged run record meets the strict VALID criterion."""
    if r.get("error"):
        return False
    dec = r.get("final_decision")
    if not dec or not str(dec).strip():
        return False

    extra = r.get("extra") or {}
    s_dec = extra.get("structured_decision")
    if isinstance(s_dec, dict):
        try:
            obj = SupplyChainDecision.model_validate(s_dec)
            if obj.validation_status == "VALID":
                return True
        except Exception:
            pass

    vo = r.get("validation_outcome") or {}
    return vo.get("passed") is True


def get_existing_valid_runs() -> List[dict]:
    """Scan logs/runs.jsonl and return all records that are strictly VALID."""
    if not os.path.exists(RUNS_FILE):
        return []
    valid = []
    with open(RUNS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if is_run_valid(r):
                    valid.append(r)
            except Exception:
                pass
    return valid


def pick_next_task(valid_runs: List[dict], scenarios_by_id: Dict[str, dict]) -> Optional[Tuple[str, dict, str]]:
    """
    Round-robin picker:
    1. Select architecture with fewest valid runs (< 14).
    2. Select scenario within that architecture with fewest valid runs.
    3. Select prompt style (react/cot/direct) with fewest valid runs for that combo.
    """
    arch_counts = Counter(r.get("architecture") for r in valid_runs)
    underfilled_archs = [a for a in ARCHITECTURES if arch_counts[a] < TARGET_PER_ARCH]
    if not underfilled_archs:
        return None  # All targets met

    # Pick architecture with lowest valid count (deterministic tie-break)
    chosen_arch = min(underfilled_archs, key=lambda a: (arch_counts[a], ARCHITECTURES.index(a)))

    # Tally scenario counts within chosen architecture
    scen_counts = Counter(
        r.get("scenario") for r in valid_runs if r.get("architecture") == chosen_arch
    )
    # Pick scenario with lowest valid count among core scenarios
    chosen_scen_id = min(CORE_SCENARIO_IDS, key=lambda sid: (scen_counts[sid], CORE_SCENARIO_IDS.index(sid)))
    chosen_scen = scenarios_by_id[chosen_scen_id]

    # Tally prompt styles for this (arch, scenario)
    style_counts = Counter(
        (r.get("extra") or {}).get("prompt_style", "react")
        for r in valid_runs
        if r.get("architecture") == chosen_arch and r.get("scenario") == chosen_scen_id
    )
    chosen_style = min(PROMPT_STYLES, key=lambda st: (style_counts[st], PROMPT_STYLES.index(st)))

    return chosen_arch, chosen_scen, chosen_style


def print_progress(valid_runs: List[dict], total_attempts: int):
    counts = Counter(r.get("architecture") for r in valid_runs)
    s_cnt = counts.get("sequential", 0)
    p_cnt = counts.get("parallel", 0)
    h_cnt = counts.get("hierarchical", 0)
    total_valid = len(valid_runs)
    print(
        f"\n[PROGRESS] valid {total_valid}/{TOTAL_TARGET} | "
        f"sequential {s_cnt}/{TARGET_PER_ARCH}, "
        f"parallel {p_cnt}/{TARGET_PER_ARCH}, "
        f"hierarchical {h_cnt}/{TARGET_PER_ARCH} | "
        f"attempts {total_attempts}"
    )


def print_final_table(valid_runs: List[dict], exhausted_indices: set):
    print("\n" + "=" * 90)
    print(" FINAL TABLE OF VALID RUNS ACROSS BALANCED MATRIX (Target: 42)")
    print("=" * 90)
    header = f"{'Architecture':14s} | {'Persona':22s} | {'Scenario':22s} | {'Prompt Style':12s} | {'Count':5s}"
    print(header)
    print("-" * 90)

    breakdown = Counter(
        (r.get("architecture"), (r.get("extra") or {}).get("persona", r.get("persona", "Unknown")), r.get("scenario"), (r.get("extra") or {}).get("prompt_style", "react"))
        for r in valid_runs
    )

    tot_tokens = 0
    tot_cost = 0.0
    for r in valid_runs:
        extra = r.get("extra") or {}
        tot_tokens += extra.get("total_tokens") or 0
        tot_cost += extra.get("estimated_cost_usd") or 0.0

    for (arch, persona, scen, style), cnt in sorted(breakdown.items()):
        short_persona = persona.split("-")[0].strip()[:20]
        print(f"{arch:14s} | {short_persona:22s} | {scen:22s} | {style:12s} | {cnt:5d}")

    print("-" * 90)
    arch_counts = Counter(r.get("architecture") for r in valid_runs)
    print(f"Total VALID Runs : {len(valid_runs)} / {TOTAL_TARGET}")
    print(f"  * Sequential   : {arch_counts.get('sequential', 0)} / {TARGET_PER_ARCH}")
    print(f"  * Parallel     : {arch_counts.get('parallel', 0)} / {TARGET_PER_ARCH}")
    print(f"  * Hierarchical : {arch_counts.get('hierarchical', 0)} / {TARGET_PER_ARCH}")
    print(f"Total Tokens     : {tot_tokens:,} tokens")
    print(f"Total Est. Cost  : ${tot_cost:.4f}")
    if exhausted_indices:
        print(f"Exhausted Keys   : {sorted(list(exhausted_indices))} (by 1-based index)")
    else:
        print("Exhausted Keys   : None")
    print("=" * 90 + "\n")


def check_prompt_ablation_status() -> bool:
    """Check if logs/prompt_ablation.jsonl has successful runs for styles."""
    if not os.path.exists(PROMPT_ABLATION_FILE):
        return False
    styles_seen = set()
    with open(PROMPT_ABLATION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not row.get("error") and row.get("style"):
                    styles_seen.add(row["style"])
            except Exception:
                pass
    return len(styles_seen) >= 3


def run_post_completion_steps(key_mgr: KeyManager):
    """Run judge backfill, prompt ablation (if needed), and report regeneration."""
    print("\n" + "=" * 80)
    print(" EXECUTING POST-COMPLETION PIPELINE (Judge Backfill -> Ablation -> Reports)")
    print("=" * 80)

    # 1. Judge backfill
    print("\n[Step 1/3] Running LLM-as-Judge backfill...")
    subprocess.run([sys.executable, "-m", "src.judge", "--backfill"], check=False)

    # 2. Prompt ablation (if styles incomplete and active key available)
    if not check_prompt_ablation_status():
        print("\n[Step 2/3] Running Prompt Ablation experiment (--repeats 1)...")
        subprocess.run([sys.executable, "-m", "src.prompt_ablation", "--repeats", "1"], check=False)
    else:
        print("\n[Step 2/3] Prompt ablation already completed with all styles. Generating ablation summary...")
        subprocess.run([sys.executable, "-m", "src.prompt_ablation", "--analyze"], check=False)

    # 3. Report regeneration
    print("\n[Step 3/3] Running Report & Chart Regeneration...")
    subprocess.run([sys.executable, "regenerate_reports.py"], check=False)
    print("\nPost-completion pipeline complete.\n")


def main():
    key_mgr = KeyManager()
    if not key_mgr.keys:
        print("ERROR: No Groq API keys found in .env (checked GROQ_API_KEY_1..N and GROQ_API_KEY).")
        return

    all_scenarios = get_all_scenarios()
    scenarios_by_id = {s["id"]: s for s in all_scenarios}

    total_attempts = 0
    all_keys_exhausted = False

    valid_runs = get_existing_valid_runs()
    print_progress(valid_runs, total_attempts)

    while len(valid_runs) < TOTAL_TARGET:
        next_pick = pick_next_task(valid_runs, scenarios_by_id)
        if not next_pick:
            print("\nAll architecture targets (14 sequential, 14 parallel, 14 hierarchical) are met!")
            break

        arch, scenario, prompt_style = next_pick
        scen_id = scenario["id"]
        print(f"\n>>> Selected Task: arch={arch} | scenario={scen_id} | style={prompt_style} (Key #{key_mgr.current_index + 1})")

        combo_attempts = 0
        combo_success = False

        while combo_attempts < 2:  # Run once; retry once if INVALID_LOGIC
            combo_attempts += 1
            total_attempts += 1

            try:
                res = run_once(
                    scenario=scenario,
                    judge_agents=False,
                    architecture=arch,
                    tools_via="mcp",
                    structured=False,
                    prompt_style=prompt_style,
                    structural_variant="json_first",
                    timeout_seconds=600,
                )

                if res and is_run_valid(res):
                    print(f"[SUCCESS] {arch} / {scen_id} [{prompt_style}] -> VALID.")
                    combo_success = True
                    break
                else:
                    v_stat = "INVALID_LOGIC"
                    if res:
                        s_dec = (res.get("extra") or {}).get("structured_decision") or {}
                        v_stat = s_dec.get("validation_status", "INVALID_LOGIC")
                    print(f"[INVALID] {arch} / {scen_id} [{prompt_style}] produced {v_stat}.")
                    if combo_attempts < 2:
                        print(f"[RETRY] Retrying {arch} / {scen_id} [{prompt_style}] once...")
                        time.sleep(5)
                        continue
                    else:
                        print(f"[RECORDED FAILED] {arch} / {scen_id} failed validation on retry. Moving on.")
                        break

            except Exception as e:
                err_str = str(e)
                if is_tpd_error(err_str):
                    print(f"\n[TPD QUOTA] Daily quota exceeded on Key #{key_mgr.current_index + 1}.")
                    rotated = key_mgr.rotate_on_tpd()
                    if not rotated:
                        print("[ALL KEYS EXHAUSTED] All Groq API keys have exhausted their daily quota.")
                        all_keys_exhausted = True
                        break
                    else:
                        print(f"[KEY ROTATION] Switched to Key #{key_mgr.current_index + 1}. Retrying same combination...")
                        combo_attempts -= 1  # Do not count rotation against combination attempt count
                        time.sleep(5)
                        continue
                elif is_tpm_error(err_str):
                    wait_s = extract_wait_seconds(err_str)
                    print(f"\n[TPM LIMIT] Rate limit on Key #{key_mgr.current_index + 1}. Waiting {wait_s:.1f}s before retry...")
                    time.sleep(wait_s)
                    combo_attempts -= 1
                    continue
                else:
                    print(f"[ERROR] Run failed with execution error: {e}")
                    break

        # Refresh valid runs count from disk
        valid_runs = get_existing_valid_runs()
        print_progress(valid_runs, total_attempts)

        if all_keys_exhausted:
            break

        # Small inter-run cooldown to prevent TPM spikes
        time.sleep(3)

    # Post-completion automated pipeline
    run_post_completion_steps(key_mgr)

    # Print final summary table
    valid_runs = get_existing_valid_runs()
    print_final_table(valid_runs, key_mgr.exhausted_indices)


if __name__ == "__main__":
    main()

