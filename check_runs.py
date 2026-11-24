import json
from collections import defaultdict

rows = [json.loads(l) for l in open("logs/runs.jsonl")]

def classify(r):
    err = r.get("error")
    if not err and r.get("final_decision"):
        return "success"
    if err and "RateLimitError" in err:
        return "rate_limit"
    return "agent_error"

stats = defaultdict(lambda: defaultdict(int))
for r in rows:
    stats[r["architecture"]][classify(r)] += 1

print(f"Total attempts: {len(rows)}\n")
total_ok = 0
for arch, c in stats.items():
    n = sum(c.values())
    real = c["success"] + c["agent_error"]  # excludes quota failures
    total_ok += c["success"]
    print(f"{arch}: {n} attempts | success={c['success']} | "
          f"rate_limit={c['rate_limit']} | agent_error={c['agent_error']}")
    if real:
        print(f"   failure rate excluding quota: {c['agent_error']}/{real} "
              f"= {c['agent_error']/real:.0%}")

print(f"\nTotal successful runs: {total_ok} (target: 40+)")