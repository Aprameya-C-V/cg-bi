import json
from pathlib import Path

from cgbi.evaluate import summarize

RUNS = Path("runs")


def rows(name):
    return {json.loads(l)["name"]: json.loads(l) for l in (RUNS / name / "results.jsonl").read_text().splitlines()}


lite = rows("gemini-3.5-flash-lite")
argon = rows("gemini-4-argon")
names = sorted(lite)
lite_cost = sum(lite[n]["cost_usd"] for n in names)

out = []
for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    merged, esc = [], 0
    for n in names:
        use_argon = lite[n].get("max_confidence", 0) < t
        src = argon[n] if use_argon else lite[n]
        esc += int(use_argon)
        merged.append({
            "gold": src["gold"], "pred": src["pred"], "latency_ms": src["latency_ms"],
            "cost_usd": src["cost_usd"], "error": src["error"],
        })
    s = summarize(merged)
    blended = (lite_cost + sum(argon[n]["cost_usd"] for n in names if lite[n].get("max_confidence", 0) < t)) / len(names) * 1000
    out.append({
        "threshold": t, "escalation": esc / len(names), "exact": s["exact_set_match"],
        "f1": s["f1"], "critical_recall": s["critical_recall"], "cost_per_1000": blended,
        "p50": sorted(m["latency_ms"] for m in merged)[len(merged)//2],
    })

for r in out:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
(RUNS / "cascade.json").write_text(json.dumps(out, indent=2))
