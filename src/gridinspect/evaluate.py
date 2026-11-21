from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import RUNS
from .data import CLASSES, Sample
from .prompts import WIDE

CRITICAL = {"crack", "thunderstrike"}


async def evaluate_model(model, samples: list[Sample], *, concurrency: int = 5) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    async def one(sample: Sample) -> None:
        async with semaphore:
            result = await model.inspect_image(sample.image, WIDE)
            pred = []
            if result.raw:
                try:
                    payload = json.loads(result.raw["candidates"][0]["content"]["parts"][0]["text"])
                except Exception:
                    try:
                        payload = json.loads(result.raw["choices"][0]["message"]["content"])
                    except Exception:
                        payload = {}
                pred = [c for c in payload.get("defects", []) if c in CLASSES]
                notes = payload.get("notes", "")
                severity = payload.get("severity", "none")
            else:
                notes, severity = "", "none"
            rows.append(
                {
                    "image": str(sample.image),
                    "name": sample.image.stem,
                    "gold": sample.classes,
                    "pred": sorted(set(pred)),
                    "severity": severity,
                    "notes": notes,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "latency_ms": result.latency_ms,
                    "cost_usd": result.cost_usd,
                    "error": result.error,
                }
            )

    await asyncio.gather(*(one(s) for s in samples))
    rows.sort(key=lambda r: r["name"])
    summary = summarize(rows)
    summary["model"] = model.model
    summary["wall_s"] = time.perf_counter() - started

    out = RUNS / model.model
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with open(out / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    exact = sum(1 for r in rows if set(r["gold"]) == set(r["pred"])) / n if n else 0
    tp = sum(len(set(r["gold"]) & set(r["pred"])) for r in rows)
    fp = sum(len(set(r["pred"]) - set(r["gold"])) for r in rows)
    fn = sum(len(set(r["gold"]) - set(r["pred"])) for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_class = {}
    for cls in CLASSES:
        gold = sum(1 for r in rows if cls in r["gold"])
        hit = sum(1 for r in rows if cls in r["gold"] and cls in r["pred"])
        per_class[cls] = {"gold": gold, "hit": hit, "recall": hit / gold if gold else 0.0}

    crit = [r for r in rows if set(r["gold"]) & CRITICAL]
    crit_hit = sum(1 for r in crit if set(r["pred"]) & CRITICAL)
    lat = [r["latency_ms"] for r in rows if not r["error"]]
    cost = sum(r["cost_usd"] for r in rows)
    return {
        "n": n,
        "errors": sum(1 for r in rows if r["error"]),
        "exact_set_match": exact,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "critical_recall": crit_hit / len(crit) if crit else 0.0,
        "critical_n": len(crit),
        "per_class": per_class,
        "cost_usd": cost,
        "cost_per_1000": cost / n * 1000 if n else 0.0,
        "latency_p50_ms": float(np.percentile(lat, 50)) if lat else 0.0,
        "latency_p95_ms": float(np.percentile(lat, 95)) if lat else 0.0,
    }
