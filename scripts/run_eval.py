from __future__ import annotations

import argparse
import asyncio
import json

from cgbi.data import load_manifest, make_split
from cgbi.evaluate import evaluate_model
from cgbi.providers import DeepSeekVision, GeminiVision


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both", choices=["gemini", "deepseek", "both"])
    parser.add_argument("--per-class", type=int, default=13)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    samples = make_split(load_manifest(), per_class=args.per_class)
    print(f"{len(samples)} images")

    models = []
    if args.model in ("gemini", "both"):
        models.append(GeminiVision("gemini-3.5-flash-lite"))
    if args.model in ("deepseek", "both"):
        models.append(DeepSeekVision("deepseek-v4.1-flash"))
    try:
        for model in models:
            print(f"== {model.model}")
            summary = await evaluate_model(model, samples, concurrency=args.concurrency)
            print(json.dumps(summary, indent=2))
    finally:
        for model in models:
            await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
