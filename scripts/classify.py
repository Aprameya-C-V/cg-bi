from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from gridinspect.providers import GeminiVision


async def main() -> None:
    model = GeminiVision("gemini-3.5-flash-lite")
    try:
        result = await model.inspect_image(Path(sys.argv[1]), __import__("gridinspect.prompts", fromlist=["WIDE"]).WIDE)
    finally:
        await model.aclose()
    payload = {}
    if result.raw:
        payload = json.loads(result.raw["candidates"][0]["content"]["parts"][0]["text"])
    payload["_cost_usd"] = result.cost_usd
    payload["_latency_ms"] = round(result.latency_ms, 1)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
