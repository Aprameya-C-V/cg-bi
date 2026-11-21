from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / "keys.env", override=False)
DATA = ROOT / "data"
CPLID = DATA / "InsulatorDataSet-master"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"


@dataclass(frozen=True)
class Price:
    input_usd_per_m: float
    output_usd_per_m: float

    def cost(self, prompt: int, completion: int) -> float:
        return (prompt * self.input_usd_per_m + completion * self.output_usd_per_m) / 1_000_000


# Published list prices, used for business cost even when the run is free tier.
PRICES = {
    "gemini-3.5-flash-lite": Price(0.30, 2.50),
    "gemini-3.8-flash": Price(0.75, 3.75),
    "deepseek-v4.1-flash": Price(0.15, 0.60),
    "gemini-4-argon": Price(2.00, 10.00),
}


def gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    path = Path.home() / ".config" / "gemini-api-key"
    return path.read_text().strip()


def go_key() -> str:
    key = os.getenv("HBC_GO_API_KEY")
    if key:
        return key
    auth = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    return json.loads(auth.read_text())["opencode-go"]["key"]


def go_base() -> str:
    return os.getenv("HBC_GO_BASE_URL", "https://opencode.ai/inference/go/openai/v1")
