from __future__ import annotations

import asyncio
import base64
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import PRICES, gemini_key, go_base, go_key


@dataclass
class VisionResult:
    defect_present: bool = False
    defect_type: str = "none"
    severity: str = "none"
    boxes: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    retries: int = 0
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


class _Base:
    model: str
    name: str

    def __init__(self, model: str, *, max_retries: int = 5, timeout: float = 120.0):
        self.model = model
        self.name = model
        self.max_retries = max_retries
        self.retries = 0
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=20.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    def _result(self, payload: dict, prompt: int, completion: int, latency: float, raw: dict) -> VisionResult:
        price = PRICES.get(self.model)
        return VisionResult(
            defect_present=bool(payload.get("defect_present", False)),
            defect_type=str(payload.get("defect_type", "none")),
            severity=str(payload.get("severity", "none")),
            boxes=list(payload.get("boxes") or []),
            notes=str(payload.get("notes", "")),
            prompt_tokens=prompt,
            completion_tokens=completion,
            latency_ms=latency,
            cost_usd=price.cost(prompt, completion) if price else 0.0,
            retries=self.retries,
            raw=raw,
        )


class GeminiVision(_Base):
    def __init__(self, model: str = "gemini-3.5-flash-lite", **kwargs):
        super().__init__(model, **kwargs)
        self._headers = {"x-goog-api-key": gemini_key(), "content-type": "application/json"}

    async def _generate(self, parts: list[dict[str, Any]]) -> VisionResult:
        self.retries = 0
        body = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "maxOutputTokens": 600},
        }
        started = time.perf_counter()
        response = None
        for attempt in range(self.max_retries + 1):
            response = await self._client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                headers=self._headers,
                json=body,
            )
            if response.status_code not in (408, 429, 500, 502, 503, 504) or attempt == self.max_retries:
                break
            self.retries += 1
            delay = float(response.headers.get("retry-after") or min(2 ** attempt, 20))
            await asyncio.sleep(delay + random.random())
        latency = (time.perf_counter() - started) * 1000
        if response is None or response.status_code != 200:
            return VisionResult(latency_ms=latency, error=f"HTTP {response.status_code if response else 'none'}: {response.text[:200] if response else ''}")
        raw = response.json()
        usage = raw.get("usageMetadata") or {}
        try:
            text = raw["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return VisionResult(latency_ms=latency, error=f"no candidate: {str(raw)[:200]}", raw=raw)
        return self._result(
            parse_json(text),
            int(usage.get("promptTokenCount") or 0),
            int(usage.get("candidatesTokenCount") or 0),
            latency,
            raw,
        )

    async def inspect_image(self, path: Path, prompt: str) -> VisionResult:
        return await self._generate(
            [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": _b64(path)}},
            ]
        )

    async def inspect_video(self, path: Path, prompt: str) -> dict[str, Any]:
        result = await self._generate(
            [
                {"text": prompt},
                {"inline_data": {"mime_type": "video/mp4", "data": _b64(path)}},
            ]
        )
        frames = parse_json(result.raw.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")) if result.raw else {}
        return {
            "frames": frames.get("frames", []),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "error": result.error,
        }


class DeepSeekVision(_Base):
    def __init__(self, model: str = "deepseek-v4.1-flash", **kwargs):
        super().__init__(model, **kwargs)
        self._headers = {
            "Authorization": f"Bearer {go_key()}",
            "content-type": "application/json",
            "User-Agent": "gridinspect/0.1",
            "x-opencode-session": "gridinspect-eval",
        }

    async def inspect_image(self, path: Path, prompt: str) -> VisionResult:
        self.retries = 0
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": 600,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(path)}},
                    ],
                }
            ],
        }
        started = time.perf_counter()
        response = None
        for attempt in range(self.max_retries + 1):
            response = await self._client.post(f"{go_base()}/chat/completions", headers=self._headers, json=body)
            if response.status_code not in (408, 429, 500, 502, 503, 504) or attempt == self.max_retries:
                break
            self.retries += 1
            delay = float(response.headers.get("retry-after") or min(2 ** attempt, 20))
            await asyncio.sleep(delay + random.random())
        latency = (time.perf_counter() - started) * 1000
        if response is None or response.status_code != 200:
            return VisionResult(latency_ms=latency, error=f"HTTP {response.status_code if response else 'none'}: {response.text[:200] if response else ''}")
        raw = response.json()
        usage = raw.get("usage") or {}
        text = raw["choices"][0]["message"].get("content") or ""
        return self._result(
            parse_json(text),
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            latency,
            raw,
        )
