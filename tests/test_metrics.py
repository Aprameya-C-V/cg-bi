from gridinspect.data import CLASSES, Sample, make_split
from gridinspect.evaluate import summarize
from gridinspect.providers import parse_json
from pathlib import Path


def test_exact_and_micro_metrics():
    rows = [
        {"gold": ["crack"], "pred": ["crack"], "latency_ms": 10, "cost_usd": 0.001, "error": ""},
        {"gold": ["corrosion"], "pred": ["surface_injure"], "latency_ms": 20, "cost_usd": 0.001, "error": ""},
        {"gold": ["crack", "corrosion"], "pred": ["crack"], "latency_ms": 30, "cost_usd": 0.001, "error": ""},
    ]
    s = summarize(rows)
    assert s["n"] == 3
    assert s["exact_set_match"] == 1 / 3
    assert s["critical_recall"] == 1.0
    assert 0 < s["f1"] < 1


def test_parse_json_handles_fences():
    assert parse_json("```json\n{\"a\": 1}\n```")["a"] == 1


def test_split_is_stratified_and_unique():
    samples = []
    for i, cls in enumerate(CLASSES * 5):
        samples.append(Sample(Path(f"{i}.jpg"), [cls], []))
    chosen = make_split(samples, per_class=3, seed=1)
    assert len({s.image for s in chosen}) == len(chosen)
    assert sum(1 for s in chosen if s.classes == ["crack"]) == 3
