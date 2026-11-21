from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .config import DATA

WTBD = DATA / "wtbd" / "WT blade defect dataset"
CLASSES = ["craze", "hide_craze", "corrosion", "surface_injure", "thunderstrike", "crack"]


@dataclass(frozen=True)
class Sample:
    image: Path
    classes: list[str]
    boxes: list[tuple[str, int, int, int, int]]

    @property
    def primary(self) -> str:
        return self.classes[0] if self.classes else "none"


def _parse(name: str) -> Sample:
    root = ET.parse(WTBD / "Annotations" / f"{name}.xml").getroot()
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        boxes.append(
            (
                obj.findtext("name") or "",
                int(box.findtext("xmin")),
                int(box.findtext("ymin")),
                int(box.findtext("xmax")),
                int(box.findtext("ymax")),
            )
        )
    classes = sorted({b[0] for b in boxes})
    return Sample(image=WTBD / "JPEGImages" / f"{name}.jpg", classes=classes, boxes=boxes)


def load_manifest() -> list[Sample]:
    return [_parse(p.stem) for p in sorted((WTBD / "JPEGImages").glob("*.jpg"))]


def make_split(samples: list[Sample], per_class: int = 13, seed: int = 13) -> list[Sample]:
    """Stratified sample by primary class, balanced so every class is represented."""
    rng = random.Random(seed)
    buckets: dict[str, list[Sample]] = {c: [] for c in CLASSES}
    for s in samples:
        for c in s.classes:
            buckets.setdefault(c, []).append(s)
    chosen: list[Sample] = []
    seen = set()
    for cls in CLASSES:
        for s in rng.sample(buckets[cls], min(per_class, len(buckets[cls]))):
            if s.image not in seen:
                seen.add(s.image)
                chosen.append(s)
    return chosen
