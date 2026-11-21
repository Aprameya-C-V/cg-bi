from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "reports" / "figures"
VIDEO = ROOT / "data" / "video"
CRIT = {"crack", "thunderstrike"}
FONT = "/System/Library/Fonts/Helvetica.ttc"


def rows(name: str) -> dict[str, dict]:
    path = ROOT / "runs" / name / "results.jsonl"
    return {json.loads(l)["name"]: json.loads(l) for l in path.read_text().splitlines()}


def font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def panel(name: str, gold: list[str], lite: list[str], argon: list[str], tag: str) -> Image.Image:
    src = ROOT / "data" / "wtbd" / "WT blade defect dataset" / "JPEGImages" / f"{name}.jpg"
    im = Image.open(src).convert("RGB").resize((520, 520))
    canvas = Image.new("RGB", (520, 660), "#f3f1eb")
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((16, 536), tag, font=font(20), fill="#1e4b3d")
    d.text((16, 566), f"annotated: {', '.join(gold) or 'none'}", font=font(15), fill="#14140f")
    d.text((16, 590), f"3.5 Lite: {', '.join(lite) or 'none'}", font=font(15), fill="#6b675c")
    d.text((16, 614), f"4 Argon: {', '.join(argon) or 'none'}", font=font(15), fill="#6b675c")
    return canvas


def make_examples() -> str:
    lite = rows("gemini-3.5-flash-lite")
    argon = rows("gemini-4-argon")
    names = sorted(lite)
    escalation = next(
        n for n in names
        if set(lite[n]["gold"]) & CRIT and not set(lite[n]["pred"]) & CRIT and set(argon[n]["pred"]) & CRIT
    )
    correct = next(n for n in names if set(lite[n]["gold"]) == set(lite[n]["pred"]))
    arg_correct = next(
        (n for n in names if n not in (escalation, correct)
         and set(argon[n]["gold"]) == set(argon[n]["pred"]) and set(argon[n]["gold"]) & CRIT),
        next(n for n in names if n not in (escalation, correct) and set(argon[n]["gold"]) == set(argon[n]["pred"])),
    )
    hard = next(
        (n for n in names if set(lite[n]["gold"]) != set(lite[n]["pred"])
         and set(argon[n]["gold"]) != set(argon[n]["pred"]) and n not in (escalation, correct, arg_correct)),
        names[-1],
    )

    picks = [
        (correct, "Correct on the bulk tier"),
        (escalation, "Missed by the bulk tier, caught after escalation"),
        (arg_correct, "Critical class named correctly by the strong tier"),
        (hard, "Wrong taxonomy on both tiers"),
    ]
    sheet = Image.new("RGB", (1040, 1320), "#f3f1eb")
    for i, (name, tag) in enumerate(picks):
        p = panel(name, lite[name]["gold"], lite[name]["pred"], argon[name]["pred"], tag)
        sheet.paste(p, ((i % 2) * 520, (i // 2) * 660))
    out = FIGS / "examples.png"
    sheet.save(out)
    return out.name


def make_video_strip() -> str:
    frame_dir = VIDEO / "frames"
    frame_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(VIDEO / "sweep.mp4"), "-vf", "fps=1", str(frame_dir / "f_%02d.png")],
        check=True,
    )
    labels = json.loads((VIDEO / "labels.json").read_text())
    result = json.loads((VIDEO / "result.json").read_text())
    preds = {f["second"]: [c.get("label", c) if isinstance(c, dict) else c for c in f.get("defects", [])] for f in result.get("frames", [])}

    cell_w, cell_h = 260, 300
    sheet = Image.new("RGB", (cell_w * 4, cell_h * 2), "#f3f1eb")
    d = ImageDraw.Draw(sheet)
    for i, item in enumerate(labels):
        frame = Image.open(frame_dir / f"f_{i+1:02d}.png").convert("RGB").resize((cell_w, cell_w))
        sheet.paste(frame, ((i % 4) * cell_w, (i // 4) * cell_h))
        gold = ", ".join(item["classes"])[:26]
        pred = ", ".join(preds.get(item["second"], []))[:26]
        d.text(((i % 4) * cell_w + 8, (i // 4) * cell_h + cell_w + 8), f"{i}s · {gold}", font=font(11), fill="#14140f")
        d.text(((i % 4) * cell_w + 8, (i // 4) * cell_h + cell_w + 26), f"AI: {pred or 'none'}", font=font(11), fill="#6b675c")
    out = FIGS / "video-strip.png"
    sheet.save(out)
    return out.name


if __name__ == "__main__":
    print(make_examples())
    print(make_video_strip())
