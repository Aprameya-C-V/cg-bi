from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
VIDEO = ROOT / "data" / "video"

PAPER, INK, MUTED, RULE, ACCENT, FILL = "#f3f1eb", "#14140f", "#6b675c", "#d8d3c6", "#1e4b3d", "#c9c3b4"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": RULE, "font.family": "Helvetica Neue", "font.size": 9,
})

CLASSES = ["craze", "hide_craze", "corrosion", "surface_injure", "thunderstrike", "crack"]


def load(name: str) -> dict:
    return json.loads((RUNS / name / "summary.json").read_text())


def fig_per_class(gem: dict, ds: dict) -> str:
    x = np.arange(len(CLASSES))
    w = 0.38
    g = [gem["per_class"][c]["recall"] for c in CLASSES]
    d = [ds["per_class"][c]["recall"] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(x - w / 2, g, w, color=ACCENT, label="Gemini 3.5 Flash Lite")
    ax.bar(x + w / 2, d, w, color=FILL, label="DeepSeek V4.1 Flash")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15)
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIGS / "per-class-recall.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def fig_cost(gem: dict, ds: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    names = ["Gemini 3.5\nFlash Lite", "DeepSeek\nV4.1 Flash"]
    vals = [
        ([gem["f1"], ds["f1"]], "class-set F1"),
        ([gem["critical_recall"], ds["critical_recall"]], "critical recall"),
        ([gem["cost_per_1000"], ds["cost_per_1000"]], "cost per 1,000 (USD)"),
    ]
    for ax, (data, title) in zip(axes, vals):
        ax.bar(names, data, color=[ACCENT, FILL], width=0.55)
        ax.set_title(title, fontsize=9, color=MUTED)
        for i, v in enumerate(data):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIGS / "model-compare.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def video_metrics() -> dict:
    labels = json.loads((VIDEO / "labels.json").read_text())
    result = json.loads((VIDEO / "result.json").read_text())
    frames = {f["second"]: [c for c in f.get("defects", []) if c in CLASSES] for f in result.get("frames", [])}
    rows = []
    for item in labels:
        gold = item["classes"]
        pred = frames.get(item["second"], [])
        rows.append({"second": item["second"], "gold": gold, "pred": pred})
    exact = sum(1 for r in rows if set(r["gold"]) == set(r["pred"])) / len(rows)
    crit = [r for r in rows if set(r["gold"]) & {"crack", "thunderstrike"}]
    crit_hit = sum(1 for r in crit if set(r["pred"]) & {"crack", "thunderstrike"})
    return {
        "frames": len(rows),
        "exact": exact,
        "critical_n": len(crit),
        "critical_hit": crit_hit,
        "rows": rows,
        "tokens": result.get("prompt_tokens", 0),
        "cost": result.get("cost_usd", 0.0),
    }


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    gem = load("gemini-3.5-flash-lite")
    ds = load("deepseek-v4.1-flash")
    pc = fig_per_class(gem, ds)
    mc = fig_cost(gem, ds)
    vid = video_metrics()

    def pct(v): return f"{v*100:.1f}%"
    def money(v): return f"${v:.2f}"

    video_rows = "".join(
        f"<tr><td>{r['second']}</td><td>{', '.join(r['gold'])}</td><td>{', '.join(r['pred']) or 'none'}</td></tr>"
        for r in vid["rows"]
    )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blade Inspection Copilot</title>
<style>
:root{{--paper:{PAPER};--ink:{INK};--muted:{MUTED};--rule:{RULE};--accent:{ACCENT}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue","Segoe UI",system-ui,sans-serif;font-size:16px;line-height:1.65}}
.bar{{height:6px;background:var(--accent)}}
main{{max-width:980px;margin:0 auto;padding:56px 24px 96px}}
.eyebrow{{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:0 0 12px}}
h1{{font-size:42px;letter-spacing:-.03em;margin:0 0 10px;font-weight:600}}
h2{{font-size:22px;margin:52px 0 12px;font-weight:600;letter-spacing:-.01em}}
h3{{font-size:13px;margin:26px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:500}}
p{{margin:10px 0}}
.lede{{font-size:18px;color:var(--muted);max-width:72ch}}
table{{width:100%;border-collapse:collapse;margin:14px 0;font-size:14px}}
th{{text-align:left;font-weight:500;color:var(--muted);border-bottom:1px solid var(--rule);padding:8px 10px;font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:.08em;text-transform:uppercase}}
td{{border-bottom:1px solid var(--rule);padding:9px 10px;font-variant-numeric:tabular-nums}}
.note{{color:var(--muted);font-size:14px}}
figure{{margin:18px 0 26px}} img{{max-width:100%;border:1px solid var(--rule);border-radius:6px;display:block}}
figcaption{{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--muted);margin-top:6px}}
.calc{{border:1px solid var(--rule);border-radius:6px;padding:20px;background:#fbfaf6;margin:18px 0}}
.calc label{{display:flex;justify-content:space-between;gap:12px;font-size:14px;margin:8px 0}}
.calc input{{width:120px;padding:6px 8px;border:1px solid var(--rule);border-radius:4px;background:var(--paper);font:inherit;font-size:14px;text-align:right}}
.out{{margin-top:14px;padding-top:14px;border-top:1px solid var(--rule);font-size:15px}}
.out b{{color:var(--accent)}}
ul{{margin:10px 0 10px 18px;padding:0}} li{{margin:6px 0}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:13px;background:#e9e4d8;padding:1px 5px;border-radius:3px}}
</style></head><body><div class="bar"></div><main>
<p class="eyebrow">Industry study · Renewable energy O&amp;M · November 2026</p>
<h1>Blade Inspection Copilot</h1>
<p class="lede">Drone photos of wind turbine blades, triaged by a multimodal model that names the defect, grades severity, and drafts the technician note. Measured on 75 real inspection images from a public drone dataset, with Gemini 3.5 Flash Lite against DeepSeek V4.1 Flash.</p>

<h2>The business problem</h2>
<p>Blade inspection is a documentation bottleneck, not a photography problem. A drone captures hundreds of images per turbine, and an engineer still writes a finding for each one: defect type, severity, location, and an action. The taxonomy is fixed across the fleet, so inconsistent wording and slow write-ups cost real hours and make trend analysis unreliable.</p>
<p>The copilot does not replace the engineer. It produces a first-pass finding per image in about two seconds, in one taxonomy, and a human approves or corrects it.</p>

<h2>Results</h2>
<table><thead><tr><th>model</th><th>images</th><th>exact set match</th><th>class-set F1</th><th>critical recall</th><th>cost / 1,000</th><th>p50</th><th>p95</th></tr></thead><tbody>
<tr><td>Gemini 3.5 Flash Lite</td><td>{gem['n']}</td><td>{pct(gem['exact_set_match'])}</td><td>{gem['f1']:.3f}</td><td>{pct(gem['critical_recall'])}</td><td>{money(gem['cost_per_1000'])}</td><td>{gem['latency_p50_ms']/1000:.2f}s</td><td>{gem['latency_p95_ms']/1000:.1f}s</td></tr>
<tr><td>DeepSeek V4.1 Flash</td><td>{ds['n']}</td><td>{pct(ds['exact_set_match'])}</td><td>{ds['f1']:.3f}</td><td>{pct(ds['critical_recall'])}</td><td>{money(ds['cost_per_1000'])}</td><td>{ds['latency_p50_ms']/1000:.2f}s</td><td>{ds['latency_p95_ms']/1000:.1f}s</td></tr>
</tbody></table>
<p class="note">Critical recall counts crack and thunderstrike together: the two classes where a miss has a maintenance consequence. Exact set match requires the model to name exactly the annotated class set.</p>
<figure><img src="figures/{mc}" alt="model comparison"><figcaption>F1, critical-defect recall and cost per 1,000 images</figcaption></figure>
<figure><img src="figures/{pc}" alt="per class recall"><figcaption>Per-class recall. Corrosion and surface damage are the readable classes; the fine-crack classes stay hard.</figcaption></figure>

<h2>What the numbers mean</h2>
<ul>
<li><b>Gemini is usable as a drafting copilot, not as a gate.</b> It catches about seven in ten critical defects and names the right class set for one image in five. That is enough to draft a finding for an engineer to approve, not enough to clear an image without review.</li>
<li><b>DeepSeek is not viable for this task at similar cost.</b> Class-set F1 0.17 and critical recall 8 percent. Its image path is cheap but the visual grounding is far behind Gemini on fine surface defects.</li>
<li><b>Cost is not the differentiator.</b> Both land near $0.40 to $0.46 per 1,000 images, which is trivial next to the labor it is meant to save. Model choice is a quality decision.</li>
</ul>

<h2>Video sweep</h2>
<p>Gemini 3.5 Flash Lite accepts video directly, so the same workflow runs on a short drone sweep. An eight-second clip built from eight inspection frames, one frame per second, produced a per-second condition log: {vid['critical_hit']} of {vid['critical_n']} critical frames flagged, exact class set on {pct(vid['exact'])} of frames, {vid['tokens']} prompt tokens.</p>
<table><thead><tr><th>second</th><th>annotated</th><th>copilot</th></tr></thead><tbody>{video_rows}</tbody></table>
<p class="note">Eight frames is a demonstration, not a benchmark. It shows the video path works and that the model tracks defects across a sweep.</p>

<h2>Business case</h2>
<p>The model numbers above are measured. The labor numbers below are inputs you can change.</p>
<div class="calc">
<label>Inspection images per month <input id="img" type="number" value="2000"></label>
<label>Manual minutes per finding <input id="manual" type="number" value="4"></label>
<label>Copilot review seconds per image <input id="review" type="number" value="45"></label>
<label>Loaded engineer cost per hour (USD) <input id="rate" type="number" value="18"></label>
<div class="out" id="out"></div>
</div>
<p class="note">Model cost uses the measured ${gem['cost_per_1000']:.2f} per 1,000 images. Review time assumes the engineer checks the draft rather than writing from scratch.</p>

<h2>What is strong, what is weak</h2>
<p><b>Strong.</b> Real drone imagery (CC BY 4.0), a fixed taxonomy, two model families measured on the same images, cost and latency per 1,000, a working video path, and a business case whose inputs are visible.</p>
<p><b>Weak.</b> No healthy images in the source dataset, so specificity and false-clear rate cannot be reported. The dataset is annotated by one source, and craze versus hide_craze is a genuinely ambiguous boundary. 75 images and one seed, so the intervals are wide. p95 includes gateway retries. The copilot drafts; it does not clear.</p>

<h2>Reproduce</h2>
<pre class="note">uv run python scripts/run_eval.py --model both --per-class 13
uv run python scripts/build_video.py
uv run python scripts/report.py</pre>
<p class="note">Data: Wind Turbine Blade Defect dataset, DOI 10.6084/m9.figshare.30210175, CC BY 4.0.</p>
</main>
<script>
function calc(){{
  const img=+document.getElementById('img').value||0;
  const manual=+document.getElementById('manual').value||0;
  const review=+document.getElementById('review').value||0;
  const rate=+document.getElementById('rate').value||0;
  const manualH=img*manual/60, coopH=img*review/3600;
  const saved=manualH-coopH;
  const money=saved*rate;
  const model=img/1000*{gem['cost_per_1000']:.4f};
  document.getElementById('out').innerHTML =
    `<b>${{saved.toFixed(0)}} engineer-hours saved per month</b> (${{manualH.toFixed(0)}}h manual vs ${{coopH.toFixed(0)}}h reviewing drafts), worth about ${{money.toFixed(0)}} at the loaded rate. Model cost about ${{model.toFixed(2)}} per month.`;
}}
['img','manual','review','rate'].forEach(id=>document.getElementById(id).addEventListener('input',calc));calc();
</script></body></html>"""
    (REPORTS / "report.html").write_text(html)
    (REPORTS / "index.html").write_text(html)
    thumbnail()
    print("wrote report.html and thumbnail.png")




def thumbnail() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.3))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.3); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 12, 6.3, color=PAPER))
    ax.add_patch(plt.Rectangle((0, 6.1), 12, 0.2, color=ACCENT))
    ax.text(0.7, 4.9, "Blade Inspection Copilot", fontsize=30, color=INK, weight="bold")
    ax.text(0.7, 4.35, "Multimodal triage and report drafting for wind turbine blade inspections", fontsize=12, color=MUTED)
    stats = [("69%", "critical defect recall"), ("$0.46", "per 1,000 images"), ("2.0s", "draft per image")]
    for i, (v, l) in enumerate(stats):
        x = 0.7 + i * 3.8
        ax.text(x, 2.5, v, fontsize=24, color=ACCENT, weight="bold")
        ax.text(x, 2.1, l, fontsize=10, color=MUTED)
    ax.text(0.7, 0.8, "Gemini 3.5 Flash Lite vs DeepSeek V4.1 Flash · 75 real drone images · CC BY 4.0 data", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(ROOT / "thumbnail.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
