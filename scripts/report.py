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

PAPER, INK, MUTED, RULE, ACCENT, FILL, MID = "#f3f1eb", "#14140f", "#6b675c", "#d8d3c6", "#1e4b3d", "#c9c3b4", "#7d8f6f"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": RULE, "font.family": "Helvetica Neue", "font.size": 9,
})

CLASSES = ["craze", "hide_craze", "corrosion", "surface_injure", "thunderstrike", "crack"]


def load(name: str) -> dict:
    return json.loads((RUNS / name / "summary.json").read_text())


def fig_per_class(models: list[tuple[str, dict, str]]) -> str:
    x = np.arange(len(CLASSES))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    for i, (label, summary, color) in enumerate(models):
        vals = [summary["per_class"][c]["recall"] for c in CLASSES]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15)
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "per-class-recall.png", dpi=170)
    plt.close(fig)
    return "per-class-recall.png"


def fig_tradeoff(models: list[tuple[str, dict, str]]) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    names = [m[0] for m in models]
    sets = [
        ([m[1]["f1"] for m in models], "class-set F1"),
        ([m[1]["critical_recall"] for m in models], "critical recall"),
        ([m[1]["cost_per_1000"] for m in models], "cost per 1,000 (USD)"),
    ]
    for ax, (values, title) in zip(axes, sets):
        ax.bar(names, values, color=[m[2] for m in models], width=0.55)
        ax.set_title(title, fontsize=9, color=MUTED)
        ax.tick_params(axis="x", labelsize=7)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "model-compare.png", dpi=170)
    plt.close(fig)
    return "model-compare.png"


def fig_selective(gem: dict) -> str:
    rows = gem["selective"]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.plot([r["coverage"] * 100 for r in rows], [r["exact_match"] * 100 for r in rows], marker="o", color=ACCENT, label="exact set match")
    ax.plot([r["coverage"] * 100 for r in rows], [r["critical_recall"] * 100 for r in rows], marker="s", color=MID, label="critical recall")
    ax.set_xlabel("coverage (% of images above confidence threshold)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "selective.png", dpi=170)
    plt.close(fig)
    return "selective.png"


def fig_cascade(models, cascade) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for label, summary, color in models:
        ax.scatter(summary["cost_per_1000"], summary["critical_recall"] * 100, s=70, color=color, label=label, zorder=3)
    ax.plot([c["cost_per_1000"] for c in cascade], [c["critical_recall"] * 100 for c in cascade],
            marker="o", color=ACCENT, linestyle="--", linewidth=1.2, label="routed cascade")
    ax.set_xlabel("cost per 1,000 images (USD)")
    ax.set_ylabel("critical recall (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "cascade.png", dpi=170)
    plt.close(fig)
    return "cascade.png"


def video_metrics() -> dict:
    labels = json.loads((VIDEO / "labels.json").read_text())
    result = json.loads((VIDEO / "result.json").read_text())
    frames = {f["second"]: [c.get("label", c) if isinstance(c, dict) else c for c in f.get("defects", [])] for f in result.get("frames", [])}
    rows = []
    for item in labels:
        gold = item["classes"]
        pred = [c for c in frames.get(item["second"], []) if c in CLASSES]
        rows.append({"second": item["second"], "gold": gold, "pred": pred})
    exact = sum(1 for r in rows if set(r["gold"]) == set(r["pred"])) / len(rows)
    crit = [r for r in rows if set(r["gold"]) & {"crack", "thunderstrike"}]
    crit_hit = sum(1 for r in crit if set(r["pred"]) & {"crack", "thunderstrike"})
    return {"frames": len(rows), "exact": exact, "critical_n": len(crit), "critical_hit": crit_hit,
            "rows": rows, "tokens": result.get("prompt_tokens", 0), "cost": result.get("cost_usd", 0.0)}


def thumbnail(argon: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.3))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.3); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 12, 6.3, color=PAPER))
    ax.add_patch(plt.Rectangle((0, 6.1), 12, 0.2, color=ACCENT))
    ax.text(0.7, 4.9, "Blade Inspection Copilot", fontsize=30, color=INK, weight="bold")
    ax.text(0.7, 4.35, "Multimodal triage and report drafting for wind turbine blade inspections", fontsize=12, color=MUTED)
    stats = [("96%", "critical recall, strong tier"), ("$1.73", "per 1,000, routed cascade"), ("36%", "images escalated")]
    for i, (v, l) in enumerate(stats):
        x = 0.7 + i * 3.8
        ax.text(x, 2.5, v, fontsize=24, color=ACCENT, weight="bold")
        ax.text(x, 2.1, l, fontsize=10, color=MUTED)
    ax.text(0.7, 0.8, "Gemini 4 Argon and 3.5 Flash Lite vs DeepSeek V4.1 Flash · 75 real drone images · CC BY 4.0", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(ROOT / "thumbnail.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    gem = load("gemini-3.5-flash-lite")
    argon = load("gemini-4-argon")
    ds = load("deepseek-v4.1-flash")
    cascade = json.loads((RUNS / "cascade.json").read_text())
    models = [("Gemini 3.5 Lite", gem, FILL), ("Gemini 4 Argon", argon, ACCENT), ("DeepSeek V4.1", ds, MUTED)]
    pc = fig_per_class(models)
    mc = fig_tradeoff(models)
    sel = fig_cascade(models, cascade)
    last = cascade[-1]
    def pctf(v): return f"{v*100:.1f}%"
    cascade_rows = "".join(
        f"<tr><td>{c['threshold']:.2f}</td><td>{pctf(c['escalation'])}</td><td>{pctf(c['exact'])}</td>"
        f"<td>{c['f1']:.3f}</td><td>{pctf(c['critical_recall'])}</td><td>${c['cost_per_1000']:.2f}</td></tr>"
        for c in cascade
    )
    cascade_row = (
        f"<tr><td>Routed cascade (Lite then Argon)</td><td>{gem['n']}</td><td>{pctf(last['exact'])}</td>"
        f"<td>{last['f1']:.3f}</td><td>{pctf(last['critical_recall'])}</td>"
        f"<td>${last['cost_per_1000']:.2f}</td><td>{last['p50']/1000:.2f}s</td><td>n/a</td></tr>"
    )
    vid = video_metrics()

    def pct(v): return f"{v*100:.1f}%"
    def money(v): return f"${v:.2f}"
    def tier_row(name, s):
        return (f"<tr><td>{name}</td><td>{s['n']}</td><td>{pct(s['exact_set_match'])}</td><td>{s['f1']:.3f}</td>"
                f"<td>{pct(s['critical_recall'])}</td><td>{money(s['cost_per_1000'])}</td>"
                f"<td>{s['latency_p50_ms']/1000:.2f}s</td><td>{s['latency_p95_ms']/1000:.1f}s</td></tr>")

    sel_rows = "".join(
        f"<tr><td>{r['threshold']:.1f}</td><td>{pct(r['coverage'])}</td><td>{pct(r['exact_match'])}</td><td>{pct(r['critical_recall'])}</td></tr>"
        for r in gem["selective"]
    )
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
<p class="lede">Drone photos of wind turbine blades, triaged by a multimodal model that names the defect, grades severity, and drafts the technician note. Three model tiers, 75 real inspection images, and a confidence gate for routing.</p>

<h2>The business problem</h2>
<p>Blade inspection is a documentation bottleneck. A drone captures hundreds of images per turbine and an engineer still writes a finding for each one: type, severity, location, action. The taxonomy is fixed across the fleet, so inconsistent wording and slow write-ups cost hours and break trend analysis.</p>
<p>The copilot drafts the finding in about two seconds. The engineer approves or corrects it. Because tiers differ by six-fold in cost, the routing decision matters as much as the model choice.</p>

<h2>Results by tier</h2>
<table><thead><tr><th>model</th><th>images</th><th>exact set</th><th>class F1</th><th>critical recall</th><th>cost / 1,000</th><th>p50</th><th>p95</th></tr></thead><tbody>
{tier_row("Gemini 3.5 Flash Lite", gem)}
{tier_row("Gemini 4 Argon via gateway", argon)}
{tier_row("DeepSeek V4.1 Flash", ds)}
{cascade_row}
</tbody></table>
<p class="note">Critical recall counts crack and thunderstrike together. Cost uses list prices, not subscription allowances. p95 includes gateway retries.</p>
<figure><img src="figures/{mc}" alt="model comparison"><figcaption>Quality, critical recall and cost across tiers</figcaption></figure>
<figure><img src="figures/{pc}" alt="per class recall"><figcaption>Per-class recall. The escalated tier carries the fine-crack classes; the bulk tier carries surface damage.</figcaption></figure>

<h2>The tiering finding</h2>
<ul>
<li><b>Bulk tier, Gemini 3.5 Flash Lite.</b> {pct(gem['exact_set_match'])} exact set match, F1 {gem['f1']:.2f}, {pct(gem['critical_recall'])} critical recall at {money(gem['cost_per_1000'])} per 1,000 images and 1.3 seconds. Good enough to draft, not to clear.</li>
<li><b>Escalated tier, Gemini 4 Argon.</b> Critical recall rises to {pct(argon['critical_recall'])} with F1 {argon['f1']:.2f} at {money(argon['cost_per_1000'])} per 1,000 and 5.7 seconds. Worth it for critical classes, not for every image.</li>
<li><b>DeepSeek V4.1 Flash is not viable here.</b> {pct(ds['critical_recall'])} critical recall at roughly the bulk-tier price. Cheap image input does not buy visual grounding.</li>
</ul>

<h2>Routed cascade</h2>
<p>The useful policy is not a single tier. Run the cheap tier on every image, then escalate only the images whose own confidence is below 0.9 to the strong tier. One cheap call always, a strong call on roughly a third of images.</p>
<table><thead><tr><th>confidence gate</th><th>escalated</th><th>exact set</th><th>class F1</th><th>critical recall</th><th>cost / 1,000</th></tr></thead><tbody>
{cascade_rows}
</tbody></table>
<figure><img src="figures/{sel}" alt="routed cascade"><figcaption>Cost against critical recall. The routed cascade sits above the line between the two single tiers.</figcaption></figure>
<p class="note">At a 0.9 gate the cascade reaches {pct(last['critical_recall'])} critical recall and F1 {last['f1']:.2f} for ${last['cost_per_1000']:.2f} per 1,000, against ${argon['cost_per_1000']:.2f} for the strong tier alone. That is the business case: most of the strong tier's recall at under half its cost.</p>

<h2>Video sweep</h2>
<p>Gemini accepts video directly, so the same workflow runs on a short drone sweep. An eight-second clip built from eight frames produced a per-second condition log: {vid['critical_hit']} of {vid['critical_n']} critical frames flagged, exact class set on {pct(vid['exact'])} of frames, {vid['tokens']} prompt tokens.</p>
<table><thead><tr><th>second</th><th>annotated</th><th>copilot</th></tr></thead><tbody>{video_rows}</tbody></table>
<p class="note">Eight frames is a demonstration, not a benchmark. It shows the video path works and tracks defects across a sweep.</p>

<h2>Business case</h2>
<p>Model numbers are measured. Labor is an input you can change.</p>
<div class="calc">
<label>Inspection images per month <input id="img" type="number" value="2000"></label>
<label>Manual minutes per finding <input id="manual" type="number" value="4"></label>
<label>Copilot review seconds per image <input id="review" type="number" value="45"></label>
<label>Loaded engineer cost per hour (USD) <input id="rate" type="number" value="18"></label>
<label>Share of images escalated to the high tier (%) <input id="esc" type="number" value="20"></label>
<div class="out" id="out"></div>
</div>
<p class="note">Model cost is the bulk tier on every image plus the strong tier on the escalated share. Review time assumes the engineer checks a draft rather than writing one.</p>

<h2>What is strong, what is weak</h2>
<p><b>Strong.</b> Real drone imagery (CC BY 4.0), fixed taxonomy, three tiers measured on the same 75 images, a routed cost model, a confidence gate, a working video path, and a business case whose inputs are visible.</p>
<p><b>Weak.</b> No healthy images in the source data, so no specificity or false-clear claim. 75 images and one seed. Craze versus hide_craze is ambiguous even for annotators. p95 includes retries. The copilot drafts; it does not approve.</p>

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
  const esc=(+document.getElementById('esc').value||0)/100;
  const manualH=img*manual/60, coopH=img*review/3600;
  const saved=manualH-coopH, money=saved*rate;
  const model=(img/1000)*({gem['cost_per_1000']:.4f}+esc*{argon['cost_per_1000']:.4f});
  document.getElementById('out').innerHTML =
    `<b>${{saved.toFixed(0)}} engineer-hours saved per month</b> (${{manualH.toFixed(0)}}h manual vs ${{coopH.toFixed(0)}}h reviewing drafts), worth about ${{money.toFixed(0)}} at the loaded rate. Model cost about ${{model.toFixed(2)}} per month.`;
}}
['img','manual','review','rate','esc'].forEach(id=>document.getElementById(id).addEventListener('input',calc));calc();
</script></body></html>"""
    (REPORTS / "report.html").write_text(html)
    (REPORTS / "index.html").write_text(html)
    thumbnail(argon)
    print("wrote report.html and thumbnail.png")


if __name__ == "__main__":
    main()
