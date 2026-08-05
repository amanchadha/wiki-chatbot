"""Metric-progression plots for RATIONALE.md, built from eval/runs/ artifacts.

Usage:
    python eval/make_plots.py     # writes eval/plots/*.png

Run numbering matches the table in RATIONALE.md section 3. Values are read
from each run's summary.json / case files; nothing is hand-entered. Runs
9-10 are judge-only replays of runs 7-8 (identical agent transcripts), so
behavior metrics are plotted once at runs 7-8 and the side-claims series
(introduced by judge-v4) is plotted at runs 9-11.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(__file__).resolve().parent / "runs"
OUT = Path(__file__).resolve().parent / "plots"
OUT.mkdir(exist_ok=True)

# RATIONALE section-3 run number -> run directory
RUN_DIRS = {
    1: "20260719T173828Z_v0-baseline",
    2: "20260719T191804Z_v0-baseline",
    3: "20260719T195322Z_v1-verify-volatile",
    4: "20260719T200819Z_v1.1-outage-fallback",
    5: "20260719T205230Z_v1.2-snippets-followup",
    6: "20260719T211816Z_v1.2-snippets-followup",
    7: "20260719T213752Z_v1.3-chain-verify-infobox",
    8: "20260719T214752Z_v1.3-chain-verify-infobox",
    9: "20260720T222422Z_v1.3-chain-verify-infobox_rejudge",
    10: "20260720T222548Z_v1.3-chain-verify-infobox_rejudge",
    11: "20260720T231040Z_v1.4-citations_rejudge",
}
BEHAVIOR_RUNS = [1, 2, 3, 4, 5, 6, 7, 8, 11]  # 9-10 replay 7-8's transcripts

# Colors: dataviz reference palette, categorical slots 1-2 (fixed order)
C1, C2 = "#2a78d6", "#eb6834"
GRID = dict(color="#00000022", linewidth=0.8)


def load(run_no):
    d = RUNS / RUN_DIRS[run_no]
    summary = json.loads((d / "summary.json").read_text())
    cases = [json.loads(p.read_text()) for p in sorted((d / "cases").glob("*.json"))]
    return summary, [c for c in cases if "error" not in c]


def tool_success_rate(cases):
    """Uniform across eras: error results are 'SEARCH ERROR:'-prefixed (new
    tool) or carry an HTTP 429 marker (old tool)."""
    calls = errs = 0
    for c in cases:
        for tr in c.get("tool_results", []):
            calls += 1
            if tr.startswith("SEARCH ERROR:") or "429" in tr[:200]:
                errs += 1
    return (calls - errs) / calls if calls else None


series = {k: {} for k in (
    "recall", "precision", "pass", "sufficiency", "article", "tool_ok",
    "faithful", "side_ok",
)}

for n in RUN_DIRS:
    s, cases = load(n)
    if n in BEHAVIOR_RUNS:
        series["recall"][n] = s.get("search_recall")
        series["precision"][n] = s.get("search_precision")
        series["pass"][n] = s.get("pass_rate")
        series["sufficiency"][n] = s.get("evidence_sufficiency_rate")
        series["article"][n] = s.get("right_article_rate")
        series["tool_ok"][n] = tool_success_rate(cases)
        nsearched = s.get("n_searched_cases") or 0
        if nsearched:
            series["faithful"][n] = 1 - len(s.get("unfaithful_ids") or []) / nsearched
    flagged = s.get("answers_with_unsupported_side_claims")
    if flagged:  # "1/22" style; judge-v4+ only
        a, b = flagged.split("/")
        series["side_ok"][n] = 1 - int(a) / int(b)


def panel(ax, title, keys, labels=None, annotate=()):
    labels = labels or [None] * len(keys)
    for key, label, color in zip(keys, labels, (C1, C2)):
        pts = sorted(series[key].items())
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                label=label)
        ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, -3), fontsize=8, color=color)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylim(0, 1.06)
    ax.set_xlim(0.5, 11.5)
    ax.set_xticks(range(1, 12))
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", **GRID)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for x, text in annotate:
        ax.axvline(x, color="#00000033", linewidth=0.8, linestyle=":")
        ax.annotate(text, (x, 0.06), fontsize=7, color="#555555", rotation=90,
                    ha="right", va="bottom")
    if any(labels):
        ax.legend(fontsize=8, frameon=False, loc="lower right")


fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), dpi=200)
notes = [(3, "429 outage"), (6, "multi-hop added")]

panel(axes[0][0], "Search decision (recall vs precision)",
      ["recall", "precision"], ["recall", "precision"], notes)
panel(axes[0][1], "Evidence sufficiency (searched cases)", ["sufficiency"],
      annotate=notes)
panel(axes[0][2], "Right-article rate (searched cases)", ["article"],
      annotate=notes)
panel(axes[1][0], "Tool-call success rate", ["tool_ok"], annotate=[(3, "429 outage")])
panel(axes[1][1], "Groundedness", ["faithful", "side_ok"],
      ["faithful answers", "no unsupported side claims"],
      [(9, "judge-v4 rubric added")])
axes[1][1].legend(fontsize=8, frameon=False, loc="lower left")
panel(axes[1][2], "Pass rate", ["pass"], annotate=notes)

for ax in axes[1]:
    ax.set_xlabel("Run # (section 3 table)", fontsize=8)

fig.suptitle("Metric progression across eval runs (from eval/runs/ artifacts)",
             fontsize=12, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = OUT / "metrics_progression.png"
fig.savefig(out, facecolor="white")
print(f"wrote {out}")
