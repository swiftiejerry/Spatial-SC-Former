"""Regenerate Figure 12, the report's central result figure.

The other eleven report figures come from report_figures.py, method_diagram.py,
marker_expression_field.py and formal_c_partitions.py. This one was written for
the report and is kept here so the figure is reproducible from the repository.

Three panels:
  a  paired differences B0 - A* for the three structural metrics; open circles
     are individual seeds, the filled diamond is the mean, the dashed line is
     no change.
  b  absolute ARI by arm, showing the true spatial graph against a
     degree-preserving random rewiring of itself.
  c  wiring evidence: spatial message norms and relation-message gradient norms
     are non-zero at every layer, so the degradation is not a missing edge.

Reads experiments/reports/advanced_blueprint_P2/strict_report.json.
Writes a vector PDF next to this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments" / "reports" / "advanced_blueprint_P2" / "strict_report.json"
OUT = ROOT / "report" / "source" / "fig12_p2_core_result.pdf"

d = json.loads(SRC.read_text(encoding="utf-8"))
paired = d["paired"]
runs = d["runs"]

INK = "#272727"
RED = "#C0392B"
BLUE = "#2E6DA4"
GREY = "#9A9A9A"
GREEN = "#3C8C5A"

plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
for k in ("axes.edgecolor", "axes.labelcolor", "xtick.color", "ytick.color", "text.color"):
    plt.rcParams[k] = INK

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

# --- panel a: paired differences -----------------------------------------
a = ax[0]
metrics = [("ARI", "ari", 0.0), ("NMI", "nmi", 0.72), ("boundary-F1", "boundary_f1", 1.44)]
for name, key, off in metrics:
    vals = paired[key]["values"]
    mean, sd = paired[key]["mean"], paired[key]["sd"]
    a.scatter([off] * 3, vals, s=70, facecolor="white", edgecolor=BLUE, linewidth=1.8,
              zorder=3, label="individual seed" if key == "ari" else None)
    a.errorbar([off], [mean], yerr=[sd], fmt="D", color=RED, markersize=8, capsize=5,
               capthick=1.6, elinewidth=1.6, zorder=4, label="mean ± s.d." if key == "ari" else None)
a.axhline(0, color=GREY, lw=1.4, ls="--", zorder=1)
a.text(0.62, 0.145, "no change", color=GREY, fontsize=9, va="bottom")
a.set_xticks([o for _, _, o in metrics])
a.set_xticklabels([n for n, _, _ in metrics], fontsize=10.5)
a.set_ylabel("paired difference  (B0 − A*)", fontsize=10.5)
a.set_title("a   Spatial edges degrade all three metrics", fontsize=11.5, loc="left", pad=9)
a.legend(fontsize=9, frameon=False, loc="lower left")
a.set_ylim(-0.30, 0.20)

# --- panel b: absolute ARI, true graph vs rewiring -------------------------
b = ax[1]
groups = [
    ("A*\n(no spatial)", [r["Astar"]["ari"] for r in runs], BLUE, -0.26, "o"),
    ("B0\n(true spatial)", [r["B0"]["ari"] for r in runs], RED, 0.0, "o"),
    ("degree-preserving\nrandom rewiring",
     [r["nulls"]["degree_rewire"]["ari"] for r in runs], GREEN, 0.26, "s"),
]
for name, vals, colour, off, marker in groups:
    b.scatter([i + off for i in range(3)], vals, s=78, facecolor="white", edgecolor=colour,
              linewidth=1.9, marker=marker, zorder=3)
    b.hlines(np.mean(vals), off - 0.14, off + 0.14, color=colour, lw=2.6, zorder=4)
for i, r in enumerate(runs):
    b.text(i, -0.028, "seed %s" % r["seed"], ha="center", fontsize=8.5, color=GREY)
b.set_xticks(range(3))
b.set_xticklabels([""] * 3)
b.set_ylabel("ARI  (absolute)", fontsize=10.5)
b.set_title("b   True spatial graph ≈ random graph", fontsize=11.5, loc="left", pad=9)
b.set_ylim(-0.04, 0.30)
b.legend(handles=[
    Line2D([], [], marker="o", ls="", mfc="white", mec=BLUE, mew=1.9, ms=8, label="A*  (no spatial edge)"),
    Line2D([], [], marker="o", ls="", mfc="white", mec=RED, mew=1.9, ms=8, label="B0  (true spatial kNN)"),
    Line2D([], [], marker="s", ls="", mfc="white", mec=GREEN, mew=1.9, ms=8, label="degree-preserving rewiring"),
], fontsize=8.6, frameon=False, loc="upper right")

# --- panel c: wiring evidence ---------------------------------------------
c = ax[2]
w = runs[0]["wiring"]
layers = [l["layer"] for l in w["layers"]]
msg = [l["weighted_message_l2_mean"] for l in w["layers"]]
grad = w["spatial_relation_msg_gradient_norms"]
x2 = np.arange(len(layers))
c.bar(x2 - 0.19, msg, width=0.36, color=BLUE, label="spatial message L2 norm")
c.bar(x2 + 0.19, grad, width=0.36, color=RED, label="relation message gradient norm")
c.set_xticks(x2)
c.set_xticklabels(["layer %d" % i for i in layers], fontsize=10)
c.set_ylabel("norm", fontsize=10.5)
c.set_title("c   The relation is live, not missing", fontsize=11.5, loc="left", pad=9)
c.legend(fontsize=8.8, frameon=False, loc="upper left")
c.set_ylim(0, max(max(msg), max(grad)) * 1.35)
c.text(0.5, 0.88, "removing spatial edges reproduces A*\nto < 1e-6; eval is bitwise identical",
       transform=c.transAxes, fontsize=8.6, color=GREY, ha="center", va="top")

for axis in ax:
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, format="pdf", bbox_inches="tight", facecolor="white")
print("wrote", OUT)
