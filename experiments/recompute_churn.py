"""Recompute the two churn numbers the README and the report quote.

Run from the repository root:

    python experiments/recompute_churn.py

It prints, per seed: how far the trained output sits from the pseudo-label,
on the legacy basis (the fixed-target regime) and on the aligned basis.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N = 3484

print("aligned basis, A* (report: 62%, 51%, 60%)")
for s in (0, 1, 42):
    d = ROOT / "experiments" / "runs" / "p2_aligned_v1" / ("Astar_seed%d" % s)
    init = np.load(d / "initial_labels.npy")
    pred = np.load(d / "pred.npy")
    print("  seed %-3d changed %4d / %d = %.1f%%" % (s, (init != pred).sum(), len(init),
                                                     100.0 * (init != pred).mean()))

print()
print("aligned basis, B0")
for s in (0, 1, 42):
    d = ROOT / "experiments" / "runs" / "p2_aligned_v1" / ("B0_seed%d" % s)
    a = np.load(ROOT / "experiments" / "runs" / "p2_aligned_v1" / ("Astar_seed%d" % s) / "pred.npy")
    b = np.load(d / "pred.npy")
    print("  seed %-3d A* vs B0 differ on %4d / %d = %.1f%%" % (s, (a != b).sum(), len(a),
                                                               100.0 * (a != b).mean()))

print()
print("legacy basis (report: 0-1.9%), from the headroom report")
h = json.loads((ROOT / "experiments" / "reports" / "fixed_pseudolabel_headroom"
                / "report.json").read_text(encoding="utf-8"))
for row in h["rows"]:
    print("  seed %-3d armA %d spots (%.3f%%), armB %d spots (%.3f%%)" % (
        row["seed"], row["arm_a_改动_spot"], 100.0 * row["arm_a_改动_spot"] / N,
        row["arm_b_改动_spot"], 100.0 * row["arm_b_改动_spot"] / N))
