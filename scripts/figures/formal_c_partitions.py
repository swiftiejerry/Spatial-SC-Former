"""Plot the registered C partitions at their original tissue coordinates."""

import importlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    drawing = importlib.import_module("report_figures")
    drawing.出版样式()
    base = drawing.载入真值()
    summary = json.loads((ROOT / "experiments/reports/goal_stage2/summary.json").read_text(encoding="utf-8"))
    if not summary["complete"]:
        raise ValueError("formal C summary is incomplete")
    fig, axes = plt.subplots(3, 2, figsize=(7, 9.6))
    provenance = []
    for i, row in enumerate(summary["runs"]):
        run = ROOT / "experiments/runs" / row["run_id"]
        for j, (artifact, metric, title) in enumerate((
            ("warmup_init_labels.npy", "initial", "Warm-up Leiden"),
            ("pred.npy", "final", "Final dynamic partition"),
        )):
            labels = np.load(run / artifact)
            unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
            small = counts / labels.size < drawing.PRED_THR
            color = [drawing.SMALL_CLUSTER if small[k] else drawing.PRED_GREYS[k % len(drawing.PRED_GREYS)] for k in inverse]
            drawing.画切片(axes[i, j], base, color, 域界色=None)
            metrics = row[metric]
            axes[i, j].set_title(
                f"Seed {row['seed']} | {title}\n"
                f"K={len(unique)}  ARI={metrics['ari']:.4f}  rare-F1={metrics['rare_f1']:.4f}",
                fontsize=8, pad=8,
            )
            provenance.append({"seed": row["seed"], "artifact": str(run / artifact),
                               "small_cluster_spots": int(counts[small].sum())})
    fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.06, hspace=0.30, wspace=0.15)
    fig.text(0.5, 0.025, "Red: spots in predicted clusters <4.5% of the tissue; red does not identify a true rare type.",
             ha="center", fontsize=7)
    out = ROOT / "experiments/reports/figures_dev"
    files = drawing.存图(fig, out, "F11_正式C初始与最终空间分区")
    metadata = {
        "figure": "F11", "files": files,
        "why": "把公开代码基线 C 的三个 seed 初始划分与最终划分逐点映射回组织，检查小簇分布及其变化。本批仍是随机建图和gene counts输入，不是完整Top-Z蓝图复现。红色仅表示预测簇占比低于4.5%，不等于真稀有域；不同面板的灰色簇号不能直接对应。真值分布见图1。",
        "source": "experiments/reports/goal_stage2/summary.json", "panels": provenance,
    }
    (out / "说明_F11.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
