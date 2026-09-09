"""F9：区分蓝图基底、公开代码基线及其现行 C 结果。

只绘制 F9。现行结果来自 goal_stage1/goal_stage2，历史数字另列来源，
不以旧 C 或旧机制轨迹填充当前状态。报告读取 why，完整来源保存在说明 JSON。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "experiments" / "reports"
OUT = REPORTS / "figures_dev"
NAME = "F9_方法与消融设计"
C名称 = "公开代码基线 C"


def 载入共享():
    path = ROOT / "scripts" / "figures" / "report_figures.py"
    spec = importlib.util.spec_from_file_location("交付图表", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["交付图表"] = module
    spec.loader.exec_module(module)
    return module


def 读json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def 数(value, places: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{places}f}"


def C结果(summary: dict) -> dict:
    """按 run 配对，不依赖可选的 paired / paired_final_minus_initial 字段。"""
    if not summary.get("complete") or summary.get("contract_violations"):
        raise ValueError("公开代码基线 C 汇总不完整或有协议违例，不能填入现行结果")
    runs = sorted(summary["runs"], key=lambda r: r["seed"])
    if [r["seed"] for r in runs] != [0, 1, 42]:
        raise ValueError("公开代码基线 C 必须包含 seed 0/1/42 各一次")
    diffs = [r["final"]["rare_f1"] - r["initial"]["rare_f1"] for r in runs]
    mechanism = {m["seed"]: m for m in summary["mechanism"]}
    active = [r["seed"] for r in runs
              if mechanism[r["seed"]]["final_predict_confident_fraction"] > 0
              and mechanism[r["seed"]]["proto_loss_final"] > 0]
    return {
        "name": C名称,
        "seeds": [r["seed"] for r in runs],
        "source_runs": [r["run_id"] for r in runs],
        "k0": [r["initial"]["n_clusters"] for r in runs],
        "final": {key: {"mean": mean(r["final"][key] for r in runs),
                         "sd": stdev(r["final"][key] for r in runs)}
                  for key in ("rare_f1", "ari")},
        "paired_final_minus_initial": {
            "rare_f1": {"values": diffs, "mean": mean(diffs), "sd": stdev(diffs)}
        },
        "active_seeds": active,
        "mechanism_pass": summary["gates"]["confidence_mechanism_cross_seed_pass"],
        "endpoint_pass": summary["gates"]["endpoint_cross_seed_pass"],
    }


def 历史背景() -> dict:
    ids = [
        "stage1_A_seed0",
        "stage1_B_k10_seed0",
        "stage2_C_t0.2_dq0.2_seed0_initA0_v3a_probe",
        "stage1_A_pr0.8_seed0",
        "stage2_C_t0.2_dq0.2_seed0_initA0_pr0.8",
        "stage2_C_t0.2_dq0.2_seed0_initA0_pr0.8_T30",
    ]
    rows = []
    for rid in ids:
        m = 读json(ROOT / "experiments" / "runs" / rid / "metrics.json")
        trace = m.get("dynamic", {}).get("refresh_trace") or []
        rows.append({"run_id": rid, "source": f"experiments/runs/{rid}/metrics.json",
                     "metrics": m["scformer"], "refresh_trace": trace})
    resolution = 读json(REPORTS / "伪标签resolution对照" / "report.json")
    return {"history": "历史结果保留，不参与现行 C 指标或门禁判定", "runs": rows,
            "initial_partitions": resolution["rows"],
            "initial_partition_source": "experiments/reports/伪标签resolution对照/report.json"}


def 图说明(stage1: dict, stage2: dict, tau: dict) -> dict:
    current = C结果(stage2)
    pair = current["paired_final_minus_initial"]["rare_f1"]
    grid = tau["tau_grid"]
    if len(grid) != 8:
        raise ValueError("F9 的预检范围说明要求八个已登记 tau 候选；请先核对新预检范围")
    tau_text = "、".join(f"{value:g}" for value in grid)
    basis = (
        "师姐原文以 Top-Z spot-gene 图和 spot/gene 双节点 Z 输入为基底。"
        "当前沿用公开代码的随机采样 gene 并集非零边、gene counts 输入与 counts KL 目标，"
        "未满足该基底，因此现行 C 称公开代码基线 C，不称完整严格蓝图复现。"
        "空间坐标还用于 A/C 的分块与 halo，关闭 spot-spot relation 不等于坐标不可见。"
    )
    result = (
        f"公开代码基线 C 的 K0 为 {'/'.join(map(str, current['k0']))}，"
        f"final-initial rare-F1 为 {pair['mean']:+.4f} ± {pair['sd']:.4f}；"
        f"结果门{'通过' if current['endpoint_pass'] else '未通过'}，"
        f"置信机制门{'通过' if current['mechanism_pass'] else '未通过'}。"
        f"最终置信比例与 prototype loss 均非零的 seed 为 {current['active_seeds']}。"
        "这只判定当前协议，不能否定修改点三或指定唯一根因。D 未开始，开关只表示计划。"
    )
    preflight = (
        f"tau 前置检查固定 δ={tau['delta_fixed']:g}，只扫八个候选 {tau_text}；"
        f"预检{'通过' if tau['preflight_pass'] else '未通过'}，"
        "结论限于这些候选及本次预检条件，不是所有 tau 无解，也不是完整 δ 敏感性分析。"
    )
    why = ("本图区分目标基底、实际实现和现行结果，避免把历史探索当作当前进度。"
           + basis + result + preflight
           + "C-A 比较的是整套动态流程，不能单独归因于 assignment 刷新。")
    return {
        "figure": "F9", "why": why,
        "caption": why + "公开代码基线 C 在 warm-up 后的 H_spot 上运行 Leiden 得到 K0，"
                   "之后 K 固定，assignment/prototype 每 T 个 epoch 更新。prototype 是簇内均值，"
                   "q=max softmax(cos(h,c)/τ)，仅 q>δ 的 spot 参与 L_proto；"
                   "warm-up 使用 L_KL，之后使用 L_KL+λ·L_proto，替代原固定分类与类内项，不加 L_spatial。"
                   "面板 b 为各组三 seed 均值，离散程度见 current_C 和原汇总；"
                   "A/B 使用表达侧固定标签，公开代码基线 C 使用各 run 的 warm-up Leiden 初始标签，"
                   "这是整套动态流程的比较，不能单独归因于 assignment 刷新。旧 C、res=0.8 和 T=5/T=30 的数字"
                   "及来源保存在 historical_background，不进入现行状态。",
        "报告端": "why 为报告实际读取的图说；caption 和结构化字段提供补充来源。",
        "current_C": current,
        "current_AB": stage1["arms"],
        "source_reports": ["experiments/reports/goal_stage1/summary.json",
                           "experiments/reports/goal_stage2/summary.json",
                           "experiments/reports/goal_stage2/tau_preflight.json"],
        "blueprint_source": "docs/ideas/scFormer修改思路.docx",
        "substrate": {"target": "Top-Z graph + spot Z + gene Z",
                      "current": "sampled gene-union nonzero edges; gene counts; KL counts",
                      "target_met": False},
        "tau_preflight": {"tau_grid": grid, "delta_fixed": tau["delta_fixed"],
                          "preflight_pass": tau["preflight_pass"], "scope": preflight},
        "historical_background": 历史背景(),
    }


def main() -> int:
    shared = 载入共享()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    # Keep F9 independently auditable: the shared style is still the source of
    # truth, but these export-critical settings remain visible in this script.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 600,
    })
    stage1 = 读json(REPORTS / "goal_stage1" / "summary.json")
    if not stage1.get("complete") or stage1.get("contract_violations"):
        raise ValueError("公开代码基线 A/B 汇总未通过完整性检查")
    info = 图说明(stage1, 读json(REPORTS / "goal_stage2" / "summary.json"),
                  读json(REPORTS / "goal_stage2" / "tau_preflight.json"))
    current = info["current_C"]
    pair = current["paired_final_minus_initial"]["rare_f1"]
    shared.出版样式()
    ink, muted, blue, red = shared.INK, shared.RULE, shared.UNLOCK, shared.SMALL_CLUSTER
    fig = plt.figure(figsize=(shared.W2, 112 / 25.4))
    ax = fig.add_axes([0.03, 0.35, 0.94, 0.52])
    ax.set(xlim=(0, 100), ylim=(0, 40))
    ax.set_axis_off()
    fig.text(0.03, 0.975, "a   Three layers that must not be conflated",
             weight="bold", fontsize=8, color=ink, va="top")
    fig.text(0.03, 0.938, "Sister's blueprint: Top-Z graph + Z inputs for BOTH spots and genes",
             fontsize=6.8, color=red, va="top")
    fig.text(0.03, 0.905, "Public-code baseline: sampled edges + gene counts; A/C still use spatial batching",
             fontsize=6.6, color=ink, va="top")
    fig.text(0.03, 0.872, "Input preflight completed; KL target, halo batching and reseeding remain declared conventions",
             fontsize=6.6, color=blue, va="top")

    framed_text = []

    def box(x, y, w, h, label, color=muted):
        patch = Rectangle((x, y), w, h, facecolor="white", edgecolor=color, lw=0.8)
        ax.add_patch(patch)
        text = ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                       fontsize=6.5, color=ink, linespacing=1.35)
        framed_text.append((patch, text))

    def arrow(start, end, color=muted, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7,
                                    lw=0.8, color=color, linestyle="--" if dashed else "-"))

    # Separate protocol rows: no arrow implies coordinates generate Top-Z.
    columns = [(0, "Protocol"), (24, "Expression-derived input"),
               (55, "Module comparison"), (82, "Evidence scope")]
    for x, label in columns:
        ax.text(x, 37, label, fontsize=7, weight="bold", va="center", color=ink)
    protocol_rows = [
        (28, "Blueprint", "Top-Z graph\nSpot Z + gene Z",
         "A / B / C / D\nSpatial and dynamic factors", "Target design", ink),
        (16, "Public-code\nbaseline", "Sampled nonzero edges\nGene counts",
         "A / B / C completed\nD not run", "Three-seed\nmodule results", ink),
        (4, "Input-wiring\npreflight", "Top-Z graph\nSpot Z + gene Z",
         "A / B / C / D\nShort runs only", "Wiring checked\nNot effectiveness", blue),
    ]
    for y, protocol, inputs, modules, scope, color in protocol_rows:
        ax.plot([0, 100], [y + 5.5, y + 5.5], lw=0.5, color=muted)
        for x, label in zip([0, 24, 55, 82], [protocol, inputs, modules, scope]):
            ax.text(x, y, label, fontsize=6.8, color=color, va="center", linespacing=1.4)

    table_ax = fig.add_axes([0.03, 0.09, 0.53, 0.23])
    table_ax.set_axis_off()
    table_ax.set(xlim=(0, 100), ylim=(0, 100))
    table_ax.text(0, 104, "b   What has actually been run (3-seed results)",
                  fontsize=7, weight="bold", color=ink)
    x = [0, 35, 54, 70, 90]
    for pos, label in zip(x, ["Run", "Protocol", "Module", "rare-F1", "ARI"]):
        table_ax.text(pos, 86, label, fontsize=6.0, color=muted)
    rows = [
        ("A", "public", "fixed labels", stage1["arms"]["A"]),
        ("B", "public", "+ spatial", stage1["arms"]["B"]),
        ("C", "public", "+ dynamic", current["final"]),
        ("D", "not run", "not run", None),
    ]
    for i, (label, spatial, dynamic, metrics) in enumerate(rows):
        values = [label, spatial, dynamic,
                  数(metrics["rare_f1"]["mean"]) if metrics else "--",
                  数(metrics["ari"]["mean"]) if metrics else "--"]
        for pos, value in zip(x, values):
            table_ax.text(pos, 66 - 19 * i, value, fontsize=6.0, color=ink, va="center")
    table_ax.plot([0, 100], [78, 78], color=muted, lw=0.6)
    table_ax.plot([0, 100], [0, 0], color=muted, lw=0.6)

    gate_ax = fig.add_axes([0.61, 0.09, 0.36, 0.23])
    gate_ax.set_axis_off()
    gate_ax.text(0, 1.04, "c   Current status", fontsize=7, weight="bold", color=ink)
    gate_ax.text(0, 0.79, "Strict-input effectiveness: NOT TESTED", fontsize=6.6, color=red)
    gate_ax.text(0, 0.55, "Confidence mechanism: " + ("PASS" if current["mechanism_pass"] else "FAIL"),
                 fontsize=6.6, color=red if not current["mechanism_pass"] else blue)
    gate_ax.text(0, 0.31, "Own-initial endpoint: " + ("PASS" if current["endpoint_pass"] else "FAIL"),
                 fontsize=6.6, color=red if not current["endpoint_pass"] else blue)
    gate_ax.text(0, 0.07, "Public C: K0 = " + "/".join(map(str, current["k0"])) + "; D not run",
                 fontsize=6.6, color=ink)
    fig.text(0.03, 0.052, f"C final - initial rare-F1: {pair['mean']:+.4f} ± {pair['sd']:.4f}. "
             "C-A compares the full dynamic workflow, not assignment refresh alone.",
             fontsize=5.9, color=ink)
    grid = info["tau_preflight"]["tau_grid"]
    fig.text(0.03, 0.021,
             f"Tau preflight: {len(grid)} candidates, {min(grid):g} to {max(grid):g}, "
             f"delta={info['tau_preflight']['delta_fixed']:g}. "
             "No conclusion about ALL tau.", fontsize=5.9, color=ink)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for patch, text in framed_text:
        outer, inner = patch.get_window_extent(renderer), text.get_window_extent(renderer)
        if not (outer.x0 <= inner.x0 <= inner.x1 <= outer.x1
                and outer.y0 <= inner.y0 <= inner.y1 <= outer.y1):
            raise ValueError(f"F9 框内文字越界：{text.get_text()}")
    for text in fig.findobj(match=plt.Text):
        if text.get_visible() and text.get_text():
            bounds = text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            if bounds.x0 < 0 or bounds.x1 > 1 or bounds.y0 < 0 or bounds.y1 > 1:
                raise ValueError(f"F9 文字超出画布：{text.get_text()}")
    texts = fig.texts + [text for axes in fig.axes for text in axes.texts]
    for i, left in enumerate(texts):
        for right in texts[i + 1:]:
            if left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer)):
                raise ValueError(f"F9 文字重叠：{left.get_text()} / {right.get_text()}")
    # 此次有界输出只写 figures_dev，不自动更新 tmp 或主报告交付目录。
    info["files"] = shared.存图(fig, OUT, NAME, preview=False)
    (OUT / "说明_F9.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] {C名称}；仅生成 F9 和说明_F9.json；框内文字与画布边界检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
