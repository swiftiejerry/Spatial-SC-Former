"""A1 消融的交付级图版（重制版）。

为什么重做
----------
上一版把每个问题都画成了柱状图或折线图。这类研究（空间转录组的域划分、稀有域检出）
在文献里的标准图语言不是柱线，而是「组织切片散点图版 + 嵌入空间散点 + 列联热图 +
流向图」，柱线只在最后当配角。图版读者第一眼要看到的是组织上的形状，不是条形的高低。

图版契约（先写结论，再画图）
----------------------------
F2/F3/F4/F6/F7/F8 保留历史探索的数据与来源，不代替公开代码基线 C 的现行结果。
固定伪标签下的近似复制、小簇出现与表示探针都只说明被测协议中的现象，不单独定位根因。
--metadata-only 只更新这些图的说明，不读取表达矩阵、不重绘图片。

证据链（一张图回答一个问题，回答重复的图不留）：

* F1  稀有域长什么样、占多少：组织散点主图 + 5 个稀有域的单独放大格 + 占比色带。
* F2  各 arm 在组织上到底切出了什么：预测分区图版 + 行归一化列联热图。
* F6  被 refine 的表示里有没有结构：嵌入空间散点图版，按真值上色。
* F3  稀有域去了哪里：真值域 → 预测簇的流向图（替代原来的召回柱状图）。
* F7  机制门过没过：按实际观测时点记录的簇位矩阵。
* F8  主终点可达吗：ARI–rare-F1 取舍与小簇结构的两面证据。

* F4  差值算不算效应：绝对水平横排点图 + 同 seed 配对差的放大格。
* F5  空间边进不进得了 batch：一个 batch 在组织上的三种取法 + 保留率与代价。

archetype：F1 / F2 / F6 是 image plate + quant，F5 是 image plate + quant，
F4 / F8 是 quantitative grid。

journal / export 契约：EI 双栏，双栏宽 178 mm、单栏宽 85 mm；正文字号 7 pt、
最小 5 pt；导出 SVG（文字可编辑）+ PDF（TrueType 42）+ PNG 1200 dpi；白底；
能直接标注就不用图例。

口径边界（画图不许越界）
------------------------
* rare-F1 为 0 的原因是划分里没有占比 < 4.5% 的簇，不是簇数。K=5 下上限是 1.0，已证。
* 小簇存在只说明「小簇成为可能」；只有小簇内稀有占比明显高过全片基线才算检出。
* 不写「空间关系无用」，只写「第一版无权重空间边在固定伪标签下没有可测变化」。
* slide_4E / Figure 7 的数字不进这张图版。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

# plot_primitives 和本文件同目录。直接跑本脚本时 sys.path[0] 就是 scripts/figures/，但
# method_diagram.py 是用 spec_from_file_location 把本文件当模块执行的，
# 那条路径下不保证 scripts/figures/ 在 sys.path 里，所以显式补一次。
_脚本目录 = str(Path(__file__).resolve().parent)
if _脚本目录 not in sys.path:
    sys.path.insert(0, _脚本目录)

import plot_primitives as 基元  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
REPORTS = ROOT / "experiments" / "reports"
H5AD = ROOT / "data" / "raw" / "human_lymph_node_A1" / "adata_RNA.h5ad"
LABEL_KEY = "final_annot"
UMAP_CACHE = REPORTS / "figures_dev" / "_缓存_嵌入二维坐标.npz"

# 专给 view_image 看的低分辨率副本。1200 dpi 交付图长边有 8000+ px，
# Codex 附图只压到 2048 px，而上游对「单次请求图片数 > 20」的请求要求
# 每张图长边 <= 2000 px，一看就把整条会话钉死在 400 IMAGE_DIMENSION_EXCEEDED。
预览目录 = ROOT / "tmp" / "看图预览"
预览上限 = 1600

# EI 双栏：178 mm / 85 mm
W2 = 178 / 25.4
W1 = 85 / 25.4
# 1.5 栏 = 双栏的 0.62。报告里 F5 就是按 0.62 倍栏宽摆的（生成技术报告.py 的图版表），
# 画多宽就摆多宽：画满 178 mm 再被版面缩到 0.62，7 pt 的正文字会变成 4.3 pt，
# 直接掉到 5 pt 底线以下。
W15 = 0.62 * 178 / 25.4

RARE_THR = 0.05   # 真值稀有类判据
PRED_THR = 0.045  # 预测稀有簇判据

# 语义配色：稀有类用饱和色，丰度类退成中性灰阶。
# 颜色本身承担「这一类要不要被注意」的语义，不是为了区分而区分。
RARE_COLORS = {
    "follicle": "#B64342",
    "medulla vessels": "#0F4D92",
    "subcapsular sinus": "#42949E",
    "hilum": "#9A4D8E",
    "trabeculae": "#E28E2C",
}
# 蜂窝铺满之后灰阶要重挑。第一版从 #E6E6E6 起步、到 #343434 收尾：切片内部最大的
# 那一类几乎是白的，读者看到的是一个空心的组织；而边缘的 capsule / pericapsular
# adipose 是近黑的一圈，比五个稀有域还抢眼。域边界画出来之后灰阶不必再承担
# 「分开相邻的两块」这件事，所以整体抬一档、范围收窄：最浅的 #DCDCDC 已经有实体感，
# 最深的 #8C8C8C 也压不过饱和的稀有色。
ABUNDANT_GREYS = ["#DCDCDC", "#CBCBCB", "#B9B9B9", "#A8A8A8", "#999999", "#8C8C8C"]
# 预测簇的灰阶单独一套，比真值那套宽。真值的域在空间上是连片的，域边界画出来就是形状，
# 灰阶只要能区分相邻块；预测出来的块是碎的，同一个判据下界线成千条，画出来是一片麻点，
# 所以 F2 的分区图不画内部界线，簇之间全靠灰阶分开——那就必须把灰阶拉开。
PRED_GREYS = ["#E2E2E2", "#C6C6C6", "#AAAAAA", "#8E8E8E", "#767676", "#626262"]
INK = "#272727"
RULE = "#767676"
SMALL_CLUSTER = "#B64342"
# 丰度类之间的域界线。比 INK 浅得多：组织内部相邻大类之间的界线很密，
# 用 INK 0.35 画会把切片内部糊成一片麻点（试过，太吵）。
域界灰 = "#8A8A8A"

# 图上写全名会被邻格切掉（pericapsular adipose tissue 就被切过），
# 长名字在图内用缩写，全名留给正文图注。
短名 = {"pericapsular adipose tissue": "pericapsular adipose"}

# F4 / F5 这两张定量图的配色单独一套。组织图里的红是「稀有类」，这里要区分的是
# 「伪标签锁着 vs 解锁」和「边留住 vs 边被切掉」——同一个红在两张图里指两件事，
# 是读者最容易被坑的地方，所以两套颜色分开命名，不互相借用。
LOCKED = "#585858"    # A0 / A / B：输出被固定伪标签钉住的那一族
UNLOCK = "#0F4D92"    # C：prototype 动态聚类；F5 里同时表示「留在 batch 内的边」
UNLOCK2 = "#3775BA"
EDGE_LOST = "#D4837D"  # F5 里被切掉的空间边。比 F8 那档浅红重一点：全片图上那些线只有
                       # 0.3 pt 宽，太浅在 600 dpi 下几乎看不见。
SEEDS = (0, 1, 42)
SEED_MARK = {0: "o", 1: "s", 42: "^"}


def 偏暗(hex_color: str, threshold: float = 140.0) -> bool:
    """底色暗就用白字。字色不能靠 hex 字符串比大小去猜。"""
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < threshold


def kNN纯度(matrix: np.ndarray, labels: np.ndarray, k: int = 10) -> tuple[float, float]:
    """邻居里同类的比例，以及随机水平（各类占比平方和）。

    图里自己算，不从别处的 json 抄：图上画的是哪份 embedding，数字就必须是那份
    embedding 算出来的，否则两边一漂就是事故。
    """
    from sklearn.neighbors import NearestNeighbors

    unit = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    finder = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(unit)
    _, index = finder.kneighbors(unit)
    purity = float((labels[index[:, 1:]] == labels[:, None]).mean())
    share = np.bincount(labels) / labels.size
    return purity, float((share ** 2).sum())


def 上游表达空间(adata) -> np.ndarray:
    """伪标签真正的来源侧：normalize_total → log1p → PCA → 前 40 个主成分。

    逐步对齐 scformer.utils.initial_clustering，没有 HVG 也没有 scale。
    参照系必须和实际产伪标签的那条路一致，自己另搭一个空间会把对比整体带偏。
    """
    import scanpy as sc

    work = adata.copy()
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    sc.tl.pca(work, svd_solver="arpack", random_state=0)
    return np.asarray(work.obsm["X_pca"][:, :40], dtype=np.float32)


def 二维投影(名: str, matrix: np.ndarray, seed: int = 0) -> np.ndarray:
    """UMAP 到 2D，结果缓存。改图很频繁，每次重算六份 UMAP 太慢。"""
    UMAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    存 = dict(np.load(UMAP_CACHE)) if UMAP_CACHE.is_file() else {}
    键 = 名 + "_" + str(matrix.shape) + "_" + f"{float(matrix.sum()):.4e}"
    if 键 in 存:
        return 存[键]
    import umap

    xy = umap.UMAP(n_neighbors=15, min_dist=0.25, metric="cosine",
                   random_state=seed).fit_transform(matrix)
    存[键] = np.asarray(xy, dtype=np.float32)
    np.savez_compressed(UMAP_CACHE, **存)
    return 存[键]


def 出版样式() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def 面板号(ax, label, x=-0.10, y=1.02, color=INK, size=9.0) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=size,
            fontweight="bold", color=color, ha="left", va="bottom")


def 存图(fig, out: Path, name: str, *, preview: bool = True) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kw in ((".svg", {}), (".pdf", {"dpi": 1200}), (".png", {"dpi": 1200})):
        p = out / (name + suffix)
        fig.savefig(p, **kw)
        paths.append(p.name)
    if preview:
        存预览(fig, name)
    量宽(fig, name)
    plt.close(fig)
    return paths


def 量宽(fig, name: str) -> float:
    """落盘宽度自检。图是 bbox="tight" 存的，figsize 不等于成图宽度，
    而 EI 跨栏上限是 178 mm——事后被排版发现比这里打一行贵得多。"""
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    宽mm = 25.4 * (bb.width + 2 * float(plt.rcParams["savefig.pad_inches"]))
    高mm = 25.4 * (bb.height + 2 * float(plt.rcParams["savefig.pad_inches"]))
    print(f"[bbox] {name}: {宽mm:.1f} x {高mm:.1f} mm"
          + ("   [warn] 超过 178 mm 跨栏上限" if 宽mm > 178.0 else ""), flush=True)
    return 宽mm


def 存预览(fig, name: str) -> Path:
    """另存一张长边 <= 预览上限 的副本，看图只看它，交付图不动。"""
    from PIL import Image

    预览目录.mkdir(parents=True, exist_ok=True)
    p = 预览目录 / (name + ".png")
    fig.savefig(p, dpi=max(40.0, 预览上限 * 0.94 / max(fig.get_size_inches())))
    # bbox="tight" 会按实际画到的范围裁剪，算出来的 dpi 不保证卡住上限，落盘后复核一次。
    with Image.open(p) as im:
        w, h = im.size
        if max(w, h) > 预览上限:
            k = 预览上限 / max(w, h)
            im.convert("RGB").resize((max(1, round(w * k)), max(1, round(h * k))),
                                     Image.LANCZOS).save(p, optimize=True)
    return p


def 读json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


历史图文 = {
    "F2": (
        "历史探索：seed 0 的 A0/A/B，以及 v2b warm-up 初始化、v3a A0-label 初始化。"
        "这些 C 不是公开代码基线 C。空间分区、小簇与列联热图用于核对预测去向；"
        "图中默认 resolution 0.5 的固定标签 run 没有低于 4.5% 的预测簇，"
        "不能推广到 resolution 0.8 或所有固定标签协议。"
    ),
    "F3": (
        "历史探索：B 与 v2b/v3a 两种旧 C 的稀有类流向，不是 T=5/T=30 对照，"
        "也不是公开代码基线 C。缎带宽度是 spot 数，红色簇条表示占比低于 4.5%。"
        "流向只描述哪些 spot 被分到同一簇，不能据此排除空间关系或认定簇位分配是唯一原因。"
    ),
    "F4": (
        "历史探索：图中只保留同 seed 的 B-A 配对差，A0/A/B 与两组旧 C 不作为同一实验条件合并。"
        "空心点对应 seed 0、1、42，实心菱形为均值及样本标准差；零线表示没有变化。"
        "本批 B-A 的 ARI/NMI 差约为 1e-3 且跨 seed 翻号，rare-F1 配对差为 0；"
        "这只能说明当前协议没有检出稳定效应。"
    ),
    "F6": (
        "历史表示诊断：30ep 纯 L_KL 后另有 1ep 的旧 preflight、100ep warmuponly、"
        "旧 A 和 v2b/v3a C；不是公开代码基线 C 的三 seed ep30 快照。"
        "投影与 kNN 标签纯度用于比较这些特定表示的域信号，低纯度提示候选瓶颈，"
        "不能证明表示是唯一根因、排除 τ/δ，或外推所有训练时长。"
        "另行 tau 预检只覆盖固定 δ=0.8 的八个候选 0.002、0.005、0.008、0.01、"
        "0.015、0.02、0.03、0.05；候选未通过不等于所有 tau 无解。"
    ),
    "F7": (
        "历史机制轨迹：seed 0 的 v2a（δ=0.5、τ=0.02）、v2b（分位 δ、warm-up 初始化）"
        "和 v3a（分位 δ、A0-label 初始化）。它们不是 T=30 对照，也不能替代公开代码基线 C "
        "的机制判定。五个簇位与高置信比例只对应图中旧协议；分位阈值下比例接近配置目标"
        "不等于绝对阈值 δ=0.8 持续工作，更不能单凭簇位保留判定表示或稀有域检测成功。"
    ),
    "F8": (
        "历史初始化诊断：面板 a 比较 resolution 扫描与旧 A0/A/B、v2b/v3a C，"
        "不含公开代码基线 C。面板 b 和 c 分别显示最小簇占比与小簇内稀有占比，"
        "并把 4.5% 预测阈值和 8.0% 全片稀有基线画成参考线；没有小簇的分辨率记为 n/a。"
        "resolution 0.2/0.8 产生了富集小簇，主要对应 follicle；这不等于多个稀有域都已被检测到。"
    ),
}


def 历史图来源(figure: str) -> list[str]:
    templates = {
        "F2": [
            "stage0_A0_seed0", "stage1_A_seed0", "stage1_B_k10_seed0",
            "stage2_C_t0.2_dq0.2_seed0_v2b_preflight",
            "stage2_C_t0.2_dq0.2_seed0_initA0_v3a_probe",
        ],
        "F3": [
            "stage1_B_k10_seed0", "stage2_C_t0.2_dq0.2_seed0_v2b_preflight",
            "stage2_C_t0.2_dq0.2_seed0_initA0_v3a_probe",
        ],
        "F4": [
            "stage0_A0_seed{seed}", "stage1_A_seed{seed}",
            "stage1_B_k10_seed{seed}",
            "stage2_C_t0.2_dq0.2_seed{seed}_v2b_preflight",
            "stage2_C_t0.2_dq0.2_seed{seed}_initA0_v3a_probe",
        ],
        "F6": [
            "stage2_C_t0.08_d0.8_seed0_preflight",
            "stage2_C_t0.2_dq0.2_seed0_warmuponly", "stage1_A_seed0",
            "stage2_C_t0.2_dq0.2_seed0_v2b_preflight",
            "stage2_C_t0.2_dq0.2_seed0_initA0_v3a_probe",
        ],
        "F7": [
            "stage2_C_t0.02_d0.5_seed0_v2a_preflight",
            "stage2_C_t0.2_dq0.2_seed0_v2b_preflight",
            "stage2_C_t0.2_dq0.2_seed0_initA0_v3a_probe",
        ],
        "F8": [
            "stage0_A0_seed0", "stage1_A_seed0", "stage1_B_k10_seed0",
            "stage2_C_t0.2_dq0.2_seed0_v2b_preflight",
            "stage2_C_t0.2_dq0.2_seed0_initA0_v3a_probe",
        ],
    }[figure]
    return [template.format(seed=seed) for template in templates
            for seed in ((0, 1, 42) if "{seed}" in template else (0,))]


def 历史图说明(figure: str, files: list[str], source_runs: list[str],
             coordinate_note: str = "") -> dict:
    text = 历史图文[figure]
    if coordinate_note:
        text += "坐标口径：" + coordinate_note + "。"
    return {
        "figure": figure, "files": files, "why": text, "caption": text,
        "history": "历史诊断，不替代公开代码基线 C",
        "source_runs": source_runs,
        "coordinate_note": coordinate_note,
        "image_status": "图面已含历史标签",
    }


def 历史图标记(fig) -> None:
    # 位于已有内容上方，避免历史标签覆盖数据或改变面板内的布局。
    fig.canvas.draw()
    top = fig.get_tightbbox(fig.canvas.get_renderer()).y1 / fig.get_size_inches()[1]
    fig.text(0.01, top + 0.018, "HISTORICAL DIAGNOSTICS | not the current public-code baseline C",
             fontsize=6.2, color=RULE, ha="left", va="bottom")


def 只更新历史说明(out: Path, selected: list[str] | None = None) -> None:
    path = out / "图表说明.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    targets = set(历史图文 if selected is None else selected)
    if targets - 历史图文.keys():
        raise ValueError("metadata-only 仅支持 F2/F3/F4/F6/F7/F8")
    found = {item["figure"] for item in data["figures"]}
    if targets - found:
        raise ValueError(f"说明清单缺少：{sorted(targets - found)}")
    for item in data["figures"]:
        figure = item["figure"]
        if figure not in targets:
            continue
        sources = 历史图来源(figure)
        note = item.get("coordinate_note", "")
        if not note and figure == "F2":
            note = ("spot 坐标为 Visium 阵列坐标，已按列距 100 µm、行距 86.6 µm "
                    "还原真实几何，每个瓦片是该 spot 的 Voronoi 胞")
        previous_status = item.get("image_status")
        item.update(历史图说明(figure, item["files"], sources, note))
        item["source_reports"] = {
            "F6": ["experiments/reports/warmup_representation_quality/report.json",
                   "experiments/reports/goal_stage2/tau_preflight.json"],
            "F8": ["experiments/reports/rare_domain_baseline_frontier/report.json"],
        }.get(figure, [])
        item["image_status"] = previous_status or "既有图片未重绘；下次单图重绘将加入历史标签并使用修正后的图面文字"
        item["source_metrics"] = [
            {"run_id": rid, "path": f"experiments/runs/{rid}/metrics.json",
             "scformer": 读json(RUNS / rid / "metrics.json").get("scformer", {})}
            for rid in sources
        ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[metadata-only] 更新 {', '.join(sorted(targets))}，未读取表达矩阵、未重绘图片")


def 载入真值() -> dict:
    import scanpy as sc

    adata = sc.read_h5ad(H5AD)
    truth = adata.obs[LABEL_KEY].astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    share = truth.value_counts(normalize=True)
    counts = truth.value_counts()
    rare = [t for t in share.index if share[t] < RARE_THR]
    abundant = [t for t in share.index if share[t] >= RARE_THR]
    颜色 = {}
    for i, t in enumerate(abundant):
        颜色[t] = ABUNDANT_GREYS[i % len(ABUNDANT_GREYS)]
    for t in rare:
        颜色[t] = RARE_COLORS.get(t, SMALL_CLUSTER)
    # 晶格几何只算一次：3484 个 spot 之间有 10231 条邻接对，每个 panel 重算一遍
    # 会明显变慢，而整套图版画的都是同一张切片。
    几何 = 基元.晶格几何(coords)
    return {"truth": truth, "coords": coords, "share": share, "counts": counts,
            "rare": rare, "abundant": abundant, "颜色": 颜色,
            "几何": 几何, "换算说明": 基元.单位换算说明(几何)}


def 组织散点(ax, coords, colors, s=1.6, alpha=0.95) -> None:
    """组织切片散点：等比例、无坐标轴，点小到能看出细带和薄环。

    交付图已经全部换成 `画切片()` 的蜂窝画法，这里只留给别处调用：散点看不出组织
    形状（3484 个点摆在整数格上是一片稀疏点云），而且直接拿阵列坐标当等比例画会把
    近圆形的淋巴结压成细长椭圆。
    """
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=s, linewidths=0,
               alpha=alpha, rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def 画切片(ax, 基: dict, 面色, *, 分区=None, 域界色: str | None = 域界灰,
        单独描边: dict | None = None, 比例尺: bool = True,
        长度_um: float = 500.0) -> dict:
    """交付图里所有组织切片的唯一画法：Voronoi 蜂窝 + 域边界 + 外轮廓 + 比例尺。

    以前每张图各画各的散点，点大小、有没有比例尺、长宽比全靠人记，收成一条路径之后
    样式漂移就没有藏身处。`分区` 给一份逐 spot 的标签，相邻标签不同的瓦片之间会画出
    公共边——这是蜂窝相对散点最大的收益：薄环和细带能直接看出是连通的还是碎的。
    `单独描边` 是 {标签: 颜色}，给需要被找到的那几个域（稀有类、低于判据的小簇）
    再描一遍自己颜色的粗边；丰度类之间只用浅灰细线，谁该被注意由颜色和线宽一起说。
    单域高亮的格子里所有界线都是「域 vs 外部」，浅灰那一遍纯属多余，
    传 `域界色=None` 关掉。
    """
    g = 基["几何"]
    coords = 基["coords"]
    基元.组织瓦片(ax, coords, 面色, 收缩=0.94, 几何=g)
    if 分区 is not None:
        标签 = np.asarray(分区)
        if 域界色 is not None:
            基元.域边界(ax, coords, 标签, 几何=g, 色=域界色, 宽=0.25)
        for 标, 色 in (单独描边 or {}).items():
            基元.域边界(ax, coords, 标签, 几何=g, 色=色, 宽=0.5, 只画={标})
    基元.组织外轮廓(ax, coords, 几何=g, 宽=0.5)
    if 比例尺:
        基元.比例尺(ax, g, 长度_um=长度_um)
    return g


def 可直标(基: dict, mask: np.ndarray) -> bool:
    """这个域能不能在切片上原地直标。

    `标签重心` 取的是中位数，散落成一圈的域（follicle 绕着囊膜、subcapsular sinus
    是一段弧）重心会落到组织正中间的空白处，标在那里等于指错地方。判据用「重心到
    该域最近一个 spot 的距离」：超过一个 spot 间距就不直标，交给单域格和图例。
    """
    g = 基["几何"]
    um = g["um"]
    cx, cy = 基元.标签重心(um, mask)
    if not np.isfinite(cx):
        return False
    return float(np.hypot(um[mask, 0] - cx, um[mask, 1] - cy).min()) <= g["pitch"]


def figure_1(基: dict, out: Path) -> dict:
    """F1 —— 稀有域长什么样、占多少。

    还原真实几何之后组织几乎是圆的（列距 100 µm、行距 86.6 µm，长宽比 1.06），
    不再是拿阵列坐标当等比例画出来的那个细长椭圆，所以 hero 是一个近正方的大格竖跨
    两行，5 个稀有域的单独格排成 2x3 贴在右侧，色块清单收成一条横带（上排丰度、
    下排稀有），比例色带独占最下一行。丰度类的名字不写在切片上：这几类在空间上互相
    穿插，质心标注必然互相压字（第一版就压成一团），名字全部交给横带。
    """
    truth, 颜色 = 基["truth"], 基["颜色"]
    rare, counts, share = 基["rare"], 基["counts"], 基["share"]
    g = 基["几何"]
    n = len(truth)
    真值 = truth.to_numpy()

    # 格子的长宽比按切片的真实长宽比配：hero 竖跨两行加一条行间距，所以它的宽度必须
    # 约等于两个小格的宽度，否则 equal aspect 会把 hero 缩小、上下各留一条白。
    fig = plt.figure(figsize=(W2, W2 * 0.656))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.0, 0.24, 0.22],
                          width_ratios=[2.30, 1.0, 1.0, 1.0],
                          hspace=0.30, wspace=0.07,
                          left=0.030, right=0.985, top=0.925, bottom=0.045)

    ax = fig.add_subplot(gs[0:2, 0])
    画切片(ax, 基, [颜色[t] for t in 真值], 分区=真值,
         单独描边={t: 颜色[t] for t in rare})
    # 标题、副标题、面板号各占一行，别叠在同一个 y 上（第一版就叠了）。
    面板号(ax, "a", x=-0.008, y=1.048)
    ax.text(0.052, 1.048, "Ground-truth tissue domains", transform=ax.transAxes,
            fontsize=7.6, fontweight="bold", va="bottom", ha="left", color=INK)
    ax.text(0.052, 1.008, str(n) + " spots, 10 annotated domains",
            transform=ax.transAxes, fontsize=6.2, va="bottom", ha="left",
            color=RULE)
    # 直标只给空间上聚成块的稀有域。hilum 23 个 spot、trabeculae 8 个，不标出来读者
    # 在切片上根本找不到；follicle 和 subcapsular sinus 散在囊膜一圈，重心会落到组织
    # 正中间的空白处，那两个只在自己的单域格和横带里给名字。
    for t in rare:
        m = (truth == t).to_numpy()
        if not 可直标(基, m):
            continue
        cx, cy = 基元.标签重心(g["um"], m)
        朝外 = np.array([cx, cy]) - g["um"].mean(axis=0)
        朝外 = 朝外 / (np.linalg.norm(朝外) + 1e-9)
        基元.白描字(ax, cx + 朝外[0] * 3.2 * g["R"], cy + 朝外[1] * 3.2 * g["R"],
                短名.get(t, t), 字号=6.0, 色=颜色[t], weight="bold",
                ha="left" if 朝外[0] >= 0 else "right")

    序 = sorted(rare, key=lambda t: -int(counts[t]))
    位置 = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2)]
    for (r, c), t in zip(位置, 序):
        axi = fig.add_subplot(gs[r, c])
        m = (truth == t).to_numpy()
        画切片(axi, 基, np.where(m, 颜色[t], 基元.背景瓦片).tolist(),
             分区=m.astype(int), 域界色=None, 单独描边={1: 颜色[t]})
        axi.set_title(t, fontsize=6.5, color=颜色[t], pad=2.0, fontweight="bold")
        axi.text(0.5, -0.01,
                 str(int(counts[t])) + " spots  " + f"{100 * share[t]:.2f}%",
                 transform=axi.transAxes, fontsize=5.9, ha="center", va="top",
                 color=RULE)
        if (r, c) == (0, 1):
            面板号(axi, "b", x=-0.075, y=1.122)
            axi.text(0.07, 1.122, "One rare domain per panel",
                     transform=axi.transAxes, fontsize=7.0, va="bottom",
                     ha="left", color=INK)

    # 2x3 里剩下的那一格写读图提示，别留一个白洞。
    axn = fig.add_subplot(gs[1, 3])
    axn.set_axis_off()
    axn.text(0.0, 0.99,
             "Each small panel shows one\nrare domain (colour) against\n"
             "all 3484 spots (grey).\n\nThey are thin rims, narrow\n"
             "bands and single spots, not\ncompact blobs, so a 5-way\n"
             "partition has to spend a\nwhole cluster slot on < 3%\n"
             "of the tissue to recover one.",
             transform=axn.transAxes, fontsize=5.7, va="top", ha="left",
             color="#3A3A3A", linespacing=1.34)

    # 色块清单收成一条横带：上排五个丰度类、下排五个稀有类，顺序按占比。竖着摆在 hero
    # 旁边的话，那一列的宽度要从 hero 身上扣，而 hero 现在是近正方的，扣掉之后它竖跨
    # 两行就不成比例了。
    axl = fig.add_subplot(gs[2, :])
    axl.set_axis_off()
    顺序 = list(share.sort_values(ascending=False).index)
    for i, t in enumerate(顺序):
        x = (i % 5) * 0.174
        y = 0.74 if i < 5 else 0.24
        是稀有 = t in rare
        axl.add_patch(plt.Rectangle((x, y - 0.17), 0.0165, 0.34,
                                    facecolor=颜色[t], edgecolor="#B5B5B5",
                                    linewidth=0.3,
                                    transform=axl.transAxes, clip_on=False))
        axl.text(x + 0.025, y,
                 短名.get(t, t) + f"  {100 * float(share[t]):.1f}%",
                 transform=axl.transAxes, fontsize=5.6, va="center",
                 ha="left", color=颜色[t] if 是稀有 else "#3A3A3A",
                 fontweight="bold" if 是稀有 else "normal")
    axl.text(0.878, 0.49, "bold = rare domain\n(< 5% of tissue)",
             transform=axl.transAxes, fontsize=5.4, va="center", ha="left",
             color=RULE, linespacing=1.35)

    axb = fig.add_subplot(gs[3, :])
    左 = 0.0
    for t in 顺序:
        w = float(share[t])
        axb.barh(0, w, left=左, height=0.5, color=颜色[t],
                 edgecolor="white", linewidth=0.5)
        if w > 0.12:
            axb.text(左 + w / 2, 0, 短名.get(t, t) + "\n" + f"{100 * w:.1f}%",
                     ha="center",
                     va="center", fontsize=5.6,
                     color="white" if 偏暗(颜色[t]) else "#272727")
        左 += w
    稀有和 = float(sum(share[t] for t in rare))
    # 稀有块是一条很窄的条，用括号把整段括起来比用箭头指某一格准确。
    axb.plot([1 - 稀有和, 1], [-0.34, -0.34], color=INK, lw=0.7)
    for x in (1 - 稀有和, 1.0):
        axb.plot([x, x], [-0.34, -0.27], color=INK, lw=0.7)
    axb.text(1.0, -0.46,
             "5 rare domains = " + str(int(sum(counts[t] for t in rare)))
             + f" spots ({100 * 稀有和:.1f}% of tissue)",
             ha="right", va="top", fontsize=6.2, color=INK)
    axb.set_xlim(0, 1)
    axb.set_ylim(-1.05, 0.35)
    axb.set_yticks([])
    axb.set_xticks([])
    for spine in axb.spines.values():
        spine.set_visible(False)
    面板号(axb, "c", x=-0.02, y=0.62)

    files = 存图(fig, out, "F1_真值分区与稀有类")
    return {"figure": "F1", "files": files,
            "why": "主终点是 rare-F1，读者必须先看到稀有域在组织上的形状和体量：薄环、"
                   "细带、小点，合计只占 8.0%。有了形状，后面 rare-F1 为 0 才能被判断成"
                   "判据问题而不是随口结论。柱状图给不了形状，所以用组织切片主图加"
                   "单独放大格，占比只用一条比例色带交代。"
                   "坐标口径：" + 基["换算说明"] + "。"}


def 载入预测(run_id: str):
    p = RUNS / run_id / "pred.npy"
    return np.load(p) if p.is_file() else None


def 列联(pred, truth, 顺序) -> np.ndarray:
    """行是真值类，列是簇；每行归一化成这一类有多少比例去了这个簇。"""
    簇 = sorted(int(v) for v in np.unique(pred))
    M = np.zeros((len(顺序), len(簇)), dtype=float)
    tv = truth.to_numpy()
    for i, t in enumerate(顺序):
        m = tv == t
        total = int(m.sum())
        if total == 0:
            continue
        for j, c in enumerate(簇):
            M[i, j] = float((pred[m] == c).sum()) / total
    return M


def figure_2(基: dict, out: Path, arms) -> dict | None:
    """F2 —— 各 arm 在组织上切出了什么，以及每个真值类被送去了哪里。

    每一行一个 arm：左列组织分区图，中列只点出低于判据的小簇，右列行归一化列联热图。
    分区图回答「切出来的块像不像域」，热图回答「稀有类去了谁的簇」，一行读完一个 arm。
    """
    truth = 基["truth"]
    可用 = [(名, rid, 载入预测(rid)) for 名, rid in arms]
    可用 = [(名, rid, p) for 名, rid, p in 可用 if p is not None]
    if not 可用:
        return None
    顺序 = list(基["share"].sort_values(ascending=False).index)
    n行 = len(可用)

    # 列序：热图放最左，真值类的名字就落在整图左边缘，不会压到别的格子里去
    # （上一版把热图放最右，10 个类名挤进了中间那格的组织图上）。
    # 切片还原真实几何之后近乎正方（长宽比 1.06），格子的宽高比要照着配：以前按竖椭圆
    # 给的宽格子，等比例会把切片缩到只占格子宽度的三分之一，剩下的全是白。
    # 热图和两列切片分成两个 gridspec：单个 gridspec 的 wspace 是全局的，切片之间要紧、
    # 热图和切片之间要松（那是「定量」和「图版」两半的分界），一个数给不了两种间距。
    fig = plt.figure(figsize=(W2, 1.10 * n行 + 0.85))
    界 = dict(top=0.925, bottom=0.050, hspace=0.26)
    # left 给到 0.160：pericapsular adipose tissue 这个类名有 1.09 英寸宽，
    # 左边距小于它就会顶出版面，tight bbox 跟着变宽。
    gs热 = fig.add_gridspec(n行, 1, left=0.160, right=0.630, **界)
    gs图 = fig.add_gridspec(n行, 2, left=0.730, right=0.985, wspace=0.10, **界)
    try:
        cmap = plt.get_cmap("rocket_r")
    except ValueError:
        cmap = plt.get_cmap("magma_r")
    im = None

    for r, (名, rid, pred) in enumerate(可用):
        sizes = np.bincount(pred)
        非空 = np.where(sizes > 0)[0]
        占比 = sizes[非空] / sizes.sum()
        小 = [int(c) for c, f in zip(非空, 占比) if f < PRED_THR]
        簇色 = {}
        gi = 0
        # 灰阶按簇大小给，最大的最浅——和 F1 里丰度类的口径一致：最大的那块退到背景去，
        # 小的反而显。按簇号给的话哪个簇拿到最深的灰纯属偶然，跨 arm 也没法比。
        for c, f in sorted(zip(非空, 占比), key=lambda kv: -kv[1]):
            if f < PRED_THR:
                簇色[int(c)] = SMALL_CLUSTER
            else:
                簇色[int(c)] = PRED_GREYS[gi % len(PRED_GREYS)]
                gi += 1

        ax0 = fig.add_subplot(gs图[r, 0])
        # 只给小簇描红边。丰度簇之间不画界线：预测分区在空间上是碎的，界线画满之后
        # 整片组织变成麻点，簇的形状反而看不见了（试过）。
        画切片(ax0, 基, [簇色[int(c)] for c in pred], 分区=pred,
             域界色=None, 单独描边={c: SMALL_CLUSTER for c in 小})
        # arm 名和关键数字写在格子外面，写在里面会压到组织上。
        ax0.text(-0.02, 1.105, 名, transform=ax0.transAxes, fontsize=6.6,
                 fontweight="bold", va="bottom", ha="left", color=INK)
        ax0.text(-0.02, 1.015,
                 "K=" + str(len(非空)) + f"   min cluster {100 * 占比.min():.2f}%",
                 transform=ax0.transAxes, fontsize=5.9, va="bottom", ha="left",
                 color=RULE)
        if r == 0:
            面板号(ax0, "b", x=-0.09, y=1.50)
            # 标题折两行：切片格子只有 0.85 英寸宽，一行摆不下的标题会横着顶出版面。
            ax0.text(0.09, 1.50, "Predicted\ndomains", transform=ax0.transAxes,
                     fontsize=7.0, va="bottom", ha="left", color=INK,
                     linespacing=1.25)

        ax1 = fig.add_subplot(gs图[r, 1])
        m = np.isin(pred, 小) if 小 else np.zeros(len(truth), dtype=bool)
        画切片(ax1, 基, np.where(m, SMALL_CLUSTER, 基元.背景瓦片).tolist(),
             分区=m.astype(int), 域界色=None,
             单独描边={1: SMALL_CLUSTER} if 小 else None)
        # 数字挪进格子里：切片是圆的，四个角本来就是死空间，写在格子外面这两行会横着
        # 顶出版面（上一版就把整图的 tight bbox 顶宽了）。
        if 小:
            稀有占比 = float(truth.isin(基["rare"]).to_numpy()[m].mean())
            基元.白描字(ax1, 0.015, 0.995,
                    str(int(m.sum())) + " spots in\nsmall clusters",
                    字号=5.7, 色=SMALL_CLUSTER, ha="left", va="top",
                    transform=ax1.transAxes)
            基元.白描字(ax1, 0.015, 0.775,
                    f"rare inside {100 * 稀有占比:.1f}%\n(baseline 8.0%)",
                    字号=5.7, ha="left", va="top",
                    色=SMALL_CLUSTER if 稀有占比 > 0.08 else "#5A5A5A",
                    transform=ax1.transAxes)
        else:
            基元.白描字(ax1, 0.015, 0.995, "no cluster\nbelow 4.5%", 字号=5.7,
                    色="#5A5A5A", ha="left", va="top",
                    transform=ax1.transAxes)
        if r == 0:
            面板号(ax1, "c", x=-0.09, y=1.50)
            ax1.text(0.09, 1.50, "Spots below\nthe 4.5% cut",
                     transform=ax1.transAxes, fontsize=7.0, va="bottom",
                     ha="left", color=INK, linespacing=1.25)

        ax2 = fig.add_subplot(gs热[r, 0])
        M = 列联(pred, truth, 顺序)
        im = ax2.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax2.set_xticks(range(M.shape[1]))
        ax2.set_xticklabels([str(int(c)) for c in 非空], fontsize=5.7)
        # 真值类的名字每行重复五遍是浪费，只在第一行标一次。
        if r == 0:
            ax2.set_yticks(range(len(顺序)))
            ax2.set_yticklabels(顺序, fontsize=5.6)
            for k, t in enumerate(顺序):
                if t in 基["rare"]:
                    ax2.get_yticklabels()[k].set_color(基["颜色"][t])
                    ax2.get_yticklabels()[k].set_fontweight("bold")
        else:
            ax2.set_yticks([])
        ax2.tick_params(length=1.4, pad=1.2)
        ax2.set_frame_on(False)
        if r == n行 - 1:
            ax2.set_xlabel("cluster id", fontsize=6.3, labelpad=1.5)
        if r == 0:
            面板号(ax2, "a", x=-0.215, y=1.50)
            ax2.text(-0.16, 1.50,
                     "Where each ground-truth domain went (row-normalised)",
                     transform=ax2.transAxes, fontsize=7.0, va="bottom",
                     ha="left", color=INK)

    if im is not None:
        cax = fig.add_axes([0.058, 0.30, 0.009, 0.26])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("share of a domain in that cluster", fontsize=5.9,
                     labelpad=3)
        cb.ax.yaxis.set_label_position("left")
        cb.ax.yaxis.set_ticks_position("left")
        cb.ax.tick_params(labelsize=5.7, length=1.4)
        cb.set_ticks([0, 0.5, 1.0])
        cb.outline.set_linewidth(0.5)

    历史图标记(fig)
    files = 存图(fig, out, "F2_各arm空间分区_seed0")
    return 历史图说明("F2", files, [rid for _, rid, _ in 可用], 基["换算说明"])


def figure_6(基: dict, out: Path) -> dict | None:
    """F6 —— 被 refine 的表示里到底有没有组织域结构。

    这一张必须是散点图版，不能是柱状图。柱状图只能告诉你 kNN 纯度是 0.233 还是 0.601，
    读者还是不知道「没有结构」长什么样；把嵌入空间投到 2D、按真值上色，一眼就能看出
    warm-up 那两格是一团糊，而表达空间和有监督那两格是分开的岛。
    稀有类画在最上层并加大点径，否则 280 个 spot 会被 3204 个盖掉。
    """
    import scanpy as sc

    truth = 基["truth"]
    codes = truth.astype("category").cat.codes.to_numpy()
    颜色 = 基["颜色"]
    是稀有 = truth.isin(基["rare"]).to_numpy()

    候选 = [
        ("Upstream expression space\n(PC1-40, pseudo-label source)", None),
        ("Historical preflight\n(30 ep L_KL + 1 ep)", "stage2_C_t0.08_d0.8_seed0_preflight"),
        ("Warm-up 100 ep\n(L_KL only)", "stage2_C_t0.2_dq0.2_seed0_warmuponly"),
        ("Stage 1 arm A, 100 ep\n(fixed pseudo-label loss)", "stage1_A_seed0"),
        ("Historical C, 50 ep\n(dynamic, warm-up init)",
         "stage2_C_t0.2_dq0.2_seed0_v2b_preflight"),
        ("Historical C, 40 ep\n(dynamic, A0-label init)",
         "stage2_C_t0.2_dq0.2_seed0_initA0_v3a_probe"),
    ]
    面板: list[tuple[str, np.ndarray]] = []
    for 名, rid in 候选:
        if rid is None:
            面板.append((名, 上游表达空间(sc.read_h5ad(H5AD))))
            continue
        p = RUNS / rid / "cell_embedding.npy"
        if p.is_file():
            面板.append((名, np.load(p).astype(np.float32)))
    if len(面板) < 2:
        return None

    统计 = []
    for 名, M in 面板:
        purity, chance = kNN纯度(M, codes)
        统计.append((名, purity, chance))

    列 = 3
    行 = int(np.ceil(len(面板) / 列))
    fig = plt.figure(figsize=(W2, 1.42 * 行 + 1.45))
    gs = fig.add_gridspec(行 + 1, 列, height_ratios=[1.0] * 行 + [0.86],
                          hspace=0.62, wspace=0.10,
                          left=0.045, right=0.985, top=0.895, bottom=0.075)
    字母 = "abcdefgh"

    for i, (名, M) in enumerate(面板):
        ax = fig.add_subplot(gs[i // 列, i % 列])
        cache_name = {
            "Historical preflight\n(30 ep L_KL + 1 ep)": "Warm-up 30 ep\n(L_KL only)",
            "Historical C, 50 ep\n(dynamic, warm-up init)": "Stage 2 C, 50 ep\n(dynamic, warm-up init)",
            "Historical C, 40 ep\n(dynamic, A0-label init)": "Stage 2 C, 40 ep\n(dynamic, A0-label init)",
        }.get(名, 名)
        xy = 二维投影(cache_name.replace("\n", " "), M)
        ax.scatter(xy[~是稀有, 0], xy[~是稀有, 1],
                   c=[颜色[t] for t in truth[~是稀有]], s=1.5, linewidths=0,
                   alpha=0.75, rasterized=True)
        ax.scatter(xy[是稀有, 0], xy[是稀有, 1],
                   c=[颜色[t] for t in truth[是稀有]], s=5.0, linewidths=0,
                   alpha=0.98, rasterized=True, zorder=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(名, fontsize=6.5, pad=2.5, color=INK, linespacing=1.35)
        purity, chance = 统计[i][1], 统计[i][2]
        超出 = purity - chance
        # 数字写在格子下面。写在格子里一定会压到点云上（上一版六格里压了四格）。
        ax.text(0.5, -0.015,
                f"kNN label purity {purity:.3f}   chance {chance:.3f}"
                f"   →  {超出:+.3f}",
                transform=ax.transAxes, fontsize=5.9, va="top", ha="center",
                color=SMALL_CLUSTER if 超出 < 0.05 else "#2E6E3A")
        面板号(ax, 字母[i], x=-0.03, y=1.24)

    # 这条是「超出随机多少」的点线带（forest 那种读法），不是柱状图：
    # 只有一个量、还要和一条参考线比，点线比条形容易读出差距。
    axq = fig.add_subplot(gs[行, :])
    名字 = [s[0].split("\n")[0] for s in 统计]
    超出值 = [s[1] - s[2] for s in 统计]
    y = np.arange(len(名字))[::-1]
    for yi, v in zip(y, 超出值):
        c = SMALL_CLUSTER if v < 0.05 else "#2E6E3A"
        axq.plot([0, v], [yi, yi], color=c, lw=1.0, solid_capstyle="butt")
        axq.plot(v, yi, "o", ms=3.4, color=c)
        axq.text(v + 0.008, yi, f"{v:+.3f}", fontsize=5.9, va="center",
                 ha="left", color=c)
    axq.axvline(0, color=RULE, lw=0.7)
    axq.set_yticks(y)
    axq.set_yticklabels(名字, fontsize=6.0)
    axq.set_xlim(-0.02, max(超出值) * 1.30 + 0.02)
    axq.set_xlabel("kNN label purity above chance", fontsize=6.4, labelpad=1.5)
    axq.tick_params(length=1.6, pad=1.5)
    面板号(axq, 字母[len(面板)], x=-0.042, y=1.04)
    axq.text(1.0, 1.04,
             "green: purity above chance by ≥ 0.05; red: smaller difference (descriptive only)",
             transform=axq.transAxes, fontsize=5.9, va="bottom", ha="right",
             color=RULE)

    历史图标记(fig)
    files = 存图(fig, out, "F6_表示空间结构")
    return 历史图说明("F6", files, [rid for _, rid in 候选 if rid is not None])


def 缎带(ax, x0: float, x1: float, y0a: float, y0b: float,
       y1a: float, y1b: float, color: str, alpha: float = 0.62,
       n: int = 64) -> None:
    """一条从左堆栈流到右堆栈的缎带。

    用 smoothstep 而不是直线：直线交叉多了以后完全看不清哪条通向哪里，
    S 形在起止两端是水平的，眼睛能顺着走。
    """
    t = np.linspace(0.0, 1.0, n)
    s = t * t * (3.0 - 2.0 * t)
    x = x0 + (x1 - x0) * t
    上 = y0a + (y1a - y0a) * s
    下 = y0b + (y1b - y0b) * s
    ax.fill_between(x, 下, 上, color=color, alpha=alpha, linewidth=0,
                    zorder=2)


def 流向面板(ax, pred: np.ndarray, truth, 顺序: list[str], 颜色: dict,
          rare: list[str], 标题: str) -> None:
    """左边真值域、右边预测簇，缎带宽度就是 spot 数。"""
    n_total = len(pred)
    簇 = sorted(int(v) for v in np.unique(pred))
    sizes = np.bincount(pred, minlength=max(簇) + 1)
    间隙 = 0.012
    左高 = {t: float((truth == t).sum()) / n_total for t in 顺序}
    右高 = {c: float(sizes[c]) / n_total for c in 簇}

    左顶 = {}
    y = 1.0
    for t in 顺序:
        左顶[t] = y
        y -= 左高[t] + 间隙
    右顶 = {}
    y = 1.0
    for c in 簇:
        右顶[c] = y
        y -= 右高[c] + 间隙

    x0, x1 = 0.10, 0.90
    宽 = 0.055
    # 先画丰度域的缎带，稀有域最后画，保证稀有的压在上面看得见。
    for 是稀有阶段 in (False, True):
        for t in 顺序:
            if (t in rare) != 是稀有阶段:
                continue
            m = (truth == t).to_numpy()
            上 = 左顶[t]
            for c in 簇:
                k = float((pred[m] == c).sum()) / n_total
                if k <= 0:
                    continue
                占 = float((pred[m] == c).sum()) / max(int(m.sum()), 1)
                r上 = 右顶[c]
                # 右侧同一簇内按真值顺序依次堆叠
                右偏 = sum(float((pred[(truth == s).to_numpy()] == c).sum()) / n_total
                          for s in 顺序[:顺序.index(t)])
                缎带(ax, x0 + 宽, x1, 上, 上 - k, r上 - 右偏, r上 - 右偏 - k,
                    颜色[t], alpha=0.80 if t in rare else 0.34)
                上 -= k
    # 小域的条太薄，标签会互相压。先算想放的位置，再自下而上顶开到最小间距，
    # 然后用一根细引线连回条上——直接叠着写是不可读的。
    最小间距 = 0.043
    调整: dict[str, float] = {}
    上一 = None
    for t in reversed(顺序):
        y = 左顶[t] - 左高[t] / 2
        if 上一 is not None and y < 上一 + 最小间距:
            y = 上一 + 最小间距
        调整[t] = y
        上一 = y
    for t in 顺序:
        ax.add_patch(plt.Rectangle((x0, 左顶[t] - 左高[t]), 宽, 左高[t],
                                   facecolor=颜色[t], edgecolor="none", zorder=4))
        真y = 左顶[t] - 左高[t] / 2
        标y = 调整[t]
        c = 颜色[t] if t in rare else "#3A3A3A"
        if abs(标y - 真y) > 1e-4:
            ax.plot([x0 - 0.012, x0], [标y, 真y], color=c, lw=0.4, zorder=5,
                    clip_on=False)
        ax.text(x0 - 0.018, 标y,
                短名.get(t, t) + f"  {int((truth == t).sum())}",
                fontsize=5.7, ha="right", va="center", color=c,
                fontweight="bold" if t in rare else "normal")
    for c in 簇:
        小 = 右高[c] < PRED_THR
        ax.add_patch(plt.Rectangle((x1, 右顶[c] - 右高[c]), 宽, 右高[c],
                                   facecolor=SMALL_CLUSTER if 小 else "#8F8F8F",
                                   edgecolor="none", zorder=4))
        ax.text(x1 + 宽 + 0.015, 右顶[c] - 右高[c] / 2,
                f"cluster {c}   {int(sizes[c])}"
                + ("  < 4.5%" if 小 else ""),
                fontsize=5.7, ha="left", va="center",
                color=SMALL_CLUSTER if 小 else "#3A3A3A",
                fontweight="bold" if 小 else "normal")
    ax.set_xlim(-0.34, 1.38)
    底部 = min(左顶[顺序[-1]] - 左高[顺序[-1]],
             右顶[簇[-1]] - 右高[簇[-1]],
             min(调整.values()))
    ax.set_ylim(底部 - 0.035, 1.04)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(标题, fontsize=6.8, pad=3.0, color=INK)


def figure_3(基: dict, out: Path, arms) -> dict | None:
    """F3 —— 稀有域到底流向了哪个簇。

    原来这里是「逐稀有类召回」的柱状图，只能读出一个 0。流向图能同时给出三件事：
    每个稀有域有多大、它整块进了谁的簇、那个簇里还混了谁。
    """
    truth = 基["truth"]
    顺序 = list(基["share"].sort_values(ascending=False).index)
    可用 = [(名, 载入预测(rid)) for 名, rid in arms]
    可用 = [(名, p) for 名, p in 可用 if p is not None]
    if not 可用:
        return None

    fig = plt.figure(figsize=(W2, 2.55))
    gs = fig.add_gridspec(1, len(可用), wspace=0.34,
                          left=0.02, right=0.98, top=0.86, bottom=0.10)
    for i, (名, pred) in enumerate(可用):
        ax = fig.add_subplot(gs[0, i])
        流向面板(ax, pred, truth, 顺序, 基["颜色"], 基["rare"], 名)
        面板号(ax, "abcdef"[i], x=0.0, y=1.02)

    历史图标记(fig)
    files = 存图(fig, out, "F3_稀有域流向")
    return 历史图说明("F3", files, [rid for _, rid in arms])


def figure_7(基: dict, out: Path, runs) -> dict | None:
    """F7: show observed refresh snapshots without interpolating between them."""
    面板 = []
    for 名, rid in runs:
        d = (读json(RUNS / rid / "metrics.json").get("dynamic") or {})
        trace = d.get("refresh_trace") or []
        if trace:
            面板.append((名, d, trace))
    if not 面板:
        return None

    fig = plt.figure(figsize=(W2, 2.65))
    gs = fig.add_gridspec(1, len(面板), wspace=0.48,
                          left=0.065, right=0.90, top=0.81, bottom=0.18)
    for i, (名, d, trace) in enumerate(面板):
        ax = fig.add_subplot(gs[0, i])
        K = max(len(t.get("cluster_sizes") or []) for t in trace)
        x = np.arange(1, len(trace) + 1)
        份 = np.zeros((K, len(trace)))
        for j, t in enumerate(trace):
            s = np.asarray(t.get("cluster_sizes") or [], dtype=float)
            if s.size < K:
                s = np.concatenate([s, np.zeros(K - s.size)])
            总 = s.sum() if s.sum() > 0 else 1.0
            份[:, j] = s / 总
        conf = np.asarray([t["confident_fraction"] for t in trace], dtype=float)
        observed = np.vstack([份, conf]) * 100
        if not np.isfinite(observed).all() or (observed < 0).any() or (observed > 100.001).any():
            raise ValueError("Invalid recorded cluster/confidence proportions")
        im = ax.imshow(observed, cmap="Blues", vmin=0, vmax=100,
                       interpolation="nearest", aspect="auto")
        for row in range(K + 1):
            for col in range(len(trace)):
                value = observed[row, col]
                label = "0" if value == 0 else ("<0.1" if value < 0.1 else f"{value:.1f}")
                ax.text(col, row, label, ha="center", va="center", fontsize=6.5,
                        color="white" if value > 55 else INK)
        ax.axhline(K - 0.5, color=INK, lw=0.8)
        ax.set_yticks(range(K + 1), [f"Slot {k}" for k in range(K)] + ["Conf."])
        ax.set_xticks(range(len(trace)), x)
        ax.tick_params(length=0)
        ax.set_xlabel("Observed refresh", fontsize=6.5, labelpad=3)
        ax.set_title(名, fontsize=6.8, pad=3.0, color=INK)
        面板号(ax, "abcdef"[i], x=-0.10, y=1.02)

    cax = fig.add_axes([0.925, 0.24, 0.014, 0.48])
    fig.colorbar(im, cax=cax, ticks=[0, 50, 100]).set_label("Spots (%)", fontsize=6.5)
    fig.text(0.065, 0.055, "Recorded snapshots only. Conf. = spots above the confidence threshold; slot IDs are local to each run.",
             fontsize=6.1, color=INK)
    历史图标记(fig)
    files = 存图(fig, out, "F7_机制门refresh轨迹")
    return 历史图说明("F7", files, [rid for _, rid in runs])


def figure_8(基: dict, out: Path) -> dict | None:
    """性能平面与逐分辨率证据并排；缺失的富集比例不编码为零。"""
    report = 读json(REPORTS / "rare_domain_baseline_frontier" / "report.json")
    rows = report.get("rows", [])
    if not rows:
        return None
    baseline = report["baseline_rare_share"] * 100
    fig = plt.figure(figsize=(W2, 3.75))
    ax = fig.add_axes([0.075, 0.19, 0.405, 0.69])
    size_ax = fig.add_axes([0.595, 0.19, 0.15, 0.69])
    enrich_ax = fig.add_axes([0.805, 0.19, 0.17, 0.69])
    for r in rows:
        color = SMALL_CLUSTER if r["n_small_clusters"] else "#9A9A9A"
        ax.scatter(r["ari"], r["rare_f1"], s=25, color=color,
                   edgecolor="white", linewidth=0.5, zorder=3)
    labels = {0.2: (0.135, 0.423), 0.8: (0.275, 0.452),
              2.0: (0.19, 0.315), 0.5: (0.27, 0.16)}
    for r in rows:
        if r["resolution"] in labels:
            ax.annotate(f"res {r['resolution']:g}, K={r['k']}",
                        (r["ari"], r["rare_f1"]), xytext=labels[r["resolution"]],
                        fontsize=6.3, arrowprops=dict(arrowstyle="-", lw=0.5, color=RULE))
    models = [("A0", "stage0_A0_seed0", "s"),
              ("A", "stage1_A_seed0", "o"),
              ("B", "stage1_B_k10_seed0", "^"),
              ("C: warm-up init", "stage2_C_t0.2_dq0.2_seed0_v2b_preflight", "D"),
              ("C: A0-label init", "stage2_C_t0.2_dq0.2_seed0_initA0_v3a_probe", "*")]
    model_rows = []
    for name, rid, marker in models:
        m = 读json(RUNS / rid / "metrics.json")["scformer"]
        ax.scatter(m["ari"], m["rare_f1"], marker=marker, s=35,
                   facecolor="none", edgecolor=UNLOCK, linewidth=0.9, zorder=5)
        model_rows.append({"label": name, "run_id": rid, "ari": m["ari"], "rare_f1": m["rare_f1"]})
        if name == "A":
            pos, label = (0.27, 0.075), "A0 / A / B"
        elif name.startswith("C:"):
            pos = (0.008, 0.115) if "warm-up" in name else (0.115, 0.058)
            label = name
        else:
            continue
        ax.annotate(label, (m["ari"], m["rare_f1"]), xytext=pos,
                    fontsize=6.0, color=UNLOCK,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#7296BB"))
    ax.set(xlim=(-0.015, 0.385), ylim=(-0.02, 0.485),
           xlabel="ARI", ylabel="Rare-F1")
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_title("Partition accuracy", loc="left", fontsize=7.5, pad=9)
    ax.text(0, -0.20, "Filled: Leiden  |  Open: historical model runs\nRed: at least one small cluster; grey: none",
            transform=ax.transAxes, fontsize=6.2, va="top", linespacing=1.5)

    y = np.arange(len(rows))
    for a in (size_ax, enrich_ax):
        a.set_ylim(len(rows) - 0.4, -0.6)
        a.set_yticks(y)
        a.tick_params(axis="y", length=0)
        a.grid(axis="y", color="#E7E7E7", linewidth=0.45, zorder=0)
        a.spines["left"].set_visible(False)
    size_ax.set_yticklabels([f"{r['resolution']:g}" for r in rows], fontsize=6.3)
    enrich_ax.set_yticklabels([])
    size_ax.set_ylabel("Leiden resolution", fontsize=6.7)
    size_ax.axvline(100 * PRED_THR, ls=(0, (3, 2)), color=SMALL_CLUSTER, lw=0.7)
    enrich_ax.axvline(baseline, ls=(0, (3, 2)), color=RULE, lw=0.7)
    for i, r in enumerate(rows):
        small = r["n_small_clusters"] > 0
        color = SMALL_CLUSTER if small else "#9A9A9A"
        size_ax.scatter(100 * r["min_cluster_share"], i, s=19, color=color, zorder=3)
        if r["small_rare_share"] is None:
            enrich_ax.text(40, i, "n/a", fontsize=6.0, va="center", ha="center", color=RULE)
        else:
            enrich_ax.scatter(100 * r["small_rare_share"], i, s=19, color=SMALL_CLUSTER, zorder=3)
    size_ax.set_xscale("log")
    size_ax.set_xlim(0.2, 140)
    size_ax.set_xticks([1, 10, 100], ["1", "10", "100"])
    size_ax.xaxis.set_minor_formatter(NullFormatter())
    enrich_ax.set_xlim(-3, 83)
    enrich_ax.set_xticks([0, 40, 80])
    size_ax.set_xlabel("Smallest cluster (%)", fontsize=6.5)
    enrich_ax.set_xlabel("Rare share (%)", fontsize=6.5)
    size_ax.set_title("Cluster size", loc="left", fontsize=7.5, pad=9)
    enrich_ax.set_title("Small-cluster content", loc="left", fontsize=7.5, pad=9)
    size_ax.text(0, -0.20, f"Dashed: {100 * PRED_THR:g}% cut", transform=size_ax.transAxes, fontsize=6.2, va="top")
    enrich_ax.text(0, -0.20, f"Dashed: tissue {baseline:.1f}%\nn/a: no small cluster",
                   transform=enrich_ax.transAxes, fontsize=6.2, va="top", linespacing=1.5)
    for a, letter in zip((ax, size_ax, enrich_ax), "abc"):
        面板号(a, letter, x=-0.15 if a is ax else -0.25, y=1.04)
    files = 存图(fig, out, "F8_稀有域基线前沿")
    entry = 历史图说明("F8", files, [r["run_id"] for r in model_rows])
    entry["source_reports"] = ["experiments/reports/rare_domain_baseline_frontier/report.json"]
    entry["source_data"] = {"leiden": rows, "historical_models": model_rows}
    entry["figure_claim"] = "Higher partition accuracy does not necessarily recover rare domains; small-cluster size and composition must be assessed together."
    entry["why"] = ("这张图把性能、最小簇规模和小簇内稀有占比拆成三个共享分辨率的证据面："
                    "面板 a 看准确率与 rare-F1 的取舍，面板 b 判断是否产生了低于 4.5% 的小簇，"
                    "面板 c 再判断这些小簇是否高于全片 8.0% 稀有基线。没有小簇的分辨率显示 n/a，"
                    "避免把缺失测量误读成零富集。")
    return entry


def figure_8_legacy(基: dict, out: Path) -> dict | None:
    """F8: show the performance landscape and the structural condition behind it."""
    前沿 = 读json(REPORTS / "rare_domain_baseline_frontier" / "report.json")
    rows = 前沿.get("rows") or []
    if not rows:
        return None
    基线 = float(前沿.get("baseline_rare_share", 0.08))

    模型 = [
        ("A0", "stage0_A0_seed0", "s"),
        ("A", "stage1_A_seed0", "o"),
        ("B k=10", "stage1_B_k10_seed0", "^"),
        ("C warm-up init", "stage2_C_t0.2_dq0.2_seed0_v2b_preflight", "D"),
        ("C A0-label init", "stage2_C_t0.2_dq0.2_seed0_initA0_v3a_probe", "*"),
    ]
    模型点 = []
    for 名, rid, mk in 模型:
        sc_b = (读json(RUNS / rid / "metrics.json").get("scformer") or {})
        if sc_b:
            模型点.append((名, float(sc_b.get("ari", 0.0)),
                          float(sc_b.get("rare_f1", 0.0)), mk))

    fig = plt.figure(figsize=(W2, 2.62))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.36,
                          left=0.075, right=0.91, top=0.84, bottom=0.19)

    ax = fig.add_subplot(gs[0, 0])
    for r in rows:
        有小簇 = bool(r.get("n_small_clusters", 0))
        富集 = r.get("small_rare_share")
        c = "#C4C4C4"
        if 有小簇:
            c = SMALL_CLUSTER if (富集 is not None and 富集 > 基线) else "#E9A6A1"
        ax.scatter(r["ari"], r["rare_f1"], s=10 + 2.0 * float(r["k"]),
                   facecolor=c, edgecolor="white", linewidths=0.5, zorder=3)
    关键 = sorted(rows, key=lambda r: -r["rare_f1"])[:3]
    默认 = next((r for r in rows if abs(float(r["resolution"]) - 0.5) < 1e-9), None)
    if 默认 is not None and 默认 not in 关键:
        关键 = 关键 + [默认]
    # 手写偏移：这几个点彼此很近，自动 offset 一定压字（上一版就压成一团）。
    偏移 = {0.2: (-30, 9), 0.8: (6, 7), 2.0: (8, 1), 0.5: (-4, -20)}
    for r in 关键:
        标 = "res " + f"{float(r['resolution']):g}" + "  K=" + str(int(r["k"]))
        if 默认 is not None and r is 默认:
            标 += "\n(upstream default)"
        dx, dy = 偏移.get(round(float(r["resolution"]), 4), (6, 5))
        ax.annotate(标, (r["ari"], r["rare_f1"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=5.9, color=INK)
    # 模型点用统一蓝色描边；名称集中放在轴下方，避免标签压住相邻的 baseline 点。
    from matplotlib.lines import Line2D
    for 名, ari, rf1, mk in 模型点:
        ax.scatter(ari, rf1, marker=mk, s=28, facecolor="none",
                   edgecolor="#0F4D92", linewidths=0.9, zorder=6)
    handles = [Line2D([0], [0], marker=mk, color="#0F4D92",
                      markerfacecolor="none", linestyle="None", markersize=4.5)
               for _, _, _, mk in 模型点]
    ax.legend(handles, [名 for 名, _, _, _ in 模型点],
              loc="upper center", bbox_to_anchor=(0.48, -0.19),
              ncol=3, frameon=False, fontsize=5.6,
              handletextpad=0.4, columnspacing=0.8, borderpad=0.1)
    ax.set_xlabel("ARI against final_annot", fontsize=6.6)
    ax.set_ylabel("rare-F1 (main endpoint)", fontsize=6.6)
    ax.set_ylim(-0.03, 0.47)
    面板号(ax, "a", x=-0.10, y=1.03)
    ax.text(0.045, 1.03, "Leiden baseline and model runs",
            transform=ax.transAxes, fontsize=7.0, va="bottom", ha="left",
            color=INK)

    ax2 = fig.add_subplot(gs[0, 1])
    res = [float(r["resolution"]) for r in rows]
    最小 = [100 * float(r["min_cluster_share"]) for r in rows]
    ax2.axhspan(0.05, 4.5, color="#F6CFCB", alpha=0.55, lw=0, zorder=1)
    ax2.axhline(4.5, color=SMALL_CLUSTER, lw=0.8, ls=(0, (3, 2)), zorder=2)
    enrichment = [100 * float(r["small_rare_share"]) if r.get("small_rare_share") is not None else 0
                  for r in rows]
    points = ax2.scatter(res, 最小, c=enrichment, cmap="YlOrRd", vmin=0, vmax=max(80, max(enrichment)),
                         s=22, edgecolor="white", linewidths=0.5, zorder=3)
    for r, yv in zip(rows, 最小):
        if abs(float(r["resolution"]) - 0.5) < 1e-9 or abs(float(r["resolution"]) - 0.8) < 1e-9:
            offset = (4, 10) if abs(float(r["resolution"]) - 0.5) > 1e-9 else (8, -30)
            ax2.annotate(f"{float(r['resolution']):g}\nK={int(r['k'])}",
                         (float(r["resolution"]), yv), textcoords="offset points",
                         xytext=offset, fontsize=5.8, color=INK)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Leiden resolution", fontsize=6.6)
    ax2.set_ylabel("smallest cluster (% of tissue)", fontsize=6.6)
    ax2.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    ax2.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
    # log 轴默认用 mathtext 画 10^n，指数被缩到母字号的 0.7 倍：6.5 pt 的刻度里
    # 就藏着 4.55 pt 的字，audit_pdf_text.py 的 5 pt 地板会 FAIL。这里换成普通十进制。
    ax2.set_yticks([0.1, 1.0, 10.0, 100.0])
    ax2.set_yticklabels(["0.1", "1", "10", "100"])
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.text(0.97, 0.06, "shaded: below 4.5% cut",
             transform=ax2.transAxes, fontsize=5.7, ha="right", va="bottom",
             color="#8A3A38")
    面板号(ax2, "b", x=-0.24, y=1.03)
    ax2.text(0.06, 1.03, "Small-cluster size",
             transform=ax2.transAxes, fontsize=7.0, va="bottom", ha="left",
             color=INK)

    cbar = fig.add_axes([0.925, 0.26, 0.014, 0.45])
    cb = fig.colorbar(points, cax=cbar, ticks=[0, 25, 50, 75])
    cb.set_label("rare share in small cluster (%)", fontsize=6.0)
    cbar.tick_params(labelsize=5.7, length=1.8)
    ax2.text(0.04, 0.96, f"colour: tissue baseline {100 * 基线:.1f}%",
             transform=ax2.transAxes, fontsize=5.8, va="top", ha="left", color=INK)

    历史图标记(fig)
    files = 存图(fig, out, "F8_稀有域基线前沿")
    entry = 历史图说明("F8", files, [rid for _, rid, _ in 模型])
    entry["source_reports"] = ["experiments/reports/rare_domain_baseline_frontier/report.json"]
    return entry


# F4 的五个 arm。名字直接当行标写在图上，所以要短到能塞进左边距，全称留给图注。
# C 的两支不是超参微调，是两种 prototype 初始化，必须分成两行：合成一行会把
# ARI 0.009 和 0.328 平均成 0.17，那个数谁都不是。
F4_ARMS = [
    ("A0  upstream", "stage0_A0_seed{seed}", LOCKED),
    ("A  spatial batching", "stage1_A_seed{seed}", LOCKED),
    ("B  + spatial k=10", "stage1_B_k10_seed{seed}", LOCKED),
    ("C  warm-up init", "stage2_C_t0.2_dq0.2_seed{seed}_v2b_preflight", UNLOCK),
    ("C  A0-label init", "stage2_C_t0.2_dq0.2_seed{seed}_initA0_v3a_probe", UNLOCK2),
]


def 逐seed指标(模板: str, key: str) -> dict[int, float]:
    """一个 arm 在各 seed 上的指标，缺的 seed 不进字典。

    键留成 seed 号而不是列表：后面要按 seed 配对相减，位置对齐靠列表下标迟早对错。
    """
    out: dict[int, float] = {}
    for seed in SEEDS:
        sc = 读json(RUNS / 模板.format(seed=seed) / "metrics.json").get("scformer") or {}
        v = sc.get(key)
        if v is not None:
            out[seed] = float(v)
    return out


def 配对差(后: dict[int, float], 前: dict[int, float]) -> dict[int, float]:
    """同 seed 相减。边缘分布会把 seed 之间的抖动混进效应里，配对差才是效应。"""
    return {s: 后[s] - 前[s] for s in sorted(set(后) & set(前))}


def figure_4(out: Path) -> dict | None:
    """F4: one claim, namely the paired effect of adding the spatial relation."""
    metrics = [("ARI", "ari", 0.001), ("NMI", "nmi", 0.001),
               ("rare-F1", "rare_f1", 0.005)]
    pairs = []
    for label, key, floor in metrics:
        control = 逐seed指标("stage1_A_seed{seed}", key)
        treated = 逐seed指标("stage1_B_k10_seed{seed}", key)
        diff = 配对差(treated, control)
        if diff:
            pairs.append((label, key, diff, floor))
    if not pairs:
        return None

    fig = plt.figure(figsize=(W2, 2.22))
    gs = fig.add_gridspec(1, len(pairs), wspace=0.42,
                          left=0.09, right=0.98, top=0.78, bottom=0.25)
    for i, (label, key, diff, floor) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, i])
        values = np.asarray([diff[s] for s in SEEDS if s in diff], dtype=float)
        scale = max(float(np.max(np.abs(values))), floor)
        limit = 1.45 * scale
        ax.axvline(0.0, color=INK, lw=0.75, zorder=1)
        for row, seed in enumerate([s for s in SEEDS if s in diff]):
            ax.scatter(diff[seed], row, marker=SEED_MARK[seed], s=34,
                       facecolor="white", edgecolor=UNLOCK, linewidths=0.9, zorder=3)
        mean_v = float(values.mean())
        sd_v = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ax.errorbar(mean_v, len(values) + 0.28, xerr=sd_v, fmt="D",
                    color=INK, markerfacecolor=INK, markersize=3.5,
                    capsize=2.0, elinewidth=0.8, zorder=4)
        ax.text(0.5, 1.02, f"mean {mean_v:+.4f} ± {sd_v:.4f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=6.0,
                color=INK)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-0.55, len(values) + 0.72)
        ax.set_yticks(range(len(values) + 1),
                      [f"seed {s}" for s in SEEDS if s in diff] + ["mean ± SD"])
        ax.set_xlabel("B − A", fontsize=6.6, labelpad=2)
        ax.set_title(label, fontsize=7.2, loc="left", pad=5, color=INK)
        ax.tick_params(axis="y", length=0, labelsize=6.0)
        ax.tick_params(axis="x", labelsize=5.8)
        ax.grid(axis="x", color="#E3E3E3", lw=0.45, zorder=0)
        if i == 0:
            ax.set_ylabel("same-seed paired comparison", fontsize=6.4)
        else:
            ax.set_yticklabels([])
    fig.text(0.09, 0.07,
             "Open symbols: seeds 0, 1 and 42. The filled diamond is the mean with sample SD; zero is no change.",
             fontsize=6.1, color=INK)
    历史图标记(fig)
    files = 存图(fig, out, "F4_跨seed配对指标")
    entry = 历史图说明("F4", files, 历史图来源("F4"))
    entry["figure_claim"] = "Under the public-code baseline, adding the unweighted spatial relation changes the three metrics only marginally and inconsistently across seeds."
    entry["main_panel"] = "B-A paired differences for ARI, NMI and rare-F1; each seed is paired with its same-seed control."
    return entry


def figure_4_legacy(out: Path) -> dict | None:
    """F4 —— 差值算不算效应。

    横排点图：arm 的名字本身就是自变量（"+ spatial k=10"），竖排放不下就得缩成
    A / B / C 再配一个图例，读者要来回对照。横排把名字写在行首，五行一眼读完。

    上下两块分工不同，不是同一件事画两遍。上排是绝对水平，回答「各 arm 落在哪个
    量级」；ARI 轴要装下 C 的 0.009 和 A 的 0.330，一个 1e-3 的差在这条轴上只有
    0.07 mm，看不见。下排两格是同 seed 配对差，各自用自己的量程，门禁结论只能在
    那里读出来。
    """
    水平 = [(名, 逐seed指标(模板, "ari"), 逐seed指标(模板, "nmi"),
            逐seed指标(模板, "rare_f1"), 色) for 名, 模板, 色 in F4_ARMS]
    if not any(行[1] for 行 in 水平):
        return None
    # 固定伪标签自己对真值的分数：A / B 的输出基本就是它，所以它是这三支的锚，
    # 不是「基线对照」。数字从 metrics.json 里的 leiden_baseline 取，不另算。
    锚 = 读json(RUNS / "stage1_A_seed0" / "metrics.json").get("leiden_baseline") or {}

    fig = plt.figure(figsize=(W2, 3.15))
    gs1 = fig.add_gridspec(1, 3, wspace=0.20, left=0.145, right=0.988,
                           top=0.935, bottom=0.585)
    gs2 = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.92], wspace=0.30,
                           left=0.058, right=0.988, top=0.415, bottom=0.115)

    def 画水平(ax, 取值, 参考, 轴名: str, xlim) -> None:
        if 参考 is not None:
            ax.axvline(参考, color=RULE, lw=0.7, ls=(0, (3, 2)), zorder=1)
            # 说明文字放在左半边的空档里，不贴轴底：贴轴底会压到 x 刻度标签。
            ax.text(xlim[0] + 0.045 * (xlim[1] - xlim[0]), 1.62,
                    f"dashed = fixed\npseudo-label {参考:.3f}", fontsize=5.6,
                    color=RULE, ha="left", va="center", linespacing=1.4)
        for i, 行 in enumerate(水平):
            名, 色 = 行[0], 行[4]
            vals = 取值(行)
            y = len(水平) - 1 - i
            if not vals:
                ax.text(xlim[0] + 0.02 * (xlim[1] - xlim[0]), y, "not run",
                        fontsize=5.6, color=RULE, va="center")
                continue
            lo, hi = min(vals.values()), max(vals.values())
            if hi > lo:
                ax.plot([lo, hi], [y, y], color=色, lw=1.6, alpha=0.28,
                        solid_capstyle="round", zorder=2)
            for seed, v in vals.items():
                # 空心 marker：三个 seed 重合时（A / B 就是重合的）实心会互相盖死，
                # 空心叠起来还能数出是三个点。
                ax.plot(v, y, marker=SEED_MARK[seed], ms=3.8, mfc="white",
                        mec=色, mew=0.75, zorder=4, clip_on=False)
        ax.set_yticks(range(len(水平)))
        ax.set_yticklabels([行[0] for 行 in 水平][::-1], fontsize=6.3)
        ax.set_ylim(-0.55, len(水平) - 0.45)
        ax.set_xlim(*xlim)
        ax.set_xlabel(轴名, fontsize=6.5, labelpad=1.5)
        ax.tick_params(axis="y", length=0)

    ax1 = fig.add_subplot(gs1[0, 0])
    画水平(ax1, lambda 行: 行[1], 锚.get("ari"), "ARI vs final_annot", (0.0, 0.40))
    面板号(ax1, "a", x=-0.235, y=1.04)
    ax1.text(0.0, 1.04, "Absolute level, three seeds each",
             transform=ax1.transAxes, fontsize=7.0, va="bottom", ha="left",
             color=INK)

    ax2 = fig.add_subplot(gs1[0, 1])
    画水平(ax2, lambda 行: 行[2], 锚.get("nmi"), "NMI vs final_annot", (0.0, 0.46))
    ax2.set_yticklabels([])
    面板号(ax2, "b", x=-0.075, y=1.04)

    ax3 = fig.add_subplot(gs1[0, 2])
    画水平(ax3, lambda 行: 行[3], None, "rare-F1 (main endpoint)", (-0.004, 0.068))
    ax3.set_yticklabels([])
    ax3.axvline(0.0, color=RULE, lw=0.7, zorder=1)
    零 = sum(1 for 行 in 水平 if 行[4] is LOCKED
             for v in 行[3].values() if v == 0.0)
    if 零:
        # 带引线指到那一叠 0 上：光写在旁边会被读成某一行的行标。
        ax3.annotate(f"{零} runs\nexactly 0", xy=(0.0, 3.0),
                     xytext=(24, 16), textcoords="offset points", fontsize=5.8,
                     color=LOCKED, ha="left", va="center", linespacing=1.4,
                     arrowprops=dict(arrowstyle="-", lw=0.45, color=LOCKED,
                                     shrinkA=0.5, shrinkB=1.5))
    面板号(ax3, "c", x=-0.075, y=1.04)

    门 = [
        ("A - A0", "sampler swap", "stage1_A_seed{seed}", "stage0_A0_seed{seed}",
         0.0235, 0.02),
        ("B - A", "spatial relation", "stage1_B_k10_seed{seed}",
         "stage1_A_seed{seed}", 0.00145, None),
    ]
    for j, (名, 副, 后模板, 前模板, 半宽, 容差) in enumerate(门):
        ax = fig.add_subplot(gs2[0, j])
        if 容差 is not None:
            ax.axvspan(-容差, 容差, color="#EDF1F7", lw=0, zorder=1)
        # 零线只画到数据行上方一点，不用 axvline 通到顶：顶上要放 mean ± sd 摘要，
        # 通到顶的线会从数字中间穿过去。
        ax.plot([0.0, 0.0], [-0.62, 1.60], color=INK, lw=0.7, zorder=2)
        摘要 = []
        for r, key in enumerate(("nmi", "ari")):
            差 = 配对差(逐seed指标(后模板, key), 逐seed指标(前模板, key))
            if not 差:
                continue
            for seed, v in 差.items():
                ax.plot(v, r, marker=SEED_MARK[seed], ms=4.0, mfc="white",
                        mec=INK, mew=0.8, zorder=4, clip_on=False)
            arr = np.asarray(list(差.values()), dtype=float)
            符号 = "".join("+" if v > 0 else "-" for v in 差.values())
            摘要.append((key.upper(), float(arr.mean()),
                        float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 符号))
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["NMI", "ARI"], fontsize=6.3)
        # 上沿留到 2.3：两行 mean ± sd 摘要要放在面板里面。上一版放在 1.42 / 1.12，
        # 那已经在面板外面，直接压到了上一排的 x 轴标签。
        ax.set_ylim(-0.62, 2.30)
        ax.set_xlim(-半宽, 半宽)
        ax.set_xlabel("paired difference, same seed", fontsize=6.5, labelpad=1.5)
        ax.tick_params(axis="y", length=0)
        for r, (标, m, s, 符号) in enumerate(摘要[::-1]):
            ax.text(0.985, 0.965 - 0.105 * r,
                    f"{标} {m:+.4f} ± {s:.4f}   signs {符号}",
                    transform=ax.transAxes, fontsize=5.6, ha="right",
                    va="center", color=INK)
        if 容差 is not None:
            ax.text(0.5, 0.035, f"shaded = ±{容差:g} drift tolerance",
                    transform=ax.transAxes, fontsize=5.6, ha="center",
                    va="bottom", color="#5C7196")
        面板号(ax, "de"[j], x=-0.115, y=1.05)
        ax.text(0.0, 1.05, 名 + "   " + 副, transform=ax.transAxes,
                fontsize=7.0, va="bottom", ha="left", color=INK)

    # 第三格不放第六张图，放读图必需的四件事：seed 怎么认、误差条和 ± 各是什么、
    # A / B 为什么动不了（输出对伪标签的 ARI 就在 0.995 以上）、主终点上 0 的分布。
    # 「散布的定义要写在图上，不许让读者猜」是交付预检的硬条目。
    ax6 = fig.add_subplot(gs2[0, 2])
    ax6.axis("off")
    ax6.text(0.0, 1.02, "How to read", transform=ax6.transAxes, fontsize=7.0,
             va="top", ha="left", color=INK)
    # 三个 seed 横排：格子只有 0.95 英寸高，竖排 11 行放不下。间距按最长的
    # "seed 42" 留，上一版按 0.135 排，字直接压到下一个 marker 上。
    for i, (seed, mk) in enumerate(SEED_MARK.items()):
        ax6.plot(0.02 + 0.34 * i, 0.885, marker=mk, ms=3.8, mfc="white",
                 mec=INK, mew=0.75, transform=ax6.transAxes, clip_on=False)
        ax6.text(0.065 + 0.34 * i, 0.885, f"seed {seed}",
                 transform=ax6.transAxes, fontsize=5.9, va="center", ha="left",
                 color=INK)
    锁 = {}
    for 名, 模板, 色 in F4_ARMS:
        v = 逐seed指标(模板, "ari_vs_pseudo")
        if v:
            锁.setdefault("C" if 色 is not LOCKED else "A / B", []).extend(v.values())
    总, 正, 最高 = 0, 0, 0.0
    for 名, 模板, 色 in F4_ARMS:
        if 色 is LOCKED:
            continue
        v = 逐seed指标(模板, "rare_f1")
        总 += len(v)
        正 += sum(1 for x in v.values() if x > 0)
        最高 = max([最高] + list(v.values()))
    # 每行都短：这一格只有 1.9 英寸宽，把两段拼成一行会溢到图外，tight bbox 一裁
    # 整张图就跟着变宽，报告里的字号也跟着漂。
    行 = ["bar = min to max", "± = SD over the three seeds"]
    for 组 in ("A / B", "C"):
        if 组 in 锁:
            if 组 == "A / B":
                行.append("ARI vs the fixed pseudo-label")
            行.append(f"    {组}  {min(锁[组]):.3f} - {max(锁[组]):.3f}")
    if 总:
        行 += [f"rare-F1 above 0: C  {正} of {总} runs",
               f"    highest {最高:.3f}"]
    ax6.text(0.0, 0.775, "\n".join(行), transform=ax6.transAxes, fontsize=5.9,
             va="top", ha="left", color=INK, linespacing=1.62)

    历史图标记(fig)
    files = 存图(fig, out, "F4_跨seed配对指标")
    return 历史图说明("F4", files, 历史图来源("F4"))


def 配窗(ax, fig, 界: tuple[float, float, float, float]) -> tuple[float, float]:
    """把数据窗口撑到和面板框一样的长宽比，并返回 (x 跨度, 每数据单位的点数)。

    ``set_aspect('equal')`` 在窗口和面板框比例不一致时会缩面板，缩完再按面板框算
    散点直径就偏小，spot 之间会漏出白缝，切片看着像散沙。先把窗口配到框的比例，
    equal 就成了空操作，换算也才对得上。
    """
    x0, x1, y0, y1 = 界
    框 = ax.get_position()
    w = 框.width * fig.get_figwidth()
    h = 框.height * fig.get_figheight()
    want = h / w
    have = (y1 - y0) / (x1 - x0)
    if have < want:
        pad = ((x1 - x0) * want - (y1 - y0)) / 2
        y0, y1 = y0 - pad, y1 + pad
    else:
        pad = ((y1 - y0) / want - (x1 - x0)) / 2
        x0, x1 = x0 - pad, x1 + pad
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # 组织图一律 y 朝下，和 F1 / F2 同向
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return x1 - x0, w * 72.0 / (x1 - x0)


def figure_5(基: dict, out: Path) -> dict | None:
    """F5 —— 空间边进不进得了 batch。

    需要这张图的理由：一条 spot-spot 边只有两个端点落在同一个 batch 里才进得了
    message passing。上游采样器先 shuffle 再切，batch 里只剩 0.8% 的空间边——
    真按那个采样器跑，arm B 等于没加空间关系，Stage 1 的任何结论都不成立。

    上排是机制，下排才是数字。0.8% 这个数单独摆出来读者没有画面：要看到「一个
    batch 的 30 个 spot 撒在整张切片上」才知道边为什么会全被切断。所以三格用同一个
    参考 spot 所在的那个 batch，把它的空间边画出来，留住的和被切掉的分色。
    """
    probe = 读json(RUNS / "stage1a_batching_probe" / "report.json")
    per_k = probe.get("per_k") or {}
    if not per_k:
        return None

    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from matplotlib.collections import LineCollection

    from spatial_scformer.graph.spatial import build_spatial_graph
    from spatial_scformer.graph.spatial_batch import add_spatial_halo, spatial_batch_order

    coords = np.asarray(基["coords"], dtype=float)
    n = coords.shape[0]
    cell = int(probe.get("cell_size", 30))
    批数 = int(probe.get("n_batches", -(-n // cell)))
    K图 = 10  # 主实验用的 k；保留率对 k 的依赖放在 d 里，地图只画一个 k
    # 建图和分批一律用原始阵列坐标：那是实验实际跑的口径。阵列坐标是各方向不等比的
    # （1 个列单位 50 µm、1 个行单位 86.6 µm），换成 µm 会换掉 kNN 的邻居集，
    # 图上的 usable / cut 计数就不再是那次实验的数了。µm 只用来画。
    um = 基["几何"]["um"]
    pitch = float(基["几何"]["pitch"])

    边 = build_spatial_graph(coords, mode="knn", k=K图).edge_index
    块 = spatial_batch_order(coords, n_batches=批数)
    带halo = add_spatial_halo(块, 边)
    # 上游采样器：RandomState(seed).choice 打乱再按 cell_size 切，和
    # scripts/stage1a_verify_spatial_batching.py 里的复现完全一致（那份产出了 probe 的数）。
    顺序 = np.random.RandomState(0).choice(n, size=n, replace=False)
    随机块 = [顺序[i * cell:(i + 1) * cell] for i in range(批数)]

    def 归属(批列) -> np.ndarray:
        表 = np.full(n, -1, dtype=np.int64)
        for bi, b in enumerate(批列):
            表[np.asarray(b, dtype=np.int64).reshape(-1)] = bi
        return 表

    随机归属, 块归属 = 归属(随机块), 归属(块)
    # 参考 spot 取离质心最近、且在两种分批下都落在满员 batch 里的那个。离质心近是为了
    # 四面都有邻居（切片边缘的块一半邻居本来就不存在）；满员是因为 3484 / 30 = 116.13，
    # 最后一个 batch 只有 4 个 spot——第一版正好命中它，图上「一个 batch 30 个 spot」
    # 变成了 4 个点，边数也从 330 掉到 41，看着像空间边本来就稀。
    参考 = -1
    for cand in np.argsort(((coords - coords.mean(axis=0)) ** 2).sum(axis=1)):
        if (len(随机块[随机归属[cand]]) == cell
                and len(块[块归属[cand]]) >= cell - 1):
            参考 = int(cand)
            break
    if 参考 < 0:
        return None

    随机核 = np.asarray(随机块[随机归属[参考]], dtype=np.int64).reshape(-1)
    空间核 = np.asarray(块[块归属[参考]], dtype=np.int64).reshape(-1)
    halo批 = next(b for b in 带halo if 参考 in set(b["core_index"]))

    def 邻边(核: set[int]) -> list[tuple[int, int]]:
        """所有至少一个端点在核里的空间边，去掉方向重复。"""
        见 = set()
        for s, t in zip(边[0].tolist(), 边[1].tolist()):
            if s in 核 or t in 核:
                见.add((min(s, t), max(s, t)))
        return sorted(见)

    def 分边(核: set[int], 成员: set[int]) -> tuple[list, list]:
        """留住 / 切断。halo 批的判据多一条「至少一端是 core」，只有 core 出梯度。"""
        留, 断 = [], []
        for s, t in 邻边(核):
            段 = [um[s], um[t]]
            同批 = s in 成员 and t in 成员
            (留 if 同批 and (s in 核 or t in 核) else 断).append(段)
        return 留, 断

    地图 = [
        ("Random batching", 随机核, 随机核, None),
        ("Median bisection", 空间核, 空间核, None),
        ("+ one-hop halo, zoom of b", np.asarray(halo批["core_index"]),
         np.asarray(halo批["cell_index"]), "zoom"),
    ]

    # 上排三格的宽高比按切片真实长宽比（1.06）配。以前按阵列坐标画，切片是细长椭圆，
    # 格子跟着又高又窄；还原几何之后再用那个格子，两边各留一条白。
    fig = plt.figure(figsize=(W15, 3.10))
    gs1 = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.30, 1.60], wspace=0.10,
                           left=0.035, right=0.988, top=0.965, bottom=0.553)
    gs2 = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.46,
                           left=0.115, right=0.975, top=0.400, bottom=0.100)

    全界 = (um[:, 0].min(), um[:, 0].max(), um[:, 1].min(), um[:, 1].max())
    # 直径按 spot 间距的倍数给，分面板设：全片图要让 30 个 core spot 在 3484 个背景点里
    # 找得到，所以 core 放到 1.7 倍；放大格反过来，spot 铺满就把边全盖住了，压到 0.62 倍
    # 才看得见连线。
    倍数 = {None: (1.25, 1.70), "zoom": (0.66, 0.72)}
    for i, (名, 核arr, 成员arr, 模式) in enumerate(地图):
        ax = fig.add_subplot(gs1[0, i])
        核 = set(int(v) for v in 核arr)
        成员 = set(int(v) for v in 成员arr)
        if 模式 == "zoom":
            pts = um[sorted(成员)]
            pad = 5.0 * pitch
            界 = (pts[:, 0].min() - pad, pts[:, 0].max() + pad,
                 pts[:, 1].min() - pad, pts[:, 1].max() + pad)
        else:
            界 = 全界
        # 极浅的蜂窝当底图。以前背景是一片灰点，空间边看着像飘在空白里；铺满之后
        # 边是画在组织上的，切断的那些也才看得出断在哪。
        画切片(ax, 基, [基元.背景瓦片] * n, 比例尺=False)
        跨度, 点每单位 = 配窗(ax, fig, 界)
        khalo, k核 = 倍数[模式]
        留, 断 = 分边(核, 成员)
        线宽 = 0.28 if 模式 != "zoom" else 0.65
        if 断:
            ax.add_collection(LineCollection(断, colors=EDGE_LOST, linewidths=线宽,
                                             zorder=2, rasterized=True))
        if 留:
            ax.add_collection(LineCollection(留, colors=UNLOCK,
                                             linewidths=线宽 * 1.5, zorder=4,
                                             rasterized=True))
        halo点 = sorted(成员 - 核)
        if halo点:
            ax.scatter(um[halo点, 0], um[halo点, 1],
                       s=max((khalo * pitch * 点每单位) ** 2, 0.6),
                       facecolor="#BFD0E4", edgecolor="white",
                       linewidths=0.25 if 模式 == "zoom" else 0.0, zorder=5)
        核列 = sorted(核)
        ax.scatter(um[核列, 0], um[核列, 1],
                   s=max((k核 * pitch * 点每单位) ** 2, 1.2),
                   facecolor=INK, edgecolor="white",
                   linewidths=0.30 if 模式 == "zoom" else 0.20, zorder=6)
        # 比例尺放在 配窗 之后：它是按最终窗口算长度的。放大格换了放大倍率，
        # 500 µm 会横占窗口的五分之一，所以那一格给 200 µm。
        基元.比例尺(ax, 基["几何"],
                长度_um=500.0 if 模式 != "zoom" else 200.0)
        # 只给计数、不给百分比。单个 batch 的分母是「碰到这个 batch 的边」，每条跨界边
        # 被两个 batch 各算一次；d 里的全局保留率分母是每条边算一次。两个数本来就不同，
        # 在图上同时摆成百分比，读者一定会当成前后矛盾。
        ax.text(0.5, -0.012, f"{len(留)} usable", transform=ax.transAxes,
                fontsize=6.0, ha="right", va="top", color=UNLOCK)
        ax.text(0.52, -0.012, f"   {len(断)} cut", transform=ax.transAxes,
                fontsize=6.0, ha="left", va="top", color=EDGE_LOST)
        面板号(ax, "abc"[i], x=-0.02 if i else -0.05, y=1.005)
        ax.text(0.135 if i == 0 else 0.085, 1.005, 名, transform=ax.transAxes,
                fontsize=6.4, va="bottom", ha="left", color=INK)
        if 模式 == "zoom":
            # 标注挂在窗口的两个空角上：halo 环在上方、core 在下方，箭头各自指回去。
            上 = min(halo点, key=lambda idx: um[idx, 1])
            ax.annotate(f"halo, {len(halo点)} cells\nmessage passing only",
                        xy=um[上], xycoords="data",
                        xytext=(0.035, 0.955), textcoords="axes fraction",
                        fontsize=5.8, color="#3F6088", linespacing=1.45,
                        ha="left", va="top",
                        arrowprops=dict(arrowstyle="-", lw=0.45, color="#8FA8C4",
                                        shrinkA=1.0, shrinkB=2.0))
            ax.annotate(f"core, {len(核)} cells\nthe only ones with gradients",
                        # 引线指到参考 spot，它在核的正中间。指最外圈的核 spot 会让引线
                        # 停在紧邻的 halo spot 上，看着像在标 halo。
                        # 文字挂右下角，并抬到比例尺上方一档：左下角让给比例尺，
                        # 贴着底边写会和「200 µm」串成一行。
                        xy=um[参考], xycoords="data",
                        xytext=(0.968, 0.135), textcoords="axes fraction",
                        fontsize=5.8, color=INK, linespacing=1.45,
                        ha="right", va="bottom",
                        arrowprops=dict(arrowstyle="-", lw=0.45, color=RULE,
                                        shrinkA=1.0, shrinkB=2.0))
        else:
            if i == 0:
                # 挂左上角：左下角让给比例尺。
                ax.text(0.02, 0.985, f"one batch of {cell},\nk = {K图}",
                        transform=ax.transAxes, fontsize=5.7, ha="left",
                        va="top", color=INK, linespacing=1.4,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec="none", alpha=0.82))
    # b 上框出 c 的取景范围：c 换了尺度，不标出来读者会以为是第三种分批方式。
    axb = fig.axes[1]
    pts = um[sorted(int(v) for v in halo批["cell_index"])]
    pad = 5.0 * pitch
    axb.add_patch(plt.Rectangle((pts[:, 0].min() - pad, pts[:, 1].min() - pad),
                               np.ptp(pts[:, 0]) + 2 * pad,
                               np.ptp(pts[:, 1]) + 2 * pad,
                               fill=False, ec=INK, lw=0.5, ls=(0, (2, 1.6)),
                               zorder=7))

    ks = sorted(per_k, key=int)
    x = np.arange(len(ks), dtype=float)
    系列 = (("random batching", "random_batching_retention_mean", "#A8A8A8", "o"),
           ("spatial bisection", "spatial_batching_retention", UNLOCK2, "s"),
           ("+ one-hop halo", "halo_retention", UNLOCK, "^"))

    ax4 = fig.add_subplot(gs2[0, 0])
    for 名, key, 色, mk in 系列:
        v = [100 * float(per_k[k][key]) for k in ks]
        ax4.plot(x, v, color=色, lw=0.9, marker=mk, ms=3.4, mfc="white",
                 mec=色, mew=0.8, zorder=3)
        # random 那条贴着轴底，右端放不下它的名字：右侧只剩不到 0.4 英寸，两行的数字标签
        # 一定会溢出到 e 的 y 轴标签上。所以它的标签居中放在自己线的上方，值收成一个区间
        # 写在第二行；另两条仍旧右端直接标注。上一版逐点标 0.83 / 0.78 / 0.80，k=15 那个
        # 数正好撞在系列名上。
        if key.startswith("random"):
            低 = min(100 * float(per_k[kk][key]) for kk in ks)
            高 = max(100 * float(per_k[kk][key]) for kk in ks)
            ax4.text(1.0, 15.0,
                     f"{名}\n{低:.2f} - {高:.2f}%, mean of 3 seeds",
                     fontsize=5.9, color="#6E6E6E", ha="center", va="bottom",
                     linespacing=1.4)
        else:
            ax4.text(x[-1] + 0.13, v[-1], 名, fontsize=5.9, color=色,
                     va="center", ha="left")
    每seed = [per_k[k].get("random_batching_retention_per_seed") or {} for k in ks]
    for xi, 表 in zip(x, 每seed):
        for val in 表.values():
            ax4.plot(xi, 100 * float(val), marker=".", ms=2.0, color="#A8A8A8",
                     zorder=4)
    # 线性轴，不用 log：0.8% 在 0-100 的轴上就该贴着轴底，这正是要传达的量级差。
    # 上一版用 log 柱，0.83% 的柱看着有 100% 那根的四成高，把灾难画成了小差距。
    ax4.set_ylim(0, 108)
    ax4.set_yticks([0, 25, 50, 75, 100])
    ax4.set_xlim(-0.30, len(ks) + 0.02)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"k = {k}" for k in ks])
    ax4.set_ylabel("spatial edges usable\ninside a batch (%)", fontsize=6.4,
                   linespacing=1.35)
    面板号(ax4, "d", x=-0.235, y=1.06)
    ax4.text(0.0, 1.06, "All 117 batches", transform=ax4.transAxes, fontsize=6.8,
             va="bottom", ha="left", color=INK)

    ax5 = fig.add_subplot(gs2[0, 1])
    核大小 = float((probe.get("spatial_batch_sizes") or {}).get("max", cell))
    ax5.axhline(核大小, color=RULE, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax5.text(-0.22, 核大小 - 3.0, f"core {核大小:.0f}", fontsize=5.8, color=RULE,
             va="top", ha="left")
    for key, 色, mk, 标, dy in (("halo_batch_size_max", UNLOCK2, "^", "max", 6.0),
                               ("halo_batch_size_mean", UNLOCK, "o", "mean", -6.5)):
        v = [float(per_k[k][key]) for k in ks]
        ax5.plot(x, v, color=色, lw=0.9, marker=mk, ms=3.4, mfc="white", mec=色,
                 mew=0.8, zorder=3)
        ax5.annotate("halo " + 标, (x[-1], v[-1]), textcoords="offset points",
                     xytext=(4.5, dy), fontsize=5.9, color=色, ha="left",
                     va="center")
    ax5.set_ylim(0, 112)
    ax5.set_xlim(-0.30, len(ks) - 0.02)
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"k = {k}" for k in ks])
    ax5.set_ylabel("cells per batch", fontsize=6.4)
    面板号(ax5, "e", x=-0.30, y=1.06)
    ax5.text(0.0, 1.06, "What 100% costs", transform=ax5.transAxes, fontsize=6.8,
             va="bottom", ha="left", color=INK)

    files = 存图(fig, out, "F5_batch内空间边保留率")
    return {"figure": "F5", "files": files,
            "why": "一条空间边只有两个端点落在同一个 batch 里才进得了 message passing，"
                   "所以这张图管的是「arm B 到底有没有空间边可传播」这个前置条件。a 里"
                   "上游采样器把一个 batch 的 30 个 spot 撒满整张切片，边几乎全断；b 换成"
                   "空间中位二分，batch 变成一块组织，多数边留在里面；c 放大看一跳 halo "
                   "补上的那圈 halo spot 会进入该 batch 的图与重构子矩阵，边就一条不丢。d 给三个 k 上"
                   "的全局保留率 0.8% → 64-74% → 100%，e 给代价：batch 从 30 个 spot 长到 "
                   "58-81 个。这一步不先做，Stage 1 量到的会是采样器，不是空间关系。"
                   "坐标口径：" + 基["换算说明"] + "；建图和分批仍在原始阵列坐标上做，"
                   "µm 只用于呈现。"}


# 这一版取代的旧文件名。旧脚本 绘制消融图表.py 里 F3 / F6 / F7 用的是别的名字，
# 不清掉的话交付目录里会同时躺着新旧两版，谁是准的说不清。
被取代 = [
    "F3_稀有类召回与簇大小_seed0",
    "F6_warmup表示质量",
]
# 八张图现在全部由本脚本生成，没有沿用旧脚本产物的条目了。
沿用: list[str] = []


def 整理清单(out: Path, manifest: list[dict]) -> None:
    """写交付清单，并把本轮没重做的条目原样继承下来。

    蓝图第 8 节要求每张图都写清「为什么需要这张图」，所以清单必须是完整的一份，
    不能只剩这次重画的几张。

    以前只继承 `沿用` 里点名的那几张，于是 `--only F8` 重画一张图就会把清单缩成
    一条，报告端所有图注一起消失。现在改成「本轮没生成的条目一律继承」。
    """
    说明 = out / "图表说明.json"
    旧 = (读json(说明).get("figures") or []) if 说明.is_file() else []
    本轮 = {e.get("figure") for e in manifest}
    继承 = [e for e in 旧 if e.get("figure") not in 本轮]
    合并 = sorted(manifest + 继承, key=lambda e: e.get("figure", ""))
    说明.write_text(json.dumps({"figures": 合并}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    删除 = []
    for stem in 被取代:
        for suffix in (".png", ".pdf", ".svg"):
            p = out / (stem + suffix)
            if p.is_file():
                p.unlink()
                删除.append(p.name)
    if 删除:
        print("[clean] 清掉被取代的旧图：" + "、".join(删除), flush=True)
    print("[ok] 交付清单 " + str(len(合并)) + " 张图 -> " + str(说明), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPORTS / "figures_dev")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--metadata-only", action="store_true",
                        help="只更新 F2/F3/F4/F6/F7/F8 的历史图说明，不读取表达矩阵或重绘")
    args = parser.parse_args()

    if args.metadata_only:
        只更新历史说明(args.out, args.only)
        return 0

    出版样式()
    基 = 载入真值()
    print("[info] " + str(len(基["truth"])) + " spots，稀有 "
          + str(len(基["rare"])) + " 类："
          + "、".join(t + " " + str(int(基["counts"][t])) for t in 基["rare"]),
          flush=True)

    manifest: list[dict] = []

    def want(name: str) -> bool:
        return args.only is None or name in args.only

    if want("F1"):
        manifest.append(figure_1(基, args.out))

    if want("F2"):
        arms = [
            ("A0  upstream scFormer", "stage0_A0_seed0"),
            ("A  spatial batching", "stage1_A_seed0"),
            ("B  + spatial relation k=10", "stage1_B_k10_seed0"),
            ("C  dynamic, warm-up init", "stage2_C_t0.2_dq0.2_seed0_v2b_preflight"),
            ("C  dynamic, A0-label init", "stage2_C_t0.2_dq0.2_seed0_initA0_v3a_probe"),
        ]
        entry = figure_2(基, args.out, arms)
        if entry is None:
            print("[warn] F2 缺 pred.npy，跳过", flush=True)
        else:
            manifest.append(entry)

    if want("F6"):
        entry = figure_6(基, args.out)
        if entry is None:
            print("[warn] F6 缺 cell_embedding.npy，跳过", flush=True)
        else:
            manifest.append(entry)

    if want("F3"):
        entry = figure_3(基, args.out, [
            ("B  + spatial relation k=10\n(fixed pseudo-label)",
             "stage1_B_k10_seed0"),
            ("C  dynamic, warm-up init", "stage2_C_t0.2_dq0.2_seed0_v2b_preflight"),
            ("C  dynamic, A0-label init",
             "stage2_C_t0.2_dq0.2_seed0_initA0_v3a_probe"),
        ])
        if entry is None:
            print("[warn] F3 缺 pred.npy，跳过", flush=True)
        else:
            manifest.append(entry)

    if want("F7"):
        entry = figure_7(基, args.out, [
            ("Fixed δ = 0.5, τ = 0.02", "stage2_C_t0.02_d0.5_seed0_v2a_preflight"),
            ("Quantile δ, warm-up init",
             "stage2_C_t0.2_dq0.2_seed0_v2b_preflight"),
            ("Quantile δ, A0-label init",
             "stage2_C_t0.2_dq0.2_seed0_initA0_v3a_probe"),
        ])
        if entry is None:
            print("[warn] F7 缺 refresh_trace，跳过", flush=True)
        else:
            manifest.append(entry)

    if want("F8"):
        entry = figure_8(基, args.out)
        if entry is None:
            print("[warn] F8 缺 rare_domain_baseline_frontier/report.json，跳过", flush=True)
        else:
            manifest.append(entry)

    if want("F4"):
        entry = figure_4(args.out)
        if entry is None:
            print("[warn] F4 缺 metrics.json，跳过", flush=True)
        else:
            manifest.append(entry)

    if want("F5"):
        entry = figure_5(基, args.out)
        if entry is None:
            print("[warn] F5 缺 stage1a_batching_probe/report.json，跳过", flush=True)
        else:
            manifest.append(entry)

    args.out.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        print("[ok] " + entry["figure"] + ": " + entry["files"][0], flush=True)
        print("     理由：" + entry["why"], flush=True)
    if args.only is None:
        整理清单(args.out, manifest)
    else:
        print("[info] --only 模式不动交付清单，避免把没重画的图从清单里删掉", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
