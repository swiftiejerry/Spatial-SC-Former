"""F10 —— 五个稀有域的 marker 表达场。

为什么单独一张
--------------
蓝图第 8 节点名了两种参考呈现方式：空间散点分区图和 marker 表达场。前者是 F1 / F2，
后者一张都没有，报告里「五个稀有域在表达上到底可分不可分」这句话因此是悬空的。
这张图只回答那一句：稀有域在表达层面有自己的 marker 场，且高表达的位置和真值标注
对得上。所以主终点 rare-F1 = 0 不是「这些域在表达里根本不存在」，而是划分 / 表示
这一侧的问题。

口径边界（画图不许越界）
------------------------
* 只说表达层面可分。不写「marker 证明模型能检测稀有域」——这张图里没有任何模型输出。
* marker 由 Wilcoxon 秩和在真值标注上选出，不是文献先验；选不干净的照实标注。
* 稀有 = 真值占比 < 5%，不改判据。
* slide_4E / Figure 7 的数字不进这张图。

两段式跑法
----------
读 h5ad + rank_genes_groups 会超过 60 秒，所以拆成两步：``--重算`` 先把 DE 结果和
要画的表达向量落盘成缓存，画图只读缓存。改版面时不必重算 DE。

helper 为什么是抄来的
---------------------
出版样式 / 存图 / 存预览 / 组织散点 这几个和 ``绘制交付图表.py`` 是同一套契约。
本来该 import，但那个文件此刻正被另一个 agent 改，import 一个半成品会在导入期就炸，
所以这里复制一份。契约有变动时两边一起改。

切片几何
--------
``obsm['spatial']`` 存的是 Visium 的整数阵列坐标 ``(array_row, array_col)``，
不是等距笛卡尔坐标。直接当散点画会把近圆形的淋巴结压成细长椭圆——第一版就是这样，
同一张切片在 F1 里是圆的、在这里是椭圆的。现在坐标一律先过
``图版基元.晶格几何()`` 还原真实几何（列距 100 µm、行距 86.6 µm），六个面板都用
``图版基元`` 的蜂窝铺砌，和 F1 / F2 同一套函数、同一套口径。
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

# 图版基元 和本文件同目录；直接跑本脚本时 sys.path[0] 就是 scripts/，
# 被别处以模块方式加载时不保证，所以显式补一次。
_脚本目录 = str(Path(__file__).resolve().parent)
if _脚本目录 not in sys.path:
    sys.path.insert(0, _脚本目录)

import 图版基元 as 基元  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "experiments" / "reports"
H5AD = ROOT / "data" / "raw" / "human_lymph_node_A1" / "adata_RNA.h5ad"
LABEL_KEY = "final_annot"
图目录 = REPORTS / "figures_dev"
缓存路径 = 图目录 / "_缓存_稀有域marker.npz"
说明路径 = 图目录 / "说明_F10.json"
图名 = "F10_稀有域marker表达场"

# 600 dpi 交付图长边 4000+ px，直接 view_image 会把整条会话钉死在
# 400 IMAGE_DIMENSION_EXCEEDED。看图只看这个目录里的缩图。
预览目录 = ROOT / "tmp" / "看图预览"
预览上限 = 1600

W2 = 178 / 25.4          # EI 双栏
RARE_THR = 0.05          # 真值稀有类判据

# 和 F1 / F2 / F3 同一套语义配色：同一个域在整份报告里永远是同一个颜色。
RARE_COLORS = {
    "follicle": "#B64342",
    "medulla vessels": "#0F4D92",
    "subcapsular sinus": "#42949E",
    "hilum": "#9A4D8E",
    "trabeculae": "#E28E2C",
}
丰度灰 = "#E4E4E4"
未检出灰 = "#E4E4E4"     # 该基因没测到的 spot：只交代组织形状，不参与色标
INK = "#272727"
RULE = "#767676"
# sequential，不用 rainbow。取 magma 的反向：浅底深高。
# 白底上「深 = 高」才有对比——正向 viridis / magma 的高端是亮黄，压在浅灰组织上直接消失，
# 真值圈也没地方站。深高还顺带让整张图在黑白打印下仍然可读。
CMAP = "magma_r"
# magma_r 最浅的那一小段要截掉。未检出的 spot 铺的是 #E4E4E4 浅灰，而 magma_r 的 0 端是
# #FCFDBF，两者亮度几乎一样、只差一点黄；散点时代它们之间有白缝隔着，蜂窝铺满之后
# 「没测到」和「刚测到」就贴在一起分不开了。从 0.14 起步，低表达端是明确的黄。
色标起点 = 0.14

# marker 的筛选口径。写成常量是为了让「哪个基因为什么被选中」可复核，
# 而不是埋在一行 if 里。
最小类内占比 = 0.60      # pts_in：该类至少 60% 的 spot 检出这个基因
最小占比倍数 = 2.0       # pts_in / pts_rest，"在该类里明显高于其他类"
最大校正p = 0.05
回退最小占比 = 0.30      # 判据全不过时，回退候选至少要这个检出率，否则画出来是散点不是场
留档条数 = 10            # 说明 json 里存多少条候选，供复核
色标分位 = 99.0          # 色标上限取「检出 spot」的这个分位数


def 诊断打印(类名: str, 排名: list[dict]) -> None:
    """判据一条都不过的时候，先看清是哪一条在卡，别直接改阈值。

    三张小表对应三种可能：DE 排名靠前的基因本来就不专一、专一的基因存在但类内检出率
    没到「最小类内占比」、以及表达倍数最大的是谁。
    """
    def 打(题: str, 行们: list[dict]) -> None:
        print("    " + 题, flush=True)
        for r in 行们[:6]:
            print("      %-12s rank %5d  pts %.2f/%.2f  ratio %5.2f  fold %5.2f  "
                  "p_adj %8.2g"
                  % (r["gene"], r["rank"], r["pts_in"], r["pts_rest"],
                     r["pts_in"] / max(r["pts_rest"], 1e-9),
                     r["fold_vs_tissue_mean"], r["pvals_adj"]), flush=True)

    显著 = [r for r in 排名 if r["pvals_adj"] < 最大校正p]
    print("  诊断 %s（显著基因 %d 个）" % (类名, len(显著)), flush=True)
    打("按 DE 排名", 排名)
    打("按 pts 倍数（显著 + pts_in >= 0.30）",
       sorted([r for r in 显著 if r["pts_in"] >= 0.30],
              key=lambda r: -r["pts_in"] / max(r["pts_rest"], 1e-9)))
    打("按表达倍数（显著 + pts_in >= 0.50）",
       sorted([r for r in 显著 if r["pts_in"] >= 0.50],
              key=lambda r: -r["fold_vs_tissue_mean"]))


# --------------------------------------------------------------------- 通用 helper

def 出版样式() -> None:
    """西文字体显式指定。simsun 的拉丁字形等宽，数字排出来像打字机，不能拿来排英文。"""
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
        "axes.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def 存预览(fig, name: str) -> Path:
    """另存一张长边 <= 预览上限 的副本。交付图一个字节不动。"""
    from PIL import Image

    预览目录.mkdir(parents=True, exist_ok=True)
    p = 预览目录 / (name + ".png")
    fig.savefig(p, dpi=max(40.0, 预览上限 * 0.94 / max(fig.get_size_inches())))
    # bbox="tight" 按实际画到的范围裁剪，算出来的 dpi 不保证卡住上限，落盘后复核。
    with Image.open(p) as im:
        w, h = im.size
        if max(w, h) > 预览上限:
            k = 预览上限 / max(w, h)
            im.convert("RGB").resize((max(1, round(w * k)), max(1, round(h * k))),
                                     Image.LANCZOS).save(p, optimize=True)
    return p


def 查宽(fig, 目标宽: float) -> None:
    """落盘前量一次 tight bbox。

    ``savefig.bbox="tight"`` 会按实际画到的范围裁剪，也会往外扩：一行没折的图注就能把
    7.0 in 的图撑到 11.8 in（第一版就是这样）。撑宽的图进报告要被按宽度缩回去，7 pt
    正文跟着缩到 4 pt 出头，直接掉出 5 pt 底线，而且从图上看不出来。所以这里出声。
    """
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    宽 = float(bb.width)
    垫 = 2 * float(plt.rcParams["savefig.pad_inches"])
    print("[bbox] %s: %.1f x %.1f mm"
          % (图名, 25.4 * (宽 + 垫), 25.4 * (float(bb.height) + 垫)), flush=True)
    if 宽 > 目标宽 + 0.02:
        print("警告：tight bbox 宽 %.2f in 超过设计宽 %.2f in，排进报告后 7 pt 正文只剩 "
              "%.1f pt，去把最长的那行文字折短" % (宽, 目标宽, 7 * 目标宽 / 宽),
              flush=True)


def 存图(fig, out: Path, name: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    查宽(fig, W2)
    paths = []
    for suffix, kw in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 600})):
        p = out / (name + suffix)
        fig.savefig(p, **kw)
        paths.append(p.name)
    存预览(fig, name)
    plt.close(fig)
    return paths


def 表达色标():
    """截掉最浅一段的 magma_r。瓦片和色条用同一个对象，两边不会漂。"""
    from matplotlib.colors import LinearSegmentedColormap

    底 = plt.get_cmap(CMAP)
    return LinearSegmentedColormap.from_list(
        CMAP + "_trunc", 底(np.linspace(色标起点, 1.0, 256)))


def 蜂窝(ax, coords, 几何: dict, 面色) -> None:
    """一格组织切片：Voronoi 蜂窝 + 切片外轮廓 + 500 µm 比例尺。

    和 F1 / F2 同一套 `图版基元` 函数、同一套口径（收缩 0.94 不描边、轮廓 0.5、
    比例尺在左下角）。切片是近圆形的，四个角本来就是空的，比例尺正好站在那儿。
    """
    基元.组织瓦片(ax, coords, 面色, 收缩=0.94, 几何=几何)
    基元.组织外轮廓(ax, coords, 几何=几何, 宽=0.5)
    基元.比例尺(ax, 几何, 长度_um=500.0)


def 域轮廓(ax, coords, 几何: dict, 标签, 域: str, 色: str, *,
        垫白: bool = False) -> None:
    """域标注：把这个域和外部之间的界线画出来，蜂窝版的「标注环」。

    逐 spot 画圈在这个尺寸下没法用：一格只有 27 mm 宽，一个 spot 占 0.26 mm，
    圈会糊成一片色块。界线说的是同一件事（这个域在哪），而且和 F1 里给稀有域描边
    是同一个函数。表达场的底色从浅黄一直到近黑，所以先垫一条白边，
    界线在整条色标上都读得出来。
    """
    if 垫白:
        基元.域边界(ax, coords, 标签, 几何=几何, 色="white", 宽=1.05, 只画={域},
                 zorder=3.6)
    基元.域边界(ax, coords, 标签, 几何=几何, 色=色, 宽=0.45, 只画={域}, zorder=3.8)


# ------------------------------------------------------------------------- 缓存

def 不合格原因(行: dict) -> list[str]:
    """三条判据都是「这个基因是不是这一类自己的」的最低要求：校正后显著、这一类里大多数
    spot 都检出、检出率明显高于其他类。任何一条不过就不能当 marker 用。
    """
    坏 = []
    if not (行["pvals_adj"] < 最大校正p):
        坏.append("p_adj >= %g" % 最大校正p)
    if 行["pts_in"] < 最小类内占比:
        坏.append("pts_in < %.2f" % 最小类内占比)
    if 行["pts_in"] < 最小占比倍数 * max(行["pts_rest"], 1e-9):
        坏.append("pts_in/pts_rest < %.1f" % 最小占比倍数)
    return 坏


def 短原因(行: dict, spot数: int) -> str:
    """图上那一行「差在哪」。按严重程度取第一条：不显著 > 不专一 > 类内检出不够。

    格宽 27 mm，5.4 pt 下最多 22 个字符，所以只留数字和门槛，别写句子。
    """
    if not (行["pvals_adj"] < 最大校正p):
        return "n=%d, none significant" % spot数
    比 = 行["pts_in"] / max(行["pts_rest"], 1e-9)
    if 比 < 最小占比倍数:
        return "best %.1fx, need %.0fx" % (比, 最小占比倍数)
    if 行["pts_in"] < 最小类内占比:
        return "in %.0f%%, need %.0f%%" % (100 * 行["pts_in"],
                                           100 * 最小类内占比)
    return ""


def 挑marker(排名: list[dict]) -> dict:
    """三档：判据全过的排名最高者 -> 显著里最专一者 -> 排名第一。

    第一档才算「这个域有自己的 marker」。扫的是整条 Wilcoxon 排名而不是前几十名：
    Wilcoxon 的 z 偏爱「到处都表达、只是这一类高一点」的基因，稀疏但专一的 marker 常
    排在几十名之后，截断候选池等于把「没有专一 marker」这个结论强加给数据。

    判据全不过时不画排名第一，而是画显著基因里 pts 倍数最高的那个（第二档）。理由是
    结论方向：这一格最后要写「不专一」，那就得先把这个域最有利的证据摆上去，否则读者
    有理由怀疑是基因挑得差。倍数最高恰好就是失败那条判据上的最优解，所以这是给对方
    最好的一手，不是挑好看的。

    连显著基因都没有（trabeculae，n=8 过不了多重校正）就退回排名第一（第三档）：
    这时候按倍数排 18085 个基因必然捞到偶然极值，那才是真的挑好看的。
    """
    通过 = [r for r in 排名 if not r["不合格原因"]]
    显著 = [r for r in 排名 if r["pvals_adj"] < 最大校正p]
    底 = {"通过数": len(通过), "通过": 通过, "显著数": len(显著)}
    if 通过:
        return dict(底, 选中=通过[0], 备选=通过[1] if len(通过) > 1 else None,
                    档=1, 稳定=True, 原因="")
    可读 = sorted([r for r in 显著 if r["pts_in"] >= 回退最小占比],
                  key=lambda r: -r["pts_in"] / max(r["pts_rest"], 1e-9))
    if 可读:
        return dict(底, 选中=可读[0], 备选=可读[1] if len(可读) > 1 else None,
                    档=2, 稳定=False,
                    原因="no gene met all three criteria; shown is the most "
                         "detection-specific significant gene with pts_in >= %.2f"
                         % 回退最小占比)
    return dict(底, 选中=排名[0], 备选=排名[1] if len(排名) > 1 else None,
                档=3, 稳定=False,
                原因="no gene reaches p_adj < %g in this domain; shown is the "
                     "top-ranked DE gene" % 最大校正p)


def 逐基因统计(X, m: np.ndarray) -> dict[str, np.ndarray]:
    """一次算完全部基因的检出率和均值。

    逐基因 ``adata[:, g].X`` 取列在 18085 个基因上跑不动，所以整块向量化：稀疏矩阵上
    ``(X > 0).sum(axis=0)`` 就是检出数，``mean(axis=0)`` 就是均值。这样候选池不必截断。
    """
    def 计(块, n):
        if hasattr(块, "getnnz"):
            检出 = np.asarray((块 > 0).sum(axis=0)).ravel()
            均值 = np.asarray(块.mean(axis=0)).ravel()
        else:
            块 = np.asarray(块)
            检出 = (块 > 0).sum(axis=0).astype(float)
            均值 = 块.mean(axis=0)
        return 检出 / max(n, 1), 均值

    pts_in, mean_in = 计(X[m], int(m.sum()))
    pts_rest, _ = 计(X[~m], int((~m).sum()))
    _, mean_all = 计(X, X.shape[0])
    return {"pts_in": pts_in, "pts_rest": pts_rest,
            "mean_in": mean_in, "mean_all": mean_all}


def 计算缓存() -> None:
    """DE + 选 marker + 抽出要画的表达向量，全部落盘。"""
    import scanpy as sc

    print("读 h5ad:", H5AD, flush=True)
    adata = sc.read_h5ad(H5AD)
    print("shape:", adata.shape, "| var 前 5:", list(adata.var_names[:5]), flush=True)

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    truth = np.asarray(adata.obs[LABEL_KEY].astype(str).to_numpy(), dtype=str)
    share = {t: float((truth == t).mean()) for t in sorted(set(truth.tolist()))}
    counts = {t: int((truth == t).sum()) for t in share}
    rare = [t for t in share if share[t] < RARE_THR]
    序 = sorted(rare, key=lambda t: -counts[t])
    print("稀有类（占比 < %.0f%%）:" % (100 * RARE_THR),
          [(t, counts[t], round(100 * share[t], 2)) for t in 序], flush=True)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("DE: rank_genes_groups wilcoxon on", LABEL_KEY, flush=True)
    sc.tl.rank_genes_groups(adata, groupby=LABEL_KEY, method="wilcoxon")
    结果 = adata.uns["rank_genes_groups"]

    X = adata.X
    列号 = {g: i for i, g in enumerate(map(str, adata.var_names))}

    def 取列(基因: str) -> np.ndarray:
        j = 列号[基因]
        col = X[:, j]
        col = col.toarray() if hasattr(col, "toarray") else np.asarray(col)
        return np.asarray(col, dtype=np.float64).ravel()

    汇总, 基因名, 向量 = [], [], []
    for 类名 in 序:
        m = truth == 类名
        统 = 逐基因统计(X, m)
        排名 = []
        for i, g in enumerate(map(str, 结果["names"][类名])):
            j = 列号[g]
            均值内 = float(统["mean_in"][j])
            均值全 = float(统["mean_all"][j])
            行 = {
                "gene": g,
                "rank": i + 1,
                "score": float(结果["scores"][类名][i]),
                "logfoldchange": float(结果["logfoldchanges"][类名][i]),
                "pvals_adj": float(结果["pvals_adj"][类名][i]),
                "pts_in": float(统["pts_in"][j]),
                "pts_rest": float(统["pts_rest"][j]),
                "mean_log1p_in": 均值内,
                "mean_log1p_all": 均值全,
                "fold_vs_tissue_mean": 均值内 / max(均值全, 1e-12),
            }
            行["不合格原因"] = 不合格原因(行)
            排名.append(行)
        判 = 挑marker(排名)
        if not 判["稳定"]:
            诊断打印(类名, 排名)
        选 = 判["选中"]
        v = 取列(选["gene"])
        汇总.append({
            "domain": 类名, "n_spots": counts[类名], "share": share[类名],
            "gene": 选["gene"], "rank": 选["rank"], "score": 选["score"],
            "logfoldchange": 选["logfoldchange"], "pvals_adj": 选["pvals_adj"],
            "pts_in": 选["pts_in"], "pts_rest": 选["pts_rest"],
            "mean_log1p_in": 选["mean_log1p_in"],
            "mean_log1p_all": 选["mean_log1p_all"],
            "fold_vs_tissue_mean": 选["fold_vs_tissue_mean"],
            "stable": 判["稳定"], "unstable_reason": 判["原因"],
            "second_choice": (判["备选"] or {}).get("gene"),
            "n_genes_passing": 判["通过数"],
            "n_genes_significant": 判["显著数"],
            "tier": 判["档"],
            "fail_short": "" if 判["稳定"] else 短原因(选, counts[类名]),
            "top_by_de_rank": 排名[:留档条数],
            "top_passing": 判["通过"][:留档条数],
        })
        基因名.append(选["gene"])
        向量.append(v)
        print("%-20s %-10s rank %5d  pts %.2f/%.2f  fold %5.2f  p_adj %8.2g  "
              "sig %4d  passing %4d  tier %d  stable=%s"
              % (类名, 选["gene"], 选["rank"], 选["pts_in"], 选["pts_rest"],
                 选["fold_vs_tissue_mean"], 选["pvals_adj"], 判["显著数"],
                 判["通过数"], 判["档"], 判["稳定"]), flush=True)
        if not 判["稳定"]:
            print("    差在哪:", 短原因(选, counts[类名]), "|", 判["原因"],
                  flush=True)

    # 点的大小按真实点距推，别写死：换数据集或换版面宽度都不用重调。
    from sklearn.neighbors import NearestNeighbors

    近 = NearestNeighbors(n_neighbors=2).fit(coords)
    d, _ = 近.kneighbors(coords)
    点距 = float(np.median(d[:, 1]))
    print("最近邻中位点距:", round(点距, 3),
          "| 组织长宽比:",
          round(float(np.ptp(coords[:, 0]) / np.ptp(coords[:, 1])), 4), flush=True)

    图目录.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        缓存路径,
        coords=coords.astype(np.float32),
        truth=truth,
        序=np.asarray(序),
        genes=np.asarray(基因名),
        expr=np.asarray(向量, dtype=np.float32),
        点距=np.asarray(点距),
        汇总=np.asarray(json.dumps(汇总, ensure_ascii=False)),
    )
    print("缓存已写:", 缓存路径, flush=True)


def 载入缓存() -> dict:
    if not 缓存路径.is_file():
        raise SystemExit("缺缓存 %s，先跑 --重算" % 缓存路径)
    z = np.load(缓存路径, allow_pickle=False)
    return {
        "coords": np.asarray(z["coords"], dtype=np.float64),
        "truth": np.asarray(z["truth"]).astype(str),
        "序": [str(t) for t in z["序"]],
        "genes": [str(g) for g in z["genes"]],
        "expr": np.asarray(z["expr"], dtype=np.float64),
        "点距": float(z["点距"]),
        "汇总": json.loads(str(z["汇总"])),
    }


# -------------------------------------------------------------------------- 图

def 定色标(缓存: dict) -> None:
    """逐基因定色标上限：检出 spot 的 p99。

    为什么按检出值的分位数而不是全片最大值：全片九成以上的 spot 是 0，最大值又常被一两个
    spot 拉高，两头一挤，域内那些真正抬起来的 spot 会落到色标下三分之一，看着和背景差不多。
    旧的 marker 图（``create_fig7_publication_quality.py``）取的也是检出值的 p99，
    这里沿用同一条口径。上限之上的 spot 饱和在最深色，逐基因记进说明 json。
    """
    for i, 记 in enumerate(缓存["汇总"]):
        v = 缓存["expr"][i]
        正 = v > 0
        记["n_positive"] = int(正.sum())
        记["colour_max"] = float(v.max())
        记["colour_vmax"] = (float(np.percentile(v[正], 色标分位))
                             if 正.any() else 1.0)
        记["colour_vmax_rule"] = "p%g of detected spots" % 色标分位


def 共位(缓存: dict) -> None:
    """「亮的地方在不在圈里」的数字版：取表达最高的 n 个 spot，看几个落在这个域里。

    n 取该域自己的 spot 数，所以命中率同时是 precision 和 recall，可以直接和该域的全片
    占比比——占比就是随机水平。图上不写这个数（每格已经四行字了），但报告正文要引用
    「表达层面对不对得上」的时候用的就是它，比 fold 更贴题。并列值按 spot 顺序打破，
    所以结果可复现。
    """
    truth = 缓存["truth"]
    for i, 记 in enumerate(缓存["汇总"]):
        v = 缓存["expr"][i]
        n = int(记["n_spots"])
        取 = np.argsort(-v, kind="stable")[:n]
        命中 = int((truth[取] == 记["domain"]).sum())
        记["top_n_hits"] = 命中
        记["top_n_precision"] = 命中 / max(n, 1)
        记["top_n_enrichment"] = (命中 / max(n, 1)) / max(float(记["share"]), 1e-12)


def figure_10(缓存: dict, out: Path) -> dict:
    """一行 6 格：最左是真值参照，右边 5 格是各稀有域自己的 marker 表达场。

    版面为什么是一行不是两行：色条加逐格数字那一块要 1.14 in，两行就得摆两套，
    再加上还原真实几何后每格 27 x 29 mm 的切片，整图会高到 170 mm 以上。
    一行 6 格是 178 x 70 mm，和 F1 一个体量；位置比对也更顺——参照格和五个域在同一条
    水平线上，眼睛左右扫一遍就够。
    """
    coords, truth = 缓存["coords"], 缓存["truth"]
    序, genes, 汇总 = 缓存["序"], 缓存["genes"], 缓存["汇总"]
    expr = 缓存["expr"]

    # 阵列坐标先还原真实几何。长宽比按 图版基元.整理切片轴 的口径算：窗口是
    # 「spot 范围 ± 外接圆半径」再加 3% 边距，边距在比值里约掉。格子的宽高比配成这个数，
    # equal aspect 才不会把格子缩小、两边留白。
    几何 = 基元.晶格几何(coords)
    um, R = 几何["um"], 几何["R"]
    长宽比 = ((float(np.ptp(um[:, 0])) + 2 * R) / (float(np.ptp(um[:, 1])) + 2 * R))
    色标 = 表达色标()
    # 参照格只分「稀有域」和「其余」，所以丰度类合成一档，免得在丰度类之间画出界线。
    参照标签 = np.where(np.isin(truth, 序), truth, "\u00b7rest")

    # 左右各留 1 个百分点：bbox="tight" 之后还要加 2 x 0.02 in 的 pad，
    # 贴着 0.006 / 0.994 摆会把落盘宽度顶到 178.1 mm，比跨栏上限高 0.1 mm。
    左, 右, 格隙 = 0.010, 0.988, 0.085
    格宽 = (右 - 左) * W2 / (6 + 5 * 格隙)                  # 英寸
    格高 = 格宽 / 长宽比
    顶白, 上题, 条隙, 条高, 下文, 底白 = 0.02, 0.32, 0.05, 0.045, 1.14, 0.02
    图高 = 顶白 + 上题 + 格高 + 条隙 + 条高 + 下文 + 底白

    def 寸(英寸: float) -> float:
        """英寸（从图底往上）换成 figure 分数。版面全部用英寸算，免得被格高牵着走。"""
        return 英寸 / 图高

    轴顶 = 寸(底白 + 下文 + 条高 + 条隙 + 格高)

    fig = plt.figure(figsize=(W2, 图高))
    gs = fig.add_gridspec(1, 6, wspace=格隙, left=左, right=右, top=轴顶,
                          bottom=寸(底白 + 下文 + 条高 + 条隙))
    gsc = fig.add_gridspec(1, 6, wspace=格隙, left=左, right=右,
                           top=寸(底白 + 下文 + 条高), bottom=寸(底白 + 下文))

    def 标题(pos, 字母: str, 主: str, 主色: str, 次: str, 次色: str = INK) -> None:
        """格宽只有 27 mm，标题任何一行超过 20 个字符就会压到隔壁格，写之前先数一遍。"""
        fig.text(pos.x0, 轴顶 + 寸(0.150), 字母, fontsize=9.0, fontweight="bold",
                 color=INK, ha="left", va="bottom")
        fig.text(pos.x0 + 0.014, 轴顶 + 寸(0.152), 主, fontsize=6.8,
                 fontweight="bold", color=主色, ha="left", va="bottom")
        fig.text(pos.x0 + 0.014, 轴顶 + 寸(0.030), 次, fontsize=6.1, color=次色,
                 ha="left", va="bottom")

    def 行y(i: int) -> float:
        """色条下面的文字块，第 i 行的顶边。"""
        return 寸(底白 + 下文 - 0.01 - i * 0.098)

    字母 = "abcdef"

    # --- 参照格：只给 5 个稀有域上色，其余统一浅灰。
    ax = fig.add_subplot(gs[0, 0])
    面色 = np.full(len(truth), 丰度灰, dtype=object)
    for t in 序:
        面色[truth == t] = RARE_COLORS[t]
    蜂窝(ax, coords, 几何, list(面色))
    for t in 序:
        域轮廓(ax, coords, 几何, 参照标签, t, RARE_COLORS[t])
    参照位 = ax.get_position()
    稀有数 = sum(int(r["n_spots"]) for r in 汇总)
    标题(参照位, 字母[0], "Ground truth", INK,
         "%d rare spots, %.1f%%"
         % (稀有数, 100 * sum(float(r["share"]) for r in 汇总)), RULE)

    # 参照格底下用色块清单代替色条：panel b-f 的标题色就是这里的色。
    for i, t in enumerate(序):
        y = 底白 + 下文 + 条高 - i * 0.098
        fig.add_artist(plt.Rectangle(
            (参照位.x0, 寸(y - 0.052)), 0.0105, 寸(0.030),
            transform=fig.transFigure, facecolor=RARE_COLORS[t], edgecolor="none"))
        fig.text(参照位.x0 + 0.0135, 寸(y - 0.037), t, fontsize=5.6,
                 color=RARE_COLORS[t], ha="left", va="center")

    # --- 五个 marker 场。
    for i, (t, g, 记) in enumerate(zip(序, genes, 汇总), start=1):
        v = expr[i - 1]
        vmax = float(记["colour_vmax"])
        正 = v > 0
        axm = fig.add_subplot(gs[0, i])
        # 没检出的 spot 铺浅灰、不进色标，只交代组织形状。读者要判断的是「测到的地方在不在
        # 圈里」，把 90% 的 0 也涂进色标，整片组织就变成底噪了。
        面色 = np.full(len(truth), 未检出灰, dtype=object)
        if 正.any():
            from matplotlib.colors import to_hex

            归一 = plt.Normalize(0.0, vmax)
            面色[正] = [to_hex(c) for c in 色标(归一(v[正]))]
        蜂窝(axm, coords, 几何, list(面色))
        域轮廓(axm, coords, 几何, np.where(truth == t, t, "\u00b7rest"), t,
             RARE_COLORS[t], 垫白=True)
        pos = axm.get_position()
        中心 = 0.5 * (pos.x0 + pos.x1)
        标题(pos, 字母[i], t, RARE_COLORS[t], g)

        axc = fig.add_subplot(gsc[0, i])
        位 = axc.get_position()
        axc.set_position([位.x0 + 0.14 * 位.width, 位.y0, 0.72 * 位.width, 位.height])
        条 = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0.0, vmax),
                                                cmap=色标),
                          cax=axc, orientation="horizontal")
        条.outline.set_linewidth(0.4)
        axc.set_xticks([])
        axc.set_yticks([])
        条位 = axc.get_position()
        fig.text(条位.x0, 行y(0), "0", fontsize=5.4, color=RULE, ha="left",
                 va="top")
        fig.text(条位.x1, 行y(0), "%.1f" % vmax, fontsize=5.4, color=RULE,
                 ha="right", va="top")
        fig.text(中心, 行y(1), "log1p expression", fontsize=5.4, color=RULE,
                 ha="center", va="top")
        fig.text(中心, 行y(2), "%.1fx tissue mean" % 记["fold_vs_tissue_mean"],
                 fontsize=6.2, color=INK, ha="center", va="top")
        fig.text(中心, 行y(3),
                 "%d spots, %.0f%% detected"
                 % (记["n_spots"], 100 * 记["pts_in"]),
                 fontsize=5.4, color=RULE, ha="center", va="top")
        # 「亮的地方在不在圈里」的数字。判据只管基因的检出模式，管不了空间是否排他：
        # hilum 的 COL1A2 判据过了，但 top 23 里只有 2 个落在 hilum，胶原在整圈囊膜都高。
        # 所以这一行必须和判据那一行并排出现，少了它「domain-specific」会被读成「标出了这个域」。
        fig.text(中心, 行y(4),
                 "%d of top %d inside" % (记["top_n_hits"], 记["n_spots"]),
                 fontsize=5.8, color=INK, ha="center", va="top")
        # 判据过没过必须两边都写，只在失败的格子里写一行会让人以为标注是临时加的。
        if 记["stable"]:
            fig.text(中心, 行y(5), "meets marker criteria", fontsize=5.6, color=RULE,
                     ha="center", va="top")
        else:
            fig.text(中心, 行y(5), "fails marker criteria", fontsize=5.6,
                     color=RARE_COLORS[t], ha="center", va="top",
                     fontweight="bold")
            fig.text(中心, 行y(6), 记["fail_short"], fontsize=5.4, color=RULE,
                     ha="center", va="top")

    # 图注必须自己折行。一行不折就会把 tight bbox 撑宽，图进报告后整体被缩小。
    fig.text(左, 行y(7),
             "Grey: every other spot in a, and every spot where the gene was not "
             "detected in b-f. Outlines enclose that domain's annotated spots.\n"
             "Genes come from Wilcoxon rank-sum on the ground-truth annotation, "
             "not from prior literature.\n"
             "Marker criteria: p_adj < %g, detected in >= %.0f%% of the domain, "
             "and detected >= %.0fx as often inside the domain as outside it.\n"
             "\"45 of top 96 inside\": of the 96 highest-expressing spots "
             "(96 = the domain's size), 45 are annotated to that domain. Chance "
             "level is the domain's share of the section, 0.2-2.8%%."
             % (最大校正p, 100 * 最小类内占比, 最小占比倍数),
             fontsize=5.6, color=RULE, ha="left", va="top", linespacing=1.35)

    files = 存图(fig, out, 图名)
    return {"figure": "F10", "files": files}


def 说明条目(缓存: dict) -> list[dict]:
    """落进说明 json 的可追溯字段。top_candidates 留着，方便复核「为什么是这个基因」。"""
    出 = []
    for 记 in 缓存["汇总"]:
        行 = {k: 记[k] for k in (
            "domain", "n_spots", "share", "gene", "rank", "score",
            "logfoldchange", "pvals_adj", "pts_in", "pts_rest",
            "mean_log1p_in", "mean_log1p_all", "fold_vs_tissue_mean",
            "colour_vmax", "colour_vmax_rule", "stable", "unstable_reason",
            "second_choice", "n_genes_passing", "n_genes_significant", "tier",
            "fail_short", "n_positive", "colour_max")}
        行.update({k: 记[k] for k in ("top_n_hits", "top_n_precision",
                                      "top_n_enrichment")})
        取 = ("gene", "rank", "score", "logfoldchange", "pvals_adj", "pts_in",
              "pts_rest", "fold_vs_tissue_mean")
        for 键 in ("top_by_de_rank", "top_passing"):
            行[键] = [{k: c[k] for k in 取} for c in 记[键]]
        出.append(行)
    return 出


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--重算", action="store_true", help="重算 DE 缓存")
    ap.add_argument("--只算", action="store_true", help="只算缓存，不画图")
    ap.add_argument("--out", type=Path, default=图目录)
    a = ap.parse_args()

    if a.重算 or a.只算 or not 缓存路径.is_file():
        计算缓存()
    if a.只算:
        return 0

    出版样式()
    缓存 = 载入缓存()
    定色标(缓存)
    共位(缓存)
    条目 = figure_10(缓存, a.out)
    # why 进报告，必须短。第一版 600 多字，排进 EI 双栏是整整一页十几行小字，
    # 读者在图注里读不完的东西等于没写。这里只留三样：这张图回答什么、follicle 那组数字、
    # 以及「图里没有模型输出」这条口径；细节全部落在 caption 字段，供复核不进报告。
    # 上限按 len() 卡 200 字符（最严的算法，中文字数只有 108），改这段前先数一遍。
    条目["why"] = (
        "五个稀有域在表达上可不可分？这决定 rare-F1 = 0 该归因于表达没信号，还是划分与"
        "表示这一侧。只有 follicle 能排除前者：CXCL13 在它 96 个 spot 里 90% 检出、"
        "域内 log1p 均值是全片 10.1 倍、表达最高的 96 个 spot 有 45 个在域内"
        "（随机水平 17.0 倍）。另四个域的判据结果逐格写在图上。"
        "图里没有模型输出，不能读成「模型能检测稀有域」。")
    条目["caption"] = (
        "蓝图第 8 节点了两种参考呈现方式，空间散点分区图有了（F1 / F2），marker 表达场"
        "一张都没有。少了它，「五个稀有域在表达上到底可分不可分」这句话在报告里是悬空的，"
        "而 rare-F1 = 0 的归因正压在这句话上：是「这些域在表达里根本不存在」，还是划分和"
        "表示这一侧的问题。这张图给的答案是分域的，不是一句「都可分」。"
        "follicle 有 CXCL13：96 个 spot 里 90% 检出、域内 log1p 均值是全片的 10.1 倍、"
        "表达最高的 96 个 spot 里 45 个落在 follicle（该域占比 2.8%，即随机水平的 17 倍），"
        "所以 follicle 这一支的 rare-F1 = 0 不能归因于「表达里没有信号」。"
        "hilum 的 COL1A2 过了判据，但表达最高的 23 个 spot 里只有 2 个在 hilum——胶原在"
        "整圈囊膜都高，所以只能说它有一个显著且偏专一的基因，不能说这个场标出了 hilum。"
        "medulla vessels、subcapsular sinus、trabeculae 没有任何基因同时满足三条判据，"
        "格子里照实写了差在哪：medulla vessels 全片最高 pts 倍数只有 1.9（12 个显著基因"
        "是一套被稀释的平滑肌信号），subcapsular sinus 最专一的 CD9 只在 48% 的域内 spot "
        "检出，trabeculae 8 个 spot 过不了多重校正。"
        "口径只到这里：这张图讲表达层面，图里没有任何模型输出，不能读成「模型能检测稀有域」，"
        "也不能反过来拿那三个域给主终点开脱。")
    条目["markers"] = {
        "selection": {
            "de": "normalize_total(1e4) -> log1p -> "
                  "sc.tl.rank_genes_groups(groupby='final_annot', method='wilcoxon')",
            "candidate_pool": "整条 Wilcoxon 排名（18085 个基因），不截断候选池",
            "criteria": {
                "pvals_adj": "< %g" % 最大校正p,
                "pts_in": ">= %.2f" % 最小类内占比,
                "pts_in / pts_rest": ">= %.1f" % 最小占比倍数,
            },
            "rule": "三档，逐域记在 domains[].tier。第一档：沿整条 Wilcoxon 排名从高到低"
                    "取第一个三条判据全过的基因，只有这一档在图上写 domain-specific。"
                    "第二档（判据全不过但有显著基因）：取显著基因里 pts 倍数最高、且 "
                    "pts_in >= %.2f 的那个——失败判据上的最优解，先把这个域最有利的证据"
                    "摆上去，再写「不专一」，读者才没法怀疑是基因挑差了。"
                    "第三档（连显著基因都没有，trabeculae n=8 过不了多重校正）：退回排名"
                    "第一，因为此时按倍数排 18085 个基因必然捞到偶然极值。"
                    "扫整条排名而不是只看前几十名，是因为 Wilcoxon 的 z 偏爱「到处都表达、"
                    "这一类高一点」的基因，稀疏但专一的 marker 常排在几十名之后，"
                    "截断候选池会把「没有专一 marker」这个结论强加给数据。" % 回退最小占比,
            "threshold_robustness": "三个不过判据的域，把 pts_in 门槛从 0.60 降到 0.50 "
                                    "照样不过：subcapsular sinus 最专一的 CD9 是 0.48，"
                                    "medulla vessels 全片最高 pts 倍数只有 1.87，"
                                    "trabeculae 卡在多重校正上、和这个门槛无关。所以"
                                    "「不过判据」不是阈值选出来的。注意这条只保证这三个域"
                                    "的判定不变：门槛动了有可能换掉 hilum 选中的基因，"
                                    "COL1A2 排在第 6，前 5 名里只要有 pts_in 落在 "
                                    "0.50-0.60 且倍数够的就会被优先取到。",
            "fold_vs_tissue_mean": "该基因在本域内的 log1p 表达均值 / 全片 log1p 均值，"
                                   "和图上的颜色是同一个量。",
            "top_n_precision": "取表达最高的 n 个 spot（n = 该域 spot 数），其中落在该域"
                               "的比例；top_n_enrichment 是它对该域全片占比（随机水平）"
                               "的倍数。图上没写这个数，但「亮的地方在不在圈里」要量化"
                               "就是它。",
            "colour_scale": "浅灰 = 该基因没检出的 spot，不进色标；检出的 spot 用 "
                            "magma_r（浅底深高），vmin=0、vmax 取检出 spot 的 p%g，"
                            "超出的饱和在最深色。逐基因的 vmax、全片最大值、检出数记在 "
                            "domains[] 的 colour_vmax / colour_max / n_positive。"
                            % 色标分位,
        },
        "domains": 说明条目(缓存),
    }
    说明路径.write_text(json.dumps(条目, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print("图:", ", ".join(条目["files"]))
    print("说明:", 说明路径)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
