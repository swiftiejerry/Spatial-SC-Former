# -*- coding: utf-8 -*-
"""图版基元：所有交付图共用的绘图原语。

为什么单独一个模块：`绘制交付图表.py` / `绘制marker表达场.py` / `绘制方法示意图.py`
都要画组织切片，之前各画各的，结果三处点大小不一样、有的有比例尺有的没有。
把「组织怎么画」这件事收到一个地方，样式漂移就没有藏身处。

最重要的一条：A1 的 `obsm['spatial']` 存的不是像素坐标，是 Visium 的**阵列坐标**
`(array_row, array_col)`（整数、最近邻距离恰好 sqrt(2)）。以前直接把它当散点画，
1.6 pt 的圆点摆在整数格上，看起来是一片稀疏的点云，组织的形状要靠读者自己脑补，
而且长宽比是错的。实际上这份坐标是规则三角晶格，可以还原成真实几何
（同行相邻 spot 中心距 100 µm，所以 1 个 col 单位 = 50 µm、1 个 row 单位 = 86.6 µm）
并按 Voronoi 蜂窝铺满，这才是空间转录组该有的呈现方式：连续的组织、
能看清的域边界、有比例尺、长宽比不撒谎。

哪一列是 row 哪一列是 col 不能靠几何判：`{(r,c) : c ≡ r mod 2}` 这个点集在两个方向上
都有间距 2 的邻居，完全对称。只能用 10x 的约定判——`array_row` 取值 0-77、
`array_col` 取值 0-127，所以取值超过 77 的那一列必然是 col。两列都不超 77 时
退回「取值范围更大的那一列是 col」，并在几何字典里标出这次是猜的。
"""

from __future__ import annotations

import numpy as np

# Visium 的固定几何：同行相邻 spot 中心距 100 µm。
# 阵列坐标里列步长是 2、行步长是 1，所以 1 个列单位 = 50 µm，
# 1 个行单位 = 100 * sin(60°) = 86.6025 µm。
VISIUM_PITCH_UM = 100.0
列单位_UM = VISIUM_PITCH_UM / 2.0
行单位_UM = VISIUM_PITCH_UM * np.sqrt(3.0) / 2.0

# 中性底色：不在当前高亮集合里的 spot 用它，保留组织形状但不抢注意力。
背景瓦片 = "#EDEDED"
背景瓦片深 = "#E0E0E0"
组织描边 = "#9A9A9A"
域描边 = "#FFFFFF"
INK = "#272727"


def 晶格几何(coords: np.ndarray) -> dict:
    """把 spot 坐标还原成真实几何，并给出蜂窝铺砌需要的一切。

    返回
    ----
    um     : (N,2) 单位 µm 的中心坐标（阵列坐标会被换算，像素坐标原样返回）
    R      : 正六边形外接圆半径（µm）。正三角晶格的 Voronoi 胞是正六边形，
             R = pitch / sqrt(3)，边长也等于 R。
    pitch  : 相邻 spot 中心距
    邻接    : (M,2) 相邻 spot 的下标对，M 约等于 3N
    是阵列坐标 : bool，用来在图注里交代坐标做过换算
    起始角  : 六边形第一个顶点的角度。100 µm 邻居所在的方向上必须是一条平边，
             所以 col 方向决定六边形是平顶还是尖顶。
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(coords, dtype=float)
    整数 = bool(np.allclose(xy, np.rint(xy), atol=1e-6))
    最近邻 = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    是阵列 = 整数 and abs(最近邻 - np.sqrt(2.0)) < 1e-6

    col轴 = 1
    col轴确定 = False
    起始角 = 0.0
    if 是阵列:
        上限 = [float(xy[:, 0].max()), float(xy[:, 1].max())]
        超 = [i for i in (0, 1) if 上限[i] > 77.0]
        if len(超) == 1:
            col轴, col轴确定 = 超[0], True
        else:
            跨度 = [float(xy[:, i].max() - xy[:, i].min()) for i in (0, 1)]
            col轴 = int(np.argmax(跨度))
        单位 = [行单位_UM, 行单位_UM]
        单位[col轴] = 列单位_UM
        单位[1 - col轴] = 行单位_UM
        um = np.column_stack([xy[:, 0] * 单位[0], xy[:, 1] * 单位[1]])
        pitch = VISIUM_PITCH_UM
        # col 方向就是 100 µm 邻居方向，那条方向上要是一条平边。
        # col 在 y 轴 → 平顶（顶点在 0°）；col 在 x 轴 → 尖顶（顶点在 90°）。
        起始角 = 0.0 if col轴 == 1 else np.pi / 2.0
    else:
        um = xy
        pitch = 最近邻
        起始角 = np.pi / 2.0

    R = pitch / np.sqrt(3.0)
    树 = cKDTree(um)
    邻接 = np.asarray(sorted(树.query_pairs(r=pitch * 1.05)), dtype=int)
    if 邻接.size == 0:
        邻接 = np.zeros((0, 2), dtype=int)
    return {"um": um, "R": float(R), "pitch": float(pitch),
            "邻接": 邻接, "是阵列坐标": 是阵列, "起始角": float(起始角),
            "col轴": int(col轴), "col轴确定": bool(col轴确定)}


def 六边形顶点(um: np.ndarray, R: float, 收缩: float = 1.0,
           起始角: float = 0.0) -> np.ndarray:
    """(N,6,2) 的顶点数组。第一个顶点在 `起始角`，100 µm 邻居方向上是一条平边。

    `收缩` < 1 会在瓦片之间留出缝。缝用留白而不是描边做，
    600 dpi 下 0.1 pt 的白描边在 PDF 里会被渲染器吃掉，留白不会。
    """
    角 = 起始角 + np.arange(6) * (np.pi / 3.0)
    偏 = np.column_stack([np.cos(角), np.sin(角)]) * (R * 收缩)
    return um[:, None, :] + 偏[None, :, :]


def 组织瓦片(ax, coords, facecolors, *, 收缩: float = 0.94,
           边色=None, 边宽: float = 0.0, alpha: float = 1.0,
           几何: dict | None = None, zorder: float = 2.0):
    """蜂窝组织图。facecolors 长度必须等于 spot 数。

    不画坐标轴、等比例、y 轴翻转（切片按图像习惯上下方向摆）。
    """
    from matplotlib.collections import PolyCollection

    g = 几何 or 晶格几何(coords)
    verts = 六边形顶点(g["um"], g["R"], 收缩, g.get("起始角", 0.0))
    集合 = PolyCollection(list(verts), facecolors=list(facecolors),
                        edgecolors="none" if 边色 is None else 边色,
                        linewidths=边宽, alpha=alpha, zorder=zorder,
                        rasterized=True, antialiased=True)
    ax.add_collection(集合)
    整理切片轴(ax, g)
    return g


def 整理切片轴(ax, 几何: dict, 边距: float = 0.03) -> None:
    um = 几何["um"]
    R = 几何["R"]
    x0, x1 = um[:, 0].min() - R, um[:, 0].max() + R
    y0, y1 = um[:, 1].min() - R, um[:, 1].max() + R
    dx, dy = x1 - x0, y1 - y0
    ax.set_xlim(x0 - dx * 边距, x1 + dx * 边距)
    ax.set_ylim(y0 - dy * 边距, y1 + dy * 边距)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def 域边界(ax, coords, labels, *, 几何: dict | None = None,
         色=INK, 宽: float = 0.45, zorder: float = 4.0,
         只画: set | None = None):
    """把标签不同的相邻瓦片之间那条公共边画出来。

    这是蜂窝图相对散点图最大的收益：域的形状不再靠颜色块的边缘去猜，
    薄环（subcapsular sinus）和细带（trabeculae）能直接看出来是连通的还是碎的。

    `只画` 给一组标签时，只画这些标签与外部之间的界线，用来做单域高亮。
    """
    from matplotlib.collections import LineCollection

    g = 几何 or 晶格几何(coords)
    um, R, 邻接 = g["um"], g["R"], g["邻接"]
    lab = np.asarray(labels)
    段 = []
    for i, j in 邻接:
        if lab[i] == lab[j]:
            continue
        if 只画 is not None and not ({lab[i], lab[j]} & 只画):
            continue
        p, q = um[i], um[j]
        d = q - p
        d = d / (np.linalg.norm(d) + 1e-12)
        n = np.array([-d[1], d[0]])
        m = (p + q) / 2.0
        段.append([m - n * (R / 2.0), m + n * (R / 2.0)])
    if 段:
        ax.add_collection(LineCollection(段, colors=色, linewidths=宽,
                                        zorder=zorder, capstyle="round"))
    return len(段)


def 组织外轮廓(ax, coords, *, 几何: dict | None = None,
           色=组织描边, 宽: float = 0.5, zorder: float = 5.0):
    """切片外缘：邻居不满 6 个的瓦片，把朝外那几条边画出来。"""
    from matplotlib.collections import LineCollection

    g = 几何 or 晶格几何(coords)
    um, R, 邻接 = g["um"], g["R"], g["邻接"]
    邻 = [[] for _ in range(len(um))]
    for i, j in 邻接:
        邻[i].append(j)
        邻[j].append(i)
    # 六个邻居方向（µm）。邻居方向垂直于六边形的边，所以比顶点角多 30°。
    基角 = g.get("起始角", 0.0) + np.pi / 6.0
    角 = 基角 + np.arange(6) * (np.pi / 3.0)
    方向 = np.column_stack([np.cos(角), np.sin(角)]) * g["pitch"]
    段 = []
    for i, nb in enumerate(邻):
        if len(nb) >= 6:
            continue
        有 = um[nb] - um[i] if nb else np.zeros((0, 2))
        for d in 方向:
            if len(有) and np.min(np.linalg.norm(有 - d, axis=1)) < g["pitch"] * 0.2:
                continue
            u = d / np.linalg.norm(d)
            n = np.array([-u[1], u[0]])
            m = um[i] + u * (g["pitch"] / 2.0)
            段.append([m - n * (R / 2.0), m + n * (R / 2.0)])
    if 段:
        ax.add_collection(LineCollection(段, colors=色, linewidths=宽,
                                        zorder=zorder, capstyle="round"))
    return len(段)


def 比例尺(ax, 几何: dict, *, 长度_um: float = 500.0, 位置=(0.04, 0.045),
        色=INK, 宽: float = 1.1, 字号: float = 5.6, 文字=None):
    """左下角比例尺。切片图没有坐标轴，尺度必须由比例尺交代。"""
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    px = x0 + (x1 - x0) * 位置[0]
    py = y1 - (y1 - y0) * 位置[1]
    ax.plot([px, px + 长度_um], [py, py], color=色, lw=宽,
            solid_capstyle="butt", zorder=8, clip_on=False)
    ax.text(px + 长度_um / 2.0, py - (y1 - y0) * 0.012,
            文字 or f"{长度_um:.0f} µm", ha="center", va="bottom",
            fontsize=字号, color=色, zorder=8)


def 白描字(ax, x, y, text, *, 字号: float = 6.0, 色=INK, 描边=2.0,
        ha="center", va="center", weight="normal", zorder: float = 9.0,
        transform=None):
    """带白色外描边的直接标注。组织图上颜色块底色深浅不一，
    直标必须自带描边，否则总有一两个域上的字读不出来。"""
    import matplotlib.patheffects as pe

    t = ax.text(x, y, text, fontsize=字号, color=色, ha=ha, va=va,
                fontweight=weight, zorder=zorder,
                transform=transform or ax.transData)
    t.set_path_effects([pe.withStroke(linewidth=描边, foreground="white"),
                        pe.Normal()])
    return t


def 标签重心(um: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """一个域放直标的位置。用中位数而不是均值：环形和分叉的域，
    均值会落到组织外面去。"""
    pts = um[mask]
    if not len(pts):
        return float("nan"), float("nan")
    return float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))


def 放大框(ax, 几何: dict, 中心_um, 半宽_um: float, *, 色="#B64342",
        宽: float = 0.6, 虚线=(0, (2.2, 1.4)), zorder: float = 7.0):
    """在切片上框出要放大的区域。inset 自己在图函数里建，
    这里只负责母图上那个虚线框和它的颜色口径统一。"""
    from matplotlib.patches import Rectangle

    cx, cy = 中心_um
    ax.add_patch(Rectangle((cx - 半宽_um, cy - 半宽_um), 2 * 半宽_um, 2 * 半宽_um,
                           fill=False, edgecolor=色, linewidth=宽,
                           linestyle=虚线, zorder=zorder))


def 哑铃(ax, y, 左, 右, *, 左色="#585858", 右色="#0F4D92", 连线="#BFBFBF",
       连宽: float = 1.6, 大小: float = 14.0, zorder: float = 3.0):
    """一行哑铃：同一个单位（同 seed / 同初始划分）前后两个值用一条棒连起来。

    配对的东西必须画成配对的。三个 seed 画三行，读者一眼看到的是
    「每一次都掉」还是「有的涨有的掉」，而不是两团重叠的边缘分布。
    """
    ax.plot([左, 右], [y, y], color=连线, lw=连宽, solid_capstyle="round",
            zorder=zorder)
    ax.scatter([左], [y], s=大小, color=左色, zorder=zorder + 1, linewidths=0)
    ax.scatter([右], [y], s=大小, color=右色, zorder=zorder + 1, linewidths=0)


def 单位换算说明(几何: dict) -> str:
    """图注里那句话，避免每张图各写一遍、写法还不一样。"""
    if not 几何["是阵列坐标"]:
        return ""
    return ("spot 坐标为 Visium 阵列坐标，已按列距 100 µm、行距 86.6 µm 还原真实几何，"
            "每个瓦片是该 spot 的 Voronoi 胞")
