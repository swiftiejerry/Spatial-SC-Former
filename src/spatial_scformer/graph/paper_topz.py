"""按论文 Methods 实现的 Top-Z 建图与 X 重构目标。

为什么要另写一份
----------------
公开代码的 `process_node` 不是论文的 Top-Z：它在该 spot 全部非零基因里按
`softmax(log(counts+1))` 随机抽 20 个，实测与论文 Top-20 Z 的重合只有 1.0-1.6%，
换个 seed 就换一整张图（Jaccard 0.030）。同时 `_build_batch` 用 batch gene 并集的
全部非零项建边，每个 spot 实际拿到中位 156 条边而不是 20 条，其中 87.2% 连的是它
自己没选的基因。数字见 `experiments/reports/偏离诊断_v2/实测核验.json`。

这个模块提供论文那一侧：

* `paper_x_matrix`  论文 Eq.1 的 X = log1p(1e4 * C / libsize)，重构目标用它。
* `gene_zscore`     论文 Eq.2 的逐基因标准化，只返回 mean / std，避免建整张稠密 Z。
* `topz_selection`  论文 Eq.3 的每 spot Top-K，确定性，tie 按基因索引升序。
* `paper_topz_batch_select`  和 `spatial_batch_select_whole` 同一个输出契约，
  只把 gene 选择换成上面那个确定性版本，方便"只改建图一项"。

公开代码那一版原样留着（`spatial_batch_select_whole`），两边可以配对比较。
"""

from __future__ import annotations

import math
import os
import pickle
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix, issparse


def _as_csr_genes_by_cells(rna_matrix: Any) -> csr_matrix:
    matrix = rna_matrix.tocsr() if issparse(rna_matrix) else csr_matrix(rna_matrix)
    return matrix


def paper_x_matrix(rna_matrix: Any) -> csr_matrix:
    """论文 Eq.1 的 X，形状与输入一致（genes x cells）。

    库大小按 cell（列）求和。非零模式与 C 完全相同：log1p 只在正值上作用，
    所以换成这个目标之后建图结构不变，改的只有重构目标本身。
    """
    counts = _as_csr_genes_by_cells(rna_matrix).tocsc(copy=True).astype(np.float64)
    library = np.asarray(counts.sum(axis=0)).ravel()
    library[library == 0] = 1.0
    # csc 的 indptr 按列切，直接按列缩放最省事
    for column in range(counts.shape[1]):
        start, end = counts.indptr[column], counts.indptr[column + 1]
        if end > start:
            counts.data[start:end] *= 1e4 / library[column]
    counts.data = np.log1p(counts.data)
    return counts.tocsr()


def gene_zscore(rna_matrix: Any) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """返回 (X, 每基因 mean, 每基因 std)。std 为 0 的基因用 1 兜底。"""
    x = paper_x_matrix(rna_matrix)
    n_cells = x.shape[1]
    total = np.asarray(x.sum(axis=1)).ravel()
    squared = np.asarray(x.multiply(x).sum(axis=1)).ravel()
    mean = total / n_cells
    variance = np.maximum(squared / n_cells - mean ** 2, 0.0)
    std = np.sqrt(variance)
    std[std == 0] = 1.0
    return x, mean, std


def topz_selection(rna_matrix: Any, k: int = 20, chunk: int = 256) -> dict[int, list[int]]:
    """论文 Eq.3：每个 spot 取 Z 最高的 k 个基因。

    确定性来自两处：Z 只由数据决定，排序用 stable argsort，所以并列的基因按索引
    升序取。换 seed 结果逐位相同——这正是公开代码那一版做不到的事。
    """
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or int(k) < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    k = int(k)
    x, mean, std = gene_zscore(rna_matrix)
    n_genes, n_cells = x.shape
    if k > n_genes:
        raise ValueError(f"k={k} exceeds the number of genes {n_genes}")
    x_cells = x.T.tocsr()                     # cells x genes，按行取更快
    selection: dict[int, list[int]] = {}
    for start in range(0, n_cells, int(chunk)):
        stop = min(start + int(chunk), n_cells)
        block = np.asarray(x_cells[start:stop].todense(), dtype=np.float64)
        z = (block - mean) / std
        # stable 排序保证并列时索引小的在前；取负值实现降序
        order = np.argsort(-z, axis=1, kind="stable")[:, :k]
        for offset in range(stop - start):
            selection[start + offset] = sorted(int(g) for g in order[offset])
    return selection


def paper_topz_batch_select(
    rna_matrix: Any,
    blocks: Sequence[np.ndarray],
    *,
    k: int = 20,
    save_path: str | None = None,
) -> tuple[list[dict[str, list[int]]], np.ndarray, dict[int, dict[str, Any]]]:
    """把确定性 Top-Z 选择装进 `batch_select_whole` 的输出契约。

    `blocks` 直接用 `spatial_batch_order` 的结果，这样"分批方式"这一项和公开代码
    那一版保持一致，唯一变的是每个 spot 选哪些基因。
    """
    matrix = _as_csr_genes_by_cells(rna_matrix)
    if save_path is not None:
        cache = os.path.join(save_path, "indices_ss.pkl")
        node_file = os.path.join(save_path, "Node_Ids.pkl")
        dic_file = os.path.join(save_path, "dic.pkl")
        if all(os.path.exists(path) for path in (cache, node_file, dic_file)):
            with open(cache, "rb") as handle:
                indices_ss = pickle.load(handle)
            with open(node_file, "rb") as handle:
                node_ids = pickle.load(handle)
            with open(dic_file, "rb") as handle:
                dic = pickle.load(handle)
            return indices_ss, node_ids, dic

    selection = topz_selection(matrix, k=k)
    node_ids = np.concatenate([np.asarray(b, dtype=np.int64) for b in blocks])
    indices_ss: list[dict[str, list[int]]] = []
    dic: dict[int, dict[str, Any]] = {}
    for block in blocks:
        genes: list[int] = []
        cells = [int(cell) for cell in np.asarray(block).reshape(-1)]
        for cell in cells:
            picked = selection[cell]
            dic[cell] = {"g": list(picked)}
            genes.extend(picked)
        indices_ss.append({"gene_index": sorted(set(genes)), "cell_index": cells})

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "indices_ss.pkl"), "wb") as handle:
            pickle.dump(indices_ss, handle)
        with open(os.path.join(save_path, "Node_Ids.pkl"), "wb") as handle:
            pickle.dump(node_ids, handle)
        with open(os.path.join(save_path, "dic.pkl"), "wb") as handle:
            pickle.dump(dic, handle)
    return indices_ss, node_ids, dic


def selection_edge_count(selection: dict[int, Sequence[int]],
                         batch: dict[str, Any]) -> int:
    """一个 batch 里按自选集建出来的有向边数（单向计数）。测试与落盘都用它。"""
    genes = set(int(g) for g in batch["gene_index"])
    total = 0
    for cell in batch["cell_index"]:
        picked = selection.get(int(cell))
        if picked is None:
            continue
        total += len(genes.intersection(int(g) for g in picked))
    return total
