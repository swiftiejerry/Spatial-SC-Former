"""Stage 1 runner: A (expression only) and B (expression + spot-spot spatial).

Blueprint alignment (modification #1):
  - spatial edges enter HGT as a third relation type, not as a post-hoc overlay
  - first version uses unweighted kNN edges, k comes from the CLI, no L_spatial
  - nothing else changes: Top-Z gene selection, pseudo-labels, losses and
    hyper-parameters stay at the A0 baseline values

Both arms use spatially contiguous batches with a one-hop halo so that the
spatial relation is actually reachable inside a batch; Stage 1a measured 0.78%
edge retention under the upstream shuffle versus 100% here.  Running arm A on the
same sampler is what isolates the sampler's own effect from the relation's.

Usage:
  $env:PYTHONNOUSERSITE='1'; $env:PYTHONIOENCODING='utf-8'
  python scripts/run_stage1.py --arm A --seed 0
"""

from __future__ import annotations

import argparse
import json
import os

# Must precede torch import for use_deterministic_algorithms to work.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/scFormer"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(UPSTREAM))

DEFAULT_H5AD = ROOT / "data/raw/human_lymph_node_A1/adata_RNA.h5ad"
RUNS = ROOT / "experiments/runs"
PROTOCOL = ROOT / "configs" / "mainline_protocol.json"

# A0 baseline hyper-parameters, held fixed across Stage 1.
HP = dict(cell_size=30, n_hidden=104, n_heads=8, n_layers=3,
          lr=5e-4, weight_decay=0.1, labsm=0.1, epochs=100)


def set_global_seed(seed: int, deterministic: bool) -> None:
    import torch
    from torch.backends import cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True)


def cluster_table(pred: np.ndarray, truth: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"cluster": pred, "truth": truth.astype(str).values})
    rows = []
    for cluster, sub in frame.groupby("cluster"):
        counts = sub["truth"].value_counts()
        rows.append({
            "cluster": int(cluster),
            "n_cells": len(sub),
            "pct": round(100 * len(sub) / len(frame), 2),
            "top_truth": counts.index[0],
            "top_n": int(counts.iloc[0]),
            "purity": round(counts.iloc[0] / len(sub), 3),
            "n_truth_types": int(sub["truth"].nunique()),
        })
    return pd.DataFrame(rows).sort_values("n_cells", ascending=False).reset_index(drop=True)


def rare_scores(pred: np.ndarray, truth: pd.Series,
                ref_thr: float = 0.05, pred_thr: float = 0.045) -> dict:
    """A0 definition: true rare = truth class share < 5%, predicted rare = cluster share < 4.5%."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    truth_share = truth.value_counts(normalize=True)
    rare_types = truth_share[truth_share < ref_thr].index
    y_true = truth.isin(rare_types).astype(int).values

    series = pd.Series(pred)
    cluster_share = series.value_counts(normalize=True)
    rare_clusters = cluster_share[cluster_share < pred_thr].index
    y_pred = series.isin(rare_clusters).astype(int).values

    return {
        "rare_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "rare_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "rare_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_true_rare": int(y_true.sum()),
        "n_pred_rare": int(y_pred.sum()),
        "rare_types": [str(t) for t in rare_types],
        "n_rare_clusters": int(len(rare_clusters)),
    }


def per_rare_type_recall(pred: np.ndarray, truth: pd.Series,
                         ref_thr: float = 0.05, pred_thr: float = 0.045) -> dict:
    """Per-class recall for each rare type; aggregate rare-F1 hides single classes."""
    truth_share = truth.value_counts(normalize=True)
    rare_types = truth_share[truth_share < ref_thr].index
    series = pd.Series(pred)
    cluster_share = series.value_counts(normalize=True)
    rare_clusters = set(cluster_share[cluster_share < pred_thr].index)
    truth_values = truth.astype(str).values
    result = {}
    for rare_type in rare_types:
        mask = truth_values == str(rare_type)
        total = int(mask.sum())
        hit = int(np.isin(pred[mask], list(rare_clusters)).sum()) if rare_clusters else 0
        result[str(rare_type)] = {
            "n_cells": total,
            "n_in_rare_cluster": hit,
            "recall": (hit / total) if total else 0.0,
        }
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 上游 segment_function 对 3484 个 spot 给的是 (resolution 0.5, n_neighbors 10)。
UPSTREAM_N_NEIGHBORS = 10


def upstream_default_resolution(n_spots: int) -> float:
    """镜像 scformer.utils.initial_clustering 里的 segment_function，只为把实际生效的
    resolution 写进 metrics.json——否则日志里只有一句 print，事后无法核对。"""
    if n_spots <= 500:
        return 0.2
    if n_spots <= 5000:
        return 0.5
    return 0.8


def upstream_pseudo_labels(rna_matrix, resolution: float | None = None,
           n_neighbors: int = UPSTREAM_N_NEIGHBORS) -> list[int]:
    """A0 那条路产生的 Leiden 伪标签；resolution=None 就是上游原封不动的默认。

    换 resolution 必须把 n_neighbors 一起显式传进去。initial_clustering 的判断是
    `if custom_resolution is None or custom_n_neighbors is None:` 才走 segment_function，
    只给 custom_resolution 会被整条忽略掉，run 会安静地跑成 0.5，看日志根本看不出来。
    显式传 n_neighbors=10 也保证"只动 resolution 一个旋钮"，别的预处理和默认那条路一致。
    """
    from scformer.utils import initial_clustering

    if resolution is None:
        return [int(x) for x in initial_clustering(rna_matrix)]
    return [int(x) for x in initial_clustering(
        rna_matrix,
        custom_resolution=float(resolution),
        custom_n_neighbors=int(n_neighbors),
    )]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["A", "B"], required=True,
                        help="A = expression only, B = expression + spot-spot spatial")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spatial-k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=HP["epochs"])
    parser.add_argument("--cell-size", type=int, default=HP["cell_size"])
    parser.add_argument("--pseudo-resolution", type=float, default=None,
                        help="上游 initial_clustering 的 Leiden resolution；省略就是上游默认 0.5。"
                             "0.5 结构上不产生低于 4.5% 判据的小簇，rare-F1 在这一支上恒为 0；"
                             "0.8 会给出一个 102 spot 的簇（73.5% 是稀有类）")
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--label-key", default="final_annot")
    parser.add_argument("--graph", choices=["upstream_sampled", "paper_topz"],
                        default="upstream_sampled",
                        help="upstream_sampled 是公开代码那一版（softmax(log(counts+1)) 随机抽 20，"
                             "与论文 Top-20 Z 的重合实测 1.0-1.6%）；paper_topz 是论文 Eq.3-4 的"
                             "确定性 Top-K，并且边只连该 spot 自己选中的基因")
    parser.add_argument("--topz-k", type=int, default=20,
                        help="论文主分析固定 K=20，只在 --graph paper_topz 下生效")
    parser.add_argument("--kl-target", choices=["counts", "paper_x"], default="counts",
                        help="counts 是公开代码那一版（对 raw counts 直接 softmax，实测 "
                             "cell->gene 方向是 one-hot）；paper_x 换成论文 Eq.1 的 X")
    parser.add_argument("--gene-input", choices=["counts", "paper_z"], default="counts",
                        help="counts 是公开代码那一版（gene 节点吃 raw counts 行）；"
                             "paper_z 换成论文 2.1 的 Z[:, j]。实测两类节点输入的 L2 中位"
                             "是 135.4 对 15.6，量级不对等")
    parser.add_argument("--runs-root", type=Path, default=RUNS)
    parser.add_argument("--tag", default="")
    parser.add_argument("--protocol-file", type=Path, default=None,
                        help="正式协议快照；传入后会把路径、SHA-256 和完整 argv 写入 metrics.json")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    import scanpy as sc
    import torch
    from scipy.sparse import csr_matrix
    from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

    from scformer.utils import initial_clustering
    from spatial_scformer.graph.spatial import build_spatial_graph
    from spatial_scformer.graph.paper_topz import paper_topz_batch_select, selection_edge_count
    from spatial_scformer.graph.spatial_batch import (
        add_spatial_halo,
        core_incident_edge_retention,
        spatial_batch_order,
        spatial_batch_select_whole,
    )
    from spatial_scformer.stage1_trainer import Stage1Config, Stage1Model

    protocol_snapshot = None
    if args.protocol_file is not None:
        protocol_path = args.protocol_file
        if not protocol_path.is_absolute():
            protocol_path = ROOT / protocol_path
        if not protocol_path.exists():
            raise SystemExit(f"protocol file not found: {protocol_path}")
        protocol_snapshot = {
            "name": json.loads(protocol_path.read_text(encoding="utf-8"))["protocol"]["name"],
            "path": str(protocol_path),
            "sha256": file_sha256(protocol_path),
            "argv": list(sys.argv),
        }

    spatial_relation = args.arm == "B"
    suffix = f"_k{args.spatial_k}" if spatial_relation else ""
    if args.pseudo_resolution is not None:
        # 伪标签的 resolution 换了就是换了一份训练目标，必须进 run_id，不然两份不同的
        # 监督信号会写进同一个目录。
        suffix += f"_pr{args.pseudo_resolution:g}"
    if args.graph == "paper_topz":
        # 建图口径必须进 run_id：两种选基因方式产出的是两张完全不同的图。
        suffix += f"_topz{args.topz_k}"
    if args.kl_target == "paper_x":
        suffix += "_klX"
    if args.gene_input == "paper_z":
        suffix += "_gz"
    run_id = f"stage1_{args.arm}{suffix}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out = args.runs_root / run_id
    (out / "graph_preprocess").mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed, args.deterministic)
    started = time.time()

    adata = sc.read_h5ad(args.h5ad)
    truth = adata.obs[args.label_key].astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    adata.X = csr_matrix(adata.X)
    rna_matrix = adata.X.T  # gene x cell, upstream convention
    n_cells = int(adata.n_obs)
    print(f"[info] {args.h5ad.name}: {n_cells} spots x {adata.shape[1]} genes, arm {args.arm}")

    res_note = ("上游默认 0.5" if args.pseudo_resolution is None
                else f"resolution {args.pseudo_resolution:g}")
    print(f"[info] initial_clustering (Leiden pseudo-labels, {res_note}) ...")
    leiden = upstream_pseudo_labels(rna_matrix, args.pseudo_resolution)
    ari_leiden = float(adjusted_rand_score(truth, leiden))
    nmi_leiden = float(normalized_mutual_info_score(truth, leiden))
    print(f"[info] pseudo K={len(set(leiden))} ARI={ari_leiden:.4f} NMI={nmi_leiden:.4f}")

    # Spatial graph is always built: arm A needs it for the halo so that both arms
    # see identical batch composition and only the relation differs.
    graph = build_spatial_graph(coords, mode="knn", k=args.spatial_k, weighted=False)
    n_batches = math.ceil(n_cells / int(args.cell_size))
    blocks = spatial_batch_order(coords, n_batches=n_batches)
    halo = add_spatial_halo(blocks, graph.edge_index)
    retention = core_incident_edge_retention(graph.edge_index, halo)

    if args.graph == "paper_topz":
        gene_batches, node_ids, gene_dic = paper_topz_batch_select(
            rna_matrix, blocks, k=args.topz_k, save_path=str(out / "graph_preprocess"),
        )
    else:
        gene_batches, node_ids, gene_dic = spatial_batch_select_whole(
            rna_matrix, coords, cell_size=args.cell_size,
            save_path=str(out / "graph_preprocess"),
        )
    selection = {int(cell): [int(g) for g in info["g"]] for cell, info in gene_dic.items()}
    if len(gene_batches) != len(halo):
        raise SystemExit("gene batches and halo batches disagree on batch count")
    batches = []
    for gene_batch, halo_batch in zip(gene_batches, halo):
        if [int(c) for c in gene_batch["cell_index"]] != list(halo_batch["core_index"]):
            raise SystemExit("core cell order mismatch between gene and halo batches")
        batches.append({
            "gene_index": gene_batch["gene_index"],
            "cell_index": halo_batch["cell_index"],
            "core_index": halo_batch["core_index"],
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] batches={len(batches)} device={device} spatial_relation={spatial_relation}")
    print(f"[info] spatial edges k={args.spatial_k}: {retention['n_edges']} directed, "
          f"core-incident retention={retention['retention']:.4f}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    config = Stage1Config(
        n_hidden=HP["n_hidden"], n_heads=HP["n_heads"], n_layers=HP["n_layers"],
        labsm=HP["labsm"], lr=HP["lr"], weight_decay=HP["weight_decay"],
        epochs=args.epochs, spatial_relation=spatial_relation,
        kl_target=args.kl_target,
        gene_input=args.gene_input,
        edge_source=("per_spot_selection" if args.graph == "paper_topz" else "batch_nonzero"),
    )
    model = Stage1Model(
        rna_matrix, batches, leiden, config, device,
        spatial_edge_index=graph.edge_index if spatial_relation else None,
        selection=selection,
    )
    # 实际喂进 HGT 的 spot-gene 边数：论文是每 cell 恰好 K 条，公开代码那一版实测中位 156 条。
    # 这个数必须随 run 落盘，否则事后没法判断这一支到底是哪张图。
    csr = rna_matrix.tocsr()
    if args.graph == "paper_topz":
        edge_counts = [selection_edge_count(selection, batch) for batch in batches]
        core_edge_counts = [
            selection_edge_count(selection, {"gene_index": batch["gene_index"],
                                             "cell_index": batch["core_index"]})
            for batch in batches
        ]
    else:
        edge_counts = [int((csr[batch["gene_index"], :][:, batch["cell_index"]] > 0).sum())
                       for batch in batches]
        core_edge_counts = [int((csr[batch["gene_index"], :][:, batch["core_index"]] > 0).sum())
                            for batch in batches]
    per_spot = [count / max(len(batch["cell_index"]), 1)
                for count, batch in zip(edge_counts, batches)]
    per_core = [count / max(len(batch["core_index"]), 1)
                for count, batch in zip(core_edge_counts, batches)]
    graph_contract = {
        "gene_selection": args.graph,
        "topz_k": int(args.topz_k) if args.graph == "paper_topz" else None,
        "edge_source": config.edge_source,
        "kl_target": config.kl_target,
        "gene_input": config.gene_input,
        "spot_gene_edges_total": int(sum(edge_counts)),
        "spot_gene_edges_per_spot_median": float(np.median(per_spot)),
        # 论文口径是"每 cell 恰好 K 条"，只有 core spot 才有对照意义：
        # halo spot 的自选基因大多不在这个 batch 的并集里，会把均值拉低。
        "spot_gene_edges_per_core_spot_median": float(np.median(per_core)),
        "batch_gene_union_median": float(np.median([len(b["gene_index"]) for b in batches])),
    }
    print(f"[info] 建图口径 {args.graph} / KL 目标 {args.kl_target}：每 spot 边数中位 "
          f"{graph_contract['spot_gene_edges_per_spot_median']:.1f}（每 core spot "
          f"{graph_contract['spot_gene_edges_per_core_spot_median']:.1f}），"
          f"batch gene 并集中位 {graph_contract['batch_gene_union_median']:.0f}")
    edge_stats = model.spatial_edge_stats()
    history = model.train()
    train_sec = time.time() - started
    torch.save(model.gnn.state_dict(), out / "node_model.pth")

    result = model.predict()
    pred_labels = result["pred_label"]
    np.save(out / "pred.npy", pred_labels)
    np.save(out / "cell_embedding.npy", result["cell_embedding"])
    np.save(out / "leiden_pseudolabels.npy", np.asarray(leiden))
    np.save(out / "Node_Ids.npy", node_ids)

    ari = float(adjusted_rand_score(truth, pred_labels))
    nmi = float(normalized_mutual_info_score(truth, pred_labels))
    ari_vs_pseudo = float(adjusted_rand_score(leiden, pred_labels))
    table = cluster_table(pred_labels, truth)
    table.to_csv(out / "cluster_sizes.csv", index=False)
    rare = rare_scores(pred_labels, truth)
    rare_leiden = rare_scores(np.asarray(leiden), truth)
    per_type = per_rare_type_recall(pred_labels, truth)
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == "cuda" else 0.0

    payload = {
        "run_id": run_id,
        "protocol": protocol_snapshot,
        "stage": "1",
        "arm": args.arm,
        "experiment": ("A (expression only, spatial batching)" if args.arm == "A"
                       else f"B (expression + spot-spot kNN k={args.spatial_k})"),
        "h5ad": str(args.h5ad),
        "h5ad_sha256": file_sha256(args.h5ad),
        "label_key": args.label_key,
        "n_spots": n_cells,
        "n_genes": int(adata.shape[1]),
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "hyperparams": {**HP, "epochs": args.epochs, "cell_size": args.cell_size},
        "graph_contract": graph_contract,
        "batching": {
            "sampler": "spatial_median_bisection_with_one_hop_halo",
            "n_batches": len(batches),
            "core_sizes": {
                "min": int(min(len(b["core_index"]) for b in batches)),
                "max": int(max(len(b["core_index"]) for b in batches)),
            },
            "batch_sizes_with_halo": {
                "min": int(min(len(b["cell_index"]) for b in batches)),
                "max": int(max(len(b["cell_index"]) for b in batches)),
            },
        },
        "spatial": {
            "relation_enabled": spatial_relation,
            "num_relations": config.num_relations,
            "mode": "knn",
            "k": args.spatial_k,
            "weighted": False,
            "n_directed_edges": int(retention["n_edges"]),
            "core_incident_retention": retention["retention"],
            **edge_stats,
        },
        "scformer": {"ari": ari, "nmi": nmi,
                     "n_clusters": int(len(set(pred_labels.tolist()))),
                     "ari_vs_pseudo": ari_vs_pseudo, **rare},
        "rare_per_type": per_type,
        "leiden_baseline": {"ari": ari_leiden, "nmi": nmi_leiden,
                            "n_clusters": len(set(leiden)),
                            "resolution": (float(args.pseudo_resolution)
                                           if args.pseudo_resolution is not None
                                           else upstream_default_resolution(n_cells)),
                            "resolution_source": ("cli --pseudo-resolution"
                                                  if args.pseudo_resolution is not None
                                                  else "upstream segment_function"),
                            "n_neighbors": UPSTREAM_N_NEIGHBORS,
                            **rare_leiden},
        "truth_distribution": {str(k): int(v) for k, v in truth.value_counts().items()},
        "history_tail": history[-5:],
        "runtime_sec": round(time.time() - started, 1),
        "train_sec": round(train_sec, 1),
        "peak_gpu_mem_gb": round(peak, 3),
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
    }
    (out / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 78)
    print(f"run_id: {run_id}")
    print(f"{'':16s} {'ARI':>8s} {'NMI':>8s} {'K':>4s} {'rare-F1':>9s} {'rare-R':>8s}")
    print(f"{'this run':16s} {ari:8.4f} {nmi:8.4f} "
          f"{len(set(pred_labels.tolist())):4d} {rare['rare_f1']:9.4f} {rare['rare_recall']:8.4f}")
    print(f"{'Leiden floor':16s} {ari_leiden:8.4f} {nmi_leiden:8.4f} "
          f"{len(set(leiden)):4d} {rare_leiden['rare_f1']:9.4f} {rare_leiden['rare_recall']:8.4f}")
    print(f"ARI vs pseudo = {ari_vs_pseudo:.4f}")
    print(f"spatial edges inside batches = {edge_stats['total_spatial_edges_in_batches']}")
    print(f"runtime {payload['runtime_sec']}s, peak GPU {payload['peak_gpu_mem_gb']} GB")
    print(f"artifacts -> {out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
