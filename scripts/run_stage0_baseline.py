"""Stage 0 · A0 baseline：upstream scFormer 原样跑 human_lymph_node A1 RNA。

只用表达矩阵，**不使用空间坐标**（`obsm['spatial']` 完全不读）。
口径 = upstream 代码原样（`batch_select_whole` 丰度抽样 + `NDR_2` + `pred`），
超参照官方 Tutorial：cell_size 30 / n_hid 104 / 8 heads / 3 layers / AdamW 5e-4 /
wd 0.1 / labsm 0.1 / 100 epochs。

同时输出三样东西用于 G0 门：
1. ARI / NMI vs `obs['final_annot']`（10 类）
2. **每个 cluster 的细胞个数**（含占比、主要对应的真实类型、纯度）
3. Leiden 伪标签自己的 ARI —— 这是"sanity floor"：如果 scFormer 打不过它，
   后面 A/B/C/D 的比较就没有意义

用法:
    python scripts/run_stage0_baseline.py                    # 默认 100 epoch, seed 0
    python scripts/run_stage0_baseline.py --epochs 5 --tag smoke
"""

from __future__ import annotations

import argparse
import json
import os

# 必须在 import torch 之前设好，配合 use_deterministic_algorithms 才能逐位复现
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "scFormer"
DEFAULT_H5AD = ROOT / "data" / "raw" / "human_lymph_node_A1" / "adata_RNA.h5ad"
DEFAULT_RUNS = ROOT / "experiments" / "runs"

HP = dict(cell_size=30, n_hidden=104, n_heads=8, n_layers=3,
          lr=5e-4, weight_decay=0.1, labsm=0.1, epochs=100)


def set_global_seed(seed: int, deterministic: bool) -> None:
    import torch
    from torch.backends import cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True)


def cluster_table(pred: np.ndarray, truth: pd.Series) -> pd.DataFrame:
    """每个 cluster 的细胞个数 + 占比 + 最匹配的真实类型 + 纯度。"""
    df = pd.DataFrame({"cluster": pred, "truth": truth.astype(str).values})
    rows = []
    for c, sub in df.groupby("cluster"):
        vc = sub["truth"].value_counts()
        rows.append({
            "cluster": int(c),
            "n_cells": len(sub),
            "pct": round(100 * len(sub) / len(df), 2),
            "top_truth": vc.index[0],
            "top_n": int(vc.iloc[0]),
            "purity": round(vc.iloc[0] / len(sub), 3),
            "n_truth_types": int(sub["truth"].nunique()),
        })
    return pd.DataFrame(rows).sort_values("n_cells", ascending=False).reset_index(drop=True)


def rare_scores(pred: np.ndarray, truth: pd.Series,
                ref_thr: float = 0.05, pred_thr: float = 0.045) -> dict:
    """官方口径：真稀有 = 真实类型占比 < 5%；预测稀有 = 簇占比 < 4.5%。"""
    from sklearn.metrics import f1_score, precision_score, recall_score

    tprop = truth.value_counts(normalize=True)
    rare_types = tprop[tprop < ref_thr].index
    y_true = truth.isin(rare_types).astype(int).values

    s = pd.Series(pred)
    cprop = s.value_counts(normalize=True)
    rare_clusters = cprop[cprop < pred_thr].index
    y_pred = s.isin(rare_clusters).astype(int).values

    return {
        "rare_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "rare_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "rare_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_true_rare": int(y_true.sum()),
        "n_pred_rare": int(y_pred.sum()),
        "rare_types": [str(t) for t in rare_types],
        "n_rare_clusters": int(len(rare_clusters)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    ap.add_argument("--label-key", default="final_annot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=HP["epochs"])
    ap.add_argument("--cell-size", type=int, default=HP["cell_size"])
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    sys.path.insert(0, str(UPSTREAM))
    import scanpy as sc
    import torch
    from scipy.sparse import csr_matrix
    from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

    from scformer.model import NDR_2, pred
    from scformer.utils import batch_select_whole, initial_clustering

    run_id = f"stage0_A0_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out = args.runs_root / run_id
    (out / "graph_preprocess").mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed, args.deterministic)
    t0 = time.time()

    adata = sc.read_h5ad(args.h5ad)
    truth = adata.obs[args.label_key].astype(str)
    print(f"[info] {args.h5ad.name}: {adata.shape[0]} spots x {adata.shape[1]} genes")
    print(f"[info] 不使用空间坐标（obsm 里有 {list(adata.obsm)} 但一律不读）")
    print(f"[info] ground truth `{args.label_key}` {truth.nunique()} 类:")
    print(truth.value_counts().to_string())

    adata.X = csr_matrix(adata.X)
    rna_matrix = adata.X.T  # gene x cell，与官方 Tutorial 完全一致

    print("\n[info] initial_clustering (Leiden 伪标签) ...")
    leiden = [int(x) for x in initial_clustering(rna_matrix)]
    ari_leiden = float(adjusted_rand_score(truth, leiden))
    nmi_leiden = float(normalized_mutual_info_score(truth, leiden))
    print(f"[info] 伪标签簇数 = {len(set(leiden))}, ARI={ari_leiden:.4f}, NMI={nmi_leiden:.4f}")

    indices, node_ids, _ = batch_select_whole(
        rna_matrix, cell_size=args.cell_size, save_path=str(out / "graph_preprocess")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] n_batch = {len(indices)}, device = {device}, "
          f"deterministic = {args.deterministic}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = NDR_2(
        rna_matrix, indices, leiden,
        n_hid=HP["n_hidden"], n_heads=HP["n_heads"], n_layers=HP["n_layers"],
        labsm=HP["labsm"], lr=HP["lr"], wd=HP["weight_decay"],
        device=device, num_types=2, num_relations=2, epochs=args.epochs,
    )
    gnn, _, _, _ = model.train_model(n_batch=len(indices))
    train_sec = time.time() - t0
    model.save_model(str(out / "node_model.pth"))

    results = pred(rna_matrix, gnn=gnn, indices=indices, device=device)
    # pred 的输出按 batch 顺序（= Node_Ids 顺序），要还原回 adata 的原始行序
    order = np.argsort(node_ids)
    pred_labels = np.asarray(results["pred_label"])[order]
    embedding = np.asarray(results["cell_embedding"])[order]

    np.save(out / "Node_Ids.npy", node_ids)
    np.save(out / "pred.npy", pred_labels)
    np.save(out / "cell_embedding.npy", embedding)
    np.save(out / "leiden_pseudolabels.npy", np.asarray(leiden))

    ari = float(adjusted_rand_score(truth, pred_labels))
    nmi = float(normalized_mutual_info_score(truth, pred_labels))
    tbl = cluster_table(pred_labels, truth)
    tbl.to_csv(out / "cluster_sizes.csv", index=False)
    rare = rare_scores(pred_labels, truth)
    rare_leiden = rare_scores(np.asarray(leiden), truth)
    peak = (torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0

    payload = {
        "run_id": run_id,
        "stage": "0", "experiment": "A0 (upstream 口径, 无空间坐标)",
        "h5ad": str(args.h5ad), "label_key": args.label_key,
        "n_spots": int(adata.shape[0]), "n_genes": int(adata.shape[1]),
        "use_spatial": False,
        "seed": args.seed, "deterministic": bool(args.deterministic),
        "hyperparams": {**HP, "epochs": args.epochs, "cell_size": args.cell_size},
        "n_batch": len(indices),
        "scformer": {"ari": ari, "nmi": nmi, "n_clusters": int(len(set(pred_labels))), **rare},
        "leiden_baseline": {"ari": ari_leiden, "nmi": nmi_leiden,
                            "n_clusters": len(set(leiden)), **rare_leiden},
        "sanity_floor_passed": bool(ari >= ari_leiden),
        "truth_distribution": {str(k): int(v) for k, v in truth.value_counts().items()},
        "runtime_sec": round(time.time() - t0, 1), "train_sec": round(train_sec, 1),
        "peak_gpu_mem_gb": round(peak, 3),
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                     encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"run_id : {run_id}   （只用表达矩阵，未使用空间坐标）")
    print("-" * 78)
    print(f"{'':22s} {'ARI':>8s} {'NMI':>8s} {'簇数':>6s} {'rare-F1':>9s} {'rare-P':>8s} {'rare-R':>8s}")
    print(f"{'scFormer (A0)':22s} {ari:8.4f} {nmi:8.4f} "
          f"{len(set(pred_labels)):6d} {rare['rare_f1']:9.4f} "
          f"{rare['rare_precision']:8.4f} {rare['rare_recall']:8.4f}")
    print(f"{'Leiden 伪标签(floor)':22s} {ari_leiden:8.4f} {nmi_leiden:8.4f} "
          f"{len(set(leiden)):6d} {rare_leiden['rare_f1']:9.4f} "
          f"{rare_leiden['rare_precision']:8.4f} {rare_leiden['rare_recall']:8.4f}")
    print("-" * 78)
    print(f"G0.4 sanity floor（scFormer 是否 >= Leiden）: "
          f"{'PASS' if payload['sanity_floor_passed'] else 'FAIL'}")
    print(f"\n每个 cluster 的细胞个数（共 {len(tbl)} 个簇）:")
    print(tbl.to_string(index=False))
    print(f"\n真稀有类（占比<5%，共 {rare['n_true_rare']} 个 spot）: {rare['rare_types']}")
    print(f"用时 {payload['runtime_sec']}s (训练 {payload['train_sec']}s), 峰值显存 {payload['peak_gpu_mem_gb']} GB")
    print(f"产物: {out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



