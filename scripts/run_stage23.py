"""Stage 2 / Stage 3 runner: prototype dynamic clustering, with or without spatial.

Blueprint alignment (modification #3 and Stage 3):
  arm C = expression only + prototype dynamic clustering
  arm D = C + spot-spot spatial relation (Full)

Both arms reuse the Stage 1 batching and hyper-parameters so the only difference
from Stage 1 is the clustering objective, and the only difference between C and D
is the spatial relation.  ``delta`` is swept rather than assumed, because the
blueprint marks 0.8 as a first test value only.

Usage:
  $env:PYTHONNOUSERSITE='1'; $env:PYTHONIOENCODING='utf-8'
  python scripts/run_stage23.py --arm C --seed 0
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/scFormer"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_H5AD = ROOT / "data/raw/human_lymph_node_A1/adata_RNA.h5ad"
RUNS = ROOT / "experiments/runs"
PROTOCOL = ROOT / "configs" / "mainline_protocol.json"

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["C", "D"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spatial-k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=HP["epochs"])
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--update-interval", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.8)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--lambda-proto", type=float, default=1.0)
    parser.add_argument("--target-k", type=int, default=None,
                        help="label-free Leiden resolution scan target; omit to use a fixed resolution")
    parser.add_argument("--leiden-resolution", type=float, default=None,
                        help="resolution of the epoch-warmup Leiden on H_spot; default 0.5 is the A0 setting")
    parser.add_argument("--free-k0", action="store_true",
                        help="let the epoch-warmup Leiden pick K0 itself instead of matching the A0 K")
    parser.add_argument("--no-preserve-k", action="store_true",
                        help="reproduce the v1 bug where an emptied cluster keeps a zero prototype and dies")
    parser.add_argument("--no-center-embedding", action="store_true",
                        help="measure cos(h_i, c_k) from the raw origin, which is what collapsed "
                             "the q range and switched L_proto off in the first three runs")
    parser.add_argument("--delta-mode", choices=["quantile", "absolute"], default="quantile",
                        help="quantile re-derives delta from the current q distribution each "
                             "refresh; absolute is the literal fixed-delta reading")
    parser.add_argument("--delta-quantile", type=float, default=0.2,
                        help="share of least-confident spots left out of L_proto when "
                             "--delta-mode quantile")
    parser.add_argument("--init-source", choices=["warmup_leiden", "a0_pseudo"],
                        default="warmup_leiden",
                        help="where the first assignment comes from. warmup_leiden is the "
                             "blueprint recipe (Leiden on H_spot at the end of warm-up); "
                             "a0_pseudo seeds it from the A0 expression-space Leiden labels "
                             "because the warm-up representation measured ARI 0.0006 on A1. "
                             "Either way the labels are refreshed every update-interval "
                             "epochs and never enter a loss.")
    parser.add_argument("--probe-every", type=int, default=0,
                        help="dump the spot embedding every N warm-up epochs so one run "
                             "shows whether a longer warm-up helps at all")
    parser.add_argument("--pseudo-resolution", type=float, default=None,
                        help="上游 initial_clustering 的 Leiden resolution，省略即上游默认 0.5。"
                             "它同时决定参照伪标签、K0（--target-k / --free-k0 未指定时）和 "
                             "--init-source a0_pseudo 的初始 assignment，所以换成 0.8 时 A 组与 "
                             "C 组仍然只差『固定 vs 动态』一个因素。0.5 结构上不产生低于 4.5% "
                             "判据的小簇（rare-F1 在这一支上恒为 0），0.8 会给出一个 102 spot、"
                             "73.5% 是稀有类的小簇")
    parser.add_argument("--cell-size", type=int, default=HP["cell_size"])
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--label-key", default="final_annot")
    parser.add_argument("--runs-root", type=Path, default=RUNS)
    parser.add_argument("--tag", default="")
    parser.add_argument("--protocol-file", type=Path, default=None,
                        help="正式协议快照；传入后会把路径、SHA-256 和完整 argv 写入 metrics.json")
    parser.add_argument("--graph", choices=["upstream_sampled", "paper_topz"],
                        default="upstream_sampled",
                        help="见 run_stage1.py 的同名开关：paper_topz 是论文 Eq.3-4 的确定性 "
                             "Top-K，并且边只连该 spot 自己选中的基因")
    parser.add_argument("--topz-k", type=int, default=20)
    parser.add_argument("--kl-target", choices=["counts", "paper_x"], default="counts",
                        help="paper_x 把重构目标换成论文 Eq.1 的 X。arm C/D 的 warm-up 只有 "
                             "L_KL，所以这一项直接决定 warm-up 表示里有没有信号")
    parser.add_argument("--gene-input", choices=["counts", "paper_z"], default="counts",
                        help="paper_z 把 gene 节点输入换成论文 2.1 的 Z[:, j]（现状是 raw counts 行）")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    import scanpy as sc
    import torch
    from scipy.sparse import csr_matrix
    from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

    stage1_runner = __import__("run_stage1")
    cluster_table = stage1_runner.cluster_table
    rare_scores = stage1_runner.rare_scores
    per_rare_type_recall = stage1_runner.per_rare_type_recall
    file_sha256 = stage1_runner.file_sha256

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

    from spatial_scformer.graph.spatial import build_spatial_graph
    from spatial_scformer.graph.paper_topz import paper_topz_batch_select, selection_edge_count
    from spatial_scformer.graph.spatial_batch import (
        add_spatial_halo,
        core_incident_edge_retention,
        spatial_batch_order,
        spatial_batch_select_whole,
    )
    from spatial_scformer.stage2_trainer import Stage2Config, Stage2Model

    spatial_relation = args.arm == "D"
    parts = [f"stage{'3' if spatial_relation else '2'}_{args.arm}"]
    if spatial_relation:
        parts.append(f"k{args.spatial_k}")
    parts.append(f"t{args.tau:g}")
    # The run_id has to say which reading of delta was in force, otherwise a quantile
    # run and a fixed-delta run collide on the same directory name.
    if args.delta_mode == "quantile":
        parts.append(f"dq{args.delta_quantile:g}")
    else:
        parts.append(f"d{args.delta:g}")
    parts.append(f"seed{args.seed}")
    if args.init_source != "warmup_leiden":
        # 初始 assignment 的来源必须进 run_id，否则两条不同的初始化路径会写进同一个目录。
        parts.append("initA0")
    if args.pseudo_resolution is not None:
        # 伪标签的 resolution 决定初始划分里有没有小簇，是这一支的主变量，必须进 run_id。
        parts.append(f"pr{args.pseudo_resolution:g}")
    if args.graph == "paper_topz":
        parts.append(f"topz{args.topz_k}")
    if args.kl_target == "paper_x":
        parts.append("klX")
    if args.gene_input == "paper_z":
        parts.append("gz")
    run_id = "_".join(parts) + (f"_{args.tag}" if args.tag else "")
    out = args.runs_root / run_id
    (out / "graph_preprocess").mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed, args.deterministic)
    started = time.time()

    adata = sc.read_h5ad(args.h5ad)
    truth = adata.obs[args.label_key].astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    adata.X = csr_matrix(adata.X)
    rna_matrix = adata.X.T
    n_cells = int(adata.n_obs)
    print(f"[info] {args.h5ad.name}: {n_cells} spots x {adata.shape[1]} genes, arm {args.arm}")

    # The A0 Leiden pseudo labels are still computed, but only as the reference the
    # dynamic labels are compared against; they do not drive any loss here.
    print("[info] reference A0 Leiden pseudo-labels (not used as a training target) ...")
    leiden = stage1_runner.上游伪标签(rna_matrix, args.pseudo_resolution)
    ari_leiden = float(adjusted_rand_score(truth, leiden))
    # K0 is a real design choice and the blueprint does not pin it, so it is explicit here.
    # Default matches the baseline's K: then C vs A differs only in how labels are produced,
    # which is what "一次只改一个主要因素" requires.  --free-k0 lets the warm-up Leiden pick K0
    # itself (resolution 0.5 gave K0=4 in the preflight, i.e. coarser than the baseline), and
    # --target-k forces a specific K0 for the sensitivity arm.  Whichever is used, the run
    # records cluster_init.selection so the report can never be ambiguous about it.
    if args.target_k is not None:
        target_k = args.target_k
        k0_source = f"forced target_k={target_k}"
    elif args.free_k0:
        target_k = None
        k0_source = "free (Leiden on H_spot decides)"
    else:
        target_k = len(set(leiden))
        k0_source = f"matched to A0 K={target_k}"
    print(f"[info] A0 pseudo K={len(set(leiden))} ARI={ari_leiden:.4f}; dynamic K0: {k0_source}")

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
            rna_matrix, coords, cell_size=args.cell_size, save_path=str(out / "graph_preprocess")
        )
    selection = {int(cell): [int(g) for g in info["g"]] for cell, info in gene_dic.items()}
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    config = Stage2Config(
        n_hidden=HP["n_hidden"], n_heads=HP["n_heads"], n_layers=HP["n_layers"],
        labsm=HP["labsm"], lr=HP["lr"], weight_decay=HP["weight_decay"],
        epochs=args.epochs, spatial_relation=spatial_relation,
        warmup_epochs=args.warmup_epochs, update_interval=args.update_interval,
        delta=args.delta, tau=args.tau, lambda_proto=args.lambda_proto,
        target_k=target_k, leiden_resolution=args.leiden_resolution, seed=args.seed,
        preserve_k=not args.no_preserve_k,
        center_embedding=not args.no_center_embedding,
        delta_mode=args.delta_mode, delta_quantile=args.delta_quantile,
        init_source=args.init_source,
        probe_every=args.probe_every, probe_dir=str(out),
        kl_target=args.kl_target,
        gene_input=args.gene_input,
        edge_source=("per_spot_selection" if args.graph == "paper_topz" else "batch_nonzero"),
    )
    model = Stage2Model(
        rna_matrix, batches, leiden, config, device,
        spatial_edge_index=graph.edge_index if spatial_relation else None,
        selection=selection,
    )
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
        # 论文口径是"每 cell 恰好 K 条"，只有 core spot 才有对照意义。
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
    if model.initial_assignment is None:
        raise RuntimeError("dynamic clustering did not record its warm-up initial assignment")
    initial_labels = np.asarray(model.initial_assignment, dtype=np.int64)
    np.save(out / "pred.npy", pred_labels)
    np.save(out / "warmup_init_labels.npy", initial_labels)
    np.save(out / "cell_embedding.npy", result["cell_embedding"])
    np.save(out / "confidence.npy", result["confidence"])
    np.save(out / "leiden_pseudolabels.npy", np.asarray(leiden))
    np.save(out / "Node_Ids.npy", node_ids)

    ari = float(adjusted_rand_score(truth, pred_labels))
    nmi = float(normalized_mutual_info_score(truth, pred_labels))
    ari_vs_pseudo = float(adjusted_rand_score(leiden, pred_labels))
    table = cluster_table(pred_labels, truth)
    table.to_csv(out / "cluster_sizes.csv", index=False)
    rare = rare_scores(pred_labels, truth)
    rare_leiden = rare_scores(np.asarray(leiden), truth)
    rare_initial = rare_scores(initial_labels, truth)
    ari_initial = float(adjusted_rand_score(truth, initial_labels))
    nmi_initial = float(normalized_mutual_info_score(truth, initial_labels))
    per_type = per_rare_type_recall(pred_labels, truth)
    confidence = result["confidence"]
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == "cuda" else 0.0

    payload = {
        "run_id": run_id,
        "protocol": protocol_snapshot,
        "stage": "3" if spatial_relation else "2",
        "arm": args.arm,
        "experiment": ("D (expression + spatial + prototype dynamic)" if spatial_relation
                       else "C (expression + prototype dynamic)"),
        "h5ad": str(args.h5ad),
        "h5ad_sha256": file_sha256(args.h5ad),
        "label_key": args.label_key,
        "n_spots": n_cells,
        "n_genes": int(adata.shape[1]),
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "hyperparams": {**HP, "epochs": args.epochs, "cell_size": args.cell_size},
        "graph_contract": graph_contract,
        "dynamic": {
            "warmup_epochs": args.warmup_epochs,
            "update_interval": args.update_interval,
            "delta": args.delta,
            "tau": args.tau,
            "lambda_proto": args.lambda_proto,
            "target_k": target_k,
            "objective": "L_KL + lambda_proto * L_proto (replaces label smoothing and -S_intra)",
            "cluster_init": model.cluster_init,
            "initial_partition": {
                "artifact": "warmup_init_labels.npy",
                "ari": ari_initial,
                "nmi": nmi_initial,
                "n_clusters": int(np.unique(initial_labels).size),
                **rare_initial,
            },
            "n_clusters_used": int(np.unique(pred_labels).size),
            "confidence_mean": float(confidence.mean()),
            # In quantile mode the threshold is re-derived, so this reports the share that
            # L_proto would actually use on the final embedding.  Reporting it against a
            # delta frozen at the last refresh gave a misleading 0.027 in the smoke run.
            "confident_fraction": float(
                (confidence > float(np.quantile(confidence, args.delta_quantile))).mean()
            ) if args.delta_mode == "quantile" else float((confidence > args.delta).mean()),
            "confidence_percentiles": {
                "p10": float(np.quantile(confidence, 0.10)),
                "p50": float(np.quantile(confidence, 0.50)),
                "p90": float(np.quantile(confidence, 0.90)),
            },
            "center_embedding": bool(not args.no_center_embedding),
            "delta_mode": args.delta_mode,
            "delta_quantile": args.delta_quantile,
            "init_source": args.init_source,
            "pseudo_resolution": (float(args.pseudo_resolution)
                                  if args.pseudo_resolution is not None
                                  else stage1_runner.上游默认resolution(n_cells)),
            "pseudo_resolution_source": ("cli --pseudo-resolution"
                                         if args.pseudo_resolution is not None
                                         else "upstream segment_function"),
            "delta_effective_final": float(model.delta_value),
            "preserve_k": bool(not args.no_preserve_k),
            "refresh_trace": getattr(model, "refresh_trace", []),
        },
        "batching": {
            "sampler": "spatial_median_bisection_with_one_hop_halo",
            "n_batches": len(batches),
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
                     "n_clusters": int(np.unique(pred_labels).size),
                     "ari_vs_pseudo": ari_vs_pseudo, **rare},
        "rare_per_type": per_type,
        "leiden_baseline": {"ari": ari_leiden,
                            "nmi": float(normalized_mutual_info_score(truth, leiden)),
                            "n_clusters": len(set(leiden)),
                            "resolution": (float(args.pseudo_resolution)
                                           if args.pseudo_resolution is not None
                                           else stage1_runner.上游默认resolution(n_cells)),
                            **rare_leiden},
        "truth_distribution": {str(k): int(v) for k, v in truth.value_counts().items()},
        "history_tail": history[-8:],
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
          f"{int(np.unique(pred_labels).size):4d} {rare['rare_f1']:9.4f} {rare['rare_recall']:8.4f}")
    print(f"{'Leiden floor':16s} {ari_leiden:8.4f} "
          f"{payload['leiden_baseline']['nmi']:8.4f} {len(set(leiden)):4d} "
          f"{rare_leiden['rare_f1']:9.4f} {rare_leiden['rare_recall']:8.4f}")
    print(f"ARI vs A0 pseudo = {ari_vs_pseudo:.4f} (Stage 1 was ~0.995)")
    print(f"confident spots (q>{float(model.delta_value):.4f}, mode={args.delta_mode}) = "
          f"{payload['dynamic']['confident_fraction']:.3f}")
    print(f"cluster init: {model.cluster_init.get('selection')} "
          f"res={model.cluster_init.get('resolution')} k={model.cluster_init.get('k')}")
    print(f"runtime {payload['runtime_sec']}s, peak GPU {payload['peak_gpu_mem_gb']} GB")
    print(f"artifacts -> {out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
