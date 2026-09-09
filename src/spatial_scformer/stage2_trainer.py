"""Stage 2 trainer: prototype-based dynamic clustering (blueprint modification #3).

What changes relative to Stage 1
--------------------------------
Stage 1 showed the fixed-pseudo-label objective dominates: ARI against the
pseudo labels stays above 0.99, so the model reproduces Leiden almost point for
point and no added relation can move rare-cluster detection.  This module
replaces that objective exactly as the blueprint prescribes:

  * epochs 1..warmup:      L = L_KL only (no clustering constraint at all)
  * at epoch == warmup:    Leiden on the learned spot embedding gives the first
                           pseudo labels and K0, then prototypes c_k = mean(h_i)
  * afterwards:            L = L_KL + lambda_proto * L_proto, and only spots with
                           confidence q_i > delta contribute
  * every update_interval: assignments and prototypes are recomputed from the
                           current embedding; K stays fixed in this first version

``L_proto`` replaces the label-smoothing classification loss and the
intra-cluster cosine term rather than being added on top, so the experiment still
isolates a single factor.  There is no ``Linear(d_h, K)`` head: cluster identity
comes from prototype cosine similarity, which is what makes a later variable-K
split/merge extension possible without resizing any layer.

Low-confidence spots are deliberately left unconstrained; that is the mechanism
intended to give rare and boundary spots room to be reassigned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .stage1_trainer import CELL_TYPE, GENE_TYPE, Stage1Config, Stage1Model


@dataclass
class Stage2Config(Stage1Config):
    """Stage 1 hyper-parameters plus the dynamic-clustering controls.

    Every value the blueprint calls tunable lives here so runs stay declarative;
    ``delta`` in particular is only a first test value and is swept explicitly.
    """

    warmup_epochs: int = 30
    update_interval: int = 10
    delta: float = 0.8
    tau: float = 0.2
    lambda_proto: float = 1.0
    leiden_neighbors: int = 10
    leiden_resolution: float | None = None
    target_k: int | None = None
    seed: int = 0
    preserve_k: bool = True
    center_embedding: bool = True
    delta_mode: str = "quantile"
    delta_quantile: float = 0.2
    init_source: str = "warmup_leiden"
    probe_every: int = 0
    probe_dir: str | None = None


def leiden_on_embedding(
    embedding: np.ndarray,
    *,
    n_neighbors: int,
    resolution: float | None,
    target_k: int | None,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster the learned embedding, optionally scanning for an exact K.

    The scan is label-free: it only inspects the number of Leiden communities, so
    no ground-truth information can leak into the pseudo labels.
    """
    import anndata as ad
    import scanpy as sc

    holder = ad.AnnData(X=np.ascontiguousarray(embedding, dtype=np.float32))
    sc.pp.neighbors(holder, n_neighbors=n_neighbors, use_rep="X", random_state=seed)

    def cluster_at(value: float) -> np.ndarray:
        sc.tl.leiden(holder, resolution=value, key_added="dyn", random_state=seed)
        return holder.obs["dyn"].astype(int).to_numpy()

    if target_k is None:
        chosen = 0.5 if resolution is None else float(resolution)
        labels = cluster_at(chosen)
        return labels, {"selection": "fixed_resolution", "resolution": chosen,
                        "k": int(np.unique(labels).size)}

    scan: list[dict[str, float | int]] = []
    best: tuple[np.ndarray, float, int] | None = None
    for candidate in np.round(np.arange(0.10, 3.001, 0.05), 4):
        labels = cluster_at(float(candidate))
        k = int(np.unique(labels).size)
        scan.append({"resolution": float(candidate), "k": k})
        if k == int(target_k):
            return labels, {"selection": "first_exact_k", "resolution": float(candidate),
                            "k": k, "scan": scan}
        if best is None or abs(k - int(target_k)) < abs(best[2] - int(target_k)):
            best = (labels, float(candidate), k)
    if best is None:
        raise RuntimeError("leiden scan produced no partition")
    return best[0], {"selection": "closest_k", "resolution": best[1], "k": best[2], "scan": scan}


def embedding_center(embedding: torch.Tensor, enabled: bool) -> torch.Tensor | None:
    """The origin that cosine similarity is measured from.

    Why this exists
    ---------------
    Three Stage 2 runs in a row silently switched ``L_proto`` off.  The cause is
    geometric, not a badly chosen hyper-parameter.  On the warm-up embedding every
    spot sits inside one narrow cone: top-1 cosine against the prototypes runs
    0.808/0.935/0.982 (min/median/max), the top1-top2 margin has median 0.016, and
    the five prototypes themselves have pairwise cosine 0.956-0.981.  A similarity
    that is almost constant carries almost no information, so
    ``q_i = max_k softmax(s_ik / tau)`` is squeezed into a band far narrower than
    [0, 1] and every fixed delta either admits nearly all spots or none of them.

    Measuring the same cosine after removing the component all spots share fixes
    the measurement instead of patching the threshold.  On the very same embedding
    the top-1 cosine spreads to 0.042/0.393/0.714, the top1-top2 margin widens to
    median 0.258, and the prototypes separate to -0.484/-0.244/0.215.  The
    blueprint formula is untouched -- it still reads cos(h_i, c_k) -- only the
    origin moves, and it moves for spots and prototypes alike, so the comparison
    stays fair.  Numbers above come from ``experiments/reports/原型几何诊断``.
    """
    if not enabled:
        return None
    return embedding.detach().mean(dim=0)


def resolve_delta(
    confidence: torch.Tensor, *, mode: str, absolute: float, quantile: float
) -> float:
    """Turn the blueprint's ``q_i > delta`` rule into a threshold that stays alive.

    ``absolute`` is the literal reading: delta is a constant.  ``quantile`` keeps
    the rule shape and re-derives delta from the current q distribution at every
    refresh, dropping the least confident ``quantile`` share of spots.  The
    blueprint says delta = 0.8 is only a first test value that must be written to
    config and swept, and the purpose of the rule is to keep low-confidence rare
    and boundary spots out of the prototype pull.  A rank-based cut expresses that
    purpose and, unlike a constant, cannot silently empty the confident set when
    the embedding geometry moves underneath it.
    """
    if mode == "quantile":
        share = min(max(float(quantile), 0.0), 0.95)
        return float(torch.quantile(confidence.detach().float(), share))
    return float(absolute)


def frame_geometry(
    embedding: torch.Tensor, centers: torch.Tensor, center: torch.Tensor | None
) -> dict[str, float]:
    """Numbers that make prototype collapse visible while a run is still going.

    ``prototype_cosine_max`` near 1 means the prototypes have merged and the
    partition is about to degenerate, which is the failure that produced ARI
    0.0029 before anyone could see it in the loss curve.
    """
    with torch.no_grad():
        points = embedding if center is None else embedding - center
        protos = centers if center is None else centers - center
        similarity = F.cosine_similarity(points.unsqueeze(1), protos.unsqueeze(0), dim=2)
        ordered = similarity.sort(dim=1, descending=True).values
        if ordered.size(1) > 1:
            gap = ordered[:, 0] - ordered[:, 1]
        else:
            gap = torch.zeros_like(ordered[:, 0])
        normalized = F.normalize(protos, dim=1)
        pairwise = normalized @ normalized.t()
        mask = ~torch.eye(protos.size(0), dtype=torch.bool, device=protos.device)
        off_diagonal = pairwise[mask]
        return {
            "top1_similarity_median": float(ordered[:, 0].median()),
            "top1_top2_gap_median": float(gap.median()),
            "prototype_cosine_median": float(off_diagonal.median()) if off_diagonal.numel() else 0.0,
            "prototype_cosine_max": float(off_diagonal.max()) if off_diagonal.numel() else 0.0,
        }


def prototypes_from_assignment(
    embedding: torch.Tensor,
    assignment: torch.Tensor,
    n_clusters: int,
    *,
    previous: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    preserve_k: bool = True,
    center: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[int, list[int]]]:
    """c_k = mean of the embeddings currently assigned to cluster k.

    Why the empty-cluster branch exists
    -----------------------------------
    The first version zeroed the prototype of any cluster that lost all of its
    spots.  Cosine similarity against a zero vector is 0, so that slot could never
    win an argmax again: cluster death was irreversible.  ``stage2_C_t0.005_d0.8_seed0``
    ended with 2 non-empty clusters out of K0=5 and ARI 0.0029 for exactly this
    reason.  The blueprint fixes K in the first version of modification #3, so a
    dying slot is a bug, not a result.

    With ``preserve_k`` the empty slot is re-seeded from the least confident spots,
    picked greedily to be as unlike the surviving prototypes as possible.  Those
    spots are the boundary and rare candidates the blueprint wants to leave room
    for, so the guard and the scientific intent point the same way.  Passing
    ``preserve_k=False`` reproduces the old behaviour for comparison.
    """
    centers = torch.zeros(n_clusters, embedding.size(1), device=embedding.device,
                          dtype=embedding.dtype)
    empty: list[int] = []
    for cluster in range(n_clusters):
        mask = assignment == cluster
        if bool(mask.any()):
            centers[cluster] = embedding[mask].mean(dim=0)
        else:
            empty.append(cluster)

    reseeded: dict[int, list[int]] = {}
    if not empty or not preserve_k:
        return centers, reseeded

    with torch.no_grad():
        # The "most unlike the survivors" search has to run in the same frame the
        # loss uses; in the raw cone every candidate looks alike (pairwise cosine
        # 0.96+) and the pick would be noise.
        framed = embedding if center is None else embedding - center
        normalized = F.normalize(framed, dim=1)
        if confidence is not None and confidence.numel() == embedding.size(0):
            per_slot = max(10, embedding.size(0) // (4 * max(n_clusters, 1)))
            take = min(embedding.size(0), per_slot * len(empty))
            pool = torch.argsort(confidence)[:take]
        else:
            pool = torch.arange(embedding.size(0), device=embedding.device)
        for cluster in empty:
            if pool.numel() == 0:
                if previous is not None:
                    centers[cluster] = previous[cluster]
                continue
            framed_centers = centers if center is None else centers - center
            similarity = normalized[pool] @ F.normalize(framed_centers, dim=1).t()
            seed = int(pool[int(torch.argmin(similarity.max(dim=1).values))])
            neighbours = (normalized @ normalized[seed]).topk(
                min(10, embedding.size(0))).indices
            centers[cluster] = embedding[neighbours].mean(dim=0)
            # The caller needs the members, not just the slot id: a re-seeded prototype
            # with nobody assigned to it never appears as a cross-entropy target, so
            # L_proto can only push spots away from it and the slot dies again.
            reseeded[int(cluster)] = [int(v) for v in neighbours.tolist()]
    return centers, reseeded


def confidence_and_assignment(
    embedding: torch.Tensor,
    centers: torch.Tensor,
    tau: float,
    center: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """s_ik = cos(h_i, c_k); p_ik = softmax(s_ik / tau); q_i = max_k p_ik.

    ``center`` shifts the origin that both arguments of the cosine are measured
    from; ``embedding_center`` explains why the raw origin leaves the cosine
    almost constant and therefore useless as a confidence signal.
    """
    if center is not None:
        embedding = embedding - center
        centers = centers - center
    similarity = F.cosine_similarity(embedding.unsqueeze(1), centers.unsqueeze(0), dim=2)
    probabilities = F.softmax(similarity / tau, dim=1)
    confidence, assignment = probabilities.max(dim=1)
    return similarity, confidence, assignment


CONFIDENCE_TAU_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
CONFIDENCE_DELTA_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)


def confidence_grid(
    embedding: torch.Tensor,
    centers: torch.Tensor,
    tau_grid: Sequence[float] = CONFIDENCE_TAU_GRID,
    delta_grid: Sequence[float] = CONFIDENCE_DELTA_GRID,
    *,
    center: torch.Tensor | None = None,
) -> dict[str, dict[str, float]]:
    """Confident-spot fraction for each (tau, delta) on the embedding at hand.

    tau and delta interact through a hard ceiling: cosine similarity lives in
    [-1, 1], so the softmax over K prototypes cannot reach a high maximum
    probability unless tau is small.  A smoke run at tau=0.2 left zero confident
    spots, which silently zeroes ``L_proto`` and turns Stage 2 back into plain
    reconstruction.  Recording the grid at cluster-init time makes that failure
    mode visible in the log within the first minutes of a run instead of after it.
    """
    with torch.no_grad():
        table: dict[str, dict[str, float]] = {}
        for tau in tau_grid:
            _, confidence, _ = confidence_and_assignment(
                embedding, centers, float(tau), center=center
            )
            table[f"{tau:g}"] = {
                f"{delta:g}": float((confidence > delta).float().mean())
                for delta in delta_grid
            }
    return table


class Stage2Model(Stage1Model):
    """Stage 1 graph machinery with the dynamic clustering objective."""

    def __init__(
        self,
        rna_matrix: Any,
        batches: Sequence[dict[str, Any]],
        pseudo_labels: Sequence[int],
        config: Stage2Config,
        device: torch.device,
        spatial_edge_index: object | None = None,
        selection: dict[int, Sequence[int]] | None = None,
    ) -> None:
        super().__init__(rna_matrix, batches, pseudo_labels, config, device,
                         spatial_edge_index=spatial_edge_index, selection=selection)
        self.config: Stage2Config = config
        self.n_clusters = int(config.target_k or np.unique(self.pseudo_labels).size)
        self.centers: torch.Tensor | None = None
        self.assignment = np.full(rna_matrix.shape[1], -1, dtype=np.int64)
        self.initial_assignment: np.ndarray | None = None
        self.cluster_init: dict[str, Any] = {}
        # Frame and threshold in force between refreshes.  Both are derived from the
        # full embedding and then held fixed, so the per-batch L_proto uses exactly
        # the same rule as the full-embedding bookkeeping.
        self.center_vector: torch.Tensor | None = None
        self.delta_value = float(config.delta)
        self.refresh_trace: list[dict[str, Any]] = []

    def _core_embedding(self, prepared: list[dict[str, Any]]) -> np.ndarray:
        """Embedding for every spot, taken from the batch where it is a core cell."""
        n_cells = self.rna_matrix.shape[1]
        embedding = np.zeros((n_cells, self.config.n_hidden), dtype=np.float32)
        filled = np.zeros(n_cells, dtype=bool)
        was_training = self.gnn.training
        self.gnn.eval()  # 关闭 dropout，确保推断的确定性
        try:
            with torch.no_grad():
                for payload in prepared:
                    node_rep = self.gnn.forward(
                        payload["node_feature"], payload["node_type"],
                        payload["edge_index"], payload["edge_type"],
                    )
                    cell_emb = node_rep[payload["node_type"] == CELL_TYPE]
                    for position in payload["core_positions"].cpu().tolist():
                        global_id = payload["cell_index"][int(position)]
                        embedding[global_id] = cell_emb[int(position)].detach().cpu().numpy()
                        filled[global_id] = True
        finally:
            self.gnn.train(was_training)
        if not bool(filled.all()):
            raise RuntimeError("some spots were never a core cell in any batch")
        return embedding

    def _synchronise_partition(
        self, embedding: torch.Tensor
    ) -> tuple[np.ndarray, torch.Tensor, int]:
        """Make refresh labels, prototypes, and final predictions one partition.

        Empty-slot reseeding changes the assignment after the first argmax.  A
        second argmax against stale prototypes could therefore produce labels
        different from the labels used by the next loss.  Recompute means and
        assignments together until the fixed-K partition is stable (or a small
        bounded number of passes is reached).
        """
        assert self.centers is not None
        labels = self.assignment.copy()
        reseeded_total = 0
        confidence = torch.zeros(embedding.size(0), device=embedding.device)
        for _ in range(4):
            _, confidence, predicted = confidence_and_assignment(
                embedding, self.centers, self.config.tau, center=self.center_vector
            )
            candidate = predicted.cpu().numpy().astype(np.int64)
            if np.unique(candidate).size < self.n_clusters:
                self.centers, reseeded = prototypes_from_assignment(
                    embedding, predicted, self.n_clusters,
                    previous=self.centers, confidence=confidence,
                    preserve_k=self.config.preserve_k, center=self.center_vector,
                )
                for cluster, members in reseeded.items():
                    if members:
                        candidate[np.asarray(members, dtype=np.int64)] = int(cluster)
                reseeded_total += len(reseeded)
                labels = candidate
                label_tensor = torch.from_numpy(labels).to(self.device)
                self.centers, _ = prototypes_from_assignment(
                    embedding, label_tensor, self.n_clusters,
                    preserve_k=False, center=self.center_vector,
                )
                self.assignment = labels
                continue

            label_tensor = torch.from_numpy(candidate).to(self.device)
            self.centers, _ = prototypes_from_assignment(
                embedding, label_tensor, self.n_clusters,
                preserve_k=False, center=self.center_vector,
            )
            _, confidence, final_assignment = confidence_and_assignment(
                embedding, self.centers, self.config.tau, center=self.center_vector
            )
            final_labels = final_assignment.cpu().numpy().astype(np.int64)
            labels = final_labels
            self.assignment = labels
            if np.unique(labels).size == self.n_clusters:
                return labels, confidence, reseeded_total

        self.assignment = labels
        return labels, confidence, reseeded_total

    def _initialise_clusters(self, prepared: list[dict[str, Any]]) -> dict[str, Any]:
        embedding = self._core_embedding(prepared)
        if self.config.init_source == "a0_pseudo":
            # 为什么留这条路：warm-up 表示里没有可聚类的组织域信号。实测同一份 A1，
            # 30 epoch 纯 L_KL 之后 Leiden 的 ARI 是 0.0006、kNN 标签纯度只比随机高
            # 0.0126、同类减异类余弦 0.0004；而它的输入侧表达空间 PCA50 是 ARI 0.2572、
            # kNN 高出随机 0.3036（见 experiments/reports/warmup表示质量）。所以蓝图
            # "Epoch 30 在当前 H_spot 上跑 Leiden 拿初始伪标签" 这一步在这份数据上拿到
            # 的是噪声，后面的 prototype 迭代只能把噪声自洽化。
            #
            # 这条路只改"初始 assignment 从哪来"：用 A0 那份表达空间 Leiden 伪标签当
            # 第 0 次 assignment，之后每 T 个 epoch 照蓝图重算 assignment 与 prototype。
            # 伪标签不再进入任何 loss，也不再被固定 —— 蓝图修改点三要解决的是"固定"，
            # 不是"存在"，这条路保留了动态纠错，只是不再假设 warm-up 表示可聚类。
            labels = np.asarray(self.pseudo_labels, dtype=np.int64)
            info = {"selection": "a0_pseudo", "resolution": None,
                    "k": int(np.unique(labels).size)}
        else:
            labels, info = leiden_on_embedding(
                embedding,
                n_neighbors=self.config.leiden_neighbors,
                resolution=self.config.leiden_resolution,
                target_k=self.config.target_k,
                seed=self.config.seed,
            )
        self.n_clusters = int(np.unique(labels).size)
        self.assignment = labels.astype(np.int64)
        self.initial_assignment = self.assignment.copy()
        tensor = torch.from_numpy(embedding).to(self.device)
        self.center_vector = embedding_center(tensor, self.config.center_embedding)
        self.centers, _ = prototypes_from_assignment(
            tensor, torch.from_numpy(self.assignment).to(self.device), self.n_clusters,
            preserve_k=self.config.preserve_k, center=self.center_vector,
        )
        self.refresh_trace = []
        info["init_source"] = str(self.config.init_source)
        info["n_clusters"] = self.n_clusters
        _, confidence, _ = confidence_and_assignment(
            tensor, self.centers, self.config.tau, center=self.center_vector
        )
        self.delta_value = resolve_delta(
            confidence, mode=self.config.delta_mode,
            absolute=self.config.delta, quantile=self.config.delta_quantile,
        )
        info["center_embedding"] = bool(self.config.center_embedding)
        info["delta_mode"] = str(self.config.delta_mode)
        info["delta_quantile"] = float(self.config.delta_quantile)
        info["delta_effective_at_init"] = self.delta_value
        info["confident_fraction_at_init"] = float((confidence > self.delta_value).float().mean())
        # Both frames are recorded so the report can show the fix worked on the
        # geometry rather than asserting it.
        info["geometry_raw"] = frame_geometry(tensor, self.centers, None)
        info["geometry_active"] = frame_geometry(tensor, self.centers, self.center_vector)
        info["confidence_grid"] = confidence_grid(
            tensor, self.centers, center=self.center_vector
        )
        self.cluster_init = info
        print(f"[stage2] cluster init: selection={info.get('selection')} "
              f"resolution={info.get('resolution')} K0={self.n_clusters} "
              f"centered={info['center_embedding']} delta_mode={info['delta_mode']} "
              f"delta={self.delta_value:.4f} confident at tau={self.config.tau:g} = "
              f"{info['confident_fraction_at_init']:.3f}", flush=True)
        for label in ("raw", "active"):
            block = info[f"geometry_{label}"]
            print(f"[stage2]   geometry[{label:>6s}] top1={block['top1_similarity_median']:.4f} "
                  f"gap={block['top1_top2_gap_median']:.4f} "
                  f"proto_cos med/max={block['prototype_cosine_median']:.4f}"
                  f"/{block['prototype_cosine_max']:.4f}", flush=True)
        for tau_key, row in info["confidence_grid"].items():
            cells = "  ".join(f"q>{delta}:{value:.3f}" for delta, value in row.items())
            print(f"[stage2]   tau={tau_key:>5s}  {cells}", flush=True)
        return info

    def _refresh_clusters(self, prepared: list[dict[str, Any]]) -> dict[str, float]:
        """Recompute assignment and prototypes from the current embedding."""
        embedding = torch.from_numpy(self._core_embedding(prepared)).to(self.device)
        assert self.centers is not None
        # The component every spot shares keeps drifting while the encoder trains, so
        # the frame is re-derived here and then frozen until the next refresh.
        self.center_vector = embedding_center(embedding, self.config.center_embedding)
        _, confidence, assignment = confidence_and_assignment(
            embedding, self.centers, self.config.tau, center=self.center_vector
        )
        previous = self.assignment.copy()
        self.assignment = assignment.cpu().numpy().astype(np.int64)
        nonempty_before = int(np.unique(self.assignment).size)
        previous_centers = self.centers
        self.centers, reseeded = prototypes_from_assignment(
            embedding, assignment, self.n_clusters,
            previous=previous_centers, confidence=confidence,
            preserve_k=self.config.preserve_k, center=self.center_vector,
        )
        # Hand the seed neighbourhood to the revived slot.  Without this the guard is a
        # ratchet: the v2a run re-seeded slots 1, 2 and 4 at every single refresh and
        # found them empty again at the next one, ending at 2 non-empty clusters and
        # ARI 0.0024.  A slot with members is a target the loss can pull toward.
        for cluster, members in reseeded.items():
            if members:
                self.assignment[np.asarray(members, dtype=np.int64)] = int(cluster)
        self.assignment, confidence_after, extra_reseeded = self._synchronise_partition(embedding)
        reseeded_count = len(reseeded) + extra_reseeded
        changed = float(np.mean(previous != self.assignment)) if previous.size else 0.0
        sizes = np.bincount(self.assignment, minlength=self.n_clusters).tolist()
        # delta and the reported confident share are measured against the prototypes
        # the next epochs will actually score against, i.e. after any reseeding.
        self.delta_value = resolve_delta(
            confidence_after, mode=self.config.delta_mode,
            absolute=self.config.delta, quantile=self.config.delta_quantile,
        )
        probes = torch.tensor([0.1, 0.5, 0.9], device=confidence_after.device)
        spread = torch.quantile(confidence_after.detach().float(), probes).tolist()
        stats = {
            "changed_fraction": changed,
            "delta_effective": self.delta_value,
            "confident_fraction": float((confidence_after > self.delta_value).float().mean()),
            "confidence_mean": float(confidence_after.mean()),
            "confidence_p10": float(spread[0]),
            "confidence_p50": float(spread[1]),
            "confidence_p90": float(spread[2]),
            "n_nonempty_before_reseed": nonempty_before,
            "n_nonempty": int(np.unique(self.assignment).size),
            "n_reseeded": reseeded_count,
            **frame_geometry(embedding, self.centers, self.center_vector),
        }
        # 塌缩要在训练中途就能看见，而不是等 100 epoch 跑完看最终 ARI。
        trace = getattr(self, "refresh_trace", None)
        if trace is None:
            trace = self.refresh_trace = []
        trace.append({**stats, "cluster_sizes": sizes, "reseeded": sorted(reseeded)})
        flags = ""
        if stats["n_nonempty"] != self.n_clusters:
            flags += "  [warn] 簇位塌缩"
        if stats["confident_fraction"] <= 0.0:
            flags += "  [warn] L_proto 被静默关闭"
        if stats["prototype_cosine_max"] > 0.99:
            flags += "  [warn] 原型合并"
        print(f"[stage2] refresh: changed={changed:.3f} "
              f"confident={stats['confident_fraction']:.3f} "
              f"delta={stats['delta_effective']:.4f} "
              f"q p10/50/90={stats['confidence_p10']:.3f}/{stats['confidence_p50']:.3f}/"
              f"{stats['confidence_p90']:.3f} "
              f"gap={stats['top1_top2_gap_median']:.4f} "
              f"proto_cos={stats['prototype_cosine_max']:.4f} "
              f"nonempty={stats['n_nonempty']}/{self.n_clusters} "
              f"reseeded={sorted(reseeded)} sizes={sizes}{flags}", flush=True)
        return stats

    def _proto_loss(
        self, core_emb: torch.Tensor, core_labels: torch.Tensor
    ) -> tuple[torch.Tensor, int, float]:
        assert self.centers is not None
        similarity, confidence, _ = confidence_and_assignment(
            core_emb, self.centers, self.config.tau, center=self.center_vector
        )
        if self.config.delta_mode == "quantile":
            # The confidence scale drifts inside an interval, not only across refreshes.
            # With delta frozen at the last refresh the smoke run kept 2% of spots in the
            # epoch right after init and 29% in the next one, so L_proto was close to off
            # again for whole epochs.  Taking the cut from the batch's own q distribution
            # makes the rule scale-free: the share of spots left out stays where it was
            # configured no matter how the geometry moves.  Batches here are contiguous
            # spatial tiles, so this reads as "drop the least trustworthy fifth of each
            # tile", which is the blueprint's intent stated locally.
            threshold = resolve_delta(
                confidence, mode="quantile", absolute=self.config.delta,
                quantile=self.config.delta_quantile,
            )
        else:
            threshold = self.delta_value
        keep = confidence > threshold
        n_kept = int(keep.sum())
        if n_kept == 0:
            return torch.zeros((), device=core_emb.device, dtype=core_emb.dtype), 0, threshold
        logits = similarity[keep] / self.config.tau
        return F.cross_entropy(logits, core_labels[keep]), n_kept, threshold

    def train(self, progress: bool = True) -> list[dict[str, float]]:
        from tqdm import tqdm

        prepared = [self._build_batch(batch) for batch in self.batches]
        history: list[dict[str, float]] = []
        for epoch in tqdm(range(self.config.epochs), desc="Epochs", disable=not progress):
            epoch_number = epoch + 1
            self.gnn.train()  # 确保训练模式（dropout 开启）
            if epoch_number == self.config.warmup_epochs + 1 and self.centers is None:
                info = self._initialise_clusters(prepared)
                history.append({"epoch": epoch_number - 0.5, "event": 1.0,
                                "init_k": float(info["n_clusters"]),
                                # init_source=a0_pseudo 时没有 resolution 这个概念，
                                # 直接 float(None) 会在第 31 个 epoch 把整个 run 炸掉。
                                # 用 -1 表示"不适用"，比塞 NaN 好：NaN 会让 metrics.json
                                # 变成不合法的 JSON。
                                "init_resolution": (float(info["resolution"])
                                                    if info.get("resolution") is not None
                                                    else -1.0)})
            elif (
                self.centers is not None
                and epoch_number > self.config.warmup_epochs + 1
                and (epoch_number - self.config.warmup_epochs - 1) % self.config.update_interval == 0
            ):
                refresh = self._refresh_clusters(prepared)
                history.append({"epoch": epoch_number - 0.5, "event": 2.0, **refresh})

            dynamic = self.centers is not None
            sums = {"loss": 0.0, "kl": 0.0, "proto": 0.0, "kept": 0.0, "delta": 0.0}
            for payload in prepared:
                node_rep = self.gnn.forward(
                    payload["node_feature"], payload["node_type"],
                    payload["edge_index"], payload["edge_type"],
                )
                cell_emb = node_rep[payload["node_type"] == CELL_TYPE]
                gene_emb = node_rep[payload["node_type"] == GENE_TYPE]
                loss_kl = self._reconstruction_loss(cell_emb, gene_emb, payload["kl_target"])

                if dynamic:
                    core_positions = payload["core_positions"]
                    core_emb = cell_emb[core_positions]
                    global_ids = [payload["cell_index"][int(p)] for p in core_positions.cpu()]
                    core_labels = torch.from_numpy(self.assignment[global_ids]).long().to(self.device)
                    loss_proto, n_kept, threshold = self._proto_loss(core_emb, core_labels)
                    loss = loss_kl + self.config.lambda_proto * loss_proto
                    sums["proto"] += float(loss_proto.detach())
                    sums["kept"] += n_kept
                    sums["delta"] += threshold
                else:
                    loss = loss_kl

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                sums["loss"] += float(loss.detach())
                sums["kl"] += float(loss_kl.detach())

            n = len(prepared)
            history.append({
                "epoch": epoch_number,
                "stage": 2.0 if dynamic else 1.0,
                "loss": sums["loss"] / n,
                "reconstruction_loss": sums["kl"] / n,
                "proto_loss": sums["proto"] / n,
                "confident_spots": sums["kept"] / n,
                "delta_batch_mean": sums["delta"] / n,
            })

            # warm-up 途中定期存一份 embedding：一次 run 就能画出"warm-up 越久，表示里的
            # 组织域信号是涨还是跌"的曲线，不用为每个 warm-up 长度单独起一次 45 分钟的 run。
            # 只在 warm-up 阶段存（centers 还没建），落盘的是全片 spot 的表示。
            if (self.config.probe_every > 0 and self.config.probe_dir
                    and not dynamic and epoch_number % self.config.probe_every == 0):
                from pathlib import Path as _Path

                folder = _Path(self.config.probe_dir)
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / f"warmup_embedding_ep{epoch_number}.npy"
                np.save(target, self._core_embedding(prepared))
                print(f"[stage2] warm-up 探针 -> {target.name}", flush=True)

        if self.centers is None:
            # warmup_epochs >= epochs: still produce clusters so the run is usable.
            self._initialise_clusters(prepared)
        else:
            # The reported partition has to come from prototypes that match the final
            # weights.  Without this the labels are up to ``update_interval`` epochs
            # stale: the smoke run refreshed to 5 balanced clusters and then reported
            # K=3, purely because one more epoch of drift happened after the last
            # refresh.  The blueprint already updates assignment every T epochs; this
            # is that update, done once more at the end.
            final = self._refresh_clusters(prepared)
            history.append({"epoch": float(self.config.epochs) + 0.5, "event": 3.0, **final})
        return history

    def predict(self) -> dict[str, np.ndarray]:
        """Cluster identity from prototype cosine similarity, not embedding argmax."""
        if self.centers is None:
            raise RuntimeError("prototypes were never initialised")
        prepared = [self._build_batch(batch) for batch in self.batches]
        embedding = self._core_embedding(prepared)
        tensor = torch.from_numpy(embedding).to(self.device)
        assignment, confidence, _ = self._synchronise_partition(tensor)
        return {
            "pred_label": assignment,
            "cell_embedding": embedding,
            "confidence": confidence.cpu().numpy().astype(np.float32),
        }



