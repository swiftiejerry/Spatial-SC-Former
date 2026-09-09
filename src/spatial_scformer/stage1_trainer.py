"""Stage 1 trainer: upstream NDR_2 semantics plus an optional spot-spot relation.

Why this is a local fork instead of a patch to ``third_party/scFormer``
---------------------------------------------------------------------
The A0 baseline must stay reproducible from the untouched upstream code, so the
upstream package is never edited.  This module re-implements ``NDR_2.train_model``
and ``pred`` loop-for-loop with three additions the blueprint's modification #1
requires:

1. ``num_relations`` can be 3, adding relation type 2 = spot -> spot, so spatial
   adjacency enters HGT message passing rather than being a post-hoc overlay.
2. Batches may carry halo cells.  A 30-spot tile is mostly boundary, so 26-36% of
   its spatial edges would still be cut; the halo pulls in one-hop spatial
   neighbours so every edge is available.  Halo cells contribute to message
   passing and reconstruction only.
3. The clustering losses (label smoothing and the intra-cluster cosine term) are
   restricted to core cells, so adding a halo cannot change how many spots the
   clustering objective sees.

Everything else - the shared 256-d encoder, the 104-d embedding used directly as
logits without a classification head, ``loss = loss_cluster + loss_kl - lll``,
``argmax`` over the embedding as the predicted cluster - is byte-for-byte the
upstream formulation, because Stage 1 must isolate one factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.nn import functional as F

CELL_TYPE = 0
GENE_TYPE = 1
REL_GENE_TO_CELL = 0
REL_CELL_TO_GENE = 1
REL_SPOT_TO_SPOT = 2


@dataclass
class Stage1Config:
    n_hidden: int = 104
    n_heads: int = 8
    n_layers: int = 3
    dropout: float = 0.3
    labsm: float = 0.1
    lr: float = 5e-4
    weight_decay: float = 0.1
    epochs: int = 100
    spatial_relation: bool = False
    # 两个论文保真开关，默认都停在公开代码的行为上，所以现有 run 逐位不变。
    # kl_target="paper_x"：重构目标换成论文 Eq.1 的 X。实测现状（raw counts）在
    #   cell->gene 方向 softmax 之后是 one-hot（最大概率 1.000、有效支撑 1.00/513），
    #   而论文明确写了 target 是 X。非零模式相同，所以这一项不改图。
    # edge_source="per_spot_selection"：边只连该 spot 自己选中的基因，实测现状是
    #   batch gene 并集的全部非零项、每 spot 中位 156 条、87.2% 不是自选边。
    kl_target: str = "counts"
    edge_source: str = "batch_nonzero"
    # gene 节点输入。公开代码喂的是 raw counts 行（`model.py:708`），论文 2.1 要求的是
    # Z[:, j]，即该基因在全部 spot 上的标准化谱。实测两类节点的输入尺度差一个量级
    # （spot 侧 L2 中位 135.4、max/median 1.18；gene 侧 15.6 / 14.2），
    # 所以这一项单独留一个开关，方便只改它一项。
    gene_input: str = "counts"

    @property
    def num_relations(self) -> int:
        return 3 if self.spatial_relation else 2

    def validate(self) -> None:
        if self.kl_target not in ("counts", "paper_x"):
            raise ValueError(f"kl_target must be counts or paper_x, got {self.kl_target!r}")
        if self.gene_input not in ("counts", "paper_z"):
            raise ValueError(f"gene_input must be counts or paper_z, got {self.gene_input!r}")
        if self.edge_source not in ("batch_nonzero", "per_spot_selection"):
            raise ValueError(
                f"edge_source must be batch_nonzero or per_spot_selection, got {self.edge_source!r}"
            )


def scale_cell_features(rna_matrix: Any) -> csr_matrix:
    """normalize_total(1e4) + log1p + scale, exactly as upstream ``NDR_2.__init__``."""
    import anndata as ad
    import scanpy as sc

    adata = ad.AnnData(rna_matrix.T)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    return csr_matrix(adata.X.T)


def _spatial_pairs_by_cell(edge_index: object) -> dict[int, list[int]]:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must be (2, n_edges), got {edges.shape}")
    mapping: dict[int, list[int]] = {}
    for source, target in zip(edges[0].tolist(), edges[1].tolist()):
        mapping.setdefault(int(source), []).append(int(target))
    return mapping


class Stage1Model:
    """Upstream-equivalent HGT trainer with optional spatial relation and halo."""

    def __init__(
        self,
        rna_matrix: Any,
        batches: Sequence[dict[str, Any]],
        pseudo_labels: Sequence[int],
        config: Stage1Config,
        device: torch.device,
        spatial_edge_index: object | None = None,
        selection: dict[int, Sequence[int]] | None = None,
    ) -> None:
        from scformer.model import GNN_from_raw
        from scformer.utils import LabelSmoothing

        config.validate()
        if config.spatial_relation and spatial_edge_index is None:
            raise ValueError("spatial_edge_index is required when spatial_relation=True")
        if config.edge_source == "per_spot_selection" and not selection:
            raise ValueError("edge_source='per_spot_selection' needs the per-spot gene selection")

        self.rna_matrix = rna_matrix
        self.batches = list(batches)
        self.pseudo_labels = np.asarray(pseudo_labels, dtype=np.int64)
        self.config = config
        self.device = device
        self.cell_features_processed = scale_cell_features(rna_matrix)
        self.selection = ({int(cell): [int(g) for g in genes]
                           for cell, genes in selection.items()} if selection else {})
        if config.kl_target == "paper_x":
            from .graph.paper_topz import paper_x_matrix

            self.kl_matrix = paper_x_matrix(rna_matrix)
        else:
            self.kl_matrix = rna_matrix
        if config.gene_input == "paper_z":
            from .graph.paper_topz import gene_zscore

            # X 留稀疏，mean / std 留向量；Z 列在 batch 里现算，避免把 18085 x 3484
            # 的稠密 Z 全建出来（那是 252 MB，而每个 batch 只用到几百个基因）。
            self.gene_x, self.gene_mean, self.gene_std = gene_zscore(rna_matrix)
        else:
            self.gene_x = None
        self.spatial_neighbours = (
            _spatial_pairs_by_cell(spatial_edge_index) if config.spatial_relation else {}
        )

        self.label_smoothing = LabelSmoothing(config.labsm)
        self.gnn = GNN_from_raw(
            in_dim=[rna_matrix.shape[0], rna_matrix.shape[1]],
            n_hid=config.n_hidden,
            num_types=2,
            num_relations=config.num_relations,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
        ).to(device)
        self.optimizer = torch.optim.AdamW(
            self.gnn.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    def _build_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        cell_index = [int(c) for c in batch["cell_index"]]
        gene_index = [int(g) for g in batch["gene_index"]]
        n_cells = len(cell_index)
        local_of = {cell: position for position, cell in enumerate(cell_index)}

        if self.config.gene_input == "paper_z":
            rows = np.asarray(self.gene_x[gene_index, :].todense(), dtype=np.float32)
            mean = self.gene_mean[gene_index].astype(np.float32)[:, None]
            std = self.gene_std[gene_index].astype(np.float32)[:, None]
            gene_feature = (rows - mean) / std      # 论文 Z[:, j]，转置后每行是一个基因
        else:
            gene_feature = self.rna_matrix[gene_index, :]
        cell_feature = self.cell_features_processed[:, cell_index].T
        gene_cell_sub = self.rna_matrix[gene_index, :][:, cell_index]
        # 论文 2.5.1：softmax 归一化作用在 (B, ŴB) 这块子矩阵上，与边怎么选无关，
        # 所以重构目标始终是整块子矩阵，只有"用 C 还是用 X"这一项在变。
        kl_sub = (gene_cell_sub if self.config.kl_target == "counts"
                  else self.kl_matrix[gene_index, :][:, cell_index])

        if self.config.edge_source == "batch_nonzero":
            gene_rows, cell_cols = np.nonzero(gene_cell_sub)
        else:
            local_gene = {int(g): position for position, g in enumerate(gene_index)}
            rows: list[int] = []
            cols: list[int] = []
            for position, cell in enumerate(cell_index):
                for gene in self.selection.get(int(cell), ()):
                    local = local_gene.get(int(gene))
                    if local is None:
                        continue
                    rows.append(local)
                    cols.append(position)
            gene_rows = np.asarray(rows, dtype=np.int64)
            cell_cols = np.asarray(cols, dtype=np.int64)
        gene_to_cell_src = (gene_rows + n_cells).tolist()
        gene_to_cell_dst = cell_cols.tolist()
        cell_to_gene_src = cell_cols.tolist()
        cell_to_gene_dst = (gene_rows + n_cells).tolist()

        sources = gene_to_cell_src + cell_to_gene_src
        targets = gene_to_cell_dst + cell_to_gene_dst
        edge_types = [REL_GENE_TO_CELL] * len(gene_to_cell_src) + [
            REL_CELL_TO_GENE
        ] * len(cell_to_gene_src)

        n_spatial = 0
        if self.config.spatial_relation:
            core = batch.get("core_index", cell_index)
            spatial_src: list[int] = []
            spatial_dst: list[int] = []
            # Only edges incident to a core cell can carry gradient for this batch,
            # and every such edge is guaranteed present because the halo is one hop.
            for cell in core:
                target_local = local_of[int(cell)]
                for neighbour in self.spatial_neighbours.get(int(cell), ()):
                    source_local = local_of.get(int(neighbour))
                    if source_local is None:
                        continue
                    spatial_src.append(source_local)
                    spatial_dst.append(target_local)
            n_spatial = len(spatial_src)
            sources += spatial_src
            targets += spatial_dst
            edge_types += [REL_SPOT_TO_SPOT] * n_spatial

        node_feature = [
            torch.tensor(cell_feature.todense(), dtype=torch.float32).to(self.device),
            torch.tensor(
                gene_feature if isinstance(gene_feature, np.ndarray) else gene_feature.todense(),
                dtype=torch.float32,
            ).to(self.device),
        ]
        node_type = torch.LongTensor(
            np.concatenate(
                [np.full(n_cells, CELL_TYPE), np.full(len(gene_index), GENE_TYPE)]
            )
        ).to(self.device)
        edge_index = torch.LongTensor([sources, targets]).to(self.device)
        edge_type = torch.LongTensor(edge_types).to(self.device)

        core_positions = torch.LongTensor(
            [local_of[int(c)] for c in batch.get("core_index", cell_index)]
        ).to(self.device)
        return {
            "node_feature": node_feature,
            "node_type": node_type,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "gene_cell_sub": gene_cell_sub,
            "kl_target": kl_sub,
            "cell_index": cell_index,
            "core_positions": core_positions,
            "n_spatial_edges": n_spatial,
            "n_spot_gene_edges": int(len(gene_to_cell_src)),
        }

    def _reconstruction_loss(self, cell_emb, gene_emb, gene_cell_sub) -> torch.Tensor:
        target = torch.tensor(gene_cell_sub.todense(), dtype=torch.float32).to(self.device)
        logp_gene_to_cell = F.log_softmax(torch.mm(gene_emb, cell_emb.t()), dim=-1)
        logp_cell_to_gene = F.log_softmax(torch.mm(cell_emb, gene_emb.t()), dim=-1)
        loss_gene_to_cell = F.kl_div(
            logp_gene_to_cell, F.softmax(target, dim=-1), reduction="mean"
        )
        loss_cell_to_gene = F.kl_div(
            logp_cell_to_gene, F.softmax(target.t(), dim=-1), reduction="mean"
        )
        return loss_gene_to_cell + loss_cell_to_gene

    def _intra_cluster_similarity(self, core_emb, core_labels) -> torch.Tensor:
        # Upstream sums (not averages) the per-label mean cosine similarity and
        # subtracts it from the loss; reproduced verbatim for comparability.
        total = torch.zeros((), dtype=core_emb.dtype, device=core_emb.device)
        labels = core_labels.detach().cpu().numpy()
        for label in sorted(set(labels.tolist())):
            mask = torch.from_numpy(labels == label).to(core_emb.device)
            group = core_emb[mask]
            if group.size(0) > 1:
                similarity = F.cosine_similarity(
                    group.unsqueeze(1), group.unsqueeze(0), dim=-1
                )
                total = total + similarity.mean()
        return total

    def train(self, progress: bool = True) -> list[dict[str, float]]:
        from tqdm import tqdm

        prepared = [self._build_batch(batch) for batch in self.batches]
        history: list[dict[str, float]] = []
        epochs = range(self.config.epochs)
        for epoch in tqdm(epochs, desc="Epochs", disable=not progress):
            sums = {"loss": 0.0, "kl": 0.0, "cluster": 0.0, "sim": 0.0}
            for payload in prepared:
                node_rep = self.gnn.forward(
                    payload["node_feature"],
                    payload["node_type"],
                    payload["edge_index"],
                    payload["edge_type"],
                )
                cell_emb = node_rep[payload["node_type"] == CELL_TYPE]
                gene_emb = node_rep[payload["node_type"] == GENE_TYPE]

                loss_kl = self._reconstruction_loss(
                    cell_emb, gene_emb, payload["kl_target"]
                )
                core_positions = payload["core_positions"]
                core_emb = cell_emb[core_positions]
                core_labels = torch.LongTensor(
                    self.pseudo_labels[
                        [payload["cell_index"][int(p)] for p in core_positions.cpu()]
                    ]
                ).to(self.device)
                loss_cluster = self.label_smoothing(core_emb, core_labels)
                similarity = self._intra_cluster_similarity(core_emb, core_labels)
                loss = loss_cluster + loss_kl - similarity

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                sums["loss"] += float(loss.detach())
                sums["kl"] += float(loss_kl.detach())
                sums["cluster"] += float(loss_cluster.detach())
                sums["sim"] += float(similarity.detach())
            n = len(prepared)
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": sums["loss"] / n,
                    "reconstruction_loss": sums["kl"] / n,
                    "cluster_loss": sums["cluster"] / n,
                    "intra_similarity": sums["sim"] / n,
                }
            )
        return history

    def predict(self) -> dict[str, np.ndarray]:
        """Core-cell predictions in original ``adata`` row order.

        Upstream ``pred`` returns batch-order outputs for every cell in every
        batch.  With halo batches a cell appears in several batches, so only the
        batch where it is a core cell is authoritative; each cell is core exactly
        once, which keeps the output length equal to ``n_cells``.
        """
        prepared = [self._build_batch(batch) for batch in self.batches]
        n_cells = self.rna_matrix.shape[1]
        labels = np.full(n_cells, -1, dtype=np.int64)
        embeddings = np.zeros((n_cells, self.config.n_hidden), dtype=np.float32)
        with torch.no_grad():
            for payload in prepared:
                node_rep = self.gnn.forward(
                    payload["node_feature"],
                    payload["node_type"],
                    payload["edge_index"],
                    payload["edge_type"],
                )
                cell_emb = node_rep[payload["node_type"] == CELL_TYPE]
                for position in payload["core_positions"].cpu().tolist():
                    global_id = payload["cell_index"][int(position)]
                    vector = cell_emb[int(position)]
                    labels[global_id] = int(vector.argmax().item())
                    embeddings[global_id] = vector.detach().cpu().numpy()
        if int((labels < 0).sum()) != 0:
            raise RuntimeError("some cells were never a core cell in any batch")
        return {"pred_label": labels, "cell_embedding": embeddings}

    def spatial_edge_stats(self) -> dict[str, int]:
        totals = [self._build_batch(batch)["n_spatial_edges"] for batch in self.batches]
        return {
            "total_spatial_edges_in_batches": int(sum(totals)),
            "min_per_batch": int(min(totals)) if totals else 0,
            "max_per_batch": int(max(totals)) if totals else 0,
        }
