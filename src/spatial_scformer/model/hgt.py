"""Lightweight heterogeneous message-passing backbone for Spatial-scFormer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

EdgeType = tuple[str, str, str]
CELL_TO_GENE: EdgeType = ("cell", "top_z", "gene")
GENE_TO_CELL: EdgeType = ("gene", "rev_top_z", "cell")
CELL_TO_CELL: EdgeType = ("cell", "spatial", "cell")


@dataclass
class HeteroScFormerOutput:
    cell_embedding: Tensor
    gene_embedding: Tensor
    cluster_logits: Tensor

    @property
    def cluster_probabilities(self) -> Tensor:
        return self.cluster_logits.softmax(dim=-1)


def make_bidirectional_relations(cell_to_gene_edge_index: Tensor,
                                 spatial_edge_index: Tensor | None = None) -> dict[EdgeType, Tensor]:
    relations = {CELL_TO_GENE: cell_to_gene_edge_index,
                 GENE_TO_CELL: cell_to_gene_edge_index[[1, 0]]}
    if spatial_edge_index is not None:
        relations[CELL_TO_CELL] = spatial_edge_index
    return relations


def _validate_node_features(name: str, features: Tensor, input_dim: int) -> None:
    if not isinstance(features, Tensor) or features.ndim != 2 or features.shape[1] != input_dim:
        raise ValueError(f"{name} features must have shape (N, {input_dim})")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError(f"{name} features must be finite floating-point values")


def _validate_edge_index(relation: EdgeType, edge_index: Tensor, *, num_source: int,
                        num_target: int, device: torch.device) -> None:
    if not isinstance(edge_index, Tensor) or edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge index for {relation} must have shape (2, E) and dtype torch.long")
    if edge_index.device != device:
        raise ValueError(f"edge index for {relation} is on the wrong device")
    if edge_index.numel() and (int(edge_index[0].min()) < 0 or int(edge_index[0].max()) >= num_source or int(edge_index[1].min()) < 0 or int(edge_index[1].max()) >= num_target):
        raise IndexError(f"edge index out of range for relation {relation}")


def _validate_edge_weight(relation: EdgeType, edge_weight: Tensor | None, edge_count: int, device: torch.device) -> None:
    if edge_weight is None:
        return
    if not isinstance(edge_weight, Tensor) or edge_weight.ndim != 1 or edge_weight.shape[0] != edge_count or edge_weight.device != device:
        raise ValueError(f"edge weight for {relation} has the wrong shape or device")
    if not edge_weight.is_floating_point() or not torch.isfinite(edge_weight).all() or torch.any(edge_weight < 0):
        raise ValueError(f"edge weight for {relation} must be finite and non-negative")


class _RelationMean(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.source_transform = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, source_features: Tensor, target_count: int, edge_index: Tensor,
                edge_weight: Tensor | None) -> Tensor:
        hidden_dim = source_features.shape[1]
        if edge_index.shape[1] == 0:
            return source_features.new_zeros((target_count, hidden_dim))
        source, target = edge_index
        messages = self.source_transform(source_features).index_select(0, source)
        weights = source_features.new_ones(edge_index.shape[1]) if edge_weight is None else edge_weight.to(dtype=source_features.dtype)
        messages = messages * weights.unsqueeze(1)
        aggregated = source_features.new_zeros((target_count, hidden_dim)).index_add(0, target, messages)
        degree = source_features.new_zeros(target_count).index_add(0, target, weights)
        return aggregated / degree.clamp_min(torch.finfo(source_features.dtype).eps).unsqueeze(1)


class _HeteroMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.cell_self = nn.Linear(hidden_dim, hidden_dim)
        self.gene_self = nn.Linear(hidden_dim, hidden_dim)
        self.cell_to_gene = _RelationMean(hidden_dim)
        self.gene_to_cell = _RelationMean(hidden_dim)
        self.cell_to_cell = _RelationMean(hidden_dim)
        self.cell_norm = nn.LayerNorm(hidden_dim)
        self.gene_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cell_features: Tensor, gene_features: Tensor, relations: Mapping[EdgeType, Tensor], edge_weights: Mapping[EdgeType, Tensor] | None) -> tuple[Tensor, Tensor]:
        def weight_for(relation: EdgeType) -> Tensor | None:
            return None if edge_weights is None else edge_weights.get(relation)
        gene_message = self.cell_to_gene(cell_features, gene_features.shape[0], relations[CELL_TO_GENE], weight_for(CELL_TO_GENE))
        cell_message = self.gene_to_cell(gene_features, cell_features.shape[0], relations[GENE_TO_CELL], weight_for(GENE_TO_CELL))
        if CELL_TO_CELL in relations:
            cell_message = cell_message + self.cell_to_cell(cell_features, cell_features.shape[0], relations[CELL_TO_CELL], weight_for(CELL_TO_CELL))
        gene_update = F.gelu(self.gene_self(gene_features) + gene_message)
        cell_update = F.gelu(self.cell_self(cell_features) + cell_message)
        return (self.cell_norm(cell_features + self.dropout(cell_update)),
                self.gene_norm(gene_features + self.dropout(gene_update)))


class LightweightHeteroScFormer(nn.Module):
    """Whole-graph encoder with an explicit K-way cluster head."""

    def __init__(self, cell_input_dim: int, gene_input_dim: int, hidden_dim: int, num_clusters: int, *, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        for value in (cell_input_dim, gene_input_dim, hidden_dim, num_clusters, num_layers):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("model dimensions must be positive integers")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.cell_input_dim, self.gene_input_dim = cell_input_dim, gene_input_dim
        self.hidden_dim, self.num_clusters = hidden_dim, num_clusters
        self.cell_projection = nn.Linear(cell_input_dim, hidden_dim)
        self.gene_projection = nn.Linear(gene_input_dim, hidden_dim)
        self.layers = nn.ModuleList(_HeteroMessageLayer(hidden_dim, dropout) for _ in range(num_layers))
        self.cluster_head = nn.Linear(hidden_dim, num_clusters)

    def _canonical_relations(self, relations: Mapping[EdgeType, Tensor], edge_weights: Mapping[EdgeType, Tensor] | None):
        if CELL_TO_GENE not in relations:
            raise KeyError(f"missing required relation {CELL_TO_GENE}")
        canonical = dict(relations)
        weights = None if edge_weights is None else dict(edge_weights)
        canonical.setdefault(GENE_TO_CELL, canonical[CELL_TO_GENE][[1, 0]])
        if weights is not None and GENE_TO_CELL not in weights and CELL_TO_GENE in weights:
            weights[GENE_TO_CELL] = weights[CELL_TO_GENE]
        unknown = set(canonical).difference({CELL_TO_GENE, GENE_TO_CELL, CELL_TO_CELL})
        if unknown:
            raise KeyError(f"unsupported relations: {sorted(unknown)}")
        return canonical, weights

    def forward(self, cell_features: Tensor, gene_features: Tensor, relations: Mapping[EdgeType, Tensor], edge_weights: Mapping[EdgeType, Tensor] | None = None) -> HeteroScFormerOutput:
        _validate_node_features("cell", cell_features, self.cell_input_dim)
        _validate_node_features("gene", gene_features, self.gene_input_dim)
        if gene_features.device != cell_features.device:
            raise ValueError("cell and gene features must be on the same device")
        canonical, weights = self._canonical_relations(relations, edge_weights)
        shapes = {CELL_TO_GENE: (cell_features.shape[0], gene_features.shape[0]), GENE_TO_CELL: (gene_features.shape[0], cell_features.shape[0]), CELL_TO_CELL: (cell_features.shape[0], cell_features.shape[0])}
        for relation, edge_index in canonical.items():
            _validate_edge_index(relation, edge_index, num_source=shapes[relation][0], num_target=shapes[relation][1], device=cell_features.device)
            _validate_edge_weight(relation, None if weights is None else weights.get(relation), edge_index.shape[1], cell_features.device)
        cells, genes = self.cell_projection(cell_features), self.gene_projection(gene_features)
        for layer in self.layers:
            cells, genes = layer(cells, genes, canonical, weights)
        return HeteroScFormerOutput(cells, genes, self.cluster_head(cells))

    @torch.no_grad()
    def predict_clusters(self, cell_features: Tensor, gene_features: Tensor, relations: Mapping[EdgeType, Tensor], edge_weights: Mapping[EdgeType, Tensor] | None = None) -> Tensor:
        return self(cell_features, gene_features, relations, edge_weights).cluster_logits.argmax(dim=-1)


SpatialScFormer = LightweightHeteroScFormer
