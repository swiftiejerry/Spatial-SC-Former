"""Training loops for fixed and prototype-based Spatial-scFormer variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from sklearn.cluster import KMeans
from torch import Tensor
from torch.nn import functional as F

from .cluster.dynamic import compute_prototypes, high_confidence_label_update, refresh_prototypes
from .losses import masked_cluster_cross_entropy, prototype_contrastive_loss
from .model.hgt import EdgeType, HeteroScFormerOutput, LightweightHeteroScFormer


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    reconstruction_weight: float = 1.0
    cluster_weight: float = 1.0
    label_smoothing: float = 0.1
    max_reconstruction_edges: int = 100_000
    mode: str = "fixed"
    warmup_epochs: int = 10
    update_interval: int = 5
    prototype_temperature: float = 0.2
    confidence_threshold: float = 0.6
    prototype_momentum: float = 0.5
    seed: int = 0

    def validate(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid epochs, learning rate, or weight decay")
        if self.reconstruction_weight < 0 or self.cluster_weight < 0 or self.max_reconstruction_edges < 1:
            raise ValueError("invalid loss weights or edge limit")
        if self.mode not in {"fixed", "dynamic"}:
            raise ValueError("mode must be 'fixed' or 'dynamic'")
        if not 0 <= self.warmup_epochs < self.epochs or self.update_interval < 1:
            raise ValueError("invalid warmup or update interval")
        if self.prototype_temperature <= 0 or not 0 <= self.confidence_threshold <= 1 or not 0 <= self.prototype_momentum < 1:
            raise ValueError("invalid prototype settings")


@dataclass
class TrainingResult:
    output: HeteroScFormerOutput
    labels: Tensor
    history: list[dict[str, float | int | str]]
    config: dict


def _kmeans_labels(embeddings: Tensor, k: int, seed: int) -> Tensor:
    labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(embeddings.detach().cpu().numpy())
    return torch.as_tensor(labels, dtype=torch.long, device=embeddings.device)


def sampled_edge_reconstruction_loss(cell_embeddings: Tensor, gene_embeddings: Tensor, cell_to_gene: Tensor, *, max_edges: int, generator: torch.Generator) -> Tensor:
    if cell_to_gene.dtype != torch.long or cell_to_gene.ndim != 2 or cell_to_gene.shape[0] != 2:
        raise ValueError("cell_to_gene must be torch.long with shape (2, E)")
    edge_count = cell_to_gene.shape[1]
    if edge_count == 0:
        return cell_embeddings.sum() * 0.0 + gene_embeddings.sum() * 0.0
    positive = cell_to_gene if edge_count <= max_edges else cell_to_gene.index_select(1, torch.randperm(edge_count, generator=generator, device=cell_to_gene.device)[:max_edges])
    cells, positive_genes = positive
    negative_genes = torch.randint(gene_embeddings.shape[0], positive_genes.shape, generator=generator, device=positive_genes.device)
    negative_genes = torch.where(negative_genes == positive_genes, (negative_genes + 1) % gene_embeddings.shape[0], negative_genes)
    scale = float(cell_embeddings.shape[1]) ** -0.5
    cell_values = cell_embeddings.index_select(0, cells)
    positive_logits = (cell_values * gene_embeddings.index_select(0, positive_genes)).sum(dim=1) * scale
    negative_logits = (cell_values * gene_embeddings.index_select(0, negative_genes)).sum(dim=1) * scale
    return 0.5 * (F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits)) + F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits)))


def train_full_graph(model: LightweightHeteroScFormer, cell_features: Tensor, gene_features: Tensor, relations: Mapping[EdgeType, Tensor], *, num_clusters: int, config: TrainingConfig, fixed_labels: Tensor | None = None, edge_weights: Mapping[EdgeType, Tensor] | None = None) -> TrainingResult:
    config.validate()
    if model.num_clusters != num_clusters:
        raise ValueError("model head width and num_clusters differ")
    if fixed_labels is not None and (fixed_labels.dtype != torch.long or fixed_labels.shape != (cell_features.shape[0],) or int(fixed_labels.min()) < 0 or int(fixed_labels.max()) >= num_clusters):
        raise ValueError("fixed_labels must be one valid long label per cell")
    if config.mode == "fixed" and fixed_labels is None:
        raise ValueError("fixed mode requires fixed_labels")
    device = cell_features.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    labels = fixed_labels.clone() if fixed_labels is not None else torch.full((cell_features.shape[0],), -1, dtype=torch.long, device=device)
    prototypes = None; confidence_mask = None; history = []
    cell_to_gene = relations[next(k for k in relations if k[0] == "cell" and k[2] == "gene")]
    for epoch in range(config.epochs):
        model.train(); output = model(cell_features, gene_features, relations, edge_weights)
        reconstruction = sampled_edge_reconstruction_loss(output.cell_embedding, output.gene_embedding, cell_to_gene, max_edges=config.max_reconstruction_edges, generator=generator)
        cluster_loss = reconstruction * 0.0; stage = "fixed" if config.mode == "fixed" else "warmup"
        if config.mode == "fixed":
            cluster_loss = F.cross_entropy(output.cluster_logits, labels, label_smoothing=config.label_smoothing)
        elif epoch >= config.warmup_epochs:
            stage = "dynamic"
            if prototypes is None:
                labels = _kmeans_labels(output.cell_embedding, num_clusters, config.seed)
                prototypes = compute_prototypes(output.cell_embedding.detach(), labels, num_clusters=num_clusters).prototypes
                confidence_mask = torch.ones_like(labels, dtype=torch.bool)
            elif (epoch - config.warmup_epochs) % config.update_interval == 0:
                update = high_confidence_label_update(output.cell_embedding.detach(), prototypes, threshold=config.confidence_threshold, tau=config.prototype_temperature, previous_labels=labels)
                labels, confidence_mask = update.labels, update.high_confidence_mask
                prototypes = refresh_prototypes(output.cell_embedding.detach(), prototypes, threshold=config.confidence_threshold, tau=config.prototype_temperature, momentum=config.prototype_momentum).state.prototypes
            cluster_loss = prototype_contrastive_loss(output.cell_embedding, prototypes, labels, confidence_mask=confidence_mask, temperature=config.prototype_temperature) + masked_cluster_cross_entropy(output.cluster_logits, labels, confidence_mask)
        loss = config.reconstruction_weight * reconstruction + config.cluster_weight * cluster_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        history.append({"epoch": epoch + 1, "stage": stage, "loss": float(loss.detach().cpu()), "reconstruction_loss": float(reconstruction.detach().cpu()), "cluster_loss": float(cluster_loss.detach().cpu()), "n_confident": int(confidence_mask.sum()) if confidence_mask is not None else len(labels), "n_label_clusters": int(torch.unique(labels[labels >= 0]).numel())})
    model.eval()
    with torch.no_grad():
        final_output = model(cell_features, gene_features, relations, edge_weights)
    if config.mode == "dynamic" and torch.any(labels < 0):
        labels = _kmeans_labels(final_output.cell_embedding, num_clusters, config.seed)
    return TrainingResult(output=final_output, labels=labels, history=history, config=asdict(config))
