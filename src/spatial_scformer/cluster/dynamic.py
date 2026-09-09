"""Prototype-based dynamic pseudo-label updates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class PrototypeState:
    prototypes: Tensor
    counts: Tensor


@dataclass
class PrototypeAssignment:
    probabilities: Tensor
    labels: Tensor
    confidence: Tensor


@dataclass
class HighConfidenceUpdate:
    labels: Tensor
    proposed_labels: Tensor
    confidence: Tensor
    high_confidence_mask: Tensor
    probabilities: Tensor


@dataclass
class PrototypeRefresh:
    state: PrototypeState
    assignment: PrototypeAssignment
    high_confidence_mask: Tensor


def _validate_embeddings(name: str, embeddings: Tensor) -> None:
    if not isinstance(embeddings, Tensor) or embeddings.ndim != 2 or not embeddings.is_floating_point() or not torch.isfinite(embeddings).all():
        raise ValueError(f"{name} must be a finite 2-D floating tensor")


def compute_prototypes(embeddings: Tensor, labels: Tensor, *, num_clusters: int | None = None) -> PrototypeState:
    _validate_embeddings("embeddings", embeddings)
    if labels.dtype != torch.long or labels.ndim != 1 or labels.shape[0] != embeddings.shape[0] or labels.device != embeddings.device:
        raise ValueError("labels must align with embeddings")
    valid = labels >= 0
    if num_clusters is None:
        if not torch.any(valid): raise ValueError("num_clusters required when all labels unassigned")
        num_clusters = int(labels[valid].max()) + 1
    if not isinstance(num_clusters, int) or num_clusters <= 0: raise ValueError("num_clusters must be positive")
    if torch.any(labels[valid] >= num_clusters): raise IndexError("label outside cluster range")
    valid_labels, valid_embeddings = labels[valid], embeddings[valid]
    sums = embeddings.new_zeros((num_clusters, embeddings.shape[1])).index_add(0, valid_labels, valid_embeddings)
    counts = torch.bincount(valid_labels, minlength=num_clusters)
    return PrototypeState(sums / counts.clamp_min(1).to(embeddings.dtype).unsqueeze(1), counts)


def assign_to_prototypes(embeddings: Tensor, prototypes: Tensor, *, tau: float = 1.0, valid_prototypes: Tensor | None = None) -> PrototypeAssignment:
    _validate_embeddings("embeddings", embeddings); _validate_embeddings("prototypes", prototypes)
    if embeddings.shape[1] != prototypes.shape[1] or embeddings.device != prototypes.device or prototypes.shape[0] == 0 or tau <= 0:
        raise ValueError("incompatible embeddings/prototypes or temperature")
    logits = F.normalize(embeddings, dim=1) @ F.normalize(prototypes, dim=1).T / tau
    if valid_prototypes is not None:
        if valid_prototypes.dtype != torch.bool or valid_prototypes.shape != (prototypes.shape[0],) or valid_prototypes.device != prototypes.device or not torch.any(valid_prototypes):
            raise ValueError("valid_prototypes must select at least one prototype")
        logits = logits.masked_fill(~valid_prototypes.unsqueeze(0), -torch.inf)
    probabilities = logits.softmax(dim=1); confidence, labels = probabilities.max(dim=1)
    return PrototypeAssignment(probabilities, labels, confidence)


def high_confidence_label_update(embeddings: Tensor, prototypes: Tensor, *, threshold: float = 0.8, tau: float = 1.0, previous_labels: Tensor | None = None, valid_prototypes: Tensor | None = None) -> HighConfidenceUpdate:
    if not 0.0 <= threshold <= 1.0: raise ValueError("threshold must be in [0, 1]")
    assignment = assign_to_prototypes(embeddings, prototypes, tau=tau, valid_prototypes=valid_prototypes)
    mask = assignment.confidence > threshold
    previous = torch.full_like(assignment.labels, -1) if previous_labels is None else previous_labels
    if previous.dtype != torch.long or previous.shape != assignment.labels.shape or previous.device != assignment.labels.device: raise ValueError("previous_labels must align")
    return HighConfidenceUpdate(torch.where(mask, assignment.labels, previous), assignment.labels, assignment.confidence, mask, assignment.probabilities)


def refresh_prototypes(embeddings: Tensor, current_prototypes: Tensor, *, threshold: float = 0.8, tau: float = 1.0, momentum: float = 0.0, valid_prototypes: Tensor | None = None) -> PrototypeRefresh:
    if not 0.0 <= threshold <= 1.0 or not 0.0 <= momentum < 1.0: raise ValueError("invalid threshold or momentum")
    assignment = assign_to_prototypes(embeddings, current_prototypes, tau=tau, valid_prototypes=valid_prototypes)
    mask = assignment.confidence > threshold
    confident_state = compute_prototypes(embeddings[mask], assignment.labels[mask], num_clusters=current_prototypes.shape[0])
    has_updates = confident_state.counts > 0
    blended = momentum * current_prototypes + (1.0 - momentum) * confident_state.prototypes
    prototypes = torch.where(has_updates.unsqueeze(1), blended, current_prototypes)
    return PrototypeRefresh(PrototypeState(prototypes, confident_state.counts), assignment, mask)
