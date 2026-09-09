"""Loss functions for Spatial-scFormer ablation stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class ScFormerLoss:
    total: Tensor
    kl: Tensor
    prototype: Tensor
    batch: Tensor


def _reduce(values: Tensor, reduction: str) -> Tensor:
    if reduction == "none": return values
    if reduction == "mean": return values.mean()
    if reduction == "sum": return values.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def symmetric_kl_from_logits(first_logits: Tensor, second_logits: Tensor, *, dim: int = -1, reduction: str = "mean") -> Tensor:
    if first_logits.shape != second_logits.shape:
        raise ValueError("KL inputs must have identical shapes")
    if not first_logits.is_floating_point() or not second_logits.is_floating_point():
        raise TypeError("KL inputs must use floating-point dtypes")
    first_logp, second_logp = F.log_softmax(first_logits, dim=dim), F.log_softmax(second_logits, dim=dim)
    first_p, second_p = first_logp.exp(), second_logp.exp()
    first_to_second = (first_p * (first_logp - second_logp)).sum(dim=dim)
    second_to_first = (second_p * (second_logp - first_logp)).sum(dim=dim)
    return _reduce(0.5 * (first_to_second + second_to_first), reduction)


def prototype_contrastive_loss(embeddings: Tensor, prototypes: Tensor, labels: Tensor, *, confidence_mask: Tensor | None = None, temperature: float = 1.0) -> Tensor:
    if embeddings.ndim != 2 or prototypes.ndim != 2 or embeddings.shape[1] != prototypes.shape[1]:
        raise ValueError("embeddings and prototypes must be compatible 2-D tensors")
    if labels.dtype != torch.long or labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must be long with one value per embedding")
    if temperature <= 0: raise ValueError("temperature must be positive")
    selected = torch.ones_like(labels, dtype=torch.bool) if confidence_mask is None else confidence_mask
    if selected.dtype != torch.bool or selected.shape != labels.shape: raise ValueError("confidence_mask shape/dtype mismatch")
    selected_labels = labels[selected]
    if selected_labels.numel() == 0: return embeddings.sum() * 0.0 + prototypes.sum() * 0.0
    if int(selected_labels.min()) < 0 or int(selected_labels.max()) >= prototypes.shape[0]: raise IndexError("prototype label out of range")
    logits = F.normalize(embeddings[selected], dim=1) @ F.normalize(prototypes, dim=1).T / temperature
    return F.cross_entropy(logits, selected_labels)


def masked_cluster_cross_entropy(cluster_logits: Tensor, labels: Tensor, confidence_mask: Tensor | None = None) -> Tensor:
    if cluster_logits.ndim != 2 or labels.dtype != torch.long or labels.shape != cluster_logits.shape[:1]:
        raise ValueError("cluster_logits and labels have incompatible shapes")
    selected = torch.ones_like(labels, dtype=torch.bool) if confidence_mask is None else confidence_mask
    if selected.dtype != torch.bool or selected.shape != labels.shape: raise ValueError("confidence_mask shape/dtype mismatch")
    if not torch.any(selected): return cluster_logits.sum() * 0.0
    return F.cross_entropy(cluster_logits[selected], labels[selected])


def assemble_scformer_loss(kl_loss: Tensor, *, prototype_loss: Tensor | None = None, batch_loss: Tensor | None = None, lambda_proto: float = 1.0, lambda_batch: float = 0.0) -> ScFormerLoss:
    if kl_loss.ndim != 0 or lambda_proto < 0 or lambda_batch < 0: raise ValueError("invalid KL loss or weights")
    zero = kl_loss * 0.0
    prototype, batch = zero if prototype_loss is None else prototype_loss, zero if batch_loss is None else batch_loss
    if prototype.ndim != 0 or batch.ndim != 0: raise ValueError("component losses must be scalar")
    return ScFormerLoss(total=kl_loss + lambda_proto * prototype + lambda_batch * batch, kl=kl_loss, prototype=prototype, batch=batch)
