"""Deterministic Top-Z cell-gene graph construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class TopZGraph:
    """A bipartite cell-to-gene edge list with deterministic edge order."""

    edge_index: np.ndarray
    scores: np.ndarray
    num_cells: int
    num_genes: int
    k: int

    @property
    def reverse_edge_index(self) -> np.ndarray:
        return self.edge_index[[1, 0]].copy()


def _validate_matrix(matrix: object) -> tuple[int, int]:
    shape = matrix.shape if sparse.issparse(matrix) else np.asarray(matrix).shape
    if len(shape) != 2:
        raise ValueError(f"expression matrix must be 2-D, got shape {shape}")
    n_cells, n_genes = int(shape[0]), int(shape[1])
    if n_genes == 0:
        raise ValueError("expression matrix must contain at least one gene")
    return n_cells, n_genes


def _stable_top_k(row: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(row).reshape(-1)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("expression matrix must contain numeric values")
    if not np.isfinite(values).all():
        raise ValueError("expression matrix contains NaN or infinite values")
    n_genes = values.size
    if k == n_genes:
        selected = np.arange(n_genes, dtype=np.int64)
    else:
        boundary = np.partition(values, n_genes - k)[n_genes - k]
        higher = np.flatnonzero(values > boundary)
        tied = np.flatnonzero(values == boundary)
        selected = np.concatenate((higher, tied[: k - higher.size])).astype(np.int64, copy=False)
    order = np.lexsort((selected, -values[selected]))
    selected = selected[order]
    return selected, values[selected]


def top_z_indices(matrix: object, k: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if k < 0:
        raise ValueError("k must be non-negative")
    n_cells, n_genes = _validate_matrix(matrix)
    selected_k = min(int(k), n_genes)
    indices = np.empty((n_cells, selected_k), dtype=np.int64)
    scores = np.empty((n_cells, selected_k), dtype=np.float64)
    if selected_k == 0:
        return indices, scores
    if sparse.issparse(matrix):
        csr = sparse.csr_matrix(matrix)
        for cell_index in range(n_cells):
            row = csr.getrow(cell_index).toarray().reshape(-1)
            indices[cell_index], scores[cell_index] = _stable_top_k(row, selected_k)
    else:
        dense = np.asarray(matrix)
        for cell_index in range(n_cells):
            indices[cell_index], scores[cell_index] = _stable_top_k(dense[cell_index], selected_k)
    return indices, scores


def build_top_z_graph(matrix: object, k: int) -> TopZGraph:
    n_cells, n_genes = _validate_matrix(matrix)
    indices, scores = top_z_indices(matrix, k)
    selected_k = indices.shape[1]
    cell_indices = np.repeat(np.arange(n_cells, dtype=np.int64), selected_k)
    edge_index = np.vstack((cell_indices, indices.reshape(-1)))
    return TopZGraph(edge_index=edge_index, scores=scores.reshape(-1),
                     num_cells=n_cells, num_genes=n_genes, k=selected_k)


def build_top_z_edges(matrix: object, k: int) -> np.ndarray:
    return build_top_z_graph(matrix, k).edge_index
