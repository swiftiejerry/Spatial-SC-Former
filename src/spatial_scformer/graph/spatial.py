"""Deterministic spatial cell-cell graph construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SpatialGraph:
    """A directed edge view of a deduplicated undirected spatial graph."""

    edge_index: np.ndarray
    distances: np.ndarray
    edge_weight: np.ndarray | None
    num_cells: int
    mode: str


def _validate_coordinates(coordinates: object) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError(f"coordinates must be 2-D, got shape {coords.shape}")
    if coords.shape[1] == 0:
        raise ValueError("coordinates must contain at least one dimension")
    if not np.isfinite(coords).all():
        raise ValueError("coordinates contain NaN or infinite values")
    return coords


def _knn_pairs(coords: np.ndarray, k: int) -> set[tuple[int, int]]:
    n_cells = coords.shape[0]
    if n_cells < 2 or k == 0:
        return set()
    k_eff = min(k, n_cells - 1)
    tree = cKDTree(coords)
    pairs: set[tuple[int, int]] = set()
    for source in range(n_cells):
        distances, neighbors = tree.query(coords[source], k=k_eff + 1)
        distances = np.atleast_1d(distances)
        neighbors = np.atleast_1d(neighbors).astype(np.int64, copy=False)
        non_self_distances = distances[neighbors != source]
        if non_self_distances.size < k_eff:
            distances, neighbors = tree.query(coords[source], k=n_cells)
            distances = np.atleast_1d(distances)
            neighbors = np.atleast_1d(neighbors).astype(np.int64, copy=False)
            non_self_distances = distances[neighbors != source]
        kth_distance = np.partition(non_self_distances, k_eff - 1)[k_eff - 1]
        radius = np.nextafter(float(kth_distance), np.inf)
        candidates = np.asarray(tree.query_ball_point(coords[source], r=radius), dtype=np.int64)
        candidates = candidates[candidates != source]
        candidate_distances = np.linalg.norm(coords[candidates] - coords[source], axis=1)
        order = np.lexsort((candidates, candidate_distances))
        for target in candidates[order[:k_eff]]:
            left, right = sorted((source, int(target)))
            pairs.add((left, right))
    return pairs


def _radius_pairs(coords: np.ndarray, radius: float) -> set[tuple[int, int]]:
    if coords.shape[0] < 2:
        return set()
    pairs = cKDTree(coords).query_pairs(radius, output_type="set")
    return {(int(left), int(right)) for left, right in pairs}


def _expand_bidirectional(coords: np.ndarray, pairs: set[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        return np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.float64)
    directed = sorted((source, target) for left, right in pairs for source, target in ((left, right), (right, left)))
    edge_index = np.asarray(directed, dtype=np.int64).T
    distances = np.linalg.norm(coords[edge_index[0]] - coords[edge_index[1]], axis=1)
    return edge_index, distances


def normalized_distance_weights(distances: object) -> np.ndarray:
    """Convert distances to reproducible weights in ``(0, 1]``."""
    values = np.asarray(distances, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("distances must be finite and non-negative")
    positive = values[values > 0]
    if positive.size == 0:
        return np.ones_like(values)
    scale = float(np.median(positive))
    return np.exp(-values / scale)


def build_spatial_graph(
    coordinates: object,
    *,
    mode: str = "knn",
    k: int = 6,
    radius: float | None = None,
    weighted: bool = False,
) -> SpatialGraph:
    """Build a bidirectional, self-loop-free spatial graph."""
    coords = _validate_coordinates(coordinates)
    normalized_mode = mode.lower()
    if normalized_mode == "knn":
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError("k must be an integer")
        if k < 0:
            raise ValueError("k must be non-negative")
        pairs = _knn_pairs(coords, int(k))
    elif normalized_mode == "radius":
        if radius is None:
            raise ValueError("radius is required when mode='radius'")
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and non-negative")
        pairs = _radius_pairs(coords, float(radius))
    else:
        raise ValueError("mode must be either 'knn' or 'radius'")
    edge_index, distances = _expand_bidirectional(coords, pairs)
    weights = normalized_distance_weights(distances) if weighted else None
    return SpatialGraph(edge_index=edge_index, distances=distances, edge_weight=weights,
                        num_cells=coords.shape[0], mode=normalized_mode)


def build_spatial_edges(coordinates: object, **kwargs: object) -> np.ndarray:
    """Compatibility helper returning only the directed edge index."""
    return build_spatial_graph(coordinates, **kwargs).edge_index
