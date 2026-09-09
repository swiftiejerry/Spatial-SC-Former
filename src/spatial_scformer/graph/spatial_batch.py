"""Spatially contiguous mini-batch construction for Spatial-scFormer.

Why this module exists
----------------------
Upstream ``scformer.utils.batch_select_whole`` shuffles cell ids with
``np.random.choice`` before cutting them into ``cell_size``-sized batches.  Each
HGT batch therefore contains a spatially random set of spots, so almost every
spot-spot spatial edge has its two endpoints in different batches and is dropped
before message passing.  Adding a spatial relation on top of that sampler would
measure a sampler artefact, not the relation.

This module keeps the exact output contract of ``batch_select_whole``
(``indices_ss``, ``Node_Ids``, ``dic``) but replaces the shuffle with a
deterministic recursive median bisection of the spatial coordinates, so each
batch is a compact tile of tissue and intra-batch spatial edges survive.

The gene selection per cell is delegated to the upstream ``process_node`` so the
spot-gene side of the graph stays byte-identical to the A0 baseline.
"""

from __future__ import annotations

import math
import os
import pickle
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix, issparse


def _validate_coordinates(coordinates: object, n_cells: int) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coordinates must be (n_cells, >=2), got {coords.shape}")
    if coords.shape[0] != n_cells:
        raise ValueError(
            f"coordinates has {coords.shape[0]} rows but RNA_matrix has {n_cells} cells"
        )
    if not np.isfinite(coords).all():
        raise ValueError("coordinates contain NaN or infinite values")
    return coords[:, :2]


def spatial_batch_order(coordinates: object, *, n_batches: int) -> list[np.ndarray]:
    """Partition cells into ``n_batches`` spatially compact, balanced blocks.

    Recursive median bisection on the wider axis.  Ties are broken by cell index
    so the partition is a pure function of the coordinates and ``n_batches``.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coordinates must be (n_cells, >=2), got {coords.shape}")
    if not isinstance(n_batches, (int, np.integer)) or isinstance(n_batches, bool):
        raise TypeError("n_batches must be an integer")
    n_batches = int(n_batches)
    if n_batches < 1:
        raise ValueError("n_batches must be >= 1")
    n_cells = coords.shape[0]
    if n_batches > n_cells:
        raise ValueError(f"n_batches={n_batches} exceeds n_cells={n_cells}")

    blocks: list[np.ndarray] = []

    def split(indices: np.ndarray, budget: int) -> None:
        if budget == 1 or indices.size <= 1:
            # Sort by cell index so every consumer (gene batching, halo expansion,
            # retention accounting) sees the same canonical order for a block.
            blocks.append(np.sort(indices))
            return
        span = coords[indices].max(axis=0) - coords[indices].min(axis=0)
        axis = 0 if span[0] >= span[1] else 1
        # lexsort keys are applied last-first: primary key is the chosen axis,
        # secondary key is the raw cell index for deterministic tie-breaking.
        order = np.lexsort((indices, coords[indices, axis]))
        ordered = indices[order]
        left_budget = budget // 2
        cut = int(round(ordered.size * left_budget / budget))
        cut = min(max(cut, left_budget), ordered.size - (budget - left_budget))
        split(ordered[:cut], left_budget)
        split(ordered[cut:], budget - left_budget)

    split(np.arange(n_cells, dtype=np.int64), n_batches)
    return blocks


def spatial_batch_select_whole(
    RNA_matrix: Any,
    coordinates: object,
    neighbor: Sequence[int] = (20,),
    cell_size: int = 30,
    save_path: str = "processed_data_spatial_subset",
) -> tuple[list[dict[str, list[int]]], np.ndarray, dict[int, dict[str, Any]]]:
    """Drop-in replacement for ``batch_select_whole`` with spatial batches.

    ``RNA_matrix`` is genes x cells, matching the upstream convention.
    Returns ``(indices_ss, Node_Ids, dic)`` with the same shapes and dtypes the
    upstream trainer expects.
    """
    from joblib import Parallel, delayed
    from scformer.utils import process_node

    indices_file = os.path.join(save_path, "indices_ss.pkl")
    node_ids_file = os.path.join(save_path, "Node_Ids.pkl")
    dic_file = os.path.join(save_path, "dic.pkl")
    if all(os.path.exists(path) for path in (indices_file, node_ids_file, dic_file)):
        with open(indices_file, "rb") as handle:
            indices_ss = pickle.load(handle)
        with open(node_ids_file, "rb") as handle:
            node_ids = pickle.load(handle)
        with open(dic_file, "rb") as handle:
            dic = pickle.load(handle)
        return indices_ss, node_ids, dic

    n_cells = RNA_matrix.shape[1]
    coords = _validate_coordinates(coordinates, n_cells)
    matrix = RNA_matrix.tocsr() if issparse(RNA_matrix) else csr_matrix(RNA_matrix)

    n_batches = math.ceil(n_cells / int(cell_size))
    blocks = spatial_batch_order(coords, n_batches=n_batches)

    node_ids = np.concatenate(blocks).astype(np.int64, copy=False)
    indices_ss: list[dict[str, list[int]]] = []
    dic: dict[int, dict[str, Any]] = {}
    for block in blocks:
        results = Parallel(n_jobs=1)(
            delayed(process_node)(int(node), matrix, list(neighbor)) for node in block
        )
        gene_indices_all: list[int] = []
        for node, gene_indices in results:
            dic[node] = {"g": gene_indices}
            gene_indices_all.extend(gene_indices)
        indices_ss.append(
            {
                "gene_index": sorted(set(gene_indices_all)),
                "cell_index": [int(node) for node in block],
            }
        )

    os.makedirs(save_path, exist_ok=True)
    with open(indices_file, "wb") as handle:
        pickle.dump(indices_ss, handle)
    with open(node_ids_file, "wb") as handle:
        pickle.dump(node_ids, handle)
    with open(dic_file, "wb") as handle:
        pickle.dump(dic, handle)
    return indices_ss, node_ids, dic


def add_spatial_halo(
    blocks: Sequence[np.ndarray], edge_index: object
) -> list[dict[str, Any]]:
    """Extend each block with the spatial neighbours of its own cells.

    A compact 30-cell tile is mostly boundary, so 26-36% of its spatial edges
    still leave the tile.  Adding a one-hop halo makes every edge incident to a
    core cell available inside the batch; the halo cells participate in message
    passing only and must be excluded from the clustering loss, which is why the
    returned dict carries ``core_size`` and ``core_index`` alongside the merged
    ``cell_index`` (core cells first, halo cells after, both index-sorted).
    """
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must be (2, n_edges), got {edges.shape}")
    neighbours: dict[int, set[int]] = {}
    for source, target in zip(edges[0].tolist(), edges[1].tolist()):
        neighbours.setdefault(int(source), set()).add(int(target))

    result: list[dict[str, Any]] = []
    for block in blocks:
        core = sorted(int(cell) for cell in np.asarray(block).reshape(-1))
        core_set = set(core)
        halo: set[int] = set()
        for cell in core:
            halo.update(neighbours.get(cell, ()))
        halo -= core_set
        result.append(
            {
                "core_index": core,
                "core_size": len(core),
                "cell_index": core + sorted(halo),
                "halo_size": len(halo),
            }
        )
    return result


def core_incident_edge_retention(
    edge_index: object, batches: Sequence[dict[str, Any]]
) -> dict[str, float]:
    """Retention of edges that have at least one endpoint in some batch core.

    With halo batches an edge is usable when a batch contains both endpoints and
    at least one of them is a core cell of that batch, because only core cells
    contribute gradients.
    """
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must be (2, n_edges), got {edges.shape}")
    n_edges = int(edges.shape[1])
    if n_edges == 0:
        return {"n_edges": 0, "n_retained": 0, "retention": 0.0}
    usable = np.zeros(n_edges, dtype=bool)
    for batch in batches:
        members = set(int(cell) for cell in batch["cell_index"])
        core = set(int(cell) for cell in batch["core_index"])
        for position in range(n_edges):
            if usable[position]:
                continue
            source = int(edges[0, position])
            target = int(edges[1, position])
            if source in members and target in members and (source in core or target in core):
                usable[position] = True
    retained = int(np.count_nonzero(usable))
    return {
        "n_edges": n_edges,
        "n_retained": retained,
        "retention": retained / n_edges,
    }


def intra_batch_edge_retention(
    edge_index: object, indices_ss: Sequence[dict[str, list[int]]]
) -> dict[str, float]:
    """Fraction of spatial edges whose two endpoints share a batch.

    This is the Stage 1a gate metric: a spatial edge can only reach HGT message
    passing when both endpoints land in the same mini-batch.
    """
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must be (2, n_edges), got {edges.shape}")
    n_edges = int(edges.shape[1])
    if n_edges == 0:
        return {"n_edges": 0, "n_retained": 0, "retention": 0.0}
    batch_of: dict[int, int] = {}
    for batch_id, batch in enumerate(indices_ss):
        for cell in batch["cell_index"]:
            batch_of[int(cell)] = batch_id
    source = np.fromiter((batch_of.get(int(c), -1) for c in edges[0]), dtype=np.int64, count=n_edges)
    target = np.fromiter((batch_of.get(int(c), -2) for c in edges[1]), dtype=np.int64, count=n_edges)
    retained = int(np.count_nonzero(source == target))
    return {
        "n_edges": n_edges,
        "n_retained": retained,
        "retention": retained / n_edges,
    }
