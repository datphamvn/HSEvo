"""Core diversity metrics: SWDI and CDI.

These functions operate purely on numeric embeddings and depend only on
``numpy`` and ``scipy`` so they can be reused by any framework without pulling
in a deep-learning stack. The cosine similarity is reimplemented in numpy to
avoid a hard ``scikit-learn`` dependency.
"""

from __future__ import annotations

from math import log
from typing import Sequence, Union

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

ArrayLike = Union[np.ndarray, Sequence]


def _as_2d_array(embeddings: ArrayLike) -> np.ndarray:
    """Coerce an arbitrary embeddings container into a 2D ``(n, d)`` array.

    Accepts a 2D array, a list of 1D vectors, or a list of ``(1, d)`` row
    vectors (as produced by the CodeT5+ embedder).
    """
    if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
        return embeddings.astype(np.float64, copy=False)

    flattened = [np.asarray(emb, dtype=np.float64).flatten() for emb in embeddings]
    if not flattened:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(flattened)


def cosine_similarity_matrix(embeddings: ArrayLike) -> np.ndarray:
    """Pairwise cosine similarity matrix, computed with numpy only."""
    matrix = _as_2d_array(embeddings)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero for all-zero vectors.
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalized = matrix / safe_norms
    similarity = normalized @ normalized.T
    return np.clip(similarity, -1.0, 1.0)


def _cluster_nodes(similarity_matrix: np.ndarray, threshold: float) -> list[list[int]]:
    """Group nodes into clusters using complete-linkage hierarchical clustering."""
    distance_matrix = 1.0 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, None)
    # Enforce symmetry and a zero diagonal so squareform accepts the matrix.
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
    np.fill_diagonal(distance_matrix, 0.0)

    condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed, method="complete")
    cluster_ids = fcluster(linkage_matrix, t=1 - threshold, criterion="distance")

    node_clusters: dict[int, list[int]] = {}
    for node, cluster_id in enumerate(cluster_ids):
        node_clusters.setdefault(cluster_id, []).append(node)
    return list(node_clusters.values())


def _shannon_entropy(proportions: Sequence[float]) -> float:
    return -sum(p * log(p) for p in proportions if p > 0)


def shannon_wiener_index(embeddings: ArrayLike, threshold: float = 0.95) -> float:
    """Shannon-Wiener Diversity Index (SWDI).

    Measures how evenly individuals are spread across clusters in the embedding
    space at a single point in time. Higher values indicate more diversity.

    Args:
        embeddings: A ``(n, d)`` array or a sequence of ``n`` embedding vectors.
        threshold: Cosine-similarity threshold for clustering; items closer than
            ``1 - threshold`` in cosine distance are merged.

    Returns:
        The SWDI value. Returns ``0.0`` when fewer than two items are provided.
    """
    matrix = _as_2d_array(embeddings)
    total_nodes = matrix.shape[0]
    if total_nodes < 2:
        return 0.0

    similarity_matrix = cosine_similarity_matrix(matrix)
    np.fill_diagonal(similarity_matrix, 1.0)

    clusters = _cluster_nodes(similarity_matrix, threshold)
    proportions = [len(cluster) / total_nodes for cluster in clusters]
    return _shannon_entropy(proportions)


def cumulative_diversity_index(embeddings: ArrayLike) -> float:
    """Cumulative Diversity Index (CDI).

    Builds a minimum spanning tree over the Euclidean distances between all
    embeddings and returns the entropy of the normalized MST edge weights,
    reflecting the overall spread of the population.

    Args:
        embeddings: A ``(n, d)`` array or a sequence of ``n`` embedding vectors.

    Returns:
        The CDI value. Returns ``0.0`` when fewer than two items are provided.
    """
    matrix = _as_2d_array(embeddings)
    if matrix.shape[0] < 2:
        return 0.0

    distance_matrix = squareform(pdist(matrix, metric="euclidean"))
    mst = minimum_spanning_tree(distance_matrix).toarray()

    mst_distances = mst[mst != 0]
    total_distance = np.sum(mst_distances)
    if total_distance == 0:
        return 0.0

    proportions = mst_distances / total_distance
    return float(-np.sum(proportions * np.log(proportions)))
