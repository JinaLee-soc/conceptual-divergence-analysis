"""Agglomerative clustering with a movable cutoff (clustering.method: hierarchical).

An optional alternative to the paper's spherical k-means: one deterministic
average-linkage tree over the passages (cosine distance), cut into k clusters.

Invariants the rest of the system relies on:

1. Cuts are pure: no small-cluster merging is applied after a cut, so every
   partition is exactly a cut of the tree and coarse cuts are unions of
   fine cuts (what the validation GUI's movable cutoff needs).
2. The tree depends only on the embedding matrix (no seeds), so a session
   can rebuild the dendrogram it was created from.

``partition_matrix`` is the dispatcher for every configuration other than
the paper's default (kmeans + auto-k); the default path never imports this
module."""

from __future__ import annotations

import numpy as np

from cda.clustering.spherical import (
    DegenerateClusterError,
    clip_k,
    kmeans_labels,
    merge_small_clusters,
    require_mean_directions,
    select_partition_by_silhouette,
    silhouette_on_sample,
    silhouette_sample_indices,
)


def linkage_matrix(X: np.ndarray) -> np.ndarray:
    """Average-linkage tree over rows of X, cosine distance.

    Average linkage (UPGMA) keeps merge heights monotone on cosine
    dissimilarities, so "cut at height h" and "cut into k clusters" describe
    the same family of partitions — which the dendrogram cutoff line needs.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    return linkage(pdist(X, metric="cosine"), method="average")


def cut_tree_at_k(Z: np.ndarray, k: int) -> np.ndarray:
    """Cut the tree into (at most) k clusters; labels 0..m-1 in first-appearance order.

    Relabeling by first appearance over the row order makes labels a pure
    function of (Z, k) — scipy's own numbering depends on internal node order,
    which is deterministic too, but first-appearance is stable to read and
    matches the "Cluster 0 is the first passage's cluster" convention tests
    pin elsewhere.
    """
    from scipy.cluster.hierarchy import fcluster

    raw = fcluster(Z, t=k, criterion="maxclust")
    remap: dict[int, int] = {}
    labels = np.empty(len(raw), dtype=int)
    for i, value in enumerate(int(x) for x in raw):
        if value not in remap:
            remap[value] = len(remap)
        labels[i] = remap[value]
    return labels


def cut_height(Z: np.ndarray, k: int) -> float:
    """The dendrogram height at which a horizontal line yields k clusters.

    Returns the midpoint between the last merge performed and the first merge
    refused — the natural place to draw the cutoff line. k <= 1 returns a
    height above the tree; k >= n_leaves returns 0.0 (every leaf its own
    cluster).
    """
    heights = Z[:, 2]
    n_leaves = Z.shape[0] + 1
    if k <= 1:
        return float(heights[-1]) * 1.05 if len(heights) else 1.0
    if k >= n_leaves:
        return 0.0
    performed = n_leaves - k  # merges done when exactly k clusters remain
    below = float(heights[performed - 1]) if performed > 0 else 0.0
    above = float(heights[performed])
    return (below + above) / 2.0


def select_cut_by_silhouette(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    max_k: int,
    silhouette_subsample: int,
    seed: int,
) -> tuple[np.ndarray | None, int, float | None, list[dict]]:
    """Silhouette-best cut over k = 2..max_k, by the k-means path's selection rule.

    A cut containing a cluster with no mean direction is logged as invalid
    and skipped by ``select_partition_by_silhouette``.
    """
    n_samples = X.shape[0]
    effective_max_k, _clip = clip_k(max_k, n_samples)
    if effective_max_k < 2:
        return None, 1, None, []
    candidates = ((k, cut_tree_at_k(Z, k), {}) for k in range(2, effective_max_k + 1))
    evaluation_indices = silhouette_sample_indices(n_samples, silhouette_subsample, seed)
    return select_partition_by_silhouette(X, candidates, evaluation_indices)


def _degenerate_fixed_k(k: int, exc: DegenerateClusterError) -> ValueError:
    """The error for a fixed-k partition that has no usable cluster centroids."""
    return ValueError(
        f"The fixed k={k} partition is unusable: {exc} Choose a different "
        "clustering.k, or remove clustering.k to let the silhouette sweep choose."
    )


def _fixed_k_result(
    X: np.ndarray, labels: np.ndarray, k: int, silhouette_subsample: int, seed: int
) -> tuple[np.ndarray | None, int, float | None, list[dict]]:
    """Score one fixed-resolution partition on the auto path's evaluation sample."""
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        return None, 1, None, [{"k": k, "silhouette": None, "fixed": True}]
    try:
        require_mean_directions(X, labels)
    except DegenerateClusterError as exc:
        raise _degenerate_fixed_k(k, exc) from exc
    evaluation_indices = silhouette_sample_indices(X.shape[0], silhouette_subsample, seed)
    score = silhouette_on_sample(X, labels, evaluation_indices)
    if score is None:
        raise ValueError(
            f"The fixed k={k} partition has no defined silhouette on the "
            f"evaluation sample of {silhouette_subsample} passages; raise "
            "clustering.silhouette_subsample or choose a different k."
        )
    return labels, n_unique, score, [{"k": k, "silhouette": round(score, 4), "fixed": True}]


def partition_matrix(
    X: np.ndarray,
    *,
    method: str,
    fixed_k: int | None,
    max_k: int,
    n_restarts: int,
    max_iter: int,
    min_cluster_size: int,
    silhouette_subsample: int,
    seed: int,
) -> tuple[np.ndarray | None, int, float | None, list[dict]]:
    """Partition X under the configured non-default method/resolution.

    Returns the ``select_k_by_silhouette`` shape (labels, k, silhouette,
    sweep_log). A fixed k is capped at n - 1 (``clip_k``); fixed-k k-means
    still merges small clusters, so it can return fewer than k clusters.
    """
    n_samples = X.shape[0]

    if method == "kmeans":
        k, _clip = clip_k(int(fixed_k), n_samples)
        if k < 2:
            return None, 1, None, []
        labels, _inertia = kmeans_labels(X, k, n_restarts=n_restarts, max_iter=max_iter, seed=seed)
        if min_cluster_size > 1 and len(np.unique(labels)) >= 2:
            try:
                labels, _merges = merge_small_clusters(X, labels, min_size=min_cluster_size)
            except DegenerateClusterError as exc:
                raise _degenerate_fixed_k(k, exc) from exc
        return _fixed_k_result(X, labels, k, silhouette_subsample, seed)

    # Hierarchical: one tree, cut at the fixed k or at the silhouette-best k.
    if n_samples < 3:
        return None, 1, None, []
    Z = linkage_matrix(X)
    if fixed_k is None:
        return select_cut_by_silhouette(
            X, Z, max_k=max_k, silhouette_subsample=silhouette_subsample, seed=seed
        )
    k, _clip = clip_k(int(fixed_k), n_samples)
    if k < 2:
        return None, 1, None, []
    return _fixed_k_result(X, cut_tree_at_k(Z, k), k, silhouette_subsample, seed)
