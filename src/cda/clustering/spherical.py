"""Spherical k-means with silhouette-based k selection (paper §3.3).

Embeddings are L2-normalized so cosine similarity is a dot product; k-means
(k-means++ seeded) runs on the unit hypersphere with cosine distance, with
centroids renormalized after every update; the optimal k maximizes
the cosine silhouette over k = 2..max_k. Clusters below the minimum size are
merged into their nearest neighbor. Everything is seeded: the same passages
and config always produce the same partition.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from cda.corpus.embeddings import Embedder
from cda.types import ClusterSummary, Passage, passage_theme_tags


class DegenerateClusterError(ValueError):
    """A cluster whose unit vectors sum to zero, so it has no mean direction."""


def unit_centroid(rows: np.ndarray) -> np.ndarray:
    """Mean direction of L2-normalized rows, renormalized to the unit sphere.

    Raises ``DegenerateClusterError`` when the rows cancel out exactly: such a
    cluster has no mean direction, and returning the zero vector would
    silently give every member the same distance.
    """
    centroid = rows.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if not norm > 0:
        raise DegenerateClusterError(
            f"A cluster of {rows.shape[0]} embeddings has no mean direction "
            "(its unit vectors sum to zero); cosine distances to it are undefined."
        )
    return centroid / norm


def centroid_order(member_X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Member positions ordered by (cosine distance to centroid, position)."""
    return np.argsort(1.0 - member_X @ centroid, kind="stable")


def contiguous_labels(labels: np.ndarray) -> np.ndarray:
    """Relabel to 0..m-1, preserving the order of the original label values."""
    return np.unique(labels, return_inverse=True)[1].astype(int).reshape(-1)


def _assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.argmax(X @ centroids.T, axis=1)


def _update_centroids(
    X: np.ndarray, labels: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Spherical k-means update: renormalized member means.

    A cluster that is empty, or whose members cancel out, is re-seeded at a
    random row so the run keeps k centroids.
    """
    sums = np.zeros((k, X.shape[1]))
    np.add.at(sums, labels, X)
    norms = np.linalg.norm(sums, axis=1)
    centroids = np.empty_like(sums)
    for c in range(k):
        centroids[c] = sums[c] / norms[c] if norms[c] > 0 else X[rng.integers(0, X.shape[0])]
    return centroids


def kmeans_labels(
    X: np.ndarray,
    k: int,
    *,
    n_restarts: int = 5,
    max_iter: int = 80,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Spherical k-means on L2-normalized X with k-means++ seeding.

    Returns (labels, inertia): labels are the cosine assignment to the final
    centroids, relabeled contiguously, and inertia is the summed cosine
    distance to those same centroids (lower is better). Restart r for this k
    draws from the independent stream ``SeedSequence([seed, k, r])``, so no
    (k, restart) pair shares a stream and ``n_restarts`` needs no upper bound.
    """
    from sklearn.cluster import kmeans_plusplus

    if n_restarts < 1 or max_iter < 1:
        raise ValueError(
            f"k-means needs n_restarts >= 1 and max_iter >= 1, got {n_restarts} and {max_iter}."
        )
    if not np.isfinite(X).all():
        raise ValueError(
            "k-means cannot run on a non-finite embedding matrix (NaN/inf values); "
            "build it with prepare_matrix or normalize_rows."
        )
    n = X.shape[0]
    if k > n:
        raise ValueError(f"k-means cannot form k={k} clusters from {n} observations.")
    if k < 1:
        raise ValueError(f"k-means needs k >= 1, got {k}.")
    if k == 1:
        return np.zeros(n, dtype=int), float(np.sum(1.0 - X @ unit_centroid(X)))

    best_labels = np.zeros(n, dtype=int)
    best_inertia = float("inf")
    for restart in range(n_restarts):
        rng = np.random.default_rng([seed, k, restart])
        # On unit vectors squared Euclidean distance is 2 * cosine distance,
        # so k-means++ D^2 sampling here is cosine-distance seeding.
        centroids, _indices = kmeans_plusplus(X, k, random_state=int(rng.integers(0, 2**32 - 1)))
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        labels = _assign(X, centroids)
        for _ in range(max_iter):
            centroids = _update_centroids(X, labels, k, rng)
            new_labels = _assign(X, centroids)
            converged = np.array_equal(new_labels, labels)
            labels = new_labels
            if converged:
                break
        # Invariant: labels == _assign(X, centroids), converged or not.
        inertia = float(np.sum(1.0 - (X @ centroids.T)[np.arange(n), labels]))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    return contiguous_labels(best_labels), best_inertia


def _undefined_reason(labels: np.ndarray) -> str | None:
    """Why the silhouette of ``labels`` is undefined, or None when it is defined."""
    n_labels = len(np.unique(labels))
    if n_labels < 2:
        return f"fewer than two clusters among the {len(labels)} evaluated passages"
    if n_labels >= len(labels):
        return f"each of the {len(labels)} evaluated passages is its own cluster"
    return None


def silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Mean cosine silhouette, or None where it is undefined.

    Undefined means fewer than two clusters, or every point its own cluster
    (``undefined_silhouette_reason`` names which).
    """
    from sklearn.metrics import silhouette_score

    if _undefined_reason(labels) is not None:
        return None
    return float(silhouette_score(X, labels, metric="cosine"))


def merge_small_clusters(
    X: np.ndarray, labels: np.ndarray, *, min_size: int = 3
) -> tuple[np.ndarray, list[dict]]:
    """Merge clusters smaller than ``min_size`` into their nearest cluster.

    Small clusters merge in ascending label order into the large cluster with
    the most similar unit centroid; if no cluster reaches ``min_size``, the
    largest is kept. The result is always relabeled contiguously.
    """
    merges_log: list[dict] = []
    labels = labels.copy()
    unique = sorted(set(int(x) for x in labels.tolist()))
    sizes = {c: int(np.sum(labels == c)) for c in unique}
    large = [c for c in unique if sizes[c] >= min_size]
    small = [c for c in unique if sizes[c] < min_size]
    if not large:
        largest = max(unique, key=lambda c: sizes[c])
        large = [largest]
        small = [c for c in unique if c != largest]

    centroids = {c: unit_centroid(X[labels == c]) for c in large}
    for cluster in small:
        source_centroid = unit_centroid(X[labels == cluster])
        target = max(large, key=lambda c: float(np.dot(source_centroid, centroids[c])))
        merges_log.append(
            {
                "from_cluster": cluster,
                "into_cluster": target,
                "reason": f"size={sizes[cluster]} < min={min_size}",
            }
        )
        labels[labels == cluster] = target
        centroids[target] = unit_centroid(X[labels == target])
    return contiguous_labels(labels), merges_log


def require_mean_directions(X: np.ndarray, labels: np.ndarray) -> None:
    """Raise ``DegenerateClusterError`` naming the first cluster with no mean direction.

    Cluster summaries, representatives and the divergence flags all need each
    cluster's unit centroid, so a partition failing this check is unusable.
    """
    for cluster in np.unique(labels):
        try:
            unit_centroid(X[labels == cluster])
        except DegenerateClusterError as exc:
            raise DegenerateClusterError(f"Cluster {int(cluster)}: {exc}") from exc


def adjusted_rand_index(labels_a, labels_b) -> float:
    """ARI from the required scikit-learn dependency; never change estimators."""
    from sklearn.metrics import adjusted_rand_score

    if labels_a is None or labels_b is None or len(labels_a) != len(labels_b):
        raise ValueError("ARI requires two complete labelings of the same observations.")
    return float(adjusted_rand_score(labels_a, labels_b))


def hydrate_embeddings(passages: list[Passage], embedder: Embedder, batch_size: int = 256) -> None:
    """Fill in missing passage embeddings, batched to bound memory."""
    if batch_size < 1:
        raise ValueError("Embedding batch_size must be positive.")
    missing = [i for i, p in enumerate(passages) if p.embedding is None]
    if any(not passages[i].text.strip() for i in missing):
        raise ValueError("Cannot embed an empty passage; check the extracted source text.")
    for start in range(0, len(missing), batch_size):
        batch_indices = missing[start : start + batch_size]
        vectors = embedder.embed([passages[i].text for i in batch_indices])
        if len(vectors) != len(batch_indices):
            raise ValueError(
                f"Incomplete embeddings: requested {len(batch_indices)}, received {len(vectors)}. "
                "No subset will be analyzed; check the embedding provider and retry."
            )
        for passage_index, vector in zip(batch_indices, vectors, strict=True):
            passages[passage_index].embedding = vector
    prepare_matrix(passages)


def normalize_rows(
    X: np.ndarray, names: list[str], *, what: str = "passage embeddings"
) -> np.ndarray:
    """L2-normalize rows, refusing NaN/inf and all-zero vectors by name.

    This is the one place that decides what an unusable embedding is. Every
    path that builds an embedding matrix (clustering, the validation readings,
    curation, the divergence flags) goes through it, because otherwise the
    same bad row fails differently in each: a NaN crashes the k sweep with an
    opaque AttributeError (NaN inertia never beats the running best, so no
    partition is ever selected) and scipy's cosine linkage, while a zero row —
    which has no direction at all — is silently kept and lands in cluster 0.

    ``names`` labels the rows for the error message (chunk ids, cluster ids,
    positions — whatever the call site has), so the researcher learns *which*
    item is bad and not merely that something is.
    """
    norms = np.linalg.norm(X, axis=1)
    bad = ~np.isfinite(X).all(axis=1) | ~np.isfinite(norms) | (norms == 0)
    if bad.any():
        examples = ", ".join(names[i] for i in np.flatnonzero(bad)[:5])
        raise ValueError(
            f"{int(bad.sum())} of {X.shape[0]} {what} are unusable (NaN/inf or "
            f"all-zero) — e.g. {examples}.\n"
            "What to do next: passage text is re-embedded at analyze time (the "
            "corpus store keeps text, not vectors), so re-ingesting produces the "
            "same numbers — the embedding model itself is what went wrong. "
            "Delete ~/.cache/huggingface and rerun once online to refetch the "
            "model; if it persists, the passages named above may be empty or "
            "whitespace-only, in which case remove those files and try again."
        )
    return X / norms[:, np.newaxis]


def prepare_matrix(passages: list[Passage]) -> tuple[np.ndarray | None, list[Passage]]:
    """Validate every observation before constructing the normalized matrix."""
    if not passages:
        return None, passages
    missing = [p.chunk_id for p in passages if p.embedding is None]
    if missing:
        raise ValueError(f"Missing embeddings for {len(missing)} passages: {missing[:5]}")
    try:
        X = np.asarray([p.embedding for p in passages], dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Embedding vectors must be numeric and have identical dimensions."
        ) from exc
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("Embedding vectors must have one common, nonzero dimension.")
    X = normalize_rows(X, [p.chunk_id for p in passages])
    return (X if len(passages) >= 2 else None), passages


def silhouette_sample_indices(n_samples: int, subsample: int, seed: int) -> np.ndarray | None:
    """Return one reproducible silhouette evaluation sample, or all rows.

    A k sweep must call this once and reuse the result for every candidate.
    Otherwise sampling noise is mixed into the comparison between k values.
    ``silhouette_on_sample`` extends it per partition so no cluster is missed.
    """
    if n_samples > subsample:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n_samples, size=subsample, replace=False))
    return None


def covering_sample(labels: np.ndarray, sample_indices: np.ndarray) -> np.ndarray:
    """The fixed sample plus the lowest-index member of every cluster it misses.

    A random subsample can miss a small cluster entirely, which would score a
    different partition than the one being selected (or leave too few clusters
    for a silhouette at all). Adding one deterministic member per missing
    cluster keeps the evaluation set fixed across candidates apart from those
    additions. Returns sorted row indices.
    """
    clusters, first_member = np.unique(labels, return_index=True)
    missing = first_member[~np.isin(clusters, labels[sample_indices])]
    if missing.size == 0:
        return sample_indices
    return np.union1d(sample_indices, missing)


def silhouette_on_sample(
    X: np.ndarray, labels: np.ndarray, sample_indices: np.ndarray | None
) -> float | None:
    """Cosine silhouette on a caller-supplied fixed evaluation set (None if undefined).

    With a subsample, every cluster of ``labels`` is represented (see
    ``covering_sample``); without one (``None``) all rows are scored.
    """
    if sample_indices is not None:
        rows = covering_sample(labels, sample_indices)
        return silhouette(X[rows], labels[rows])
    return silhouette(X, labels)


def undefined_silhouette_reason(
    labels: np.ndarray, sample_indices: np.ndarray | None
) -> str | None:
    """Why ``silhouette_on_sample`` returned None for this partition, for the logs."""
    if sample_indices is not None:
        labels = labels[covering_sample(labels, sample_indices)]
    return _undefined_reason(labels)


def clip_k(requested_k: int, n_samples: int) -> tuple[int, dict]:
    """Cap a requested k (or max_k) at n_samples - 1 and say so.

    Returns (effective_k, record); the record is empty when nothing was
    clipped and otherwise names both values for the run's diagnostics.
    """
    effective = min(int(requested_k), n_samples - 1)
    if effective < requested_k:
        return effective, {"k_clipped_from": int(requested_k), "k_clipped_to": effective}
    return effective, {}


def select_partition_by_silhouette(
    X: np.ndarray,
    candidates: Iterable[tuple[int, np.ndarray, dict]],
    evaluation_indices: np.ndarray | None,
) -> tuple[np.ndarray | None, int, float | None, list[dict]]:
    """Keep the candidate partition with the highest silhouette.

    ``candidates`` yields (requested_k, labels, log_fields); labels is None
    when the candidate could not be built (its log_fields say why).
    Partitions with fewer than two clusters, a cluster with no mean direction
    (logged as ``invalid``), or an undefined silhouette (logged with the
    reason as ``silhouette_undefined``) are logged with ``silhouette: None``
    and never selected, so one degenerate k never aborts the sweep. Ties keep the earliest candidate. Returns (labels, k,
    silhouette, sweep_log); labels and silhouette are None when no candidate
    is valid.
    """
    best_score: float | None = None
    best_labels = None
    best_k = 1
    sweep_log: list[dict] = []
    for k, labels, log_fields in candidates:
        if labels is not None:
            try:
                require_mean_directions(X, labels)
            except DegenerateClusterError as exc:
                labels, log_fields = None, {**log_fields, "invalid": str(exc)}
        if labels is None:
            sweep_log.append({"k": k, "silhouette": None, **log_fields})
            continue
        n_unique = len(np.unique(labels))
        score = silhouette_on_sample(X, labels, evaluation_indices) if n_unique >= 2 else None
        entry: dict = {"k": k, "silhouette": None if score is None else round(score, 4)}
        if score is None:
            entry["n_unique_clusters"] = n_unique
            entry["silhouette_undefined"] = undefined_silhouette_reason(labels, evaluation_indices)
        entry.update(log_fields)
        sweep_log.append(entry)
        if score is not None and (best_score is None or score > best_score):
            best_score, best_labels, best_k = score, labels, n_unique
    return best_labels, best_k, best_score, sweep_log


def _kmeans_candidates(
    X: np.ndarray,
    ks: Iterable[int],
    *,
    n_restarts: int,
    max_iter: int,
    min_cluster_size: int,
    seed: int,
) -> Iterator[tuple[int, np.ndarray, dict]]:
    """Post-merge k-means partition for each requested k.

    A k whose small-cluster merge produces a cluster with no mean direction
    is yielded with labels None and the reason, for the selector to skip.
    """
    for k in ks:
        labels, inertia = kmeans_labels(X, k, n_restarts=n_restarts, max_iter=max_iter, seed=seed)
        log_fields: dict = {"inertia": round(inertia, 4)}
        n_kmeans = len(np.unique(labels))
        if n_kmeans < k:
            # Duplicate embeddings can leave k-means with empty clusters.
            log_fields["kmeans_unique_clusters"] = n_kmeans
        merges: list[dict] = []
        if min_cluster_size > 1:
            try:
                labels, merges = merge_small_clusters(X, labels, min_size=min_cluster_size)
            except DegenerateClusterError as exc:
                yield k, None, {**log_fields, "invalid": f"small-cluster merge: {exc}"}
                continue
        n_final = len(np.unique(labels))
        if n_final < k:
            log_fields["final_k"] = n_final
        if merges:
            log_fields["small_cluster_merges"] = len(merges)
        yield k, labels, log_fields


def select_k_by_silhouette(
    X: np.ndarray,
    *,
    max_k: int,
    n_restarts: int,
    max_iter: int,
    min_cluster_size: int,
    silhouette_subsample: int,
    seed: int,
) -> tuple[np.ndarray | None, int, float | None, list[dict]]:
    """Silhouette-best post-merge k-means partition over k = 2..max_k.

    Every k is scored on one fixed evaluation sample (extended by one member
    of any cluster it misses, see ``covering_sample``), so the reported k and
    silhouette describe the same final partition and are comparable across k.
    max_k is capped at n - 1 (see ``clip_k``). Returns (labels, k,
    silhouette, sweep_log); labels and silhouette are None when no valid
    partition exists.
    """
    n_samples = X.shape[0]
    effective_max_k, _clip = clip_k(max_k, n_samples)
    if effective_max_k < 2:
        return None, 1, None, []
    candidates = _kmeans_candidates(
        X,
        range(2, effective_max_k + 1),
        n_restarts=n_restarts,
        max_iter=max_iter,
        min_cluster_size=min_cluster_size,
        seed=seed,
    )
    evaluation_indices = silhouette_sample_indices(n_samples, silhouette_subsample, seed)
    return select_partition_by_silhouette(X, candidates, evaluation_indices)


def _summarize_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    with_embeddings: list[Passage],
    n_representatives: int,
) -> tuple[list[dict], list[dict]]:
    """Per-cluster prompt entries and distance diagnostics.

    Representatives are the ``n_representatives`` members nearest the unit
    centroid, ordered by (distance, position).
    """
    unique_clusters = sorted(set(int(x) for x in labels.tolist()))
    if len(unique_clusters) < 2:
        raise ValueError("Cluster summaries need a partition with at least two clusters.")
    centroids = {c: unit_centroid(X[labels == c]) for c in unique_clusters}

    cluster_entries: list[dict] = []
    per_cluster_stats: list[dict] = []
    for cluster in unique_clusters:
        member_positions = np.flatnonzero(labels == cluster)
        member_X = X[member_positions]
        mean_intra = float(np.mean(1.0 - member_X @ centroids[cluster]))
        mean_nearest = min(
            float(np.mean(1.0 - member_X @ centroids[other]))
            for other in unique_clusters
            if other != cluster
        )
        per_cluster_stats.append(
            {
                "cluster_id": cluster,
                "n": len(member_positions),
                "mean_intra_dist": round(mean_intra, 4),
                "mean_nearest_dist": round(mean_nearest, 4),
            }
        )

        order = centroid_order(member_X, centroids[cluster])[:n_representatives]
        representatives = [with_embeddings[int(member_positions[j])] for j in order]
        cluster_entries.append(
            {
                "cluster_id": cluster,
                "n_passages": len(member_positions),
                "representatives": [p.text for p in representatives],
                "sources": [p.source for p in representatives],
                "rep_theme_tags": [passage_theme_tags(p) for p in representatives],
                "rep_roles": [f"top{rank + 1}" for rank in range(len(representatives))],
            }
        )
    return cluster_entries, per_cluster_stats


def cluster_passages(
    passages: list[Passage],
    embedder: Embedder,
    *,
    label: str,
    max_k: int,
    n_representatives: int,
    n_restarts: int = 5,
    max_iter: int = 80,
    min_cluster_size: int = 3,
    silhouette_subsample: int = 2000,
    seed: int = 42,
    method: str = "kmeans",
    fixed_k: int | None = None,
) -> ClusterSummary:
    """Cluster one corpus and summarize it for the LLM.

    Attaches cluster ids to the passages and selects the ``n_representatives``
    passages closest to each cluster centroid (paper's appendix: five
    centroid-nearest samples).

    ``method``/``fixed_k`` are the optional resolution overrides
    (``clustering.method`` / ``clustering.k``); the default pair is the
    paper's silhouette-selected spherical k-means, unchanged.
    """
    prefix = f"[{label}] " if label else ""
    n_total = len(passages)
    if n_total < 2:
        for p in passages:
            p.cluster_id = "Cluster 0"
        return ClusterSummary(
            used=False,
            n_total_passages=n_total,
            n_clusters=1,
            silhouette=0.0,
            note=f"{prefix}Too few passages for clustering",
        )

    hydrate_embeddings(passages, embedder)
    # n_total >= 2 and hydration either embeds every passage or raises, so X
    # is the full matrix in passage order.
    X, with_embeddings = prepare_matrix(passages)

    if method == "kmeans" and fixed_k is None:
        labels, best_k, best_score, sweep_log = select_k_by_silhouette(
            X,
            max_k=max_k,
            n_restarts=n_restarts,
            max_iter=max_iter,
            min_cluster_size=min_cluster_size,
            silhouette_subsample=silhouette_subsample,
            seed=seed,
        )
    else:
        # Lazy import: the paper's default path never touches the optional
        # method module (or scipy).
        from cda.clustering.hierarchy import partition_matrix

        labels, best_k, best_score, sweep_log = partition_matrix(
            X,
            method=method,
            fixed_k=fixed_k,
            max_k=max_k,
            n_restarts=n_restarts,
            max_iter=max_iter,
            min_cluster_size=min_cluster_size,
            silhouette_subsample=silhouette_subsample,
            seed=seed,
        )
    if labels is None:
        for p in passages:
            p.cluster_id = "Cluster 0"
        return ClusterSummary(
            used=False,
            n_total_passages=n_total,
            n_clusters=1,
            silhouette=0.0,
            note=f"{prefix}No valid partition found",
        )

    for i, passage in enumerate(with_embeddings):
        passage.cluster_id = f"Cluster {int(labels[i])}"

    cluster_entries, per_cluster_stats = _summarize_clusters(
        X, labels, with_embeddings, n_representatives
    )

    note = f"{prefix}Embedding-level cluster separation detected"
    diagnostics = {
        "n_passages": n_total,
        "n_with_embeddings": int(X.shape[0]),
        "k_sweep": sweep_log,
        "best_k": best_k,
        "best_silhouette": round(best_score, 4),
        "min_cluster_size": min_cluster_size,
        "per_cluster": per_cluster_stats,
        **clip_k(max_k if fixed_k is None else fixed_k, int(X.shape[0]))[1],
    }
    if method != "kmeans" or fixed_k is not None:
        # Recorded only when the run departs from the paper's default, so
        # default-path outputs (and the demo fixtures) stay byte-identical.
        diagnostics["clustering_method"] = method
        diagnostics["fixed_k"] = fixed_k
    return ClusterSummary(
        used=True,
        n_total_passages=n_total,
        n_clusters=best_k,
        silhouette=best_score,
        note=note,
        clusters=cluster_entries,
        diagnostics=diagnostics,
    )
