"""Spherical k-means, silhouette k-selection, merging, and ARI."""

import numpy as np
import pytest

import cda.clustering.spherical as spherical
from cda.clustering.spherical import (
    adjusted_rand_index,
    cluster_passages,
    kmeans_labels,
    merge_small_clusters,
    select_k_by_silhouette,
    silhouette,
)
from tests.conftest import make_passages


def three_blob_matrix(n_per=20, seed=0):
    """Three well-separated directions on the unit sphere."""
    rng = np.random.default_rng(seed)
    anchors = []
    for i in range(3):
        a = np.random.default_rng(100 + i).normal(size=16)
        anchors.append(a / np.linalg.norm(a))
    rows, truth = [], []
    for label, anchor in enumerate(anchors):
        for _ in range(n_per):
            v = anchor + 0.05 * rng.normal(size=16)
            rows.append(v / np.linalg.norm(v))
            truth.append(label)
    return np.asarray(rows), np.asarray(truth)


def test_kmeans_recovers_separated_clusters():
    X, truth = three_blob_matrix()
    labels, _ = kmeans_labels(X, 3, seed=42)
    assert adjusted_rand_index(labels, truth) > 0.95


def test_kmeans_deterministic_given_seed():
    X, _ = three_blob_matrix()
    labels_a, inertia_a = kmeans_labels(X, 3, seed=42)
    labels_b, inertia_b = kmeans_labels(X, 3, seed=42)
    assert np.array_equal(labels_a, labels_b)
    assert inertia_a == inertia_b


def test_silhouette_prefers_true_k():
    X, truth = three_blob_matrix()
    labels, k, score, sweep = select_k_by_silhouette(
        X,
        max_k=6,
        n_restarts=5,
        max_iter=80,
        min_cluster_size=2,
        silhouette_subsample=500,
        seed=42,
    )
    assert k == 3
    assert score > 0.5
    assert adjusted_rand_index(labels, truth) > 0.95
    assert [entry["k"] for entry in sweep] == [2, 3, 4, 5, 6]


def test_silhouette_single_cluster_is_invalid():
    X, _ = three_blob_matrix()
    assert silhouette(X, np.zeros(X.shape[0], dtype=int)) is None


def test_silhouette_matches_sklearn_when_a_cluster_is_singleton():
    """A singleton sample silhouette is zero, not a rewarded score of one."""
    from sklearn.metrics import silhouette_score

    X = np.asarray([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    labels = np.asarray([0, 0, 1])
    assert silhouette(X, labels) == pytest.approx(silhouette_score(X, labels, metric="cosine"))


def test_k_sweep_reuses_one_silhouette_evaluation_sample(monkeypatch):
    X, _truth = three_blob_matrix(n_per=20)
    seen: list[tuple[int, ...] | None] = []
    real = spherical.silhouette_on_sample

    def recording_score(X_arg, labels, sample_indices):
        seen.append(None if sample_indices is None else tuple(sample_indices.tolist()))
        return real(X_arg, labels, sample_indices)

    monkeypatch.setattr(spherical, "silhouette_on_sample", recording_score)
    select_k_by_silhouette(
        X,
        max_k=6,
        n_restarts=2,
        max_iter=40,
        min_cluster_size=2,
        silhouette_subsample=30,
        seed=42,
    )
    assert len(seen) == 5
    assert seen[0] is not None
    assert all(sample == seen[0] for sample in seen)


def test_k_selection_scores_post_merge_partitions(monkeypatch):
    """If requested k=3 merges to the k=2 partition, it cannot win as k=3."""
    X = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.98, -0.1],
            [0.0, 1.0],
            [0.1, 0.99],
            [-0.1, 0.98],
        ]
    )
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    def labels_for_k(_X, k, **_kwargs):
        if k == 2:
            return np.asarray([0, 0, 0, 1, 1, 1]), 1.0
        return np.asarray([0, 0, 0, 1, 1, 2]), 0.9

    monkeypatch.setattr(spherical, "kmeans_labels", labels_for_k)
    labels, k, _score, sweep = select_k_by_silhouette(
        X,
        max_k=3,
        n_restarts=1,
        max_iter=1,
        min_cluster_size=2,
        silhouette_subsample=100,
        seed=42,
    )
    assert k == 2
    assert np.array_equal(labels, np.asarray([0, 0, 0, 1, 1, 1]))
    assert sweep[1]["final_k"] == 2
    assert sweep[1]["small_cluster_merges"] == 1


def test_merge_small_clusters():
    X, _ = three_blob_matrix(n_per=10)
    labels = np.zeros(30, dtype=int)
    labels[10:20] = 1
    labels[20:] = 2
    labels[29] = 3  # a singleton carved off cluster 2
    merged, log = merge_small_clusters(X, labels, min_size=3)
    assert len(set(merged.tolist())) == 3
    assert log and log[0]["from_cluster"] == 3
    # The singleton came from blob 2, so it must merge back into that blob.
    assert merged[29] == merged[20]


def test_ari_properties():
    a = [0, 0, 1, 1, 2, 2]
    assert adjusted_rand_index(a, a) == 1.0
    assert adjusted_rand_index(a, [2, 2, 0, 0, 1, 1]) == 1.0  # relabeling invariant
    assert adjusted_rand_index(a, [0, 1, 2, 0, 1, 2]) < 0.2
    with pytest.raises(ValueError, match="complete labelings"):
        adjusted_rand_index(None, a)


def test_cluster_passages_summary(anchored_embedder):
    texts = [f"The alpha reading of novelty appears in study {i}." for i in range(8)]
    texts += [f"A beta account of novelty as market value, case {i}." for i in range(8)]
    passages = make_passages(texts)
    summary = cluster_passages(
        passages,
        anchored_embedder,
        label="literature",
        max_k=4,
        n_representatives=3,
        min_cluster_size=2,
        seed=42,
    )
    assert summary.used
    assert summary.n_clusters == 2
    assert summary.silhouette > 0.5
    assert all(len(c["representatives"]) == 3 for c in summary.clusters)
    assert all(p.cluster_id.startswith("Cluster ") for p in passages)
    assert {c["rep_roles"][0] for c in summary.clusters} == {"top1"}


def test_cluster_passages_too_few(fake_embedder):
    summary = cluster_passages(
        make_passages(["only one passage"]),
        fake_embedder,
        label="researcher",
        max_k=8,
        n_representatives=5,
    )
    assert not summary.used
    assert summary.n_clusters == 1


# --- degenerate embeddings ----------------------------------------------


def test_nan_embedding_is_refused_by_name():
    """One NaN row used to make every k-means restart's inertia NaN, so no
    partition was ever selected and the sweep crashed with an opaque
    "'NoneType' object has no attribute 'tolist'" far from the cause."""
    import pytest

    from cda.clustering.spherical import prepare_matrix

    passages = make_passages(["a", "b", "c"])
    for p in passages:
        p.embedding = [0.5] * 16
    passages[1].embedding = [float("nan")] * 16
    with pytest.raises(ValueError, match="NaN/inf or all-zero"):
        prepare_matrix(passages)


def test_zero_embedding_is_refused_by_name():
    """A zero vector has no direction: scipy's cosine pdist turned it into a
    NaN that crashed linkage (hierarchical), while k-means silently assigned
    it to cluster 0."""
    import pytest

    from cda.clustering.spherical import prepare_matrix

    passages = make_passages(["a", "b", "c"])
    for p in passages:
        p.embedding = [0.5] * 16
    passages[2].embedding = [0.0] * 16
    with pytest.raises(ValueError, match="paper.pdf::w2"):
        prepare_matrix(passages)


def test_kmeans_refuses_non_finite_matrix_instead_of_returning_none():
    import pytest

    X = np.full((6, 4), np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        kmeans_labels(X, 2, seed=42)


def test_normalize_rows_names_the_offending_row_and_advises_the_model_cache():
    """The remedy must match where bad vectors come from: passage text is
    re-embedded at analyze time, so re-ingesting cannot fix a broken model."""
    import pytest

    from cda.clustering.spherical import normalize_rows

    X = np.asarray([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError) as excinfo:
        normalize_rows(X, ["first", "second", "third"])
    message = str(excinfo.value)
    assert "second" in message and "first" not in message
    assert "~/.cache/huggingface" in message
    assert "cda ingest" not in message


def test_normalize_rows_leaves_good_vectors_normalized():
    X = np.asarray([[3.0, 4.0], [0.0, 2.0]])
    from cda.clustering.spherical import normalize_rows

    out = normalize_rows(X, ["a", "b"])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert np.allclose(out[0], [0.6, 0.8])
