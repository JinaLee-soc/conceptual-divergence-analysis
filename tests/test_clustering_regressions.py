"""Regression tests for clustering stability and input validation."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import cda.clustering.spherical as spherical
from cda.clustering.emergence import track_emergence
from cda.clustering.hierarchy import partition_matrix
from cda.clustering.joint import _cluster_entry
from cda.clustering.saturation import rank_literature, saturation_select
from cda.clustering.spherical import (
    cluster_passages,
    kmeans_labels,
    merge_small_clusters,
    normalize_rows,
    select_k_by_silhouette,
    silhouette,
)
from cda.divergence.flags import compute_joint_flags
from cda.types import JointClustering, Passage
from tests.conftest import make_passages
from tests.test_clustering import three_blob_matrix

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

SELECT_KWARGS = dict(
    n_restarts=5, max_iter=80, min_cluster_size=3, silhouette_subsample=2000, seed=42
)
PARTITION_KWARGS = dict(
    max_k=8, n_restarts=5, max_iter=80, min_cluster_size=3, silhouette_subsample=200, seed=42
)


def noisy_directions(n, d, k, spread, seed):
    """Synthetic set: k random directions plus isotropic noise, normalized."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d))
    truth = rng.integers(0, k, n)
    X = centers[truth] + spread * rng.normal(size=(n, d))
    return normalize_rows(X, [str(i) for i in range(n)]), truth


# --- 1. k-means++ seeding, independent streams, consistent inertia -------------


@pytest.mark.parametrize("data_seed", [1, 2, 3, 4, 5])
def test_kmeans_matches_sklearn_kmeanspp_on_separated_unit_vectors(data_seed):
    k = 3 + data_seed % 4
    X, _truth = noisy_directions(300, 50, k, 0.5, data_seed)
    ours, _ = kmeans_labels(X, k, n_restarts=5, max_iter=80, seed=42)
    reference = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit_predict(X)
    assert adjusted_rand_score(ours, reference) >= 0.99


@pytest.mark.parametrize(("data_seed", "true_k"), [(4, 7), (5, 8), (11, 8), (29, 8)])
def test_silhouette_sweep_recovers_true_k_where_uniform_seeding_chose_k_minus_1(data_seed, true_k):
    X, _truth = noisy_directions(400, 64, true_k, 0.6, data_seed)
    _labels, k, _score, _sweep = select_k_by_silhouette(X, max_k=10, **SELECT_KWARGS)
    assert k == true_k


def test_restart_streams_never_collide_across_k(monkeypatch):
    seen = []
    real = np.random.default_rng

    def recording_rng(seed=None):
        seen.append(tuple(np.atleast_1d(seed).tolist()))
        return real(seed)

    monkeypatch.setattr(spherical.np.random, "default_rng", recording_rng)
    X, _ = three_blob_matrix(n_per=4)
    kmeans_labels(X, 2, n_restarts=101, max_iter=2, seed=42)
    streams_k2 = set(seen)
    seen.clear()
    kmeans_labels(X, 3, n_restarts=1, max_iter=2, seed=42)
    # The old seed + k*100 + restart scheme mapped (k=2, restart=100) and
    # (k=3, restart=0) onto the same integer seed.
    assert streams_k2.isdisjoint(set(seen))


def test_inertia_and_labels_refer_to_the_same_final_centroids(monkeypatch):
    returned = []
    real = spherical._update_centroids

    def recording_update(X, labels, k, rng):
        centroids = real(X, labels, k, rng)
        returned.append(centroids)
        return centroids

    monkeypatch.setattr(spherical, "_update_centroids", recording_update)
    X, _ = noisy_directions(400, 16, 6, 1.5, 0)  # far from converging in 2 steps
    labels, inertia = kmeans_labels(X, 6, n_restarts=1, max_iter=2, seed=42)
    final = returned[-1]
    assert np.array_equal(labels, np.argmax(X @ final.T, axis=1))
    assert inertia == pytest.approx(float(np.sum(1.0 - (X @ final.T)[np.arange(400), labels])))


def test_kmeans_validates_restart_and_iteration_counts():
    X, _ = three_blob_matrix(n_per=4)
    with pytest.raises(ValueError, match="n_restarts"):
        kmeans_labels(X, 2, n_restarts=0)
    with pytest.raises(ValueError, match="cannot form"):
        kmeans_labels(X, X.shape[0] + 1)


# --- 2. one evaluation sample for fixed-k and auto paths ------------------------


@pytest.mark.parametrize("k", [3, 4, 5])
def test_fixed_k_hierarchical_silhouette_equals_auto_sweep_entry(k):
    X, _ = noisy_directions(600, 32, 5, 0.9, 3)
    auto = partition_matrix(X, method="hierarchical", fixed_k=None, **PARTITION_KWARGS)
    fixed = partition_matrix(X, method="hierarchical", fixed_k=k, **PARTITION_KWARGS)
    sweep = {entry["k"]: entry["silhouette"] for entry in auto[3]}
    assert fixed[3][0]["silhouette"] == sweep[k]


def test_fixed_k_kmeans_silhouette_equals_auto_sweep_entry():
    X, _ = noisy_directions(600, 32, 5, 0.9, 3)
    auto = select_k_by_silhouette(
        X,
        max_k=6,
        n_restarts=5,
        max_iter=80,
        min_cluster_size=3,
        silhouette_subsample=200,
        seed=42,
    )
    fixed = partition_matrix(X, method="kmeans", fixed_k=4, **PARTITION_KWARGS)
    sweep = {entry["k"]: entry["silhouette"] for entry in auto[3]}
    assert fixed[3][0]["silhouette"] == sweep[4]


# --- 3. undefined silhouette is None, never -1.0 --------------------------------


def test_undefined_silhouette_is_none_and_never_selected():
    X, _ = three_blob_matrix()
    assert silhouette(X, np.arange(X.shape[0])) is None
    # A 1-point evaluation sample, extended by one member of the other
    # cluster, leaves each evaluated passage its own cluster: undefined.
    labels, k, score, sweep = select_k_by_silhouette(
        X, max_k=2, **{**SELECT_KWARGS, "silhouette_subsample": 1}
    )
    assert labels is None and k == 1 and score is None
    assert sweep and all(entry["silhouette"] is None for entry in sweep)
    assert all("its own cluster" in entry["silhouette_undefined"] for entry in sweep)


def test_degenerate_input_reports_no_score():
    X = normalize_rows(np.ones((20, 4)), [str(i) for i in range(20)])
    labels, _k, score, _sweep = select_k_by_silhouette(X, max_k=5, **SELECT_KWARGS)
    assert labels is None and score is None


def test_fixed_k_with_undefined_silhouette_raises():
    X, _ = three_blob_matrix()
    with pytest.raises(ValueError, match="no defined silhouette"):
        partition_matrix(
            X, method="hierarchical", fixed_k=3, **{**PARTITION_KWARGS, "silhouette_subsample": 2}
        )


# --- 4. clipping is recorded; no silent substitutions --------------------------


def test_clipped_max_k_and_fixed_k_are_recorded(anchored_embedder):
    texts = [f"alpha usage {i}" for i in range(3)] + [f"beta usage {i}" for i in range(3)]
    summary = cluster_passages(
        make_passages(texts), anchored_embedder, label="t", max_k=40, n_representatives=2
    )
    assert summary.diagnostics["k_clipped_from"] == 40
    assert summary.diagnostics["k_clipped_to"] == 5
    fixed = cluster_passages(
        make_passages(texts),
        anchored_embedder,
        label="t",
        max_k=8,
        n_representatives=2,
        method="hierarchical",
        fixed_k=40,
    )
    assert fixed.diagnostics["k_clipped_from"] == 40


def test_unclipped_run_records_no_clip(anchored_embedder):
    texts = [f"alpha usage {i}" for i in range(4)] + [f"beta usage {i}" for i in range(4)]
    summary = cluster_passages(
        make_passages(texts), anchored_embedder, label="t", max_k=3, n_representatives=2
    )
    assert "k_clipped_from" not in summary.diagnostics


def test_cluster_summary_refuses_single_cluster_instead_of_zero_nearest_distance():
    X, _ = three_blob_matrix(n_per=3)
    passages = make_passages([f"p{i}" for i in range(X.shape[0])])
    with pytest.raises(ValueError, match="at least two clusters"):
        spherical._summarize_clusters(X, np.zeros(X.shape[0], dtype=int), passages, 5)


def test_similarity_ranking_refuses_missing_scores(fake_embedder):
    passages = make_passages(["a one", "b two", "c three"])
    passages[1].similarity = None
    with pytest.raises(ValueError, match="no retrieval similarity"):
        rank_literature(passages, "term", fake_embedder, mode="similarity")
    with pytest.raises(ValueError, match="mode"):
        rank_literature(passages, "term", fake_embedder, mode="similarty")


def test_saturation_requires_a_positive_streak():
    with pytest.raises(ValueError, match="consecutive_increments"):
        saturation_select(
            make_passages(["a", "b"]),
            start=1,
            step=1,
            max_passages=2,
            ari_threshold=0.8,
            consecutive_increments=0,
            max_k=2,
            n_restarts=1,
            max_iter=5,
            min_cluster_size=1,
            silhouette_subsample=100,
            seed=1,
        )


# --- 5. flags decided on exact counts -------------------------------------------


def _entry(cluster, n_researcher, n_literature, n_literature_total):
    n = n_researcher + n_literature
    X = np.tile(np.eye(3)[cluster], (n, 1))
    roles = ["researcher"] * n_researcher + ["literature"] * n_literature
    passages = [
        Passage(chunk_id=str(i), source="s", year="2020", text="t", similarity=0.0)
        for i in range(n)
    ]
    entry, _ = _cluster_entry(
        cluster=cluster,
        X=X,
        member_indices=list(range(n)),
        roles=roles,
        passages=passages,
        n_literature_total=n_literature_total,
        n_representatives=5,
    )
    return entry


def test_joint_concentration_uses_exact_shares_not_rounded_display_values():
    # 4001 / 5001 = 0.80004 > 0.80, displayed as 0.8.
    clusters = [_entry(0, 4001, 1000, 20000), _entry(1, 5, 19000, 20000)]
    assert clusters[0]["researcher_share"] == 0.8
    joint = JointClustering(
        used=True,
        n_clusters=2,
        silhouette=0.5,
        note="",
        n_researcher=4006,
        n_literature=20000,
        clusters=clusters,
    )
    assert compute_joint_flags(joint, {})["concentrated"] == ["Cluster 0"]


def test_joint_share_exactly_at_threshold_is_not_concentrated():
    clusters = [_entry(0, 8, 2, 100), _entry(1, 1, 98, 100)]
    joint = JointClustering(
        used=True,
        n_clusters=2,
        silhouette=0.5,
        note="",
        n_researcher=9,
        n_literature=100,
        clusters=clusters,
    )
    assert compute_joint_flags(joint, {})["concentrated"] == []


# --- 6. contiguous labels -------------------------------------------------------


def test_final_partition_labels_are_contiguous_with_exact_duplicates():
    # Seed 1 once selected a partition labelled [0, 1, 3, 4].
    rng = np.random.default_rng(1)
    m = rng.integers(3, 7)
    directions = rng.normal(size=(m, 8))
    counts = rng.integers(3, 12, size=m)
    X = normalize_rows(np.repeat(directions, counts, axis=0), [str(i) for i in range(counts.sum())])
    labels, k, _score, _sweep = select_k_by_silhouette(X, max_k=12, **SELECT_KWARGS)
    assert sorted(set(labels.tolist())) == list(range(k))


def test_kmeans_and_merge_return_contiguous_labels():
    X = np.repeat(np.eye(4)[:3], [5, 5, 5], axis=0)
    labels, _ = kmeans_labels(X, 6, seed=42)
    assert sorted(set(labels.tolist())) == list(range(len(set(labels.tolist()))))
    merged, _log = merge_small_clusters(X, np.array([0] * 5 + [4] * 5 + [7] * 5), min_size=3)
    assert sorted(set(merged.tolist())) == [0, 1, 2]


# --- 7. benchmarks ----------------------------------------------------------------


def test_bootstrap_jaccard_restricts_both_clusterings_to_shared_points():
    from bootstrap_stability import jaccard_best_match

    reference_points = {"a", "b", "c", "d", "e"}
    replicate = {"a": 0, "b": 0, "c": 0, "x": 0, "y": 0, "d": 1, "e": 1}
    assert jaccard_best_match({"a", "b", "c"}, replicate, reference_points) == 1.0


def test_bootstrap_unscored_cluster_is_na_not_dissolved():
    from bootstrap_stability import band, score_replicate

    assert band(None) == "n/a"
    ari, jaccards = score_replicate({"a": 0, "b": 1}, {"z": 0, "w": 0})
    assert ari is None
    assert jaccards == {0: None, 1: None}


def test_order_stability_table_excludes_canonical_trial():
    from clustering_order_stability import render

    results = [{"trial": 0, "k": 4, "silhouette": 0.08}] + [
        {"trial": i, "k": k, "silhouette": 0.08} for i, k in enumerate([5, 5, 4, 5], start=1)
    ]
    text = render("t", "joint", 10, 10, results)
    assert "| 4 | 1 | 25% ← canonical |" in text
    assert "| 5 | 3 | 75% |" in text
    assert "3/4 shuffles" in text


def test_order_stability_canonical_k_shown_even_when_no_shuffle_lands_there():
    from clustering_order_stability import render

    results = [{"trial": 0, "k": 3, "silhouette": 0.1}] + [
        {"trial": i, "k": 4, "silhouette": 0.1} for i in range(1, 6)
    ]
    text = render("t", "joint", 10, 10, results)
    assert "| 3 | 0 | 0% ← canonical |" in text
    assert "does NOT match" in text


def test_saturation_diagnostics_refuses_zero_term_vector():
    from saturation_diagnostics import probe_b_anchor_coverage

    class ZeroEmbedder:
        def embed(self, texts):
            return [[0.0, 0.0, 0.0] for _ in texts]

    with pytest.raises(ValueError, match="unusable"):
        probe_b_anchor_coverage([], "term", ZeroEmbedder(), 0)


def test_comparison_sweep_skips_undefined_silhouettes():
    import clustering_comparison

    X, _ = three_blob_matrix()
    labels, k, score = clustering_comparison.sweep_best(
        lambda k: np.arange(X.shape[0]) % k, X, range(2, 4), np.arange(2), 1
    )
    assert labels is None and score is None


# --- 8. shared helpers and representative order -------------------------------


def test_unit_centroid_refuses_cancelling_rows():
    with pytest.raises(ValueError, match="no mean direction"):
        spherical.unit_centroid(np.array([[1.0, 0.0], [-1.0, 0.0]]))


def test_representative_ties_break_by_position():
    # 40 identical members: every distance ties, so (distance, index) must
    # return the first positions; numpy's default argsort does not.
    X = np.vstack([np.tile([1.0, 0.0, 0.0], (40, 1)), np.tile([0.0, 1.0, 0.0], (5, 1))])
    labels = np.array([0] * 40 + [1] * 5)
    passages = make_passages([f"p{i}" for i in range(45)])
    entries, _stats = spherical._summarize_clusters(X, labels, passages, 5)
    assert entries[0]["representatives"] == [f"p{i}" for i in range(5)]


def test_emergence_uses_requested_representative_count(anchored_embedder):
    from datetime import datetime, timedelta

    passages = make_passages(
        [f"alpha usage of the term, case {i}." for i in range(8)]
        + [f"beta reading of the term, case {i}." for i in range(8)]
    )
    base = datetime(2019, 1, 7, 9)
    for i, passage in enumerate(passages):
        passage.metadata["date_added"] = (base + timedelta(days=7 * i)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    result = track_emergence(
        passages,
        anchored_embedder,
        start=8,
        step=4,
        match_threshold=0.8,
        max_k=4,
        n_restarts=3,
        max_iter=50,
        min_cluster_size=2,
        silhouette_subsample=200,
        seed=42,
        n_representatives=5,
    )
    assert all(len(c["representatives"]) == 5 for c in result["final"]["clusters"])
