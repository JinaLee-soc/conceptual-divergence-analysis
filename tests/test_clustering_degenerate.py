"""Degenerate clustering candidates, collapsed k, and flag thresholds."""

import numpy as np
import pytest

import cda.clustering.hierarchy as hierarchy
import cda.clustering.spherical as spherical
from cda.clustering.hierarchy import partition_matrix, select_cut_by_silhouette
from cda.clustering.spherical import select_k_by_silhouette, unit_centroid
from cda.divergence.flags import compute_flags, compute_joint_flags
from cda.types import JointClustering
from tests.test_flags import FLAGS_CONFIG, build_corpora

SWEEP = {"n_restarts": 3, "max_iter": 50, "silhouette_subsample": 2000, "seed": 42}

# Six unit vectors where {0, 1, 4, 5} and {0, 1, 2, 3, 4, 5} both sum to zero.
CANCELLING = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [-1.0, 0.0]])


def assert_every_cluster_has_a_direction(X, labels):
    for cluster in np.unique(labels):
        unit_centroid(X[labels == cluster])


def invalid_entries(sweep_log):
    return {entry["k"]: entry for entry in sweep_log if "invalid" in entry}


# --- a degenerate merge in the k-means sweep -----------------------------


def test_sweep_skips_a_k_whose_merge_cancels_out(monkeypatch):
    """k=3 merges into a cluster summing to zero; the sweep must still pick k=2."""
    candidates = {
        2: np.asarray([0, 0, 0, 1, 1, 1]),
        3: np.asarray([0, 0, 0, 0, 1, 2]),  # both singletons merge into 0 -> sum 0
    }
    monkeypatch.setattr(spherical, "kmeans_labels", lambda _X, k, **_kw: (candidates[k], 1.0))
    labels, k, score, sweep = select_k_by_silhouette(
        CANCELLING, max_k=3, min_cluster_size=3, **SWEEP
    )
    assert k == 2 and score is not None
    assert np.array_equal(labels, candidates[2])
    invalid = invalid_entries(sweep)
    assert set(invalid) == {3}
    assert invalid[3]["silhouette"] is None
    assert "small-cluster merge" in invalid[3]["invalid"]
    assert "no mean direction" in invalid[3]["invalid"]


def test_seven_point_sweep_survives_a_cancelling_candidate():
    """A cancelling candidate must not abort a sweep over seven points."""
    X = np.asarray(
        [
            [-1.0, 0.0],
            [0.707, -0.707],
            [-0.707, 0.707],
            [0.0, 1.0],
            [-0.707, 0.707],
            [0.707, -0.707],
            [-0.707, -0.707],
        ]
    )
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    labels, _k, _score, sweep = select_k_by_silhouette(X, max_k=6, min_cluster_size=3, **SWEEP)
    assert invalid_entries(sweep), "the input should produce at least one degenerate k"
    if labels is not None:
        assert_every_cluster_has_a_direction(X, labels)


def test_sweep_never_selects_a_partition_with_a_cancelling_cluster(monkeypatch):
    """Without any merge, a k-means partition whose cluster sums to zero is invalid."""
    candidates = {
        2: np.asarray([0, 0, 1, 1, 0, 0]),  # cluster 0 = {+x, +x, -x, -x}
        3: np.asarray([0, 0, 0, 1, 2, 2]),
    }
    monkeypatch.setattr(spherical, "kmeans_labels", lambda _X, k, **_kw: (candidates[k], 1.0))
    labels, k, _score, sweep = select_k_by_silhouette(
        CANCELLING, max_k=3, min_cluster_size=1, **SWEEP
    )
    assert k == 3 and np.array_equal(labels, candidates[3])
    assert set(invalid_entries(sweep)) == {2}
    assert invalid_entries(sweep)[2]["invalid"].startswith("Cluster 0:")


def test_fixed_k_kmeans_with_a_cancelling_merge_raises_naming_k():
    X = np.asarray([[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    with pytest.raises(ValueError, match=r"fixed k=2 partition is unusable.*no mean direction"):
        partition_matrix(X, method="kmeans", fixed_k=2, max_k=6, min_cluster_size=3, **SWEEP)


# --- the hierarchical sweep and fixed cut ---------------------------------


def test_hierarchical_sweep_skips_a_cut_with_a_cancelling_cluster(monkeypatch):
    cuts = {2: np.asarray([0, 0, 1, 1, 0, 0]), 3: np.asarray([0, 0, 0, 1, 2, 2])}
    monkeypatch.setattr(hierarchy, "cut_tree_at_k", lambda _Z, k: cuts[k])
    labels, k, _score, sweep = select_cut_by_silhouette(
        CANCELLING, np.zeros((5, 4)), max_k=3, silhouette_subsample=2000, seed=42
    )
    assert k == 3 and np.array_equal(labels, cuts[3])
    assert set(invalid_entries(sweep)) == {2}


def test_hierarchical_fixed_cut_with_a_cancelling_cluster_raises_naming_k(monkeypatch):
    monkeypatch.setattr(hierarchy, "cut_tree_at_k", lambda _Z, k: np.asarray([0, 0, 1, 1, 0, 0]))
    with pytest.raises(ValueError, match=r"fixed k=2 partition is unusable"):
        partition_matrix(
            CANCELLING, method="hierarchical", fixed_k=2, max_k=6, min_cluster_size=1, **SWEEP
        )


# --- k-means ending with fewer clusters than requested --------------------


@pytest.mark.parametrize("min_cluster_size", [1, 3])
def test_sweep_log_records_collapsed_k_from_duplicate_embeddings(min_cluster_size):
    a, b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    X = np.asarray([a] * 6 + [b] * 6)
    _labels, k, _score, sweep = select_k_by_silhouette(
        X, max_k=5, min_cluster_size=min_cluster_size, **SWEEP
    )
    assert k == 2
    by_k = {entry["k"]: entry for entry in sweep}
    assert "final_k" not in by_k[2] and "kmeans_unique_clusters" not in by_k[2]
    for requested in (3, 4, 5):
        assert by_k[requested]["final_k"] == 2
        assert by_k[requested]["kmeans_unique_clusters"] == 2


# --- non-finite or out-of-range flag thresholds ---------------------------


def joint_clustering():
    rows = [(9, 1), (0, 20), (5, 5)]
    return JointClustering(
        used=True,
        n_clusters=len(rows),
        silhouette=0.1,
        note="",
        n_researcher=sum(r for r, _ in rows),
        n_literature=sum(lit for _, lit in rows),
        clusters=[
            {"cluster_id": i, "n_researcher": r, "n_literature": lit}
            for i, (r, lit) in enumerate(rows)
        ],
    )


BAD_SHARES = [float("nan"), float("inf"), -float("inf"), -0.1, 1.5]
SHARE_KEYS = ["researcher_share_threshold", "literature_share_threshold"]


@pytest.mark.parametrize("key", SHARE_KEYS)
@pytest.mark.parametrize("value", BAD_SHARES)
def test_joint_flags_refuse_bad_share_thresholds_by_name(key, value):
    with pytest.raises(ValueError, match=rf"flags\.{key} must be a finite share between 0 and 1"):
        compute_joint_flags(joint_clustering(), {key: value})


@pytest.mark.parametrize("key", SHARE_KEYS)
@pytest.mark.parametrize("value", BAD_SHARES)
def test_cross_assignment_flags_refuse_bad_share_thresholds_by_name(key, value):
    researcher, literature = build_corpora()
    with pytest.raises(ValueError, match=rf"flags\.{key} must be a finite share between 0 and 1"):
        compute_flags(researcher, literature, {**FLAGS_CONFIG, key: value})


@pytest.mark.parametrize("value", [0, 1, "0.5"])
def test_share_threshold_bounds_are_inclusive(value):
    result = compute_joint_flags(joint_clustering(), {"researcher_share_threshold": value})
    assert result["thresholds"]["researcher_share_threshold"] == float(value)
