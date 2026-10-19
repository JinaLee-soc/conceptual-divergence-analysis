"""Resolution controls: clustering.method (hierarchical) and clustering.k (fixed).

Covers the config validation in ``resolution_params``, the hierarchy module's
determinism and nesting guarantees, the dispatcher, and the wiring through
``cluster_passages`` — including that the default configuration never records
the override diagnostics (the byte-identity guarantee the demo tripwire
depends on).
"""

import numpy as np
import pytest

import cda.clustering.spherical as spherical
from cda.clustering.hierarchy import (
    cut_height,
    cut_tree_at_k,
    linkage_matrix,
    partition_matrix,
    select_cut_by_silhouette,
)
from cda.clustering.spherical import cluster_passages
from cda.config import ConfigError, load_config
from cda.pipeline import resolution_params, run_stability_check
from tests.conftest import AnchoredFakeEmbedder, make_passages
from tests.test_clustering import three_blob_matrix

PARTITION_KWARGS = dict(
    max_k=6,
    n_restarts=5,
    max_iter=80,
    min_cluster_size=2,
    silhouette_subsample=200,
    seed=42,
)


# --- resolution_params ---------------------------------------------------


def test_defaults_are_the_papers_method():
    assert resolution_params(load_config()) == {"method": "kmeans", "fixed_k": None}


def test_fixed_k_parsed_from_int_and_auto_string():
    config = load_config(overrides={"clustering": {"k": 7}})
    assert resolution_params(config)["fixed_k"] == 7
    config = load_config(overrides={"clustering": {"k": "AUTO"}})
    assert resolution_params(config)["fixed_k"] is None


def test_invalid_method_is_a_config_error():
    config = load_config(overrides={"clustering": {"method": "leiden"}})
    with pytest.raises(ConfigError, match="clustering.method"):
        resolution_params(config)


@pytest.mark.parametrize("bad", [1, 0, -3, "twelve"])
def test_invalid_k_is_a_config_error(bad):
    config = load_config(overrides={"clustering": {"k": bad}})
    with pytest.raises(ConfigError, match="clustering.k"):
        resolution_params(config)


@pytest.mark.parametrize(
    ("section", "overrides", "match"),
    [
        ("clustering", {"method": "hierarchial"}, "clustering.method"),
        ("clustering", {"k": "three"}, "clustering.k"),
        ("saturation", {"scope": "prmopt"}, "saturation.scope"),
    ],
)
def test_analyze_term_rejects_typo_configs_before_any_work(section, overrides, match):
    """A mistyped method/k/scope fails before the embedder or corpus loads.

    These settings used to be validated only inside Stage 2, after the
    embedding model had loaded and retrieval had run — minutes on a real
    corpus for what is a typo. analyze_term is called here with no embedder,
    no collection, and no corpus on disk: the ConfigError must arrive before
    any of those would be touched (a regression re-orders the failure into an
    embedder/corpus error instead)."""
    from cda.pipeline import analyze_term

    config = load_config(overrides={section: overrides})
    with pytest.raises(ConfigError, match=match):
        analyze_term("novelty", config)


# --- the tree ------------------------------------------------------------


def test_tree_and_cuts_are_deterministic():
    X, _ = three_blob_matrix()
    Z_a, Z_b = linkage_matrix(X), linkage_matrix(X)
    assert np.array_equal(Z_a, Z_b)
    assert np.array_equal(cut_tree_at_k(Z_a, 4), cut_tree_at_k(Z_b, 4))


def test_cut_recovers_separated_blobs():
    X, truth = three_blob_matrix()
    labels = cut_tree_at_k(linkage_matrix(X), 3)
    from cda.clustering.spherical import adjusted_rand_index

    assert adjusted_rand_index(labels, truth) > 0.95


def test_cut_labels_are_first_appearance_ordered():
    X, _ = three_blob_matrix()
    labels = cut_tree_at_k(linkage_matrix(X), 3)
    seen: list[int] = []
    for value in labels.tolist():
        if value not in seen:
            seen.append(value)
    assert seen == sorted(seen)  # 0, 1, 2 in order of first appearance


def test_coarse_cuts_nest_fine_cuts():
    """Every cluster at k=2 is an exact union of clusters at k=5 — the
    property flat k-means lacks and the recut affordance relies on."""
    X, _ = three_blob_matrix()
    Z = linkage_matrix(X)
    coarse, fine = cut_tree_at_k(Z, 2), cut_tree_at_k(Z, 5)
    for fine_cluster in set(fine.tolist()):
        parents = set(coarse[fine == fine_cluster].tolist())
        assert len(parents) == 1  # each fine cluster sits inside ONE coarse cluster


def test_cut_height_decreases_with_k_and_separates_cuts():
    X, _ = three_blob_matrix()
    Z = linkage_matrix(X)
    h2, h3, h5 = (cut_height(Z, k) for k in (2, 3, 5))
    assert h2 >= h3 >= h5 > 0.0
    from scipy.cluster.hierarchy import fcluster

    assert len(set(fcluster(Z, t=h3, criterion="distance").tolist())) == 3


# --- the dispatcher ------------------------------------------------------


def test_fixed_k_kmeans_partition():
    X, _ = three_blob_matrix()
    labels, k, score, sweep = partition_matrix(X, method="kmeans", fixed_k=3, **PARTITION_KWARGS)
    assert k == 3 and len(set(labels.tolist())) == 3
    assert sweep == [{"k": 3, "silhouette": round(score, 4), "fixed": True}]


def test_fixed_k_hierarchical_partition():
    X, _ = three_blob_matrix()
    labels, k, _score, sweep = partition_matrix(
        X, method="hierarchical", fixed_k=5, **PARTITION_KWARGS
    )
    assert k == 5 and len(set(labels.tolist())) == 5
    assert sweep[0]["fixed"] is True


def test_hierarchical_auto_selects_true_blob_count():
    X, _ = three_blob_matrix()
    labels, k, score, sweep = select_cut_by_silhouette(
        X, linkage_matrix(X), max_k=6, silhouette_subsample=200, seed=42
    )
    assert k == 3 and score > 0.5
    assert [entry["k"] for entry in sweep] == [2, 3, 4, 5, 6]


def test_hierarchical_auto_k_reuses_one_silhouette_sample(monkeypatch):
    X, _ = three_blob_matrix()
    seen: list[tuple[int, ...] | None] = []
    real = spherical.silhouette_on_sample

    def recording_score(X_arg, labels, sample_indices):
        seen.append(None if sample_indices is None else tuple(sample_indices.tolist()))
        return real(X_arg, labels, sample_indices)

    # The cut sweep shares spherical.select_partition_by_silhouette.
    monkeypatch.setattr(spherical, "silhouette_on_sample", recording_score)
    select_cut_by_silhouette(X, linkage_matrix(X), max_k=6, silhouette_subsample=30, seed=42)
    assert len(seen) == 5
    assert seen[0] is not None
    assert all(sample == seen[0] for sample in seen)


def test_fixed_k_capped_by_pool_size():
    X, _ = three_blob_matrix(n_per=2)  # 6 points
    labels, k, _score, _sweep = partition_matrix(
        X, method="hierarchical", fixed_k=40, **PARTITION_KWARGS
    )
    assert labels is not None and k == 5  # capped at n - 1


# --- wiring through cluster_passages ------------------------------------


def _topic_passages():
    texts = [f"{topic} usage number {i}" for topic in ("alpha", "beta", "gamma") for i in range(6)]
    return make_passages(texts)


def test_cluster_passages_default_records_no_override_diagnostics():
    summary = cluster_passages(
        _topic_passages(),
        AnchoredFakeEmbedder(),
        label="literature",
        max_k=5,
        n_representatives=2,
        min_cluster_size=2,
    )
    assert summary.used
    assert "clustering_method" not in summary.diagnostics


def test_cluster_passages_fixed_k_and_hierarchical():
    for method in ("kmeans", "hierarchical"):
        passages = _topic_passages()
        summary = cluster_passages(
            passages,
            AnchoredFakeEmbedder(),
            label="literature",
            max_k=5,
            n_representatives=2,
            min_cluster_size=2,
            method=method,
            fixed_k=3,
        )
        assert summary.used and summary.n_clusters == 3, method
        assert summary.diagnostics["clustering_method"] == method
        assert summary.diagnostics["fixed_k"] == 3
        assert {p.cluster_id for p in passages} == {"Cluster 0", "Cluster 1", "Cluster 2"}


# --- stability check and report line ------------------------------------


def test_stability_check_skipped_under_resolution_overrides():
    config = load_config(overrides={"clustering": {"method": "hierarchical"}})
    result = run_stability_check(
        [], [], config, mode="joint", embedder=AnchoredFakeEmbedder(), progress=lambda _m: None
    )
    assert result["enabled"] is False
    assert "silhouette k-selection" in result["reason"]


def test_stability_skip_reason_rendered():
    from cda.divergence.report import render_stability_markdown

    text = render_stability_markdown({"enabled": False, "reason": "clustering.k is fixed"})
    assert "skipped: clustering.k is fixed" in text
