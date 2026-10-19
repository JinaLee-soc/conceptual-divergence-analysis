"""Clustering edge cases: silhouette samples that miss a cluster, runs with no
valid partition, flag-threshold types, and the benchmarks' sampling guards."""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from cda.clustering.hierarchy import linkage_matrix, partition_matrix, select_cut_by_silhouette
from cda.clustering.order_stability import (
    INDEPENDENT,
    JOINT,
    RESOLUTION_OVERRIDE_REASON,
    run_trials,
    summarize,
)
from cda.clustering.spherical import (
    covering_sample,
    select_partition_by_silhouette,
    silhouette,
    silhouette_on_sample,
    silhouette_sample_indices,
)
from cda.config import load_config
from cda.divergence import report
from cda.divergence.flags import _thresholds
from cda.types import ClusterSummary, JointClustering
from tests.conftest import make_passages

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import bootstrap_stability  # noqa: E402
import clustering_order_stability  # noqa: E402
import saturation_diagnostics  # noqa: E402

PARTITION_KWARGS = dict(n_restarts=3, max_iter=50, min_cluster_size=1, seed=42)
TRIAL_KWARGS = dict(
    min_cluster_size=2, silhouette_subsample=200, n_restarts=5, max_iter=80, seed=42
)


# --- B4: the silhouette evaluation sample covers every cluster ---------------


def _outlier_outside_sample(n=300, subsample=100, seed=42):
    """A tight blob plus one far outlier that the random evaluation sample misses."""
    rng = np.random.default_rng(0)
    X = np.zeros((n, 8))
    X[:, 0] = 1.0
    X += 0.05 * rng.standard_normal((n, 8))
    sample = set(silhouette_sample_indices(n, subsample, seed).tolist())
    outlier = next(i for i in range(n) if i not in sample)
    X[outlier] = 0.0
    X[outlier, 1] = 1.0
    return X / np.linalg.norm(X, axis=1, keepdims=True), outlier


def test_covering_sample_adds_the_lowest_index_member_of_each_missing_cluster():
    labels = np.asarray([0, 1, 1, 2, 2, 0, 1, 2])
    sample = np.asarray([0, 5])  # only cluster 0
    assert covering_sample(labels, sample).tolist() == [0, 1, 3, 5]
    # Nothing missing: the fixed sample is returned unchanged.
    assert covering_sample(labels, np.asarray([0, 1, 3])).tolist() == [0, 1, 3]


def test_fixed_k_scores_a_cluster_the_random_sample_missed():
    X, outlier = _outlier_outside_sample()
    labels, k, score, _log = partition_matrix(
        X,
        method="hierarchical",
        fixed_k=2,
        max_k=4,
        silhouette_subsample=100,
        **PARTITION_KWARGS,
    )
    assert k == 2 and score is not None
    assert int(np.sum(labels == labels[outlier])) == 1  # the outlier is its own cluster


def test_auto_sweep_keeps_a_candidate_whose_cluster_the_sample_missed():
    X, _outlier = _outlier_outside_sample()
    _labels, k, _score, sweep = select_cut_by_silhouette(
        X, linkage_matrix(X), max_k=3, silhouette_subsample=100, seed=42
    )
    assert sweep[0]["k"] == 2 and sweep[0]["silhouette"] is not None
    assert k == 2


def test_without_subsampling_every_row_is_scored_unchanged():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 6))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    labels = (X[:, 0] > 0).astype(int)
    assert silhouette_sample_indices(40, 2000, 42) is None
    assert silhouette_on_sample(X, labels, None) == silhouette(X, labels)


def test_undefined_silhouette_is_logged_with_its_reason():
    X = np.eye(4)
    _labels, _k, score, sweep = select_partition_by_silhouette(
        X, [(4, np.arange(4), {})], evaluation_indices=None
    )
    assert score is None
    assert "its own cluster" in sweep[0]["silhouette_undefined"]


# --- C1: no valid partition is never reported as robust structure -----------


def _identical(n, prefix, embedder):
    passages = make_passages([f"{prefix} {i}" for i in range(n)], source=f"{prefix}.pdf")
    for passage in passages:
        passage.embedding = [1.0] * 8
    return passages


def test_trials_without_a_partition_are_marked_and_not_called_robust(fake_embedder):
    researcher = _identical(6, "r", fake_embedder)
    literature = _identical(20, "l", fake_embedder)
    results = run_trials(
        researcher,
        literature,
        fake_embedder,
        mode=JOINT,
        trials=6,
        seed_base=0,
        n_representatives=3,
        clustering_kwargs=TRIAL_KWARGS,
        joint_max_k=5,
        researcher_max_k=5,
        literature_max_k=5,
    )
    assert all(r["partition_found"] is False and r["silhouette"] is None for r in results)
    summary = summarize(results, JOINT, expected_k=1)
    assert summary["canonical_partition_found"] is False
    assert summary["no_partition_trials"] == 6
    assert summary["silhouette_min"] is None and summary["k_match_rate"] == 0.0
    text = report.render_stability_markdown({**summary, "enabled": True})
    assert "no valid partition was found in 6 of 6 orderings" in text
    assert "robust, not just reproducible" not in text


def test_independent_trials_mark_each_corpus(fake_embedder):
    researcher = _identical(6, "r", fake_embedder)
    literature = _identical(10, "l", fake_embedder)
    results = run_trials(
        researcher,
        literature,
        fake_embedder,
        mode=INDEPENDENT,
        trials=2,
        seed_base=0,
        n_representatives=3,
        clustering_kwargs=TRIAL_KWARGS,
        joint_max_k=5,
        researcher_max_k=5,
        literature_max_k=5,
    )
    assert results[0]["researcher_partition_found"] is False
    assert results[0]["literature_partition_found"] is False
    summary = summarize(results, INDEPENDENT)
    assert summary["researcher"]["no_partition_trials"] == 2


def test_shuffles_without_a_partition_leave_the_silhouette_range_and_the_rate():
    results = [
        {"trial": 0, "k": 3, "silhouette": 0.5, "partition_found": True},
        {"trial": 1, "k": 3, "silhouette": 0.4, "partition_found": True},
        {"trial": 2, "k": 1, "silhouette": None, "partition_found": False},
    ]
    summary = summarize(results, JOINT)
    assert summary["k_seen"] == [3]
    assert (summary["silhouette_min"], summary["silhouette_max"]) == (0.4, 0.5)
    assert summary["k_match_rate"] == 0.5
    text = report.render_stability_markdown({**summary, "enabled": True})
    assert "no valid partition was found in 1 of 3 orderings" in text
    assert "robust" not in text


def test_report_header_shows_na_silhouette_without_a_partition():
    empty = ClusterSummary(used=False, n_total_passages=4, n_clusters=1, silhouette=0.0, note="x")
    joint = JointClustering(
        used=False, n_clusters=1, silhouette=0.0, note="x", n_researcher=2, n_literature=2
    )
    header = report._header("m", {}, empty, empty, joint, datetime(2026, 1, 1))
    assert "joint_silhouette=n/a" in header
    assert "sil=n/a, N=4" in header and "0.000" not in header


# --- D10: flag thresholds are validated like flag_params ---------------------


@pytest.mark.parametrize("value", [True, 0.5, -1, "two"])
def test_underrepresented_threshold_must_be_a_whole_nonnegative_number(value):
    key = "underrepresented_max_researcher_passages"
    with pytest.raises(ValueError, match=f"flags.{key}"):
        _thresholds({key: value})


@pytest.mark.parametrize("key", ["researcher_share_threshold", "literature_share_threshold"])
def test_share_thresholds_reject_booleans(key):
    with pytest.raises(ValueError, match=f"flags.{key}"):
        _thresholds({key: True})


def test_valid_thresholds_are_unchanged():
    assert _thresholds({"underrepresented_max_researcher_passages": 2.0}) == (0.8, 0.1, 2)


# --- A8: the order-stability benchmark mirrors the pipeline's refusal -------


@pytest.mark.parametrize("clustering", [{"k": 6}, {"method": "hierarchical"}])
def test_order_stability_benchmark_refuses_resolution_overrides(clustering, tmp_path, monkeypatch):
    config_file = tmp_path / "cfg.yaml"
    lines = [f"  {key}: {value}" for key, value in clustering.items()]
    config_file.write_text("clustering:\n" + "\n".join(lines) + "\n", encoding="utf-8")

    def no_work(*_args, **_kwargs):
        raise AssertionError("the benchmark did work before refusing")

    monkeypatch.setattr(clustering_order_stability, "get_embedder", no_work)
    monkeypatch.setattr(clustering_order_stability, "extract_passages", no_work)
    monkeypatch.setattr(
        sys, "argv", ["x", "--config", str(config_file), "--term", "novelty", "--trials", "2"]
    )
    assert clustering_order_stability.main() == 2
    config = load_config(config_file)
    assert clustering_order_stability.refusal_reason(config) == RESOLUTION_OVERRIDE_REASON
    assert clustering_order_stability.refusal_reason(load_config()) is None


def test_pipeline_check_and_benchmark_share_one_refusal_text(fake_embedder):
    from cda.pipeline import run_stability_check

    config = load_config(overrides={"clustering": {"k": 4}})
    result = run_stability_check(
        [], [], config, mode=JOINT, embedder=fake_embedder, progress=lambda _m: None
    )
    assert result["reason"] == RESOLUTION_OVERRIDE_REASON


def test_benchmark_render_marks_no_partition_trials():
    results = [
        {"trial": 0, "k": 1, "silhouette": None, "partition_found": False},
        {"trial": 1, "k": 1, "silhouette": None, "partition_found": False},
    ]
    text = clustering_order_stability.render("t", JOINT, 3, 3, results)
    assert "n/a (no valid partition)" in text
    assert "looks robust" not in text and "No valid joint partition" in text


# --- B7 / E3: bootstrap and saturation benchmarks ----------------------------


@pytest.mark.parametrize("fraction", [0.0, -0.2, 1.5, float("nan")])
def test_bootstrap_refuses_a_fraction_outside_zero_one(fraction, monkeypatch, capsys):
    def no_work(*_args, **_kwargs):
        raise AssertionError("work started before the fraction was checked")

    monkeypatch.setattr(bootstrap_stability, "load_config", no_work)
    monkeypatch.setattr(sys, "argv", ["x", "--config", "c.yaml", "--fraction", str(fraction)])
    with pytest.raises(SystemExit) as exc:
        bootstrap_stability.main()
    assert exc.value.code == 2
    assert "--fraction must be in (0, 1]" in capsys.readouterr().err


def test_bootstrap_refuses_a_single_source_pool(monkeypatch, capsys, fake_embedder):
    literature = make_passages(["one", "two", "three"], source="only.pdf")
    monkeypatch.setattr(bootstrap_stability, "load_config", lambda _path: load_config())
    monkeypatch.setattr(bootstrap_stability, "get_embedder", lambda _c: fake_embedder)
    monkeypatch.setattr(bootstrap_stability, "get_collection", lambda _c: None)
    monkeypatch.setattr(
        bootstrap_stability, "extract_passages", lambda *_a, **_k: ([], literature, {})
    )

    def no_clustering(*_args, **_kwargs):
        raise AssertionError("clustering started on an unresamplable pool")

    monkeypatch.setattr(bootstrap_stability, "cluster_and_snapshot", no_clustering)
    monkeypatch.setattr(sys, "argv", ["x", "--config", "c.yaml"])
    assert bootstrap_stability.main() == 2
    assert "at least 2 sources" in capsys.readouterr().out


def _bootstrap_config(mode, tmp_path):
    return load_config(
        overrides={
            "paths": {"output_dir": str(tmp_path)},
            "clustering": {"mode": mode, "min_cluster_size": 2, "silhouette_subsample": 200},
            "saturation": {"enabled": False},
        }
    )


@pytest.mark.parametrize("mode", [JOINT, INDEPENDENT])
def test_bootstrap_clusters_the_way_analyze_does(mode, tmp_path, anchored_embedder):
    researcher = make_passages([f"my alpha reading {i}" for i in range(5)], source="mine.docx")
    literature = make_passages(
        [f"alpha study {i}" for i in range(8)] + [f"beta study {i}" for i in range(8)],
        source="lit.pdf",
    )
    labels, k = bootstrap_stability.cluster_and_snapshot(
        "term", literature, _bootstrap_config(mode, tmp_path), anchored_embedder, researcher
    )
    assert k >= 2
    researcher_ids = {p.chunk_id for p in researcher}
    assert bool(researcher_ids & set(labels)) is (mode == JOINT)


def test_saturation_probe_d_plan_is_decided_from_the_source_count():
    runnable, skipped = saturation_diagnostics.plan_probe_d(4)
    assert runnable == [2]
    assert len(skipped) == 2 and "only 4 source(s)" in skipped[0]


def test_saturation_probe_d_skips_are_stated_before_probe_b(monkeypatch, capsys, fake_embedder):
    literature = make_passages(["one", "two", "three"], source="a.pdf")
    monkeypatch.setattr(saturation_diagnostics, "load_config", lambda _path: load_config())
    monkeypatch.setattr(saturation_diagnostics, "get_embedder", lambda _c: fake_embedder)
    monkeypatch.setattr(saturation_diagnostics, "get_collection", lambda _c: None)
    monkeypatch.setattr(
        saturation_diagnostics, "extract_passages", lambda *_a, **_k: ([], literature, {})
    )

    class Stop(Exception):
        pass

    def stop(*_args, **_kwargs):
        raise Stop

    monkeypatch.setattr(saturation_diagnostics, "probe_b_ordering", stop)
    monkeypatch.setattr(sys, "argv", ["x", "--config", "c.yaml"])
    with pytest.raises(Stop):
        saturation_diagnostics.main()
    assert "probe D: dropping 2 sources skipped" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("ranking", "label"), [("term", "term anchor"), ("similarity", "retrieval similarity")]
)
def test_saturation_probe_b_labels_the_ranking_it_used(ranking, label, monkeypatch, fake_embedder):
    config = load_config(overrides={"saturation": {"ranking": ranking}})
    used = []

    def recording_rank(passages, term, embedder, *, mode):
        used.append(mode)
        return list(passages)

    monkeypatch.setattr(saturation_diagnostics, "rank_literature", recording_rank)
    monkeypatch.setattr(saturation_diagnostics, "run_rule", lambda _ranked, _config: {})
    rows, _ranked = saturation_diagnostics.probe_b_ordering(
        make_passages(["a", "b"]), "term", config, fake_embedder
    )
    assert used == [ranking]
    assert rows[0]["ordering"] == f"{label} (pipeline)"
    assert rows[1]["ordering"] == f"reversed {label}"
