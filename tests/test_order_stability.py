"""Order-sensitivity of the clustering step."""

from cda.clustering.order_stability import INDEPENDENT, JOINT, run_trials, shuffled, summarize
from cda.pipeline import _clustering_kwargs
from tests.conftest import make_passages

KWARGS = dict(min_cluster_size=2, silhouette_subsample=200, n_restarts=5, max_iter=80, seed=42)


def _corpora(anchored_embedder):
    """Same shape as the joint-clustering fixture: researcher writes only
    about alpha; literature covers alpha and beta, so a real (if modest)
    cluster boundary exists to be stable or unstable across orderings."""
    researcher = make_passages(
        [f"my alpha reading of the term, case {i}" for i in range(6)], source="mine.docx"
    )
    literature = make_passages(
        [f"an alpha treatment of the term, study {i}" for i in range(8)]
        + [f"a beta treatment of the term, study {i}" for i in range(10)],
        source="lit.pdf",
    )
    return researcher, literature


def test_shuffled_is_a_permutation_not_a_mutation(fake_embedder):
    passages = make_passages([f"p{i}" for i in range(10)])
    original_ids = [p.chunk_id for p in passages]
    result = shuffled(passages, seed=1)
    assert sorted(p.chunk_id for p in result) == sorted(original_ids)
    assert [p.chunk_id for p in passages] == original_ids  # input list untouched


def test_shuffled_is_deterministic_per_seed(fake_embedder):
    passages = make_passages([f"p{i}" for i in range(20)])
    a = [p.chunk_id for p in shuffled(passages, seed=7)]
    b = [p.chunk_id for p in shuffled(passages, seed=7)]
    c = [p.chunk_id for p in shuffled(passages, seed=8)]
    assert a == b
    assert a != c


def test_run_trials_joint_mode_trial_zero_is_canonical_order(anchored_embedder):
    researcher, literature = _corpora(anchored_embedder)
    results = run_trials(
        researcher,
        literature,
        anchored_embedder,
        mode=JOINT,
        trials=4,
        seed_base=0,
        n_representatives=3,
        clustering_kwargs=KWARGS,
        joint_max_k=5,
        researcher_max_k=5,
        literature_max_k=5,
    )
    assert len(results) == 4
    assert all("k" in r and "silhouette" in r for r in results)
    # Trial 0 (canonical, unshuffled) reproduces cluster_joint called directly
    # on the same unshuffled input.
    from cda.clustering.joint import cluster_joint

    direct = cluster_joint(
        list(researcher),
        list(literature),
        anchored_embedder,
        max_k=5,
        n_representatives=3,
        **KWARGS,
    )
    assert results[0]["k"] == direct.n_clusters
    assert results[0]["silhouette"] == direct.silhouette


def test_run_trials_independent_mode_tracks_both_corpora(anchored_embedder):
    researcher, literature = _corpora(anchored_embedder)
    results = run_trials(
        researcher,
        literature,
        anchored_embedder,
        mode=INDEPENDENT,
        trials=4,
        seed_base=0,
        n_representatives=3,
        clustering_kwargs=KWARGS,
        joint_max_k=5,
        researcher_max_k=5,
        literature_max_k=5,
    )
    assert len(results) == 4
    assert all(
        {"researcher_k", "researcher_silhouette", "literature_k", "literature_silhouette"}
        <= r.keys()
        for r in results
    )


def test_summarize_joint_reports_match_rate_against_canonical():
    results = [
        {"trial": 0, "k": 3, "silhouette": 0.2},
        {"trial": 1, "k": 3, "silhouette": 0.19},
        {"trial": 2, "k": 5, "silhouette": 0.15},
        {"trial": 3, "k": 3, "silhouette": 0.21},
    ]
    summary = summarize(results, JOINT)
    assert summary["mode"] == "joint"
    assert summary["canonical_k"] == 3
    assert summary["trials"] == 4
    assert summary["shuffled_trials"] == 3
    # 2 of the 3 shuffles match k=3. Trial 0 is the canonical order, not a
    # shuffle, so it is not evidence that reordering preserves k.
    assert summary["k_match_rate"] == round(2 / 3, 4)
    assert summary["k_seen"] == [3, 5]
    assert summary["silhouette_min"] == 0.15
    assert summary["silhouette_max"] == 0.21


def test_summarize_joint_full_agreement_is_1_0():
    results = [{"trial": i, "k": 4, "silhouette": 0.1} for i in range(5)]
    assert summarize(results, JOINT)["k_match_rate"] == 1.0


def test_summarize_independent_scores_each_corpus_separately():
    results = [
        {
            "trial": 0,
            "researcher_k": 2,
            "researcher_silhouette": 0.1,
            "literature_k": 4,
            "literature_silhouette": 0.09,
        },
        {
            "trial": 1,
            "researcher_k": 2,
            "researcher_silhouette": 0.11,
            "literature_k": 6,
            "literature_silhouette": 0.08,
        },
    ]
    summary = summarize(results, INDEPENDENT)
    assert summary["mode"] == "independent"
    assert summary["researcher"]["canonical_k"] == 2
    assert summary["researcher"]["k_match_rate"] == 1.0
    assert summary["literature"]["canonical_k"] == 4
    # The one shuffle disagreed, so no shuffle preserved k: 0.0, not the 0.5
    # the old rate reported by counting the canonical run as a match.
    assert summary["literature"]["k_match_rate"] == 0.0


def test_run_trials_uses_pipelines_own_clustering_kwargs_shape(anchored_embedder):
    """_clustering_kwargs's output (what pipeline.py actually passes) must be
    accepted as-is by run_trials — the real integration point."""
    from cda.config import load_config

    config = load_config(overrides={"clustering": {"silhouette_subsample": 200}})
    researcher, literature = _corpora(anchored_embedder)
    results = run_trials(
        researcher,
        literature,
        anchored_embedder,
        mode=JOINT,
        trials=2,
        seed_base=0,
        n_representatives=3,
        clustering_kwargs=_clustering_kwargs(config),
        joint_max_k=5,
        researcher_max_k=5,
        literature_max_k=5,
    )
    assert len(results) == 2


def test_shuffled_does_not_alias_the_caller_s_passages():
    """The trials must not write their cluster ids onto the live passages.

    ``cluster_passages``/``cluster_joint`` assign ``passage.cluster_id`` in
    place. A shallow copy would hand the trials the caller's own Passage
    objects, so after the check the pipeline's lists would carry the *last
    shuffled trial's* labels — and independent-mode ``compute_flags`` reads
    exactly that field.
    """
    original = make_passages(["alpha one", "alpha two", "beta one"])
    for passage in original:
        passage.cluster_id = "Cluster 0"

    trial = shuffled(original, seed=1)
    assert {id(p) for p in trial}.isdisjoint({id(p) for p in original})

    for passage in trial:
        passage.cluster_id = "Cluster 9"
    assert [p.cluster_id for p in original] == ["Cluster 0"] * 3


def test_run_trials_leaves_the_caller_s_cluster_ids_untouched(anchored_embedder):
    researcher = make_passages([f"my alpha take {i}" for i in range(6)], source="mine.docx")
    literature = make_passages(
        [f"an alpha study {i}" for i in range(8)] + [f"a beta study {i}" for i in range(10)],
        source="lit.pdf",
    )
    for passage in researcher + literature:
        passage.cluster_id = "SENTINEL"

    run_trials(
        researcher,
        literature,
        anchored_embedder,
        mode="joint",
        trials=3,
        seed_base=0,
        n_representatives=3,
        clustering_kwargs=dict(min_cluster_size=2, silhouette_subsample=200),
        joint_max_k=5,
        researcher_max_k=5,
        literature_max_k=5,
    )
    assert all(p.cluster_id == "SENTINEL" for p in researcher + literature)


def test_match_rate_can_reach_zero():
    """The canonical run is not a shuffle, so counting it made 0% unreachable.

    A term whose k never once survived reordering used to report `1/trials`
    — 20% on five trials — which reads like partial agreement rather than
    none at all.
    """
    results = [{"k": 7, "silhouette": 0.10}] + [{"k": k, "silhouette": 0.10} for k in (3, 5, 9, 12)]
    summary = summarize(results, "joint")
    assert summary["canonical_k"] == 7
    assert summary["trials"] == 5
    assert summary["shuffled_trials"] == 4
    assert summary["k_match_rate"] == 0.0


def test_match_rate_is_one_when_every_shuffle_agrees():
    results = [{"k": 2, "silhouette": 0.34} for _ in range(5)]
    summary = summarize(results, "joint")
    assert summary["k_match_rate"] == 1.0
    assert summary["shuffled_trials"] == 4


def test_match_rate_counts_only_shuffles():
    """Two of four shuffles agreeing is 50%, not the 60% the old rate gave."""
    results = [{"k": 4, "silhouette": 0.09}] + [{"k": k, "silhouette": 0.09} for k in (4, 4, 6, 3)]
    assert summarize(results, "joint")["k_match_rate"] == 0.5


def test_match_rate_is_none_not_zero_when_no_shuffles_ran():
    """ "Nothing was tested" must not read as "nothing agreed"."""
    summary = summarize([{"k": 3, "silhouette": 0.2}], "joint")
    assert summary["shuffled_trials"] == 0
    assert summary["k_match_rate"] is None


def test_canonical_reproduced_compares_trial_zero_to_the_callers_k():
    """Trial 0 re-clusters the caller's own order; it must land on the caller's k.

    The field is a real comparison, not an assertion: it is True when trial 0
    reproduces the pipeline's k, False when it does not (nondeterminism, which
    invalidates the row), and absent entirely when the caller did not supply
    the k to compare against — a check that was not run is never reported as
    passed.
    """
    results = [{"k": 3, "silhouette": 0.2}, {"k": 5, "silhouette": 0.2}]
    assert summarize(results, "joint", expected_k=3)["canonical_reproduced"] is True
    assert summarize(results, "joint", expected_k=4)["canonical_reproduced"] is False
    assert "canonical_reproduced" not in summarize(results, "joint")


def test_canonical_reproduced_is_per_corpus_in_independent_mode():
    results = [
        {
            "researcher_k": 2,
            "researcher_silhouette": 0.3,
            "literature_k": 4,
            "literature_silhouette": 0.2,
        },
        {
            "researcher_k": 2,
            "researcher_silhouette": 0.3,
            "literature_k": 4,
            "literature_silhouette": 0.2,
        },
    ]
    summary = summarize(results, "independent", expected_researcher_k=2, expected_literature_k=9)
    assert summary["researcher"]["canonical_reproduced"] is True
    assert summary["literature"]["canonical_reproduced"] is False
