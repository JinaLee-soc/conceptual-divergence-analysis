"""Joint clustering: shared partition, per-corpus views, composition."""

import pytest

from cda.clustering.joint import cluster_joint, composition_note, corpus_views
from cda.divergence.flags import compute_joint_flags, render_flags_markdown
from cda.divergence.report import render_joint_composition_markdown
from tests.conftest import make_passages

FLAGS_CONFIG = {
    "researcher_share_threshold": 0.80,
    "literature_share_threshold": 0.10,
    "underrepresented_max_researcher_passages": 0,
}

KWARGS = dict(max_k=5, n_representatives=3, min_cluster_size=2, silhouette_subsample=200)


def _corpora(anchored_embedder):
    """Researcher writes only about alpha; the literature covers alpha and beta.

    The anchored fake embedder puts every "alpha" text near one anchor and
    every "beta" text near another, so the joint partition must separate them —
    which makes the beta cluster a literature register the researcher never
    enters, the finding joint clustering exists to surface.
    """
    researcher = make_passages(
        [f"my alpha reading of the term, case {i}" for i in range(6)], source="mine.docx"
    )
    literature = make_passages(
        [f"an alpha treatment of the term, study {i}" for i in range(8)]
        + [f"a beta treatment of the term, study {i}" for i in range(10)],
        source="lit.pdf",
    )
    return researcher, literature


def test_joint_partition_is_shared_and_marks_researcher_location(anchored_embedder):
    researcher, literature = _corpora(anchored_embedder)
    joint = cluster_joint(researcher, literature, anchored_embedder, **KWARGS)

    assert joint.used
    assert joint.n_clusters >= 2
    assert joint.n_researcher == 6 and joint.n_literature == 18

    # Every passage from both pools carries a cluster id from the ONE partition.
    ids = {p.cluster_id for p in researcher + literature}
    assert ids <= {f"Cluster {i}" for i in range(joint.n_clusters)}

    # The researcher's own passages occupy some clusters and not others.
    with_researcher = [e for e in joint.clusters if e["n_researcher"]]
    without_researcher = [e for e in joint.clusters if not e["n_researcher"]]
    assert with_researcher and without_researcher

    # Composition adds up to the embedded totals, and shares are consistent.
    assert sum(e["n_researcher"] for e in joint.clusters) == 6
    assert sum(e["n_literature"] for e in joint.clusters) == 18
    for entry in joint.clusters:
        total = entry["n_researcher"] + entry["n_literature"]
        assert entry["n_passages"] == total
        assert entry["researcher_share"] == pytest.approx(entry["n_researcher"] / total, abs=1e-4)


def test_researcher_and_literature_land_in_the_same_alpha_cluster(anchored_embedder):
    """The point of pooling: shared usage shows up as shared membership."""
    researcher, literature = _corpora(anchored_embedder)
    cluster_joint(researcher, literature, anchored_embedder, **KWARGS)

    researcher_clusters = {p.cluster_id for p in researcher}
    alpha_lit = {p.cluster_id for p in literature if "alpha" in p.text}
    beta_lit = {p.cluster_id for p in literature if "beta" in p.text}
    assert researcher_clusters & alpha_lit  # same region as the alpha literature
    assert not (researcher_clusters & beta_lit)  # never in the beta region


def test_corpus_views_share_cluster_ids_and_omit_empty_ones(anchored_embedder):
    researcher, literature = _corpora(anchored_embedder)
    joint = cluster_joint(researcher, literature, anchored_embedder, **KWARGS)
    researcher_view, literature_view = corpus_views(joint)

    assert researcher_view.used and literature_view.used
    # Ids are the joint ones, so an id in both views names the same region.
    researcher_ids = {c["cluster_id"] for c in researcher_view.clusters}
    literature_ids = {c["cluster_id"] for c in literature_view.clusters}
    assert researcher_ids <= literature_ids
    assert researcher_ids != literature_ids  # a literature-only register exists

    # Each view counts only its own members, and reports the joint silhouette.
    assert sum(c["n_passages"] for c in researcher_view.clusters) == 6
    assert sum(c["n_passages"] for c in literature_view.clusters) == 18
    assert researcher_view.silhouette == joint.silhouette == literature_view.silhouette
    assert "JOINT" in researcher_view.note
    # Representatives in a view come from that corpus only.
    for cluster in researcher_view.clusters:
        assert all(source == "mine.docx" for source in cluster["sources"])


def test_view_of_a_corpus_with_no_passages_is_unused(anchored_embedder):
    """A drafts-less researcher side must not look like a valid clustering."""
    _researcher, literature = _corpora(anchored_embedder)
    joint = cluster_joint([], literature, anchored_embedder, **KWARGS)
    researcher_view, literature_view = corpus_views(joint)
    assert not researcher_view.used
    assert "no passages entered" in researcher_view.note
    assert literature_view.used


def test_too_few_passages_degrades_honestly(fake_embedder):
    joint = cluster_joint(make_passages(["only one"]), [], fake_embedder, **KWARGS)
    assert not joint.used
    assert "Too few passages" in joint.note
    researcher_view, _literature_view = corpus_views(joint)
    assert not researcher_view.used


def test_joint_clustering_is_deterministic(anchored_embedder):
    first = cluster_joint(*_corpora(anchored_embedder), anchored_embedder, **KWARGS)
    second = cluster_joint(*_corpora(anchored_embedder), anchored_embedder, **KWARGS)
    assert first.n_clusters == second.n_clusters
    assert first.silhouette == second.silhouette
    assert [e["n_researcher"] for e in first.clusters] == [
        e["n_researcher"] for e in second.clusters
    ]


def test_joint_flags_use_direct_membership(anchored_embedder):
    researcher, literature = _corpora(anchored_embedder)
    joint = cluster_joint(researcher, literature, anchored_embedder, **KWARGS)
    flags = compute_joint_flags(joint, FLAGS_CONFIG)

    assert flags["available"] and flags["basis"] == "joint-membership"
    # Rows mirror the partition's own counts: no cross-assignment step.
    by_cluster = {row["cluster"]: row for row in flags["rows"]}
    for entry in joint.clusters:
        row = by_cluster[f"Cluster {entry['cluster_id']}"]
        assert row["n_researcher_assigned"] == entry["n_researcher"]
        assert row["n_literature"] == entry["n_literature"]
    # The literature-only cluster receives no researcher passages at all.
    assert flags["underrepresented"]

    text = render_flags_markdown(flags)
    assert "clustered **together**" in text
    assert "n researcher members" in text


def test_joint_flags_thresholds_are_exceeded_not_met():
    """A cluster at exactly 80% researcher share is not flagged."""
    from cda.types import JointClustering

    joint = JointClustering(
        used=True,
        n_clusters=2,
        silhouette=0.2,
        note="test",
        n_researcher=9,
        n_literature=47,
        clusters=[
            {
                "cluster_id": 0,
                "n_passages": 45,
                "n_researcher": 0,
                "n_literature": 45,
                "researcher_share": 0.0,
                "literature_share": 45 / 47,
                "researcher": {"representatives": [], "sources": []},
            },
            {
                "cluster_id": 1,
                "n_passages": 10,
                "n_researcher": 8,
                "n_literature": 2,
                "researcher_share": 0.8,  # exactly at the threshold
                "literature_share": 2 / 47,
                "researcher": {"representatives": ["mine"], "sources": ["mine.docx"]},
            },
        ],
    )
    flags = compute_joint_flags(joint, FLAGS_CONFIG)
    assert flags["concentrated"] == []
    assert flags["underrepresented"] == ["Cluster 0"]

    # Nudge the counts past the threshold (9/11 > 0.8) and it flags; the
    # flags read counts, not the rounded display share.
    joint.clusters[1].update({"n_researcher": 9, "n_passages": 11})
    assert compute_joint_flags(joint, FLAGS_CONFIG)["concentrated"] == ["Cluster 1"]


def test_joint_flags_unavailable_below_two_clusters(fake_embedder):
    joint = cluster_joint(make_passages(["one"]), make_passages(["two"]), fake_embedder, **KWARGS)
    flags = compute_joint_flags(joint, FLAGS_CONFIG)
    assert not flags["available"]
    assert "fewer than two" in flags["reason"]


def test_composition_note_lists_every_cluster(anchored_embedder):
    joint = cluster_joint(*_corpora(anchored_embedder), anchored_embedder, **KWARGS)
    note = composition_note(joint)
    assert note.startswith("Joint cluster composition")
    for entry in joint.clusters:
        assert f"C{entry['cluster_id']} {entry['n_researcher']}/{entry['n_literature']}" in note


def test_composition_markdown_names_both_kinds_of_absence(anchored_embedder):
    joint = cluster_joint(*_corpora(anchored_embedder), anchored_embedder, **KWARGS)
    text = render_joint_composition_markdown(joint)
    assert "| Shared cluster |" in text
    assert "none of your passages" in text  # the beta register
    assert "your most typical passage" in text


def test_joint_flags_are_unavailable_when_one_side_is_empty(anchored_embedder):
    """With zero literature passages every cluster is trivially 100%
    researcher and 0% of the literature — each would pass the concentration
    test and read as a finding. Independent mode reports this honestly, and
    joint mode must not do worse."""
    from cda.clustering.joint import cluster_joint
    from cda.divergence.flags import compute_joint_flags
    from tests.conftest import make_passages

    researcher = make_passages(
        [f"alpha reading {i}" for i in range(4)] + [f"beta reading {i}" for i in range(4)],
        source="mine.docx",
    )
    joint = cluster_joint(
        researcher,
        [],
        anchored_embedder,
        max_k=4,
        n_representatives=2,
        n_restarts=2,
        max_iter=20,
        min_cluster_size=2,
        silhouette_subsample=100,
        seed=42,
    )
    flags = compute_joint_flags(joint, {})
    assert flags["available"] is False
    assert "no literature passages" in flags["reason"]
