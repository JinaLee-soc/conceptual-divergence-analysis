"""Curation semantics: revisions, verdict carry-forward, degenerate-edit guards."""

from datetime import datetime

import pytest

from cda.config import load_config
from cda.validation.curate import apply_pending_edit, rebuild_partition_clusters
from cda.validation.session import create_session, has_verdict, record_verdict

NOW = datetime(2026, 7, 14, 11, 0, 0)


def alpha_beta_session(tmp_path):
    """Three literature clusters over anchored topics: 0=alpha, 1=beta, 2=gamma."""
    passages = {}
    assignments = {}
    for cluster, topic in ((0, "alpha"), (1, "beta"), (2, "gamma")):
        for i in range(3):
            chunk_id = f"{topic}::{i}"
            passages[chunk_id] = {
                "text": f"{topic} usage of the term, case {i}",
                "side": "literature",
                "source": f"{topic}.abstract",
            }
            assignments[chunk_id] = cluster
    partitions = [
        {
            "revision": 0,
            "side": "literature",
            "kind": "original",
            "assignments": assignments,
            "clusters": [
                {
                    "cluster_id": c,
                    "n_passages": 3,
                    "silhouette": 0.2,
                    "shown": [f"{t}::{i}" for i in range(3)],
                }
                for c, t in ((0, "alpha"), (1, "beta"), (2, "gamma"))
            ],
        }
    ]
    return create_session(
        "novelty",
        params={"seed": 42, "embedding_model": str(load_config().get("embedding.model"))},
        sides={"literature": {"validated": True, "note": None}},
        partitions=partitions,
        passages=passages,
        readings=[],
        output_dir=tmp_path,
        now=NOW,
    )


def judge_all_original_clusters(session):
    """Blind verdicts on every revision-0 cluster: the first edit requires them."""
    for partition in session["partitions"]:
        if partition["revision"] != 0:
            continue
        for cluster in partition["clusters"]:
            if not has_verdict(session, partition["side"], cluster["cluster_id"], 0):
                record_verdict(
                    session,
                    side=partition["side"],
                    cluster_id=cluster["cluster_id"],
                    revision=0,
                    verdict="coherent",
                    now=NOW,
                )


def pending_merge(session, clusters=(0, 1)):
    session["pending_edit"] = {
        "edit_id": len(session["edits"]) + 1,
        "kind": "merge",
        "side": "literature",
        "clusters": list(clusters),
        "revision": 0,
        "status": "pending",
        "agent_advice": {"recommendation": "merge", "argument": "same usage", "valid": True},
    }
    return session["pending_edit"]


def test_reject_logs_and_changes_nothing(tmp_path, anchored_embedder):
    session_dir, session = alpha_beta_session(tmp_path)
    pending_merge(session)
    edit = apply_pending_edit(
        session,
        "reject",
        load_config(),
        embedder=anchored_embedder,
        session_dir=session_dir,
        now=NOW,
        progress=lambda _m: None,
    )
    assert edit["status"] == "rejected"
    assert session["pending_edit"] is None
    assert len(session["partitions"]) == 1  # no new revision


def test_merge_carries_untouched_verdicts_and_retires_affected(tmp_path, anchored_embedder):
    session_dir, session = alpha_beta_session(tmp_path)
    for cluster in (0, 1, 2):
        record_verdict(
            session,
            side="literature",
            cluster_id=cluster,
            revision=0,
            verdict="coherent",
            now=NOW,
        )
    pending_merge(session, clusters=(0, 1))
    edit = apply_pending_edit(
        session,
        "apply-merge",
        load_config(),
        embedder=anchored_embedder,
        session_dir=session_dir,
        now=NOW,
        progress=lambda _m: None,
    )
    assert edit["status"] == "applied"
    partition = session["partitions"][-1]
    assert partition["revision"] == 1 and partition["kind"] == "curated"
    # Cluster 1 was absorbed into 0; cluster 2 untouched.
    assert set(partition["assignments"].values()) == {0, 2}
    assert partition["affected_clusters"] == [0]
    # Untouched cluster 2's verdict carried forward; merged cluster 0 needs a fresh one.
    assert has_verdict(session, "literature", 2, 1)
    assert not has_verdict(session, "literature", 0, 1)
    carried = [v for v in session["verdicts"] if v.get("carried_from_revision") == 0]
    assert [v["cluster_id"] for v in carried] == [2]
    # Audit trail records silhouettes on both sides of the edit.
    assert set(edit["silhouette_before"]) == {"0", "1"}
    assert set(edit["silhouette_after"]) == {"0"}


def test_split_semantic_and_mechanical_paths(tmp_path, anchored_embedder):
    session_dir, session = alpha_beta_session(tmp_path)
    judge_all_original_clusters(session)
    members = sorted(k for k, v in session["partitions"][0]["assignments"].items() if v == 0)
    session["pending_edit"] = {
        "edit_id": 1,
        "kind": "split",
        "side": "literature",
        "clusters": [0],
        "revision": 0,
        "status": "pending",
        "agent_advice": {"recommendation": "split", "argument": "two usages", "valid": True},
        "split_proposals": {
            "semantic": {
                "assignments": {members[0]: "A", members[1]: "B", members[2]: "B"},
                "a_gloss": "ga",
                "b_gloss": "gb",
            },
            "mechanical": {members[0]: "A", members[1]: "A", members[2]: "B"},
        },
    }
    edit = apply_pending_edit(
        session,
        "apply-semantic",
        load_config(),
        embedder=anchored_embedder,
        session_dir=session_dir,
        now=NOW,
        progress=lambda _m: None,
    )
    partition = session["partitions"][-1]
    new_cluster = max(partition["assignments"].values())
    assert new_cluster == 3  # fresh id, no collision with untouched 1 and 2
    assert partition["assignments"][members[0]] == 0
    assert partition["assignments"][members[1]] == new_cluster
    assert partition["affected_clusters"] == [0, new_cluster]
    assert edit["applied_proposal"] == "semantic"


def test_unusable_semantic_proposal_is_a_friendly_error(tmp_path, anchored_embedder):
    _dir, session = alpha_beta_session(tmp_path)
    session["pending_edit"] = {
        "edit_id": 1,
        "kind": "split",
        "side": "literature",
        "clusters": [0],
        "revision": 0,
        "status": "pending",
        "agent_advice": {"recommendation": "split", "argument": "x", "valid": False},
        "split_proposals": {"semantic": None, "mechanical": {"alpha::0": "A", "alpha::1": "B"}},
    }
    with pytest.raises(ValueError, match="mechanical split or reject"):
        apply_pending_edit(
            session,
            "apply-semantic",
            load_config(),
            embedder=anchored_embedder,
            now=NOW,
            progress=lambda _m: None,
        )


def test_degenerate_split_is_refused_with_the_suggestion_intact(tmp_path, anchored_embedder):
    """An all-A "split" would change nothing while retiring the verdict.

    The refusal happens before the pending edit is cleared, so the researcher
    can still reject it properly (keeping the audit trail) instead of the
    suggestion silently vanishing.
    """
    _dir, session = alpha_beta_session(tmp_path)
    members = sorted(k for k, v in session["partitions"][0]["assignments"].items() if v == 0)
    session["pending_edit"] = {
        "edit_id": 1,
        "kind": "split",
        "side": "literature",
        "clusters": [0],
        "revision": 0,
        "status": "pending",
        "agent_advice": {"recommendation": "split", "argument": "x", "valid": True},
        "split_proposals": {
            "semantic": None,
            "mechanical": {member: "A" for member in members},
        },
    }
    with pytest.raises(ValueError, match="every passage on one side"):
        apply_pending_edit(
            session,
            "apply-mechanical",
            load_config(),
            embedder=anchored_embedder,
            now=NOW,
            progress=lambda _m: None,
        )
    assert session["pending_edit"] is not None  # still decidable
    assert len(session["partitions"]) == 1  # no revision was burned


def test_first_edit_records_real_before_silhouettes(tmp_path, anchored_embedder):
    """Revision 0 stores no per-cluster silhouettes, so the first edit must
    recompute them — an audit trail whose "before" column is always None
    cannot show any edit's effect."""
    session_dir, session = alpha_beta_session(tmp_path)
    for cluster in session["partitions"][0]["clusters"]:
        cluster["silhouette"] = None  # what run_validation actually stores
    judge_all_original_clusters(session)
    pending_merge(session, clusters=(0, 1))
    edit = apply_pending_edit(
        session,
        "apply-merge",
        load_config(),
        embedder=anchored_embedder,
        session_dir=session_dir,
        now=NOW,
        progress=lambda _m: None,
    )
    assert set(edit["silhouette_before"]) == {"0", "1"}
    assert all(isinstance(value, float) for value in edit["silhouette_before"].values())
    assert all(isinstance(value, float) for value in edit["silhouette_after"].values())


def test_decision_validated_against_kind(tmp_path, anchored_embedder):
    _dir, session = alpha_beta_session(tmp_path)
    pending_merge(session)
    with pytest.raises(ValueError, match="apply-merge"):
        apply_pending_edit(
            session,
            "apply-semantic",
            load_config(),
            embedder=anchored_embedder,
            now=NOW,
            progress=lambda _m: None,
        )
    session["pending_edit"] = None
    with pytest.raises(ValueError, match="No pending suggestion"):
        apply_pending_edit(
            session,
            "reject",
            load_config(),
            embedder=anchored_embedder,
            now=NOW,
            progress=lambda _m: None,
        )


def test_rebuild_partition_clusters_recomputes_shown_and_silhouettes(anchored_embedder):
    passages = {
        f"{topic}::{i}": {"text": f"{topic} case {i}", "side": "literature"}
        for topic in ("alpha", "beta")
        for i in range(3)
    }
    assignments = {key: (0 if key.startswith("alpha") else 1) for key in passages}
    clusters, silhouettes = rebuild_partition_clusters(
        assignments, passages, anchored_embedder, n_shown=2
    )
    assert [c["cluster_id"] for c in clusters] == [0, 1]
    assert all(len(c["shown"]) == 2 for c in clusters)
    # Anchored topics separate cleanly: positive silhouettes for both clusters.
    assert all(value > 0 for value in silhouettes.values())


# --- recut (move the dendrogram cutoff) ---------------------------------


def hierarchical_joint_session(tmp_path, embedder):
    """A joint hierarchical session whose revision 0 is a real cut (k=3) of
    the tree over all stored passages: researcher passages in the alpha
    topic, literature in alpha/beta/gamma."""
    import numpy as np

    from cda.clustering.hierarchy import cut_tree_at_k, linkage_matrix

    passages = {}
    for topic in ("alpha", "beta", "gamma"):
        for i in range(3):
            passages[f"{topic}::L{i}"] = {
                "text": f"{topic} usage of the term in the literature, case {i}",
                "side": "literature",
                "source": f"{topic}.abstract",
            }
    for i in range(2):
        passages[f"alpha::R{i}"] = {
            "text": f"alpha usage of the term by the researcher, case {i}",
            "side": "researcher",
            "source": "draft.docx",
        }

    chunk_ids = sorted(passages)
    X = np.asarray(embedder.embed([passages[c]["text"] for c in chunk_ids]), dtype=float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    labels = cut_tree_at_k(linkage_matrix(X), 3)
    pooled = {chunk: int(label) for chunk, label in zip(chunk_ids, labels, strict=True)}

    partitions = []
    for side in ("researcher", "literature"):
        assignments = {c: v for c, v in pooled.items() if passages[c]["side"] == side}
        partitions.append(
            {
                "revision": 0,
                "side": side,
                "kind": "original",
                "assignments": assignments,
                "clusters": [
                    {"cluster_id": c, "n_passages": n, "silhouette": None, "shown": []}
                    for c, n in sorted(
                        {
                            v: sum(1 for x in assignments.values() if x == v)
                            for v in assignments.values()
                        }.items()
                    )
                ],
            }
        )
    return create_session(
        "novelty",
        params={
            "seed": 42,
            "embedding_model": str(load_config().get("embedding.model")),
            "clustering_mode": "joint",
            "clustering_method": "hierarchical",
        },
        sides={
            "researcher": {"validated": True, "note": None},
            "literature": {"validated": True, "note": None},
        },
        partitions=partitions,
        passages=passages,
        readings=[],
        output_dir=tmp_path,
        now=NOW,
    )


def test_recut_requires_joint_hierarchical(tmp_path, anchored_embedder):
    from cda.validation.curate import apply_recut

    _dir, session = alpha_beta_session(tmp_path)  # no method params recorded
    with pytest.raises(ValueError, match="joint hierarchical"):
        apply_recut(session, 2, load_config(), embedder=anchored_embedder, now=NOW)


def test_recut_refused_while_an_edit_is_pending(tmp_path, anchored_embedder):
    from cda.validation.curate import apply_recut

    _dir, session = hierarchical_joint_session(tmp_path, anchored_embedder)
    session["pending_edit"] = {"kind": "merge"}
    with pytest.raises(ValueError, match="pending"):
        apply_recut(session, 2, load_config(), embedder=anchored_embedder, now=NOW)


def test_recut_k_bounds_are_validated(tmp_path, anchored_embedder):
    from cda.validation.curate import apply_recut

    _dir, session = hierarchical_joint_session(tmp_path, anchored_embedder)
    for bad in (1, len(session["passages"])):
        with pytest.raises(ValueError, match="between 2 and"):
            apply_recut(session, bad, load_config(), embedder=anchored_embedder, now=NOW)


def test_recut_carries_verdicts_for_unchanged_clusters_only(tmp_path, anchored_embedder):
    """Coarsening k=3 to k=2 merges two topic clusters; the third keeps its
    pooled membership, so it keeps its id and its verdict on every side that
    holds it, while the merged clusters retire to the inspection queue."""
    from cda.validation.curate import apply_recut
    from cda.validation.session import current_revision, partition_for

    session_dir, session = hierarchical_joint_session(tmp_path, anchored_embedder)
    for side in ("researcher", "literature"):
        for cluster in partition_for(session, side, 0)["clusters"]:
            record_verdict(
                session,
                side=side,
                cluster_id=cluster["cluster_id"],
                revision=0,
                verdict="coherent",
                notes="",
                now=NOW,
            )

    edit = apply_recut(
        session, 2, load_config(), embedder=anchored_embedder, session_dir=session_dir, now=NOW
    )
    assert edit["status"] == "applied" and edit["kind"] == "recut" and edit["k"] == 2
    assert len(edit["clusters_kept"]) == 1  # exactly one topic cluster survived intact
    kept = edit["clusters_kept"][0]

    for side in ("researcher", "literature"):
        revision = current_revision(session, side)
        partition = partition_for(session, side, revision)
        assert revision == 1 and partition["kind"] == "recut"
        for cluster in partition["clusters"]:
            expected = cluster["cluster_id"] == kept
            assert has_verdict(session, side, cluster["cluster_id"], revision) is expected, (
                side,
                cluster["cluster_id"],
            )
        changed = [c["cluster_id"] for c in partition["clusters"] if c["cluster_id"] != kept]
        # Changed clusters never reuse a retired id.
        assert all(c not in (0, 1, 2) for c in changed)


def test_recut_is_deterministic(tmp_path, anchored_embedder):
    from cda.validation.curate import apply_recut
    from cda.validation.session import partition_for

    _dir_a, session_a = hierarchical_joint_session(tmp_path / "a", anchored_embedder)
    _dir_b, session_b = hierarchical_joint_session(tmp_path / "b", anchored_embedder)
    judge_all_original_clusters(session_a)
    judge_all_original_clusters(session_b)
    apply_recut(session_a, 2, load_config(), embedder=anchored_embedder, now=NOW)
    apply_recut(session_b, 2, load_config(), embedder=anchored_embedder, now=NOW)
    for side in ("researcher", "literature"):
        assert (
            partition_for(session_a, side, 1)["assignments"]
            == partition_for(session_b, side, 1)["assignments"]
        )


class BrokenEmbedder:
    """Embedder whose second vector is unusable (a broken model, not a bad store)."""

    def __init__(self, bad_value):
        self.bad_value = bad_value

    def embed(self, texts):
        return [[self.bad_value] * 4 if i == 1 else [1.0, 0.0, 0.0, 0.0] for i in range(len(texts))]


@pytest.mark.parametrize("bad_value", [float("nan"), 0.0])
def test_rebuild_partition_clusters_refuses_unusable_embeddings_by_name(bad_value):
    """Silhouettes and the centroid-nearest 'shown' list go into the audit
    trail, so an unusable vector must be named rather than quietly averaged in."""
    passages = {
        chunk_id: {"text": chunk_id, "side": "literature", "source": "s.abstract"}
        for chunk_id in ("alpha::0", "alpha::1", "beta::0")
    }
    assignments = {"alpha::0": 0, "alpha::1": 0, "beta::0": 1}
    with pytest.raises(ValueError, match="alpha::1"):
        rebuild_partition_clusters(assignments, passages, BrokenEmbedder(bad_value), n_shown=2)
