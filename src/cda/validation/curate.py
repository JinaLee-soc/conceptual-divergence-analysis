"""Applying the researcher's curation decisions.

Curation edits partitions inside the validation session only. Untouched
clusters keep their ids and their verdicts (carried forward, provenance
kept); affected clusters start over without a verdict, so the inspection
queue naturally returns to them. Nothing here touches ``cda analyze`` or its
outputs: the curated partition lives and dies in the session directory as a
record of what the researcher judged, and is never re-analyzed (see
docs/validation_protocol.md).

Silhouettes are recomputed at apply time (embeddings rehydrated from the
stored passage texts with the same pinned embedder), so every edit's effect
on cluster quality is recorded in the audit trail.

Concurrency contract: both apply functions mutate a session the CALLER
loaded, so a caller whose session directory can be open elsewhere (two GUI
tabs, a tab plus a CLI) must hold ``session.session_lock(session_dir)``
across its whole load→apply→save cycle. The functions do not take the lock
themselves: it is not reentrant, and a lock taken here would start after the
caller's load — leaving exactly the stale-snapshot window it exists to close.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_samples

from cda.clustering.hierarchy import cut_tree_at_k, linkage_matrix
from cda.clustering.spherical import normalize_rows
from cda.config import Config
from cda.validation.metrics import blind_verdicts
from cda.validation.session import current_revision, partition_for, save_session


def _embedding_matrix(texts: list[str], embedder, names: list[str]) -> np.ndarray:
    """Re-embed stored passage texts, refusing unusable vectors by chunk id.

    ``names`` are the chunk ids of ``texts``, in the same order: a NaN or
    all-zero row here would silently corrupt every recomputed silhouette and
    the centroid-nearest "shown" list written into the audit trail, so it goes
    through the same guard the clustering path uses (spherical.normalize_rows).
    """
    X = np.asarray(embedder.embed(texts), dtype=float)
    return normalize_rows(X, names)


def _per_cluster_silhouettes(X: np.ndarray, labels: list[int]) -> dict[int, float | None]:
    """Per-cluster mean cosine silhouette (sklearn; a singleton's sample scores 0).

    None for every cluster when the silhouette is undefined (fewer than two
    clusters, or every passage its own cluster).
    """
    labels_arr = np.asarray(labels)
    unique = sorted(int(x) for x in set(labels_arr.tolist()))
    if not 2 <= len(unique) <= len(labels_arr) - 1:
        return {cluster: None for cluster in unique}
    scores = silhouette_samples(X, labels_arr, metric="cosine")
    return {cluster: round(float(scores[labels_arr == cluster].mean()), 4) for cluster in unique}


def require_session_embedder(session: dict, config: Config) -> None:
    """Refuse to re-embed with a model other than the one the session recorded.

    Recomputed silhouettes, recuts, and splits are only comparable with the
    session's original clustering when the passages are embedded the same way.
    """
    params = session["params"]
    recorded = params.get("embedding_model")
    if recorded is None:
        raise ValueError(
            "This validation session did not record its embedding model, so a "
            "curation edit cannot verify it would re-embed the passages the "
            "same way. Start a new session with: cda validate --term "
            f"{session['term']}"
        )
    configured = str(config.get("embedding.model"))
    if configured != recorded:
        raise ValueError(
            f"The session was clustered with embedding model {recorded!r}, but "
            f"the configuration now names {configured!r}. Set embedding.model "
            "back to the session's model before curating."
        )
    if "embedding_revision" in params:
        configured_revision = config.get("embedding.revision", None)
        if configured_revision != params["embedding_revision"]:
            raise ValueError(
                f"The session was clustered with embedding revision "
                f"{params['embedding_revision']!r}, but the configuration now names "
                f"{configured_revision!r}. Set embedding.revision back to the "
                "session's revision before curating."
            )


def require_original_verdicts(session: dict, sides: list[str]) -> None:
    """Refuse the first curation edit until every revision-0 cluster is judged.

    Agreement is computed over the original clustering only; a cluster that an
    edit retires before it was judged could never enter it.
    """
    missing = []
    for side in sides:
        judged = blind_verdicts(session, side, 0)
        clusters = partition_for(session, side, 0)["clusters"]
        unjudged = [c["cluster_id"] for c in clusters if c["cluster_id"] not in judged]
        if unjudged:
            missing.append(f"{side} cluster(s) {', '.join(str(c) for c in unjudged)}")
    if missing:
        raise ValueError(
            "Record a blind verdict on every cluster of the original clustering "
            "before curating; the agreement evidence covers only those clusters. "
            f"Still unjudged: {'; '.join(missing)}."
        )


def rebuild_partition_clusters(
    assignments: dict[str, int],
    passages: dict[str, dict],
    embedder,
    *,
    n_shown: int,
) -> tuple[list[dict], dict[int, float]]:
    """Cluster entries (size, silhouette, centroid-nearest shown ids) for a
    partition, recomputed from stored texts. No re-clustering happens here."""
    chunk_ids = sorted(assignments)
    X = _embedding_matrix(
        [passages[chunk_id]["text"] for chunk_id in chunk_ids], embedder, chunk_ids
    )
    labels = [assignments[chunk_id] for chunk_id in chunk_ids]
    silhouettes = _per_cluster_silhouettes(X, labels)

    clusters = []
    for cluster in sorted(set(labels)):
        member_indices = [i for i, label in enumerate(labels) if label == cluster]
        centroid = X[member_indices].mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroid = centroid / norm if norm > 0 else centroid
        ranked = sorted(member_indices, key=lambda i: (-float(X[i] @ centroid), chunk_ids[i]))
        clusters.append(
            {
                "cluster_id": cluster,
                "n_passages": len(member_indices),
                "silhouette": silhouettes[cluster],
                "shown": [chunk_ids[i] for i in ranked[:n_shown]],
            }
        )
    return clusters, silhouettes


def apply_recut(
    session: dict,
    k: int,
    config: Config,
    *,
    embedder,
    session_dir: Path | None = None,
    now: datetime,
    progress: Callable[[str], None] = print,
) -> dict:
    """Move the dendrogram cutoff: re-cut the session's tree into ``k`` clusters.

    Available only for sessions whose original clustering was joint AND
    hierarchical (``clustering.mode: joint`` + ``clustering.method:
    hierarchical``): there, revision 0 is by construction a cut of one
    deterministic average-linkage tree over the pooled passages, and this
    function produces another cut of the *same* tree — rebuilt from the
    stored passage texts with the pinned embedder, so no tree needs to be
    serialized. Clusters whose membership is unchanged keep their ids and
    their verdicts; every other cluster gets a fresh id and returns to the
    inspection queue. Both validated sides advance one revision together
    (the tree is shared, so a cut is inherently a joint operation).

    Free and local: no LLM call is involved.
    """
    params = session.get("params", {})
    if (
        params.get("clustering_method") != "hierarchical"
        or params.get("clustering_mode") != "joint"
    ):
        raise ValueError(
            "Moving the cutoff needs a session created from joint hierarchical "
            "clustering (clustering.mode: joint + clustering.method: hierarchical); "
            f"this session recorded mode={params.get('clustering_mode')!r}, "
            f"method={params.get('clustering_method')!r}. Re-run cda validate "
            "with those settings, or curate with merge/split instead."
        )
    if session.get("pending_edit"):
        raise ValueError(
            "Decide the pending merge/split suggestion first (apply or reject); "
            "a recut would silently invalidate the clusters it refers to."
        )
    k = int(k)
    chunk_ids = sorted(session["passages"])
    if not 2 <= k <= len(chunk_ids) - 1:
        raise ValueError(
            f"The cutoff must yield between 2 and {len(chunk_ids) - 1} clusters, got {k}."
        )
    sides = sorted(s for s, info in session["sides"].items() if info.get("validated"))
    require_original_verdicts(session, sides)
    require_session_embedder(session, config)

    X = _embedding_matrix([session["passages"][c]["text"] for c in chunk_ids], embedder, chunk_ids)
    labels = cut_tree_at_k(linkage_matrix(X), k)
    new_members: dict[int, set[str]] = {}
    for chunk_id, label in zip(chunk_ids, (int(x) for x in labels), strict=True):
        new_members.setdefault(label, set()).add(chunk_id)

    revision_of = {side: current_revision(session, side) for side in sides}

    # Old clusters, pooled across sides (ids are shared under joint mode).
    old_members: dict[int, set[str]] = {}
    for side in sides:
        for chunk_id, cluster in partition_for(session, side, revision_of[side])[
            "assignments"
        ].items():
            old_members.setdefault(int(cluster), set()).add(chunk_id)

    # Membership-identical clusters keep their ids (and, below, their
    # verdicts); every changed cluster gets a fresh id so a retired verdict
    # can never be misread as covering a different passage set.
    members_to_old_id = {frozenset(v): c for c, v in old_members.items()}
    all_ids = [
        int(c)
        for partition in session["partitions"]
        for c in set(partition["assignments"].values())
    ]
    next_id = max(all_ids, default=-1) + 1
    id_of: dict[int, int] = {}
    kept: list[int] = []
    for label in sorted(new_members):
        old_id = members_to_old_id.get(frozenset(new_members[label]))
        if old_id is not None:
            id_of[label] = old_id
            kept.append(old_id)
        else:
            id_of[label] = next_id
            next_id += 1
    pooled_assignments = {
        chunk_id: id_of[label] for label, members in new_members.items() for chunk_id in members
    }

    n_shown = config.get_int("validation.passages_shown_per_cluster", 8, minimum=1)
    edit = {
        "edit_id": len(session["edits"]) + 1,
        "kind": "recut",
        "side": "joint",
        "clusters": sorted(set(pooled_assignments.values())),
        "k": k,
        "clusters_kept": sorted(kept),
        "revisions": {},
        "silhouette_after": {},
        "status": "applied",
        "decided_at": now.isoformat(timespec="seconds"),
    }
    for side in sides:
        side_chunks = {
            chunk_id for chunk_id, meta in session["passages"].items() if meta.get("side") == side
        }
        assignments = {
            chunk_id: cluster
            for chunk_id, cluster in pooled_assignments.items()
            if chunk_id in side_chunks
        }
        if not assignments:
            continue
        clusters, silhouettes = rebuild_partition_clusters(
            assignments, session["passages"], embedder, n_shown=n_shown
        )
        new_revision = revision_of[side] + 1
        session["partitions"].append(
            {
                "revision": new_revision,
                "side": side,
                "kind": "recut",
                "derived_from": revision_of[side],
                "edit_id": edit["edit_id"],
                "affected_clusters": sorted(set(assignments.values()) - set(kept)),
                "assignments": assignments,
                "clusters": clusters,
            }
        )
        edit["revisions"][side] = new_revision
        edit["silhouette_after"][side] = {str(c): s for c, s in silhouettes.items()}

        # Verdicts survive only for clusters whose pooled membership is
        # unchanged (they kept their id by construction).
        latest: dict[int, dict] = {}
        for verdict in session["verdicts"]:
            if verdict["side"] == side and verdict["revision"] == revision_of[side]:
                latest[verdict["cluster_id"]] = verdict
        for cluster_id in sorted(set(assignments.values()) & set(kept)):
            if cluster_id in latest:
                carried = dict(latest[cluster_id])
                carried["revision"] = new_revision
                carried["carried_from_revision"] = revision_of[side]
                session["verdicts"].append(carried)

    session["edits"].append(edit)
    progress(
        f"[validate] recut at k={k}: {len(kept)} cluster(s) unchanged (verdicts "
        f"kept), {k - len(kept)} new cluster(s) need verdicts"
    )
    if session_dir:
        save_session(session_dir, session)
    return edit


def apply_pending_edit(
    session: dict,
    decision: str,
    config: Config,
    *,
    embedder,
    session_dir: Path | None = None,
    now: datetime,
    progress: Callable[[str], None] = print,
) -> dict:
    """Act on the pending suggestion: reject, or apply one of its proposals.

    Decisions: ``reject``; ``apply-merge`` for merges; ``apply-semantic`` or
    ``apply-mechanical`` for splits. Applying writes partition revision r+1,
    carries untouched clusters' verdicts forward, and leaves the affected
    clusters verdict-less (back into the inspection queue).
    """
    edit = session.get("pending_edit")
    if not edit:
        raise ValueError("No pending suggestion to decide; run a merge or split suggestion first.")
    valid = (
        ("reject", "apply-merge")
        if edit["kind"] == "merge"
        else (
            "reject",
            "apply-semantic",
            "apply-mechanical",
        )
    )
    if decision not in valid:
        raise ValueError(
            f"decision must be one of {', '.join(valid)} for a {edit['kind']}, got {decision!r}"
        )

    # Validate an applyable split proposal BEFORE the pending edit is cleared:
    # a refusal must leave the suggestion in place so the researcher can still
    # reject it (or apply the other proposal) — not lose it from the trail.
    letter_of = None
    if edit["kind"] == "split" and decision != "reject":
        proposal = edit["split_proposals"].get(
            "semantic" if decision == "apply-semantic" else "mechanical"
        )
        if decision == "apply-semantic":
            if proposal is None:
                raise ValueError(
                    "The semantic split proposal was unusable (the agent's reply "
                    "did not parse); apply the mechanical split or reject."
                )
            letter_of = proposal["assignments"]
        else:
            letter_of = proposal
        if set(letter_of.values()) != {"A", "B"}:
            raise ValueError(
                "This split proposal puts every passage on one side, so applying "
                "it would change nothing while still retiring the cluster's "
                "verdict and bumping the revision. Reject it instead."
            )

    if decision != "reject":
        require_original_verdicts(session, [edit["side"]])
        require_session_embedder(session, config)

    session["pending_edit"] = None
    edit["decided_at"] = now.isoformat(timespec="seconds")
    if decision == "reject":
        edit["status"] = "rejected"
        session["edits"].append(edit)
        if session_dir:
            save_session(session_dir, session)
        return edit

    side, revision = edit["side"], edit["revision"]
    old = partition_for(session, side, revision)
    assignments = dict(old["assignments"])
    old_silhouettes = {c["cluster_id"]: c.get("silhouette") for c in old["clusters"]}
    # The revision-0 partition stores no per-cluster silhouettes (they first
    # appear when an edit recomputes them), so on the first edit the stored
    # values are all None and "silhouette before" would record nothing.
    # Recompute them from the same embeddings the after-values will use, so
    # the audit trail can always show the edit's effect.
    if any(old_silhouettes.get(c) is None for c in edit["clusters"]):
        chunk_ids = sorted(assignments)
        X = _embedding_matrix(
            [session["passages"][c]["text"] for c in chunk_ids], embedder, chunk_ids
        )
        labels = [assignments[chunk_id] for chunk_id in chunk_ids]
        for cluster, value in _per_cluster_silhouettes(X, labels).items():
            if old_silhouettes.get(cluster) is None:
                old_silhouettes[cluster] = value

    if edit["kind"] == "merge":
        keep, absorb = sorted(edit["clusters"])
        for chunk_id, cluster in assignments.items():
            if cluster == absorb:
                assignments[chunk_id] = keep
        affected = [keep]
        edit["applied_proposal"] = "merge"
        edit["silhouette_before"] = {str(c): old_silhouettes.get(c) for c in edit["clusters"]}
    else:
        split_cluster = edit["clusters"][0]
        # The fresh id must be new across EVERY partition ever stored in the
        # session, exactly like apply_recut's scan: under joint mode the two
        # sides share cluster ids, so max() over one side's current
        # assignments could mint an id already meaning a different region on
        # the other side — or resurrect an id an earlier merge retired,
        # letting a stale verdict read as covering the new passage set.
        all_ids = (
            int(c) for partition in session["partitions"] for c in partition["assignments"].values()
        )
        new_cluster = max(all_ids, default=-1) + 1
        for chunk_id, letter in letter_of.items():
            assignments[chunk_id] = split_cluster if letter == "A" else new_cluster
        affected = [split_cluster, new_cluster]
        edit["applied_proposal"] = decision.removeprefix("apply-")
        edit["silhouette_before"] = {str(split_cluster): old_silhouettes.get(split_cluster)}

    n_shown = config.get_int("validation.passages_shown_per_cluster", 8, minimum=1)
    clusters, silhouettes = rebuild_partition_clusters(
        assignments, session["passages"], embedder, n_shown=n_shown
    )
    new_revision = revision + 1
    session["partitions"].append(
        {
            "revision": new_revision,
            "side": side,
            "kind": "curated",
            "derived_from": revision,
            "edit_id": edit["edit_id"],
            "affected_clusters": affected,
            "assignments": assignments,
            "clusters": clusters,
        }
    )
    edit["silhouette_after"] = {str(c): silhouettes.get(c) for c in affected}
    edit["status"] = "applied"
    session["edits"].append(edit)

    # Untouched clusters keep their verdicts: carry the latest verdict per
    # cluster forward with its provenance, so only the affected clusters
    # return to the inspection queue.
    latest: dict[int, dict] = {}
    for verdict in session["verdicts"]:
        if verdict["side"] == side and verdict["revision"] == revision:
            latest[verdict["cluster_id"]] = verdict
    surviving = {c["cluster_id"] for c in clusters} - set(affected)
    for cluster_id in sorted(surviving):
        if cluster_id in latest:
            carried = dict(latest[cluster_id])
            carried["revision"] = new_revision
            carried["carried_from_revision"] = revision
            session["verdicts"].append(carried)

    progress(
        f"[validate] applied {edit['kind']} on {side}: revision {new_revision}, "
        f"clusters {affected} need fresh verdicts"
    )
    if session_dir:
        save_session(session_dir, session)
    return edit
