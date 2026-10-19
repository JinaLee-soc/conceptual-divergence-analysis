"""Joint clustering of researcher and literature passages (the default mode).

Joint mode uses one shared partition: pool the researcher's
term-sentences with the selected literature term-sentences, run the identical
seeded spherical k-means over the pool, and then read off *where the
researcher's passages sit* inside each shared cluster.

Two properties follow, and they are the reason this is the default:

1. **Cluster ids mean the same thing on both sides.** "Cluster 3" is one
   region of embedding space containing some researcher passages and some
   literature passages, so a divergence is a membership fact, not a
   centroid-matching inference.
2. **The paper's >80% / <10% flags become literal.** §3.3 speaks of "the
   researcher contributing >80% of instances in a cluster" — a statement that
   only type-checks when the two corpora share a partition. Under joint
   clustering the flags are direct membership counts; no cross-assignment
   step stands between the passages and the number.

What is unchanged from independent clustering: the k-means itself (same `select_k_by_silhouette`, same
seeds, same small-cluster merging), the saturation subsampling that bounds the
literature pool, and the appendix's prompt format — the two cluster-structure
blocks are still emitted, now as two views of one partition (``corpus_views``).

Independent clustering remains available (``clustering.mode: independent`` /
``cda analyze --independent``) and reproduces the published pipeline exactly.
"""

from __future__ import annotations

import numpy as np

from cda.clustering.spherical import (
    centroid_order,
    clip_k,
    hydrate_embeddings,
    prepare_matrix,
    select_k_by_silhouette,
    unit_centroid,
)
from cda.corpus.embeddings import Embedder
from cda.types import ClusterSummary, JointClustering, Passage, passage_theme_tags

RESEARCHER = "researcher"
LITERATURE = "literature"


def _pool(
    researcher_passages: list[Passage], literature_passages: list[Passage]
) -> tuple[list[Passage], list[str]]:
    """Researcher passages first, then literature, with a parallel role list.

    Both pools arrive already sorted by the pipeline's total-order key, so
    concatenating them is itself deterministic — which is what keeps the
    joint partition (and therefore the assembled prompt) reproducible.
    """
    pooled = list(researcher_passages) + list(literature_passages)
    roles = [RESEARCHER] * len(researcher_passages) + [LITERATURE] * len(literature_passages)
    return pooled, roles


def _empty_result(
    researcher_passages: list[Passage], literature_passages: list[Passage], note: str
) -> JointClustering:
    """Degenerate outcome: every passage lands in one cluster, honestly labeled."""
    for passage in researcher_passages + literature_passages:
        passage.cluster_id = "Cluster 0"
    return JointClustering(
        used=False,
        n_clusters=1,
        silhouette=0.0,
        note=note,
        n_researcher=len(researcher_passages),
        n_literature=len(literature_passages),
    )


def _cluster_entry(
    *,
    cluster: int,
    X: np.ndarray,
    member_indices: list[int],
    roles: list[str],
    passages: list[Passage],
    n_literature_total: int,
    n_representatives: int,
) -> tuple[dict, dict]:
    """One shared cluster: composition, marked representatives, distance stats.

    Representatives are the centroid-nearest members (the appendix's
    selection), kept in one ranking across both corpora and *marked* with the
    corpus they came from — so the report and the prompt show at a glance
    whether the researcher sits at a cluster's core or on its rim.
    """
    member_X = X[member_indices]
    centroid = unit_centroid(member_X)
    distances = 1.0 - member_X @ centroid
    ranking = centroid_order(member_X, centroid).tolist()

    member_roles = [roles[i] for i in member_indices]
    n_researcher = member_roles.count(RESEARCHER)
    n_members = len(member_indices)

    def take(candidate_ranks: list[int]) -> dict:
        chosen = candidate_ranks[:n_representatives]
        return {
            "representatives": [passages[member_indices[i]].text for i in chosen],
            "sources": [passages[member_indices[i]].source for i in chosen],
            "rep_theme_tags": [passage_theme_tags(passages[member_indices[i]]) for i in chosen],
            "rep_roles": [f"top{rank + 1}" for rank in range(len(chosen))],
            "rep_corpora": [member_roles[i] for i in chosen],
        }

    entry = {
        "cluster_id": cluster,
        "n_passages": n_members,
        "n_researcher": n_researcher,
        "n_literature": n_members - n_researcher,
        # Display values only: the flags recompute both shares exactly from
        # the counts. researcher_share is the paper's >80% axis (share of this
        # cluster); literature_share the <10% axis (share of all literature).
        "researcher_share": round(n_researcher / n_members, 4),
        "literature_share": (
            round((n_members - n_researcher) / n_literature_total, 4)
            if n_literature_total
            else None
        ),
        **take(ranking),
        RESEARCHER: take([i for i in ranking if member_roles[i] == RESEARCHER]),
        LITERATURE: take([i for i in ranking if member_roles[i] == LITERATURE]),
    }

    stats = {
        "cluster_id": cluster,
        "n": n_members,
        "n_researcher": n_researcher,
        "n_literature": n_members - n_researcher,
        "mean_intra_dist": round(float(np.mean(distances)), 4),
    }
    return entry, stats


def cluster_joint(
    researcher_passages: list[Passage],
    literature_passages: list[Passage],
    embedder: Embedder,
    *,
    max_k: int,
    n_representatives: int,
    n_restarts: int = 5,
    max_iter: int = 80,
    min_cluster_size: int = 3,
    silhouette_subsample: int = 2000,
    seed: int = 42,
    method: str = "kmeans",
    fixed_k: int | None = None,
) -> JointClustering:
    """Cluster both corpora together and record where the researcher sits.

    Attaches shared cluster ids to every passage in both pools (the same
    ``Cluster N`` labels the independent path uses, so everything downstream
    — validation, the GUI, the flags — reads them unchanged) and returns the
    per-cluster composition.
    """
    pooled, roles = _pool(researcher_passages, literature_passages)
    if len(pooled) < 2:
        return _empty_result(
            researcher_passages, literature_passages, "[joint] Too few passages for clustering"
        )

    hydrate_embeddings(pooled, embedder)
    # len(pooled) >= 2 and hydration embeds every passage or raises, so X is
    # the full pooled matrix and rows align with ``roles``.
    X, with_embeddings = prepare_matrix(pooled)
    embedded_roles = roles

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
        # Same resolution overrides as cluster_passages; the default path
        # stays on the paper's silhouette-selected k-means above.
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
        return _empty_result(
            researcher_passages, literature_passages, "[joint] No valid k-means partition found"
        )

    for i, passage in enumerate(with_embeddings):
        passage.cluster_id = f"Cluster {int(labels[i])}"

    n_literature_total = embedded_roles.count(LITERATURE)
    entries: list[dict] = []
    per_cluster_stats: list[dict] = []
    for cluster in sorted(set(int(x) for x in labels.tolist())):
        member_indices = np.flatnonzero(labels == cluster).tolist()
        entry, stats = _cluster_entry(
            cluster=cluster,
            X=X,
            member_indices=member_indices,
            roles=embedded_roles,
            passages=with_embeddings,
            n_literature_total=n_literature_total,
            n_representatives=n_representatives,
        )
        entries.append(entry)
        per_cluster_stats.append(stats)

    note = "[joint] Embedding-level cluster separation detected across the pooled passages"
    diagnostics = {
        "mode": "joint",
        "n_pooled": len(pooled),
        "n_with_embeddings": int(X.shape[0]),
        "n_researcher_embedded": embedded_roles.count(RESEARCHER),
        "n_literature_embedded": n_literature_total,
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
    return JointClustering(
        used=True,
        n_clusters=best_k,
        silhouette=best_score,
        note=note,
        n_researcher=len(researcher_passages),
        n_literature=len(literature_passages),
        clusters=entries,
        diagnostics=diagnostics,
    )


def _view(joint: JointClustering, corpus: str) -> ClusterSummary:
    """One corpus's slice of the joint partition, in the appendix's shape.

    Clusters with no passages from this corpus are omitted from its block —
    their absence is exactly the finding the composition table reports, and
    printing an empty cluster would only invite the model to invent content
    for it.
    """
    n_total = joint.n_researcher if corpus == RESEARCHER else joint.n_literature
    if not joint.used or not joint.clusters:
        return ClusterSummary(
            used=False,
            n_total_passages=n_total,
            n_clusters=1,
            silhouette=0.0,
            note=joint.note,
        )

    count_key = "n_researcher" if corpus == RESEARCHER else "n_literature"
    # Only clusters this corpus actually occupies (see the docstring).
    clusters = [
        {
            "cluster_id": entry["cluster_id"],
            "n_passages": entry[count_key],
            **{
                key: entry[corpus][key]
                for key in (
                    "representatives",
                    "sources",
                    "rep_roles",
                    "rep_corpora",
                    "rep_theme_tags",
                )
            },
        }
        for entry in joint.clusters
        if entry[count_key] > 0
    ]
    occupied = len(clusters)
    if not occupied:
        # This corpus contributed nothing to the pool (no drafts and no
        # mywork-tagged chunks, say), so there is no view of it to report.
        return ClusterSummary(
            used=False,
            n_total_passages=n_total,
            n_clusters=1,
            silhouette=0.0,
            note=f"[{corpus}] no passages entered the joint clustering",
        )
    return ClusterSummary(
        used=True,
        n_total_passages=n_total,
        n_clusters=occupied,
        # The silhouette belongs to the joint partition, not to this slice of
        # it; the note says so, so the number is never read as a per-corpus
        # separation score.
        silhouette=joint.silhouette,
        note=(
            f"[{corpus}] view of one JOINT partition of researcher + literature "
            f"passages (joint k={joint.n_clusters}, silhouette computed on the "
            f"pooled passages); {occupied} of {joint.n_clusters} shared clusters "
            f"contain {corpus} passages"
        ),
        clusters=clusters,
        diagnostics={"mode": "joint", "joint_k": joint.n_clusters, "occupied_clusters": occupied},
    )


def corpus_views(joint: JointClustering) -> tuple[ClusterSummary, ClusterSummary]:
    """(researcher_summary, literature_summary) as views of the joint partition.

    Lets the whole downstream pipeline — prompt assembly, the audit, the
    report, the GUI, ``cda validate`` — keep consuming two ``ClusterSummary``
    objects, while the ids inside them refer to shared clusters.
    """
    return _view(joint, RESEARCHER), _view(joint, LITERATURE)


def composition_note(joint: JointClustering) -> str:
    """Compact per-cluster ``researcher/literature`` counts for the prompt."""
    parts = [
        f"C{entry['cluster_id']} {entry['n_researcher']}/{entry['n_literature']}"
        for entry in joint.clusters
    ]
    return (
        "Joint cluster composition (researcher/literature passages per shared "
        "cluster): " + ", ".join(parts)
    )
