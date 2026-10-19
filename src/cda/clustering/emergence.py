"""Cluster emergence over library-acquisition time (EXPERIMENTAL, post-paper).

``cda emergence`` asks a longitudinal question the paper's pipeline does not:
as the researcher added works to her personal library, when did each usage
cluster for a term coalesce into distinct structure? The timeline is Zotero's
"Date Added" — the researcher's *reading trajectory* — NOT the field's
publication history (publication year is a different axis and is reported
elsewhere). Literature passages are ordered by acquisition date, cumulative
snapshots are clustered with the same seeded spherical k-means the paper
uses, and each snapshot's clusters are matched to the final clustering by
centroid cosine similarity, yielding a birth point and growth curve per
final cluster.

Passages without a ``date_added`` cannot be placed on the timeline and are
excluded (bucketing them anywhere would fabricate history); the coverage
rate is reported prominently instead.
"""

from __future__ import annotations

import numpy as np

from cda.clustering.spherical import (
    adjusted_rand_index,
    centroid_order,
    hydrate_embeddings,
    prepare_matrix,
    select_k_by_silhouette,
    unit_centroid,
)
from cda.corpus.embeddings import Embedder
from cda.corpus.zotero import normalize_date_added
from cda.types import Passage


def passage_date_added(passage: Passage, date_key: str = "date_added") -> str | None:
    """Normalized acquisition datetime from chunk metadata, or None.

    ``date_key`` is the metadata field holding the acquisition timestamp —
    ``date_added`` (Zotero) or ``date_added_fs`` (filesystem). The timeline
    must read the same clock the window filter used, or one report would
    silently mix two histories.
    """
    raw = (passage.metadata or {}).get(date_key, "")
    return normalize_date_added(str(raw)) or None


def _cluster_ids_and_centroids(
    X: np.ndarray, labels: np.ndarray
) -> tuple[list[int], np.ndarray, list[int]]:
    """Renumbered cluster order (sorted original labels), unit centroids, sizes."""
    unique = sorted(set(int(x) for x in labels.tolist()))
    centroids = np.asarray([unit_centroid(X[labels == cluster]) for cluster in unique])
    sizes = [int(np.sum(labels == cluster)) for cluster in unique]
    return unique, centroids, sizes


def _match_to_final(
    snap_centroids: np.ndarray, final_centroids: np.ndarray, threshold: float
) -> tuple[list[dict], list[int]]:
    """Greedy one-to-one centroid matching, highest cosine first.

    One-to-one matching handles k drift naturally: with fewer snapshot
    clusters some final clusters stay unmatched (not yet born); with more,
    the surplus is reported as transient structure. Tie-breaks are total,
    so matching is deterministic.
    """
    pairs = []
    for i in range(snap_centroids.shape[0]):
        for j in range(final_centroids.shape[0]):
            pairs.append((float(snap_centroids[i] @ final_centroids[j]), i, j))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))

    snap_taken: set[int] = set()
    final_taken: set[int] = set()
    matches: list[dict] = []
    for cosine, i, j in pairs:
        if cosine < threshold:
            break
        if i in snap_taken or j in final_taken:
            continue
        snap_taken.add(i)
        final_taken.add(j)
        matches.append({"snapshot_cluster": i, "final_cluster": j, "cosine": round(cosine, 4)})
    unmatched = [i for i in range(snap_centroids.shape[0]) if i not in snap_taken]
    return matches, unmatched


def track_emergence(
    passages: list[Passage],
    embedder: Embedder,
    *,
    start: int,
    step: int,
    match_threshold: float,
    max_k: int,
    n_restarts: int,
    max_iter: int,
    min_cluster_size: int,
    silhouette_subsample: int,
    seed: int,
    date_key: str = "date_added",
    n_representatives: int = 3,
) -> dict:
    """Cluster cumulative acquisition-ordered snapshots; match to the final result.

    Snapshots are count-based (``start``, ``start+step``, …, all dated
    passages), not calendar-based: personal-library acquisition is bursty, so
    calendar buckets yield degenerate snapshots, while count increments give
    every snapshot equal statistical mass and reuse the saturation sweep's
    prefix-ARI machinery. Calendar time survives as each snapshot's ``as_of``
    date (the acquisition date of its newest passage). ``date_key`` names the
    clock — the same one the caller's window filter used. Each final cluster
    lists its ``n_representatives`` centroid-nearest passages.
    """
    if start < 1:
        raise ValueError(
            f"emergence.start must be at least 1, got {start}. It is the number of "
            "passages in the first snapshot, so 0 or a negative value has no timeline "
            "meaning."
        )
    if step < 1:
        raise ValueError(
            f"emergence.step must be at least 1, got {step}. It is how many passages "
            "each snapshot adds."
        )
    with_dates = [(passage_date_added(p, date_key), p) for p in passages]
    n_undated = sum(1 for d, _ in with_dates if d is None)
    # Total order: acquisition datetime, then chunk_id — the pipeline's
    # tiebreaker convention, so results never depend on DB return order.
    dated = sorted(
        ((d, p) for d, p in with_dates if d is not None),
        key=lambda pair: (pair[0], pair[1].chunk_id),
    )
    n_total = len(passages)
    n_dated = len(dated)

    result: dict = {
        "status": "ok",
        "params": {
            "start": start,
            "step": step,
            "match_threshold": match_threshold,
            "max_k": max_k,
            "seed": seed,
            "date_key": date_key,
        },
        "coverage": {
            "n_literature": n_total,
            "n_dated": n_dated,
            "n_undated": n_undated,
            "rate": round(n_dated / n_total, 4) if n_total else 0.0,
        },
        "snapshots": [],
        "final": None,
    }

    if n_dated < start:
        fix = (
            "Filesystem dates are recorded when PDFs are ingested from disk, so "
            "re-ingest your PDFs (cda ingest --pdf-dir ... --reingest) or use "
            "--date-source zotero"
            if date_key == "date_added_fs"
            else "Re-ingest with your Zotero CSV (cda ingest --pdf-dir ... "
            "--zotero-csv ... --reingest)"
        )
        result["status"] = "skipped"
        result["reason"] = (
            f"only {n_dated} of {n_total} literature passages carry a '{date_key}' "
            f"timestamp, fewer than the first snapshot size (emergence.start = {start}). "
            f"{fix}, or lower emergence.start."
        )
        return result

    ordered = [p for _, p in dated]
    dates = [d for d, _ in dated]
    hydrate_embeddings(ordered, embedder)

    sweep_sizes = list(range(start, n_dated + 1, step))
    if sweep_sizes[-1] != n_dated:
        sweep_sizes.append(n_dated)

    # Raw per-snapshot state kept for the matching pass; index-aligned with
    # result["snapshots"].
    raw: list[tuple[np.ndarray | None, np.ndarray | None]] = []
    previous_labels = None
    previous_n = 0

    for n in sweep_sizes:
        as_of = dates[n - 1][:10]
        X, _with_embeddings = prepare_matrix(ordered[:n])
        if X is None:
            result["snapshots"].append(
                {"n": n, "as_of": as_of, "status": "skipped", "reason": "matrix prep failed"}
            )
            raw.append((None, None))
            continue

        labels, best_k, sil, _sweep = select_k_by_silhouette(
            X,
            max_k=max_k,
            n_restarts=n_restarts,
            max_iter=max_iter,
            min_cluster_size=min_cluster_size,
            silhouette_subsample=silhouette_subsample,
            seed=seed,
        )
        record: dict = {
            "n": n,
            "as_of": as_of,
            "n_with_embeddings": int(X.shape[0]),
            "k": best_k,
            "silhouette": round(float(sil), 4) if sil is not None else None,
        }
        if labels is None:
            record["status"] = "no_partition"
            result["snapshots"].append(record)
            raw.append((None, None))
            previous_labels = None
            previous_n = 0
            continue

        record["status"] = "ok"
        # ARI vs the previous snapshot on the overlapping prefix — the
        # saturation sweep's structural-stability measure.
        if previous_labels is not None and 0 < previous_n <= X.shape[0]:
            ari = adjusted_rand_index(previous_labels[:previous_n], labels[:previous_n])
            record["ari_vs_prev"] = round(float(ari), 4)
        else:
            record["ari_vs_prev"] = None
        result["snapshots"].append(record)
        raw.append((X, labels))
        previous_labels = labels
        previous_n = int(X.shape[0])

    final_X, final_labels = raw[-1]
    if final_labels is None:
        result["status"] = "degraded"
        result["reason"] = "the final clustering (all dated passages) found no valid partition"
        return result

    final_unique, final_centroids, final_sizes = _cluster_ids_and_centroids(final_X, final_labels)
    final_record = result["snapshots"][-1]

    # Match every earlier successful snapshot to the final clustering.
    for index in range(len(raw) - 1):
        X, labels = raw[index]
        if labels is None:
            continue
        _unique, centroids, sizes = _cluster_ids_and_centroids(X, labels)
        matches, unmatched = _match_to_final(centroids, final_centroids, match_threshold)
        result["snapshots"][index]["matches"] = matches
        result["snapshots"][index]["unmatched_snapshot_clusters"] = unmatched
        result["snapshots"][index]["_sizes"] = sizes  # stripped below

    # Per final cluster: birth, growth, persistence, representatives.
    clusters: list[dict] = []
    for j, original_label in enumerate(final_unique):
        growth: list[dict] = []
        birth_n = None
        birth_date = None
        matched_cosines: list[float] = []
        n_post_birth = 0
        for index in range(len(raw) - 1):
            record = result["snapshots"][index]
            if record.get("status") != "ok":
                continue
            match = next((m for m in record.get("matches", []) if m["final_cluster"] == j), None)
            size = record["_sizes"][match["snapshot_cluster"]] if match else 0
            growth.append({"n": record["n"], "as_of": record["as_of"], "size": size})
            if match:
                if birth_n is None:
                    birth_n = record["n"]
                    birth_date = record["as_of"]
                matched_cosines.append(match["cosine"])
            if birth_n is not None:
                n_post_birth += 1
        # Every final cluster exists by the final snapshot at the latest.
        if birth_n is None:
            birth_n = final_record["n"]
            birth_date = final_record["as_of"]
        growth.append(
            {"n": final_record["n"], "as_of": final_record["as_of"], "size": final_sizes[j]}
        )

        member_positions = np.flatnonzero(final_labels == original_label)
        order = centroid_order(final_X[member_positions], final_centroids[j])
        nearest = [ordered[int(member_positions[r])] for r in order[:n_representatives]]
        representatives = [p.text for p in nearest]
        rep_sources = [p.source for p in nearest]

        clusters.append(
            {
                "final_cluster": j,
                "n_passages": final_sizes[j],
                "birth_n": birth_n,
                "birth_date": birth_date,
                "persistence": (
                    round(len(matched_cosines) / n_post_birth, 4) if n_post_birth else None
                ),
                "mean_match_cosine": (
                    round(sum(matched_cosines) / len(matched_cosines), 4)
                    if matched_cosines
                    else None
                ),
                "growth": growth,
                "representatives": representatives,
                "sources": rep_sources,
            }
        )

    for record in result["snapshots"]:
        record.pop("_sizes", None)

    result["final"] = {
        "n": final_record["n"],
        "k": len(final_unique),
        "silhouette": final_record["silhouette"],
        "clusters": clusters,
    }
    return result
