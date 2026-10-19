"""Saturation-based subsampling of the literature corpus (paper §3.3).

Common theoretical terms can retrieve thousands of literature passages —
too many to cluster and summarize usefully. Candidates are ranked by cosine
similarity to the target term's embedding (an external anchor, deliberately
independent of both researcher and literature registers, so ranking cannot
smooth away the divergences CDA is meant to measure). Batches are added
incrementally until the cluster structure stabilizes: adjusted Rand index
between consecutive increments (on the overlapping prefix) at or above the
threshold for the configured number of consecutive increments. When the cap
is reached without saturation, the report flags it.
"""

from __future__ import annotations

import numpy as np

from cda.clustering.spherical import (
    adjusted_rand_index,
    hydrate_embeddings,
    normalize_rows,
    prepare_matrix,
    select_k_by_silhouette,
)
from cda.corpus.embeddings import Embedder
from cda.types import Passage


def rank_literature(
    passages: list[Passage],
    term: str,
    embedder: Embedder,
    *,
    mode: str = "term",
) -> list[Passage]:
    """Rank passages for the incremental sweep.

    "term" (default, the paper's method): cosine similarity to the target
    term's embedding. "similarity": reuse the retrieval similarity scores,
    which every passage must carry. Ties keep the input order.
    """
    if mode not in ("term", "similarity"):
        raise ValueError(f"rank_literature mode must be 'term' or 'similarity', got {mode!r}.")
    if not passages:
        return []

    hydrate_embeddings(passages, embedder)
    X, _ = prepare_matrix(passages)
    if X is None:
        return list(passages)
    if mode == "similarity":
        unscored = [p.chunk_id for p in passages if p.similarity is None]
        if unscored:
            raise ValueError(
                f"{len(unscored)} passages carry no retrieval similarity "
                f"(e.g. {', '.join(unscored[:5])}); rank them with mode='term'."
            )
        scores = [float(p.similarity) for p in passages]
    else:
        vectors = embedder.embed([term])
        if len(vectors) != 1:
            raise ValueError("Term anchor requires exactly one embedding; check the provider.")
        anchor = np.asarray(vectors, dtype=float)
        if anchor.shape != (1, X.shape[1]):
            raise ValueError("Term anchor and passage embeddings must have identical dimensions.")
        anchor = normalize_rows(anchor, [term], what="term embeddings")[0]
        scores = (X @ anchor).tolist()
    scored = sorted(zip(scores, passages, strict=True), key=lambda pair: pair[0], reverse=True)
    return [p for _, p in scored]


def saturation_select(
    ranked_passages: list[Passage],
    *,
    start: int,
    step: int,
    max_passages: int,
    ari_threshold: float,
    consecutive_increments: int,
    max_k: int,
    n_restarts: int,
    max_iter: int,
    min_cluster_size: int,
    silhouette_subsample: int,
    seed: int,
) -> tuple[list[Passage], dict]:
    """Grow the corpus until cluster structure stabilizes; select that N.

    Saturation is declared at the EARLIER of the consecutive high-ARI steps —
    the smallest N at which structure had already stabilized (conservative).
    Returns (selected_passages, diagnostics).
    """
    if consecutive_increments < 1:
        raise ValueError(
            f"saturation needs consecutive_increments >= 1, got {consecutive_increments}."
        )
    n_candidates = len(ranked_passages)
    diagnostics: dict = {
        "enabled": True,
        "n_candidates": n_candidates,
        "start": start,
        "step": step,
        "max_passages": max_passages,
        "ari_threshold": ari_threshold,
        "consecutive_increments": consecutive_increments,
        "history": [],
        "saturation_reached": False,
        "n_selected": n_candidates,
    }

    if n_candidates < start:
        diagnostics["reason"] = (
            f"fewer candidates ({n_candidates}) than start ({start}); no selection applied"
        )
        return ranked_passages, diagnostics

    sweep_sizes = list(range(start, min(max_passages, n_candidates) + 1, step))
    if not sweep_sizes:
        # Config validation (saturation_params) rejects max_passages < start
        # before a pipeline run; this guards direct callers. Falling through
        # would return the ENTIRE candidate pool while the report still
        # printed the configured cap.
        raise ValueError(
            f"saturation sweep is empty: start={start}, step={step}, "
            f"max_passages={max_passages}, candidates={n_candidates}. "
            "Raise saturation.max_passages above saturation.start."
        )
    if sweep_sizes[-1] < min(max_passages, n_candidates):
        sweep_sizes.append(min(max_passages, n_candidates))

    previous_labels = None
    previous_n = 0
    high_ari_streak = 0
    n_selected = sweep_sizes[-1]

    for index, n in enumerate(sweep_sizes):
        subset = ranked_passages[:n]
        X, _with_embeddings = prepare_matrix(subset)
        if X is None:
            diagnostics["history"].append(
                {"n": n, "status": "skipped", "reason": "matrix prep failed"}
            )
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
            "n_with_embeddings": int(X.shape[0]),
            "k": best_k,
            "silhouette": round(float(sil), 4) if sil is not None else None,
        }

        if labels is None:
            record["status"] = "no_partition"
            diagnostics["history"].append(record)
            previous_labels = None
            previous_n = 0
            high_ari_streak = 0
            continue

        # ARI vs the previous increment, restricted to the overlapping prefix
        # (the first previous_n passages appear in both clusterings).
        if previous_labels is not None and 0 < previous_n <= X.shape[0]:
            ari = adjusted_rand_index(previous_labels[:previous_n], labels[:previous_n])
            record["ari_vs_prev"] = round(float(ari), 4)
            high_ari_streak = high_ari_streak + 1 if ari >= ari_threshold else 0
        else:
            record["ari_vs_prev"] = None
            high_ari_streak = 0

        diagnostics["history"].append(record)

        if high_ari_streak >= consecutive_increments:
            plateau_index = index - (consecutive_increments - 1)
            n_selected = sweep_sizes[plateau_index]
            diagnostics["saturation_reached"] = True
            diagnostics["plateau_step_index"] = plateau_index
            break

        previous_labels = labels
        previous_n = int(X.shape[0])

    diagnostics["n_selected"] = n_selected
    if not diagnostics["saturation_reached"]:
        diagnostics["reason"] = "ARI threshold not met within the cap; using final N"
    return ranked_passages[:n_selected], diagnostics
