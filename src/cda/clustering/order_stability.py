"""Order-sensitivity of the clustering step.

k-means++ seeding draws initial centroids by array position, so the
canonical passage order makes a run reproducible but not necessarily
robust: a weakly separated term can land on a different k under another
order. This module re-clusters the identical passage set under N seeded
shuffles (trial 0 is the canonical order) and records how k and silhouette
vary. Every knob is passed in, so it is shared without circular imports by
the pipeline's built-in check (``clustering.stability_check``) and
``benchmarks/clustering_order_stability.py``."""

from __future__ import annotations

import copy
import random

from cda.clustering.joint import cluster_joint
from cda.clustering.spherical import cluster_passages
from cda.corpus.embeddings import Embedder
from cda.types import Passage

JOINT = "joint"
INDEPENDENT = "independent"


def detached(passages: list[Passage]) -> list[Passage]:
    """Per-passage copies, so a trial cannot write back onto the caller's list.

    ``cluster_passages`` and ``cluster_joint`` assign ``passage.cluster_id`` in
    place. Handing them the pipeline's own Passage objects would leave the live
    lists carrying the last trial's labels, and independent-mode
    ``compute_flags`` reads that field — the stability check would silently
    decide the divergence flags. The copy is shallow per passage: ``embedding``
    and ``metadata`` are shared, which is safe because clustering only writes
    ``cluster_id``.
    """
    return [copy.copy(passage) for passage in passages]


def shuffled(passages: list[Passage], seed: int) -> list[Passage]:
    """Detached copies of ``passages`` in a fresh, independent order."""
    reordered = detached(passages)
    random.Random(seed).shuffle(reordered)
    return reordered


def run_trials(
    researcher: list[Passage],
    literature: list[Passage],
    embedder: Embedder,
    *,
    mode: str,
    trials: int,
    seed_base: int,
    n_representatives: int,
    clustering_kwargs: dict,
    joint_max_k: int,
    researcher_max_k: int,
    literature_max_k: int,
) -> list[dict]:
    """Cluster the identical passage sets under ``trials`` orderings.

    Trial 0 is always the exact order the caller passed in (the pipeline's
    real, canonical run — recomputing it here is cheap, since embeddings are
    already hydrated, and it doubles as an internal consistency check: it
    must reproduce the caller's own clustering exactly). Trials 1..N-1 are
    independently seeded shuffles. Returns one dict per trial: joint mode
    carries ``k``/``silhouette``; independent mode carries
    ``researcher_k``/``researcher_silhouette``/``literature_k``/
    ``literature_silhouette``, since the two corpora can drift independently.
    """
    if trials > 1000:
        raise ValueError(
            "trials above 1,000 would reuse shuffle seeds across the two "
            "corpora (the literature stream is offset by 1,000). Run fewer "
            "trials per invocation and vary seed_base across invocations."
        )
    results: list[dict] = []
    for trial in range(trials):
        if trial == 0:
            # Trial 0 re-runs the caller's own order, but on copies: it must
            # verify the canonical clustering, not overwrite it.
            r_trial, l_trial = detached(researcher), detached(literature)
        else:
            r_trial = shuffled(researcher, seed_base + trial)
            l_trial = shuffled(literature, seed_base + 1000 + trial)

        if mode == JOINT:
            joint = cluster_joint(
                r_trial,
                l_trial,
                embedder,
                max_k=joint_max_k,
                n_representatives=n_representatives,
                **clustering_kwargs,
            )
            results.append({"trial": trial, "k": joint.n_clusters, "silhouette": joint.silhouette})
        else:
            researcher_summary = cluster_passages(
                r_trial,
                embedder,
                label="researcher",
                max_k=researcher_max_k,
                n_representatives=n_representatives,
                **clustering_kwargs,
            )
            literature_summary = cluster_passages(
                l_trial,
                embedder,
                label="literature",
                max_k=literature_max_k,
                n_representatives=n_representatives,
                **clustering_kwargs,
            )
            results.append(
                {
                    "trial": trial,
                    "researcher_k": researcher_summary.n_clusters,
                    "researcher_silhouette": researcher_summary.silhouette,
                    "literature_k": literature_summary.n_clusters,
                    "literature_silhouette": literature_summary.silhouette,
                }
            )
    return results


def _corpus_summary(ks: list[int], sils: list[float], expected_k: int | None = None) -> dict:
    """Stats over the trials, with the canonical run kept out of the rate.

    ``ks[0]`` is the canonical order re-clustered, not a shuffle. Counting it
    as a match would keep ``k_match_rate`` from falling below ``1/trials`` — a
    term whose k never once survived reordering would report 20% on five
    trials, which reads like partial agreement rather than none. The rate is
    over the shuffled trials alone, so 0.0 means what it says. With no
    shuffled trials at all the rate is ``None``, not 0.0 — "nothing was
    tested" must not read as "nothing agreed".

    ``canonical_reproduced`` is the consistency check trial 0 performs:
    re-clustering the caller's own order must land on the k the caller's
    own pipeline run produced (``expected_k``). False means the clustering
    is not deterministic for this input, which invalidates the rest of the
    row. The key is only present when the caller supplied ``expected_k`` —
    a check that was not run is omitted, never reported as passed.
    """
    canonical_k = ks[0]
    shuffled_ks = ks[1:]
    matches = sum(1 for k in shuffled_ks if k == canonical_k)
    summary = {
        "canonical_k": canonical_k,
        "trials": len(ks),
        "shuffled_trials": len(shuffled_ks),
        "k_match_rate": round(matches / len(shuffled_ks), 4) if shuffled_ks else None,
        "k_seen": sorted(set(ks)),
        "silhouette_min": round(min(sils), 4),
        "silhouette_max": round(max(sils), 4),
    }
    if expected_k is not None:
        summary["canonical_reproduced"] = canonical_k == expected_k
    return summary


def summarize(
    results: list[dict],
    mode: str,
    *,
    expected_k: int | None = None,
    expected_researcher_k: int | None = None,
    expected_literature_k: int | None = None,
) -> dict:
    """Compact stats for a report line or diagnostics — not the full table.

    ``k_match_rate`` is the share of the *shuffled* trials that landed on the
    same k as the canonical run: 1.0 means every reordering agreed, and 0.0
    means none did. The canonical run is excluded from the rate because
    counting it would make 0.0 unreachable — a term whose k never once
    survived reordering would still report ``1/trials``.

    The ``expected_*`` kwargs are the k values the caller's own pipeline run
    actually produced (joint mode uses ``expected_k``; independent mode uses
    the per-corpus pair). When given, each corpus summary carries
    ``canonical_reproduced``: trial 0 re-clustered the caller's order, so it
    must land on the caller's k, and False flags nondeterminism that
    invalidates the row.
    """
    if mode == JOINT:
        return {
            "mode": JOINT,
            **_corpus_summary(
                [r["k"] for r in results],
                [r["silhouette"] for r in results],
                expected_k=expected_k,
            ),
        }
    return {
        "mode": INDEPENDENT,
        "trials": len(results),
        "researcher": _corpus_summary(
            [r["researcher_k"] for r in results],
            [r["researcher_silhouette"] for r in results],
            expected_k=expected_researcher_k,
        ),
        "literature": _corpus_summary(
            [r["literature_k"] for r in results],
            [r["literature_silhouette"] for r in results],
            expected_k=expected_literature_k,
        ),
    }
