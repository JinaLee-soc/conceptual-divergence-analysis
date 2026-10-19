#!/usr/bin/env python3
"""Bootstrap stability of the literature clusters, resampling the LIBRARY.

Answers "had my personal library been a slightly different sample of papers,
would the same clusters appear?" — the internal-stability complement to the
run-to-run stability protocol. The resampling unit is therefore the SOURCE
(paper), never the passage: passages from one paper are correlated, and
passage-level resampling would understate library-level variability.

Protocol (free, local, no LLM):
  1. Compute the reference clustering exactly as `cda analyze` would today
     (Stage 1 retrieval, the configured saturation scope, spherical k-means
     with silhouette k-selection).
  2. For each of B replicates, subsample a fraction f of the library's
     sources WITHOUT replacement (Ben-Hur-style subsampling; sampling with
     replacement would duplicate points, which distorts k-means centroids
     and silhouettes), induce the passage pool, and re-run the same
     selection + clustering, including k re-selection.
  3. Report, per reference cluster, the Hennig-style bootstrap Jaccard
     (best-matching replicate cluster, averaged over replicates) with the
     conventional bands: >= 0.85 highly stable, 0.75-0.85 stable,
     0.60-0.75 pattern-level, < 0.60 dissolved. Plus: the distribution of
     selected k across replicates, and the ARI between reference and
     replicate partitions on their shared passages.

Known approximation, stated in the output: retrieval is not re-executed
against the resampled library. The replicate pool is the reference pool
filtered by source membership, so the semantic channel's top-N is thinned,
not backfilled. This slightly understates pool sizes in replicates.

The researcher side is deliberately out of scope: drafts are not a sample
from a population of drafts, so "bootstrapping" them has no referent.

Usage:
    python benchmarks/bootstrap_stability.py --demo --term novelty --replicates 10
    python benchmarks/bootstrap_stability.py --term temporality --config my-config.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from cda.clustering.spherical import adjusted_rand_index, cluster_passages
from cda.config import load_config
from cda.corpus.embeddings import get_embedder
from cda.corpus.ingest import get_collection
from cda.extraction.windows import safe_term_slug
from cda.pipeline import (
    _clustering_kwargs,
    extract_passages,
    resolution_params,
    select_literature_by_saturation,
)

_CLUSTER_LABEL = re.compile(r"^Cluster (\d+)$")

BANDS = (
    (0.85, "highly stable"),
    (0.75, "stable"),
    (0.60, "pattern-level"),
    (0.0, "dissolved"),
)


def band(jaccard: float | None) -> str:
    """Hennig band for a mean Jaccard; "n/a" when the cluster was never scored."""
    if jaccard is None:
        return "n/a"
    for threshold, label in BANDS:
        if jaccard >= threshold:
            return label
    return "dissolved"


def cluster_and_snapshot(term, pool, config, embedder) -> tuple[dict[str, int], int]:
    """Scope-aware selection + clustering; returns ({chunk_id: label}, k).

    Runs the pipeline's own saturation step (honoring saturation.scope) and
    spherical k-means, then snapshots labels — snapshots matter because
    cluster_passages writes cluster_id onto shared Passage objects and the
    next replicate overwrites them.
    """
    selected, _info, _ran = select_literature_by_saturation(
        term, list(pool), config, embedder=embedder
    )
    summary = cluster_passages(
        selected,
        embedder,
        label="literature",
        max_k=int(config.get("clustering.literature_max_k", 15)),
        n_representatives=int(config.get("clustering.representatives_per_cluster", 5)),
        **_clustering_kwargs(config),
        **resolution_params(config),
    )
    if not summary.used:
        return {}, 0
    labels: dict[str, int] = {}
    for passage in selected:
        match = _CLUSTER_LABEL.match(str(passage.cluster_id))
        if match:
            labels[passage.chunk_id] = int(match.group(1))
    return labels, summary.n_clusters


def jaccard_best_match(
    reference_members: set[str], replicate: dict[str, int], reference_points: set[str]
) -> float | None:
    """Hennig (2007): best Jaccard of a reference cluster over replicate clusters.

    Both clusterings are restricted to the points they share (clustered in
    the reference and in the replicate), so points the reference never saw
    cannot dilute the score. None when the reference cluster has no shared
    point (nothing to compare).
    """
    shared = reference_points & set(replicate)
    present = reference_members & shared
    if not present:
        return None
    best = 0.0
    for cluster in set(replicate.values()):
        members = {chunk for chunk in shared if replicate[chunk] == cluster}
        best = max(best, len(present & members) / len(present | members))
    return best


def score_replicate(
    reference: dict[str, int], replicate: dict[str, int]
) -> tuple[float | None, dict[int, float | None]]:
    """ARI on the shared passages (None if none) and each reference cluster's Jaccard."""
    shared = sorted(set(replicate) & set(reference))
    ari = (
        adjusted_rand_index(
            np.array([reference[c] for c in shared]), np.array([replicate[c] for c in shared])
        )
        if shared
        else None
    )
    reference_points = set(reference)
    jaccards = {
        cluster: jaccard_best_match(
            {chunk for chunk, label in reference.items() if label == cluster},
            replicate,
            reference_points,
        )
        for cluster in sorted(set(reference.values()))
    }
    return ari, jaccards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term", default="novelty")
    parser.add_argument("--demo", action="store_true", help="Run on the toy corpus")
    parser.add_argument("--config", help="Config file for a real corpus")
    parser.add_argument(
        "--replicates", type=int, default=30, help="Bootstrap replicates (default 30)"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.8,
        help="Fraction of library sources kept per replicate (default 0.8)",
    )
    args = parser.parse_args()

    if args.demo:
        from cda.corpus.embeddings import get_embedder as _get_embedder
        from cda.demo import demo_config, ensure_toy_corpus_ingested, find_demo_dir

        demo_dir = find_demo_dir()
        config = demo_config(demo_dir)
        embedder = _get_embedder(config)
        ensure_toy_corpus_ingested(config, demo_dir, embedder)
    else:
        config = load_config(args.config)
        embedder = get_embedder(config)
    seed = int(config.get("clustering.seed", 42))

    collection = get_collection(config)
    _researcher, literature, _stats = extract_passages(
        args.term, config, embedder=embedder, collection=collection, progress=lambda _m: None
    )
    sources = sorted({p.source for p in literature})
    print(
        f"[bootstrap] term='{args.term}' pool={len(literature)} passages from "
        f"{len(sources)} sources | scope={config.get('saturation.scope', 'sample')} "
        f"| B={args.replicates} f={args.fraction}"
    )

    reference, reference_k = cluster_and_snapshot(args.term, literature, config, embedder)
    if not reference:
        print("[bootstrap] the reference clustering produced no valid partition; nothing to test")
        return 1
    reference_clusters = {
        cluster: {chunk for chunk, label in reference.items() if label == cluster}
        for cluster in sorted(set(reference.values()))
    }
    print(f"[bootstrap] reference: k={reference_k}, {len(reference)} clustered passages")

    keep = max(2, round(args.fraction * len(sources)))
    per_cluster: dict[int, list[float]] = {c: [] for c in reference_clusters}
    aris: list[float] = []
    ks: list[int] = []
    failed = 0

    for b in range(args.replicates):
        rng = np.random.default_rng([seed, b])
        sampled = set(rng.choice(sources, size=keep, replace=False))
        pool = [p for p in literature if p.source in sampled]
        labels, k = cluster_and_snapshot(args.term, pool, config, embedder)
        if not labels:
            failed += 1
            print(f"  replicate {b + 1}/{args.replicates}: no valid partition")
            continue
        ks.append(k)
        ari, jaccards = score_replicate(reference, labels)
        if ari is not None:
            aris.append(ari)
        for cluster, score in jaccards.items():
            if score is not None:
                per_cluster[cluster].append(score)
        ari_text = "n/a (no shared passages)" if ari is None else f"{ari:.3f}"
        print(f"  replicate {b + 1}/{args.replicates}: k={k}, ARI={ari_text}")

    now = datetime.now()
    lines = [
        f'# Bootstrap stability: "{args.term}" ({now:%Y-%m-%d %H:%M})',
        "",
        f"- Library: {len(sources)} sources, {len(literature)} retrieved passages; "
        f"replicates keep {keep} sources ({args.fraction:.0%}, without replacement)",
        f"- Saturation scope: {config.get('saturation.scope', 'sample')} | seed {seed} "
        f"| replicates {args.replicates} ({failed} without a valid partition)",
        f"- Reference: k={reference_k} over {len(reference)} clustered passages",
        "",
        "Approximation: retrieval is not re-executed per replicate; the pool is the",
        "reference retrieval filtered by sampled sources (semantic channel thinned,",
        "not backfilled).",
        "",
        "| reference cluster | size | mean Jaccard | sd | verdict (Hennig bands) |",
        "|---|---:|---:|---:|---|",
    ]
    results: dict = {
        "term": args.term,
        "generated": now.isoformat(timespec="seconds"),
        "n_sources": len(sources),
        "n_passages": len(literature),
        "fraction": args.fraction,
        "replicates": args.replicates,
        "failed_replicates": failed,
        "scope": str(config.get("saturation.scope", "sample")),
        "reference_k": reference_k,
        "k_distribution": dict(sorted(Counter(ks).items())),
        "ari_mean": round(float(np.mean(aris)), 4) if aris else None,
        "ari_min": round(float(np.min(aris)), 4) if aris else None,
        "ari_max": round(float(np.max(aris)), 4) if aris else None,
        "clusters": {},
    }
    for cluster, scores in per_cluster.items():
        mean = float(np.mean(scores)) if scores else None
        sd = float(np.std(scores)) if scores else None
        verdict = band(mean)
        lines.append(
            f"| Cluster {cluster} | {len(reference_clusters[cluster])} "
            f"| {'n/a' if mean is None else f'{mean:.3f}'} "
            f"| {'n/a' if sd is None else f'{sd:.3f}'} | {verdict} |"
        )
        results["clusters"][str(cluster)] = {
            "size": len(reference_clusters[cluster]),
            "mean_jaccard": None if mean is None else round(mean, 4),
            "sd": None if sd is None else round(sd, 4),
            "n_scores": len(scores),
            "verdict": verdict,
        }
    k_line = ", ".join(f"k={k}: {n}" for k, n in sorted(Counter(ks).items()))
    lines += [
        "",
        f"Selected k across replicates: {k_line or 'n/a'}",
        (
            f"ARI vs reference on shared passages: mean {results['ari_mean']} "
            f"(min {results['ari_min']}, max {results['ari_max']})"
            if aris
            else "ARI vs reference: n/a"
        ),
        "",
        "Reading: Jaccard >= 0.85 highly stable; 0.75-0.85 stable; 0.60-0.75 a",
        "pattern that reappears but with shifting membership; < 0.60 dissolves",
        "under library resampling and should not be reported as a register",
        "without corroboration (e.g. a blind verdict in `cda validate`).",
        "",
    ]
    report_text = "\n".join(lines)
    try:
        print("\n" + report_text)
    except UnicodeEncodeError:
        # Windows consoles often use a legacy codepage that cannot encode
        # corpus text. Degrade the console copy rather than die before the
        # report reaches disk (the file below is UTF-8 and unaffected).
        import sys

        encoding = sys.stdout.encoding or "utf-8"
        print("\n" + report_text.encode(encoding, errors="replace").decode(encoding))

    out_dir = config.path("paths.output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bootstrap_{safe_term_slug(args.term)}_{now:%Y%m%d_%H%M%S}"
    (out_dir / f"{stem}.md").write_text(report_text, encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[bootstrap] written: {out_dir / (stem + '.md')} (+ .json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
