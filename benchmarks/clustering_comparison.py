#!/usr/bin/env python3
"""Clustering-algorithm comparison on identical CDA passage sets.

Runs the paper's spherical k-means against two standard alternatives —
sklearn k-means (euclidean) and agglomerative clustering (cosine, average
linkage) — on exactly the passage sets a CDA analysis would cluster, and
reports: chosen k per method, cosine silhouette on a common basis, pairwise
ARI between solutions, and agreement with the LLM registers of the most
recent CDA report for the term (when one exists).

Usage:
    python benchmarks/clustering_comparison.py --demo --term novelty
    python benchmarks/clustering_comparison.py --config my-config.yaml --term novelty

See benchmarks/README.md for pointing this at a real corpus.
"""

from __future__ import annotations

import argparse
from datetime import datetime

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from cda.clustering.spherical import (
    adjusted_rand_index,
    hydrate_embeddings,
    merge_small_clusters,
    prepare_matrix,
    select_k_by_silhouette,
    silhouette_on_sample,
    silhouette_sample_indices,
)
from cda.config import load_config
from cda.corpus.embeddings import get_embedder
from cda.corpus.ingest import get_collection
from cda.extraction.windows import safe_term_slug
from cda.pipeline import extract_passages, select_literature_by_saturation


def build_passage_sets(term: str, config, embedder):
    """The exact researcher/literature passage sets `cda analyze` would cluster.

    Reuses the pipeline's own Stage 1 and saturation step so the benchmark
    can never drift from what a real analysis does — the whole point of the
    comparison is that the *inputs* are identical.
    """
    collection = get_collection(config)
    researcher, literature, _stats = extract_passages(
        term, config, embedder=embedder, collection=collection, progress=lambda _msg: None
    )
    literature, _info, _ran = select_literature_by_saturation(
        term, literature, config, embedder=embedder
    )
    return researcher, literature


def sweep_best(labels_for_k, X_common, k_range, evaluation_indices, min_cluster_size):
    """Pick the best post-merge partition on one fixed evaluation set."""
    best = (None, 1, None)
    for k in k_range:
        labels = labels_for_k(k)
        if labels is None:
            continue
        if min_cluster_size > 1:
            labels, _merges = merge_small_clusters(X_common, labels, min_size=min_cluster_size)
        final_k = len(set(labels.tolist()))
        if final_k < 2:
            continue
        score = silhouette_on_sample(X_common, labels, evaluation_indices)
        if score is not None and (best[2] is None or score > best[2]):
            best = (labels, final_k, score)
    return best


def compare_methods(passages, embedder, config, label: str, seed: int, max_k=None) -> dict:
    """Cluster one passage set three ways on the same embedding matrix."""
    from sklearn.cluster import AgglomerativeClustering, KMeans

    hydrate_embeddings(passages, embedder)
    X_norm, _with_embeddings = prepare_matrix(passages)
    if X_norm is None:
        return {"label": label, "error": "too few passages with embeddings"}
    clustering = config.section("clustering")
    if max_k is None:
        max_k = int(
            clustering.get("researcher_max_k" if label == "researcher" else "literature_max_k", 8)
        )
    k_range = range(2, min(max_k, X_norm.shape[0] - 1) + 1)
    silhouette_subsample = int(clustering.get("silhouette_subsample", 2000))
    min_cluster_size = int(clustering.get("min_cluster_size", 3))
    evaluation_indices = silhouette_sample_indices(X_norm.shape[0], silhouette_subsample, seed)

    spherical_labels, spherical_k, spherical_sil, _sweep = select_k_by_silhouette(
        X_norm,
        max_k=max_k,
        n_restarts=int(clustering.get("kmeans_restarts", 5)),
        max_iter=int(clustering.get("kmeans_max_iter", 80)),
        min_cluster_size=min_cluster_size,
        silhouette_subsample=silhouette_subsample,
        seed=seed,
    )

    standard_labels, standard_k, standard_sil = sweep_best(
        lambda k: KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X_norm),
        X_norm,
        k_range,
        evaluation_indices,
        min_cluster_size,
    )
    agglomerative_labels, agglomerative_k, agglomerative_sil = sweep_best(
        lambda k: AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(X_norm),
        X_norm,
        k_range,
        evaluation_indices,
        min_cluster_size,
    )

    solutions = {
        "spherical k-means (CDA)": (spherical_labels, spherical_k, spherical_sil),
        "standard k-means (euclidean on normalized embeddings)": (
            standard_labels,
            standard_k,
            standard_sil,
        ),
        "agglomerative (cosine/average)": (
            agglomerative_labels,
            agglomerative_k,
            agglomerative_sil,
        ),
    }
    ari = {}
    names = list(solutions)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            la, lb = solutions[a][0], solutions[b][0]
            ari[f"{a} vs {b}"] = (
                round(adjusted_rand_index(la, lb), 3) if la is not None and lb is not None else None
            )
    evaluation_n = (
        int(len(evaluation_indices)) if evaluation_indices is not None else int(X_norm.shape[0])
    )
    return {
        "label": label,
        "n": int(X_norm.shape[0]),
        "evaluation_n": evaluation_n,
        "solutions": solutions,
        "ari": ari,
    }


def llm_register_count(config, term: str) -> int | None:
    """Researcher-register count from the newest CDA report for the term."""
    import re

    out_dir = config.path("paths.output_dir")
    reports = sorted(
        p
        for p in out_dir.glob(f"drift_{safe_term_slug(term)}_*.md")
        if not p.name.endswith(("_prompt.md", "_system.md")) and "_verification_" not in p.name
    )
    if not reports:
        return None
    text = reports[-1].read_text(encoding="utf-8")
    return len(re.findall(r"^###\s+Cluster\s+\S+", text, flags=re.MULTILINE)) or None


def render(results: list[dict], term: str, llm_registers: int | None) -> str:
    lines = [
        f'# Clustering comparison — term: "{term}"',
        "",
        "Same passage set and same L2-normalized embedding matrix for all three algorithms.",
        "Every k and method is scored by cosine silhouette on one fixed evaluation sample,",
        "after applying the same minimum-cluster-size rule (common basis).",
        "",
    ]
    for result in results:
        lines.append(f"## {result['label']} corpus")
        if "error" in result:
            lines += ["", f"*{result['error']}*", ""]
            continue
        lines += [
            "",
            f"n = {result['n']} passages; fixed silhouette evaluation n = {result['evaluation_n']}",
            "",
            "| Method | chosen k | cosine silhouette |",
            "|---|---:|---:|",
        ]
        for name, (labels, k, sil) in result["solutions"].items():
            lines.append(
                f"| {name} | {k if labels is not None else '—'} "
                f"| {'—' if sil is None else f'{sil:.3f}'} |"
            )
        lines += ["", "| Solution pair | ARI |", "|---|---:|"]
        for pair, value in result["ari"].items():
            lines.append(f"| {pair} | {value if value is not None else '—'} |")
        lines.append("")
    if llm_registers is not None:
        lines += [
            "## Agreement with LLM registers",
            "",
            f"The most recent CDA report for this term formed **{llm_registers}** researcher "
            "registers from the spherical partition (see the report's cluster–register audit "
            "for the mapping). Where an alternative algorithm's ARI against the spherical "
            "solution is high, the LLM registers would be equally reachable from it; a low "
            "ARI means register boundaries depend on the clustering choice — worth reporting.",
        ]
    else:
        lines += [
            "*(No CDA report found for this term in the output directory — run "
            "`cda analyze`/`cda demo` first to add the LLM-register comparison.)*",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term", default="novelty")
    parser.add_argument("--demo", action="store_true", help="Run on the toy corpus")
    parser.add_argument("--config", help="Config file for a real corpus")
    parser.add_argument(
        "--per-corpus",
        action="store_true",
        help=(
            "Compare the algorithms on the researcher and literature sets "
            "separately, which is what clustering.mode: independent partitions. "
            "The default pools them, matching the joint mode an analysis uses."
        ),
    )
    args = parser.parse_args()

    if args.demo:
        from cda.demo import demo_config, ensure_toy_corpus_ingested, find_demo_dir

        demo_dir = find_demo_dir()
        config = demo_config(demo_dir)
        embedder = get_embedder(config)
        ensure_toy_corpus_ingested(config, demo_dir, embedder)
    else:
        config = load_config(args.config)
        embedder = get_embedder(config)

    seed = int(config.get("clustering.seed"))
    researcher, literature = build_passage_sets(args.term, config, embedder)
    print(f"[benchmark] {len(researcher)} researcher / {len(literature)} literature passages")

    if args.per_corpus:
        results = [
            compare_methods(researcher, embedder, config, "researcher", seed),
            compare_methods(literature, embedder, config, "literature", seed),
        ]
    else:
        # Compare on the partition an analysis actually computes: one
        # spherical k-means over the two corpora pooled. Comparing the
        # per-corpus partitions instead would justify an algorithm for a
        # configuration the default path never runs.
        results = [
            compare_methods(
                list(researcher) + list(literature),
                embedder,
                config,
                "pooled (researcher + literature)",
                seed,
                max_k=int(config.get("clustering.joint_max_k", 15)),
            )
        ]
    report = render(results, args.term, llm_register_count(config, args.term))
    # Print before writing: a multi-minute benchmark's result should survive
    # a filename or disk problem at the very last step.
    try:
        print(report)
    except UnicodeEncodeError:
        # Windows consoles often use a legacy codepage that cannot encode
        # corpus text. Degrade the console copy rather than die before the
        # report reaches disk (the file below is UTF-8 and unaffected).
        import sys

        encoding = sys.stdout.encoding or "utf-8"
        print(report.encode(encoding, errors="replace").decode(encoding))

    out_dir = config.path("paths.output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"benchmark_clustering_{safe_term_slug(args.term)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    out_path.write_text(report, encoding="utf-8")
    print(f"[benchmark] written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
