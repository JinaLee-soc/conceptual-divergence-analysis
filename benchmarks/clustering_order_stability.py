#!/usr/bin/env python3
"""Order-sensitivity of the clustering step, for one term (opt-in diagnostic).

Clusters the *identical* passage set — the exact selection `cda analyze`
would make (Stage 1 extraction + saturation subsampling, untouched) — under
N independently shuffled orderings, and reports how much the chosen k and
silhouette actually vary.

Why this exists: k-means++ seeding draws initial centroids by array
position, so the pipeline's one canonical order makes `cda analyze`
reproducible but not necessarily robust; a weakly separated term can land
on a different k under another order. Shares and the most common k are
computed over the shuffled orderings only, matching
`cda.clustering.order_stability.summarize`.

`cda analyze` also runs a small built-in version of this check by default
(`clustering.stability_check`, 30 shuffles by default, folded into every
report's "Clustering Stability" section) so you are
never entirely without this signal. Reach for this script when you want more
trials, the full per-trial table, or a standalone report to cite.

Usage:
    python benchmarks/clustering_order_stability.py --demo --term novelty
    python benchmarks/clustering_order_stability.py --config my-config.yaml --term novelty --trials 30

No LLM call, no cost — this is a clustering-only diagnostic.
See benchmarks/README.md for pointing this at a real corpus.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from cda.clustering.order_stability import JOINT, resolution_override_reason, run_trials
from cda.config import load_config
from cda.corpus.embeddings import get_embedder
from cda.corpus.ingest import get_collection
from cda.extraction.windows import safe_term_slug
from cda.pipeline import (
    _clustering_kwargs,
    clustering_mode,
    extract_passages,
    resolution_params,
    select_literature_by_saturation,
)


def _labels(results: list[dict]) -> list[str]:
    return ["canonical (cda analyze)"] + [f"shuffle {r['trial']}" for r in results[1:]]


def _distribution_table(ks: list[int], canonical_k: int, found: list[bool]) -> list[str]:
    """Where the shuffled orderings (ks[1:]) landed; trial 0 is the canonical run.

    Orderings without a valid partition get their own row: their k is a
    placeholder, not a landing.
    """
    counts = Counter(k for k, ok in zip(ks[1:], found[1:], strict=True) if ok)
    no_partition = found[1:].count(False)
    shuffles = len(ks) - 1
    lines = ["| k | shuffled orderings landing here | share |", "|---:|---:|---:|"]
    landed = set(counts) | ({canonical_k} if found[0] else set())
    for k in sorted(landed):
        marker = " ← canonical" if found[0] and k == canonical_k else ""
        lines.append(f"| {k} | {counts[k]} | {counts[k] / shuffles:.0%}{marker} |")
    if no_partition or not found[0]:
        marker = " ← canonical" if not found[0] else ""
        lines.append(
            f"| no valid partition | {no_partition} | {no_partition / shuffles:.0%}{marker} |"
        )
    return lines


def refusal_reason(config) -> str | None:
    """Why this benchmark must not run under ``config``, or None.

    The canonical trial claims to be what `cda analyze` produced, and the
    shuffles measure the silhouette k-selection; with clustering.method or
    clustering.k overriding that selection, both claims would be false.
    Mirrors the pipeline's built-in check.
    """
    return resolution_override_reason(resolution_params(config))


def _sil(value: float | None) -> str:
    return "n/a (no valid partition)" if value is None else f"{value:.4f}"


def _sil_range(sils: list[float | None]) -> str:
    found = [s for s in sils if s is not None]
    return f"{min(found):.4f}–{max(found):.4f}" if found else "n/a (no valid partition)"


def _reading_this(
    label: str, ks: list[int], sils: list[float | None], canonical_k: int, found: list[bool]
) -> str:
    missing = found.count(False)
    if not found[0]:
        return (
            f"No valid {label} partition was found in {missing} of {len(ks)} orderings, "
            "including the canonical run. There is no cluster structure here to call "
            "robust or fragile; the k shown is a placeholder, not a result."
        )
    missing_note = (
        f" No valid partition was found in {missing} of {len(ks)} orderings; those "
        "trials are counted as disagreeing and left out of the silhouette range."
        if missing
        else ""
    )
    # A trial without a partition is never an agreement (its k is a placeholder).
    ks = [k if ok else None for k, ok in zip(ks, found, strict=True)]
    counts = Counter(ks[1:])  # trial 0 is the canonical re-run, not a reordering
    shuffles = len(ks) - 1
    if set(ks) == {canonical_k}:
        if shuffles < 5:
            # Agreement from one or two reorderings is not evidence of
            # robustness — a verdict must not outrun its sample.
            return (
                f"Every ordering landed on the same {label} k — but only "
                f"{shuffles} shuffled ordering(s) were run. That is agreement, "
                "not yet robustness; rerun with more --trials before reading "
                "it as a property of the term."
            )
        return (
            f"Every ordering landed on the same {label} k across {shuffles} shuffled "
            "orderings. This term's cluster structure looks robust to how the "
            "passages happen to be ordered — canonical ordering makes the run "
            "reproducible, and this result says the structure is *also* not "
            "fragile."
        )
    top = max(counts.values())
    modal = sorted((k for k, n in counts.items() if n == top), key=lambda k: (k is None, k))
    modal_text = " / ".join("no partition" if k is None else f"k={k}" for k in modal)
    agree = "agrees with" if canonical_k in modal else "does NOT match"
    return (
        f"{label.capitalize()} k varied across orderings ({len(set(ks))} distinct values "
        f"over {len(ks)} trials; silhouette ranged {_sil_range(sils)}). "
        f"The canonical run's k={canonical_k} {agree} the most common value among "
        f"the shuffled orderings ({modal_text}, {top}/{shuffles} shuffles). This is a "
        "property of the *term* — likely weak cluster separation — not a bug: the "
        "canonical run is still fully reproducible (same order every time), but its "
        "specific k should be read as one plausible partition among several nearby "
        "ones, not as the term's one true register count."
        f"{missing_note}"
    )


def _found(results: list[dict], key: str) -> list[bool]:
    return [bool(r.get(key, True)) for r in results]


def render(term: str, mode: str, n_researcher: int, n_literature: int, results: list[dict]) -> str:
    if len(results) < 2:
        raise ValueError(
            "render needs the canonical run plus at least one shuffled ordering; "
            "a verdict from zero shuffles would be a claim with no evidence"
        )
    trials = len(results)
    labels = _labels(results)
    lines = [
        f'# Clustering order-stability — term: "{term}"',
        "",
        f"Mode: **{mode}** | {n_researcher} researcher / {n_literature} literature passages "
        "(identical set across every trial — only order varies) | "
        f"{trials} trials (1 canonical + {trials - 1} independently shuffled).",
        "",
    ]

    if mode == JOINT:
        ks = [r["k"] for r in results]
        sils = [r["silhouette"] for r in results]
        found = _found(results, "partition_found")
        canonical_k = ks[0]
        lines += [
            f"**Canonical run** (what `cda analyze` produces today): k={canonical_k}, "
            f"silhouette={_sil(sils[0])}.",
            "",
            f"**Across all {trials} orderings:** k ranged {min(ks)}–{max(ks)} "
            f"({len(set(ks))} distinct values); silhouette ranged {_sil_range(sils)}; "
            f"no valid partition in {found.count(False)} of {trials}.",
            "",
            *_distribution_table(ks, canonical_k, found),
            "",
            "| Trial | k | silhouette |",
            "|---|---:|---:|",
        ]
        for label, r in zip(labels, results, strict=True):
            lines.append(f"| {label} | {r['k']} | {_sil(r['silhouette'])} |")
        lines += ["", "## Reading this", _reading_this("joint", ks, sils, canonical_k, found)]
    else:
        r_ks = [r["researcher_k"] for r in results]
        r_sils = [r["researcher_silhouette"] for r in results]
        l_ks = [r["literature_k"] for r in results]
        l_sils = [r["literature_silhouette"] for r in results]
        r_found = _found(results, "researcher_partition_found")
        l_found = _found(results, "literature_partition_found")
        lines += [
            f"**Canonical run:** researcher k={r_ks[0]} (silhouette={_sil(r_sils[0])}), "
            f"literature k={l_ks[0]} (silhouette={_sil(l_sils[0])}). Independent mode "
            "clusters the two corpora separately, so each has its own stability picture.",
            "",
            "### Researcher corpus",
            "",
            *_distribution_table(r_ks, r_ks[0], r_found),
            "",
            "### Literature corpus",
            "",
            *_distribution_table(l_ks, l_ks[0], l_found),
            "",
            "| Trial | researcher k | researcher sil. | literature k | literature sil. |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, r in zip(labels, results, strict=True):
            lines.append(
                f"| {label} | {r['researcher_k']} | {_sil(r['researcher_silhouette'])} "
                f"| {r['literature_k']} | {_sil(r['literature_silhouette'])} |"
            )
        lines += [
            "",
            "## Reading this",
            _reading_this("researcher", r_ks, r_sils, r_ks[0], r_found),
            "",
            _reading_this("literature", l_ks, l_sils, l_ks[0], l_found),
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term", required=True)
    parser.add_argument("--demo", action="store_true", help="Run on the toy corpus")
    parser.add_argument("--config", help="Config file for a real corpus")
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Independent shuffles to compare against the canonical run (default 20)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Offset for the shuffle seeds, for a fresh independent sample of orderings",
    )
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials is the number of shuffled orderings and must be at least 1")

    if args.demo:
        from cda.demo import demo_config, ensure_toy_corpus_ingested, find_demo_dir

        demo_dir = find_demo_dir()
        config = demo_config(demo_dir)
    else:
        config = load_config(args.config)
    reason = refusal_reason(config)
    if reason:
        # Refuse before any work: the "canonical (cda analyze)" row would be an
        # auto-k k-means partition that this config's analyze never produces.
        print(f"[benchmark] order-stability check not applicable: {reason}.")
        return 2
    embedder = get_embedder(config)
    if args.demo:
        ensure_toy_corpus_ingested(config, demo_dir, embedder)

    collection = get_collection(config)
    researcher, literature, _stats = extract_passages(
        args.term, config, embedder=embedder, collection=collection, progress=lambda _msg: None
    )
    literature, _info, _ran = select_literature_by_saturation(
        args.term, literature, config, embedder=embedder
    )
    print(
        f"[benchmark] {len(researcher)} researcher / {len(literature)} literature passages "
        f"(post-saturation); running 1 canonical + {args.trials} shuffled orderings"
    )

    mode = clustering_mode(config)
    # run_trials counts the canonical re-run as trial 0, so ask for one more
    # trial than the number of shuffles the user requested — "--trials 20"
    # must mean twenty *shuffled* orderings, as the help text promises.
    results = run_trials(
        researcher,
        literature,
        embedder,
        mode=mode,
        trials=args.trials + 1,
        seed_base=args.seed_base,
        n_representatives=int(config.get("clustering.representatives_per_cluster", 5)),
        clustering_kwargs=_clustering_kwargs(config),
        joint_max_k=int(config.get("clustering.joint_max_k", 15)),
        researcher_max_k=int(config.get("clustering.researcher_max_k", 8)),
        literature_max_k=int(config.get("clustering.literature_max_k", 15)),
    )
    report = render(args.term, mode, len(researcher), len(literature), results)
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
        f"benchmark_order_stability_{safe_term_slug(args.term)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    out_path.write_text(report, encoding="utf-8")
    print(f"[benchmark] written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
