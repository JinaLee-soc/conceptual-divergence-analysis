#!/usr/bin/env python3
"""Why did the saturation threshold fire where it did? Four probes.

Each probe discriminates one candidate mechanism behind a premature (or
lucky) ARI-plateau, using the same seeded machinery the pipeline runs
(free, local, no LLM):

  A. Coarseness confound - on IDENTICAL data, cluster with different seeds
     at fixed k and measure pairwise ARI. If ARI >= threshold is attainable
     at k=2 but not at fine k, the plateau rule structurally rewards coarse
     structure: "stable" and "coarse" are the same event.
  B. Ordering - run the real stopping rule under the term-anchor ordering,
     seeded random permutations, and the reversed (periphery-first) order.
     If the anchor order saturates earliest, the homogenized term-central
     prefix is doing the work. Also reports mean cosine-to-anchor of the
     selected vs excluded passages at the observed N*.
  C. Rule sensitivity - run the sweep to the END without stopping, then
     replay the recorded ARI series against a grid of thresholds and
     consecutive-increment requirements: where would each rule have fired?
  D. Perturbation - drop m random sources (m small), rerun the real rule,
     and report how often the reached/not-reached outcome flips.

Usage:
    python benchmarks/saturation_diagnostics.py --demo --term novelty
    python benchmarks/saturation_diagnostics.py --term temporality --config my-config.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from cda.clustering.saturation import rank_literature, saturation_select
from cda.clustering.spherical import (
    adjusted_rand_index,
    kmeans_labels,
    normalize_rows,
    prepare_matrix,
)
from cda.config import load_config
from cda.corpus.embeddings import get_embedder
from cda.corpus.ingest import get_collection
from cda.extraction.windows import safe_term_slug
from cda.pipeline import _clustering_kwargs, extract_passages


def sweep_params(config) -> dict:
    saturation = config.section("saturation")
    return dict(
        start=int(saturation.get("start", 50)),
        step=int(saturation.get("step", 25)),
        ari_threshold=float(saturation.get("ari_threshold", 0.8)),
        consecutive_increments=int(saturation.get("consecutive_increments", 2)),
        max_k=int(config.get("clustering.literature_max_k", 15)),
        **_clustering_kwargs(config),
    )


def run_rule(ranked, config, **overrides):
    params = sweep_params(config)
    params.update(overrides)
    max_passages = params.pop("max_passages", int(config.get("saturation.max_passages", 1200)))
    _selected, info = saturation_select(ranked, max_passages=max_passages, **params)
    return info


def probe_a_seed_ari(literature, config, embedder, ns, ks, n_seeds=8):
    """Pairwise same-data ARI across k-means seeds, per (n, k)."""
    results = []
    base_seed = int(config.get("clustering.seed", 42))
    kwargs = _clustering_kwargs(config)
    for n in ns:
        X, _with = prepare_matrix(literature[:n])
        if X is None:
            continue
        for k in ks:
            if k >= X.shape[0]:
                continue
            labelings = [
                kmeans_labels(
                    X,
                    k,
                    n_restarts=kwargs["n_restarts"],
                    max_iter=kwargs["max_iter"],
                    seed=base_seed + 1000 * s,
                )[0]
                for s in range(n_seeds)
            ]
            pair_aris = [float(adjusted_rand_index(a, b)) for a, b in combinations(labelings, 2)]
            results.append(
                {
                    "n": n,
                    "k": k,
                    "mean_ari": round(float(np.mean(pair_aris)), 4),
                    "min_ari": round(float(np.min(pair_aris)), 4),
                    "share_above_threshold": round(
                        float(np.mean([a >= 0.8 for a in pair_aris])), 3
                    ),
                }
            )
    return results


def probe_b_ordering(literature, term, config, embedder, n_random=3):
    """The real stopping rule under different candidate orderings."""
    seed = int(config.get("clustering.seed", 42))
    anchor_ranked = rank_literature(list(literature), term, embedder, mode="term")
    orderings = [("anchor (pipeline)", anchor_ranked), ("reversed anchor", anchor_ranked[::-1])]
    for r in range(n_random):
        rng = np.random.default_rng([seed, 7000 + r])
        permuted = list(literature)
        rng.shuffle(permuted)
        orderings.append((f"random #{r + 1}", permuted))
    rows = []
    for name, ranked in orderings:
        info = run_rule(ranked, config)
        rows.append(
            {
                "ordering": name,
                "reached": bool(info.get("saturation_reached")),
                "n_selected": info.get("n_selected"),
                "k_at_selection": next(
                    (
                        h.get("k")
                        for h in reversed(info.get("history", []))
                        if h.get("n") == info.get("n_selected")
                    ),
                    None,
                ),
            }
        )
    return rows, anchor_ranked


def probe_b_anchor_coverage(anchor_ranked, term, embedder, n_star):
    """Mean cosine-to-anchor for the selected prefix vs the excluded rest."""
    term_vec = normalize_rows(
        np.asarray(embedder.embed([term]), dtype=float), [term], what="term embeddings"
    )[0]

    def mean_cos(passages):
        if not passages:
            return None
        # rank_literature hydrated every passage; normalize_rows refuses bad rows.
        X = normalize_rows(
            np.asarray([p.embedding for p in passages], dtype=float),
            [p.chunk_id for p in passages],
        )
        return round(float(np.mean(X @ term_vec)), 4)

    return {
        "selected_mean_cos": mean_cos(anchor_ranked[:n_star]),
        "excluded_mean_cos": mean_cos(anchor_ranked[n_star:]),
    }


def probe_c_rule_grid(anchor_ranked, config):
    """Full no-stop sweep once; replay the ARI series under a rule grid."""
    info = run_rule(
        anchor_ranked, config, consecutive_increments=10**9, max_passages=len(anchor_ranked)
    )
    history = info.get("history", [])
    series = [(h["n"], h.get("ari_vs_prev")) for h in history if "n" in h]
    grid = []
    for threshold in (0.70, 0.75, 0.80, 0.85, 0.90):
        for consecutive in (2, 3, 4):
            streak = 0
            fired_at = None
            for index, (_n, ari) in enumerate(series):
                streak = streak + 1 if (ari is not None and ari >= threshold) else 0
                if streak >= consecutive:
                    fired_at = series[index - (consecutive - 1)][0]
                    break
            grid.append({"threshold": threshold, "consecutive": consecutive, "n_star": fired_at})
    return grid, history


def probe_d_perturbation(literature, term, config, embedder, ms=(2, 5, 10), reps=5):
    """Drop m random sources; does the reached/not outcome flip?"""
    seed = int(config.get("clustering.seed", 42))
    sources = sorted({p.source for p in literature})
    baseline = run_rule(rank_literature(list(literature), term, embedder, mode="term"), config)
    baseline_reached = bool(baseline.get("saturation_reached"))
    rows = []
    for m in ms:
        flips = 0
        n_stars = []
        for r in range(reps):
            rng = np.random.default_rng([seed, 9000 + 100 * m + r])
            dropped = set(rng.choice(sources, size=m, replace=False))
            pool = [p for p in literature if p.source not in dropped]
            info = run_rule(rank_literature(pool, term, embedder, mode="term"), config)
            if bool(info.get("saturation_reached")) != baseline_reached:
                flips += 1
            n_stars.append(info.get("n_selected"))
        rows.append(
            {"sources_dropped": m, "reps": reps, "outcome_flips": flips, "n_stars": n_stars}
        )
    return rows, baseline_reached, baseline.get("n_selected")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term", default="novelty")
    parser.add_argument("--demo", action="store_true", help="Run on the toy corpus")
    parser.add_argument("--config", help="Config file for a real corpus")
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

    collection = get_collection(config)
    _researcher, literature, _stats = extract_passages(
        args.term, config, embedder=embedder, collection=collection, progress=lambda _m: None
    )
    print(f"[satdiag] term='{args.term}' pool={len(literature)} passages")

    print("[satdiag] probe B: ordering (anchor / reversed / random x3) ...")
    ordering_rows, anchor_ranked = probe_b_ordering(literature, args.term, config, embedder)
    observed = next(r for r in ordering_rows if r["ordering"] == "anchor (pipeline)")
    n_star = observed["n_selected"] or len(literature)
    coverage = probe_b_anchor_coverage(anchor_ranked, args.term, embedder, n_star)

    print("[satdiag] probe A: same-data ARI across seeds at fixed k ...")
    full_n = len([p for p in anchor_ranked if p.embedding is not None])
    ns = sorted({min(n_star, full_n), min(4 * n_star, full_n), full_n})
    seed_rows = probe_a_seed_ari(anchor_ranked, config, embedder, ns=ns, ks=(2, 5, 13))

    print("[satdiag] probe C: full sweep without stopping + rule grid ...")
    grid, history = probe_c_rule_grid(anchor_ranked, config)

    print("[satdiag] probe D: drop-m source perturbations ...")
    perturb_rows, baseline_reached, baseline_n = probe_d_perturbation(
        literature, args.term, config, embedder
    )

    now = datetime.now()
    lines = [
        f'# Saturation diagnostics: "{args.term}" ({now:%Y-%m-%d %H:%M})',
        "",
        f"Pool: {len(literature)} passages. Observed rule: ARI >= "
        f"{config.get('saturation.ari_threshold')} x "
        f"{config.get('saturation.consecutive_increments')} consecutive; "
        f"observed N*={n_star} (reached={observed['reached']}).",
        "",
        "## A. Same-data ARI across k-means seeds (coarseness confound)",
        "",
        "| n | k | mean pairwise ARI | min | share >= 0.8 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in seed_rows:
        lines.append(
            f"| {r['n']} | {r['k']} | {r['mean_ari']} | {r['min_ari']} "
            f"| {r['share_above_threshold']:.0%} |"
        )
    lines += [
        "",
        "If ARI >= threshold is common at k=2 and rare at fine k on IDENTICAL",
        "data, the plateau rule conflates stability with coarseness.",
        "",
        "## B. Ordering",
        "",
        "| ordering | reached | N* | k at N* |",
        "|---|---|---:|---:|",
    ]
    for r in ordering_rows:
        lines.append(
            f"| {r['ordering']} | {r['reached']} | {r['n_selected']} | {r['k_at_selection']} |"
        )
    lines += [
        "",
        f"Anchor coverage at N*={n_star}: selected passages' mean cosine to the "
        f"term anchor {coverage['selected_mean_cos']} vs excluded "
        f"{coverage['excluded_mean_cos']} (the gap measures how term-central "
        "the clustered sample is).",
        "",
        "## C. Where would other rules have fired? (one no-stop sweep)",
        "",
        "| threshold | consecutive | N* (None = never) |",
        "|---:|---:|---:|",
    ]
    for r in grid:
        lines.append(f"| {r['threshold']} | {r['consecutive']} | {r['n_star']} |")
    lines += [
        "",
        "## D. Outcome sensitivity to dropping m sources",
        "",
        f"Baseline: reached={baseline_reached}, N*={baseline_n}",
        "",
        "| sources dropped | reps | outcome flips | N* per rep |",
        "|---:|---:|---:|---|",
    ]
    for r in perturb_rows:
        lines.append(
            f"| {r['sources_dropped']} | {r['reps']} | {r['outcome_flips']} | {r['n_stars']} |"
        )
    report = "\n".join(lines) + "\n"
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        # Windows consoles often use a legacy codepage that cannot encode
        # corpus text. Degrade the console copy rather than die before the
        # report reaches disk (the file below is UTF-8 and unaffected).
        import sys

        encoding = sys.stdout.encoding or "utf-8"
        print("\n" + report.encode(encoding, errors="replace").decode(encoding))

    out_dir = config.path("paths.output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"satdiag_{safe_term_slug(args.term)}_{now:%Y%m%d_%H%M%S}"
    (out_dir / f"{stem}.md").write_text(report, encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "term": args.term,
                "generated": now.isoformat(timespec="seconds"),
                "pool": len(literature),
                "observed": observed,
                "anchor_coverage": coverage,
                "probe_a_seed_ari": seed_rows,
                "probe_b_ordering": ordering_rows,
                "probe_c_rule_grid": grid,
                "probe_c_history": history,
                "probe_d_perturbation": perturb_rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[satdiag] written: {out_dir / (stem + '.md')} (+ .json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
