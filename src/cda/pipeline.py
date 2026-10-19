"""The end-to-end analysis for one term (the paper's three stages).

1. Passage extraction — literal term retrieval, term-sentence windows, MyWork
   resolution, draft inclusion, and (when configured) the acquisition-date
   window that fixes the corpus to a point in time.
2. Divergence mapping — saturation-based literature subsampling, then
   spherical k-means: by default over the two corpora *pooled* (joint
   clustering, so cluster ids mean the same thing on both sides), or per
   corpus independently, which reproduces the published pipeline.
3. Register assignment + divergence analysis — the LLM call (live or mock),
   computed divergence flags, the cluster–register audit, and the
   timestamped markdown report.

``dry_run=True`` stops after assembling the prompt and prints it in full —
nothing is sent. This is the inspect-before-you-send mechanism the
documentation points to.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from cda.clustering.joint import cluster_joint, corpus_views
from cda.clustering.saturation import rank_literature, saturation_select
from cda.clustering.spherical import cluster_passages
from cda.config import Config, ConfigError
from cda.corpus.drafts import load_drafts, normalize_drafts
from cda.corpus.embeddings import Embedder, get_embedder
from cda.corpus.ingest import get_collection
from cda.corpus.timeframe import describe_window, filter_by_date_added, timeframe_from_config
from cda.divergence.flags import compute_flags, compute_joint_flags
from cda.divergence.report import render_report, write_outputs
from cda.extraction.draft_passages import extract_draft_passages
from cda.extraction.mywork import mywork_classifier
from cda.extraction.retrieval import build_passages, keyword_hits
from cda.llm.mock import RecordingProvider
from cda.llm.providers import fixture_model, get_provider
from cda.provenance import build_manifest, input_snapshot, sample_snapshot, seal_run
from cda.registers.assemble import build_user_prompt, configured_theme_lines
from cda.registers.audit import audit_report
from cda.registers.grounding import check_quotations
from cda.registers.prompt_loader import load_system_prompt
from cda.types import ClusterSummary, JointClustering, Passage

JOINT = "joint"
INDEPENDENT = "independent"


def _passage_sort_key(passage: Passage) -> tuple[float, str, str, str]:
    """Semantic relevance first, then year/source, chunk_id as total-order tiebreaker.

    Clustering, representative selection — and thus the assembled prompt and
    fixture fingerprints — must not depend on database return order. Defined
    once so extraction and any reuse sort identically. (Year compares as a
    string, so "unknown" sorts above dated years at equal similarity; only
    totality and stability matter here.)
    """
    return (passage.similarity, passage.year, passage.source, passage.chunk_id)


SILHOUETTE_SUBSAMPLE_FLOOR = 100


def _silhouette_subsample(config: Config) -> int:
    """How many passages the silhouette is evaluated on, never below a usable floor.

    A tiny evaluation sample leaves most candidate partitions with a singleton
    or missing cluster, so every silhouette is undefined and the k sweep
    reports "no valid partition" instead of choosing one.
    """
    value = config.get_int("clustering.silhouette_subsample", 2000)
    if value < SILHOUETTE_SUBSAMPLE_FLOOR:
        raise ConfigError(
            f"clustering.silhouette_subsample must be at least {SILHOUETTE_SUBSAMPLE_FLOOR} "
            f"passages, got {value}. Smaller evaluation samples leave the silhouette "
            "undefined for every candidate k; the default is 2000."
        )
    return value


def _clustering_kwargs(config: Config) -> dict:
    """The k-means knobs shared by both corpora and the saturation sweep."""
    return dict(
        n_restarts=config.get_int("clustering.kmeans_restarts", 5, minimum=1),
        max_iter=config.get_int("clustering.kmeans_max_iter", 80, minimum=1),
        min_cluster_size=config.get_int("clustering.min_cluster_size", 3, minimum=1),
        silhouette_subsample=_silhouette_subsample(config),
        # minimum=0: numpy's default_rng refuses negative seeds, and without
        # the bound here the refusal would surface as a raw ValueError deep
        # inside the k-sweep instead of a config error naming the key.
        seed=config.get_int("clustering.seed", 42, minimum=0),
    )


def representatives_per_cluster(config: Config) -> int:
    """How many sample passages each cluster contributes to the prompt.

    Read through ``get_int`` with a floor of 1: ``0`` would send the LLM
    clusters with no passages at all to read, and a negative value would slice
    representatives off the *end* of the centroid-distance ranking — the
    passages furthest from the cluster centre, which is the opposite of what
    the paper's prompt shows.
    """
    return config.get_int("clustering.representatives_per_cluster", 5, minimum=1)


def resolution_params(config: Config) -> dict:
    """The optional resolution overrides: ``clustering.method`` and ``clustering.k``.

    Both default to the paper's method (silhouette-selected spherical
    k-means). ``method: hierarchical`` swaps in one average-linkage tree cut
    at the chosen resolution; ``k: <int>`` pins the resolution instead of
    letting the silhouette sweep pick it. They apply to the final partition
    (joint, or per-corpus under independent mode) — the saturation sweep's
    internal clustering stays the paper's rule either way.
    """
    clustering = config.section("clustering")
    method = str(clustering.get("method", "kmeans")).strip().lower()
    if method not in ("kmeans", "hierarchical"):
        raise ConfigError(
            f"clustering.method must be 'kmeans' or 'hierarchical', got {method!r}. "
            "'kmeans' is the paper's method; 'hierarchical' builds one "
            "average-linkage tree and cuts it at the chosen resolution. "
            "See config.example.yaml."
        )
    raw_k = clustering.get("k", "auto")
    if isinstance(raw_k, str) and raw_k.strip().lower() == "auto":
        fixed_k = None
    else:
        try:
            fixed_k = int(raw_k)
        except (TypeError, ValueError):
            raise ConfigError(
                f"clustering.k must be 'auto' or an integer >= 2, got {raw_k!r}. "
                "'auto' selects k by silhouette (the paper's method); an integer "
                "fixes the resolution. See config.example.yaml."
            ) from None
        if fixed_k < 2:
            raise ConfigError(
                f"clustering.k must be 'auto' or an integer >= 2, got {fixed_k}. "
                "A one-cluster partition carries no divergence information."
            )
    max_k_params(config)  # validated with the other resolution settings
    return {"method": method, "fixed_k": fixed_k}


def max_k_params(config: Config) -> dict[str, int]:
    """The k-sweep ceilings; their defaults live only in defaults.yaml.

    Each needs at least two clusters to sweep to; a typo like `joint_max_k: 1`
    would otherwise surface as a degenerate "no valid partition" note that
    blames the data instead of the config.
    """
    return {
        "joint": config.get_int("clustering.joint_max_k", minimum=2),
        "researcher": config.get_int("clustering.researcher_max_k", minimum=2),
        "literature": config.get_int("clustering.literature_max_k", minimum=2),
    }


def saturation_scope(config: Config) -> str:
    """The validated ``saturation.scope``: ``"sample"`` or ``"prompt"``.

    A function (rather than an inline check in the saturation step) so
    ``analyze_term`` can validate it up front with the other typo-only
    settings, before the embedding model loads.
    """
    scope = str(config.section("saturation").get("scope", "sample"))
    if scope not in ("sample", "prompt"):
        raise ConfigError(
            f"saturation.scope must be 'sample' or 'prompt', got {scope!r}. "
            "'sample' is the paper's method (subsampling feeds the clustering); "
            "'prompt' clusters the full retrieved pool. See config.example.yaml."
        )
    return scope


def saturation_ranking(config: Config) -> str:
    """The validated ``saturation.ranking``: ``"term"`` or ``"similarity"``.

    Treating anything that is not "similarity" as term mode would let a typo
    (``ranking: trem``) silently swap the paper's external anchor for the
    retrieval scores — a different analysis, reported as if it were the
    paper's.
    """
    ranking = str(config.get("saturation.ranking", "term")).strip().lower()
    if ranking not in ("term", "similarity"):
        raise ConfigError(
            f"saturation.ranking must be 'term' or 'similarity', got {ranking!r}. "
            "'term' is the paper's method (cosine similarity to the target term's "
            "embedding, an external anchor); 'similarity' reuses the retrieval "
            "scores. See config.example.yaml."
        )
    return ranking


def saturation_params(config: Config) -> dict:
    """Validated saturation knobs, checked up front with the other settings.

    ``max_passages < start`` would make the sweep empty and return the whole
    pool under a cap it never applied. That cross-field rule applies only when
    the sweep can run (``enabled: true`` and ``scope: sample``); types and
    spellings are validated either way.
    """
    enabled = config.get_bool("saturation.enabled", True)
    scope = saturation_scope(config)
    ranking = saturation_ranking(config)
    start = config.get_int("saturation.start", minimum=1)
    step = config.get_int("saturation.step", minimum=1)
    max_passages = config.get_int("saturation.max_passages", minimum=1)
    if enabled and scope == "sample" and max_passages < start:
        raise ConfigError(
            f"saturation.max_passages ({max_passages}) is below saturation.start "
            f"({start}); the sweep would have no sizes to try. Raise max_passages "
            "or lower start."
        )
    ari_threshold = config.get_float("saturation.ari_threshold", minimum=0.0, maximum=1.0)
    consecutive = config.get_int("saturation.consecutive_increments", minimum=1)
    return {
        "enabled": enabled,
        "scope": scope,
        "ranking": ranking,
        "start": start,
        "step": step,
        "max_passages": max_passages,
        "ari_threshold": ari_threshold,
        "consecutive_increments": consecutive,
    }


def retrieval_params(config: Config) -> dict:
    """Validated retrieval knobs (typo-only settings, checked up front).

    ``window_mode`` matters most: a misspelling falling through an if/else
    into context mode would silently change the paper's unit of analysis
    (§3.2) with no warning anywhere in the report.
    """
    window_mode = str(config.get("retrieval.window_mode", "sentence")).strip().lower()
    if window_mode not in ("sentence", "context"):
        raise ConfigError(
            f"retrieval.window_mode must be 'sentence' or 'context', got "
            f"{config.get('retrieval.window_mode')!r}. 'sentence' is the paper's "
            "unit of analysis (§3.2); 'context' adds ±1 sentence around each hit."
        )
    min_chars = config.get_int("retrieval.min_window_chars", minimum=0)
    # Floor of 4: the truncation adds a 3-character ellipsis, so a smaller cap
    # would produce windows longer than the cap (see extraction/windows.py).
    max_chars = config.get_int("retrieval.max_window_chars", minimum=0)
    if 0 < max_chars < 4 or (max_chars and max_chars < min_chars):
        raise ConfigError(
            f"retrieval.max_window_chars ({max_chars}) is below "
            f"retrieval.min_window_chars ({min_chars})."
        )
    config.get_int("retrieval.keyword_max_chunks", minimum=0)
    return {
        "window_mode": window_mode,
        "min_window_chars": min_chars,
        "max_window_chars": max_chars,
    }


def flag_params(config: Config) -> dict:
    """Validated divergence-flag thresholds, checked before any paid work.

    The share thresholds are proportions: a percent-style value like ``80``
    can never match a 0–1 share, so every run would quietly report "no
    flags" — reading as a genuine finding of no divergence rather than a
    config mistake.
    """
    researcher = config.get_float("flags.researcher_share_threshold", minimum=0.0, maximum=1.0)
    literature = config.get_float("flags.literature_share_threshold", minimum=0.0, maximum=1.0)
    underrepresented = config.get_int("flags.underrepresented_max_researcher_passages", minimum=0)
    return {
        "researcher_share_threshold": researcher,
        "literature_share_threshold": literature,
        "underrepresented_max_researcher_passages": underrepresented,
    }


def clustering_mode(config: Config) -> str:
    """``"joint"`` (default) or ``"independent"`` (the published pipeline)."""
    mode = str(config.get("clustering.mode", default=JOINT)).strip().lower()
    if mode not in (JOINT, INDEPENDENT):
        raise ConfigError(
            f"clustering.mode must be '{JOINT}' or '{INDEPENDENT}', got '{mode}'. "
            f"'{JOINT}' pools the researcher's and the literature's passages into one "
            f"k-means partition (the default); '{INDEPENDENT}' clusters each corpus "
            "separately and cross-assigns, reproducing the paper's pipeline."
        )
    return mode


def _fail_if_window_emptied_literature(counts: dict, window: dict) -> None:
    """Refuse to analyze an empty literature corpus the window just created.

    An empty pool would otherwise produce a report full of "clustering
    unavailable" notes, and the most common cause is mundane and fixable: a
    corpus ingested before "Date Added" was recorded has no dates at all, so
    every passage looks undated and every window drops everything. Say which
    case it is.
    """
    if counts["n_before"] == 0 or counts["n_kept"] > 0:
        return
    if counts["n_undated_dropped"] == counts["n_before"]:
        if window.get("date_source") == "filesystem":
            # The fix below (re-ingest with a Zotero CSV) records *Zotero*
            # dates; it can never produce a filesystem date, so repeating it
            # here would send the user in a circle. Filesystem dates exist
            # only for corpora ingested from PDF files on disk — JSONL-based
            # corpora (the demo's included) have none, ever.
            raise ValueError(
                f"--date-source filesystem is set, but none of the "
                f"{counts['n_before']} literature passages carry a filesystem "
                "date. Filesystem dates are recorded when PDFs are ingested "
                "from disk, so a corpus ingested from JSONL records (the demo "
                "corpus included) has none. Use --date-source zotero, or "
                "re-ingest your PDFs with cda ingest --pdf-dir ... --reingest. "
                "Or set timeframe.include_undated: true / --include-undated to "
                "analyze anyway, accepting that the window is then advisory."
            )
        raise ValueError(
            f"An acquisition-date window is set, but none of the "
            f"{counts['n_before']} literature passages carry a 'Date Added', so all "
            "were dropped. Your corpus was most likely ingested before that field "
            "was recorded: re-ingest it once with\n"
            "  cda ingest --pdf-dir ... --zotero-csv ... --reingest\n"
            "(for the demo store, delete demo/_runtime and rerun). Or set "
            "timeframe.include_undated: true / --include-undated to analyze it "
            "anyway, accepting that the window is then advisory."
        )
    raise ValueError(
        f"The acquisition-date window ({describe_window(window)}) excludes all "
        f"{counts['n_before']} literature passages found for this term. Widen the "
        "window, or check that the bounds are the ones you meant."
    )


def extract_passages(
    term: str,
    config: Config,
    *,
    embedder: Embedder,
    collection,
    drafts_only: bool = False,
    draft_texts: dict[str, str] | None = None,
    progress: Callable[[str], None] = print,
) -> tuple[list[Passage], list[Passage], dict]:
    """Stage 1 — literal retrieval, MyWork resolution, timeframe, draft inclusion.

    Returns (researcher_passages, literature_passages, stats), both pools
    sorted deterministically. Exposed as a function so `cda benchmark` can
    cluster the exact same passage sets a real analysis would.
    """
    progress(f"[1/3] extracting passages for '{term}'")
    contains = keyword_hits(
        collection,
        term,
        max_chunks=config.get_int("retrieval.keyword_max_chunks", minimum=0),
        progress=progress,
    )

    researcher_passages, literature_passages, stats = build_passages(
        term, contains, mywork_classifier(config.section("mywork")), config
    )

    # The acquisition-date window is applied to the corpus pools BEFORE drafts
    # join them: drafts are exempt by construction (they are the manuscript
    # under examination, not works acquired during a period), so the exemption
    # is structural rather than a special case inside the filter.
    window = timeframe_from_config(config)
    if window["active"]:
        researcher_passages, researcher_counts = filter_by_date_added(researcher_passages, window)
        literature_passages, literature_counts = filter_by_date_added(literature_passages, window)
        stats["timeframe"] = {
            "window": window,
            "researcher": researcher_counts,
            "literature": literature_counts,
        }
        stats["researcher_passages"] = len(researcher_passages)
        stats["literature_passages"] = len(literature_passages)
        progress(
            f"      timeframe: {literature_counts['n_kept']} of "
            f"{literature_counts['n_before']} literature passages inside the window"
        )
        _fail_if_window_emptied_literature(literature_counts, window)

    # drafts_only: scope the researcher corpus to the supplied drafts alone by
    # dropping the mywork-tagged corpus chunks — analyze one manuscript rather
    # than the researcher's whole indexed program. Literature is untouched.
    if drafts_only:
        stats["corpus_researcher_passages_dropped"] = len(researcher_passages)
        researcher_passages = []
        stats["researcher_passages"] = 0

    # Researcher drafts are the fifth MyWork signal: always the researcher's.
    if draft_texts is None:
        draft_texts = load_drafts(config.path("paths.drafts_dir"), progress=progress)
    draft_texts = normalize_drafts(draft_texts)
    stats["input_snapshot"] = input_snapshot(contains, draft_texts)
    if draft_texts:
        draft_passages = extract_draft_passages(term, draft_texts, embedder, config)
        researcher_passages.extend(draft_passages)
        stats["draft_passages"] = len(draft_passages)
        stats["researcher_passages"] = len(researcher_passages)

    researcher_passages.sort(key=_passage_sort_key, reverse=True)
    literature_passages.sort(key=_passage_sort_key, reverse=True)
    return researcher_passages, literature_passages, stats


def select_literature_by_saturation(
    term: str,
    literature_passages: list[Passage],
    config: Config,
    *,
    embedder: Embedder,
) -> tuple[list[Passage], dict, bool]:
    """Stage 2a — saturation-based subsampling of the literature corpus.

    Ranks candidates by similarity to the term embedding (external anchor)
    and grows the sample until cluster structure stabilizes (ARI plateau).
    Returns (selected_passages, saturation_info, ran); ``ran`` is True only
    when the sweep executed and is also recorded as ``saturation_info["ran"]``.
    ``saturation_info`` always carries ``n_candidates`` and ``n_selected``.

    The returned list is always in the canonical ``_passage_sort_key`` order,
    never the sweep's ranking order: k-means seeds centroids by array
    position, so the clustering must depend only on *which* passages were
    selected, not on how the selection was made.
    """
    params = saturation_params(config)
    n_candidates = len(literature_passages)
    saturation_info: dict = {
        "enabled": params["enabled"],
        "scope": params["scope"],
        "ranking": params["ranking"],
        "n_candidates": n_candidates,
        "n_selected": n_candidates,
    }
    ran = False

    if not params["enabled"]:
        saturation_info["reason"] = "saturation disabled: full pool clustered"
    elif params["scope"] == "prompt":
        # The full retrieved pool feeds the clustering, the flags, and the
        # silhouettes; the prompt only ever carries k x representatives_per_cluster
        # passages. The paper's reported runs used scope "sample".
        saturation_info["reason"] = "scope=prompt: full pool clustered; no subsampling applied"
    elif n_candidates < params["start"]:
        saturation_info.update(
            {
                "saturation_reached": False,
                "reason": "pool below sweep start; selection skipped",
            }
        )
    else:
        ranked = rank_literature(literature_passages, term, embedder, mode=params["ranking"])
        literature_passages, sweep_info = saturation_select(
            ranked,
            start=params["start"],
            step=params["step"],
            max_passages=params["max_passages"],
            ari_threshold=params["ari_threshold"],
            consecutive_increments=params["consecutive_increments"],
            max_k=max_k_params(config)["literature"],
            **_clustering_kwargs(config),
        )
        saturation_info.update(sweep_info)
        saturation_info["max_passages"] = params["max_passages"]
        ran = True
    saturation_info["ran"] = ran
    literature_passages.sort(key=_passage_sort_key, reverse=True)
    return literature_passages, saturation_info, ran


def _write_passage_map(
    *,
    term,
    researcher_passages,
    literature_passages,
    joint,
    report_file,
    config,
    progress,
):
    """Draw the passage map beside the report, when the config asks for it.

    A drawing failure must never cost a paid analysis: the report is already
    on disk by the time this runs, so a missing matplotlib or a backend
    problem is reported and stepped over rather than raised.
    """
    if not config.get_bool("figure.enabled", False):
        return None
    from cda.divergence.figure import draw_passage_map

    path = Path(report_file).with_suffix(".png")
    try:
        figure_path = draw_passage_map(
            term=term,
            researcher_passages=researcher_passages,
            literature_passages=literature_passages,
            joint=joint,
            path=path,
            dpi=config.get_int("figure.dpi", 300, minimum=1),
        )
    except Exception as exc:  # noqa: BLE001 - the report must survive this
        progress(f"[warn] the passage map could not be drawn ({exc})")
        return None
    if figure_path is None:
        # draw_passage_map declines when fewer than 3 passages carry
        # embeddings — a two-point plot shows no structure. Silence here
        # would read as a failure; say why there is no map and that nothing
        # else is missing.
        progress(
            "[note] passage map skipped: fewer than 3 passages carry embeddings, "
            "too few to draw a map. The report itself is complete and unaffected."
        )
    return figure_path


def map_divergence(
    term: str,
    researcher_passages: list[Passage],
    literature_passages: list[Passage],
    stats: dict,
    config: Config,
    *,
    embedder: Embedder,
    progress: Callable[[str], None] = print,
) -> tuple[ClusterSummary, ClusterSummary, list[Passage], dict, JointClustering | None]:
    """Stage 2 — saturation-subsample the literature, then cluster.

    Returns (researcher_summary, literature_summary, literature_passages,
    saturation_info, joint). The literature pool is returned because saturation
    may have subsampled it; ``joint`` is the shared partition under joint mode
    and ``None`` under independent mode. ``stats`` is updated in place.

    Saturation runs before clustering in both modes: it bounds how much
    literature the prompt can carry, which is a question about the prompt
    budget and not about how the passages are partitioned.
    """
    mode = clustering_mode(config)
    progress(
        f"[2/3] {mode} clustering of {len(researcher_passages)} researcher / "
        f"{len(literature_passages)} literature passages"
    )
    representatives = representatives_per_cluster(config)

    literature_passages, saturation_info, ran = select_literature_by_saturation(
        term, literature_passages, config, embedder=embedder
    )
    if ran:
        stats["literature_passages_before_saturation"] = saturation_info["n_candidates"]
        stats["literature_passages"] = len(literature_passages)
        progress(
            f"      saturation: N*={saturation_info['n_selected']} of "
            f"{saturation_info['n_candidates']} "
            f"(reached={saturation_info['saturation_reached']})"
        )

    resolution = resolution_params(config)
    max_k = max_k_params(config)
    if mode == JOINT:
        joint = cluster_joint(
            researcher_passages,
            literature_passages,
            embedder,
            max_k=max_k["joint"],
            n_representatives=representatives,
            **_clustering_kwargs(config),
            **resolution,
        )
        researcher_summary, literature_summary = corpus_views(joint)
        progress(
            f"      joint k={joint.n_clusters}, silhouette={joint.silhouette:.3f}; "
            f"researcher passages present in "
            f"{sum(1 for c in joint.clusters if c['n_researcher'])} of "
            f"{joint.n_clusters} shared clusters"
        )
        return researcher_summary, literature_summary, literature_passages, saturation_info, joint

    researcher_summary = cluster_passages(
        researcher_passages,
        embedder,
        label="researcher",
        max_k=max_k["researcher"],
        n_representatives=representatives,
        **_clustering_kwargs(config),
        **resolution,
    )
    literature_summary = cluster_passages(
        literature_passages,
        embedder,
        label="literature",
        max_k=max_k["literature"],
        n_representatives=representatives,
        **_clustering_kwargs(config),
        **resolution,
    )
    return researcher_summary, literature_summary, literature_passages, saturation_info, None


def run_stability_check(
    researcher_passages: list[Passage],
    literature_passages: list[Passage],
    config: Config,
    *,
    mode: str,
    embedder: Embedder,
    expected_k: int | None = None,
    expected_researcher_k: int | None = None,
    expected_literature_k: int | None = None,
    progress: Callable[[str], None] = print,
) -> dict:
    """LLM-free order-sensitivity check, folded into every live report.

    Re-clusters the identical, already-selected passage set under a few
    independently shuffled orderings (``clustering.stability_check.trials``,
    31 by default, including the canonical run itself) and summarizes how often
    the chosen k matches across them. A canonical run is fully reproducible
    (same passages, same order, same k, every time) — this measures the
    separate question of whether that k is *robust*, i.e. whether a
    differently-ordered run of the identical passages would plausibly have
    landed on the same structure. See
    ``benchmarks/clustering_order_stability.py`` for a deeper, standalone
    version of this same check (many more trials, full per-trial table).

    The ``expected_*`` kwargs are the k values this run's real clustering
    produced (joint mode: ``expected_k``; independent mode: the per-corpus
    pair). Trial 0 re-clusters the canonical order, so it must land on the
    same k — the summaries carry that comparison as ``canonical_reproduced``
    and a False prints a warning, because it means the stability figures
    cannot be trusted.

    Returns ``{"enabled": False}`` when the config disables the check or a
    trial count under 2 makes it meaningless; otherwise the ``summarize()``
    shape from ``clustering/order_stability.py``, with ``"enabled": True``.
    """
    from cda.clustering.order_stability import run_trials, summarize

    trials = config.get_int("clustering.stability_check.trials", 5, minimum=0)
    if not config.get_bool("clustering.stability_check.enabled", True) or trials < 2:
        return {"enabled": False, "trials": trials}
    resolution = resolution_params(config)
    if resolution["method"] != "kmeans" or resolution["fixed_k"] is not None:
        # The check measures how the silhouette k-selection reacts to passage
        # order. A fixed k cannot vary, and the hierarchical tree is
        # order-independent by construction, so the statistic would be
        # vacuous either way.
        return {
            "enabled": False,
            "trials": trials,
            "reason": (
                "clustering.method/clustering.k override the paper's "
                "silhouette k-selection, which is what this check measures"
            ),
        }

    max_k = max_k_params(config)
    results = run_trials(
        researcher_passages,
        literature_passages,
        embedder,
        mode=mode,
        trials=trials,
        seed_base=config.get_int("clustering.stability_check.seed_base", 0, minimum=0),
        n_representatives=representatives_per_cluster(config),
        clustering_kwargs=_clustering_kwargs(config),
        joint_max_k=max_k["joint"],
        researcher_max_k=max_k["researcher"],
        literature_max_k=max_k["literature"],
    )
    summary = summarize(
        results,
        mode,
        expected_k=expected_k,
        expected_researcher_k=expected_researcher_k,
        expected_literature_k=expected_literature_k,
    )
    summary["enabled"] = True
    if mode == JOINT:
        progress(
            f"      stability: k={summary['canonical_k']} matched "
            f"{summary['k_match_rate']:.0%} of {summary['shuffled_trials']} shuffled orderings"
        )
        if summary.get("canonical_reproduced") is False:
            progress(
                "      WARNING: re-clustering the canonical order did not reproduce "
                f"the run's own k={expected_k} — the clustering is not deterministic "
                "for this input and the stability figures above are unreliable"
            )
    else:
        shuffled = summary["researcher"]["shuffled_trials"]
        progress(
            "      stability: researcher k matched "
            f"{summary['researcher']['k_match_rate']:.0%}, literature k matched "
            f"{summary['literature']['k_match_rate']:.0%} of {shuffled} shuffled orderings"
        )
        for corpus_label in ("researcher", "literature"):
            if summary[corpus_label].get("canonical_reproduced") is False:
                progress(
                    f"      WARNING: re-clustering the canonical {corpus_label} order did "
                    "not reproduce the run's own k — the clustering is not deterministic "
                    "for this input and the stability figures above are unreliable"
                )
    return summary


def _print_prompt_block(text: str) -> None:
    """print() that survives a console that cannot encode the prompt.

    The assembled prompt carries arbitrary corpus text. On Windows,
    *redirected* stdout defaults to cp1252, so printing a prompt with (say)
    curly quotes would die with a UnicodeEncodeError rewrapped as a bare
    "charmap codec" message — ending the dry run after the expensive work.
    Degrade the console copy instead: the on-disk ``*_prompt.md`` file is
    UTF-8 and authoritative either way.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))
        print(
            f"[note] some characters above could not be shown in this console's "
            f"encoding ({encoding}) and were replaced; the saved *_prompt.md "
            "file is UTF-8 and authoritative."
        )


def _print_dry_run(system_prompt: str, user_prompt: str) -> None:
    """Print the fully assembled prompt for the inspect-before-you-send flow."""
    print("\n" + "=" * 72)
    print("SYSTEM PROMPT")
    print("=" * 72)
    _print_prompt_block(system_prompt)
    print("\n" + "=" * 72)
    print("USER PROMPT")
    print("=" * 72)
    _print_prompt_block(user_prompt)
    print("=" * 72)
    print("DRY RUN: this is the complete prompt; nothing was sent to any API.")


def validated_settings(config: Config) -> dict:
    """Every setting that can only be wrong by a typo, validated before any work.

    A mistyped date, clustering mode, method/k override, retrieval window,
    flag threshold, saturation, LLM setting or theme entry must not surface after the
    corpus has been opened and the embedding model loaded, or silently change
    what is analyzed.
    """
    return {
        "mode": clustering_mode(config),
        "resolution": resolution_params(config),
        "representatives": representatives_per_cluster(config),
        "saturation": saturation_params(config),
        "retrieval": retrieval_params(config),
        "flags": flag_params(config),
        "clustering": _clustering_kwargs(config),
        "timeframe": timeframe_from_config(config),
        "max_tokens": config.get_int("llm.max_tokens", minimum=1),
        "temperature": config.get_float("llm.temperature", minimum=0.0),
        "theme_lines": configured_theme_lines(config),
    }


def _prepare_output_dir(config: Config) -> Path:
    """The output directory, created and checked for writability up front.

    An --out-dir pointing at an existing file would otherwise surface only
    after the extraction and clustering stages, and after the paid LLM call.
    """
    out_dir = config.path("paths.output_dir")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(
            f"Cannot use {out_dir} as the output directory ({exc}). "
            "--out-dir / paths.output_dir must name a directory (not an "
            "existing file) that you have permission to create."
        ) from exc
    if not os.access(out_dir, os.W_OK):
        raise ConfigError(
            f"The output directory {out_dir} is not writable by this user. "
            "Fix its permissions or point --out-dir somewhere writable."
        )
    return out_dir


def _build_provider(
    config: Config,
    *,
    mock: bool,
    record_fixtures: bool,
    overwrite_fixtures: bool,
    fixtures_dir: str | Path | None,
):
    """The analysis provider, built before any work so a missing key fails fast."""
    provider = get_provider(config, mock=mock, fixtures_dir=fixtures_dir)
    if not record_fixtures or mock:
        return provider
    if fixtures_dir is None:
        raise ValueError("--record-fixtures requires --fixtures-dir")
    return RecordingProvider(
        provider,
        Path(fixtures_dir),
        fingerprint_model=fixture_model(str(config.get("llm.model"))),
        overwrite=overwrite_fixtures,
    )


def _too_few_researcher_passages(
    term: str, researcher_passages: list[Passage], stats: dict, config: Config
) -> dict | None:
    """The 'skipped' result when the researcher corpus is below the minimum, else None."""
    minimum = config.get_int("analysis.min_researcher_passages", minimum=0)
    if len(researcher_passages) >= minimum:
        return None
    return {
        "term": term,
        "status": "skipped",
        "reason": (
            f"Only {len(researcher_passages)} researcher passages found; minimum is "
            f"{minimum}. Add drafts to {config.path('paths.drafts_dir')}, tag your "
            "papers 'mywork' in Zotero and re-ingest, or add filename patterns / "
            "publication titles to the mywork section of your config."
        ),
        "stats": stats,
    }


def _call_llm(provider, settings: dict, system_prompt: str, user_prompt: str, progress) -> str:
    progress(f"[3/3] register assignment via {provider.name} ({provider.model})")
    return provider.complete(
        system=system_prompt,
        user=user_prompt,
        max_tokens=settings["max_tokens"],
        temperature=settings["temperature"],
    )


def _postprocess_reply(
    llm_body: str,
    user_prompt: str,
    settings: dict,
    *,
    researcher_passages: list[Passage],
    literature_passages: list[Passage],
    researcher_summary: ClusterSummary,
    literature_summary: ClusterSummary,
    joint: JointClustering | None,
) -> tuple[dict, dict, dict]:
    """Deterministic checks of the reply: (flags, audit, grounding).

    Under joint clustering the flags read the shared partition's membership
    directly — the paper's ">80% of a cluster's instances" as written.
    """
    flags = (
        compute_joint_flags(joint, settings["flags"])
        if joint is not None
        else compute_flags(researcher_passages, literature_passages, settings["flags"])
    )
    audit = audit_report(llm_body, researcher_summary.n_clusters, literature_summary.n_clusters)
    grounding = check_quotations(llm_body, user_prompt)
    return flags, audit, grounding


def _cluster_result(summary: ClusterSummary) -> dict:
    return {
        "k": summary.n_clusters,
        "silhouette": summary.silhouette,
        "n": summary.n_total_passages,
    }


def analyze_term(
    term: str,
    config: Config,
    *,
    mock: bool = False,
    dry_run: bool = False,
    record_fixtures: bool = False,
    overwrite_fixtures: bool = False,
    fixtures_dir: str | Path | None = None,
    embedder: Embedder | None = None,
    collection=None,
    return_passages: bool = False,
    drafts_only: bool = False,
    draft_texts: dict[str, str] | None = None,
    persist: bool = True,
    progress: Callable[[str], None] = print,
) -> dict:
    """Run CDA for one term — the paper's three stages, in order.

    Stage 1 extracts passages, Stage 2 maps divergence by clustering, Stage 3
    assigns registers with the LLM and writes the report. ``dry_run=True``
    stops after assembling the prompt. ``return_passages=True`` also returns
    the clustered passages for the GUI visualizations (it changes nothing
    about the analysis). ``overwrite_fixtures`` lets ``record_fixtures``
    replace an existing replayable fixture for the same prompt.
    """
    if not persist and not dry_run:
        raise ValueError("persist=False is supported only for a dry-run preview.")
    term = term.strip()
    if not term:
        return {"term": term, "status": "skipped", "reason": "empty term"}

    settings = validated_settings(config)
    mode = settings["mode"]
    out_dir = _prepare_output_dir(config) if persist else None
    provider = (
        None
        if dry_run
        else _build_provider(
            config,
            mock=mock,
            record_fixtures=record_fixtures,
            overwrite_fixtures=overwrite_fixtures,
            fixtures_dir=fixtures_dir,
        )
    )
    embedder = embedder or get_embedder(config)
    collection = collection if collection is not None else get_collection(config)

    # ---- Stage 1: passage extraction ------------------------------------
    researcher_passages, literature_passages, stats = extract_passages(
        term,
        config,
        embedder=embedder,
        collection=collection,
        drafts_only=drafts_only,
        draft_texts=draft_texts,
        progress=progress,
    )
    skipped = _too_few_researcher_passages(term, researcher_passages, stats, config)
    if skipped:
        return skipped
    extracted_snapshot = sample_snapshot(researcher_passages, literature_passages)

    # ---- Stage 2: divergence mapping -------------------------------------
    (
        researcher_summary,
        literature_summary,
        literature_passages,
        saturation_info,
        joint,
    ) = map_divergence(
        term,
        researcher_passages,
        literature_passages,
        stats,
        config,
        embedder=embedder,
        progress=progress,
    )
    # LLM-free but several extra clusterings, so the fast dry-run preview skips it.
    stability = (
        None
        if dry_run
        else run_stability_check(
            researcher_passages,
            literature_passages,
            config,
            mode=mode,
            embedder=embedder,
            expected_k=joint.n_clusters if joint is not None else None,
            expected_researcher_k=researcher_summary.n_clusters,
            expected_literature_k=literature_summary.n_clusters,
            progress=progress,
        )
    )

    # ---- Stage 3: register assignment + divergence analysis --------------
    system_prompt = load_system_prompt(config)
    user_prompt = build_user_prompt(
        term,
        researcher_summary,
        literature_summary,
        saturation_info,
        config,
        joint=joint,
        timeframe=stats.get("timeframe"),
        researcher_passages=researcher_passages,
    )
    now = datetime.now()
    diagnostics = {
        "term": term,
        "clustering_mode": mode,
        "researcher": researcher_summary.diagnostics,
        "literature": literature_summary.diagnostics,
        "saturation": saturation_info,
        "timeframe": stats.get("timeframe"),
        "joint": joint.diagnostics if joint is not None else None,
        "stability": stability,
    }

    if not persist:
        # Public pasted-text preview: no private text on disk or stdout.
        return {
            "term": term,
            "status": "dry-run",
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "diagnostics": diagnostics,
            "stats": stats,
        }

    provenance = build_manifest(
        term=term,
        config=config,
        embedder=embedder,
        inputs=stats.get("input_snapshot"),
        extracted=extracted_snapshot,
        researcher=researcher_passages,
        literature=literature_passages,
        stats=stats,
        diagnostics=diagnostics,
        mock=mock,
        dry_run=dry_run,
        now=now,
    )
    run_files = dict(
        out_dir=out_dir,
        term=term,
        prompt_text=user_prompt,
        diagnostics=diagnostics,
        now=now,
        system_prompt_text=system_prompt,
        provenance=provenance,
    )
    # The prompt, system prompt, diagnostics, and manifest are on disk before
    # the LLM call, so a failed or interrupted call loses none of them.
    written = write_outputs(**run_files, report_text=None)
    run_files["base"] = written["base"]

    if dry_run:
        written["receipt_file"] = seal_run(written, status="dry-run")
        _print_dry_run(system_prompt, user_prompt)
        return {
            "term": term,
            "status": "dry-run",
            "clustering_mode": mode,
            "stats": stats,
            "saturation": saturation_info,
            "timeframe": stats.get("timeframe"),
            **written,
        }

    llm_body = _call_llm(provider, settings, system_prompt, user_prompt, progress)
    run_files["llm_response_text"] = llm_body
    write_outputs(**run_files, report_text=None)

    flags, audit, grounding = _postprocess_reply(
        llm_body,
        user_prompt,
        settings,
        researcher_passages=researcher_passages,
        literature_passages=literature_passages,
        researcher_summary=researcher_summary,
        literature_summary=literature_summary,
        joint=joint,
    )
    report_text = render_report(
        term=term,
        llm_body=llm_body,
        model=provider.model,
        stats=stats,
        researcher=researcher_summary,
        literature=literature_summary,
        saturation=saturation_info,
        flags=flags,
        audit=audit,
        grounding=grounding,
        config=config,
        now=now,
        joint=joint,
        stability=stability,
    )
    written = write_outputs(**run_files, report_text=report_text)
    progress(f"[done] report: {written['report_file']}")

    figure_path = _write_passage_map(
        term=term,
        researcher_passages=researcher_passages,
        literature_passages=literature_passages,
        joint=joint,
        report_file=written["report_file"],
        config=config,
        progress=progress,
    )
    if figure_path:
        written["figure_file"] = str(figure_path)
        progress(f"[done] passage map: {figure_path}")
    written["receipt_file"] = seal_run(written, status="complete")
    passages_out = (
        {"passages": {"researcher": researcher_passages, "literature": literature_passages}}
        if return_passages
        else {}
    )
    return {
        "term": term,
        "status": "ok",
        "model": provider.model,
        "clustering_mode": mode,
        "stats": stats,
        "saturation": saturation_info,
        "timeframe": stats.get("timeframe"),
        "flags": flags,
        "audit": audit,
        "grounding": grounding,
        # The GUI's joint composition chart reads this; None in independent mode.
        "joint": joint,
        "stability": stability,
        **passages_out,
        "researcher_cluster": _cluster_result(researcher_summary),
        "literature_cluster": _cluster_result(literature_summary),
        **written,
    }
