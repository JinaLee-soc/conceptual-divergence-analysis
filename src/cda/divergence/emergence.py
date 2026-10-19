"""``cda emergence`` — orchestrate, render, and write the emergence report.

EXPERIMENTAL, post-paper, LLM-free: reuses the pipeline's Stage-1 passage
extraction and the seeded clustering, then hands the literature side to
``cda.clustering.emergence.track_emergence``. Because no provider is ever
constructed, the command behaves identically with and without an API key.

Outputs follow the drift-report convention: ``emergence_<term>_<timestamp>.md``
plus ``emergence_<term>_<timestamp>_diagnostics.json``, sharing one
second-precision timestamp so reruns never overwrite earlier reports.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from cda import __version__
from cda.clustering.emergence import track_emergence
from cda.config import Config
from cda.corpus.embeddings import get_embedder
from cda.corpus.ingest import get_collection
from cda.corpus.timeframe import DATE_SOURCES, describe_timeframe, timeframe_from_config
from cda.extraction.windows import safe_term_slug
from cda.outputs import unique_run_basename, write_text_atomic
from cda.pipeline import _clustering_kwargs, extract_passages, representatives_per_cluster


def run_emergence(
    term: str,
    config: Config,
    *,
    embedder=None,
    collection=None,
    progress: Callable[[str], None] = print,
) -> dict:
    """Track how the literature clusters for ``term`` emerged over acquisition time.

    Runs on the FULL literature pool — no saturation subsampling. Saturation
    bounds a prompt budget; emergence describes the library itself, so
    truncating it would truncate the very history being measured.
    """
    term = term.strip()
    if not term:
        raise ValueError("--term is empty. Pass the term whose emergence to track.")
    window = timeframe_from_config(config)  # fail fast on a mistyped window bound
    embedder = embedder or get_embedder(config)
    collection = collection if collection is not None else get_collection(config)

    _researcher, literature, stats = extract_passages(
        term, config, embedder=embedder, collection=collection, progress=progress
    )
    if not literature:
        window_hint = (
            " The acquisition-date window may have removed them all: "
            f"{describe_timeframe(stats['timeframe'])}"
            if (stats.get("timeframe", {}).get("window") or {}).get("active")
            else ""
        )
        raise ValueError(
            f"No literature passages found for '{term}'. "
            f"Check the term spelling, or run cda ingest first.{window_hint}"
        )

    # The timeline must read the same clock the window filter reads
    # (timeframe.date_source / --date-source): filtering on the filesystem
    # clock while ordering by Zotero's would silently mix two histories in
    # one report — precisely the case the flag exists for (a rebuilt Zotero
    # library collapses every Zotero date onto the rebuild day). The value
    # comes from the validated window, not the raw config: the validator
    # normalizes case/whitespace, so re-reading the raw key here would crash
    # on a value like "Zotero" that just passed validation.
    date_source = window["date_source"]
    date_key = DATE_SOURCES[date_source]
    progress(
        f"[emergence] tracking cluster emergence over {len(literature)} literature "
        f"passages ({date_source} acquisition dates)"
    )
    emergence = track_emergence(
        literature,
        embedder,
        start=config.get_int("emergence.start", 20, minimum=1),
        step=config.get_int("emergence.step", 10, minimum=1),
        match_threshold=config.get_float(
            "emergence.match_threshold", 0.80, minimum=0.0, maximum=1.0
        ),
        max_k=config.get_int("clustering.literature_max_k", minimum=2),
        date_key=date_key,
        n_representatives=representatives_per_cluster(config),
        **_clustering_kwargs(config),
    )
    emergence["term"] = term
    emergence["date_source"] = date_source

    if emergence["coverage"]["n_dated"] == 0:
        if date_source == "filesystem":
            raise ValueError(
                "No passages carry a filesystem acquisition date, so there is no "
                "timeline to analyze under --date-source filesystem. Filesystem "
                "dates are recorded when PDFs are ingested from disk — a corpus "
                "ingested from JSONL records (the demo corpus included) has none. "
                "Use --date-source zotero, or re-ingest your PDFs: "
                "cda ingest --pdf-dir ... --reingest."
            )
        raise ValueError(
            "No passages carry a 'Date Added' timestamp, so there is no "
            "acquisition timeline to analyze. Re-ingest with your Zotero CSV: "
            "cda ingest --pdf-dir ... --zotero-csv ... --reingest. For the demo "
            "store, delete demo/_runtime and rerun this command."
        )

    now = datetime.now()
    report_text = render_emergence_report(
        term=term,
        emergence=emergence,
        stats=stats,
        config=config,
        date_source=date_source,
        now=now,
    )
    written = write_emergence_outputs(
        out_dir=config.path("paths.output_dir"),
        term=term,
        report_text=report_text,
        diagnostics=emergence,
        now=now,
    )
    progress(f"[emergence] report: {written['report_file']}")
    return {"status": emergence["status"], "emergence": emergence, **written}


def _timeline_table(snapshots: list[dict]) -> str:
    lines = [
        "| N passages | as of | k | silhouette | ARI vs previous |",
        "|---:|---|---:|---:|---:|",
    ]
    for record in snapshots:
        if record.get("status") != "ok":
            lines.append(
                f"| {record['n']} | {record['as_of']} | n/a | n/a | ({record['status']}) |"
            )
            continue
        ari = record.get("ari_vs_prev")
        lines.append(
            f"| {record['n']} | {record['as_of']} | {record['k']} "
            f"| {record['silhouette']} | {ari if ari is not None else 'n/a'} |"
        )
    return "\n".join(lines)


def _clusters_section(final: dict) -> str:
    lines = [
        "| final cluster | passages | born at N | born (date) | persistence | mean match cosine |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for cluster in final["clusters"]:
        persistence = cluster["persistence"]
        cosine = cluster["mean_match_cosine"]
        lines.append(
            f"| Cluster {cluster['final_cluster']} | {cluster['n_passages']} "
            f"| {cluster['birth_n']} | {cluster['birth_date']} "
            f"| {persistence if persistence is not None else 'n/a'} "
            f"| {cosine if cosine is not None else 'n/a'} |"
        )
    for cluster in final["clusters"]:
        lines.append(f"\n**Cluster {cluster['final_cluster']}**, most typical passages:")
        for text, source in zip(cluster["representatives"], cluster["sources"], strict=True):
            lines.append(f'- "{text}" (*{source}*)')
    return "\n".join(lines)


def _stability_line(snapshots: list[dict]) -> str:
    aris = [
        record["ari_vs_prev"]
        for record in snapshots
        if record.get("status") == "ok" and record.get("ari_vs_prev") is not None
    ]
    if not aris:
        return "Too few snapshots for a stability reading."
    return (
        f"ARI between consecutive snapshots ranged {min(aris)} to {max(aris)} "
        f"(last: {aris[-1]}). Values near 1 mean the cluster structure was "
        "already settled; low values mark reorganizations of the term's "
        "usage structure as the library grew."
    )


_CLOCK_TEXT = {
    "zotero": (
        'The timeline is Zotero\'s "Date Added": when *you* added each work to '
        "your library (your reading trajectory), **not** when the field wrote "
        "it. Publication years appear in the run diagnostics, never on this axis."
    ),
    "filesystem": (
        "The timeline is each PDF's arrival on disk (filesystem birth or "
        "modification time, `--date-source filesystem`): when the file landed "
        "in your library folder, **not** when the field wrote it or when "
        "Zotero recorded it. Publication years appear in the run diagnostics, "
        "never on this axis."
    ),
}

_UNDATED_REMEDY = {
    "zotero": (
        "\n> ⚠️ Passages without a 'Date Added' are excluded from the "
        "timeline (placing them would fabricate history). To raise "
        "coverage, re-ingest with your Zotero CSV: "
        "`cda ingest --pdf-dir ... --zotero-csv ... --reingest`."
    ),
    "filesystem": (
        "\n> ⚠️ Passages without a filesystem acquisition date are excluded "
        "from the timeline (placing them would fabricate history). Filesystem "
        "dates are recorded when PDFs are ingested from disk; to raise "
        "coverage, re-ingest the PDFs: `cda ingest --pdf-dir ... --reingest`."
    ),
}


def render_emergence_report(
    *,
    term: str,
    emergence: dict,
    stats: dict,
    config: Config,
    date_source: str,
    now: datetime,
) -> str:
    if date_source not in _CLOCK_TEXT:
        raise ValueError(f"date_source must be one of {sorted(_CLOCK_TEXT)}, got {date_source!r}")
    coverage = emergence["coverage"]
    params = emergence["params"]
    header = (
        f"<!-- generated: {now.isoformat()} | cda emergence (EXPERIMENTAL) "
        f"| cda: v{__version__} -->\n"
    )
    sections = [
        header,
        f'# Cluster Emergence: "{term}"',
        "\n**EXPERIMENTAL** (post-paper addition; no LLM involved). Cumulative "
        "snapshots of the literature, ordered by library-acquisition date, are "
        "clustered with the pipeline's seeded spherical k-means; each snapshot's "
        "clusters are matched to the final clustering by centroid cosine "
        f"similarity (threshold {params['match_threshold']}).",
        "\n## What this measures\n",
        _CLOCK_TEXT[date_source],
        "\n## Coverage\n",
        f"- Literature passages for the term: {coverage['n_literature']}",
        f"- With an acquisition date: {coverage['n_dated']} ({coverage['rate'] * 100:.1f}%)",
        f"- Excluded (no date): {coverage['n_undated']}",
    ]
    if coverage["n_undated"]:
        sections.append(_UNDATED_REMEDY[date_source])
    if emergence["status"] in ("skipped", "degraded"):
        sections += ["\n## Result\n", f"**{emergence['status']}**: {emergence['reason']}"]
        return "\n".join(sections) + "\n"

    timeframe = stats.get("timeframe") or {}
    if (timeframe.get("window") or {}).get("active"):
        sections.append(
            "\n> This run was restricted to an acquisition-date window, so the "
            "timeline below ends where the window does: "
            f"{describe_timeframe(timeframe)}"
        )

    sections += [
        "\n## Timeline\n",
        _timeline_table(emergence["snapshots"]),
        "\n## Final clusters and their births\n",
        _clusters_section(emergence["final"]),
        "\n## Structural stability\n",
        _stability_line(emergence["snapshots"]),
        "\n## Run Parameters\n",
        f"- Snapshots: start {params['start']}, step {params['step']} "
        f"| match threshold {params['match_threshold']} "
        f"| max k {params['max_k']} | seed {params['seed']}",
        f"- Embedding: `{config.get('embedding.model')}` "
        "| Full literature pool (no saturation subsampling: emergence "
        "describes the library, not a prompt budget)",
        "",
    ]
    return "\n".join(sections)


def write_emergence_outputs(
    *,
    out_dir: Path,
    term: str,
    report_text: str,
    diagnostics: dict,
    now: datetime,
) -> dict[str, str]:
    """Write the report and diagnostics; return both paths directly (no globbing)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = safe_term_slug(term)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    base = unique_run_basename(
        out_dir, f"emergence_{slug}_{timestamp}", (".md", "_diagnostics.json")
    )

    report_path = out_dir / f"{base}.md"
    write_text_atomic(report_path, report_text)

    diag_path = out_dir / f"{base}_diagnostics.json"
    write_text_atomic(diag_path, json.dumps(diagnostics, indent=2, default=str))

    return {"report_file": str(report_path), "diagnostics_file": str(diag_path)}
