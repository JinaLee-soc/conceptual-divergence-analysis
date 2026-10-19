"""`cda diagnose`: a reflexive corpus-composition report against OpenAlex.

Framing matters here. A researcher-curated corpus is not a biased sample of
"the literature" — curation is constitutive of the method: the corpus encodes
the researcher's audience and reading program (paper §3.1). What a curator
still needs is *visibility*: which venues dominate, which fields the corpus
actually covers, how publication years are distributed, and which large areas
of the declared fields the corpus doesn't touch. This module computes exactly
that and presents it as material for informed curation decisions — never as a
defect score.

Data flow: unique works are read from the local corpus's chunk metadata,
resolved against OpenAlex by DOI (batched, cached, polite pool), aggregated,
and rendered as a timestamped markdown report in the output directory.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable
from datetime import datetime

from cda.config import Config
from cda.corpus.ingest import corpus_lock, get_collection
from cda.corpus.timeframe import (
    DATE_SOURCES,
    describe_window,
    normalize_date_added,
    timeframe_from_config,
)
from cda.corpus.zotero import normalize_doi
from cda.diagnose.openalex import (
    OpenAlexCache,
    list_fields,
    top_topics_for_field,
    works_by_dois,
)
from cda.outputs import unique_run_basename, write_text_atomic

# --- corpus metadata ---------------------------------------------------------


def corpus_works(collection, window: dict | None = None) -> list[dict]:
    """One record per distinct source document in the corpus.

    ``window`` is an acquisition-date window from
    ``timeframe.timeframe_from_config``. When it is active, works acquired
    outside it are dropped, so the report describes the same corpus an
    ``--added-to`` analysis ran against rather than the whole store. Every
    chunk of one PDF carries that PDF's acquisition date, so filtering at the
    work level and at the chunk level agree.
    """
    window = window or {}
    active = bool(window.get("active"))
    date_key = DATE_SOURCES[window.get("date_source", "zotero")]
    added_from, added_to = window.get("added_from"), window.get("added_to")
    include_undated = bool(window.get("include_undated", False))
    works: dict[str, dict] = {}
    offset = 0
    while True:
        page = collection.get(include=["metadatas"], limit=1000, offset=offset)
        metadatas = page.get("metadatas") or []
        if not metadatas:
            break
        for metadata in metadatas:
            metadata = metadata or {}
            filename = str(metadata.get("pdf_filename") or "")
            # Distinct-work identity is the unique relative path: basenames can
            # collide across subfolders (a/paper.pdf, b/paper.pdf), and keying on
            # the basename would merge them into one work and skew every count.
            key = str(metadata.get("pdf_relpath") or filename)
            if not key or key in works:
                continue
            if active:
                date_added = normalize_date_added(str(metadata.get(date_key) or "")) or None
                if date_added is None:
                    if not include_undated:
                        continue
                elif (added_from and date_added < added_from) or (
                    added_to and date_added > added_to
                ):
                    continue
            works[key] = {
                "filename": filename or key,
                "relpath": key,
                "title": str(metadata.get("title") or ""),
                "year": str(metadata.get("year") or ""),
                # The same normalization works_by_dois applies, so the
                # "resolved" membership check compares like with like — a
                # "doi:"-prefixed or dx.doi.org-wrapped field is a fine DOI
                # and must not be misreported as unknown to OpenAlex.
                "doi": normalize_doi(str(metadata.get("doi") or "")),
            }
        offset += len(metadatas)
    return list(works.values())


# --- aggregations (pure functions, unit-tested offline) ----------------------


def venue_concentration(resolved: list[dict], top_n: int = 10) -> dict:
    """Top venues by share plus a Herfindahl–Hirschman concentration index.

    Shares and the HHI are over works with a known venue; works OpenAlex
    lists without one are counted separately, never pooled into a venue.
    """
    venues = Counter()
    n_unknown = 0
    for work in resolved:
        primary = work.get("primary_location") or {}
        source = (primary.get("source") or {}).get("display_name")
        if source:
            venues[source] += 1
        else:
            n_unknown += 1
    total = sum(venues.values())
    top = [(venue, count, count / total) for venue, count in venues.most_common(top_n)]
    return {
        "n_works": total,
        "n_unknown_venue": n_unknown,
        "n_venues": len(venues),
        "top": top,
        "top_share": sum(share for _, _, share in top),
        "hhi": round(sum((count / total) ** 2 for count in venues.values()), 4) if total else None,
    }


def topic_distribution(resolved: list[dict], top_n: int = 15) -> list[tuple[str, str, int]]:
    """(topic name, field name, count) for each work's primary topic."""
    counts: Counter = Counter()
    fields: dict[str, str] = {}
    for work in resolved:
        topic = work.get("primary_topic") or {}
        name = topic.get("display_name")
        if not name:
            continue
        counts[name] += 1
        fields[name] = (topic.get("field") or {}).get("display_name", "")
    return [(name, fields.get(name, ""), count) for name, count in counts.most_common(top_n)]


def work_topic_names(resolved: list[dict]) -> set[str]:
    """Every OpenAlex topic any corpus work carries (primary or secondary)."""
    names = set()
    for work in resolved:
        for topic in [work.get("primary_topic") or {}, *(work.get("topics") or [])]:
            if topic.get("display_name"):
                names.add(topic["display_name"])
    return names


def field_shares(resolved: list[dict]) -> list[tuple[str, int, float]]:
    counts: Counter = Counter()
    for work in resolved:
        field = ((work.get("primary_topic") or {}).get("field") or {}).get("display_name")
        counts[field or "(no field)"] += 1
    total = sum(counts.values())
    return [(field, count, count / total) for field, count in counts.most_common()] if total else []


def temporal_profile(resolved: list[dict]) -> dict:
    years = sorted(
        int(work["publication_year"])
        for work in resolved
        if str(work.get("publication_year") or "").isdigit()
    )
    if not years:
        return {"n_dated": 0}
    by_decade = Counter(f"{(year // 10) * 10}s" for year in years)
    return {
        "n_dated": len(years),
        "min": years[0],
        "median": statistics.median(years),
        "max": years[-1],
        "by_decade": sorted(by_decade.items()),
    }


TOPICS_LISTED = 10  # how many untouched topics the report prints; the count is never truncated


def underrepresented_topics(field_baseline: list[dict], corpus_topic_names: set[str]) -> list[str]:
    """Large topics of a declared field with zero corpus presence.

    Returns every one of them. The renderer decides how many to print; the
    count must come from this full list, because a truncated list whose
    length is reported as the count understates the finding.
    """
    return [
        topic.get("display_name", "")
        for topic in field_baseline
        if topic.get("display_name") and topic["display_name"] not in corpus_topic_names
    ]


# --- report ------------------------------------------------------------------

_FRAMING = """\
> **How to read this report.** A researcher-curated corpus is supposed to be
> selective — curation encodes your reading program and audience, and that
> selectivity is what makes Conceptual Divergence Analysis meaningful. This
> diagnostic makes the corpus's structure *visible* (venue concentration,
> field coverage, temporal profile, untouched areas) so curation stays a
> deliberate act rather than an accident. Nothing below is a defect score,
> and "underrepresented" means "absent relative to a field's overall
> volume," not "missing where it should be present."
"""


def gather_composition(
    config: Config,
    *,
    declared_fields: list[str] | None = None,
    collection=None,
    progress: Callable[[str], None] = print,
) -> dict:
    """Resolve the corpus against OpenAlex and compute every aggregate.

    Returns a bundle (works, resolved/unresolved lists, venue/topic/field/
    temporal aggregates, declared-field baselines) that
    ``render_diagnosis_markdown`` turns into the report. Kept separate from
    rendering so the network + aggregation logic reads on its own.
    """
    collection = collection if collection is not None else get_collection(config)
    window = timeframe_from_config(config)
    with corpus_lock(config, progress):
        works = corpus_works(collection, window)
    if window["active"]:
        progress(f"[diagnose] corpus timeframe: {describe_window(window)}")
    with_doi = [work for work in works if work["doi"]]
    progress(f"[diagnose] {len(works)} distinct works in corpus; {len(with_doi)} carry DOIs")
    if not works and window["active"]:
        # An empty result under an active window is a window problem, not a
        # metadata problem — re-ingesting with a Zotero CSV would change
        # nothing. analyze already makes this distinction; diagnose must too.
        raise ValueError(
            f"The acquisition-date window ({describe_window(window)}) excludes "
            "every work in the corpus, so there is nothing to diagnose. Widen "
            "the window (--added-from/--added-to), or check --date-source."
        )
    if not with_doi:
        raise ValueError(
            "No DOIs in corpus metadata — the diagnostic resolves works via "
            "DOI. Re-ingest with a Zotero CSV export (--zotero-csv) so works "
            "carry DOIs."
        )

    cache = OpenAlexCache(config.path("paths.corpus_dir") / "openalex_cache.sqlite")
    try:
        resolved_by_doi = works_by_dois(
            [work["doi"] for work in with_doi], cache, progress=progress
        )
        resolved = list(resolved_by_doi.values())
        unresolved = [work for work in with_doi if work["doi"] not in resolved_by_doi]
        progress(
            f"[diagnose] resolved {len(resolved)} works via OpenAlex "
            f"({len({w['doi'] for w in unresolved})} DOIs unknown to OpenAlex)"
        )

        declared = [name.strip() for name in (declared_fields or []) if name.strip()]
        baselines: list[tuple[str, list[dict]]] = []
        if declared:
            all_fields = {field["display_name"].lower(): field for field in list_fields(cache)}
            unknown = [name for name in declared if name.lower() not in all_fields]
            if unknown:
                valid = ", ".join(sorted(f["display_name"] for f in all_fields.values()))
                raise ValueError(
                    f"Declared field(s) {', '.join(repr(n) for n in unknown)} match no "
                    f"OpenAlex field. Valid names: {valid}."
                )
            for name in declared:
                field = all_fields[name.lower()]
                field_id = field["id"].rsplit("/", 1)[-1]
                baselines.append((field["display_name"], top_topics_for_field(field_id, cache)))
    finally:
        cache.close()

    return {
        "window": window,
        "works": works,
        "with_doi": with_doi,
        "resolved": resolved,
        "unresolved": unresolved,
        "venue_stats": venue_concentration(resolved),
        "topics": topic_distribution(resolved),
        "fields": field_shares(resolved),
        "temporal": temporal_profile(resolved),
        "baselines": baselines,
        "corpus_topic_names": work_topic_names(resolved),
    }


def render_diagnosis_markdown(data: dict, now: datetime) -> str:
    """Render the composition report from a `gather_composition` bundle."""
    works, with_doi = data["works"], data["with_doi"]
    resolved, unresolved = data["resolved"], data["unresolved"]
    venue_stats, temporal = data["venue_stats"], data["temporal"]
    # One OpenAlex work per distinct DOI: PDFs sharing a DOI are counted once
    # above, so resolved + unresolvable + sharing adds up to works with DOIs.
    n_unresolved_dois = len({work["doi"] for work in unresolved})
    n_sharing_doi = len(with_doi) - len({work["doi"] for work in with_doi})
    hhi = "n/a" if venue_stats["hhi"] is None else venue_stats["hhi"]

    lines = [
        f"<!-- generated: {now.isoformat()} | cda diagnose -->",
        "",
        "# Corpus Composition Report",
        "",
        _FRAMING,
        f"- Distinct works in corpus: **{len(works)}**",
        *(
            [f"- Corpus timeframe: {describe_window(data['window'])}"]
            if (data.get("window") or {}).get("active")
            else []
        ),
        f"- Works with DOIs: {len(with_doi)} · resolved via OpenAlex: {len(resolved)} · "
        f"unresolvable: {n_unresolved_dois} · sharing a DOI with another work: "
        f"{n_sharing_doi} · no DOI: {len(works) - len(with_doi)}",
        "",
        "## Venue concentration",
        "",
        f"{venue_stats['n_venues']} venues across {venue_stats['n_works']} resolved works "
        f"with a known venue ({venue_stats['n_unknown_venue']} more have none in OpenAlex "
        "and are left out of the shares). "
        f"Top {len(venue_stats['top'])} venues hold **{venue_stats['top_share']:.0%}** of "
        f"those works; Herfindahl–Hirschman index **{hhi}** "
        "(0 → dispersed, 1 → single venue).",
        "",
        "| Venue | works | share |",
        "|---|---:|---:|",
    ]
    for venue, count, share in venue_stats["top"]:
        lines.append(f"| {venue} | {count} | {share:.1%} |")

    lines += [
        "",
        "## Field and topic coverage",
        "",
        "| Field (primary topic's field) | works | share |",
        "|---|---:|---:|",
    ]
    for field, count, share in data["fields"]:
        lines.append(f"| {field} | {count} | {share:.1%} |")
    lines += [
        "",
        "Top topics in the corpus:",
        "",
        "| Topic | field | works |",
        "|---|---|---:|",
    ]
    for name, field, count in data["topics"]:
        lines.append(f"| {name} | {field} | {count} |")

    if data["baselines"]:
        lines += ["", "## Structurally untouched areas of your declared fields", ""]
        for field_name, baseline in data["baselines"]:
            missing = underrepresented_topics(baseline, data["corpus_topic_names"])
            shown = missing[:TOPICS_LISTED]
            lines.append(
                f"**{field_name}** — of the field's {len(baseline)} largest topics, "
                f"{len(missing)} have no corpus presence:"
            )
            lines += [f"- {topic}" for topic in shown] or ["- (none — full coverage)"]
            if len(missing) > len(shown):
                lines.append(f"- … and {len(missing) - len(shown)} more")
            lines.append("")
    else:
        lines += [
            "",
            '*No declared fields configured — pass `--fields "Social Sciences"` '
            "(or set `diagnose.declared_fields` in your config) to compare "
            "coverage against a field baseline.*",
        ]

    if temporal.get("n_dated"):
        lines += [
            "",
            "## Temporal profile",
            "",
            f"{temporal['n_dated']} dated works, {temporal['min']}–{temporal['max']} "
            f"(median {temporal['median']}).",
            "",
            "| Decade | works |",
            "|---|---:|",
        ]
        for decade, count in temporal["by_decade"]:
            lines.append(f"| {decade} | {count} |")

    if unresolved:
        lines += [
            "",
            "## Unresolvable DOIs",
            "",
            f"{n_unresolved_dois} DOIs ({len(unresolved)} works) are not recognized by "
            "OpenAlex (often book chapters, very recent items, or DOI typos):",
            "",
        ]
        lines += [
            f"- {work.get('relpath') or work.get('filename', '')} (`{work['doi']}`)"
            for work in unresolved[:20]
        ]
        if len(unresolved) > 20:
            lines.append(f"- … and {len(unresolved) - 20} more")

    return "\n".join(lines) + "\n"


def run_diagnose(
    config: Config,
    *,
    declared_fields: list[str] | None = None,
    collection=None,
    progress: Callable[[str], None] = print,
) -> dict:
    """Resolve, aggregate, render, and write the diagnosis report."""
    data = gather_composition(
        config, declared_fields=declared_fields, collection=collection, progress=progress
    )
    now = datetime.now()
    report_text = render_diagnosis_markdown(data, now)

    out_dir = config.path("paths.output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = unique_run_basename(
        out_dir, f"corpus_diagnosis_{now.strftime('%Y%m%d_%H%M%S')}", (".md",)
    )
    report_path = out_dir / f"{base}.md"
    write_text_atomic(report_path, report_text)
    progress(f"[diagnose] report: {report_path}")
    return {
        "report_file": str(report_path),
        "n_works": len(data["works"]),
        "n_resolved": len(data["resolved"]),
        "venue_hhi": data["venue_stats"]["hhi"],
    }
