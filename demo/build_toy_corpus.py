#!/usr/bin/env python3
"""Build the demo toy corpus from OpenAlex (CC0 metadata + abstracts).

Fetches ~120–150 highly-cited works from three OpenAlex topics adjacent to
the sociology of science / science studies, keeps only metadata and abstracts
(OpenAlex data is CC0 — no copyrighted full text ever enters this
repository), and writes demo/toy_corpus/abstracts.jsonl.

Topic filters (rather than free-text search) keep the corpus definition
stable and reproducible: the same three topic IDs always denote the same
research areas. The topics were seeded from canonical papers in the field
(Merton's Matthew Effect, Uzzi et al.'s atypical combinations, Oreskes'
climate-consensus essay, Gieryn's boundary-work).

The result is committed so the demo works offline; re-running this script
regenerates it. Requires OPENALEX_MAILTO in the environment or .env
(polite pool).

Usage:
    python demo/build_toy_corpus.py [--max-works 150]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from cda.diagnose.openalex import get_json, reconstruct_abstract

# Three OpenAlex topics adjacent to sociology of science / science studies,
# chosen so the corpus discusses both demo terms ("novelty", "consensus")
# from several literature registers: scientometric novelty/impact,
# philosophy-of-science epistemics, and consensus in climate science.
TOPICS = [
    ("T10102", "scientometrics and bibliometrics research"),
    ("T10778", "Philosophy and History of Science"),
    ("T14329", "Science and Climate Studies"),
]
# The demo analyzes these terms, so the corpus must actually discuss them.
# Abstracts are selected client-side by whole-word match (the pipeline's unit
# of analysis is the literal term-sentence), walking each topic's most-cited
# works via cursor pagination — this uses only plain filter queries, which
# are cheap for OpenAlex and independent of their full-text search service.
DEMO_TERMS = ("novelty", "consensus")
TERM_BEARING_PER_TOPIC = 40
GENERIC_PER_TOPIC = 10  # top-cited works kept regardless, for corpus texture
MAX_PAGES_PER_TOPIC = 15  # safety cap: 15 × 200 works scanned per topic
MIN_ABSTRACT_CHARS = 400

OUT_PATH = Path(__file__).parent / "toy_corpus" / "abstracts.jsonl"

_TERM_PATTERNS = {term: re.compile(rf"\b{term}\b", re.IGNORECASE) for term in DEMO_TERMS}


def slugify(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_length].rstrip("_") or "untitled"


def iter_topic_works(topic_id: str):
    """Yield a topic's English articles with abstracts, most-cited first."""
    cursor = "*"
    for _page in range(MAX_PAGES_PER_TOPIC):
        response = get_json(
            "works",
            {
                "filter": (
                    f"primary_topic.id:{topic_id},has_abstract:true,language:en,type:article"
                ),
                "sort": "cited_by_count:desc",
                "per-page": 200,
                "cursor": cursor,
            },
        )
        yield from response.get("results", [])
        cursor = (response.get("meta") or {}).get("next_cursor")
        if not cursor:
            return
        time.sleep(0.25)  # polite pacing between pages


def to_record(work: dict, topic_label: str) -> dict | None:
    """Normalize one OpenAlex work; None when the abstract is unusable."""
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    if len(abstract) < MIN_ABSTRACT_CHARS:
        return None
    title = (work.get("title") or "").strip()
    if not title:
        return None
    matched_terms = [t for t, pattern in _TERM_PATTERNS.items() if pattern.search(abstract)]
    year = work.get("publication_year") or ""
    venue = ""
    primary = work.get("primary_location") or {}
    if primary.get("source"):
        venue = primary["source"].get("display_name") or ""
    authors = [
        (authorship.get("author") or {}).get("display_name", "")
        for authorship in (work.get("authorships") or [])[:3]
    ]
    openalex_id = (work.get("id") or "").rsplit("/", 1)[-1]
    return {
        # "source" is the file-name-like identifier passages cite in reports.
        "source": f"{year}_{slugify(title)}.abstract",
        "openalex_id": openalex_id,
        "doi": (work.get("doi") or "").replace("https://doi.org/", ""),
        "title": title,
        "year": str(year),
        "venue": venue,
        "authors": authors,
        "topic": topic_label,
        "demo_terms": matched_terms,
        "abstract": abstract,
        "provenance": "OpenAlex (CC0), abstract reconstructed from abstract_inverted_index",
    }


def assign_demo_date_added(records: list[dict]) -> None:
    """Give every record a deterministic synthetic acquisition date.

    The fictional demo researcher "collected" the corpus one topic at a time
    (TOPICS order), one work a week from 2019-01-07, reading each topic
    oldest-first. Topic-staggered acquisition makes cluster births visible in
    `cda emergence --demo`. The rule is a pure function of committed fields,
    so regenerating the corpus never shuffles the dates. Dates are UTC, like
    every acquisition time CDA stores.
    """
    topic_order = {label: index for index, (_id, label) in enumerate(TOPICS)}
    reading_order = sorted(
        records,
        key=lambda r: (topic_order.get(r["topic"], len(TOPICS)), r["year"], r["source"]),
    )
    first_day = datetime(2019, 1, 7, 9, 0, 0, tzinfo=UTC)
    for rank, record in enumerate(reading_order):
        record["date_added"] = (first_day + timedelta(days=7 * rank)).strftime("%Y-%m-%d %H:%M:%S")


def rewrite_dates_only() -> int:
    """Add/refresh date_added on the committed jsonl — no network needed."""
    records = []
    with OUT_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    assign_demo_date_added(records)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[done] refreshed date_added on {len(records)} records in {OUT_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-works", type=int, default=150)
    parser.add_argument(
        "--dates-only",
        action="store_true",
        help="Only add/refresh the synthetic date_added field on the committed "
        "jsonl (offline; no OpenAlex fetch)",
    )
    args = parser.parse_args()

    if args.dates_only:
        return rewrite_dates_only()

    records: dict[str, dict] = {}
    for topic_id, label in TOPICS:
        print(f"[fetch] topic {topic_id} ({label}) …")
        term_bearing = 0
        generic = 0
        scanned = 0
        for work in iter_topic_works(topic_id):
            scanned += 1
            record = to_record(work, label)
            if record is None:
                continue
            key = record["openalex_id"] or record["title"].lower()
            if key in records:
                continue
            if record["demo_terms"] and term_bearing < TERM_BEARING_PER_TOPIC:
                records[key] = record
                term_bearing += 1
            elif not record["demo_terms"] and generic < GENERIC_PER_TOPIC:
                records[key] = record
                generic += 1
            if term_bearing >= TERM_BEARING_PER_TOPIC and generic >= GENERIC_PER_TOPIC:
                break
        print(
            f"        scanned {scanned} works → kept {term_bearing} term-bearing "
            f"+ {generic} generic"
        )

    ordered = sorted(records.values(), key=lambda r: r["openalex_id"])[: args.max_works]

    # Make "source" unique even if two works share a year+title slug.
    seen_sources: dict[str, int] = {}
    for record in ordered:
        base = record["source"]
        count = seen_sources.get(base, 0)
        seen_sources[base] = count + 1
        if count:
            record["source"] = base.replace(".abstract", f"_{count + 1}.abstract")

    assign_demo_date_added(ordered)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(ordered)} works to {OUT_PATH}")
    if len(ordered) < 100:
        print("[warn] fewer than 100 usable works — consider raising PER_TOPIC before committing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
