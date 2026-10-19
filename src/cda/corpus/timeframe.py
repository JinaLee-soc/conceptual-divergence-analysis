"""Restricting an analysis to a window of library-acquisition time.

A personal literature corpus grows continuously, so "the corpus" is not a
fixed object: an analysis run in July sees works that a run in April could
not. That makes results hard to compare against each other and impossible to
reproduce later. This module fixes the corpus to a *timeframe* — a window over
Zotero's "Date Added", the same acquisition axis ``cda emergence`` uses (never
publication year, which is the field's clock, not the reader's).

    cda analyze --term novelty --added-to 2026-05-01

analyzes the library exactly as it stood at the end of 1 May 2026, no matter
how much has been added since; ``--added-from`` opens the window's other end,
so a pair of bounds isolates one period of reading.

Two rules keep the filter honest rather than convenient:

- **Undated passages are dropped, and counted.** A work whose "Date Added" is
  missing cannot be placed on either side of a cutoff; keeping it would let
  post-cutoff material into a snapshot silently. The report states how many
  were dropped, and ``timeframe.include_undated: true`` overrides the rule
  when a corpus has no Zotero metadata at all.
- **Drafts are never filtered.** Drafts are read from disk at analysis time,
  not acquired into the library; they are the manuscript being examined, not
  evidence about a period. A snapshot of your reading is still a snapshot of
  *your* reading.
"""

from __future__ import annotations

import re
from typing import Any

from cda.corpus.zotero import normalize_date_added
from cda.types import Passage

_HAS_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}")
_HAS_SECONDS = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")

# Two acquisition clocks can exist in chunk metadata, both stored in UTC;
# ``timeframe.date_source`` picks one and every report names it. They answer
# different questions and can disagree by years, so the choice is explicit:
#   zotero      — Zotero's "Date Added": when the *record* entered the library.
#                 The paper's clock. Collapses onto one date if the library was
#                 ever rebuilt or re-imported.
#   filesystem  — when the *file* landed on disk. Survives a Zotero rebuild and
#                 needs no Zotero export, but tracks the download, not the read.
DATE_SOURCES = {"zotero": "date_added", "filesystem": "date_added_fs"}


def parse_bound(value: str, *, end_of_day: bool) -> str:
    """A window bound as UTC ``YYYY-MM-DD HH:MM:SS``; raises on unreadable input.

    Bounds are UTC, like every stored acquisition time; a bound with an
    explicit zone is converted to UTC. A bare date as the *upper* bound is
    padded to 23:59:59 so ``--added-to 2026-05-01`` includes that whole day,
    and an upper bound to the minute is padded to its 59th second.
    Unparseable or impossible dates raise: a silently dropped filter would
    report a snapshot that was never taken.
    """
    raw = str(value).strip()
    try:
        normalized = normalize_date_added(raw)
    except ValueError as exc:
        raise ValueError(f"Could not read '{value}' as a date: {exc}") from exc
    if not normalized:
        raise ValueError(
            f"Could not read '{value}' as a date. Use YYYY-MM-DD "
            "(or 'YYYY-MM-DD HH:MM:SS' to bound to the second); dates are UTC."
        )
    if end_of_day and not _HAS_TIME.search(raw):
        return f"{normalized[:10]} 23:59:59"
    if end_of_day and not _HAS_SECONDS.search(raw):
        # Stored times carry seconds; "10:00" means through 10:00:59.
        return f"{normalized[:16]}:59"
    return normalized


def timeframe_from_config(config) -> dict[str, Any]:
    """Normalize the ``timeframe`` config section into a window description.

    Returns ``{"active": bool, "added_from": str|None, "added_to": str|None,
    "include_undated": bool}``. Inactive when neither bound is set, which is
    the default — an unconfigured run analyzes the whole corpus, as before.
    """
    section = config.section("timeframe")
    raw_from = section.get("added_from")
    raw_to = section.get("added_to")
    added_from = parse_bound(raw_from, end_of_day=False) if raw_from else None
    added_to = parse_bound(raw_to, end_of_day=True) if raw_to else None
    if added_from and added_to and added_from > added_to:
        raise ValueError(
            f"timeframe.added_from ({added_from}) is later than timeframe.added_to "
            f"({added_to}); no passage can satisfy both."
        )
    date_source = str(section.get("date_source", "zotero")).strip().lower()
    if date_source not in DATE_SOURCES:
        raise ValueError(
            f"timeframe.date_source must be one of {sorted(DATE_SOURCES)}, got "
            f"'{date_source}'. 'zotero' reads Zotero's \"Date Added\" (when the record "
            "entered your library); 'filesystem' reads when the PDF file arrived on "
            "disk, which survives a rebuilt Zotero library."
        )
    return {
        "active": bool(added_from or added_to),
        "added_from": added_from,
        "added_to": added_to,
        # get_bool, not bool(): a quoted `include_undated: "false"` is True
        # under a bare cast, which would silently keep every undated passage.
        "include_undated": config.get_bool("timeframe.include_undated", False),
        "date_source": date_source,
    }


def _passage_date(passage: Passage, metadata_key: str = "date_added") -> str | None:
    raw = (passage.metadata or {}).get(metadata_key, "")
    return normalize_date_added(str(raw)) or None


def filter_by_date_added(
    passages: list[Passage], window: dict[str, Any]
) -> tuple[list[Passage], dict[str, Any]]:
    """Keep the passages acquired inside ``window``; report what was dropped.

    Drafts (``metadata["draft"]``) pass through untouched — see the module
    docstring. Returns (kept, counts) where counts always includes
    ``n_before``/``n_kept`` so the caller can surface the reduction even when
    nothing was dropped.
    """
    counts = {
        "n_before": len(passages),
        "n_kept": len(passages),
        "n_outside_window": 0,
        "n_undated": 0,
        "n_undated_dropped": 0,
        "n_drafts_exempt": 0,
    }
    if not window.get("active"):
        return passages, counts

    added_from = window.get("added_from")
    added_to = window.get("added_to")
    include_undated = window.get("include_undated", False)
    metadata_key = DATE_SOURCES[window.get("date_source", "zotero")]

    kept: list[Passage] = []
    for passage in passages:
        if (passage.metadata or {}).get("draft"):
            counts["n_drafts_exempt"] += 1
            kept.append(passage)
            continue
        date_added = _passage_date(passage, metadata_key)
        if date_added is None:
            counts["n_undated"] += 1
            if include_undated:
                kept.append(passage)
            else:
                counts["n_undated_dropped"] += 1
            continue
        if (added_from and date_added < added_from) or (added_to and date_added > added_to):
            counts["n_outside_window"] += 1
            continue
        kept.append(passage)

    counts["n_kept"] = len(kept)
    return kept, counts


def describe_window(window: dict[str, Any]) -> str:
    """One human-readable line for the window itself (no counts)."""
    if not window.get("active"):
        return "whole corpus (no acquisition-date window)"
    added_from = window.get("added_from")
    added_to = window.get("added_to")
    if added_from and added_to:
        span = f"added between {added_from[:10]} and {added_to[:10]} (inclusive)"
    elif added_to:
        span = f"added on or before {added_to[:10]} (corpus snapshot as of that date)"
    else:
        span = f"added on or after {added_from[:10]}"
    undated = (
        "undated passages included"
        if window.get("include_undated")
        else "undated passages excluded"
    )
    clock = {
        "zotero": 'Zotero "Date Added"',
        "filesystem": "file arrival on disk",
    }.get(window.get("date_source", "zotero"), window.get("date_source"))
    return f"{span} (UTC), by {clock}; {undated}"


def describe_timeframe(info: dict[str, Any]) -> str:
    """The report/prompt line: the window plus what it removed."""
    window = info.get("window") or {}
    if not window.get("active"):
        return "Corpus timeframe: whole corpus (no acquisition-date window applied)."
    parts = [f"Corpus timeframe: {describe_window(window)}."]
    for label, counts in (
        ("researcher", info.get("researcher") or {}),
        ("literature", info.get("literature") or {}),
    ):
        # A pool that was empty before the filter has nothing to report (a
        # drafts-only researcher side, typically).
        if not counts or not counts["n_before"]:
            continue
        detail = (
            f"{label}: {counts['n_kept']} of {counts['n_before']} passages kept "
            f"({counts['n_outside_window']} outside the window"
        )
        if counts["n_undated_dropped"]:
            detail += f", {counts['n_undated_dropped']} undated"
        elif counts["n_undated"]:
            detail += f", {counts['n_undated']} undated kept"
        parts.append(detail + ")")
    return " ".join(parts)
