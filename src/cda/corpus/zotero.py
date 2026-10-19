"""Zotero metadata via a library CSV export.

CDA never talks to Zotero directly: you export your library once
(File → Export Library… → Format: CSV) and pass the file to ``cda ingest``.
This keeps the pipeline offline, needs no Zotero API key, and works with any
Zotero version. Metadata (title, year, DOI, tags) is resolved per PDF at
ingest time and written into chunk metadata, so analysis runs never need
Zotero at all.

The ``mywork`` tag is how you mark your own papers inside Zotero: tag your
items "mywork" and MyWork resolution will classify their passages as
researcher-authored (see docs/corpus_setup_zotero.md).
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path, PureWindowsPath

from dateutil.parser import isoparse

from cda.extraction.windows import YEAR_PATTERN


def normalize_key(filename: str) -> str:
    """Case/punctuation-insensitive key for matching file names.

    Accents are folded to their ASCII skeleton (NFKD, combining marks dropped)
    so NFC and NFD spellings of one name give one key. The key is empty for a
    fully non-ASCII name; callers treat "" as "no key" and never index or
    look it up. A path with backslashes (a Zotero export written on Windows)
    is split as a Windows path.
    """
    filename = str(filename)
    base = (PureWindowsPath if "\\" in filename else Path)(filename).name.lower()
    base = re.sub(r"\.pdf$", "", base)
    base = unicodedata.normalize("NFKD", base)
    base = "".join(ch for ch in base if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", base)


def normalize_tag(tag: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(tag).strip().lower().lstrip("#"))


# A Date Added value embedded in an export cell: ISO date, optional clock with
# optional fractional seconds, optional zone (optionally after a space).
_DATE_ADDED = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[ T](?P<hour>\d{2}):\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:\s?(?:Z|[+-]\d{2}(?::?\d{2})?))?)?"
)


def normalize_date_added(value: str) -> str:
    """A "Date Added" value as UTC ``YYYY-MM-DD HH:MM:SS``; "" when it holds no ISO date.

    Zoned values are converted to UTC. Naive values are taken as UTC, which is
    what Zotero writes, so results never depend on the machine's time zone.
    A bare date means midnight; fractional seconds are dropped. The
    normalized form sorts lexicographically in time order, which timeframe
    filtering and ``cda emergence`` rely on. An impossible calendar date or
    clock time (24:00 included) raises ``ValueError``. Non-ISO forms such as
    Excel's "1/2/2024 3:04" give "": their day/month order is ambiguous.
    """
    match = _DATE_ADDED.search(str(value or ""))
    if not match:
        return ""
    # isoparse wants the zone directly after the clock: "03:04:05 +09:00".
    iso = re.sub(r"\s(?=[+-]\d{2}(?::?\d{2})?$|Z$)", "", match.group(0))
    try:
        if match.group("hour") == "24":
            raise ValueError("hour must be in 0..23")
        stamp = isoparse(iso)
    except ValueError as exc:
        raise ValueError(f"Date Added {str(value).strip()!r} is not a real date ({exc}).") from exc
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(UTC).replace(tzinfo=None)
    return stamp.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class ZoteroRecord:
    title: str
    year: str
    doi: str
    date_added: str = ""  # normalized "YYYY-MM-DD HH:MM:SS", "" when absent
    tags: list[str] = field(default_factory=list)
    attachment_keys: list[str] = field(default_factory=list)  # normalized file stems

    @property
    def is_mywork(self) -> bool:
        return any(normalize_tag(t) == "mywork" for t in self.tags)


def _split_multi(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


# One column CDA reads is not proof of a Zotero export, but a file carrying
# none of these beyond "Title" definitely isn't one (column names are
# case-sensitive on purpose: Paperpile writes "Publication year").
_ZOTERO_MARKER_COLUMNS = (
    "Item Type",
    "Publication Year",
    "Date Added",
    "Manual Tags",
    "Automatic Tags",
    "File Attachments",
)


def _require_zotero_headers(fieldnames, path: Path) -> None:
    """Refuse a CSV that is clearly not a Zotero export (column names are case-sensitive)."""
    headers = [h.strip() for h in (fieldnames or []) if h and h.strip()]
    if "Title" in headers and any(marker in headers for marker in _ZOTERO_MARKER_COLUMNS):
        return
    shown = ", ".join(headers[:12]) or "(no header row)"
    raise ValueError(
        f"{path} does not look like a Zotero CSV export — found columns: "
        f"{shown}. Expected Zotero's column names ('Title', 'Item Type', "
        "'Publication Year', 'Date Added', …; they are case-sensitive). "
        "If this library lives in Paperpile, Mendeley, or EndNote, import it "
        "into Zotero first, then File → Export Library… → Format: CSV."
    )


@contextmanager
def _raised_csv_field_limit(limit: int = 10_000_000) -> Iterator[None]:
    """Raise csv's process-global field-size cap for one read, then restore it.

    Zotero Notes/Abstract columns can exceed the 128 KB default.
    """
    previous = csv.field_size_limit()
    csv.field_size_limit(max(previous, limit))
    try:
        yield
    finally:
        csv.field_size_limit(previous)


def _warn_unread_dates(path: Path, values: list[str], progress: Callable[[str], None]) -> None:
    if values:
        progress(
            f"[warn] {path.name}: {len(values)} record(s) have a Date Added that is not "
            f"an ISO date (for example {values[0]!r}); they are undated, so a timeframe "
            "window drops them. Re-export the library from Zotero rather than "
            "re-saving the CSV in a spreadsheet program."
        )


def load_zotero_csv(
    path: str | Path, *, progress: Callable[[str], None] = print
) -> list[ZoteroRecord]:
    """Parse a Zotero CSV export into records.

    Only the columns CDA needs are read; extra columns are ignored, so any
    Zotero export profile works. A row whose Date Added is not a real date
    raises ``ValueError`` naming the row; rows whose Date Added is not an ISO
    date at all are left undated, counted, and reported through ``progress``.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Zotero export not found: {path}\n"
            "In Zotero: File → Export Library… → Format: CSV, then pass the "
            "file with --zotero-csv."
        )
    records: list[ZoteroRecord] = []
    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise FileNotFoundError(f"Could not open Zotero export {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        try:
            with _raised_csv_field_limit():
                rows = list(reader)
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{path} is not UTF-8 (byte {exc.start}: {exc.reason}). Zotero "
                "exports are UTF-8; this file was probably re-saved by another "
                "program (Excel does this). Re-export from Zotero, or re-save "
                "the file as UTF-8."
            ) from exc
        except csv.Error as exc:
            raise ValueError(
                f"Could not parse {path} as CSV ({exc}). The file may be "
                "corrupt or not a CSV at all — re-export it from Zotero "
                "(File → Export Library… → Format: CSV)."
            ) from exc
        _require_zotero_headers(reader.fieldnames, path)
        unread_dates: list[str] = []
        for row_number, row in enumerate(rows, start=1):
            title = (row.get("Title") or "").strip()
            year_raw = (row.get("Publication Year") or row.get("Date") or "").strip()
            year_match = YEAR_PATTERN.search(year_raw)
            try:
                date_added = normalize_date_added(row.get("Date Added") or "")
            except ValueError as exc:
                raise ValueError(
                    f"{path}, record {row_number} ({title or 'untitled'}): {exc} "
                    "Fix this item's Date Added in Zotero and re-export."
                ) from exc
            if not date_added and (row.get("Date Added") or "").strip():
                unread_dates.append(row["Date Added"].strip())
            tags = _split_multi(row.get("Manual Tags", "")) + _split_multi(
                row.get("Automatic Tags", "")
            )
            attachments = _split_multi(row.get("File Attachments", ""))
            attachment_keys = [
                normalize_key(att) for att in attachments if att.lower().endswith(".pdf")
            ]
            records.append(
                ZoteroRecord(
                    title=title,
                    year=year_match.group(0) if year_match else "",
                    doi=(row.get("DOI") or "").strip(),
                    date_added=date_added,
                    tags=tags,
                    attachment_keys=attachment_keys,
                )
            )
    _warn_unread_dates(path, unread_dates, progress)
    return records


def _unique_index(pairs: Iterable[tuple[str, ZoteroRecord]]) -> dict[str, ZoteroRecord | None]:
    """Map key → record. A key shared by distinct records maps to ``None``.

    ``None`` marks the key as ambiguous: attaching either record could give a
    file another work's title, acquisition date, and mywork tag, so callers
    attach nothing for it and report why. Empty keys are never indexed.
    """
    index: dict[str, ZoteroRecord | None] = {}
    for key, record in pairs:
        if not key:
            continue
        if key not in index:
            index[key] = record
        elif index[key] != record:
            index[key] = None
    return index


def build_filename_index(records: list[ZoteroRecord]) -> dict[str, ZoteroRecord | None]:
    """Map normalized PDF attachment stems to their Zotero record (see ``_unique_index``)."""
    return _unique_index((key, record) for record in records for key in record.attachment_keys)


def count_ambiguous(index: dict[str, ZoteroRecord | None]) -> int:
    return sum(1 for record in index.values() if record is None)


# --- content-based fallback matching -----------------------------------------
# When a file's name does not match its Zotero attachment, its own DOI, then
# its title, identify it. Every match records how it was made.

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
# Enough of a long title to be distinctive without demanding an exact match
# through line breaks, hyphenation, and subtitle punctuation.
_TITLE_PREFIX_CHARS = 40
_DOI_SEARCH_CHARS = 4000  # the DOI lives on the first page
# The opening region searched for library titles: the first page's title
# block and abstract, before the body starts quoting other works at length.
_TITLE_SEARCH_CHARS = 3000


def normalize_doi(value: str) -> str:
    """Lowercased bare DOI: strips resolver prefixes and trailing punctuation."""
    text = str(value or "").strip().lower()
    text = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.rstrip(".,;)]}").strip()


def normalize_title(value: str) -> str:
    """Letters and digits only, lowercased, accents folded like ``normalize_key``."""
    text = unicodedata.normalize("NFKD", str(value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text)


def build_doi_index(records: list[ZoteroRecord]) -> dict[str, ZoteroRecord | None]:
    """Map normalized DOI to record (see ``_unique_index``)."""
    return _unique_index((normalize_doi(record.doi), record) for record in records)


# Section-like titles that many library items share (editorials, front
# matter, replies). They appear on nearly every first page, so as title keys
# they would make every content match ambiguous; they are not indexed.
_GENERIC_TITLES = frozenset(
    normalize_title(title)
    for title in (
        "Introduction",
        "Editorial",
        "Preface",
        "Foreword",
        "Conclusion",
        "Conclusions",
        "Reply",
        "Comment",
        "Erratum",
        "Correction",
        "Book Review",
        "Index",
        "Abstract",
        "Acknowledgments",
        "Acknowledgements",
        "Response",
    )
)


def build_title_index(records: list[ZoteroRecord]) -> dict[str, ZoteroRecord | None]:
    """Map a normalized title key to record (see ``_unique_index``).

    The key is the first ``_TITLE_PREFIX_CHARS`` normalized characters, or
    the whole normalized title when it is shorter. Short titles are indexed
    too: a short library title quoted on another work's first page must
    make that page ambiguous instead of leaving a cited work's title as the
    only match. Generic titles (``_GENERIC_TITLES``) are not indexed.
    """
    return _unique_index(
        (normalize_title(record.title)[:_TITLE_PREFIX_CHARS], record)
        for record in records
        if normalize_title(record.title) not in _GENERIC_TITLES
    )


def _distinct(records: list[ZoteroRecord | None]) -> list[ZoteroRecord | None]:
    """``records`` without repeats (several keys can lead to one record)."""
    unique: list[ZoteroRecord | None] = []
    for record in records:
        if record not in unique:
            unique.append(record)
    return unique


def match_by_content(
    text: str,
    doi_index: dict[str, ZoteroRecord | None],
    title_index: dict[str, ZoteroRecord | None],
) -> tuple[ZoteroRecord | None, str]:
    """Find the Zotero record for a PDF from its own opening text.

    Returns (record, outcome): outcome is "doi" or "title" with a record, or
    "ambiguous_doi" / "ambiguous_title" / "short_title_only" / "" without one. DOI is tried first,
    in the first ``_DOI_SEARCH_CHARS`` characters; titles are searched in
    the first ``_TITLE_SEARCH_CHARS``. The opening pages also cite other
    works, so an opening naming more than one distinct library work, or one
    key shared by several records, attaches nothing.
    """
    head = text[:_DOI_SEARCH_CHARS]
    dois = {normalize_doi(candidate) for candidate in DOI_PATTERN.findall(head)}
    matched = [doi_index[doi] for doi in sorted(dois) if doi in doi_index]
    if len(matched) == 1 and matched[0] is not None:
        return matched[0], "doi"
    if matched:
        return None, "ambiguous_doi"
    opening = normalize_title(text[:_TITLE_SEARCH_CHARS])
    hits = [(key, record) for key, record in title_index.items() if key in opening]
    titled = _distinct([record for _, record in hits])
    if len(titled) > 1 or (titled and titled[0] is None):
        return None, "ambiguous_title"
    if titled:
        # A short title ("Novelty") is a common phrase, not a distinctive
        # key: it can make an opening ambiguous but never attach on its own.
        if any(len(key) >= _TITLE_PREFIX_CHARS for key, _ in hits):
            return titled[0], "title"
        return None, "short_title_only"
    return None, ""


def record_fields(record: ZoteroRecord) -> dict[str, str]:
    """The chunk-metadata fields one Zotero record contributes."""
    fields: dict[str, str] = {}
    if record.title:
        fields["title"] = record.title
    if record.year:
        fields["year"] = record.year
    if record.doi:
        fields["doi"] = record.doi
    if record.date_added:
        fields["date_added"] = record.date_added
    if record.tags:
        fields["zotero_tags"] = "; ".join(record.tags)
    return fields


def match_by_filename(
    pdf_path: str | Path, index: dict[str, ZoteroRecord | None]
) -> tuple[ZoteroRecord | None, str]:
    """(record, "filename"), (None, "ambiguous_filename"), or (None, "").

    A name that normalizes to an empty key (fully non-ASCII) is never looked
    up; such files fall back to content-based matching.
    """
    key = normalize_key(Path(pdf_path).name)
    if not key or key not in index:
        return None, ""
    record = index[key]
    return (record, "filename") if record is not None else (None, "ambiguous_filename")
