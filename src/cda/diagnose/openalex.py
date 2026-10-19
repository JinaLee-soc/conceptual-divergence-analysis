"""Minimal OpenAlex client utilities.

Used by the toy-corpus builder (demo/build_toy_corpus.py) and the corpus
diagnostics (`cda diagnose`). Follows OpenAlex "polite pool" etiquette: every
request carries a mailto identifier from the OPENALEX_MAILTO environment
variable, requests are paced, and diagnostics responses are cached locally in
SQLite so repeated runs cost OpenAlex nothing. OpenAlex metadata and
abstracts are CC0.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, quote_plus

import requests

from cda.corpus.zotero import normalize_doi

OPENALEX_BASE = "https://api.openalex.org"
_PACING_SECONDS = 0.15  # polite spacing between uncached requests


class OpenAlexError(RuntimeError):
    """OpenAlex request problems, with the next step in the message."""


def polite_mailto() -> str:
    mailto = os.environ.get("OPENALEX_MAILTO", "").strip()
    if not mailto or "@" not in mailto:
        raise OpenAlexError(
            "OPENALEX_MAILTO is not set (or is not an email address).\n"
            "OpenAlex asks API users to identify themselves with an email for "
            "the faster 'polite pool'. Copy .env.example to .env and fill in "
            "OPENALEX_MAILTO=you@example.org, then rerun."
        )
    return mailto


def _redact_mailto(text: str, mailto: str) -> str:
    """Strip the polite-pool email from text bound for an error message."""
    text = re.sub(r"mailto=[^&\s'\")]*", "mailto=<redacted>", text)
    for form in {mailto, quote(mailto, safe=""), quote_plus(mailto)}:
        text = text.replace(form, "<redacted>")
    return text


def get_json(
    path: str,
    params: dict | None = None,
    *,
    max_retries: int = 4,
    timeout: float = 30.0,
) -> dict:
    """GET an OpenAlex endpoint with polite mailto and backoff on 429/5xx."""
    params = dict(params or {})
    mailto = polite_mailto()
    params["mailto"] = mailto
    url = f"{OPENALEX_BASE}/{path.lstrip('/')}"

    last_error: str | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_error = _redact_mailto(str(exc), mailto)
            time.sleep(min(2**attempt, 20))
            continue
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as exc:
                # A 200 with a non-JSON body is almost always a captive
                # portal or intercepting proxy answering instead of
                # OpenAlex; the bare JSON parse error named neither.
                raise OpenAlexError(
                    f"OpenAlex returned a non-JSON response from {url} — "
                    "usually a captive portal or an intercepting proxy on "
                    "this network. Open https://api.openalex.org in a "
                    "browser to check, then rerun."
                ) from exc
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2**attempt, 20))
            last_error = f"HTTP {response.status_code} from {url}"
            continue
        if response.status_code in (401, 403, 407):
            # OpenAlex needs no credentials, so an auth/forbidden status
            # almost never comes from OpenAlex: it is a campus or corporate
            # proxy, a VPN filter, or a firewall answering in its place.
            # Sending the reader to "check the query parameters" (below) sent
            # them to debug a query that was fine.
            raise OpenAlexError(
                f"OpenAlex returned HTTP {response.status_code} for {url}. "
                "The OpenAlex API needs no login, so this is usually a "
                "network filter, VPN, or proxy answering instead of "
                "OpenAlex. Open https://api.openalex.org in a browser on "
                "this machine to check; if it is blocked, ask your network "
                "administrator to allow api.openalex.org, or rerun off the "
                "restricted network. `cda diagnose` is the only command that "
                "contacts OpenAlex; the rest of the pipeline does not need it."
            )
        raise OpenAlexError(
            f"OpenAlex returned HTTP {response.status_code} for {url}. "
            "Check the query parameters; see https://docs.openalex.org/."
        )
    raise OpenAlexError(
        f"OpenAlex request failed after {max_retries} attempts: {last_error}. "
        "That is usually a dropped connection, an OpenAlex outage, or a "
        "proxy or firewall between you and api.openalex.org. Open "
        "https://api.openalex.org in a browser on this machine to tell them "
        "apart, then retry."
    )


def _cache_unusable_error(path: Path, exc: Exception) -> OpenAlexError:
    return OpenAlexError(
        f"The OpenAlex response cache ({path}) could not be used: {exc}. "
        "If the message says 'database is locked', another `cda diagnose` — "
        "from the GUI or another terminal — is using it; let that run finish, "
        "then rerun. Otherwise the cache file may be damaged; deleting it is "
        "safe (responses are simply re-fetched from OpenAlex)."
    )


class OpenAlexCache:
    """URL-keyed response cache (SQLite) for the diagnostics.

    Lives next to the corpus (gitignored with it). Diagnostics re-runs then
    read from disk instead of re-querying OpenAlex.

    SQLite allows one writer at a time, so a GUI diagnose and a CLI diagnose
    sharing a corpus can collide here. The connection waits up to 30 s for
    the other writer (sqlite's own busy timeout), and a lock that outlasts
    the wait surfaces as an ``OpenAlexError`` saying which run to let finish
    — not as a bare ``sqlite3.OperationalError`` traceback.
    """

    _BUSY_TIMEOUT_SECONDS = 30

    def __init__(self, path: str | Path):
        self._path = Path(path)
        try:
            self._conn = sqlite3.connect(str(path), timeout=self._BUSY_TIMEOUT_SECONDS)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS openalex_cache ("
                "key TEXT PRIMARY KEY, payload TEXT NOT NULL, fetched_at TEXT NOT NULL)"
            )
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            raise _cache_unusable_error(self._path, exc) from exc

    def get(self, key: str) -> dict | None:
        try:
            row = self._conn.execute(
                "SELECT payload FROM openalex_cache WHERE key = ?", (key,)
            ).fetchone()
        except sqlite3.OperationalError as exc:
            raise _cache_unusable_error(self._path, exc) from exc
        return json.loads(row[0]) if row else None

    def put(self, key: str, payload: dict) -> None:
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO openalex_cache (key, payload, fetched_at) VALUES (?, ?, ?)",
                (key, json.dumps(payload), datetime.now(UTC).isoformat(timespec="seconds")),
            )
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            raise _cache_unusable_error(self._path, exc) from exc

    def close(self) -> None:
        self._conn.close()


def cached_get_json(path: str, params: dict | None, cache: OpenAlexCache | None) -> dict:
    """get_json with a read-through cache and polite pacing on misses."""
    key = path + "?" + json.dumps(params or {}, sort_keys=True)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
    payload = get_json(path, params)
    if cache is not None:
        cache.put(key, payload)
    time.sleep(_PACING_SECONDS)
    return payload


def works_by_dois(
    dois: list[str],
    cache: OpenAlexCache | None = None,
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, dict]:
    """Resolve many DOIs to OpenAlex works, batched 40 per request.

    Returns {normalized_doi: work}. DOIs OpenAlex doesn't know are simply
    absent from the result (callers report them as unresolvable).

    DOIs go through the same ``normalize_doi`` ingest uses (resolver
    prefixes and "doi:" stripped, lowercased): a Zotero field holding
    "doi:10.1/x" or a dx.doi.org URL is a fine DOI and must not be
    misreported as unknown to OpenAlex. A value still containing ``|`` or
    ``,`` after normalization cannot be sent at all — both characters are
    OpenAlex filter *syntax*, so one malformed field would corrupt (``|``)
    or crash (``,``) the whole 40-DOI batch it rides in. Such values are
    announced with the offending field value and left out of the batch;
    they then surface as unresolvable, which is what they are.
    """
    resolved: dict[str, dict] = {}
    normalized: list[str] = []
    for raw in dois:
        doi = normalize_doi(raw)
        if not doi:
            continue
        if "|" in doi or "," in doi:
            progress(
                f"  [diagnose] the DOI field {str(raw).strip()!r} contains "
                "'|' or ',' — not a character a DOI query can carry, so it "
                "cannot be sent to OpenAlex and is reported as unresolvable. "
                "Fix this item's DOI in Zotero and re-ingest to include it."
            )
            continue
        normalized.append(doi)
    for start in range(0, len(normalized), 40):
        batch = normalized[start : start + 40]
        response = cached_get_json(
            "works",
            {"filter": "doi:" + "|".join(batch), "per-page": len(batch)},
            cache,
        )
        for work in response.get("results", []):
            work_doi = normalize_doi(work.get("doi") or "")
            if work_doi:
                resolved[work_doi] = work
    return resolved


def list_fields(cache: OpenAlexCache | None = None) -> list[dict]:
    """The ~26 top-level OpenAlex fields (id, display_name)."""
    response = cached_get_json("fields", {"per-page": 50}, cache)
    return response.get("results", [])


def top_topics_for_field(
    field_id: str, cache: OpenAlexCache | None = None, n: int = 25
) -> list[dict]:
    """The field's biggest topics by works count — the coverage baseline."""
    response = cached_get_json(
        "topics",
        {"filter": f"field.id:{field_id}", "sort": "works_count:desc", "per-page": n},
        cache,
    )
    return response.get("results", [])


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Rebuild abstract text from OpenAlex's abstract_inverted_index.

    OpenAlex stores abstracts as {word: [positions...]} for legal reasons;
    placing each word back at its positions restores the original text.
    """
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted_index.items():
        for index in indexes:
            positions.append((int(index), word))
    if not positions:
        return ""
    positions.sort(key=lambda pair: pair[0])
    return " ".join(word for _, word in positions)
