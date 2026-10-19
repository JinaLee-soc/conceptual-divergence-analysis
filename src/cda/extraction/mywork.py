"""MyWork resolution: separating researcher-authored passages from literature.

The paper (§3.3) describes a five-signal priority algorithm. Four signals are
evaluated here, in the configurable priority order, against each retrieved
chunk's file name and metadata:

  filename_patterns — glob patterns from config, matched against the file name
                      and its stem (manual overrides, e.g. working papers)
  metadata_flags    — explicit flags in chunk metadata (self_authored, is_self,
                      self, author_type, owner)
  zotero_tags       — a 'mywork' tag among the Zotero tags written into chunk
                      metadata at ingest
  profile_titles    — token overlap between the file name and the researcher's
                      declared publication titles (English stop words excluded)

The fifth signal — direct draft inclusion — is a separate channel: files from
the drafts directory never pass through this resolver and are always
researcher-authored (see extraction/draft_passages.py).

Every classification records the signal that fired, so reports can show why
each passage landed where it did.
"""

from __future__ import annotations

import fnmatch
import re
import unicodedata
from collections.abc import Callable
from functools import partial
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from cda.config import Config, ConfigError
from cda.corpus.zotero import normalize_tag

_TRUTHY = {"1", "true", "yes", "self", "researcher", "mywork"}
_FLAG_KEYS = ("self_authored", "is_self", "self", "author_type", "owner")
KNOWN_SIGNALS = ("filename_patterns", "metadata_flags", "zotero_tags", "profile_titles")


def _as_list(value) -> list:
    """Wrap a bare YAML scalar into a one-element list (never iterate a string)."""
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _fold(text: str) -> str:
    """NFKC, casefolded, accents removed: one spelling for NFC and NFD names."""
    text = unicodedata.normalize("NFKC", text).casefold()
    decomposed = unicodedata.normalize("NFKD", text)
    # Recompose so Hangul jamo from NFKD become syllables again.
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    )


def title_tokens(text: str) -> set[str]:
    """Content tokens of a title or file stem: no English stop words, no short tokens.

    Tokens are runs of Unicode letters and digits, so accented and non-Latin
    titles keep whole words. The three-character minimum applies to ASCII
    tokens only: a two-syllable Korean or two-character CJK word is a word.
    """
    tokens = set(re.findall(r"[^\W_]+", _fold(text)))
    return {t for t in tokens if len(t) > 2 or not t.isascii()} - ENGLISH_STOP_WORDS


def load_mywork_rules(mywork_config: dict) -> dict:
    """Validate the ``mywork`` config section into classification rules."""
    priority = [str(s) for s in _as_list(mywork_config.get("priority"))] or list(KNOWN_SIGNALS)
    unknown = [s for s in priority if s not in KNOWN_SIGNALS]
    if unknown:
        raise ValueError(
            f"Unknown MyWork signal(s) in mywork.priority: {unknown}. "
            f"Valid signals: {list(KNOWN_SIGNALS)}"
        )
    section = Config({"mywork": mywork_config})
    min_overlap = section.get_int("mywork.title_match_min_overlap", 3, minimum=1)
    min_ratio = section.get_float("mywork.title_match_min_ratio", 0.55, minimum=0.0, maximum=1.0)
    titles = [str(t) for t in _as_list(mywork_config.get("publication_titles")) if str(t).strip()]
    tokens = [title_tokens(title) for title in titles]
    unusable = [t for t, tok in zip(titles, tokens, strict=True) if len(tok) < min_overlap]
    if unusable:
        raise ConfigError(
            f"mywork.publication_titles entries {unusable} have fewer than "
            f"title_match_min_overlap={min_overlap} content words (stop words and "
            "Latin-script words under three letters are not counted), so they could never match. "
            "Add a filename_patterns entry for those papers, or lower "
            "title_match_min_overlap."
        )
    patterns = [str(p).strip().lower() for p in _as_list(mywork_config.get("filename_patterns"))]
    return {
        "priority": priority,
        "patterns": [p for p in patterns if p],
        "title_tokens": tokens,
        "min_overlap": min_overlap,
        "min_ratio": min_ratio,
    }


def _filename_patterns(rules: dict, filename: str, metadata: dict) -> str | None:
    path = Path(filename)
    names = (path.name.lower(), path.stem.lower())
    for pattern in rules["patterns"]:
        if any(fnmatch.fnmatch(name, pattern) for name in names):
            return f"pattern:{pattern}"
    return None


def _metadata_flags(rules: dict, filename: str, metadata: dict) -> str | None:
    for key in _FLAG_KEYS:
        if key in metadata and str(metadata[key]).strip().lower() in _TRUTHY:
            return f"metadata:{key}"
    return None


def _zotero_tags(rules: dict, filename: str, metadata: dict) -> str | None:
    tags = str(metadata.get("zotero_tags") or "").split(";")
    if any(normalize_tag(t) == "mywork" for t in tags if t.strip()):
        return "zotero:mywork"
    return None


def _profile_titles(rules: dict, filename: str, metadata: dict) -> str | None:
    file_tokens = title_tokens(Path(filename).stem)
    for tokens in rules["title_tokens"]:
        overlap = len(file_tokens & tokens)
        if overlap >= rules["min_overlap"] and overlap / len(tokens) >= rules["min_ratio"]:
            return "profile:title_match"
    return None


_SIGNALS = {
    "filename_patterns": _filename_patterns,
    "metadata_flags": _metadata_flags,
    "zotero_tags": _zotero_tags,
    "profile_titles": _profile_titles,
}


def classify_source(rules: dict, filename: str, metadata: dict) -> tuple[bool, str]:
    """Return (is_researcher, reason). The first matching signal wins."""
    for signal in rules["priority"]:
        reason = _SIGNALS[signal](rules, filename, metadata or {})
        if reason:
            return True, reason
    return False, "default:literature"


def mywork_classifier(mywork_config: dict) -> Callable[[str, dict], tuple[bool, str]]:
    """``classify(filename, metadata) -> (is_researcher, reason)`` for these rules."""
    return partial(classify_source, load_mywork_rules(mywork_config))
