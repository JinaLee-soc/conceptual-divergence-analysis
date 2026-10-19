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
    """NFKC, casefolded, Latin accents removed: one spelling for NFC and NFD names.

    Only marks on Latin letters are accents; a Japanese dakuten or a
    Cyrillic breve changes the letter itself, so it is kept.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    kept: list[str] = []
    base_is_latin = False
    for ch in unicodedata.normalize("NFKD", text):
        if not unicodedata.combining(ch):
            base_is_latin = unicodedata.name(ch, "").startswith("LATIN")
        elif base_is_latin:
            continue
        kept.append(ch)
    # Recompose so Hangul jamo and kana with dakuten become one character again.
    return unicodedata.normalize("NFC", "".join(kept))


def title_tokens(text: str) -> set[str]:
    """Content tokens of a title or file stem: no English stop words, no short tokens.

    Tokens are runs of Unicode letters and digits, so accented and non-Latin
    titles keep whole words. The three-character minimum applies to ASCII
    tokens only: a two-syllable Korean or two-character CJK word is a word.
    """
    tokens = set(re.findall(r"[^\W_]+", _fold(text)))
    return {t for t in tokens if len(t) > 2 or not t.isascii()} - ENGLISH_STOP_WORDS


def _fold_name(name: str) -> str:
    """NFC and casefolded, so a pattern matches NFC and NFD spellings of a file name alike."""
    return unicodedata.normalize("NFC", name).casefold()


def _unusable_titles_error(unusable: list[tuple[str, set[str]]], min_overlap: int) -> ConfigError:
    """Name each title that can never match and how many usable words it has."""
    detail = "; ".join(f"{title!r} has {len(tokens)}" for title, tokens in unusable)
    short_latin = any(
        t.isascii() and len(t) <= 2
        for title, _ in unusable
        for t in re.findall(r"[^\W_]+", _fold(title))
    )
    not_counted = (
        "English stop words and Latin-script words under three letters are not counted"
        if short_latin
        else "English stop words are not counted"
    )
    return ConfigError(
        f"mywork.publication_titles entries have fewer usable words than "
        f"title_match_min_overlap={min_overlap}, so they could never match a file name "
        f"({detail}). Words are runs of letters and digits between spaces or punctuation, "
        f"so a title written without spaces is one word; {not_counted}. Add a "
        "filename_patterns entry for those papers (for example '*keyword*'), or lower "
        "title_match_min_overlap."
    )


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
    unusable = [(t, tok) for t, tok in zip(titles, tokens, strict=True) if len(tok) < min_overlap]
    if unusable:
        raise _unusable_titles_error(unusable, min_overlap)
    patterns = [
        _fold_name(str(p).strip()) for p in _as_list(mywork_config.get("filename_patterns"))
    ]
    return {
        "priority": priority,
        "patterns": [p for p in patterns if p],
        "title_tokens": tokens,
        "min_overlap": min_overlap,
        "min_ratio": min_ratio,
    }


def _filename_patterns(rules: dict, filename: str, metadata: dict) -> str | None:
    path = Path(_fold_name(filename))
    names = (path.name, path.stem)
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
