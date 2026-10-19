"""Term-sentence extraction.

The sentence containing the target term is the unit of analysis (paper §3.2).
Sentences come from pysbd's rule-based English segmenter. The default
preserves each complete term-bearing sentence occurrence; explicit nonzero
character limits enable legacy context expansion/truncation.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from importlib.metadata import version

import pysbd

from cda.config import ConfigError

# Stored with every ingested chunk; analysis refuses chunks segmented by any
# other segmenter, because sentence positions would not line up.
# "+blocks-v2": long texts are segmented in independent blocks (see
# ``split_sentences``), which can place boundaries differently from one pysbd
# pass over the whole text.
SENTENCE_SEGMENTER = f"pysbd-{version('pysbd')}+blocks-v2"

# pysbd's run time grows quadratically with input length, so text is segmented
# near this character target, extending it when a sentence is longer.
_BLOCK_CHARS = 10_000
# How far back from the block bound a sentence boundary is looked for.
_CUT_SEARCH_CHARS = 2_000


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


_WORD_CHAR = re.compile(r"[^\W_]")  # a letter or digit in any script


def _sentence_starts(text: str, pieces: list[str]) -> list[int]:
    """Where each pysbd piece begins in ``text``, located in order.

    pysbd's own offsets can overlap and its pieces can omit trailing
    punctuation (" . . .", a detached "?!"), so pieces are found by text.
    Text between pieces that pysbd left out may only be punctuation or
    whitespace; losing a letter or digit raises.
    """
    starts: list[int] = []
    cursor = 0
    found = 0
    for piece in pieces:
        found = text.find(piece, cursor)
        if found < 0 or _WORD_CHAR.search(text[cursor:found]):
            found = -1
            break
        starts.append(found)
        cursor = found + len(piece)
    if found < 0 or _WORD_CHAR.search(text[cursor:]):
        raise ValueError(
            f"Sentence segmentation ({SENTENCE_SEGMENTER}) did not cover the text "
            f"beginning {text[:80]!r}; no sentence may be dropped."
        )
    return starts


_SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]*\Z")


def _new_segmenter() -> pysbd.Segmenter:
    return pysbd.Segmenter(language="en", clean=False, char_span=True)


def _cut_point(block: str, limit: int) -> int:
    """Where to cut ``block`` so the head holds at most about ``limit`` characters.

    Use the last sentence start pysbd finds near the bound after sentence-final
    punctuation. Search farther for a long sentence; retain the text whole if
    no boundary is found.
    """
    # Extend the search when one sentence runs past the target, rather than
    # handing the entire remaining document to the quadratic segmenter.
    for high in range(limit, len(block) + _CUT_SEARCH_CHARS, _CUT_SEARCH_CHARS):
        high = min(high, len(block))
        low = max(0, high - _CUT_SEARCH_CHARS)
        window = block[low:high]
        cursor, candidates = 0, []
        for span in _new_segmenter().segment(window):
            piece = span.sent.strip()
            found = window.find(piece, cursor) if piece else -1
            if found < 0:
                break
            candidates.append(found)
            cursor = found + len(piece)
        for start in reversed(candidates):
            cut = low + start
            if cut > 0 and block[cut - 1] == " " and _SENTENCE_END.search(block, low, cut - 1):
                return cut
    # Neither a size limit nor whitespace alone establishes a sentence end.
    # Keep an unusually long sentence whole, even if it exceeds the target.
    return len(block)


def _blocks(text: str, limit: int = _BLOCK_CHARS) -> list[str]:
    """Normalized blocks ending at detected sentence starts.

    A blank line can be a PDF page break within a sentence. It is normalized
    like other whitespace. The limit is a performance target, never grounds
    for inventing a sentence boundary.
    """
    blocks = []
    rest = normalize_text(text)
    while len(rest) > limit:
        cut = _cut_point(rest, limit)
        blocks.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        blocks.append(rest)
    return blocks


def _segment_block(text: str) -> list[str]:
    """Sentences of one whitespace-normalized block (see ``split_sentences``)."""
    spans = _new_segmenter().segment(text)
    pieces = [span.sent.strip() for span in spans if span.sent.strip()]
    starts = _sentence_starts(text, pieces)
    if not starts or starts[0] > 0:
        starts.insert(0, 0)
    bounds = [b for b in starts[1:] if text[b - 1] == " "]
    edges = [0, *bounds, len(text)]
    return [text[a:b].strip() for a, b in zip(edges, edges[1:], strict=False)]


def split_sentences(text: str) -> list[str]:
    """Sentences of whitespace-normalized ``text``; together they hold every character.

    Invariants: no character is dropped (punctuation pysbd leaves uncovered
    joins the preceding sentence, or stands alone at the start; an uncovered
    letter or digit raises), and boundaries fall only at whitespace (pysbd
    can split inside a token such as ".NET"; such pieces are rejoined), so
    each sentence is a run of whole words. A segmenter holds per-call state,
    so each call builds its own.

    pysbd runs separately on blocks of about ``_BLOCK_CHARS`` characters cut
    after a detected sentence end. An unusually long sentence may exceed the
    target size; neither a blank line nor the size limit splits it.
    """
    sentences: list[str] = []
    for block in _blocks(text):
        block_sentences = _segment_block(block)
        for sentence in block_sentences:
            if sentences and not _WORD_CHAR.search(sentence):
                sentences[-1] = f"{sentences[-1]} {sentence}"
            else:
                sentences.append(sentence)
    return sentences


def term_regex(term: str) -> re.Pattern[str]:
    r"""Whole-word, case-insensitive match (plain substring for non-word terms).

    The term is NFC-normalized, matching the NFC corpus and draft text. Each
    ``\b`` is applied only where the term's edge character is a word
    character, so terms like "C++" or ".NET" stay matchable. A term without
    ASCII letters or digits (e.g. 참신성) is a plain substring, because
    agglutinative languages attach particles directly to the term.
    """
    term = unicodedata.normalize("NFC", term)
    escaped = re.escape(term)
    if not re.search(r"[A-Za-z0-9]", term):
        return re.compile(escaped, flags=re.IGNORECASE)
    lead = r"\b" if re.match(r"\w", term) else ""
    trail = r"\b" if re.search(r"\w\Z", term) else ""
    return re.compile(rf"{lead}{escaped}{trail}", flags=re.IGNORECASE)


def safe_term_slug(term: str) -> str:
    """Filesystem-safe, deterministic slug; distinct terms get distinct slugs.

    The term is NFC-normalized first, so composed and decomposed spellings of
    one term share a slug. A term with no ASCII letters or digits gets a short
    content hash instead of an empty slug.
    """
    term = unicodedata.normalize("NFC", term)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", term).strip("_").lower()
    if slug:
        return slug
    digest = hashlib.sha256(term.encode("utf-8")).hexdigest()[:8]
    return f"term_{digest}"


def extract_term_occurrences(
    text: str,
    regex: re.Pattern[str],
    *,
    window_mode: str = "sentence",
    min_window_chars: int = 0,
    max_window_chars: int = 0,
) -> list[tuple[int, str]]:
    """Term-bearing sentence positions and windows, preserving real repetitions."""
    return sentence_windows(
        split_sentences(text),
        regex,
        window_mode=window_mode,
        min_window_chars=min_window_chars,
        max_window_chars=max_window_chars,
    )


def sentence_windows(
    sentences: list[str],
    regex: re.Pattern[str],
    *,
    window_mode: str = "sentence",
    min_window_chars: int = 0,
    max_window_chars: int = 0,
) -> list[tuple[int, str]]:
    """``extract_term_occurrences`` over sentences that are already segmented."""
    windows: list[tuple[int, str]] = []

    for i, sentence in enumerate(sentences):
        if not regex.search(sentence):
            continue

        if window_mode == "sentence":
            window = normalize_text(sentence)
            if min_window_chars and len(window) < min_window_chars:
                # Too short to embed meaningfully — add ±1 sentence of context.
                window = normalize_text(" ".join(sentences[max(0, i - 1) : i + 2]))
        elif window_mode == "context":  # ±1 sentence around the hit
            window = normalize_text(" ".join(sentences[max(0, i - 1) : i + 2]))
        else:
            raise ValueError(
                f"retrieval.window_mode must be 'sentence' or 'context', got {window_mode!r}."
            )

        if max_window_chars and len(window) > max_window_chars:
            # An explicit legacy cap must not erase the literal match. Preserve
            # the whole term-bearing window rather than change the sample.
            candidate = window[: max_window_chars - 3] + "..."
            if regex.search(candidate):
                window = candidate
        windows.append((i, window))
    return windows


def window_settings(config) -> dict:
    """Window keyword arguments for ``extract_term_occurrences`` from config."""
    max_chars = config.get_int("retrieval.max_window_chars", minimum=0)
    # Truncation appends "..." after slicing to max_chars - 3, so a cap of 1-3
    # would lengthen the window it is meant to bound.
    if 0 < max_chars < 4:
        raise ConfigError("retrieval.max_window_chars must be 0 (unbounded) or at least 4.")
    return {
        "window_mode": str(config.get("retrieval.window_mode")).strip().lower(),
        "min_window_chars": config.get_int("retrieval.min_window_chars", minimum=0),
        "max_window_chars": max_chars,
    }


YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
# A new-style arXiv identifier (YYMM.NNNNN, optional version); its leading
# digits are not a year.
_ARXIV_ID = re.compile(r"(?<!\d)\d{4}\.\d{4,5}(?:v\d+)?(?!\d)")


def infer_year(filename: str, metadata: dict) -> str:
    """Publication year from chunk metadata, else the file name, else 'unknown'.

    arXiv identifiers in the file name ("1905.01234v2.pdf") are skipped.
    """
    for key in ("year", "publication_year", "date", "issued"):
        value = metadata.get(key)
        if not value:
            continue
        match = YEAR_PATTERN.search(str(value))
        if match:
            return match.group(0)
    match = YEAR_PATTERN.search(_ARXIV_ID.sub(" ", filename))
    return match.group(0) if match else "unknown"
