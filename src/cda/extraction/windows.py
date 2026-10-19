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
# "+blocks-v3": long texts are segmented in independent blocks (see
# ``split_sentences``), which can place boundaries differently from one pysbd
# pass over the whole text; CJK full stops end sentences without a following
# space, and a block without any sentence end is cut at its size bound.
SENTENCE_SEGMENTER = f"pysbd-{version('pysbd')}+blocks-v3"

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
# Chinese and Japanese end sentences with full-width marks and no space after
# them, so a boundary after one of these needs no whitespace.
_CJK_SENTENCE_END = re.compile(
    r"[\u3002\uff01\uff1f\uff0e][\"'\u201d\u2019)\]\uff09\u300d\u300f\u3011\u300b\u3009]*\Z"
)


def _new_segmenter() -> pysbd.Segmenter:
    return pysbd.Segmenter(language="en", clean=False, char_span=True)


def _after_cjk_end(text: str, position: int) -> bool:
    """Whether ``text[:position]`` ends with a CJK sentence end, closing marks and one space allowed."""
    position -= position > 0 and text[position - 1] == " "
    return bool(_CJK_SENTENCE_END.search(text, max(0, position - 16), position))


def _is_boundary(text: str, position: int) -> bool:
    """A sentence may start at ``position``: after whitespace or right after a CJK full stop."""
    return position > 0 and (text[position - 1] == " " or _after_cjk_end(text, position))


def _cut_point(block: str, limit: int) -> int | None:
    """Where to cut ``block`` so the head holds about ``limit`` characters; ``None`` if nowhere.

    Use the last sentence start pysbd finds near the bound after sentence-final
    punctuation (a CJK full stop needs no space after it). Search farther for
    a long sentence; ``None`` means the rest of the block has no sentence end.
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
            if cut > 0 and _after_cjk_end(block, cut):
                return cut
    return None


def _block_spans(text: str, limit: int = _BLOCK_CHARS) -> list[tuple[int, int, bool]]:
    """(start, end, forced) of each block of whitespace-normalized ``text``.

    When the rest of the text has no sentence end, it is cut at the bound so
    pysbd never sees more than ``limit`` characters of it. Such a forced cut
    is not a sentence boundary: ``forced`` marks a block whose first sentence
    continues the previous block's last one.
    """
    spans = []
    start, forced, exhausted = 0, False, False
    while len(text) - start > limit:
        cut = None if exhausted else _cut_point(text[start:], limit)
        # Once a search finds no sentence end, none lies further on either.
        exhausted = cut is None
        end = start + (limit if exhausted else cut)
        spans.append((start, end - (text[end - 1] == " "), forced))
        start, forced = end + (end < len(text) and text[end] == " "), exhausted
    if start < len(text):
        spans.append((start, len(text), forced))
    return spans


def _blocks(text: str, limit: int = _BLOCK_CHARS) -> list[str]:
    """Normalized blocks ending at detected sentence starts.

    A blank line can be a PDF page break within a sentence. It is normalized
    like other whitespace. The limit is a performance target, never grounds
    for inventing a sentence boundary.
    """
    rest = normalize_text(text)
    return [rest[a:b] for a, b, _ in _block_spans(rest, limit)]


def _segment_block(text: str) -> list[tuple[int, int]]:
    """(start, end) of the sentences of one whitespace-normalized block."""
    spans = _new_segmenter().segment(text)
    pieces = [span.sent.strip() for span in spans if span.sent.strip()]
    starts = _sentence_starts(text, pieces)
    if not starts or starts[0] > 0:
        starts.insert(0, 0)
    bounds = [b for b in starts[1:] if _is_boundary(text, b)]
    edges = [0, *bounds, len(text)]
    return [(a, b - (text[b - 1] == " ")) for a, b in zip(edges, edges[1:], strict=False)]


def sentence_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Whitespace-normalized ``text`` and the (start, end) of each of its sentences.

    Neighbouring sentences are separated by one space, or by nothing after a
    CJK full stop. See ``split_sentences`` for the invariants.
    """
    normalized = normalize_text(text)
    sentences: list[tuple[int, int]] = []
    for block_start, block_end, forced in _block_spans(normalized):
        block = normalized[block_start:block_end]
        for i, (a, b) in enumerate(_segment_block(block)):
            a, b = a + block_start, b + block_start
            joins_previous = forced and i == 0
            if sentences and (joins_previous or not _WORD_CHAR.search(normalized[a:b])):
                sentences[-1] = (sentences[-1][0], b)
            else:
                sentences.append((a, b))
    return normalized, sentences


def split_sentences(text: str) -> list[str]:
    """Sentences of whitespace-normalized ``text``; together they hold every character.

    Invariants: no character is dropped (punctuation pysbd leaves uncovered
    joins the preceding sentence, or stands alone at the start; an uncovered
    letter or digit raises), and boundaries fall only at whitespace or right
    after a CJK full stop (pysbd can split inside a token such as ".NET"; such
    pieces are rejoined). A segmenter holds per-call state, so each call
    builds its own.

    pysbd runs separately on blocks of about ``_BLOCK_CHARS`` characters cut
    after a detected sentence end. An unusually long sentence may exceed the
    target size; neither a blank line nor the size limit splits it. Text with
    no sentence end at all is handed to pysbd in pieces of at most the block
    size, and the pieces are rejoined.
    """
    normalized, spans = sentence_spans(text)
    return [normalized[a:b] for a, b in spans]


def literal_pattern(term: str) -> str:
    r"""The escaped NFC term; any run of whitespace between its words matches ``\s+``.

    A phrase typed with a double space or a no-break space still matches the
    corpus, whose whitespace is normalized to single spaces.
    """
    term = unicodedata.normalize("NFC", term)
    words = term.split()
    return r"\s+".join(re.escape(word) for word in words) if words else re.escape(term)


def term_regex(term: str) -> re.Pattern[str]:
    r"""Whole-word, case-insensitive match (plain substring for non-word terms).

    The term is NFC-normalized, matching the NFC corpus and draft text, and
    the whitespace between its words matches any whitespace (see
    ``literal_pattern``). Each ``\b`` is applied only where the term's edge
    character is a word character, so terms like "C++" or ".NET" stay
    matchable. A term without ASCII letters or digits (e.g. 참신성) is a plain
    substring, because agglutinative languages attach particles directly to
    the term.
    """
    escaped = literal_pattern(term)
    term = unicodedata.normalize("NFC", term).strip() or term
    if not re.search(r"[A-Za-z0-9]", term):
        return re.compile(escaped, flags=re.IGNORECASE)
    lead = r"\b" if re.match(r"\w", term) else ""
    trail = r"\b" if re.search(r"\w\Z", term) else ""
    return re.compile(rf"{lead}{escaped}{trail}", flags=re.IGNORECASE)


def term_identity(term: str) -> str:
    """Compare saved terms without changing punctuation or literal word order."""
    return unicodedata.normalize("NFC", " ".join(term.split())).lower()


def safe_term_slug(term: str) -> str:
    """Filesystem-safe, deterministic display slug.

    The term is NFC-normalized first, so composed and decomposed spellings of
    one term share a slug. A term with no ASCII letters or digits gets a short
    content hash instead of an empty slug. Different terms can share a slug
    (C++ and C#, for example); result lookup must compare the saved term too.
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
