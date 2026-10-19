"""Quotation-grounding check — a computed anti-hallucination guard.

The strongest *automatic* hallucination check available in this
architecture: every passage the LLM quotes in its report must occur
verbatim in the prompt it was given. The assembled user prompt is saved for
every run, so the check is fully deterministic and costs nothing.

Interpretation: an unverified quotation is not necessarily fabricated — the
model may paraphrase, splice, or lightly edit — but unverified spans are
exactly where a human should compare the report against the saved
``drift_<term>_<timestamp>_prompt.md`` before trusting a quoted claim. The
report lists them for that purpose.
"""

from __future__ import annotations

import re
import unicodedata

# Double quotes only (straight or curly): single quotes collide with
# apostrophes far too often to be a reliable quotation marker. A quote may
# soft-wrap across a single newline but never across a blank line. A curly
# pair may enclose straight inner quotes; otherwise every mark is paired
# left to right, and even empty pairs are consumed so pairing stays aligned.
# The min_words filter, not the regex, decides what counts as a citation.
_SPAN = r"(?:[^{excluded}\n]|\n(?!\s*\n)){{0,4000}}?"
_QUOTE_RE = re.compile(
    "“(?P<curly>" + _SPAN.format(excluded="“”") + ")”"
    '|["“](?P<plain>' + _SPAN.format(excluded='"“”') + ')["”]'
)
_ELLIPSIS_RE = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…")
_QUOTE_MARK_RE = re.compile(r'["“”]')
_ANY_QUOTE_MARK_RE = re.compile(r"[\"“”'‘’]")
AMBIGUOUS_PREFIX = "[ambiguous nesting, not checked] "


def _normalize(text: str) -> str:
    """Whitespace/typography-insensitive form for substring matching.

    All quote marks and apostrophes map to one character, so a quotation whose
    inner quotes changed style still matches the passage it reproduces.
    """
    text = _ANY_QUOTE_MARK_RE.sub("'", text)
    text = _ELLIPSIS_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_latin_letter(char: str) -> bool:
    return char.isalpha() and unicodedata.name(char, "").startswith("LATIN")


def _is_ambiguous_pair(body: str, match: re.Match) -> bool:
    """A pair whose closing mark runs into a word, or whose opening mark follows
    one, is probably an inner quote paired with an outer one; text between the
    two pairs would then escape checking.

    Only a Latin letter touching the outside of a mark signals that nesting.
    Footnote digits ('"…"1', '"…"¹'), markdown underscores ('_"…"_') and
    letters of other scripts (a Korean particle: '“…”라고') are ordinary
    neighbours of a complete quotation.
    """
    before = body[match.start() - 1] if match.start() > 0 else ""
    after = body[match.end()] if match.end() < len(body) else ""
    return _is_latin_letter(before) or _is_latin_letter(after)


def _find_fragment(fragment: str, text: str, cursor: int) -> int:
    """End offset of ``fragment`` in ``text`` at or after cursor, on word boundaries; -1 if absent."""
    pattern = re.escape(fragment)
    if re.match(r"\w", fragment[0]):
        pattern = r"(?<!\w)" + pattern
    if re.match(r"\w", fragment[-1]):
        pattern += r"(?!\w)"
    found = re.compile(pattern).search(text, cursor)
    return found.end() if found else -1


def _is_grounded(fragments: list[str], passages: list[tuple[str, str]], source: str | None) -> bool:
    """True when one passage (from the cited source, if any) holds every fragment in order."""
    if not fragments:
        return False
    for candidate_source, text in passages:
        if source and source != candidate_source:
            continue
        cursor = 0
        for fragment in fragments:
            cursor = _find_fragment(fragment, text, cursor)
            if cursor < 0:
                break
        else:
            return True
    return False


def _without_final_punctuation(fragments: list[str]) -> list[str] | None:
    """The fragments with the quotation's final . ? ! removed, or None if it has none.

    Writers often close a quotation with sentence punctuation the source has
    elsewhere ('"…fields." Then' for '…fields, and …'); the retry still needs a
    whole-word match.
    """
    if not fragments:
        return None
    last = fragments[-1].rstrip(".?!").rstrip()
    if last == fragments[-1]:
        return None
    return fragments[:-1] + [last] if last else None


def _is_grounded_allowing_final_punctuation(
    fragments: list[str], passages: list[tuple[str, str]], source: str | None
) -> bool:
    if _is_grounded(fragments, passages, source):
        return True
    trimmed = _without_final_punctuation(fragments)
    return trimmed is not None and _is_grounded(trimmed, passages, source)


_PASSAGE_RE = re.compile(
    r'Sample \d+ \[[^\]]+\] \[(?P<source>[^\]]+)\]: ["“](?P<text>.*?)["”](?=\s*(?:Sample \d+|\n|$))',
    re.DOTALL,
)
_SOURCE_SUFFIX = re.compile(
    r"^\s*[-–—]\s*\*?\[?([^\]\n*]+?\.(?:pdf|md|txt|docx|abstract))\]?\*?", re.IGNORECASE
)


def _preview(quote: str) -> str:
    return quote if len(quote) <= 110 else quote[:107] + "…"


def check_quotations(llm_body: str, user_prompt: str, *, min_words: int = 6) -> dict:
    """Check quotations against a single representative and its cited source.

    Labels shorter than min_words are omitted unless a source is attached.
    A composed quotation cannot borrow fragments from different passages, and
    fragments match on word boundaries. A pair with ambiguous nesting is never
    checked and counts as unverified, whatever its length.
    This checks textual grounding, not whether the interpretation is sound.
    """
    passages = [(m["source"], _normalize(m["text"])) for m in _PASSAGE_RE.finditer(user_prompt)]
    n_quotes = n_verified = n_ambiguous = 0
    n_marks_paired = 0
    unverified = []
    for match in _QUOTE_RE.finditer(llm_body):
        n_marks_paired += 2
        quote = match.group("curly") if match.group("curly") is not None else match.group("plain")
        if _is_ambiguous_pair(llm_body, match):
            n_quotes += 1
            n_ambiguous += 1
            unverified.append(AMBIGUOUS_PREFIX + _preview(quote))
            continue
        suffix = _SOURCE_SUFFIX.match(llm_body[match.end() :])
        source = suffix.group(1).strip() if suffix else None
        if not source and len(quote.split()) < min_words:
            continue
        fragments = [_normalize(x) for x in _ELLIPSIS_RE.split(quote) if x.strip()]
        n_quotes += 1
        if _is_grounded_allowing_final_punctuation(fragments, passages, source):
            n_verified += 1
            # Straight inner quotes of a verified curly pair were checked with it.
            n_marks_paired += len(_QUOTE_MARK_RE.findall(quote))
        else:
            unverified.append(_preview(quote))
    unpaired = max(0, len(_QUOTE_MARK_RE.findall(llm_body)) - n_marks_paired)
    return {
        "n_quotes": n_quotes,
        "n_verified": n_verified,
        "n_ambiguous": n_ambiguous,
        "unverified": unverified,
        "unpaired_marks": unpaired,
    }


def render_grounding_markdown(grounding: dict) -> str:
    n_quotes = grounding.get("n_quotes", 0)
    n_verified = grounding.get("n_verified", 0)
    unverified = grounding.get("unverified", [])
    unpaired = grounding.get("unpaired_marks", 0)

    if n_quotes == 0 and not unpaired:
        return (
            "*No source-attributed or substantive quotations (≥6 words) found; nothing to verify.*"
        )

    lines = [
        f"**{n_verified}/{n_quotes}** quoted passages verified verbatim against "
        "individual prompt passages (typography-normalized; cited filenames checked when supplied)."
    ]
    if unverified:
        lines += [
            "",
            f"⚠️ **{len(unverified)} quotation(s) could not be verified**. Check "
            "them against the saved `_prompt.md` file before relying on them:",
            "",
        ]
        lines += [f"- “{preview}”" for preview in unverified]
        lines += [
            "",
            "*Unverified does not mean fabricated (the model may paraphrase or "
            "splice), but these spans deserve a manual look.*",
        ]
    n_ambiguous = grounding.get("n_ambiguous", 0)
    if n_ambiguous:
        lines += [
            "",
            f"⚠️ **{n_ambiguous} quotation(s) had ambiguous nesting** (a quote mark "
            "directly touching a word, as when a quoted passage contains its own "
            "quotes). They were NOT checked and count as unverified; the text "
            "between such marks may not come from the prompt.",
        ]
    if unpaired:
        lines += [
            "",
            f"⚠️ **{unpaired} quotation mark(s) fell outside any checkable "
            "pair** (an unusually long quotation, a quote crossing a blank "
            "line, or an unbalanced mark). Whatever they enclose was NOT "
            "checked — compare those spans against the saved `_prompt.md`.",
        ]
    if not unverified and not unpaired:
        lines += [
            "",
            "*Every checked quotation occurs in one representative passage "
            "in the material it was shown; short unattributed labels are excluded.*",
        ]
    return "\n".join(lines)
