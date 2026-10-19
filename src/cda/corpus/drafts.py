"""Researcher draft loading (.docx, .md, .txt, .pdf).

Drafts are read at analysis time from the configured drafts directory and are
always classified as researcher-authored — the "direct draft inclusion"
signal of MyWork resolution (paper §3.4). Every draft is read whole; a draft
that cannot be read stops the run. Nothing here writes to the corpus.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path

from cda.config import ConfigError
from cda.extraction.windows import split_sentences

SUPPORTED_EXTENSIONS = {".docx", ".md", ".txt", ".pdf"}

# Placeholders docx2python writes into extracted text (note references, note
# labels, images, hyperlinks); they are markup, not the researcher's prose.
_DOCX2PYTHON_MARKUP = (
    (re.compile(r"----(?:footnote|endnote)-?\d+----"), ""),
    (re.compile(r"(?m)^(?:footnote|endnote)-?\d+\)\t"), ""),
    (re.compile(r"----Image alt text---->[^<]*<"), ""),
    (re.compile(r"----media/[^\s]*?----"), ""),
    (re.compile(r'<a href="[^"]*">(.*?)</a>', re.DOTALL), r"\1"),
    # A symbol-font character (Word's <w:sym>, e.g. Symbol or Wingdings):
    # docx2python wraps an HTML character reference in a span. Only the span
    # content is unescaped; the researcher's own "&amp;" or "<b>" stays as typed.
    (
        re.compile(r"<span style=font-family:[^<>]*>([^<>]*)</span>"),
        lambda match: html.unescape(match.group(1)),
    ),
)

# Draft formats CDA does not read; skipping one is announced, while stray
# files (.DS_Store, images, .bib) are skipped quietly.
_UNSUPPORTED_DRAFT_EXTENSIONS = {
    ".doc": "save it as .docx (in Word: File → Save As)",
    ".rtf": "save it as .docx or plain .txt",
    ".odt": "save it as .docx (in LibreOffice: File → Save As)",
    ".pages": "export it as .docx or .pdf (in Pages: File → Export To)",
}


def _normalize_text(text: str) -> str:
    """Collapsed whitespace, NFC — the same normal form as ingested corpus text."""
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", text).strip())


def _read_docx(path: Path) -> str:
    """Body (tables and text boxes included), footnotes, and endnotes of a .docx.

    Headers, footers, and comments are not part of the manuscript text.
    """
    from docx2python import docx2python
    from docx2python.iterators import iter_paragraphs

    with docx2python(path) as content:
        parts = [content.body, content.footnotes, content.endnotes]
        text = "\n".join(paragraph for part in parts for paragraph in iter_paragraphs(part))
    for pattern, replacement in _DOCX2PYTHON_MARKUP:
        text = pattern.sub(replacement, text)
    return text


def _decode_bomless_utf16(raw: bytes, path: Path, progress: Callable[[str], None]) -> str:
    """Decode UTF-16 without a byte-order mark, guessing the byte order.

    ASCII-heavy UTF-16 has a NUL beside every character; which side of each
    byte pair holds the NUL gives the byte order. The guess is announced.
    """
    head = raw[:2000]
    pairs = range(0, len(head) - 1, 2)
    nul_first = sum(1 for i in pairs if head[i] == 0)
    nul_second = sum(1 for i in pairs if head[i + 1] == 0)
    codec = "utf-16-be" if nul_first > nul_second else "utf-16-le"
    progress(
        f"  [draft] {path.name} has no byte-order mark but looks like UTF-16 "
        f"(a NUL between every character); reading it as {codec} — re-save it "
        "as UTF-8 (plain text) if the text below looks wrong"
    )
    return raw.decode(codec, errors="replace")


def _read_text_draft(path: Path, progress: Callable[[str], None]) -> str:
    """Read a .txt/.md draft, honoring UTF-8 and UTF-16 (with or without a BOM)."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")  # the BOM states the byte order; nothing to guess
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        if b"\x00" in raw[:1000]:  # BOM-less UTF-16 (rare but real)
            return _decode_bomless_utf16(raw, path, progress)
        progress(
            f"  [draft] {path.name} is not UTF-8; reading with replacement "
            "characters — re-save it as UTF-8 (plain text) for a clean read"
        )
        return raw.decode("utf-8", errors="replace")
    if "\x00" in text[:2000]:
        # BOM-less UTF-16 of ASCII text is byte-valid UTF-8 with a NUL
        # between every character — the exact silent-garbage case.
        return _decode_bomless_utf16(raw, path, progress)
    return text


def _read_pdf(path: Path) -> str:
    import fitz  # pymupdf

    with fitz.open(str(path)) as doc:
        return "\n".join(page.get_text("text") for page in doc)


# Share of sentences two drafts may have in common before they count as one
# document in two files (a .docx and its PDF export).
NEAR_DUPLICATE_OVERLAP = 0.9
# Below this many distinct sentences, shared sentences are not evidence of one
# document (two notes can both say "Novelty."); only identical text is refused.
NEAR_DUPLICATE_MIN_SENTENCES = 3
_SHOWN_TEXT_CHARS = 80


def _sentence_keys(text: str) -> set[str]:
    """Hashes of the draft's sentences, by their letters only.

    Case, punctuation, spacing, and digits are ignored, so page numbers a PDF
    export adds (alone or glued to a sentence) do not make sentences differ.
    """
    keys = set()
    for sentence in split_sentences(text):
        letters = "".join(re.findall(r"[^\W\d_]+", sentence.casefold()))
        if letters:
            keys.add(hashlib.sha256(letters.encode("utf-8")).hexdigest())
    return keys


def _identical_drafts_error(first: str, second: str, text: str) -> ConfigError:
    shown = f" (both contain only {text!r})" if len(text) <= _SHOWN_TEXT_CHARS else ""
    return ConfigError(
        f"Drafts '{first}' and '{second}' have identical text{shown}, so every term "
        "sentence in them would be counted twice. Remove one copy from the drafts "
        "directory and rerun."
    )


def _refuse_duplicate_drafts(texts: dict[str, str]) -> None:
    """Two files with the same or nearly the same text would count researcher sentences twice.

    Nearly the same: Jaccard overlap of their sentence sets (``_sentence_keys``)
    of at least ``NEAR_DUPLICATE_OVERLAP``, for drafts that each have at least
    ``NEAR_DUPLICATE_MIN_SENTENCES`` distinct sentences.
    """
    first_by_text: dict[str, str] = {}
    for name, text in texts.items():
        if text in first_by_text:
            raise _identical_drafts_error(first_by_text[text], name, text)
        first_by_text[text] = name
    keyed = [(name, _sentence_keys(text)) for name, text in texts.items()]
    keyed = [(name, keys) for name, keys in keyed if len(keys) >= NEAR_DUPLICATE_MIN_SENTENCES]
    for i, (first, first_keys) in enumerate(keyed):
        for second, second_keys in keyed[i + 1 :]:
            overlap = len(first_keys & second_keys) / len(first_keys | second_keys)
            if overlap >= NEAR_DUPLICATE_OVERLAP:
                raise ConfigError(
                    f"Drafts '{first}' and '{second}' share {overlap:.0%} of their "
                    "sentences (ignoring case, punctuation, and numbers such as page "
                    "numbers), so they look like one document saved twice (for example "
                    "a .docx and its PDF export), and every term sentence in them would "
                    "be counted twice. Keep one of them in the drafts directory and rerun."
                )


def normalize_drafts(texts: dict[str, str]) -> dict[str, str]:
    """Give supplied drafts the same normalization and duplicate checks as files."""
    if any(
        not isinstance(name, str) or not name.strip() or not isinstance(text, str)
        for name, text in texts.items()
    ):
        raise ConfigError("Drafts must map nonempty source names to text strings.")
    normalized = {name: _normalize_text(text) for name, text in texts.items() if text.strip()}
    _refuse_duplicate_drafts(normalized)
    return normalized


def load_drafts(
    drafts_dir: str | Path, *, progress: Callable[[str], None] = print
) -> dict[str, str]:
    """Read every supported draft under ``drafts_dir`` (recursive).

    Returns {relative filename: normalized full text}. Missing directory is
    not an error — running without drafts is a supported mode; a note is
    emitted so the absence is visible. A draft that cannot be parsed, or two
    drafts with identical text, raise ``ConfigError``.
    """
    drafts_dir = Path(drafts_dir).expanduser()
    if not drafts_dir.is_dir():
        progress(f"[note] drafts directory not found ({drafts_dir}); analyzing without drafts")
        return {}

    texts: dict[str, str] = {}
    for path in sorted(drafts_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("~$"):
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            fix = _UNSUPPORTED_DRAFT_EXTENSIONS.get(suffix)
            if fix:
                progress(
                    f"  [draft] skipped {path.relative_to(drafts_dir)}: "
                    f"{suffix} is not supported — {fix}. Supported formats: "
                    ".docx, .md, .txt, .pdf."
                )
            continue
        rel_name = str(path.relative_to(drafts_dir))
        try:
            if suffix == ".docx":
                full = _read_docx(path)
            elif suffix == ".pdf":
                full = _read_pdf(path)
            else:
                full = _read_text_draft(path, progress)
        except Exception as exc:
            raise ConfigError(
                f"Could not read draft {path} ({type(exc).__name__}: {exc}). Repair or "
                "re-export the file, or move it out of the drafts directory, and rerun."
            ) from exc
        full = _normalize_text(full)
        if full:
            texts[rel_name] = full
            progress(f"  [draft] loaded {rel_name} ({len(full.split())} words)")
        else:
            progress(
                f"  [draft] {rel_name} contains no extractable text (an empty "
                "file, or a scanned PDF without OCR?) — it will not count as "
                "researcher writing in this analysis"
            )
    _refuse_duplicate_drafts(texts)
    return texts
