"""Corpus edge cases: file names, folders, legacy stores, and non-English text.

Each test pins one defect: exclude patterns and MyWork patterns missing
decomposed (NFD) names, a legacy root-unknown source blocking its own rename,
phrases typed with odd whitespace, context windows over a chunk overlap, an
unreadable folder crashing ingest, hidden draft files, PDF running heads,
generic Zotero titles, CJK segmentation, date-bound typos, Windows paths in a
Zotero export, docx list markers and hyperlinks, symlinked folders, and the
Zotero summary of an unchanged rerun.
"""

from __future__ import annotations

import csv
import os
import sys
import unicodedata
import zipfile
from pathlib import Path

import pysbd
import pytest

import cda.corpus.ingest as ing
from cda.config import ConfigError, load_config
from cda.corpus.drafts import _read_docx, load_drafts, strip_running_lines
from cda.corpus.ingest import get_collection, ingest_pdf_dir, ingest_records, select_pdfs
from cda.corpus.timeframe import parse_bound
from cda.corpus.zotero import (
    ZoteroRecord,
    build_title_index,
    load_zotero_csv,
    match_by_content,
    normalize_key,
)
from cda.extraction import windows
from cda.extraction.mywork import _fold, load_mywork_rules, mywork_classifier
from cda.extraction.retrieval import build_passages, keyword_hits, literal_filter
from cda.extraction.windows import split_sentences, term_regex

NFD = unicodedata.normalize("NFD", "Été")  # a folder name as macOS often stores it


def _quiet(_message: str) -> None:
    pass


def _body(seed: str, n: int = 60) -> str:
    return " ".join(f"Sentence {seed} number {i} talks about novelty in science." for i in range(n))


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A PDF folder whose '.pdf' files are plain text, read as their extracted text."""
    lib = tmp_path / "pdfs"
    lib.mkdir()
    monkeypatch.setattr(ing, "extract_pdf_text", lambda path: Path(path).read_text())
    return lib


def _stored(config) -> list[str]:
    metadatas = get_collection(config).get(include=["metadatas"])["metadatas"]
    return sorted({m["pdf_relpath"] for m in metadatas})


def _literature(config, term: str, mywork: dict | None = None):
    hits = keyword_hits(get_collection(config), term, progress=_quiet)
    return build_passages(term, hits, mywork_classifier(mywork or {}), config)


# --- exclude patterns match decomposed folder names ----------------------


def test_nfc_exclude_pattern_skips_an_nfd_folder_before_ingest(tmp_config, fake_embedder, library):
    (library / NFD).mkdir()
    (library / NFD / "a.pdf").write_text(_body("a"))
    (library / "b.pdf").write_text(_body("b"))
    messages: list[str] = []
    stats = ingest_pdf_dir(
        tmp_config,
        fake_embedder,
        library,
        exclude=[unicodedata.normalize("NFC", "Été")],
        prune_missing=True,
        progress=messages.append,
    )
    assert stats["pdfs_excluded"] == 1 and stats["pdfs_ingested"] == 1
    assert stats["pdfs_pruned"] == 0  # never ingested, so nothing to prune
    assert _stored(tmp_config) == ["b.pdf"]


def test_exclusion_compares_paths_and_patterns_in_nfc():
    assert ing._is_excluded(Path(NFD) / "a.pdf", [unicodedata.normalize("NFC", "Été")])
    assert ing._is_excluded(Path("Été") / "a.pdf", [NFD + "/"])


# --- MyWork filename patterns match decomposed names ---------------------


def test_pdf_filename_is_stored_nfc(tmp_config, fake_embedder, library):
    name = unicodedata.normalize("NFD", "이지나_참신성_2020.pdf")
    (library / name).write_text(_body("a"))
    ingest_pdf_dir(tmp_config, fake_embedder, library, progress=_quiet)
    metadata = get_collection(tmp_config).get(include=["metadatas"])["metadatas"][0]
    assert metadata["pdf_filename"] == unicodedata.normalize("NFC", name)


def test_filename_pattern_matches_nfd_names_and_ignores_case():
    classify = mywork_classifier({"filename_patterns": ["이지나*", "*CAFÉ*"]})
    assert classify(unicodedata.normalize("NFD", "이지나_참신성_2020.pdf"), {})[0]
    assert classify(unicodedata.normalize("NFD", "notes_café_2021.pdf"), {})[0]
    assert not classify("other.pdf", {})[0]


# --- sources stored before PDF roots were recorded -----------------------


def _strip_roots_and_boundaries(config, *, boundaries: bool) -> None:
    """Rewrite the store as an older version left it."""
    collection = get_collection(config)
    stored = collection.get(include=["metadatas"])
    patch = {"pdf_root": None}
    if boundaries:
        patch.update(sentence_context=None, sentence_start=None, sentence_lengths=None)
    collection.update(ids=stored["ids"], metadatas=[dict(patch) for _ in stored["ids"]])


@pytest.mark.parametrize("reingest", [False, True])
def test_renamed_legacy_root_unknown_source_is_replaced(
    tmp_config, fake_embedder, library, reingest
):
    (library / "old.pdf").write_text(_body("a"))
    ingest_pdf_dir(tmp_config, fake_embedder, library, progress=_quiet)
    _strip_roots_and_boundaries(tmp_config, boundaries=True)
    (library / "old.pdf").rename(library / "new.pdf")
    stats = ingest_pdf_dir(
        tmp_config, fake_embedder, library, reingest=reingest, prune_missing=True, progress=_quiet
    )
    assert stats["pdfs_renamed"] == 1 and stats["pdfs_duplicates_skipped"] == 0
    assert _stored(tmp_config) == ["new.pdf"]
    _, literature, _ = _literature(tmp_config, "novelty")
    assert literature and {p.source for p in literature} == {"new.pdf"}


def test_root_unknown_source_at_its_own_path_is_adopted(tmp_config, fake_embedder, library):
    (library / "a.pdf").write_text(_body("a"))
    ingest_pdf_dir(tmp_config, fake_embedder, library, progress=_quiet)
    _strip_roots_and_boundaries(tmp_config, boundaries=False)
    messages: list[str] = []
    stats = ingest_pdf_dir(tmp_config, fake_embedder, library, progress=messages.append)
    metadatas = get_collection(tmp_config).get(include=["metadatas"])["metadatas"]
    assert {m["pdf_root"] for m in metadatas} == {
        unicodedata.normalize("NFC", str(library.resolve()))
    }
    assert not any("root unknown" in m for m in messages)
    assert stats["pdfs_root_adopted"] == 1 and stats["pdfs_root_unknown"] == 0
    assert stats["pdfs_skipped"] == 1  # adopted, then reused: nothing re-embedded


def test_legacy_chunk_error_names_every_affected_source():
    hits = {
        f"{name}::c0": {
            "document": "Novelty cut here",
            "metadata": {"chunk_index": 0, "total_chunks": 1, "pdf_relpath": name},
        }
        for name in ("first.pdf", "sub/second.pdf")
    }
    with pytest.raises(ConfigError, match=r"(?s)2 source.*first\.pdf.*sub/second\.pdf"):
        build_passages("novelty", hits, mywork_classifier({}), load_config())


# --- phrases typed with a double space or a no-break space ---------------


@pytest.mark.parametrize("term", ["Matthew  effect", "Matthew\u00a0effect", " Matthew effect "])
def test_phrase_whitespace_matches_any_whitespace(tmp_config, fake_embedder, term):
    text = "The Matthew effect matters. Other words follow here. " * 3
    ingest_records(
        tmp_config, fake_embedder, [{"source": "doc", "abstract": text}], progress=_quiet
    )
    assert term_regex(term).search("the Matthew effect")
    assert term_regex(term).search("the Matthew\n\teffect")
    hits = keyword_hits(get_collection(tmp_config), term, progress=_quiet)
    _, literature, _ = build_passages(term, hits, mywork_classifier({}), tmp_config)
    assert len(literature) == 3


def test_literal_filter_allows_any_whitespace_between_words():
    assert literal_filter("novelty") == {"$regex": "(?i)novelty"}
    assert literal_filter("Matthew\u00a0 effect") == {"$regex": r"(?i)Matthew\s+effect"}


# --- context windows over a sentence spanning the chunk overlap ---------


def test_context_window_over_chunk_overlap_is_the_union(tmp_path, fake_embedder):
    pre = [f"Filler sentence {i} has six words." for i in range(65)]  # 390 words
    long = "Start " + " ".join("novelty" if i == 60 else f"w{i}" for i in range(118)) + " ends."
    post = [f"Tail sentence {i} has six words." for i in range(80)]
    config = load_config(
        overrides={
            "paths": {"corpus_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o")},
            "retrieval": {"window_mode": "context"},
        }
    )
    text = " ".join([*pre, long, *post])
    ingest_records(config, fake_embedder, [{"source": "doc", "abstract": text}], progress=_quiet)
    _, literature, stats = _literature(config, "novelty")
    assert [p.text for p in literature] == [f"{pre[-1]} {long} {post[0]}"]
    assert stats["overlap_occurrences_removed"] == 1


# --- an unreadable folder holding a stored source ------------------------


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions that apply to the current user",
)
def test_unreadable_folder_is_reported_not_pruned_or_renamed(tmp_config, fake_embedder, library):
    (library / "sub").mkdir()
    (library / "sub" / "a.pdf").write_text(_body("a"))
    (library / "b.pdf").write_text(_body("b"))
    ingest_pdf_dir(tmp_config, fake_embedder, library, progress=_quiet)
    (library / "c.pdf").write_text(_body("a"))  # the same bytes as sub/a.pdf
    os.chmod(library / "sub", 0)
    messages: list[str] = []
    try:
        stats = ingest_pdf_dir(
            tmp_config, fake_embedder, library, prune_missing=True, progress=messages.append
        )
    finally:
        os.chmod(library / "sub", 0o755)
    assert stats["pdfs_unchecked"] == 1 and stats["pdfs_pruned"] == 0
    assert stats["pdfs_duplicates_skipped"] == 1 and stats["pdfs_renamed"] == 0
    assert _stored(tmp_config) == ["b.pdf", "sub/a.pdf"]
    assert any("could not check; not pruned" in m and "sub/a.pdf" in m for m in messages)
    assert any(m.startswith("[done]") for m in messages)


# --- hidden files in the drafts directory --------------------------------


def test_hidden_and_appledouble_drafts_are_skipped_with_a_note(tmp_path):
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "paper.md").write_text("Novelty is our claim.")
    (drafts / "._paper.docx").write_bytes(b"\x00\x05\x16\x07 AppleDouble resource fork")
    (drafts / ".notes.md").write_text("Private novelty notes.")
    (drafts / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    messages: list[str] = []
    texts = load_drafts(drafts, progress=messages.append)
    assert list(texts) == ["paper.md"]
    assert any("skipped 2 hidden file(s)" in m for m in messages)


# --- a PDF export with running heads is a near-duplicate -----------------


def _pdf_export(path: Path, words: list[str], *, head: str = "NOVELTY AND RECOGNITION") -> None:
    import fitz

    doc = fitz.open()
    for number, start in enumerate(range(0, len(words), 250), start=1):
        page = doc.new_page()
        page.insert_text((72, 40), head, fontsize=9)
        body = " ".join(words[start : start + 250])
        assert page.insert_textbox(fitz.Rect(72, 60, 520, 780), body, fontsize=9) >= 0
        page.insert_text((290, 810), str(number), fontsize=9)
    doc.save(path)


def test_pdf_export_with_running_heads_is_refused_as_a_duplicate(tmp_path):
    import random

    rng = random.Random(1)
    vocab = "novelty science claims reviewers citation field journals reward risk".split()
    sentences = [
        " ".join(rng.choice(vocab) for _ in range(rng.randint(14, 24))).capitalize() + "."
        for _ in range(200)
    ]
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "paper.md").write_text(" ".join(sentences))
    _pdf_export(drafts / "paper.pdf", " ".join(sentences).split())
    with pytest.raises(ConfigError, match="paper.md.*paper.pdf"):
        load_drafts(drafts, progress=_quiet)


def test_running_lines_are_removed_from_pdf_draft_text(tmp_path):
    import random

    rng = random.Random(2)
    vocab = "novelty science claims reviewers citation field journals reward risk".split()
    words = [rng.choice(vocab) for _ in range(1000)]
    _pdf_export(tmp_path / "paper.pdf", words)
    text = load_drafts(tmp_path, progress=_quiet)["paper.pdf"]
    assert "RECOGNITION" not in text
    assert text == " ".join(words)  # page numbers are gone too


def test_strip_running_lines_keeps_lines_seen_on_fewer_pages():
    pages = ["Journal 1\nBody one.\n1", "Journal 2\nBody two.\n2", "Other head\nBody three.\n3"]
    assert strip_running_lines(pages) == [
        "Journal 1\nBody one.",
        "Journal 2\nBody two.",
        "Other head\nBody three.",
    ]


# --- generic library titles do not make every title match ambiguous -----


def test_generic_titles_are_not_title_keys():
    distinctive = ZoteroRecord("Novelty and Recognition in the Sciences of Today", "2020", "")
    generic = [ZoteroRecord(t, "2019", "") for t in ("Introduction", "BOOK  review", "Editorial")]
    index = build_title_index([distinctive, *generic])
    assert len(index) == 1
    opening = "Novelty and Recognition in the Sciences of Today. Introduction. Editorial note."
    assert match_by_content(opening, {}, index) == (distinctive, "title")


# --- CJK sentences and blocks --------------------------------------------


def test_cjk_full_stops_end_sentences_without_a_space():
    text = "新颖性很重要。我们测量了它！结果如何？最后一句。"
    assert split_sentences(text) == ["新颖性很重要。", "我们测量了它！", "结果如何？", "最后一句。"]
    # English keeps whole-word boundaries.
    assert split_sentences("Use .NET today. Then stop.") == ["Use .NET today.", "Then stop."]


def test_pysbd_never_sees_more_than_a_block_of_unpunctuated_text(monkeypatch):
    lengths: list[int] = []
    original = pysbd.Segmenter.segment

    def recording(self, text):
        lengths.append(len(text))
        return original(self, text)

    monkeypatch.setattr(pysbd.Segmenter, "segment", recording)
    for text in ("新颖性是科学中的重要概念。" * 2000, "novelty claim is made here " * 1500):
        lengths.clear()
        sentences = split_sentences(text)
        assert max(lengths) <= windows._BLOCK_CHARS
        assert "".join(sentences).replace(" ", "") == text.replace(" ", "")
    assert len(sentences) == 1  # no sentence end: the forced cuts are rejoined


def test_cjk_corpus_passages_are_single_sentences(tmp_config, fake_embedder):
    sentence = "科学研究中的新颖性往往受到评审者的质疑。我们测量了期刊组合的非典型性。"
    text = "\n".join((sentence * 40)[i : i + 30] for i in range(0, len(sentence) * 40, 30))
    ingest_records(tmp_config, fake_embedder, [{"source": "zh", "abstract": text}], progress=_quiet)
    _, literature, _ = _literature(tmp_config, "新颖性")
    assert len(literature) == 40
    assert all(
        p.text.replace(" ", "") == "科学研究中的新颖性往往受到评审者的质疑。" for p in literature
    )


def test_segmenter_identity_marks_cjk_segmentation():
    assert windows.SENTENCE_SEGMENTER == f"pysbd-{pysbd.__version__}+blocks-v3"


# --- date bounds ----------------------------------------------------------


def test_one_digit_hour_bound_is_zero_padded():
    assert parse_bound("2026-05-01 9:30", end_of_day=False) == "2026-05-01 09:30:00"
    assert parse_bound("2026-05-01T9:30", end_of_day=True) == "2026-05-01 09:30:59"


@pytest.mark.parametrize(
    "value", ["2026-05-01 10:00 PM", "2026-05-01 10:00am", "2026-05-01 10:00 EST"]
)
def test_am_pm_and_zone_names_are_refused(value):
    with pytest.raises(ConfigError, match="AM/PM and time-zone names"):
        parse_bound(value, end_of_day=True)


# --- Windows paths in a Zotero export ------------------------------------


def test_windows_attachment_paths_give_the_file_basename(tmp_path):
    assert normalize_key(r"C:\Users\me\Zotero\storage\AB12\Smith 2020.pdf") == "smith2020"
    export = tmp_path / "export.csv"
    with export.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Title", "Item Type", "File Attachments"])
        writer.writerow(["Smith", "journalArticle", r"C:\Zotero\storage\AB12\smith2020.pdf"])
    assert load_zotero_csv(export, progress=_quiet)[0].attachment_keys == ["smith2020"]


# --- docx list markers and hyperlinks ------------------------------------

_W = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
)
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _paragraph(text: str, num: int | None = None, extra: str = "") -> str:
    numbered = (
        f'<w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="{num}"/></w:numPr></w:pPr>'
        if num
        else ""
    )
    return f'<w:p>{numbered}<w:r><w:t xml:space="preserve">{text}</w:t></w:r>{extra}</w:p>'


def _write_docx(path: Path, body: str) -> None:
    numbering = (
        f'<?xml version="1.0" encoding="UTF-8"?><w:numbering {_W}>'
        '<w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="1"/>'
        '<w:numFmt w:val="decimal"/></w:lvl></w:abstractNum>'
        '<w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/>'
        "</w:lvl></w:abstractNum>"
        '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>'
        '<w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num></w:numbering>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?><Relationships '
        'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{_REL}/numbering" Target="numbering.xml"/>'
        f'<Relationship Id="rId9" Type="{_REL}/hyperlink" Target="https://example.org/x" '
        'TargetMode="External"/></Relationships>'
    )
    types = (
        '<?xml version="1.0" encoding="UTF-8"?><Types '
        'xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/numbering.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/></Types>'
    )
    root = (
        '<?xml version="1.0" encoding="UTF-8"?><Relationships '
        'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{_REL}/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    document = f'<?xml version="1.0" encoding="UTF-8"?><w:document {_W}><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", types)
        archive.writestr("_rels/.rels", root)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/numbering.xml", numbering)
        archive.writestr("word/_rels/document.xml.rels", rels)


def test_docx_list_markers_and_hyperlink_markup_are_removed_typed_html_kept(tmp_path):
    link = '<w:hyperlink r:id="rId9"><w:r><w:t>novelty portal</w:t></w:r></w:hyperlink>'
    body = (
        _paragraph("First numbered claim on novelty.", 1)
        + _paragraph("Second numbered claim on novelty.", 1)
        + _paragraph("A bullet about novelty.", 2)
        + _paragraph("See the ", extra=link + '<w:r><w:t xml:space="preserve"> page.</w:t></w:r>')
        + _paragraph("Typed &lt;a href=&quot;x&quot;&gt;link&lt;/a&gt; and &lt;b&gt; stay.")
    )
    _write_docx(tmp_path / "draft.docx", body)
    assert _read_docx(tmp_path / "draft.docx").splitlines() == [
        "First numbered claim on novelty.",
        "Second numbered claim on novelty.",
        "A bullet about novelty.",
        "See the novelty portal page.",
        'Typed <a href="x">link</a> and <b> stay.',
    ]


# --- symlinked folders, Japanese marks, CJK title messages ---------------


@pytest.mark.skipif(sys.platform == "win32", reason="directory symlinks need privileges")
def test_symlinked_subfolders_are_followed_once(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "linked.pdf").write_text("x")
    lib = tmp_path / "pdfs"
    lib.mkdir()
    (lib / "top.pdf").write_text("x")
    (lib / "link").symlink_to(outside, target_is_directory=True)
    (outside / "loop").symlink_to(lib, target_is_directory=True)  # a cycle
    kept, _ = select_pdfs(lib)
    assert sorted(str(p.relative_to(lib)) for p in kept) == ["link/linked.pdf", "top.pdf"]


def test_fold_keeps_japanese_voicing_marks_and_drops_latin_accents():
    assert _fold("ガラスの研究") == "ガラスの研究"
    assert _fold(unicodedata.normalize("NFD", "パン")) == "パン"
    assert _fold("Café NAÏVE") == "cafe naive"


def test_cjk_title_error_counts_usable_words_without_blaming_latin_words():
    with pytest.raises(ConfigError) as error:
        load_mywork_rules({"publication_titles": ["科学中的新颖性研究"]})
    message = str(error.value)
    assert "has 1" in message and "filename_patterns" in message
    assert "Latin" not in message


# --- Zotero summary of an unchanged rerun -------------------------------


def test_zotero_summary_counts_unchanged_sources(tmp_config, fake_embedder, library, tmp_path):
    (library / "smith2020.pdf").write_text(_body("a"))
    (library / "jones2021.pdf").write_text(_body("b"))
    export = tmp_path / "export.csv"
    with export.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Title", "Item Type", "File Attachments"])
        writer.writerow(["Smith paper", "journalArticle", "/z/smith2020.pdf"])
        writer.writerow(["Jones paper", "journalArticle", "/z/jones2021.pdf"])
    ingest_pdf_dir(tmp_config, fake_embedder, library, export, progress=_quiet)
    messages: list[str] = []
    stats = ingest_pdf_dir(tmp_config, fake_embedder, library, export, progress=messages.append)
    summary = next(m for m in messages if m.startswith("[zotero] matched"))
    assert "2 unchanged" in summary
    assert stats["zotero_unchanged"] == 2
