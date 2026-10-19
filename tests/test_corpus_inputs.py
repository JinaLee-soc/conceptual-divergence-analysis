"""Corpus metadata, document formats, and sentence extraction regressions.

Each test pins one defect: quadratic sentence segmentation of long texts,
full re-ingest after any Zotero CSV change, excluded sources kept silently,
lost Date Added time zones, non-Latin and accented MyWork titles, arXiv
identifiers read as years, docx2python symbol markup in drafts, and a draft
counted twice through its PDF export.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pysbd
import pytest

import cda.corpus.ingest as ing
from cda.config import ConfigError, load_config
from cda.corpus.drafts import _read_docx, load_drafts, normalize_drafts
from cda.corpus.ingest import get_collection, ingest_pdf_dir
from cda.corpus.timeframe import parse_bound
from cda.corpus.zotero import load_zotero_csv, normalize_date_added
from cda.extraction import windows
from cda.extraction.mywork import mywork_classifier, title_tokens
from cda.extraction.windows import infer_year, normalize_text, split_sentences

CSV_HEADER = "Title,Item Type,Publication Year,Date Added,Manual Tags,File Attachments\n"
SENTENCES = (
    "Smith et al. (2020) argue that novelty is rewarded in the long run. "
    "In Fig. 2, we plot novelty scores by field and year. The U.S. NSF funds it. "
)


def _quiet(_message: str) -> None:
    pass


def _config(tmp_path):
    return load_config(
        overrides={
            "paths": {
                "corpus_dir": str(tmp_path / "corpus"),
                "drafts_dir": str(tmp_path / "drafts"),
                "output_dir": str(tmp_path / "out"),
            }
        }
    )


@pytest.fixture
def text_pdfs(monkeypatch):
    """Treat each '.pdf' as a UTF-8 text file and count how often one is read."""
    reads: list[str] = []

    def read(path):
        reads.append(Path(path).name)
        return Path(path).read_text(encoding="utf-8")

    monkeypatch.setattr(ing, "extract_pdf_text", read)
    return reads


def _stored(config) -> dict:
    """{relpath: (ids, metadatas, embeddings)} of the stored chunks."""
    result = get_collection(config).get(include=["metadatas", "embeddings"])
    stored: dict = {}
    for chunk_id, metadata, embedding in zip(
        result["ids"], result["metadatas"], result["embeddings"], strict=True
    ):
        ids, metadatas, embeddings = stored.setdefault(metadata["pdf_relpath"], ([], [], []))
        ids.append(chunk_id)
        metadatas.append(metadata)
        embeddings.append([float(x) for x in embedding])
    return stored


# --- block segmentation --------------------------------------------------


def test_long_text_is_segmented_in_bounded_blocks(monkeypatch):
    """pysbd is quadratic in input length; no pysbd call may see the whole long text."""
    lengths: list[int] = []
    original = pysbd.Segmenter.segment

    def recording(self, text):
        lengths.append(len(text))
        return original(self, text)

    monkeypatch.setattr(pysbd.Segmenter, "segment", recording)
    text = SENTENCES * 300  # about 40,000 characters, no paragraph breaks
    sentences = split_sentences(text)
    assert max(lengths) <= 10_000
    assert " ".join(sentences) == normalize_text(text)
    assert len(sentences) == 900
    assert sentences[1] == "In Fig. 2, we plot novelty scores by field and year."


def test_block_cuts_fall_after_sentence_ends_not_abbreviations():
    text = normalize_text(SENTENCES * 40)
    blocks = windows._blocks(text, 1_000)
    assert len(blocks) > 3
    assert " ".join(blocks) == text
    assert all(block.endswith(("run.", "year.", "funds it.")) for block in blocks)


def test_paragraph_whitespace_keeps_punctuation_with_the_previous_sentence():
    text = "Novelty matters here.\n\n...\n\nConsensus follows.\n \nIt holds"
    sentences = split_sentences(text)
    assert sentences == ["Novelty matters here. ...", "Consensus follows.", "It holds"]
    assert " ".join(sentences) == normalize_text(text)


def test_segmenter_identity_marks_block_segmentation():
    """Stored corpora segmented whole-text must be refreshed by the next ingest."""
    assert windows.SENTENCE_SEGMENTER == f"pysbd-{pysbd.__version__}+blocks-v3"


# --- a new Zotero export updates metadata in place -----------------------


def _ingest(config, fake_embedder, lib, csv_path, messages=None):
    progress = messages.append if messages is not None else _quiet
    return ingest_pdf_dir(config, fake_embedder, lib, csv_path, progress=progress)


def test_unrelated_zotero_change_reuses_every_source(tmp_path, fake_embedder, text_pdfs):
    config = _config(tmp_path)
    lib = tmp_path / "lib"
    lib.mkdir()
    for i in range(3):
        (lib / f"p{i}.pdf").write_text(f"Novelty paper {i}.")
    export = tmp_path / "z.csv"
    first_row = "Paper zero,journalArticle,2020,2024-01-02 03:04:05,,/x/p0.pdf\n"
    export.write_text(CSV_HEADER + first_row)
    _ingest(config, fake_embedder, lib, export)
    before = _stored(config)

    export.write_text(CSV_HEADER + first_row + "Unrelated new item,book,2026,2026-10-01,,\n")
    text_pdfs.clear()
    stats = _ingest(config, fake_embedder, lib, export)

    after = _stored(config)
    assert stats["pdfs_refreshed"] == 0 and stats["pdfs_ingested"] == 0
    assert stats["pdfs_skipped"] == 3 and stats["metadata_updated"] == 0
    assert "p0.pdf" not in text_pdfs  # matched by filename: its text is not needed
    sha = ing.file_digest(export)
    for relpath, (ids, metadatas, embeddings) in after.items():
        assert ids == before[relpath][0] and embeddings == before[relpath][2]
        assert {m["zotero_csv_sha256"] for m in metadatas} == {sha}


def test_changed_zotero_fields_are_updated_without_reembedding(tmp_path, fake_embedder, text_pdfs):
    config = _config(tmp_path)
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "lee2021.pdf").write_text("Padding sentence with novelty words. " * 200)
    export = tmp_path / "z.csv"
    export.write_text(CSV_HEADER + "Mine,journalArticle,2021,2024-01-02,,/s/lee2021.pdf\n")
    _ingest(config, fake_embedder, lib, export)
    ids, _, embeddings = _stored(config)["lee2021.pdf"]
    assert len(ids) > 1

    export.write_text(CSV_HEADER + "Mine,journalArticle,2021,2024-01-02,mywork,/s/lee2021.pdf\n")
    messages: list[str] = []
    stats = _ingest(config, fake_embedder, lib, export, messages)
    new_ids, metadatas, new_embeddings = _stored(config)["lee2021.pdf"]
    assert stats["metadata_updated"] == 1 and stats["pdfs_refreshed"] == 0
    assert new_ids == ids and new_embeddings == embeddings
    assert {m.get("zotero_tags") for m in metadatas} == {"mywork"}
    assert any("updated zotero_tags" in m for m in messages)

    # The item leaves the library: its fields go, and the reason is recorded.
    export.write_text(CSV_HEADER + "Other,journalArticle,2019,2024-01-02,,/s/other.pdf\n")
    stats = _ingest(config, fake_embedder, lib, export)
    _, metadatas, _ = _stored(config)["lee2021.pdf"]
    assert stats["metadata_updated"] == 1
    assert all("title" not in m and "zotero_tags" not in m for m in metadatas)
    assert {m["zotero_unmatched_reason"] for m in metadatas} == {"no_match"}


# --- stored sources under a now-excluded path -----------------------------


def _library_with_excluded_folder(tmp_path, fake_embedder):
    config = _config(tmp_path)
    lib = tmp_path / "lib"
    (lib / "skip").mkdir(parents=True)
    (lib / "kept.pdf").write_text("Novelty matters here.")
    (lib / "skip" / "old.pdf").write_text("Novelty in a folder excluded later.")
    ingest_pdf_dir(config, fake_embedder, lib, progress=_quiet)
    return config, lib


def test_excluded_stored_sources_are_listed_and_kept(tmp_path, fake_embedder, text_pdfs):
    config, lib = _library_with_excluded_folder(tmp_path, fake_embedder)
    messages: list[str] = []
    stats = ingest_pdf_dir(config, fake_embedder, lib, exclude=["skip"], progress=messages.append)
    assert stats["pdfs_stored_excluded"] == 1 and stats["pdfs_pruned"] == 0
    assert set(_stored(config)) == {"kept.pdf", "skip/old.pdf"}
    assert any("now excluded" in m and "skip/old.pdf" in m for m in messages)


@pytest.mark.parametrize("selection", [{"exclude": ["skip/"]}, {"recursive": False}])
def test_excluded_stored_sources_are_pruned(tmp_path, fake_embedder, text_pdfs, selection):
    config, lib = _library_with_excluded_folder(tmp_path, fake_embedder)
    stats = ingest_pdf_dir(
        config, fake_embedder, lib, prune_missing=True, progress=_quiet, **selection
    )
    assert stats["pdfs_pruned"] == 1 and stats["pdfs_missing"] == 0
    assert set(_stored(config)) == {"kept.pdf"}


# --- Date Added parsing ----------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "2024-01-02T03:04:05.123+09:00",
        "2024-01-02 03:04:05 +09:00",
        "2024-01-02 03:04:05 +0900",
        "2024-01-01T18:04:05,9Z",
    ],
)
def test_zoned_date_added_keeps_its_zone(value):
    assert normalize_date_added(value) == "2024-01-01 18:04:05"


def test_non_iso_date_added_is_undated_counted_and_warned(tmp_path):
    export = tmp_path / "z.csv"
    export.write_text(
        CSV_HEADER
        + "A,book,2020,1/2/2024 3:04,,\n"
        + "B,book,2020,2/3/2024 4:05,,\n"
        + "C,book,2020,2024-01-02 03:04:05,,\n"
    )
    messages: list[str] = []
    records = load_zotero_csv(export, progress=messages.append)
    assert [r.date_added for r in records] == ["", "", "2024-01-02 03:04:05"]
    assert len(messages) == 1
    assert "2 record(s)" in messages[0] and "'1/2/2024 3:04'" in messages[0]


def test_upper_bound_to_the_minute_includes_the_whole_minute():
    assert parse_bound("2026-05-01 10:00", end_of_day=True) == "2026-05-01 10:00:59"
    assert parse_bound("2026-05-01 10:00", end_of_day=False) == "2026-05-01 10:00:00"
    assert parse_bound("2026-05-01 10:00:30", end_of_day=True) == "2026-05-01 10:00:30"


# --- MyWork title tokens ----------------------------------------------------


def test_accented_titles_keep_whole_words_in_both_unicode_forms():
    title = "Méritocratie académique sélective en sciences"
    expected = {"meritocratie", "academique", "selective", "sciences"}
    assert title_tokens(title) == expected
    assert title_tokens(unicodedata.normalize("NFD", title)) == expected


@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_korean_title_tokenizes_and_matches_its_file(form):
    title = "과학계의 성별 인정 격차"
    assert title_tokens(title) == {"과학계의", "성별", "인정", "격차"}
    classify = mywork_classifier({"publication_titles": [title]})
    filename = unicodedata.normalize(form, "과학계의_성별_인정_격차.pdf")
    assert classify(filename, {}) == (True, "profile:title_match")
    assert classify("unrelated_paper.pdf", {})[0] is False


# --- arXiv identifiers are not years -----------------------------------------


@pytest.mark.parametrize("filename", ["1905.01234v2.pdf", "2005.14165.pdf"])
def test_arxiv_identifier_is_not_a_year(filename):
    assert infer_year(filename, {}) == "unknown"


def test_year_beside_an_arxiv_identifier_is_still_found():
    assert infer_year("smith_2019_2005.14165.pdf", {}) == "2019"


# --- docx2python symbol markup ------------------------------------------------


def _docx_with_symbol_run(path):
    """A minimal .docx (plain zipfile) whose paragraph holds a Symbol-font character."""
    import zipfile

    head = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    ns = (
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    )
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ctype = "application/vnd.openxmlformats-officedocument.wordprocessingml"
    body = (
        '<w:p><w:r><w:t xml:space="preserve">R&amp;D &lt;b&gt; &amp;amp; x &lt; y; '
        "the effect is </w:t></w:r>"
        '<w:r><w:sym w:font="Symbol" w:char="F061"/></w:r>'
        '<w:r><w:t xml:space="preserve"> = 0.3</w:t></w:r></w:p>'
    )
    parts = {
        "[Content_Types].xml": f'{head}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'<Override PartName="/word/document.xml" ContentType="{ctype}.document.main+xml"/></Types>',
        "_rels/.rels": f'{head}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{rel}/officeDocument" Target="word/document.xml"/></Relationships>',
        "word/document.xml": f"{head}<w:document {ns}><w:body>{body}</w:body></w:document>",
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, xml in parts.items():
            archive.writestr(name, xml)
    return path


def test_symbol_font_markup_is_decoded_and_literal_prose_kept(tmp_path):
    path = _docx_with_symbol_run(tmp_path / "d.docx")
    assert _read_docx(path) == "R&D <b> &amp; x < y; the effect is a = 0.3"


# --- near-duplicate drafts ------------------------------------------------------


def test_draft_and_its_pdf_export_are_refused_naming_both():
    body = " ".join(
        f"Sentence {i} discusses novelty claim number {chr(97 + i)}." for i in range(20)
    )
    pdf_text = body.replace("claim number k.", "claim number k. 2") + " 3"
    with pytest.raises(ConfigError, match=r"'ms\.docx' and 'ms\.pdf' share 100%"):
        normalize_drafts({"ms.docx": body, "ms.pdf": pdf_text})


def test_near_duplicate_files_are_refused_at_load(tmp_path):
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    body = " ".join(f"Point {chr(97 + i)} concerns novelty in science." for i in range(20))
    (drafts / "a.md").write_text(body)
    (drafts / "b.txt").write_text(body + " 14")
    with pytest.raises(ConfigError, match="a.md' and 'b.txt"):
        load_drafts(drafts, progress=_quiet)


def test_distinct_drafts_sharing_some_sentences_are_kept():
    shared = "Novelty is rising. Consensus is falling."
    drafts = {
        "a.md": shared + " First paper argues one thing.",
        "b.md": shared + " Second differs.",
    }
    assert set(normalize_drafts(drafts)) == {"a.md", "b.md"}


def test_short_drafts_sharing_their_sentences_are_kept():
    drafts = {"a.md": "Novelty. Novelty.", "b.md": "Novelty. 2"}
    assert set(normalize_drafts(drafts)) == {"a.md", "b.md"}


def test_identical_short_drafts_message_shows_the_text():
    with pytest.raises(ConfigError, match=r"identical text \(both contain only 'TODO'\)"):
        normalize_drafts({"a.md": "TODO", "b.md": "TODO"})
