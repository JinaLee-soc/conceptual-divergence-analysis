"""Regression tests for corpus ingestion and passage extraction."""

from __future__ import annotations

import csv
import os
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

import cda.corpus.ingest as ingest_module
from cda.config import ConfigError, load_config
from cda.corpus.drafts import load_drafts
from cda.corpus.ingest import (
    chunk_words,
    extract_pdf_text,
    file_acquisition_date,
    get_collection,
    ingest_pdf_dir,
    ingest_records,
    sentence_context_records,
)
from cda.corpus.zotero import (
    ZoteroRecord,
    build_doi_index,
    build_filename_index,
    build_title_index,
    load_zotero_csv,
    match_by_content,
    match_by_filename,
    normalize_date_added,
)
from cda.extraction.mywork import classify_source, load_mywork_rules, mywork_classifier
from cda.extraction.retrieval import build_passages, keyword_hits
from cda.extraction.windows import SENTENCE_SEGMENTER, infer_year, split_sentences

PINNED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # pragma: allowlist secret


def quiet(_message: str) -> None:
    pass


@pytest.fixture
def config(tmp_path):
    return load_config(
        overrides={
            "paths": {"corpus_dir": str(tmp_path / "corpus"), "drafts_dir": str(tmp_path / "d")}
        }
    )


@pytest.fixture
def text_pdfs(monkeypatch):
    """Treat each '.pdf' as a UTF-8 text file, so a test can edit its content."""
    monkeypatch.setattr(ingest_module, "extract_pdf_text", lambda p: Path(p).read_text())


def write_zotero_csv(path: Path, rows: list[dict]) -> Path:
    fields = ["Title", "Publication Year", "DOI", "Date Added", "Manual Tags", "File Attachments"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


# --- 1. stale re-ingest and missing sources -----------------------------------


def test_changed_pdf_is_reingested(tmp_path, config, fake_embedder, text_pdfs):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "a.pdf").write_text("Old version says novelty is rare. " * 10)
    ingest_pdf_dir(config, fake_embedder, lib, progress=quiet)

    (lib / "a.pdf").write_text("Corrected version says consensus only. " * 10)
    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=quiet)

    assert stats["pdfs_stale"] == 1 and stats["pdfs_refreshed"] == 1
    documents = get_collection(config).get(include=["documents"])["documents"]
    assert documents and all("Corrected" in d and "Old" not in d for d in documents)


def test_changed_zotero_csv_refreshes_metadata(tmp_path, config, fake_embedder, text_pdfs):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "lee2021.pdf").write_text("Padding sentence with novelty words. " * 6)
    row = {"Title": "Mine", "Publication Year": "2021", "File Attachments": "/s/lee2021.pdf"}
    export = tmp_path / "lib.csv"
    write_zotero_csv(export, [row])
    ingest_pdf_dir(config, fake_embedder, lib, export, progress=quiet)

    write_zotero_csv(export, [dict(row, **{"Manual Tags": "mywork"})])
    stats = ingest_pdf_dir(config, fake_embedder, lib, export, progress=quiet)

    # Only metadata changed: updated in place, not re-ingested.
    assert stats["metadata_updated"] == 1 and stats["pdfs_stale"] == 0
    metadatas = get_collection(config).get(include=["metadatas"])["metadatas"]
    assert {m.get("zotero_tags") for m in metadatas} == {"mywork"}


def test_chunks_from_another_segmenter_are_reingested(tmp_path, config, fake_embedder, text_pdfs):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "a.pdf").write_text("Novelty matters. " * 10)
    ingest_pdf_dir(config, fake_embedder, lib, progress=quiet)
    collection = get_collection(config)
    stored = collection.get(include=["metadatas"])
    collection.update(
        ids=stored["ids"],
        metadatas=[dict(m, sentence_segmenter="regex-v1") for m in stored["metadatas"]],
    )

    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=quiet)

    assert stats["pdfs_stale"] == 1 and stats["pdfs_ingested"] == 1
    metadatas = get_collection(config).get(include=["metadatas"])["metadatas"]
    assert {m["sentence_segmenter"] for m in metadatas} == {SENTENCE_SEGMENTER}


def test_sources_missing_from_disk_are_reported_and_kept(
    tmp_path, config, fake_embedder, text_pdfs
):
    lib = tmp_path / "lib"
    lib.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (lib / name).write_text(f"Paper {name} mentions novelty. " * 6)
    ingest_pdf_dir(config, fake_embedder, lib, progress=quiet)
    (lib / "b.pdf").unlink()

    messages: list[str] = []
    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=messages.append)

    assert stats["pdfs_missing"] == 1 and stats["pdfs_pruned"] == 0
    assert any("b.pdf" in m and "--prune-missing" in m for m in messages)
    stored = {
        m["pdf_relpath"] for m in get_collection(config).get(include=["metadatas"])["metadatas"]
    }
    assert stored == {"a.pdf", "b.pdf"}


def test_prune_missing_removes_sources_missing_from_disk(
    tmp_path, config, fake_embedder, text_pdfs
):
    lib = tmp_path / "lib"
    lib.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (lib / name).write_text(f"Paper {name} mentions novelty. " * 6)
    ingest_pdf_dir(config, fake_embedder, lib, progress=quiet)
    (lib / "b.pdf").unlink()

    stats = ingest_pdf_dir(config, fake_embedder, lib, prune_missing=True, progress=quiet)

    assert stats["pdfs_missing"] == 1 and stats["pdfs_pruned"] == 1
    stored = {
        m["pdf_relpath"] for m in get_collection(config).get(include=["metadatas"])["metadatas"]
    }
    assert stored == {"a.pdf"}


def test_changed_record_text_is_reingested(config, fake_embedder):
    ingest_records(config, fake_embedder, [{"source": "a", "abstract": "Old novelty."}])
    stats = ingest_records(config, fake_embedder, [{"source": "a", "abstract": "New novelty."}])
    assert stats["records_ingested"] == 1
    assert get_collection(config).get(include=["documents"])["documents"] == ["New novelty."]


# --- 2. Zotero keys shared by several records ---------------------------------


def _two_records(**shared) -> list[ZoteroRecord]:
    other = ZoteroRecord(title="Someone Else's Paper", year="2018", doi="10.1/aaa", tags=[])
    mine = ZoteroRecord(title="My Own Paper", year="2021", doi="10.1/bbb", tags=["mywork"])
    for record in (other, mine):
        for key, value in shared.items():
            setattr(record, key, value)
    return [other, mine]


def test_shared_attachment_name_attaches_nothing():
    index = build_filename_index(_two_records(attachment_keys=["fulltext"]))
    assert match_by_filename("fulltext.pdf", index) == (None, "ambiguous_filename")


def test_shared_doi_attaches_nothing():
    index = build_doi_index(_two_records(doi="10.1000/ccc"))
    assert match_by_content("doi:10.1000/ccc", index, {}) == (None, "ambiguous_doi")


def test_shared_exact_title_attaches_nothing():
    title = "A Sufficiently Long Title Shared By Two Distinct Items"
    index = build_title_index(_two_records(title=title))
    assert match_by_content(title, {}, index) == (None, "ambiguous_title")


def test_ingest_records_the_ambiguity_reason(tmp_path, config, fake_embedder, text_pdfs):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "fulltext.pdf").write_text("A paper about novelty without identifiers. " * 6)
    export = write_zotero_csv(
        tmp_path / "lib.csv",
        [
            {"Title": "Someone Else", "File Attachments": "/z/A/fulltext.pdf"},
            {"Title": "Mine", "Manual Tags": "mywork", "File Attachments": "/z/B/fulltext.pdf"},
        ],
    )
    messages: list[str] = []
    stats = ingest_pdf_dir(config, fake_embedder, lib, export, progress=messages.append)

    assert stats["zotero_ambiguous"] == 1 and stats["zotero_by_filename"] == 0
    (metadata,) = get_collection(config).get(include=["metadatas"])["metadatas"]
    assert metadata["zotero_unmatched_reason"] == "ambiguous_filename"
    assert "zotero_tags" not in metadata
    assert any("more than one Zotero item" in m for m in messages)


# --- 3. case-insensitive literal retrieval ------------------------------------


def test_keyword_hits_find_every_letter_case(config, fake_embedder):
    pad = " ".join(["Pad words here."] * 10)
    records = [
        {"source": "a", "abstract": pad + " The SARS-CoV-2 pandemic changed science."},
        {"source": "b", "abstract": pad + " The Sars-cov-2 case."},
        {"source": "c", "abstract": pad + " The iPhone changed science."},
    ]
    ingest_records(config, fake_embedder, records, progress=quiet)
    collection = get_collection(config)
    assert sorted(keyword_hits(collection, "sars-cov-2", progress=quiet)) == ["a::c0", "b::c0"]
    assert sorted(keyword_hits(collection, "IPHONE", progress=quiet)) == ["c::c0"]


# --- 4. drafts are read whole and fail loudly ---------------------------------


def _docx_with_table_notes_and_text_box(path: Path) -> Path:
    ns = (
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:v="urn:schemas-microsoft-com:vml"'
    )

    def para(text):
        return f'<w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'

    body = (
        para("Body text without the term.")
        + '<w:p><w:r><w:t xml:space="preserve">A claim.</w:t></w:r>'
        + '<w:r><w:footnoteReference w:id="1"/></w:r><w:r><w:endnoteReference w:id="1"/></w:r></w:p>'
        + "<w:tbl><w:tr><w:tc>"
        + para("Novelty appears in a table cell.")
        + "</w:tc></w:tr></w:tbl>"
        + "<w:p><w:r><w:pict><v:shape><v:textbox><w:txbxContent>"
        + para("Novelty appears in a text box.")
        + "</w:txbxContent></v:textbox></v:shape></w:pict></w:r></w:p>"
    )
    head = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ctype = "application/vnd.openxmlformats-officedocument.wordprocessingml"
    parts = {
        "[Content_Types].xml": f'{head}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'<Override PartName="/word/document.xml" ContentType="{ctype}.document.main+xml"/>'
        f'<Override PartName="/word/footnotes.xml" ContentType="{ctype}.footnotes+xml"/>'
        f'<Override PartName="/word/endnotes.xml" ContentType="{ctype}.endnotes+xml"/></Types>',
        "_rels/.rels": f'{head}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{rel}/officeDocument" Target="word/document.xml"/></Relationships>',
        "word/_rels/document.xml.rels": f'{head}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rIdF" Type="{rel}/footnotes" Target="footnotes.xml"/>'
        f'<Relationship Id="rIdE" Type="{rel}/endnotes" Target="endnotes.xml"/></Relationships>',
        "word/document.xml": f"{head}<w:document {ns}><w:body>{body}</w:body></w:document>",
        "word/footnotes.xml": f'{head}<w:footnotes {ns}><w:footnote w:id="1">'
        + para("Novelty is defined in a footnote.")
        + "</w:footnote></w:footnotes>",
        "word/endnotes.xml": f'{head}<w:endnotes {ns}><w:endnote w:id="1">'
        + para("Novelty is defined in an endnote.")
        + "</w:endnote></w:endnotes>",
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, xml in parts.items():
            archive.writestr(name, xml)
    return path


def test_docx_drafts_include_tables_notes_and_text_boxes(tmp_path):
    _docx_with_table_notes_and_text_box(tmp_path / "paper.docx")
    text = load_drafts(tmp_path, progress=quiet)["paper.docx"]
    for passage in (
        "Novelty appears in a table cell.",
        "Novelty appears in a text box.",
        "Novelty is defined in a footnote.",
        "Novelty is defined in an endnote.",
    ):
        assert passage in text
    assert "----" not in text and "footnote1)" not in text


def test_long_text_draft_is_not_truncated(tmp_path):
    (tmp_path / "long.txt").write_text(
        "Filler words go here. " * 17_000 + "Our novelty claim ends it."
    )
    assert load_drafts(tmp_path, progress=quiet)["long.txt"].endswith("Our novelty claim ends it.")


def test_long_pdf_draft_is_not_truncated(tmp_path):
    import fitz

    document = fitz.open()
    for number in range(151):
        document.new_page().insert_text((72, 72), f"Page {number} filler.")
    document[-1].insert_text((72, 120), "Our novelty claim is on the last page.")
    document.save(tmp_path / "book.pdf")
    document.close()
    assert "novelty claim is on the last page" in load_drafts(tmp_path, progress=quiet)["book.pdf"]


def test_identical_drafts_are_refused(tmp_path):
    (tmp_path / "paper.md").write_text("We define novelty as recombination.")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "paper.md").write_text("We define  novelty as recombination.\n")
    with pytest.raises(ConfigError, match=r"paper\.md.*sub/paper\.md"):
        load_drafts(tmp_path, progress=quiet)


def test_unparseable_draft_raises(tmp_path):
    (tmp_path / "broken.docx").write_bytes(b"not a zip archive")
    with pytest.raises(ConfigError, match="broken.docx"):
        load_drafts(tmp_path, progress=quiet)


# --- 5. pysbd sentence segmentation -------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ('They called it "novelty." Then novelty again.', 2),
        ("Smith et al. (2019) argue that novelty is rewarded.", 1),
        ("See Fig. 2 for novelty rates.", 1),
        ("The U.S. novelty premium is large.", 1),
    ],
)
def test_sentence_boundaries(text, expected):
    sentences = split_sentences(text)
    assert len(sentences) == expected
    assert " ".join(sentences) == text


def test_sentence_boundaries_fall_between_words():
    text = "Applications built on .NET are described here. The .NET runtime is analyzed."
    assert split_sentences(text) == [
        "Applications built on .NET are described here.",
        "The .NET runtime is analyzed.",
    ]


def test_ingest_records_the_segmenter(config, fake_embedder):
    ingest_records(config, fake_embedder, [{"source": "a", "abstract": "Novelty."}], progress=quiet)
    (metadata,) = get_collection(config).get(include=["metadatas"])["metadatas"]
    assert metadata["sentence_segmenter"] == SENTENCE_SEGMENTER


def test_analysis_rejects_chunks_from_another_segmenter():
    metadata = {
        "pdf_relpath": "a.pdf",
        "chunk_index": 0,
        "sentence_start": 0,
        "sentence_context": "Novelty.",
        "sentence_segmenter": "regex-v1",
    }
    hits = {"a::c0": {"document": "Novelty.", "metadata": metadata}}
    with pytest.raises(ConfigError, match="cda ingest"):
        build_passages("novelty", hits, mywork_classifier({}), load_config())


# --- 6. MyWork settings --------------------------------------------------------


@pytest.mark.parametrize(
    "setting",
    [
        {"title_match_min_overlap": True},
        {"title_match_min_overlap": 2.9},
        {"title_match_min_ratio": True},
        {"title_match_min_ratio": float("nan")},
        {"title_match_min_ratio": 1.5},
    ],
)
def test_mywork_numbers_are_validated(setting):
    with pytest.raises(ConfigError, match="mywork.title_match_min"):
        load_mywork_rules(setting)


def test_filename_pattern_matches_name_with_extension():
    classify = mywork_classifier({"filename_patterns": ["lee_2020.pdf"]})
    assert classify("lee_2020.pdf", {}) == (True, "pattern:lee_2020.pdf")
    assert not classify("lee_2021.pdf", {})[0]


def test_stop_words_do_not_count_toward_a_title_match():
    rules = load_mywork_rules({"publication_titles": ["Gender and the Science of the Laboratory"]})
    assert rules["title_tokens"] == [{"gender", "science", "laboratory"}]
    assert not classify_source(rules, "the_and_of_gender_paper.pdf", {})[0]


def test_title_that_can_never_match_is_refused():
    with pytest.raises(ConfigError, match="Gender and Science"):
        mywork_classifier({"publication_titles": ["Gender and Science"]})


# --- 7. UTC acquisition times -------------------------------------------------


@pytest.fixture
def seoul_time(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Seoul")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_zoned_date_added_is_stored_in_utc(seoul_time):
    assert normalize_date_added("2024-01-01T03:00:00Z") == "2024-01-01 03:00:00"
    assert normalize_date_added("2024-01-01 03:00:00") == "2024-01-01 03:00:00"


def test_filesystem_clock_is_utc(tmp_path, seoul_time):
    path = tmp_path / "a.pdf"
    path.write_text("x")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    stat = path.stat()
    stamp = getattr(stat, "st_birthtime", None) or stat.st_mtime
    expected = datetime.fromtimestamp(stamp, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
    assert file_acquisition_date(path) == expected


def test_impossible_date_added_in_export_is_rejected(tmp_path):
    export = write_zotero_csv(tmp_path / "lib.csv", [{"Title": "T", "Date Added": "2026-02-30"}])
    with pytest.raises(ValueError, match="record 1"):
        load_zotero_csv(export)


def test_window_description_names_utc():
    from cda.corpus.timeframe import describe_window

    window = {"active": True, "added_from": None, "added_to": "2026-05-01 23:59:59"}
    assert "(UTC)" in describe_window(window)


# --- 8. overlapping chunks with different context windows ----------------------


@pytest.mark.parametrize("position", [40, 49])
@pytest.mark.parametrize("retrieval", [{"window_mode": "context"}, {"min_window_chars": 200}])
def test_overlap_keeps_the_wider_context_window(tmp_path, fake_embedder, position, retrieval):
    config = load_config(
        overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}, "retrieval": retrieval}
    )
    sentences = [f"Filler sentence number {i} has exactly ten words here ok." for i in range(60)]
    sentences[position] = "Here the novelty term appears in this ten word sentence."
    ingest_records(config, fake_embedder, [{"source": "doc", "abstract": " ".join(sentences)}])
    hits = keyword_hits(get_collection(config), "novelty", progress=quiet)
    assert len(hits) == 2  # the sentence lies in the overlap of two chunks

    _, literature, stats = build_passages("novelty", hits, mywork_classifier({}), config)

    expected = " ".join(sentences[position - 1 : position + 2])
    assert [p.text for p in literature] == [expected]
    assert stats["overlap_occurrences_removed"] == 1


# --- 9. small overlaps never drop the tail --------------------------------------


@pytest.mark.parametrize("overlap", [0, 5, 19])
def test_every_word_is_chunked(overlap):
    text = " ".join(f"w{i}" for i in range(1000)) + " The novelty claim ends here."
    chunks = chunk_words(text, 500, overlap)
    assert {w for chunk in chunks for w in chunk.split()} == set(text.split())
    contexts = sentence_context_records(text, 500, overlap, len(chunks))
    assert any("novelty" in context for _, context, _ in contexts)


# --- 10. years need digit boundaries ---------------------------------------------


def test_infer_year_requires_digit_boundaries():
    assert infer_year("0003122420912345.pdf", {}) == "unknown"
    assert infer_year("smith_2019_paper.pdf", {}) == "2019"
    assert infer_year("x.pdf", {"year": "c2019"}) == "2019"


# --- 11. pinned embedding snapshot ----------------------------------------------


def test_embedding_revision_is_pinned_by_default():
    assert load_config().get("embedding.revision") == PINNED_REVISION


def test_provenance_records_the_resolved_snapshot(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub

    from cda.corpus.embeddings import SentenceTransformerEmbedder

    snapshot = tmp_path / "snapshots" / PINNED_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    requested = []

    def cached(repo, filename, revision=None):
        requested.append(revision)
        return str(snapshot / filename)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    embedder = SentenceTransformerEmbedder("org/model", revision=PINNED_REVISION)
    embedder._model = SimpleNamespace(device="cpu")

    result = embedder.provenance()

    assert requested == [PINNED_REVISION]
    assert result["resolved_revision"] == PINNED_REVISION
    assert result["files"] == {"config.json": result["files"]["config.json"]}


# --- 12. loud failures instead of silent skips -------------------------------------


def test_unreadable_pdf_raises_from_the_extractor(tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(RuntimeError):
        extract_pdf_text(tmp_path / "bad.pdf")


def test_unreadable_pdf_is_counted_and_reported(tmp_path, config, fake_embedder):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "bad.pdf").write_bytes(b"%PDF-1.4 garbage")
    messages: list[str] = []
    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=messages.append)
    assert stats["pdfs_unreadable"] == 1
    assert any("bad.pdf" in m and "unreadable" in m for m in messages)


def test_empty_chunk_document_raises():
    hits = {"a::c0": {"document": "", "metadata": {"pdf_relpath": "a.pdf"}}}
    with pytest.raises(ConfigError, match="no text"):
        build_passages("novelty", hits, mywork_classifier({}), load_config())


def test_record_without_text_raises(config, fake_embedder):
    records = [{"source": "a", "abstract": "Novelty."}, {"source": "b", "abstract": "  "}]
    with pytest.raises(ValueError, match=r"positions \[1\]"):
        ingest_records(config, fake_embedder, records, progress=quiet)
