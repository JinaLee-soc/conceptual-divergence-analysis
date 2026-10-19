"""Regression tests for ingest and extraction data integrity.

Each test pins one failure that lost, duplicated, or misattributed corpus
data: sentence boundaries recomputed at analysis, dropped trailing
punctuation, deletion before a refresh succeeded, unreadable files aborting
an ingest, stripped Zotero fields, one file stored twice, pruning of
root-unknown sources, cited-title Zotero matches, and embedding provenance
outside the default cache.
"""

from __future__ import annotations

import os
import random
import sys
from types import SimpleNamespace

import pytest

import cda.corpus.ingest as ing
from cda.config import ConfigError, load_config
from cda.corpus.ingest import get_collection, ingest_pdf_dir, ingest_records
from cda.extraction.draft_passages import extract_draft_passages
from cda.extraction.mywork import mywork_classifier
from cda.extraction.retrieval import build_passages, keyword_hits
from cda.extraction.windows import split_sentences, term_regex


def _config(tmp_path, **corpus):
    overrides = {
        "paths": {
            "corpus_dir": str(tmp_path / "corpus"),
            "drafts_dir": str(tmp_path / "drafts"),
            "output_dir": str(tmp_path / "out"),
        }
    }
    if corpus:
        overrides["corpus"] = corpus
    return load_config(overrides=overrides)


def _quiet(_message: str) -> None:
    return None


def _stored(config) -> dict[str, dict]:
    """{pdf_relpath: metadata of one chunk} for the store."""
    metadatas = get_collection(config).get(include=["metadatas"])["metadatas"]
    return {m["pdf_relpath"]: m for m in metadatas}


@pytest.fixture
def text_pdfs(monkeypatch):
    """PDF extraction reads the file as plain text, so tests write text files."""
    monkeypatch.setattr(ing, "extract_pdf_text", lambda path: path.read_text())


# --- analysis uses the ingest-time sentence boundaries ----------------------

# Fragments typical of extracted PDF text: running heads, figure captions,
# reference entries, and abbreviations that segment differently in isolation.
_PDF_FRAGMENTS = [
    "Smith et al. (2020) argue that novelty is rewarded in the long run.",
    "1 Introduction",
    "2. Data and methods",
    "Table 1. Descriptive statistics",
    "Fig. 3. Novelty over time",
    "Reviewers penalize novelty when it is unfamiliar.",
    "We measure atypical combinations of cited journals.",
    "12",
    "Journal of Informetrics 14 (2020) 101–115",
    "1. Uzzi B, Mukherjee S. Atypical combinations and scientific impact. Science. 2013;342:468–72.",
    "Novelty claims appear in abstracts more often over time.",
    "The U.S. National Science Foundation funds high-risk research.",
    "In Fig. 2, we plot novelty scores by field and year.",
    "et al.",
    "p. 4",
    "Vol. 80, No. 5",
]


def test_analysis_sentences_equal_ingest_sentences_for_pdf_like_text(tmp_path, fake_embedder):
    """Re-segmenting a chunk's context alone disagreed with whole-document
    segmentation: wrong passages, duplicates, or a "Conflicting overlapping
    sentence records" crash."""
    rng = random.Random(1)
    regex = term_regex("novelty")
    for trial in range(8):
        text = "\n".join(rng.choice(_PDF_FRAGMENTS) for _ in range(rng.randint(150, 400)))
        config = _config(tmp_path / str(trial))
        ingest_records(
            config, fake_embedder, [{"source": "doc", "abstract": text}], progress=_quiet
        )
        hits = keyword_hits(get_collection(config), "novelty", progress=_quiet)
        _, literature, _ = build_passages("novelty", hits, mywork_classifier({}), config)
        expected = sorted(s for s in split_sentences(text) if regex.search(s))
        assert sorted(p.text for p in literature) == expected


def test_chunks_without_stored_sentence_lengths_are_refused_and_reingested(tmp_path, fake_embedder):
    config = _config(tmp_path)
    record = {"source": "doc", "abstract": "Novelty matters. Reviewers agree."}
    ingest_records(config, fake_embedder, [record], progress=_quiet)
    collection = get_collection(config)
    stored = collection.get(include=["documents", "embeddings", "metadatas"])
    collection.delete(ids=stored["ids"])
    collection.add(
        ids=stored["ids"],
        documents=stored["documents"],
        embeddings=stored["embeddings"],
        metadatas=[
            {k: v for k, v in m.items() if k != "sentence_lengths"} for m in stored["metadatas"]
        ],
    )
    hits = keyword_hits(collection, "novelty", progress=_quiet)
    with pytest.raises(ConfigError, match="re-ingest"):
        build_passages("novelty", hits, mywork_classifier({}), config)
    assert ingest_records(config, fake_embedder, [record], progress=_quiet)["records_ingested"] == 1
    assert all("sentence_lengths" in m for m in collection.get()["metadatas"])


# --- trailing spaced dots and detached punctuation ---------------------------


@pytest.mark.parametrize(
    "ending", [". . .", ". .", ". . . .", "x . . .", "?!", "!?", "pp. ?!", "Vol. ?!"]
)
def test_trailing_punctuation_pysbd_drops_is_kept(ending):
    text = "Novelty is rewarded in science. " + ending
    sentences = split_sentences(text)
    assert " ".join(sentences) == text
    assert sentences[0].startswith("Novelty is rewarded in science.")


def test_uncovered_letters_still_raise():
    from cda.extraction.windows import _sentence_starts

    with pytest.raises(ValueError, match="did not cover"):
        _sentence_starts("Alpha beta. Gamma delta.", ["Alpha beta."])


@pytest.mark.parametrize("ending", [" . . .", "…", "..."])
def test_draft_ending_in_an_ellipsis_is_read(tmp_path, ending):
    drafts = {"intro.md": "Novelty claims are rising. TODO: finish this" + ending}
    passages = extract_draft_passages("novelty", drafts, None, _config(tmp_path))
    assert [p.text for p in passages] == ["Novelty claims are rising."]


# --- a failed refresh never deletes the stored version ------------------


def test_failed_embedding_keeps_the_stored_version(tmp_path, fake_embedder, text_pdfs):
    config = _config(tmp_path)
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "a.pdf").write_text("Novelty matters in science.")
    ingest_pdf_dir(config, fake_embedder, lib, progress=_quiet)
    before = get_collection(config).get()["ids"]

    class FailingEmbedder:
        name = "failing"

        def embed(self, texts):
            raise RuntimeError("embedding backend failed")

    (lib / "a.pdf").write_text("Novelty matters in science, revised.")
    with pytest.raises(RuntimeError, match="embedding backend failed"):
        ingest_pdf_dir(config, FailingEmbedder(), lib, progress=_quiet)
    assert get_collection(config).get()["ids"] == before


def test_failed_segmentation_keeps_the_stored_version(
    tmp_path, fake_embedder, text_pdfs, monkeypatch
):
    config = _config(tmp_path)
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "a.pdf").write_text("Novelty matters in science.")
    (lib / "b.pdf").write_text("Another paper about novelty.")
    ingest_pdf_dir(config, fake_embedder, lib, progress=_quiet)
    old_text = _stored(config)["a.pdf"]["sentence_context"]

    original = ing.sentence_context_records

    def segment(text, *args):
        if "revised" in text:
            raise ValueError("Sentence segmentation did not cover the text.")
        return original(text, *args)

    monkeypatch.setattr(ing, "sentence_context_records", segment)
    (lib / "a.pdf").write_text("Novelty matters in science, revised.")
    (lib / "c.pdf").write_text("A new paper about novelty.")
    messages: list[str] = []
    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=messages.append)
    stored = _stored(config)
    assert set(stored) == {"a.pdf", "b.pdf", "c.pdf"}
    assert stored["a.pdf"]["sentence_context"] == old_text
    assert stats["pdfs_unreadable"] == 1 and stats["pdfs_kept_previous"] == 1
    assert any("previously stored version was kept" in m for m in messages)


def _encrypted_pdf(path):
    import fitz

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Novelty behind a password.")
    doc.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")


def _plain_pdf(path, text):
    import fitz

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    doc.save(str(path))


def test_encrypted_pdf_is_unreadable_and_the_ingest_continues(tmp_path, fake_embedder):
    config = _config(tmp_path)
    lib = tmp_path / "lib"
    lib.mkdir()
    _plain_pdf(lib / "a.pdf", "Novelty matters in a.")
    _encrypted_pdf(lib / "b_locked.pdf")
    _plain_pdf(lib / "c.pdf", "Novelty matters in c.")
    messages: list[str] = []
    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=messages.append)
    assert set(_stored(config)) == {"a.pdf", "c.pdf"}
    assert stats["pdfs_unreadable"] == 1
    assert any("b_locked.pdf" in m and "could not be read" in m for m in messages)


def test_pdf_that_becomes_encrypted_keeps_its_previous_version(tmp_path, fake_embedder):
    config = _config(tmp_path)
    lib = tmp_path / "lib"
    lib.mkdir()
    _plain_pdf(lib / "a.pdf", "Novelty matters in a.")
    ingest_pdf_dir(config, fake_embedder, lib, progress=_quiet)
    _encrypted_pdf(lib / "a.pdf")
    messages: list[str] = []
    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=messages.append)
    assert "Novelty matters in a." in _stored(config)["a.pdf"]["sentence_context"]
    assert stats["pdfs_kept_previous"] == 1
    assert any("a.pdf (previous version kept)" in m for m in messages)


# --- re-ingest without --zotero-csv keeps Zotero fields ----------------------


def test_refresh_without_zotero_csv_keeps_zotero_fields(tmp_path, fake_embedder, text_pdfs):
    config = _config(tmp_path)
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "mine.pdf").write_text("Novelty matters in my own paper.")
    csv_path = tmp_path / "z.csv"
    csv_path.write_text(
        "Title,Item Type,Publication Year,Date Added,Manual Tags,File Attachments\n"
        "My paper,journalArticle,2020,2024-01-02 03:04:05,mywork,/x/mine.pdf\n"
    )
    ingest_pdf_dir(config, fake_embedder, lib, csv_path, progress=_quiet)
    fields = ("title", "year", "date_added", "zotero_tags", "zotero_match", "zotero_csv_sha256")
    before = {key: _stored(config)["mine.pdf"][key] for key in fields}

    (lib / "mine.pdf").write_text("Novelty matters in my own paper, revised.")
    messages: list[str] = []
    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=messages.append)
    after = _stored(config)["mine.pdf"]
    assert stats["pdfs_refreshed"] == 1
    assert {key: after.get(key) for key in fields} == before
    assert "revised" in after["sentence_context"]
    assert any("kept its existing Zotero metadata" in m for m in messages)


# --- one file is stored once ---------------------------------------------------


def test_nested_roots_do_not_store_one_file_twice(tmp_path, fake_embedder, text_pdfs):
    config = _config(tmp_path)
    lib = tmp_path / "lib"
    (lib / "sub").mkdir(parents=True)
    (lib / "sub" / "x.pdf").write_text("Novelty matters here.")
    ingest_pdf_dir(config, fake_embedder, lib / "sub", progress=_quiet)
    messages: list[str] = []
    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=messages.append)
    assert set(_stored(config)) == {"x.pdf"}
    assert get_collection(config).count() == 1
    assert stats["pdfs_duplicates_skipped"] == 1
    assert any("sub/x.pdf" in m and "'x.pdf'" in m for m in messages)


def test_renamed_file_replaces_its_old_name(tmp_path, fake_embedder, text_pdfs):
    """A case-only rename (or any rename): the old name is gone under its
    exact spelling, so the new name replaces it instead of adding a copy."""
    config = _config(tmp_path)
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "Smith2020.pdf").write_text("Novelty matters here.")
    ingest_pdf_dir(config, fake_embedder, lib, progress=_quiet)
    os.rename(lib / "Smith2020.pdf", lib / "smith2020.pdf")
    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=_quiet)
    assert set(_stored(config)) == {"smith2020.pdf"}
    assert stats["pdfs_renamed"] == 1


def test_missing_source_check_respects_letter_case(tmp_path, fake_embedder, text_pdfs):
    config = _config(tmp_path)
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "Smith2020.pdf").write_text("Novelty matters here.")
    ingest_pdf_dir(config, fake_embedder, lib, progress=_quiet)
    os.rename(lib / "Smith2020.pdf", lib / "smith2020.pdf")
    pdf_root = str(lib.resolve())
    missing, _ = ing._missing_sources(get_collection(config), lib, pdf_root)
    assert missing == ["Smith2020.pdf"]


def test_identical_copies_in_one_library_are_stored_once(tmp_path, fake_embedder, text_pdfs):
    config = _config(tmp_path)
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "a.pdf").write_text("Novelty matters here.")
    (lib / "copy_of_a.pdf").write_text("Novelty matters here.")
    messages: list[str] = []
    stats = ingest_pdf_dir(config, fake_embedder, lib, progress=messages.append)
    assert set(_stored(config)) == {"a.pdf"}
    assert stats["pdfs_duplicates_skipped"] == 1
    assert any("copy_of_a.pdf" in m and "'a.pdf'" in m for m in messages)


# --- sources without a recorded root are never pruned -------------------------


def test_root_unknown_sources_are_not_pruned(tmp_path, fake_embedder, text_pdfs):
    config = _config(tmp_path)
    lib_a, lib_b = tmp_path / "libA", tmp_path / "libB"
    lib_a.mkdir()
    lib_b.mkdir()
    (lib_a / "a1.pdf").write_text("Novelty one.")
    (lib_b / "b1.pdf").write_text("Novelty three.")
    ingest_pdf_dir(config, fake_embedder, lib_a, progress=_quiet)
    collection = get_collection(config)
    stored = collection.get(include=["documents", "embeddings", "metadatas"])
    collection.delete(ids=stored["ids"])
    collection.add(  # a store written before pdf_root was recorded
        ids=stored["ids"],
        documents=stored["documents"],
        embeddings=stored["embeddings"],
        metadatas=[{k: v for k, v in m.items() if k != "pdf_root"} for m in stored["metadatas"]],
    )
    messages: list[str] = []
    stats = ingest_pdf_dir(
        config, fake_embedder, lib_b, prune_missing=True, progress=messages.append
    )
    assert set(_stored(config)) == {"a1.pdf", "b1.pdf"}
    assert stats["pdfs_pruned"] == 0 and stats["pdfs_root_unknown"] == 1
    assert any("root unknown; not pruned" in m and "a1.pdf" in m for m in messages)


# --- a cited library title does not become a PDF's Zotero record ------------


def test_short_title_quoted_with_another_makes_the_match_ambiguous():
    from cda.corpus.zotero import ZoteroRecord, build_doi_index, build_title_index, match_by_content

    records = [
        ZoteroRecord(
            title="Gendered Citation Returns to Novelty Claims in Science", year="2024", doi=""
        ),
        ZoteroRecord(title="A Reply to Lee", year="2025", doi=""),
    ]
    text = (
        "A Reply to Lee\nJohn Smith\nThis comment responds to Gendered Citation "
        "Returns to Novelty Claims in Science (Lee 2024)."
    )
    assert match_by_content(text, build_doi_index(records), build_title_index(records)) == (
        None,
        "ambiguous_title",
    )
    alone = "A Reply to Lee\nJohn Smith\nThis comment responds to a recent article."
    # A short title alone is too common a phrase to attach a record.
    record, outcome = match_by_content(alone, build_doi_index(records), build_title_index(records))
    assert (record, outcome) == (None, "short_title_only")


def test_titles_beyond_the_opening_region_are_not_matched():
    from cda.corpus.zotero import (
        _TITLE_SEARCH_CHARS,
        ZoteroRecord,
        build_doi_index,
        build_title_index,
        match_by_content,
    )

    records = [ZoteroRecord(title="A Reply to Lee", year="2025", doi="")]
    text = "x " * _TITLE_SEARCH_CHARS + "A Reply to Lee"
    assert match_by_content(text, build_doi_index(records), build_title_index(records)) == (
        None,
        "",
    )


def test_cited_title_is_not_attached_at_ingest(tmp_path, fake_embedder, text_pdfs):
    config = _config(tmp_path)
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "commentary_download.pdf").write_text(
        "A Reply to Lee\nJohn Smith\nThis comment responds to Gendered Citation Returns "
        "to Novelty Claims in Science (Lee 2024). Novelty claims are overrated."
    )
    csv_path = tmp_path / "z.csv"
    csv_path.write_text(
        "Title,Item Type,Publication Year,Date Added,Manual Tags,File Attachments\n"
        "Gendered Citation Returns to Novelty Claims in Science,journalArticle,2024,"
        "2024-01-02 03:04:05,mywork,/x/lee2024.pdf\n"
        "A Reply to Lee,journalArticle,2025,2025-06-01 00:00:00,,/x/smith_reply.pdf\n"
    )
    ingest_pdf_dir(config, fake_embedder, lib, csv_path, progress=_quiet)
    metadata = _stored(config)["commentary_download.pdf"]
    assert "zotero_tags" not in metadata and "title" not in metadata
    assert metadata["zotero_unmatched_reason"] == "ambiguous_title"


# --- embedding provenance from a non-default cache -----------------------------


def _fake_sentence_transformers(monkeypatch, loaded: list):
    class FakeModel:
        device = "cpu"

        def __init__(self, name, **kwargs):
            loaded.append(kwargs)

        def encode(self, texts, **kwargs):
            return [[0.0] for _ in texts]

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeModel)
    )


def test_provenance_resolves_the_snapshot_in_sentence_transformers_home(tmp_path, monkeypatch):
    from cda.corpus.embeddings import SentenceTransformerEmbedder

    revision = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
    repo = tmp_path / "st_home" / "models--org--model"
    (repo / "snapshots" / revision).mkdir(parents=True)
    (repo / "snapshots" / revision / "config.json").write_text("{}")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(revision)
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path / "st_home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "default_cache"))
    loaded: list = []
    _fake_sentence_transformers(monkeypatch, loaded)

    embedder = SentenceTransformerEmbedder("org/model")
    embedder.embed(["hello"])
    result = embedder.provenance()
    assert loaded[0]["cache_folder"] == str(tmp_path / "st_home")
    assert result["resolved_revision"] == revision
    assert list(result["files"]) == ["config.json"]


def test_unresolvable_provenance_is_recorded_not_raised(tmp_path, monkeypatch):
    from cda.corpus.embeddings import SentenceTransformerEmbedder

    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path / "empty_home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "default_cache"))
    _fake_sentence_transformers(monkeypatch, [])
    embedder = SentenceTransformerEmbedder("org/model")
    embedder.embed(["hello"])
    result = embedder.provenance()
    assert result["resolved_revision"] is None
    assert result["status"].startswith("unresolved")


def test_short_library_title_alone_never_attaches():
    from cda.corpus.zotero import ZoteroRecord, build_doi_index, build_title_index, match_by_content

    short = ZoteroRecord(title="Novelty", year="2020", doi="")
    records = [short]
    text = "A study of scientific novelty and its reception among reviewers."
    record, outcome = match_by_content(text, build_doi_index(records), build_title_index(records))
    assert record is None
    assert outcome == "short_title_only"


def test_short_title_only_outcome_is_recorded_without_an_ambiguity_warning(tmp_path):
    from collections import Counter
    from pathlib import Path

    from cda.corpus.ingest import _zotero_fields
    from cda.corpus.zotero import ZoteroRecord, build_doi_index, build_title_index

    records = [ZoteroRecord(title="Novelty", year="2020", doi="")]
    zotero = {"filename": {}, "doi": build_doi_index(records), "title": build_title_index(records)}
    messages: list[str] = []
    stats: Counter = Counter()
    fields = _zotero_fields(
        Path(tmp_path / "paper.pdf"),
        "paper.pdf",
        "On novelty in science.",
        zotero,
        False,
        stats,
        messages.append,
    )
    assert fields == {"zotero_unmatched_reason": "short_title_only"}
    assert stats["zotero_ambiguous"] == 0 and not messages
