"""Filename-dependent Zotero matching and recovery from failed file-property reads."""

from pathlib import Path

import numpy as np
import pytest

import cda.corpus.ingest as ing
from cda.config import load_config
from cda.extraction.mywork import mywork_classifier


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    monkeypatch.setattr(ing, "extract_pdf_text", lambda path: Path(path).read_text())
    return root, config


def _quiet(_message):
    pass


def _rows(config):
    return ing.get_collection(config).get(include=["documents", "metadatas", "embeddings"])


def _forbid_rebuild(*_args, **_kwargs):
    pytest.fail("an unchanged PDF was extracted or embedded again")


@pytest.mark.parametrize("old_name", ["unmatched.pdf", "other.pdf"])
@pytest.mark.parametrize("defer_export", [False, True])
def test_rename_rematches_the_same_export(
    tmp_path, monkeypatch, library, fake_embedder, old_name, defer_export
):
    root, config = library
    csv = tmp_path / "zotero.csv"
    csv.write_text(
        "Title,Item Type,Publication Year,File Attachments,Manual Tags\n"
        "Researcher's paper,journalArticle,2024,matched.pdf,#MyWork\n"
        "Other paper,journalArticle,2020,other.pdf,literature\n"
    )
    old = root / old_name
    old.write_text("Novelty has a stable meaning. " * 30)
    ing.ingest_pdf_dir(config, fake_embedder, root, csv, progress=_quiet)
    before = _rows(config)
    old.rename(root / "matched.pdf")
    monkeypatch.setattr(fake_embedder, "embed", _forbid_rebuild)
    monkeypatch.setattr(ing, "extract_pdf_text", _forbid_rebuild)
    if defer_export:
        ing.ingest_pdf_dir(config, fake_embedder, root, progress=_quiet)
    ing.ingest_pdf_dir(config, fake_embedder, root, csv, progress=_quiet)
    after = _rows(config)
    assert after["documents"] == before["documents"]
    assert np.allclose(after["embeddings"], before["embeddings"], rtol=1e-6, atol=1e-7)
    classify = mywork_classifier({"priority": ["zotero_tags"]})
    for meta in after["metadatas"]:
        assert meta["title"] == "Researcher's paper"
        assert meta["year"] == "2024"
        assert meta["zotero_match_filename"] == "matched.pdf"
        assert "zotero_unmatched_reason" not in meta
        assert classify("matched.pdf", meta) == (True, "zotero:mywork")
    stats = ing.ingest_pdf_dir(config, fake_embedder, root, csv, progress=_quiet)
    assert stats["zotero_unchanged"] == 1


def test_old_store_with_stale_match_is_repaired_once(tmp_path, monkeypatch, library, fake_embedder):
    root, config = library
    csv = tmp_path / "zotero.csv"
    csv.write_text(
        "Title,Item Type,Publication Year,File Attachments,Manual Tags\n"
        "Researcher's paper,journalArticle,2024,matched.pdf,#MyWork\n"
    )
    old = root / "unmatched.pdf"
    old.write_text("Novelty has a stable meaning. " * 30)
    ing.ingest_pdf_dir(config, fake_embedder, root, csv, progress=_quiet)
    old.rename(root / "matched.pdf")
    ing.ingest_pdf_dir(config, fake_embedder, root, progress=_quiet)
    collection = ing.get_collection(config)
    rows = _rows(config)
    # A store from before filename tracking can hold this no-match result
    # under the corrected name while already carrying the current CSV hash.
    collection.update(
        ids=rows["ids"], metadatas=[{"zotero_match_filename": None}] * len(rows["ids"])
    )
    monkeypatch.setattr(fake_embedder, "embed", _forbid_rebuild)
    monkeypatch.setattr(ing, "extract_pdf_text", _forbid_rebuild)
    stats = ing.ingest_pdf_dir(config, fake_embedder, root, csv, progress=_quiet)
    assert stats["metadata_updated"] == 1
    assert all(m["zotero_tags"] == "#MyWork" for m in _rows(config)["metadatas"])
    stats = ing.ingest_pdf_dir(config, fake_embedder, root, csv, progress=_quiet)
    assert stats["zotero_unchanged"] == 1 and stats["metadata_updated"] == 0


@pytest.mark.parametrize("prune", [False, True])
@pytest.mark.parametrize("error", [PermissionError, FileNotFoundError, OverflowError])
def test_failed_properties_keep_the_old_copy_and_continue(
    monkeypatch, library, fake_embedder, prune, error
):
    root, config = library
    old = root / "original.pdf"
    old.write_text("Novelty has an earlier stored meaning. " * 30)
    ing.ingest_pdf_dir(config, fake_embedder, root, progress=_quiet)
    before = _rows(config)
    old.rename(root / "aa.pdf")
    (root / "zz.pdf").write_text("Novelty has a separate new meaning. " * 30)
    real_date = ing.file_acquisition_date

    def broken_date(path):
        if path.name == "aa.pdf":
            raise error("file properties unavailable after hashing")
        return real_date(path)

    monkeypatch.setattr(ing, "file_acquisition_date", broken_date)
    stats = ing.ingest_pdf_dir(config, fake_embedder, root, prune_missing=prune, progress=_quiet)
    after = _rows(config)
    assert stats["pdfs_unreadable"] == 1
    assert stats["pdfs_kept_previous"] == 1
    assert stats["pdfs_pruned"] == 0
    assert {m["pdf_relpath"] for m in after["metadatas"]} == {"original.pdf", "zz.pdf"}
    old_rows = ing.get_collection(config).get(
        where={"pdf_relpath": "original.pdf"}, include=["documents", "metadatas", "embeddings"]
    )
    assert old_rows["ids"] == before["ids"]
    assert old_rows["documents"] == before["documents"]
    assert old_rows["metadatas"] == before["metadatas"]
    assert np.array_equal(old_rows["embeddings"], before["embeddings"])
    monkeypatch.setattr(ing, "file_acquisition_date", real_date)
    stats = ing.ingest_pdf_dir(config, fake_embedder, root, prune_missing=prune, progress=_quiet)
    assert stats["pdfs_unreadable"] == 0
    assert {m["pdf_relpath"] for m in _rows(config)["metadatas"]} == {"aa.pdf", "zz.pdf"}


@pytest.mark.parametrize("failure", ["unreadable", "no_text", "segmentation"])
@pytest.mark.parametrize("rebuild", ["reingest", "chunk_settings"])
def test_failed_rebuild_of_a_rename_keeps_the_old_copy_under_prune(
    tmp_path, monkeypatch, library, fake_embedder, failure, rebuild
):
    # A rename that cannot reuse the stored chunks (--reingest, or new chunk
    # settings) is rebuilt from the file; when that rebuild fails, the old
    # name is the document's only stored version and must survive pruning.
    root, config = library
    old = root / "original.pdf"
    old.write_text("Novelty has an earlier stored meaning. " * 30)
    ing.ingest_pdf_dir(config, fake_embedder, root, progress=_quiet)
    before = _rows(config)
    old.rename(root / "aa.pdf")
    reingest = rebuild == "reingest"
    if not reingest:
        config = load_config(
            overrides={
                "paths": {"corpus_dir": str(tmp_path / "corpus")},
                "corpus": {"chunk_size_words": 400},
            }
        )
    real_extract, real_segment = ing.extract_pdf_text, ing.sentence_context_records

    def broken_extract(path):
        raise ValueError("extractor failed")

    def broken_segment(*_args):
        raise ValueError("Sentence segmentation did not cover the text.")

    if failure == "unreadable":
        monkeypatch.setattr(ing, "extract_pdf_text", broken_extract)
    elif failure == "no_text":
        monkeypatch.setattr(ing, "extract_pdf_text", lambda path: "")
    else:
        monkeypatch.setattr(ing, "sentence_context_records", broken_segment)
    stats = ing.ingest_pdf_dir(
        config, fake_embedder, root, reingest=reingest, prune_missing=True, progress=_quiet
    )
    assert stats["pdfs_pruned"] == 0
    assert stats["pdfs_kept_previous"] == 1
    old_rows = ing.get_collection(config).get(
        where={"pdf_relpath": "original.pdf"}, include=["documents", "metadatas", "embeddings"]
    )
    assert old_rows["ids"] == before["ids"]
    assert old_rows["documents"] == before["documents"]
    assert np.array_equal(old_rows["embeddings"], before["embeddings"])
    monkeypatch.setattr(ing, "extract_pdf_text", real_extract)
    monkeypatch.setattr(ing, "sentence_context_records", real_segment)
    stats = ing.ingest_pdf_dir(
        config, fake_embedder, root, reingest=reingest, prune_missing=True, progress=_quiet
    )
    assert stats["pdfs_ingested"] == 1 and stats["pdfs_renamed"] == 1
    assert {m["pdf_relpath"] for m in _rows(config)["metadatas"]} == {"aa.pdf"}
