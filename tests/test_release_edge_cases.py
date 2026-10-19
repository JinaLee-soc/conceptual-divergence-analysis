"""Data preservation and literal retrieval at source and sentence boundaries."""

import numpy as np
import pytest

import cda.corpus.ingest as ing
from cda.extraction.windows import normalize_text, split_sentences


def _snapshot(collection):
    stored = collection.get(include=["documents", "metadatas", "embeddings"])
    return {
        i: (d, m, np.asarray(e).tolist())
        for i, d, m, e in zip(
            stored["ids"],
            stored["documents"],
            stored["metadatas"],
            stored["embeddings"],
            strict=True,
        )
    }


@pytest.fixture
def library(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    lib.mkdir()
    monkeypatch.setattr(ing, "extract_pdf_text", lambda path: path.read_text())
    return lib


def test_long_sentence_is_never_split_at_a_block_size():
    sentence = "Novelty " + "persists " * 1600 + "through this entire sentence."
    assert split_sentences(sentence) == [normalize_text(sentence)]


def test_docx_style_paragraph_break_does_not_split_a_sentence():
    # PDF extractors can put a blank line at a page boundary mid-sentence.
    text = "Novelty persists across\n\npage boundaries in this sentence."
    assert split_sentences(text) == [normalize_text(text)]


@pytest.mark.parametrize("kind", ["pdf", "record"])
def test_failed_store_restores_previous_source(
    tmp_config, fake_embedder, library, monkeypatch, kind
):
    path = library / "paper.pdf"
    path.write_text("Novelty has an earlier interpretation.")

    def run(text):
        if kind == "pdf":
            path.write_text(text)
            ing.ingest_pdf_dir(tmp_config, fake_embedder, library, progress=lambda _: None)
        else:
            ing.ingest_records(
                tmp_config,
                fake_embedder,
                [{"source": "paper", "abstract": text}],
                progress=lambda _: None,
            )

    run("Novelty has an earlier interpretation.")
    collection = ing.get_collection(tmp_config)
    before = _snapshot(collection)
    real = type(collection).add
    failed = False

    def fail_once(self, **kwargs):
        nonlocal failed
        real(self, **kwargs)  # also exercise a backend that wrote before raising
        if not failed:
            failed = True
            raise RuntimeError("injected database write failure")

    monkeypatch.setattr(type(collection), "add", fail_once)
    with pytest.raises(RuntimeError, match="injected database write failure"):
        run("Novelty has a changed interpretation.")
    assert _snapshot(collection) == before


def test_changed_file_that_becomes_duplicate_does_not_keep_old_text(
    tmp_config, fake_embedder, library
):
    a, b = library / "a.pdf", library / "b.pdf"
    a.write_text("Novelty uses the common interpretation.")
    b.write_text("Novelty uses the old unrelated interpretation.")
    ing.ingest_pdf_dir(tmp_config, fake_embedder, library, progress=lambda _: None)
    b.write_text(a.read_text())
    ing.ingest_pdf_dir(tmp_config, fake_embedder, library, progress=lambda _: None)
    values = list(_snapshot(ing.get_collection(tmp_config)).values())
    assert len(values) == 1
    assert values[0][0] == a.read_text()


def test_pruning_excluded_copy_keeps_selected_identical_copy(tmp_config, fake_embedder, library):
    (library / "excluded").mkdir()
    a = library / "excluded" / "a.pdf"
    a.write_text("Novelty uses the common interpretation.")
    ing.ingest_pdf_dir(tmp_config, fake_embedder, library, progress=lambda _: None)
    b = library / "b.pdf"
    b.write_text(a.read_text())
    ing.ingest_pdf_dir(
        tmp_config,
        fake_embedder,
        library,
        exclude=["excluded"],
        prune_missing=True,
        progress=lambda _: None,
    )
    values = list(_snapshot(ing.get_collection(tmp_config)).values())
    assert len(values) == 1
    assert values[0][1]["pdf_relpath"] == "b.pdf"


def test_same_csv_refreshes_match_when_shared_basename_appears(
    tmp_config, fake_embedder, library, tmp_path
):
    csv = tmp_path / "library.csv"
    csv.write_text(
        "Title,Item Type,Publication Year,Date Added,Manual Tags,File Attachments\nA title,journalArticle,2020,2026-01-01,mywork,paper.pdf\n"
    )
    a = library / "paper.pdf"
    a.write_text("Novelty holds in the first paper.")
    ing.ingest_pdf_dir(tmp_config, fake_embedder, library, csv, progress=lambda _: None)
    assert next(iter(_snapshot(ing.get_collection(tmp_config)).values()))[1]["title"] == "A title"
    (library / "other").mkdir()
    (library / "other" / "paper.pdf").write_text("Novelty holds in the different second paper.")
    ing.ingest_pdf_dir(tmp_config, fake_embedder, library, csv, progress=lambda _: None)
    metadata = [v[1] for v in _snapshot(ing.get_collection(tmp_config)).values()]
    assert len(metadata) == 2
    assert all(
        "title" not in m and m["zotero_unmatched_reason"] == "shared_basename" for m in metadata
    )


def test_old_digest_of_changed_copy_does_not_suppress_new_source(
    tmp_config, fake_embedder, library
):
    original = "Novelty has the original interpretation."
    z = library / "z.pdf"
    z.write_text(original)
    ing.ingest_pdf_dir(tmp_config, fake_embedder, library, progress=lambda _: None)
    (library / "a.pdf").write_text(original)
    z.write_text("Novelty has a different interpretation now.")
    ing.ingest_pdf_dir(tmp_config, fake_embedder, library, progress=lambda _: None)
    values = list(_snapshot(ing.get_collection(tmp_config)).values())
    assert {v[1]["pdf_relpath"] for v in values} == {"a.pdf", "z.pdf"}


def test_metadata_refresh_repairs_every_chunk_not_only_first(
    tmp_config, fake_embedder, library, tmp_path
):
    csv = tmp_path / "library.csv"
    csv.write_text(
        "Title,Item Type,Publication Year,Date Added,Manual Tags,File Attachments\nA title,journalArticle,2020,2026-01-01,mywork,paper.pdf\n"
    )
    (library / "paper.pdf").write_text("Novelty has many meanings. " * 300)
    ing.ingest_pdf_dir(tmp_config, fake_embedder, library, csv, progress=lambda _: None)
    collection = ing.get_collection(tmp_config)
    ids = collection.get()["ids"]
    assert len(ids) > 1
    collection.update(
        ids=ids[1:], metadatas=[{"title": "Wrong title", "doi": "stale"}] * (len(ids) - 1)
    )
    ing.ingest_pdf_dir(tmp_config, fake_embedder, library, csv, progress=lambda _: None)
    assert all(m["title"] == "A title" and "doi" not in m for m in collection.get()["metadatas"])


def test_metadata_update_failure_rolls_back(
    tmp_config, fake_embedder, library, tmp_path, monkeypatch
):
    csv = tmp_path / "library.csv"
    csv.write_text("Title,Item Type,File Attachments\nA title,journalArticle,paper.pdf\n")
    (library / "paper.pdf").write_text("Novelty has many meanings. " * 300)
    ing.ingest_pdf_dir(tmp_config, fake_embedder, library, csv, progress=lambda _: None)
    collection = ing.get_collection(tmp_config)
    before = _snapshot(collection)
    csv.write_text("Title,Item Type,File Attachments\nChanged title,journalArticle,paper.pdf\n")
    update = type(collection).update
    failed = False

    def fail_once(self, **kwargs):
        nonlocal failed
        update(self, **kwargs)
        if not failed:
            failed = True
            raise RuntimeError("injected metadata update failure")

    monkeypatch.setattr(type(collection), "update", fail_once)
    stats = ing.ingest_pdf_dir(tmp_config, fake_embedder, library, csv, progress=lambda _: None)
    assert stats["pdfs_kept_previous"] == 1
    assert _snapshot(collection) == before


@pytest.mark.parametrize("first_kind", ["pdf", "record"])
def test_pdf_and_record_names_cannot_replace_each_other(
    tmp_config, fake_embedder, library, first_kind
):
    (library / "paper.pdf").write_text("Novelty comes from the PDF library.")
    record = [{"source": "paper.pdf", "abstract": "Novelty comes from the unrelated record."}]

    def pdf():
        ing.ingest_pdf_dir(tmp_config, fake_embedder, library, progress=lambda _: None)

    def records():
        ing.ingest_records(tmp_config, fake_embedder, record, progress=lambda _: None)

    first, second = (pdf, records) if first_kind == "pdf" else (records, pdf)
    first()
    before = _snapshot(ing.get_collection(tmp_config))
    with pytest.raises(ValueError, match="source|Source"):
        second()
    assert _snapshot(ing.get_collection(tmp_config)) == before


def test_metadata_rollback_failure_does_not_claim_previous_version_is_kept(
    tmp_config, fake_embedder, library, tmp_path, monkeypatch
):
    csv = tmp_path / "library.csv"
    csv.write_text("Title,Item Type,File Attachments\nA title,journalArticle,paper.pdf\n")
    (library / "paper.pdf").write_text("Novelty has many meanings.")
    ing.ingest_pdf_dir(tmp_config, fake_embedder, library, csv, progress=lambda _: None)
    collection = ing.get_collection(tmp_config)
    csv.write_text("Title,Item Type,File Attachments\nChanged title,journalArticle,paper.pdf\n")

    def fail(self, **kwargs):
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(type(collection), "update", fail)
    with pytest.raises(RuntimeError, match="restor|incomplete"):
        ing.ingest_pdf_dir(tmp_config, fake_embedder, library, csv, progress=lambda _: None)


def test_literal_phrase_survives_a_nonoverlapping_chunk_edge(tmp_config, fake_embedder):
    from cda.config import load_config
    from cda.extraction.retrieval import build_passages, keyword_hits

    tmp_config = load_config(
        overrides={
            "paths": {"corpus_dir": str(tmp_config.path("paths.corpus_dir"))},
            "corpus": {"chunk_size_words": 20, "chunk_overlap_words": 0},
        }
    )
    sentence = " ".join(["word"] * 19 + ["social", "novelty", "matters."])
    ing.ingest_records(
        tmp_config,
        fake_embedder,
        [{"source": "paper", "abstract": sentence}],
        progress=lambda _: None,
    )
    collection = ing.get_collection(tmp_config)
    assert all("social novelty" not in doc for doc in collection.get()["documents"])
    hits = keyword_hits(collection, "social novelty", progress=lambda _: None)
    _, passages, _ = build_passages(
        "social novelty", hits, lambda *_: (False, "literature"), tmp_config
    )
    assert [p.text for p in passages] == [sentence]


def test_phrase_candidates_without_full_match_do_not_consume_cap(tmp_config, fake_embedder):
    from cda.extraction.retrieval import keyword_hits

    ing.ingest_records(
        tmp_config,
        fake_embedder,
        [
            {"source": "a", "abstract": "Novelty alone does not match the phrase."},
            {"source": "b", "abstract": "Social novelty matters."},
        ],
        progress=lambda _: None,
    )
    messages = []
    hits = keyword_hits(
        ing.get_collection(tmp_config),
        "social novelty",
        batch_size=1,
        max_chunks=1,
        progress=messages.append,
    )
    assert [hit["document"] for hit in hits.values()] == ["Social novelty matters."]
    assert not messages


def test_file_read_error_keeps_previous_source_and_continues(
    tmp_config, fake_embedder, library, monkeypatch
):
    (library / "a.pdf").write_text("Novelty has a stored interpretation.")
    ing.ingest_pdf_dir(tmp_config, fake_embedder, library, progress=lambda _: None)
    before = _snapshot(ing.get_collection(tmp_config))
    (library / "b.pdf").write_text("Novelty has another interpretation.")
    real = ing.file_digest

    def unreadable(path):
        if path.name == "a.pdf":
            raise PermissionError("injected file access failure")
        return real(path)

    monkeypatch.setattr(ing, "file_digest", unreadable)
    stats = ing.ingest_pdf_dir(tmp_config, fake_embedder, library, progress=lambda _: None)
    after = _snapshot(ing.get_collection(tmp_config))
    assert all(after[key] == value for key, value in before.items())
    assert len(after) == 2
    assert stats["pdfs_unreadable"] == stats["pdfs_kept_previous"] == 1


def test_long_first_sentence_does_not_disable_blocking_for_the_rest():
    from cda.extraction.windows import _blocks

    first = "Novelty " + "persists " * 1600 + "through this entire sentence."
    rest = " Consensus then holds. Other sentences follow." * 500
    sentences = split_sentences(first + rest)
    assert sentences[0] == normalize_text(first)
    assert len(sentences) == 1001
    assert len(_blocks(first + rest)) > 2


def test_pruning_does_not_treat_an_inaccessible_folder_as_missing(
    tmp_config, fake_embedder, library, monkeypatch
):
    from pathlib import Path

    (library / "paper.pdf").write_text("Novelty has a stored interpretation.")
    ing.ingest_pdf_dir(tmp_config, fake_embedder, library, progress=lambda _: None)
    before = _snapshot(ing.get_collection(tmp_config))
    real = ing.os.listdir

    def denied(path):
        if Path(path) == library:
            raise PermissionError("injected directory access failure")
        return real(path)

    monkeypatch.setattr(ing.os, "listdir", denied)
    messages: list[str] = []
    stats = ing.ingest_pdf_dir(
        tmp_config, fake_embedder, library, prune_missing=True, progress=messages.append
    )
    assert _snapshot(ing.get_collection(tmp_config)) == before
    assert stats["pdfs_unchecked"] == 1 and stats["pdfs_pruned"] == 0
    assert any("could not check; not pruned" in m and "paper.pdf" in m for m in messages)
