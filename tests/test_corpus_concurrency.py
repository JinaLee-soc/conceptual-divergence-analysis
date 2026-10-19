"""Readers must see a complete corpus even while another process refreshes it."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import cda.corpus.ingest as ingest
from cda.config import load_config
from cda.pipeline import extract_passages


@pytest.mark.parametrize("layout", ["standard", "direct", "symlink"])
def test_extraction_waits_for_source_replacement(
    tmp_path, tmp_config, fake_embedder, monkeypatch, layout
):
    ingest.ingest_records(
        tmp_config,
        fake_embedder,
        [
            {"source": "updated.abstract", "abstract": "Novelty was the old definition."},
            {"source": "stable.abstract", "abstract": "Novelty is the stable definition."},
        ],
        progress=lambda _: None,
    )
    collection = ingest.get_collection(tmp_config)
    corpus = tmp_config.path("paths.corpus_dir")
    if layout == "direct":
        corpus = corpus / "chroma"
    elif layout == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(corpus, target_is_directory=True)
        corpus = alias
    reader_config = load_config(overrides={"paths": {"corpus_dir": str(corpus)}})
    deleted, resume, reader_observed, reader_finished = [threading.Event() for _ in range(4)]
    store = ingest._store_chunks

    def pause_before_add(*args, **kwargs):
        deleted.set()
        assert resume.wait(10)
        return store(*args, **kwargs)

    def progress(message):
        if "another ingest" in message:
            reader_observed.set()

    def read():
        try:
            return extract_passages(
                "novelty",
                reader_config,
                embedder=fake_embedder,
                collection=collection,
                draft_texts={},
                progress=progress,
            )
        finally:
            reader_finished.set()
            reader_observed.set()

    monkeypatch.setattr(ingest, "_store_chunks", pause_before_add)
    monkeypatch.setattr(ingest, "_INGEST_LOCK_POLL_SECONDS", 0.01)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(
            ingest.ingest_records,
            tmp_config,
            fake_embedder,
            [
                {"source": "updated.abstract", "abstract": "Novelty has the new definition."},
            ],
            progress=lambda _: None,
        )
        try:
            assert deleted.wait(10)
            reader = pool.submit(read)
            assert reader_observed.wait(10)
            completed_during_write = reader_finished.is_set()
        finally:
            resume.set()
        writer.result(timeout=10)
        _, literature, _ = reader.result(timeout=10)
    assert not completed_during_write
    assert {p.text for p in literature} == {
        "Novelty has the new definition.",
        "Novelty is the stable definition.",
    }


def test_writer_waits_for_all_retrieval_pages(tmp_config, fake_embedder, monkeypatch):
    from types import SimpleNamespace

    import cda.pipeline as pipeline
    from cda.extraction.retrieval import keyword_hits

    ingest.ingest_records(
        tmp_config,
        fake_embedder,
        [
            {"source": "first.abstract", "abstract": "Novelty is first."},
            {"source": "second.abstract", "abstract": "Novelty is second."},
        ],
        progress=lambda _: None,
    )
    collection = ingest.get_collection(tmp_config)
    first_page, resume, writer_observed, writer_finished = [threading.Event() for _ in range(4)]

    def get(**kwargs):
        result = collection.get(**kwargs)
        if kwargs.get("offset") == 0:
            first_page.set()
            assert resume.wait(10)
        return result

    def paged(*args, **kwargs):
        return keyword_hits(*args, **kwargs, batch_size=1)

    def write_progress(message):
        if "another ingest" in message:
            writer_observed.set()

    def write():
        try:
            return ingest.ingest_records(
                tmp_config,
                fake_embedder,
                [
                    {"source": "second.abstract", "abstract": "Novelty changed."},
                ],
                progress=write_progress,
            )
        finally:
            writer_finished.set()
            writer_observed.set()

    monkeypatch.setattr(pipeline, "keyword_hits", paged)
    monkeypatch.setattr(ingest, "_INGEST_LOCK_POLL_SECONDS", 0.01)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(
            extract_passages,
            "novelty",
            tmp_config,
            embedder=fake_embedder,
            collection=SimpleNamespace(get=get),
            draft_texts={},
            progress=lambda _: None,
        )
        try:
            assert first_page.wait(10)
            writer = pool.submit(write)
            assert writer_observed.wait(10)
            completed_during_read = writer_finished.is_set()
        finally:
            resume.set()
        _, literature, _ = reader.result(timeout=10)
        writer.result(timeout=10)
    assert not completed_during_read
    assert {p.text for p in literature} == {"Novelty is first.", "Novelty is second."}


def test_failed_retrieval_releases_corpus_lock(tmp_config, fake_embedder):
    from types import SimpleNamespace

    from cda.config import ConfigError

    def broken_get(**kwargs):
        raise RuntimeError("unavailable store")

    with pytest.raises(ConfigError, match="incomplete"):
        extract_passages(
            "novelty",
            tmp_config,
            embedder=fake_embedder,
            collection=SimpleNamespace(get=broken_get),
            draft_texts={},
            progress=lambda _: None,
        )
    assert not (tmp_config.path("paths.corpus_dir") / "chroma" / ".ingest.lock").exists()


def test_diagnosis_waits_for_corpus_writes(tmp_config, monkeypatch):
    from types import SimpleNamespace

    from cda.diagnose.report import gather_composition

    def get(**kwargs):
        raise AssertionError("read started while the corpus was being updated")

    monkeypatch.setattr(ingest, "_INGEST_LOCK_TIMEOUT_SECONDS", 0)
    with ingest.corpus_lock(tmp_config, lambda _: None):
        with pytest.raises(RuntimeError, match="ingest.lock"):
            gather_composition(
                tmp_config, collection=SimpleNamespace(get=get), progress=lambda _: None
            )
