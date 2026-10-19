"""Boundary regressions for retrieval and ingestion, using synthetic sources only."""

from types import SimpleNamespace

import pytest

from cda.config import Config, ConfigError
from cda.corpus.ingest import get_collection, ingest_pdf_dir, ingest_records
from cda.extraction.retrieval import keyword_hits


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_nonfinite_integer_config_has_actionable_error(value):
    with pytest.raises(ConfigError, match="integer"):
        Config({"x": value}).get_int("x")


def test_duplicate_retrieval_ids_abort_instead_of_overwrite():
    collection = SimpleNamespace(
        get=lambda **kw: {
            "ids": ["same", "same"],
            "documents": ["Novelty one.", "Novelty two."],
            "metadatas": [{}, {}],
        }
    )
    with pytest.raises(ConfigError, match="duplicate"):
        keyword_hits(collection, "novelty")


def test_repeated_retrieval_page_aborts():
    def get(**kwargs):
        if kwargs["offset"] >= 2:
            return {"ids": []}
        return {"ids": ["same"], "documents": ["Novelty."], "metadatas": [{}]}

    with pytest.raises(ConfigError, match="duplicate"):
        keyword_hits(SimpleNamespace(get=get), "novelty", batch_size=1)


@pytest.mark.parametrize("options", [{"batch_size": 0}, {"batch_size": -1}, {"max_chunks": -1}])
def test_invalid_retrieval_bounds_are_refused(options):
    collection = SimpleNamespace(get=lambda **kw: pytest.fail("invalid bounds reached store"))
    with pytest.raises(ValueError):
        keyword_hits(collection, "novelty", **options)


def test_changed_record_metadata_is_refreshed(tmp_config, fake_embedder):
    record = {
        "source": "toy",
        "abstract": "Novelty matters.",
        "year": "2020",
        "zotero_tags": "mywork",
    }
    ingest_records(tmp_config, fake_embedder, [record])
    record.update(year="2021", zotero_tags="literature")
    assert ingest_records(tmp_config, fake_embedder, [record])["records_ingested"] == 1
    meta = get_collection(tmp_config).get(include=["metadatas"])["metadatas"][0]
    assert meta["year"] == "2021" and meta["zotero_tags"] == "literature"
    del record["zotero_tags"]
    ingest_records(tmp_config, fake_embedder, [record])
    assert (
        "zotero_tags" not in get_collection(tmp_config).get(include=["metadatas"])["metadatas"][0]
    )


def test_changed_chunk_settings_refresh_records(tmp_config, fake_embedder):
    records = [{"source": "toy", "abstract": "Novelty matters. " * 300}]
    ingest_records(tmp_config, fake_embedder, records)
    tmp_config.data["corpus"].update(chunk_size_words=100, chunk_overlap_words=20)
    assert ingest_records(tmp_config, fake_embedder, records)["records_ingested"] == 1
    documents = get_collection(tmp_config).get(include=["documents"])["documents"]
    assert all(len(text.split()) <= 100 for text in documents)


def test_identical_source_ids_refused_before_ingestion(tmp_config, fake_embedder):
    records = [
        {"source": "toy", "abstract": text} for text in ["Novelty first.", "Novelty second."]
    ]
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        ingest_records(tmp_config, fake_embedder, records)
    assert not tmp_config.path("paths.corpus_dir").exists()


def test_different_pdf_roots_cannot_overwrite_same_relative_source(
    tmp_config, fake_embedder, tmp_path, monkeypatch
):
    monkeypatch.setattr("cda.corpus.ingest.extract_pdf_text", lambda path: path.read_text())
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "paper.pdf").write_text("Novelty first.")
    (second / "paper.pdf").write_text("Novelty second.")
    ingest_pdf_dir(tmp_config, fake_embedder, first)
    before = get_collection(tmp_config).get(include=["documents", "metadatas"])
    with pytest.raises(ValueError, match="root"):
        ingest_pdf_dir(tmp_config, fake_embedder, second)
    assert get_collection(tmp_config).get(include=["documents", "metadatas"]) == before


def test_supplied_decomposed_draft_text_matches_literal_term(tmp_config, fake_embedder):
    import unicodedata

    from cda.extraction.draft_passages import extract_draft_passages

    passages = extract_draft_passages(
        "café",
        {"draft": unicodedata.normalize("NFD", "The café matters.")},
        fake_embedder,
        tmp_config,
    )
    assert len(passages) == 1


def test_supplied_duplicate_drafts_are_refused(tmp_config, fake_embedder):
    from cda.extraction.draft_passages import extract_draft_passages

    with pytest.raises(ConfigError, match="identical text"):
        extract_draft_passages(
            "novelty",
            {"a": "Novelty matters.", "b": "Novelty  matters."},
            fake_embedder,
            tmp_config,
        )


def test_empty_library_can_prune_last_missing_source(
    tmp_config, fake_embedder, tmp_path, monkeypatch
):
    monkeypatch.setattr("cda.corpus.ingest.extract_pdf_text", lambda p: p.read_text())
    library = tmp_path / "library"
    library.mkdir()
    path = library / "only.pdf"
    path.write_text("Novelty matters.")
    ingest_pdf_dir(tmp_config, fake_embedder, library)
    path.unlink()
    stats = ingest_pdf_dir(tmp_config, fake_embedder, library, prune_missing=True)
    assert stats["pdfs_pruned"] == 1 and get_collection(tmp_config).count() == 0


def test_damaged_chunk_counts_are_refreshed(tmp_config, fake_embedder):
    records = [{"source": "toy", "abstract": "Novelty matters."}]
    ingest_records(tmp_config, fake_embedder, records)
    collection = get_collection(tmp_config)
    result = collection.get(include=["metadatas"])
    collection.update(ids=result["ids"], metadatas=[dict(result["metadatas"][0], total_chunks=0)])
    assert ingest_records(tmp_config, fake_embedder, records)["records_ingested"] == 1


def test_changed_pdf_chunk_settings_refresh(tmp_config, fake_embedder, tmp_path, monkeypatch):
    monkeypatch.setattr("cda.corpus.ingest.extract_pdf_text", lambda p: p.read_text())
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "a.pdf").write_text("Novelty matters. " * 300)
    ingest_pdf_dir(tmp_config, fake_embedder, lib)
    tmp_config.data["corpus"].update(chunk_size_words=100, chunk_overlap_words=20)
    assert ingest_pdf_dir(tmp_config, fake_embedder, lib)["pdfs_refreshed"] == 1


def test_provenance_ignores_pdf_root_location_but_not_metadata():
    from cda.provenance import input_snapshot, sample_snapshot
    from tests.conftest import make_passages

    hits = {"a": {"document": "Novelty.", "metadata": {"pdf_root": "/machine/one", "year": "2020"}}}
    before = input_snapshot(hits, {})
    passages = make_passages(["Novelty."])
    passages[0].metadata = dict(hits["a"]["metadata"])
    sample = sample_snapshot(passages, [])
    hits["a"]["metadata"]["pdf_root"] = "/machine/two"
    passages[0].metadata["pdf_root"] = "/machine/two"
    assert before == input_snapshot(hits, {})
    assert sample == sample_snapshot(passages, [])
    hits["a"]["metadata"]["year"] = "2021"
    assert before != input_snapshot(hits, {})
