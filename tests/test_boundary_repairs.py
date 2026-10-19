"""Failure and boundary contracts using synthetic inputs only."""

import pytest

from cda.config import ConfigError, load_config
from cda.corpus.ingest import get_collection, ingest_records
from cda.extraction.mywork import mywork_classifier
from cda.extraction.retrieval import build_passages, keyword_hits
from cda.extraction.windows import SENTENCE_SEGMENTER
from cda.registers.assemble import format_cluster_block
from cda.types import ClusterSummary


def test_later_page_failure_never_returns_partial_hits():
    class Collection:
        def get(self, **kwargs):
            if kwargs["offset"]:
                raise OSError("database unavailable")
            return {"ids": ["a"], "documents": ["Novelty."], "metadatas": [{}]}

    with pytest.raises(ConfigError, match="incomplete"):
        keyword_hits(Collection(), "novelty", batch_size=1)


def test_original_sentence_survives_word_chunk_boundaries(tmp_config, fake_embedder):
    sentence = " ".join(["Before"] + ["word"] * 448 + ["novelty"] + ["word"] * 170) + "."
    record = {"source": "synthetic.abstract", "abstract": sentence, "zotero_tags": "topic"}
    ingest_records(tmp_config, fake_embedder, [record], progress=lambda _: None)
    collection = get_collection(tmp_config)
    hits = keyword_hits(collection, "novelty")
    _, literature, _ = build_passages("novelty", hits, mywork_classifier({}), tmp_config)
    assert [p.text for p in literature] == [sentence]


def test_legacy_indexed_chunks_require_reingestion():
    hits = {
        "x": {
            "document": "Novelty cut here",
            "metadata": {"chunk_index": 0, "total_chunks": 2, "pdf_filename": "source.pdf"},
        }
    }
    with pytest.raises(ConfigError, match="re-ingest"):
        build_passages("novelty", hits, mywork_classifier({}), load_config())


def test_prompt_preserves_each_representatives_own_tags():
    summary = ClusterSummary(
        True,
        2,
        1,
        0.1,
        "",
        clusters=[
            {
                "cluster_id": 0,
                "n_passages": 2,
                "representatives": ["Novelty in science.", "Novelty in art."],
                "sources": ["science.pdf", "art.pdf"],
                "rep_roles": ["top1", "top2"],
                "rep_theme_tags": [["science"], ["art"]],
            }
        ],
    )
    prompt = format_cluster_block(summary)
    assert "Theme tags for Sample 1 (source metadata): science" in prompt
    assert "Theme tags for Sample 2 (source metadata): art" in prompt


@pytest.mark.parametrize("mode", ["joint", "independent"])
def test_clustering_keeps_tag_to_sentence_association(mode, anchored_embedder):
    from cda.clustering.joint import cluster_joint, corpus_views
    from cda.clustering.spherical import cluster_passages
    from tests.conftest import make_passages

    passages = make_passages(
        [f"{topic} novelty {i}." for topic in ["alpha", "beta"] for i in range(6)]
    )
    for p in passages:
        p.metadata["zotero_tags"] = "MyWork; " + p.text.split()[0]
    options = dict(max_k=3, n_representatives=3, min_cluster_size=2)
    if mode == "joint":
        joint = cluster_joint(passages, [], anchored_embedder, **options)
        summary, _ = corpus_views(joint)
    else:
        summary = cluster_passages(passages, anchored_embedder, label="researcher", **options)
    assert summary.used
    for entry in summary.clusters:
        for text, tags in zip(entry["representatives"], entry["rep_theme_tags"], strict=True):
            assert tags == [text.split()[0]]


def test_original_context_does_not_change_indexed_word_windows(tmp_config, fake_embedder):
    from cda.corpus.ingest import chunk_words, sentence_context_records

    sentence = " ".join(["Novelty"] + ["token"] * 619) + "."
    chunks = chunk_words(sentence, 500, 100)
    contexts = sentence_context_records(sentence, 500, 100, len(chunks))
    assert [len(c.split()) for c in chunks] == [500, 220]
    assert contexts == [(0, sentence, [len(sentence)])] * 2


def test_legacy_records_upgrade_once(tmp_config, fake_embedder):
    from cda.corpus.ingest import _add_chunks

    collection = get_collection(tmp_config, create=True)
    _add_chunks(
        collection,
        fake_embedder,
        "old",
        ["Novelty."],
        {"pdf_relpath": "old.abstract", "pdf_filename": "old.abstract", "total_chunks": 1},
    )
    record = {"source": "old.abstract", "abstract": "Novelty."}
    assert ingest_records(tmp_config, fake_embedder, [record])["records_ingested"] == 1
    assert ingest_records(tmp_config, fake_embedder, [record])["records_ingested"] == 0
    stored = collection.get(include=["metadatas"])["metadatas"]
    assert len(stored) == 1 and stored[0]["sentence_context"] == "Novelty."


def test_pipeline_retrieval_failure_never_calls_model(tmp_config, fake_embedder, monkeypatch):
    from cda.pipeline import analyze_term

    class Provider:
        def complete(self, **kwargs):
            pytest.fail("An incomplete retrieval must never reach the model")

    class Collection:
        def get(self, **kwargs):
            if kwargs["offset"]:
                raise OSError("database offline")
            return {
                "ids": [str(i) for i in range(1000)],
                "documents": ["Novelty."] * 1000,
                "metadatas": [{}] * 1000,
            }

    monkeypatch.setattr("cda.pipeline.get_provider", lambda *a, **kw: Provider())
    with pytest.raises(ConfigError, match="incomplete"):
        analyze_term("novelty", tmp_config, collection=Collection(), embedder=fake_embedder)
    assert not list(tmp_config.path("paths.output_dir").glob("*"))


def test_whole_sentence_context_does_not_merge_equal_basenames():
    hits = {
        name: {
            "document": "Novelty.",
            "metadata": {
                "pdf_filename": "fulltext.pdf",
                "pdf_relpath": name,
                "chunk_index": 0,
                "sentence_context": "Novelty.",
                "sentence_start": 0,
                "sentence_lengths": "[8]",
                "sentence_segmenter": SENTENCE_SEGMENTER,
            },
        }
        for name in ["a/fulltext.pdf", "b/fulltext.pdf"]
    }
    _, literature, _ = build_passages("novelty", hits, mywork_classifier({}), load_config())
    assert len(literature) == 2


def test_grounding_reads_tagged_sample_without_treating_tags_as_evidence():
    from cda.registers.grounding import check_quotations

    prompt = 'Sample 1 [top1] [a.pdf]: "Novelty matters."\nTheme tags for Sample 1 (source metadata): science'
    assert check_quotations('"Novelty matters." - *a.pdf*', prompt)["n_verified"] == 1
    assert check_quotations('"science" - *a.pdf*', prompt)["n_verified"] == 0
