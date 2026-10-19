"""Regression cases for input validation and report integrity; all inputs are synthetic."""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from cda.clustering.spherical import (
    adjusted_rand_index,
    hydrate_embeddings,
    prepare_matrix,
    silhouette,
)
from cda.corpus.ingest import get_collection, ingest_records
from cda.divergence.report import write_outputs
from cda.extraction.draft_passages import extract_draft_passages
from cda.extraction.mywork import mywork_classifier
from cda.extraction.retrieval import build_passages, keyword_hits
from tests.conftest import make_passages


@pytest.mark.parametrize("count", [1, 3])
def test_embedding_cardinality_must_match(count):
    passages = make_passages(["Novelty one.", "Novelty two."])
    embedder = SimpleNamespace(embed=lambda _: [[1.0, 0.0]] * count)
    with pytest.raises(ValueError, match="Incomplete embeddings"):
        hydrate_embeddings(passages, embedder)
    assert all(p.embedding is None for p in passages)


@pytest.mark.parametrize("vectors", [[None, [1, 0]], [[1, 0], [1]], [[], []]])
def test_matrix_never_silently_drops_or_accepts_invalid_vectors(vectors):
    passages = make_passages(["One.", "Two."])
    for p, vector in zip(passages, vectors, strict=True):
        p.embedding = vector
    with pytest.raises(ValueError):
        prepare_matrix(passages)


def test_identical_vectors_have_zero_silhouette():
    X = np.ones((4, 3)) / np.sqrt(3)
    assert silhouette(X, np.array([0, 0, 1, 1])) == 0.0


def test_ari_single_cluster_partitions_agree():
    assert adjusted_rand_index([0, 0, 0], [1, 1, 1]) == 1.0


def test_ari_does_not_fall_back_when_dependency_missing(monkeypatch):
    import builtins

    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "sklearn.metrics":
            raise ImportError("test missing dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(ImportError):
        adjusted_rand_index([0, 0], [1, 1])


def test_overlapping_chunks_preserve_distinct_repeated_sentences(tmp_config, fake_embedder):
    text = "Novelty matters. " + "Other words. " * 220 + "Novelty matters. " + "End words. " * 100
    ingest_records(tmp_config, fake_embedder, [{"source": "toy", "abstract": text}])
    hits = keyword_hits(get_collection(tmp_config), "novelty")
    _, passages, stats = build_passages("novelty", hits, mywork_classifier({}), tmp_config)
    assert [p.text for p in passages] == ["Novelty matters."] * 2
    assert len({p.metadata["sentence_index"] for p in passages}) == 2
    assert stats["overlap_occurrences_removed"] >= 1


def test_drafts_keep_repetitions_within_and_between_sources(tmp_config, fake_embedder):
    passages = extract_draft_passages(
        "novelty", {"a": "Novelty. Novelty.", "b": "Novelty."}, fake_embedder, tmp_config
    )
    assert len(passages) == 3


@pytest.mark.parametrize(
    "field", ["prompt_text", "report_text", "system_prompt_text", "diagnostics"]
)
def test_run_base_reuse_rejects_changes_without_touching_any_file(tmp_path, field):
    args = dict(
        out_dir=tmp_path,
        term="novelty",
        now=datetime(2026, 1, 1),
        base="same",
        prompt_text="prompt1",
        report_text="report1",
        system_prompt_text="system1",
        diagnostics={"n": 1},
    )
    write_outputs(**args)
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    args[field] = {"n": 2} if field == "diagnostics" else "changed"
    with pytest.raises(FileExistsError):
        write_outputs(**args)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == original


def test_two_phase_output_writes_are_idempotent(tmp_path):
    args = dict(
        out_dir=tmp_path,
        term="novelty",
        now=datetime(2026, 1, 1),
        base="same",
        prompt_text="prompt",
        diagnostics={},
    )
    write_outputs(**args, report_text=None)
    write_outputs(**args, report_text="report")
    write_outputs(**args, report_text="report")


def test_missing_term_anchor_never_switches_ranking_silently():
    from cda.clustering.saturation import rank_literature

    passages = make_passages(["One.", "Two."])
    for p in passages:
        p.embedding = [1.0, 0.0]
    with pytest.raises(ValueError, match="Term anchor"):
        rank_literature(passages, "novelty", SimpleNamespace(embed=lambda _: []))
