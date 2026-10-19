"""run_validation end to end (offline): real store, stub reader, session out."""

import json
from pathlib import Path

from cda.config import load_config
from cda.corpus.ingest import get_collection, ingest_records
from cda.validation.run import run_validation
from cda.validation.session import load_session


class OracleReader:
    """Always-valid reader replies, keyed by nothing (constant judgment)."""

    model = "claude-sonnet-4-6"

    def __init__(self, judgment="coherent"):
        self.judgment = judgment
        self.calls = 0

    def complete(self, *, system, user, max_tokens, temperature):
        self.calls += 1
        return json.dumps(
            {"gloss": "one usage", "judgment": self.judgment, "rationale": "consistent"}
        )


def build_store(tmp_path, fake_embedder):
    config = load_config(
        overrides={
            "paths": {
                "corpus_dir": str(tmp_path / "corpus"),
                "drafts_dir": str(tmp_path / "drafts"),
                "output_dir": str(tmp_path / "outputs"),
            },
            "clustering": {"min_cluster_size": 2, "literature_max_k": 4},
        }
    )
    records = [
        {
            "source": f"2020_{topic}_{i}.abstract",
            "abstract": f"An {topic} reading of novelty, case {i}. "
            f"Novelty recurs in the {topic} literature, variant {i}.",
        }
        for topic in ("alpha", "beta")
        for i in range(6)
    ]
    ingest_records(config, fake_embedder, records, progress=lambda _m: None)
    return config


def test_run_validation_creates_session_and_report(tmp_path, anchored_embedder):
    config = build_store(tmp_path, anchored_embedder)
    reader = OracleReader()
    result = run_validation(
        "novelty",
        config,
        embedder=anchored_embedder,
        collection=get_collection(config),
        providers={"claude-sonnet-4-6": reader},
        progress=lambda _m: None,
    )
    assert result["status"] == "ok"
    session_dir = Path(result["session_dir"])
    assert session_dir.name.startswith("validation_novelty_")
    session = load_session(session_dir)

    # Literature side clustered and validated; no drafts -> researcher side skipped.
    assert session["sides"]["literature"]["validated"] is True
    assert session["sides"]["researcher"]["validated"] is False
    partition = session["partitions"][0]
    assert partition["side"] == "literature" and partition["revision"] == 0
    n_clusters = len(partition["clusters"])
    assert n_clusters >= 2
    # One reading per cluster from the single reader; all parsed as valid.
    assert reader.calls == n_clusters
    assert len(session["readings"]) == n_clusters
    assert all(r["valid"] for r in session["readings"])
    # No verdicts yet; the report exists and states the protocol.
    assert session["verdicts"] == []
    # The run's embedding model is part of the record: the report is
    # re-rendered for the session's whole life, and the configured model can
    # change in between, so the run parameters must not read current config.
    assert session["params"]["embedding_model"] == str(config.get("embedding.model"))
    report = Path(result["report_file"]).read_text(encoding="utf-8")
    assert "Clustering Validation" in report
    assert "BLIND" in report
    assert "Session open: not yet endorsed." in report
    assert "—" not in report  # boilerplate convention: no em-dashes


def test_run_validation_shown_passages_capped_and_blind(tmp_path, anchored_embedder):
    config = build_store(tmp_path, anchored_embedder)
    captured = []

    class CapturingReader(OracleReader):
        def complete(self, *, system, user, max_tokens, temperature):
            captured.append(user)
            return super().complete(
                system=system, user=user, max_tokens=max_tokens, temperature=temperature
            )

    run_validation(
        "novelty",
        config,
        embedder=anchored_embedder,
        collection=get_collection(config),
        providers={"claude-sonnet-4-6": CapturingReader()},
        progress=lambda _m: None,
    )
    n_shown = int(config.get("validation.passages_shown_per_cluster"))
    for user in captured:
        # Numbered passages only, capped, and blind to labels/sources.
        first_line_counts = user.count("\n1. ")
        assert first_line_counts == 1
        assert f"\n{n_shown + 1}. " not in user
        assert ".abstract" not in user and "Cluster" not in user


def test_run_validation_errors_when_nothing_to_validate(tmp_path, fake_embedder):
    config = load_config(
        overrides={
            "paths": {
                "corpus_dir": str(tmp_path / "corpus"),
                "drafts_dir": str(tmp_path / "drafts"),
                "output_dir": str(tmp_path / "outputs"),
            }
        }
    )
    ingest_records(
        config,
        fake_embedder,
        [{"source": "one.abstract", "abstract": "novelty appears exactly once here"}],
        progress=lambda _m: None,
    )
    import pytest

    with pytest.raises(ValueError, match="Neither corpus side"):
        run_validation(
            "novelty",
            config,
            embedder=fake_embedder,
            collection=get_collection(config),
            providers={"claude-sonnet-4-6": OracleReader()},
            progress=lambda _m: None,
        )


def test_live_validate_fails_on_a_missing_key_before_any_heavy_stage(tmp_path, monkeypatch):
    """A missing ANTHROPIC_API_KEY must surface before extraction and
    clustering, matching analyze_term's up-front provider check. No corpus
    store exists under this config and no embedder is passed: if provider
    construction happened after the heavy stages, this test would die on the
    missing store (or try to download the real embedding model) instead."""
    import pytest

    from cda.llm.base import LLMError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY is not set"):
        run_validation("novelty", config, mock=False, progress=lambda _m: None)


def test_live_validate_fails_on_a_misconfigured_second_reader_up_front(tmp_path, monkeypatch):
    """Reader 2+'s provider is constructed before extraction too: an unset
    OPENAI_API_KEY for an OpenAI-compatible second reader used to surface only
    inside read_clusters, after reader 1's calls had already been paid for."""
    import pytest

    from cda.llm.base import LLMError

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = load_config(
        overrides={
            "paths": {"corpus_dir": str(tmp_path / "corpus")},
            "validation": {"readers": ["claude-sonnet-4-6", "openai:gpt-4o"]},
        }
    )
    with pytest.raises(LLMError, match="OPENAI_API_KEY is not set"):
        run_validation("novelty", config, mock=False, progress=lambda _m: None)


def test_mock_validate_fails_on_a_missing_fixtures_dir_before_any_heavy_stage(tmp_path):
    """Mock mode takes the same early path: no fixtures directory is refused
    before the store is even opened (none exists under this config)."""
    import pytest

    from cda.llm.base import LLMError

    config = load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})
    with pytest.raises(LLMError, match="fixtures directory"):
        run_validation("novelty", config, mock=True, fixtures_dir=None, progress=lambda _m: None)


def test_shown_by_cluster_refuses_unusable_embeddings_by_name():
    """A NaN row does not raise on its own here — it poisons the sort keys, so
    the researcher and the reader agent would see an arbitrary set of passages."""
    import pytest

    from cda.validation.run import _shown_by_cluster
    from tests.conftest import make_passages

    passages = make_passages(["a", "b", "c"])
    for p in passages:
        p.cluster_id = "Cluster 0"
        p.embedding = [1.0, 0.0, 0.0, 0.0]
    passages[1].embedding = [float("nan")] * 4
    with pytest.raises(ValueError, match=r"paper\.pdf::w1 \(cluster 0\)"):
        _shown_by_cluster(passages, 2)


def test_shown_by_cluster_refuses_zero_vector_by_name():
    import pytest

    from cda.validation.run import _shown_by_cluster
    from tests.conftest import make_passages

    passages = make_passages(["a", "b", "c"])
    for p in passages:
        p.cluster_id = "Cluster 3"
        p.embedding = [1.0, 0.0, 0.0, 0.0]
    passages[2].embedding = [0.0] * 4
    with pytest.raises(ValueError, match=r"paper\.pdf::w2 \(cluster 3\)"):
        _shown_by_cluster(passages, 2)
