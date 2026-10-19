"""Demo plumbing and OpenAlex utilities (fully offline)."""

import json
from pathlib import Path

import pytest

from cda.demo import demo_config, find_demo_dir, load_toy_records
from cda.diagnose.openalex import OpenAlexError, polite_mailto, reconstruct_abstract


def test_reconstruct_abstract_roundtrip():
    text = "novelty is ratified by audiences repairing their expectations"
    inverted: dict[str, list[int]] = {}
    for position, word in enumerate(text.split()):
        inverted.setdefault(word, []).append(position)
    assert reconstruct_abstract(inverted) == text


def test_reconstruct_abstract_handles_empty():
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


def test_polite_mailto_required(monkeypatch):
    monkeypatch.delenv("OPENALEX_MAILTO", raising=False)
    with pytest.raises(OpenAlexError, match="OPENALEX_MAILTO"):
        polite_mailto()
    monkeypatch.setenv("OPENALEX_MAILTO", "someone@example.org")
    assert polite_mailto() == "someone@example.org"


def _make_demo_dir(tmp_path, n_records=3):
    toy = tmp_path / "demo" / "toy_corpus"
    toy.mkdir(parents=True)
    with (toy / "abstracts.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(n_records):
            handle.write(
                json.dumps({"source": f"2020_paper_{i}.abstract", "abstract": f"text {i}"}) + "\n"
            )
    (toy / "drafts").mkdir()
    return tmp_path / "demo"


def test_find_demo_dir_explicit_and_missing(tmp_path):
    demo_dir = _make_demo_dir(tmp_path)
    assert find_demo_dir(demo_dir) == demo_dir.resolve()
    with pytest.raises(FileNotFoundError, match="root of the cloned repository"):
        find_demo_dir(tmp_path / "elsewhere")


def test_demo_config_points_into_runtime(tmp_path):
    demo_dir = _make_demo_dir(tmp_path)
    config = demo_config(demo_dir)
    assert str(config.path("paths.corpus_dir")).endswith("_runtime/corpus")
    assert str(config.path("paths.drafts_dir")).endswith("toy_corpus/drafts")
    # Paper defaults untouched — the demo runs the real pipeline.
    assert config.get("saturation.max_passages") == 1200
    assert config.get("llm.model") == "claude-sonnet-4-6"


def test_demo_config_drafts_override_for_own_text_mode(tmp_path):
    """The paste-your-own-text mode swaps only the researcher side."""
    demo_dir = _make_demo_dir(tmp_path)
    own = tmp_path / "your_text_drafts"
    config = demo_config(demo_dir, drafts_dir=own)
    assert config.path("paths.drafts_dir") == own
    # Literature side and parameters are untouched.
    assert str(config.path("paths.corpus_dir")).endswith("_runtime/corpus")
    assert config.get("saturation.max_passages") == 1200


def test_load_toy_records(tmp_path):
    demo_dir = _make_demo_dir(tmp_path, n_records=2)
    records = load_toy_records(demo_dir)
    assert len(records) == 2
    assert records[0]["source"] == "2020_paper_0.abstract"


def test_partial_first_ingest_is_completed_on_next_run(tmp_path, fake_embedder):
    """A first demo run interrupted mid-indexing leaves a partial store; the
    next run must complete it (per-record idempotent ingest), not skip it —
    a partial corpus assembles a different prompt and the fixture replay
    would fail forever with a misleading message."""
    from cda.corpus.ingest import get_collection, ingest_records
    from cda.demo import ensure_toy_corpus_ingested

    demo_dir = _make_demo_dir(tmp_path, n_records=3)
    config = demo_config(demo_dir)
    records = load_toy_records(demo_dir)
    ingest_records(config, fake_embedder, records[:1], progress=lambda _msg: None)  # "interrupt"
    ensure_toy_corpus_ingested(config, demo_dir, fake_embedder)
    stored = get_collection(config).get(include=["metadatas"])["metadatas"]
    assert {m["pdf_relpath"] for m in stored} == {r["source"] for r in records}


def test_ingest_records_copies_date_added_without_touching_identity(tmp_path, fake_embedder):
    """The date_added whitelist entry is additive: chunk ids and documents
    must be byte-identical with and without the field — the recorded demo
    fixtures depend on that identity."""
    from cda.corpus.ingest import get_collection, ingest_records
    from cda.demo import demo_config

    record = {"source": "2020_paper_0.abstract", "abstract": "novelty appears here"}
    stores = {}
    for label, extra in (("without", {}), ("with", {"date_added": "2020-05-04 09:00:00"})):
        config = demo_config(_make_demo_dir(tmp_path / label))
        ingest_records(config, fake_embedder, [dict(record, **extra)], progress=lambda _m: None)
        stores[label] = get_collection(config).get(include=["metadatas", "documents"])

    assert stores["with"]["ids"] == stores["without"]["ids"]
    assert stores["with"]["documents"] == stores["without"]["documents"]
    assert stores["with"]["metadatas"][0]["date_added"] == "2020-05-04 09:00:00"
    assert "date_added" not in stores["without"]["metadatas"][0]


def test_toy_corpus_records_have_deterministic_date_added():
    """Every committed record carries the synthetic acquisition date, exactly
    as assign_demo_date_added would recompute it from committed fields."""
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "demo"))
    try:
        from build_toy_corpus import assign_demo_date_added
    finally:
        sys.path.pop(0)

    with (repo_root / "demo" / "toy_corpus" / "abstracts.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    committed = {r["source"]: r["date_added"] for r in records}

    recomputed = [dict(r) for r in records]
    assign_demo_date_added(recomputed)
    assert {r["source"]: r["date_added"] for r in recomputed} == committed


def test_repo_demo_drafts_are_marked_synthetic():
    """Every committed demo draft must declare itself fictional."""
    from pathlib import Path

    drafts_dir = Path(__file__).resolve().parents[1] / "demo" / "toy_corpus" / "drafts"
    draft_files = sorted(drafts_dir.glob("*.md"))
    assert len(draft_files) >= 3
    for path in draft_files:
        head = path.read_text(encoding="utf-8")[:400]
        assert "SYNTHETIC DOCUMENT" in head, f"{path.name} lacks the synthetic header"
        body = path.read_text(encoding="utf-8").lower()
        assert "novelty" in body or "consensus" in body


@pytest.mark.integration
def test_validate_demo_runs_offline_from_committed_fixtures(tmp_path):
    """The repo's rule: every command works under --mock-llm from a fresh clone.

    The synthetic reading fixtures went missing once while two documents
    described them as present, and nothing failed until someone tried the
    command. Marked integration because it needs the embedding model.
    """
    from cda.corpus.embeddings import get_embedder
    from cda.demo import demo_config, ensure_toy_corpus_ingested, find_demo_dir
    from cda.validation.run import run_validation

    demo_dir = find_demo_dir()
    # The dedicated parameter, not extra_overrides={"paths": ...} — that key
    # is refused so a session can never silently land in the shared
    # demo/_runtime/outputs the GUI sweeps and resumes from.
    config = demo_config(demo_dir, output_dir=str(tmp_path))
    embedder = get_embedder(config)
    ensure_toy_corpus_ingested(config, demo_dir, embedder)

    result = run_validation(
        "novelty",
        config,
        mock=True,
        fixtures_dir=demo_dir / "fixtures" / "validation",
        embedder=embedder,
        progress=lambda _m: None,
    )
    assert result["status"] == "ok"
    report = Path(result["report_file"]).read_text(encoding="utf-8")
    # Blinding: no reading is shown before a verdict is recorded.
    assert "withheld until you record a verdict" in report
