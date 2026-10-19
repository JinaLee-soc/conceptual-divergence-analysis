"""Run integrity and reproducibility boundaries, using synthetic inputs only."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cda.corpus.embeddings import SentenceTransformerEmbedder
from cda.corpus.ingest import ingest_records
from cda.pipeline import analyze_term
from cda.provenance import compare_runs, digest, input_snapshot, sample_snapshot, verify_receipt
from tests.conftest import make_passages


@pytest.fixture
def ready(tmp_config, fake_embedder):
    ingest_records(
        tmp_config,
        fake_embedder,
        [{"source": "toy", "abstract": "Novelty has one meaning. Novelty has another meaning."}],
        progress=lambda _: None,
    )
    return dict(
        term="novelty",
        config=tmp_config,
        embedder=fake_embedder,
        draft_texts={"draft": "Novelty matters. Novelty varies."},
        progress=lambda _: None,
    )


def test_two_runs_record_identical_inputs_and_samples(ready):
    runs = [analyze_term(**ready, dry_run=True) for _ in range(2)]
    manifests = [json.loads(Path(run["manifest_file"]).read_text()) for run in runs]
    assert runs[0]["base"] != runs[1]["base"]
    assert (
        compare_runs(Path(runs[0]["receipt_file"]), Path(runs[1]["receipt_file"]))["status"]
        == "exact agreement"
    )
    for key in [
        "inputs",
        "extracted_sample",
        "analyzed_sample",
        "config_sha256",
        "code",
        "environment",
    ]:
        assert manifests[0][key] == manifests[1][key]
    assert manifests[0]["environment"]["packages"]
    assert manifests[0]["code"]["package_files"]["pipeline.py"]
    assert manifests[0]["embedding"]["status"].startswith("custom")
    for run in runs:
        assert verify_receipt(Path(run["receipt_file"]))["status"] == "verified"


def test_complete_mock_run_and_tamper_detection(ready, monkeypatch):
    provider = SimpleNamespace(
        name="synthetic", model="test", complete=lambda **_: "Synthetic result."
    )
    monkeypatch.setattr("cda.pipeline.get_provider", lambda *a, **kw: provider)
    run = analyze_term(**ready, mock=True)
    receipt_path = Path(run["receipt_file"])
    assert json.loads(receipt_path.read_text())["status"] == "complete"
    assert verify_receipt(receipt_path)["status"] == "verified"
    Path(run["prompt_file"]).write_text("changed")
    with pytest.raises(ValueError, match="changed run artifact"):
        verify_receipt(receipt_path)


def test_model_failure_leaves_prepared_manifest_without_completion(ready, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("synthetic provider failure")

    monkeypatch.setattr(
        "cda.pipeline.get_provider",
        lambda *a, **kw: SimpleNamespace(name="test", model="test", complete=fail),
    )
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        analyze_term(**ready, mock=True)
    output = ready["config"].path("paths.output_dir")
    assert len(list(output.glob("*_manifest.json"))) == 1
    assert not list(output.glob("*_receipt.json"))


def test_input_and_sample_digests_detect_changes():
    before = input_snapshot({"a": {"document": "Novelty."}}, {"draft": "Novelty."})
    after = input_snapshot({"a": {"document": "Changed."}}, {"draft": "Novelty."})
    assert before["sha256"] != after["sha256"]
    passages = make_passages(["Novelty.", "Other novelty."])
    before = sample_snapshot(passages, [])
    passages.reverse()
    assert before["sha256"] != sample_snapshot(passages, [])["sha256"]
    with pytest.raises(ValueError, match="Duplicate"):
        sample_snapshot([passages[0]] * 2, [])


def test_model_manifest_hashes_local_weights_and_revision(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"synthetic weights")
    embedder = SentenceTransformerEmbedder(str(tmp_path), revision="requested")

    # A tiny stand-in with SentenceTransformer's indexing/device interface.
    class Model(list):
        device = "cpu"

    embedder._model = Model(
        [
            SimpleNamespace(
                auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash="resolved"))
            )
        ]
    )
    result = embedder.provenance()
    assert result["resolved_revision"] == "resolved"
    assert result["requested_revision"] == "requested"
    assert result["files"]["weights.bin"]
    assert result["files_sha256"] == digest(result["files"])


def test_no_provenance_or_environment_write_for_public_preview(ready, monkeypatch):
    monkeypatch.setattr(
        "cda.pipeline.build_manifest",
        lambda **kwargs: pytest.fail("preview must not gather a manifest"),
    )
    result = analyze_term(**ready, dry_run=True, persist=False)
    assert result["status"] == "dry-run"
    assert not ready["config"].path("paths.output_dir").exists()


def test_reference_comparison_rejects_changed_config(ready):
    first = analyze_term(**ready, dry_run=True)
    ready["config"].data["clustering"]["seed"] += 1
    second = analyze_term(**ready, dry_run=True)
    with pytest.raises(ValueError, match="config_sha256"):
        compare_runs(Path(second["receipt_file"]), Path(first["receipt_file"]))
