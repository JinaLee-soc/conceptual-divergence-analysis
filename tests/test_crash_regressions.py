"""Regressions for crashes that aborted a run late or discarded a finished reply."""

import datetime as dt
import importlib.metadata
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from cda.config import ConfigError, load_config
from cda.corpus.ingest import ingest_records
from cda.corpus.timeframe import timeframe_from_config
from cda.llm.base import LLMError, TruncatedReplyError
from cda.pipeline import analyze_term
from cda.provenance import _environment_differences, digest, environment_snapshot

# --- unquoted YAML dates ------------------------------------------------


def _write_config(tmp_path: Path, timeframe_yaml: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""paths:
  corpus_dir: {tmp_path / "corpus"}
  drafts_dir: {tmp_path / "drafts"}
  output_dir: {tmp_path / "outputs"}
clustering:
  min_cluster_size: 2
  silhouette_subsample: 200
  researcher_max_k: 4
  literature_max_k: 5
saturation:
  start: 8
  step: 4
  max_passages: 60
timeframe:
{timeframe_yaml}
""",
        encoding="utf-8",
    )
    return path


def test_unquoted_yaml_dates_load_as_iso_text(tmp_path):
    path = _write_config(
        tmp_path,
        "  added_from: 2019-01-01\n  added_to: 2026-05-01 12:30:00+02:00\n",
    )
    config = load_config(path)
    assert config.get("timeframe.added_from") == "2019-01-01"
    # An aware datetime is converted to UTC rather than losing its zone.
    assert config.get("timeframe.added_to") == "2026-05-01 10:30:00"
    window = timeframe_from_config(config)
    assert window["added_from"] == "2019-01-01 00:00:00"
    assert window["added_to"] == "2026-05-01 10:30:00"


def test_date_objects_in_overrides_and_lists_become_text():
    config = load_config(
        overrides={
            "timeframe": {
                "added_from": dt.date(2000, 1, 1),
                "added_to": dt.datetime(2026, 5, 1, 8, 15, 0),
            },
            "themes": [{"tag": "t", "description": dt.date(2020, 2, 3)}],
        }
    )
    assert config.get("timeframe.added_from") == "2000-01-01"
    assert config.get("timeframe.added_to") == "2026-05-01 08:15:00"
    assert config.get("themes")[0]["description"] == "2020-02-03"
    digest(config.data)  # JSON-serializable throughout


def test_digest_names_the_key_of_a_non_json_value():
    with pytest.raises(ConfigError, match=r"timeframe\.added_to"):
        digest({"timeframe": {"added_to": dt.date(2026, 5, 1)}})
    with pytest.raises(ConfigError, match=r"themes\[1\]"):
        digest({"themes": ["ok", object()]})


def test_unquoted_date_config_completes_a_dry_run(tmp_path, fake_embedder):
    """Before the fix this crashed in the manifest after clustering."""
    config = load_config(
        _write_config(tmp_path, "  added_from: 2000-01-01\n  include_undated: true\n")
    )
    ingest_records(
        config,
        fake_embedder,
        [{"source": "toy", "abstract": "Novelty has one meaning. Novelty has another meaning."}],
        progress=lambda _: None,
    )
    run = analyze_term(
        term="novelty",
        config=config,
        embedder=fake_embedder,
        draft_texts={"draft": "Novelty matters. Novelty varies."},
        progress=lambda _: None,
        dry_run=True,
    )
    manifest = json.loads(Path(run["manifest_file"]).read_text(encoding="utf-8"))
    assert manifest["config"]["timeframe"]["added_from"] == "2000-01-01"


# --- environment inventory ----------------------------------------------


def _nameless_distribution(tmp_path: Path) -> importlib.metadata.Distribution:
    info = tmp_path / "broken-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nVersion: 1.0\n", encoding="utf-8")
    return importlib.metadata.Distribution.at(info)


def test_environment_snapshot_records_a_nameless_distribution(tmp_path, monkeypatch):
    real = list(importlib.metadata.distributions())
    broken = _nameless_distribution(tmp_path)
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [broken, *real])
    first = environment_snapshot()
    unnamed = [p for p in first["packages"] if p["name"].startswith("<unnamed:")]
    assert unnamed == [{"name": f"<unnamed:{broken._path}>", "version": "1.0"}]

    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [*reversed(real), broken])
    assert environment_snapshot()["packages"] == first["packages"]


def _manifest(packages: list[dict]) -> dict:
    return {
        "environment": {"python": "x", "platform": "p", "machine": "m", "packages": packages},
        "embedding": {"name": "e"},
    }


def _differences(current: list[dict], reference: list[dict]) -> dict:
    return _environment_differences([_manifest(current), _manifest(reference)])


def test_duplicate_distributions_are_not_collapsed():
    both = [{"name": "numpy", "version": "1.26"}, {"name": "numpy", "version": "2.0"}]
    for single in ("2.0", "1.26"):
        differences = _differences(both, [{"name": "numpy", "version": single}])
        assert differences == {
            "duplicate_packages": {"numpy": {"current": ["1.26", "2.0"], "reference": [single]}}
        }
    assert _differences(both, list(reversed(both))) == {}


def test_package_names_compare_in_pep503_form():
    assert (
        _differences(
            [{"name": "PyYAML", "version": "6"}, {"name": "Typing.Extensions", "version": "4"}],
            [{"name": "pyyaml", "version": "6"}, {"name": "typing_extensions", "version": "4"}],
        )
        == {}
    )
    assert _differences(
        [{"name": "PyYAML", "version": "6"}], [{"name": "pyyaml", "version": "5"}]
    ) == {"packages": {"pyyaml": {"current": "6", "reference": "5"}}}


# --- OpenAI-compatible stop reasons -------------------------------------


def _openai_reply(monkeypatch, finish_reason):
    from cda.llm.providers import OpenAICompatibleProvider

    reply = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason, message=SimpleNamespace(content=" Full reply. ")
            )
        ]
    )
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **_kw: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kw: reply))
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return OpenAICompatibleProvider(model="local-model").complete(
        system="s", user="u", max_tokens=100, temperature=0.0
    )


@pytest.mark.parametrize(
    "finish_reason", ["stop", None, "eos", "eos_token", "end_turn", "stop_sequence"]
)
def test_openai_normal_finish_reasons_keep_the_reply(monkeypatch, finish_reason):
    assert _openai_reply(monkeypatch, finish_reason) == "Full reply."


def test_openai_length_is_still_a_truncated_reply(monkeypatch):
    with pytest.raises(TruncatedReplyError):
        _openai_reply(monkeypatch, "length")


@pytest.mark.parametrize("finish_reason", ["content_filter", "tool_calls"])
def test_openai_other_finish_reasons_are_refused_by_name(monkeypatch, finish_reason):
    with pytest.raises(LLMError, match=finish_reason) as error:
        _openai_reply(monkeypatch, finish_reason)
    assert not isinstance(error.value, TruncatedReplyError)
