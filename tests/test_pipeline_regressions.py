"""Regression tests for pipeline configuration and output consistency."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cda.config import ConfigError, load_config
from cda.corpus.ingest import ingest_records
from cda.llm.base import LLMError, prompt_fingerprint
from cda.registers.audit import _classify_mapping_line, audit_report, literature_mapping_counts
from cda.registers.grounding import check_quotations, render_grounding_markdown
from cda.types import ClusterSummary, Passage

# The demo prompt line whose passage itself contains a quotation.
DEMO_PROMPT = (
    'Sample 3 [top3] [draft_01_novelty_as_audience_repair.md]: "What analysts of science '
    'call novelty, the operators I shadowed called "the paperwork of surprise." The '
    'distinction matters."\n'
    'Sample 1 [top1] [draft_02.md]: "If novelty is a repair accomplishment and consensus '
    'is a risk-redistribution device, both proxies point at the wrong scene."\n'
)


# ---- 1. quotation grounding ------------------------------------------------


def test_invented_span_between_inner_quote_marks_is_not_verified():
    """Left-to-right pairing used to verify the outer fragments and never
    look at the invented text between the inner marks."""
    body = (
        'The researcher writes "What analysts of science call novelty, the operators I '
        'shadowed called "an invented phrase that never appears anywhere in the corpus at all" here."'
    )
    result = check_quotations(body, DEMO_PROMPT)
    assert result["n_ambiguous"] >= 1
    assert result["n_verified"] < result["n_quotes"]
    assert any(item.startswith("[ambiguous nesting, not checked]") for item in result["unverified"])
    assert "ambiguous nesting" in render_grounding_markdown(result)


def test_legitimate_quote_of_a_passage_with_inner_quotes_still_verifies():
    curly = (
        "As the draft puts it, “What analysts of science call novelty, the operators I "
        'shadowed called "the paperwork of surprise." The distinction matters.”'
    )
    single = (
        'As the draft puts it, "What analysts of science call novelty, the operators I '
        "shadowed called 'the paperwork of surprise.' The distinction matters.\""
    )
    for body in (curly, single):
        result = check_quotations(body, DEMO_PROMPT)
        assert (result["n_quotes"], result["n_verified"], result["n_ambiguous"]) == (1, 1, 0)


def test_short_attributed_quote_matches_whole_words_only():
    novel = check_quotations('"novel" - draft_01_novelty_as_audience_repair.md', DEMO_PROMPT)
    novelty = check_quotations('"novelty" - draft_01_novelty_as_audience_repair.md', DEMO_PROMPT)
    assert novel["n_verified"] == 0 and novel["n_quotes"] == 1
    assert novelty["n_verified"] == 1


@pytest.mark.parametrize("elision", ["[...]", "[…]", "[ ... ]"])
def test_bracketed_elision_is_treated_like_an_ellipsis(elision):
    body = (
        f'"If novelty is a repair accomplishment {elision} both proxies point at the wrong scene."'
    )
    assert check_quotations(body, DEMO_PROMPT)["n_verified"] == 1


# ---- 2.-5. cluster-register audit ------------------------------------------

_TABLE = """### Cluster A: x
## 2. Researcher vs. Literature Divergence

| Register | k-means clusters absorbed | note |
|---|---|---|
| **L-R1** | Clusters 0 and 3 | |
| **L-R2** | clusters 1, 2 & 5 | |
| **L-R3** | C4 | |
| **L-R4** | none of them | |
| **L-R3** | C6 | |
"""


def test_table_accepts_plural_and_short_cluster_forms_and_notes_dropped_rows():
    audit = audit_report(_TABLE, k_researcher=1, k_literature=7)
    lit = audit["literature"]
    assert (lit["merge"], lit["one_to_one"], lit["split"]) == (2, 1, 0)
    assert any("'L-R4'" in note and "no parseable" in note for note in audit["notes"])
    assert any("'L-R3'" in note and "more than once" in note for note in audit["notes"])


@pytest.mark.parametrize(
    "line",
    [
        "- **k-means/LLM mapping:** no merge; Cluster 4 splits into L-R2 and L-R3",
        "- **k-means/LLM mapping:** not 1:1 - Cluster 4 is split across L-R2/L-R3",
        "- **k-means/LLM mapping:** without merging, cluster 2 is split in two",
    ],
)
def test_negation_stops_at_the_clause_boundary(line):
    assert _classify_mapping_line(line) == "split"


def test_negation_within_the_clause_still_applies():
    line = "- **k-means/LLM mapping:** does not merge or split; it maps 1:1 onto cluster 0."
    assert _classify_mapping_line(line) == "one_to_one"


_V5_REPORT = """## 1. Polysemy Map
### Cluster A: repair
- **k-means/LLM mapping:** 1:1 with embedding cluster 0
### Cluster B: rationing
- **k-means/LLM mapping:** 1:1 with embedding cluster 1

## 2. Researcher vs. Literature Divergence
### Literature Register I: originality
- **k-means/LLM mapping:** 1:1 with embedding cluster 0
### Literature Register II: quality
- **k-means/LLM mapping:** 1:1 with embedding cluster 1

k-means/LLM divergence rate: researcher {researcher}; literature {literature}.

## 3. Actionable Recommendations
"""


def test_stated_rates_in_the_v5_format_are_cross_checked():
    agree = audit_report(
        _V5_REPORT.format(researcher="0/2 (two 1:1)", literature="0/2 (two 1:1)"), 2, 2
    )
    assert agree["notes"] == []
    disagree = audit_report(
        _V5_REPORT.format(researcher="50%", literature="2/2 (two splits)"), 2, 2
    )
    assert any("literature rate 0.0 disagrees" in n and "1.0" in n for n in disagree["notes"])
    assert any("researcher rate 0.0 disagrees" in n and "0.5" in n for n in disagree["notes"])


def test_literature_bullets_are_never_counted_as_researcher_declarations():
    report = """## 1. Polysemy Map
### Cluster 0: repair
- **k-means/LLM mapping:** 1:1 with Cluster 0
### Cluster 1: rationing
- **k-means/LLM mapping:** 1:1 with Cluster 1

## 2. Divergence
- **L-R1 mapping:** merges clusters 0 and 2
- **L-R2 mapping:** 1:1 with cluster 1
"""
    audit = audit_report(report, 2, 3)
    assert audit["researcher"]["one_to_one"] == 2
    assert audit["researcher"]["merge"] == 0
    assert audit["researcher"]["divergence_rate"] == 0.0
    assert literature_mapping_counts(report) == {"one_to_one": 1, "merge": 1, "split": 0}


def test_zero_researcher_registers_is_reported_as_zero():
    audit = audit_report("No headings at all.", 2, 2)
    assert audit["researcher"]["llm_registers"] == 0


# ---- 6. saturation note and parameters -------------------------------------


def _literature(n: int) -> list[Passage]:
    return [
        Passage(chunk_id=f"c{i}", source="a.pdf", year="2020", text="x novelty", similarity=0.5)
        for i in range(n)
    ]


def test_disabled_saturation_states_the_true_literature_count():
    from cda.pipeline import select_literature_by_saturation
    from cda.registers.assemble import _saturation_note

    config = load_config(overrides={"saturation": {"enabled": False}})
    _, info, ran = select_literature_by_saturation(
        "novelty", _literature(56), config, embedder=None
    )
    assert not ran
    assert (info["n_candidates"], info["n_selected"]) == (56, 56)
    assert "all 56 retrieved literature passages" in _saturation_note(info)


def test_saturation_selection_uses_the_validated_parameters():
    from cda.pipeline import select_literature_by_saturation

    config = load_config(
        overrides={
            "saturation": {"enabled": True, "scope": "sample", "start": 50, "max_passages": 10}
        }
    )
    with pytest.raises(ConfigError, match="max_passages"):
        select_literature_by_saturation("novelty", _literature(60), config, embedder=None)


def test_saturation_ranking_is_normalized_before_entering_the_prompt():
    from cda.registers.assemble import _saturation_note

    note = _saturation_note(
        {
            "n_candidates": 100,
            "n_selected": 50,
            "ranking": " Term ",
            "saturation_reached": True,
            "ari_threshold": 0.8,
            "consecutive_increments": 2,
        }
    )
    assert "external anchor" in note and " Term " not in note


# ---- 7. run comparison -----------------------------------------------------


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


def test_compare_runs_ignores_output_location(ready, tmp_path):
    from cda.pipeline import analyze_term
    from cda.provenance import compare_runs

    first = analyze_term(**ready, dry_run=True)
    ready["config"].data["paths"]["output_dir"] = str(tmp_path / "elsewhere")
    second = analyze_term(**ready, dry_run=True)
    result = compare_runs(Path(second["receipt_file"]), Path(first["receipt_file"]))
    assert result["status"] == "exact agreement"


def test_machine_differences_are_reported_and_fail_only_when_strict(ready, monkeypatch):
    import cda.provenance as provenance
    from cda.pipeline import analyze_term

    first = analyze_term(**ready, dry_run=True)
    real = provenance.environment_snapshot
    monkeypatch.setattr(
        provenance, "environment_snapshot", lambda: {**real(), "platform": "other-os-1.0"}
    )
    second = analyze_term(**ready, dry_run=True)
    current, reference = Path(second["receipt_file"]), Path(first["receipt_file"])

    result = provenance.compare_runs(current, reference)
    assert result["environment_differences"]["platform"]["current"] == "other-os-1.0"
    with pytest.raises(ValueError, match="environment.platform"):
        provenance.compare_runs(current, reference, strict_environment=True)

    command = [sys.executable, "-m", "cda.provenance", str(current), "--compare", str(reference)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    strict = subprocess.run([*command, "--strict-environment"], capture_output=True, text=True)
    assert strict.returncode != 0 and "environment.platform" in strict.stderr


def test_manifest_records_the_configured_embedding_revision(ready):
    from cda.pipeline import analyze_term
    from cda.provenance import digest

    ready["config"].data["embedding"]["revision"] = "abc123"
    run = analyze_term(**ready, dry_run=True)
    manifest = json.loads(Path(run["manifest_file"]).read_text())
    assert manifest["embedding"]["configured_revision"] == "abc123"
    analysis = {key: value for key, value in manifest["config"].items() if key != "paths"}
    assert manifest["config_sha256"] == digest(analysis)


# ---- 8. figure -------------------------------------------------------------


def _figure_passages(side: str, cluster: int, n: int, offset: float) -> list[Passage]:
    return [
        Passage(
            chunk_id=f"{side}{cluster}-{i}",
            source="s",
            year="2020",
            text="t",
            similarity=0.0,
            cluster_id=f"Cluster {cluster}",
            embedding=[offset + 0.01 * i, float(cluster), 0.1 * i],
        )
        for i in range(n)
    ]


def _draw(monkeypatch, tmp_path, joint, name="map.png"):
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    from cda.divergence import figure

    labels, colours = [], []
    real_annotate, real_scatter = Axes.annotate, Axes.scatter

    def annotate(self, text, *args, **kwargs):
        labels.append(text)
        return real_annotate(self, text, *args, **kwargs)

    def scatter(self, *args, **kwargs):
        colours.append(list(kwargs["c"]))
        return real_scatter(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "annotate", annotate)
    monkeypatch.setattr(Axes, "scatter", scatter)
    figure.draw_passage_map(
        term="novelty",
        researcher_passages=_figure_passages("r", 0, 3, 1.0),
        literature_passages=_figure_passages("l", 0, 3, -1.0) + _figure_passages("l", 1, 2, 0.0),
        joint=joint,
        path=tmp_path / name,
        dpi=40,
    )
    return labels, colours


def test_independent_figure_never_shares_colours_or_labels_across_corpora(monkeypatch, tmp_path):
    labels, (literature_colours, researcher_colours) = _draw(monkeypatch, tmp_path, None)
    assert labels == ["R0", "L0", "L1"]
    assert not set(literature_colours) & set(researcher_colours)


def test_joint_figure_keeps_shared_labels_and_colours(monkeypatch, tmp_path):
    joint = SimpleNamespace(used=True)
    labels, (literature_colours, researcher_colours) = _draw(monkeypatch, tmp_path, joint)
    assert labels == ["C0 (3R)", "C1"]
    assert researcher_colours[0] == literature_colours[0]


def test_figure_refuses_to_overwrite_and_rejects_unknown_cluster_ids(monkeypatch, tmp_path):
    from cda.divergence.figure import _cluster_index

    _draw(monkeypatch, tmp_path, None)
    with pytest.raises(FileExistsError):
        _draw(monkeypatch, tmp_path, None)
    garbage = Passage(chunk_id="x", source="s", year="", text="", similarity=0, cluster_id="?")
    with pytest.raises(ValueError, match="unrecognized cluster id"):
        _cluster_index(garbage)


# ---- 9. raw reply persisted before post-processing ----------------------------


def test_llm_reply_is_saved_before_post_processing_and_sealed(ready, monkeypatch):
    from cda.pipeline import analyze_term
    from cda.provenance import verify_receipt

    reply = "### Cluster A: synthetic\nSynthetic reply."
    provider = SimpleNamespace(name="synthetic", model="test", complete=lambda **_: reply)
    monkeypatch.setattr("cda.pipeline.get_provider", lambda *a, **kw: provider)

    def fail(*_args, **_kwargs):
        raise RuntimeError("post-processing failed")

    monkeypatch.setattr("cda.pipeline.audit_report", fail)
    with pytest.raises(RuntimeError, match="post-processing failed"):
        analyze_term(**ready, mock=True)
    output = ready["config"].path("paths.output_dir")
    saved = list(output.glob("*_llm_response.txt"))
    assert len(saved) == 1 and saved[0].read_text(encoding="utf-8") == reply

    monkeypatch.undo()
    monkeypatch.setattr("cda.pipeline.get_provider", lambda *a, **kw: provider)
    run = analyze_term(**ready, mock=True)
    receipt = json.loads(Path(run["receipt_file"]).read_text())
    assert "llm_response_file" in receipt["artifacts"]
    assert verify_receipt(Path(run["receipt_file"]))["status"] == "verified"


# ---- 10. providers and verify ----------------------------------------------


def test_anthropic_refusal_stop_reason_is_rejected(monkeypatch):
    from cda.llm.providers import AnthropicProvider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider(model="claude-sonnet-4-6")
    provider._client.messages = SimpleNamespace(
        create=lambda **_kw: SimpleNamespace(
            stop_reason="refusal", content=[SimpleNamespace(text="I can't help with that.")]
        )
    )
    with pytest.raises(LLMError, match="refusal"):
        provider.complete(system="s", user="u", max_tokens=5, temperature=0.0)


def test_fixture_fingerprint_keeps_non_anthropic_prefixes_only():
    from cda.llm.providers import fixture_model

    assert fixture_model("openai:gpt-4o") == "openai:gpt-4o"
    assert fixture_model("claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert fixture_model("anthropic:claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert prompt_fingerprint(fixture_model("openai:claude-x"), "s", "u") != prompt_fingerprint(
        fixture_model("claude-x"), "s", "u"
    )


def test_openai_spec_does_not_replay_an_anthropic_fixture(tmp_path):
    from cda.llm.base import FixtureMissError
    from cda.llm.providers import get_provider

    key = prompt_fingerprint("claude-x", "s", "u")
    (tmp_path / f"{key}.json").write_text(json.dumps({"response": "anthropic reply"}))
    anthropic = get_provider(
        load_config(overrides={"llm": {"model": "claude-x"}}), mock=True, fixtures_dir=tmp_path
    )
    assert anthropic.complete(system="s", user="u", max_tokens=5, temperature=0) == (
        "anthropic reply"
    )
    openai = get_provider(
        load_config(overrides={"llm": {"model": "openai:claude-x"}}),
        mock=True,
        fixtures_dir=tmp_path,
    )
    with pytest.raises(FixtureMissError):
        openai.complete(system="s", user="u", max_tokens=5, temperature=0)


def test_verify_labels_the_primary_model_from_the_report_header(tmp_path, monkeypatch):
    from cda.registers import verify as verify_mod

    base = "drift_novelty_20260801_090000"
    (tmp_path / f"{base}.md").write_text(
        "<!-- generated: 2026-08-01T09:00:00 | model: claude-opus-recorded | cda: v1 -->\n"
        "### Cluster A: repair\n",
        encoding="utf-8",
    )
    (tmp_path / f"{base}_prompt.md").write_text('TARGET TERM: "novelty"\nprompt', encoding="utf-8")
    (tmp_path / f"{base}_system.md").write_text("system", encoding="utf-8")
    second = SimpleNamespace(
        name="stub", model="haiku", complete=lambda **_: "### Cluster R1: replayed\n"
    )
    monkeypatch.setattr(verify_mod, "resolve_model_spec", lambda _spec: second)
    config = load_config(
        overrides={
            "paths": {"output_dir": str(tmp_path)},
            "llm": {"model": "claude-config-now", "verification_model": "haiku"},
        }
    )
    result = verify_mod.run_verify("novelty", config, progress=lambda _m: None)
    text = Path(result["verification_file"]).read_text(encoding="utf-8")
    assert "`claude-opus-recorded` (primary)" in text
    assert "claude-config-now" not in text


def test_comparable_body_cuts_at_the_structural_marker():
    from cda.divergence.report import END_OF_LLM_BODY
    from cda.registers.verify import extract_register_structure

    report = (
        f"### Cluster A: repair\n{END_OF_LLM_BODY}\n## Renamed computed section\n"
        "### Cluster Z: computed boilerplate\nSemantic distance scale: 1 = same\n"
    )
    structure = extract_register_structure(report)
    assert structure == {"researcher_register_labels": ["repair"], "distance_ratings": []}


# ---- 11.-12. report wording and fail-loud cleanups ----------------------------


def test_clustering_mode_typo_is_a_config_error():
    from cda.pipeline import clustering_mode

    with pytest.raises(ConfigError, match="clustering.mode"):
        clustering_mode(load_config(overrides={"clustering": {"mode": "pooled"}}))


def test_max_k_ceilings_come_from_config_only():
    from cda.pipeline import max_k_params

    config = load_config(overrides={"clustering": {"joint_max_k": 7}})
    assert max_k_params(config)["joint"] == 7
    del config.data["clustering"]["researcher_max_k"]
    with pytest.raises(ConfigError, match="researcher_max_k"):
        max_k_params(config)


def test_cluster_block_refuses_missing_roles():
    from cda.registers.assemble import format_cluster_block

    entry = {"cluster_id": 0, "n_passages": 1, "representatives": ["t"], "sources": ["a.pdf"]}
    with pytest.raises(KeyError):
        format_cluster_block(ClusterSummary(True, 1, 1, 0.5, "note", [entry]))


def test_unusable_theme_entry_is_refused_not_dropped():
    from cda.registers.assemble import _researcher_context

    config = load_config(overrides={"themes": ["ok-tag", {"description": "no tag"}]})
    with pytest.raises(ConfigError, match="themes entry"):
        _researcher_context(config)


class _StubLive:
    name = "stub"
    model = "claude-sonnet-4-6"

    def __init__(self):
        self.calls = 0

    def complete(self, **_kwargs):
        self.calls += 1
        return f"live answer {self.calls}"


def test_recording_refuses_to_replace_a_replayable_fixture_before_paying(tmp_path):
    from cda.llm.mock import RecordingProvider

    live = _StubLive()
    RecordingProvider(live, tmp_path).complete(system="s", user="u", max_tokens=5, temperature=0)
    with pytest.raises(LLMError, match="refusing to replace"):
        RecordingProvider(live, tmp_path).complete(
            system="s", user="u", max_tokens=5, temperature=0
        )
    assert live.calls == 1
    RecordingProvider(live, tmp_path, overwrite=True).complete(
        system="s", user="u", max_tokens=5, temperature=0
    )
    fixture = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert fixture["response"] == "live answer 2"


def test_recording_heals_a_corrupt_fixture(tmp_path):
    from cda.llm.mock import RecordingProvider

    key = prompt_fingerprint("claude-sonnet-4-6", "s", "u")
    (tmp_path / f"{key}.json").write_text("{half", encoding="utf-8")
    RecordingProvider(_StubLive(), tmp_path).complete(
        system="s", user="u", max_tokens=5, temperature=0
    )
    assert json.loads((tmp_path / f"{key}.json").read_text())["response"] == "live answer 1"


def test_outputs_import_does_not_touch_the_umask(monkeypatch, tmp_path):
    import cda.outputs as outputs

    def forbidden(_mask):
        raise AssertionError("os.umask called")

    monkeypatch.setattr(os, "umask", forbidden)
    reloaded = importlib.reload(outputs)
    reloaded.write_text_atomic(tmp_path / "a.md", "text")
    reloaded.write_text_immutable(tmp_path / "b.md", "text")
    assert (tmp_path / "b.md").read_text() == "text"
