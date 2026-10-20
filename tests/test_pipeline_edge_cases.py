"""Edge cases in grounding, the audit parser, verification, config validation,
fixture replay, output installation, provenance and the passage-map palettes.
"""

from __future__ import annotations

import errno
import json
import os

import numpy as np
import pytest

from cda.config import ConfigError, load_config
from cda.registers.audit import _classify_mapping_line, audit_report
from cda.registers.grounding import check_quotations
from cda.registers.verify import extract_register_structure
from cda.types import ClusterSummary, JointClustering

# ---- quote nesting in non-Latin text -----------------------------------

PROMPT = (
    "LITERATURE PASSAGES (cluster summary):\n"
    'Sample 1 [researcher] [mine.pdf]: "평가 이론은 프로그램 설계의 핵심 근거가 된다 그리고 '
    '실무자는 이를 따른다."\n'
    'Sample 2 [literature] [lit.pdf]: "Evaluation theory is the core basis of program '
    'design in every field."\n'
)


def _result(body: str, prompt: str = PROMPT) -> tuple[int, int, int, int]:
    result = check_quotations(body, prompt)
    return (
        result["n_quotes"],
        result["n_verified"],
        result["n_ambiguous"],
        result["unpaired_marks"],
    )


def test_straight_quotes_nested_in_korean_text_hide_no_invented_span():
    body = (
        '연구자는 "평가 이론은 프로그램 설계의 핵심 근거가 된다 "완전히 조작된 문장"이 '
        '이어진다" 라고 썼다.'
    )
    quotes, verified, ambiguous, _ = _result(body)
    assert ambiguous >= 1
    assert verified == 0 and quotes >= 1


@pytest.mark.parametrize("letter", ["λ", "д", "가"])
def test_any_script_letter_touching_a_straight_mark_is_ambiguous(letter):
    body = f'They write "Evaluation theory is the core basis of program design"{letter} here.'
    assert _result(body)[2] == 1


def test_curly_marks_keep_letters_outside_unambiguous():
    body = "그들은 “Evaluation theory is the core basis of program design”라고 썼다."
    assert _result(body) == (1, 1, 0, 0)


@pytest.mark.parametrize("neighbour", ["1", "¹", "_"])
def test_digits_and_underscores_next_to_straight_marks_stay_unambiguous(neighbour):
    body = f'They write "Evaluation theory is the core basis of program design"{neighbour} here.'
    assert _result(body) == (1, 1, 0, 0)


def test_spaced_inner_straight_quote_is_ambiguous():
    body = (
        'The author wrote "Evaluation theory is the core basis " fabricated claim here " '
        'of program design in every field".'
    )
    quotes, verified, ambiguous, _ = _result(body)
    assert ambiguous == 2
    assert verified == 0


def test_spaced_closing_mark_before_an_attribution_is_not_ambiguous():
    body = '"Evaluation theory is the core basis of program design " - lit.pdf and "x"'
    assert _result(body)[2] == 0


# ---- grounding false-unverified cases ----------------------------------


def test_source_filename_containing_a_bracket_is_parsed():
    prompt = 'Sample 1 [top1] [draft [v2].md]: "Novelty is a repair accomplishment of the lab."\n'
    assert (
        check_quotations('"Novelty is a repair accomplishment" - draft [v2].md', prompt)[
            "n_verified"
        ]
        == 1
    )


def test_spaced_ellipsis_is_an_elision():
    body = '"Evaluation theory is the core basis . . . design in every field"'
    assert _result(body) == (1, 1, 0, 0)


def test_blockquote_continuation_prefix_is_ignored():
    body = '> "Evaluation theory is the core basis\n> of program design in every field"'
    assert _result(body) == (1, 1, 0, 0)


def test_final_semicolon_is_retried_like_sentence_punctuation():
    body = 'They write "Evaluation theory is the core basis of program;" Then more.'
    assert _result(body) == (1, 1, 0, 0)


def test_short_curly_label_with_straight_inner_quotes_leaves_no_unpaired_marks():
    assert _result('the “"evaluation" term” is used') == (0, 0, 0, 0)


def test_unverified_curly_pair_lists_its_inner_marks_without_counting_them_unpaired():
    result = check_quotations('the “"evaluation" term is totally invented here today” is', PROMPT)
    assert result["unverified"] == ['"evaluation" term is totally invented here today']
    assert result["unpaired_marks"] == 0


# ---- elided quotes ------------------------------------------------------


def test_ellipsis_tokens_do_not_count_toward_the_word_minimum():
    # Four real words: a short label, not a checked citation.
    assert _result('They say "novelty is ... rewarded unevenly ...".')[0] == 0


def test_elided_quote_built_from_one_word_fragments_is_not_verified():
    body = 'They say "evaluation ... is ... the ... of ... in ... field".'
    quotes, verified, _, _ = _result(body)
    assert (quotes, verified) == (1, 0)


def test_elided_quote_with_substantive_fragments_still_verifies():
    body = '"Evaluation theory is the core … of program design"'
    assert _result(body) == (1, 1, 0, 0)


# ---- audit negations and the register table -----------------------------

_FIELD = "- **k-means/LLM mapping:** "


@pytest.mark.parametrize(
    "mapping",
    [
        "1:1 (cannot be merged or split)",
        "never merged or split (1:1)",
        "None merged and none split",
        "1:1 (merge count 0)",
        "a split of Cluster 2 is not observed",
        "a merge with Cluster 1 does not occur",
        "a split was not found",
        "Split? No. Merge? No.",
    ],
)
def test_negated_mapping_forms_are_one_to_one(mapping):
    assert _classify_mapping_line(_FIELD + mapping) == "one_to_one"


def test_negation_does_not_reach_a_word_outside_its_list():
    assert _classify_mapping_line(_FIELD + "Cluster 3 is not split, merged with Cluster 4") == (
        "merge"
    )


def test_merges_naming_clusters_zero_and_one_is_still_a_merge():
    assert _classify_mapping_line(_FIELD + "merges 0 and 1") == "merge"


_TABLE = """### Cluster A: x
## 2. Researcher vs. Literature Divergence
| {first} | {second} | k-means clusters absorbed |
|---|---|---|
| Cluster 0 | L-R1 | Clusters 0 and 1 |
| Cluster 0 | L-R2 | Cluster 2 |
| Cluster 3 | L-R3 | Cluster 3 |
"""


def test_table_prefers_the_absorbed_column_and_keys_rows_by_register():
    report = _TABLE.format(first="Closest k-means cluster", second="Literature register")
    audit = audit_report(report, 1, 4)
    literature = audit["literature"]
    assert (literature["merge"], literature["one_to_one"], literature["split"]) == (1, 2, 0)
    assert not any("more than once" in note for note in audit["notes"])


def test_table_without_a_register_column_says_so():
    report = _TABLE.format(first="Closest k-means cluster", second="Definition")
    notes = audit_report(report, 1, 4)["notes"]
    assert any("no column headed 'register'" in note for note in notes)


# ---- distance ratings ---------------------------------------------------


@pytest.mark.parametrize(
    ("line", "ratings"),
    [
        ("Semantic distance: B: 2; C: 5", [2, 5]),
        ("Semantic distance from Cluster 2: 4", [4]),
        ("- **Semantic distance from other clusters:** A–B: 3 (adjacent usage)", [3]),
        ("Distance to B rated 4", [4]),
        ("Semantic distance scale: 1 = terminological variants; 5 = incommensurable", []),
        ("Flag pairs with distance >= 3", []),
        ("distance to Cluster 3: 7", []),
    ],
)
def test_distance_ratings_follow_a_label(line, ratings):
    assert extract_register_structure(line)["distance_ratings"] == ratings


# ---- settings refused before retrieval -------------------------


class _Untouchable:
    """Records any use: the pipeline must not reach retrieval with a bad config."""

    name = "untouchable"

    def __init__(self, touched: list[str]):
        self._touched = touched

    def __getattr__(self, attribute):
        self._touched.append(attribute)
        raise AssertionError(f"retrieval started before config validation ({attribute})")


def _config(tmp_path, **overrides):
    return load_config(overrides={"paths": {"output_dir": str(tmp_path / "outputs")}, **overrides})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"figure": {"enabled": "maybe"}}, "figure.enabled"),
        ({"figure": {"dpi": 0}}, "figure.dpi"),
        ({"llm": {"temperature": 1.5}}, "llm.temperature"),
        ({"llm": {"model": "openai:gpt-4o", "temperature": 2.5}}, "llm.temperature"),
        ({"distance_scale": [1, 2]}, "distance_scale"),
        ({"themes": [{"tag": None}]}, "themes entry"),
        ({"themes": [{"tag": "x", "description": None}]}, "description"),
        ({"themes": [{"tag": "x", "description": "  "}]}, "description"),
    ],
)
def test_bad_settings_fail_before_retrieval(tmp_path, overrides, message):
    from cda.pipeline import analyze_term

    touched: list[str] = []
    with pytest.raises(ConfigError, match=message):
        analyze_term(
            "novelty",
            _config(tmp_path, **overrides),
            mock=True,
            fixtures_dir=tmp_path,
            embedder=_Untouchable(touched),
            collection=_Untouchable(touched),
            progress=lambda _message: None,
        )
    assert touched == []


def test_openai_temperature_up_to_two_is_accepted(tmp_path):
    from cda.pipeline import validated_settings

    config = _config(tmp_path, llm={"model": "openai:gpt-4o", "temperature": 1.5})
    assert validated_settings(config)["temperature"] == 1.5


def test_verify_checks_the_temperature_before_reading_any_run(tmp_path):
    from cda.registers.verify import run_verify

    config = _config(tmp_path, llm={"verification_model": "claude-haiku-4-5", "temperature": 1.5})
    with pytest.raises(ConfigError, match="llm.temperature"):
        run_verify("novelty", config, progress=lambda _message: None)


def test_theme_without_description_key_is_a_bare_tag(tmp_path):
    from cda.pipeline import validated_settings

    lines = validated_settings(_config(tmp_path, themes=[{"tag": "repair"}]))["theme_lines"]
    assert lines == ["  #repair"]


def test_max_window_chars_message_names_the_real_constraint(tmp_path):
    from cda.pipeline import validated_settings

    with pytest.raises(ConfigError) as caught:
        validated_settings(_config(tmp_path, retrieval={"max_window_chars": 2}))
    message = str(caught.value)
    assert "0 (unbounded)" in message and "at least" in message
    assert "below retrieval.min_window_chars (0)" not in message


# ---- mock replay prefers the configured model's recording --------------


def _fixture(directory, model: str, system: str, user: str, response: str) -> None:
    from cda.llm.base import prompt_fingerprint

    path = directory / f"{prompt_fingerprint(model, system, user)}.json"
    path.write_text(json.dumps({"model": model, "response": response}), encoding="utf-8")


def test_mock_provider_falls_back_and_names_the_replayed_model(tmp_path):
    from cda.llm.mock import MockProvider

    _fixture(tmp_path, "claude-code-session", "s", "u", "authored reply")
    provider = MockProvider(
        tmp_path,
        "claude-sonnet-4-6",
        fallbacks=[("claude-code-session", "claude-code-session")],
    )
    reply = provider.complete(system="s", user="u", max_tokens=10, temperature=0.0)
    assert reply == "authored reply"
    assert provider.model == "claude-code-session"


def test_mock_provider_prefers_the_configured_model_recording(tmp_path):
    from cda.llm.mock import MockProvider

    _fixture(tmp_path, "claude-code-session", "s", "u", "authored reply")
    _fixture(tmp_path, "claude-sonnet-4-6", "s", "u", "live recording")
    provider = MockProvider(
        tmp_path,
        "claude-sonnet-4-6",
        fallbacks=[("claude-code-session", "claude-code-session")],
    )
    assert provider.complete(system="s", user="u", max_tokens=10, temperature=0.0) == (
        "live recording"
    )
    assert provider.model == "claude-sonnet-4-6"


def test_mock_demo_keeps_the_configured_model_and_falls_back_to_the_authored_one(
    tmp_path, monkeypatch
):
    import cda.demo
    import cda.pipeline

    toy = tmp_path / "demo" / "toy_corpus"
    toy.mkdir(parents=True)
    (toy / "abstracts.jsonl").write_text('{"source": "a.abstract", "abstract": "t"}\n')
    captured = {}

    def fake_analyze(term, config, **kwargs):
        captured["model"] = config.get("llm.model")
        captured["fallbacks"] = kwargs.get("fallback_fixture_models")
        return {"status": "skipped", "reason": "test"}

    monkeypatch.setattr(cda.demo, "get_embedder", lambda config: None)
    monkeypatch.setattr(cda.demo, "ensure_toy_corpus_ingested", lambda *args: None)
    monkeypatch.setattr(cda.pipeline, "analyze_term", fake_analyze)
    cda.demo.run_demo(term="novelty", mock=True, demo_dir=tmp_path / "demo")
    assert captured["model"] == load_config().get("llm.model")
    assert tuple(captured["fallbacks"]) == ("claude-code-session",)


# ---- install_new without hard links -------------------------------------


@pytest.mark.parametrize("code", [errno.EPERM, errno.ENOTSUP, errno.EXDEV, errno.EMLINK])
def test_install_new_copies_when_hard_links_are_unsupported(tmp_path, monkeypatch, code):
    from cda import outputs

    def no_links(source, target):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(outputs.os, "link", no_links)
    target = tmp_path / "report.md"
    outputs.write_text_immutable(target, "body")
    assert target.read_text(encoding="utf-8") == "body"
    outputs.write_text_immutable(target, "body")  # identical replay accepted
    with pytest.raises(FileExistsError):
        outputs.write_text_immutable(target, "changed")
    assert target.read_text(encoding="utf-8") == "body"
    assert [p.name for p in tmp_path.iterdir()] == ["report.md"]


def test_install_new_propagates_other_link_errors(tmp_path, monkeypatch):
    from cda import outputs

    def denied(source, target):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(outputs.os, "link", denied)
    with pytest.raises(PermissionError):
        outputs.write_text_immutable(tmp_path / "report.md", "body")
    assert not (tmp_path / "report.md").exists()


# ---- embedding provenance -----------------------------------------------


def test_short_model_alias_resolves_like_sentence_transformers():
    from cda.corpus.embeddings import hub_model_id

    assert hub_model_id("all-MiniLM-L6-v2") == "sentence-transformers/all-MiniLM-L6-v2"
    assert hub_model_id("org/model") == "org/model"
    assert hub_model_id("bert-base-uncased") == "bert-base-uncased"


def test_provenance_looks_up_the_cache_under_the_resolved_id(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub

    from cda.corpus.embeddings import SentenceTransformerEmbedder

    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    repos = []

    def cached(repo, filename, revision=None):
        repos.append(repo)
        return str(snapshot / filename)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    embedder._model = SimpleNamespace(device="cpu")
    result = embedder.provenance()
    assert repos == ["sentence-transformers/all-MiniLM-L6-v2"]
    assert result["resolved_revision"] == "abc123"


def test_embedding_identity_ignores_snapshot_extras_and_status_text():
    from cda.provenance import _embedding_identity

    def manifest(files, **extra):
        return {
            "embedding": {
                "name": "all-MiniLM-L6-v2",
                "resolved_revision": "abc",
                "files": files,
                "status": "local model files hashed",
                **extra,
            }
        }

    weights = {"model.safetensors": "w", "config.json": "c", "tokenizer.json": "t"}
    base = manifest(weights)
    extras = manifest(
        {**weights, "README.md": "r", "onnx/model.onnx": "o"},
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        status="hashed on another machine",
    )
    assert _embedding_identity(base) == _embedding_identity(extras)
    changed = manifest({**weights, "model.safetensors": "other"})
    assert _embedding_identity(base) != _embedding_identity(changed)


# ---- passage-map palettes -----------------------------------------------


def _rgb(colour: str) -> np.ndarray:
    from matplotlib.colors import to_rgb

    return np.array(to_rgb(colour))


def test_joint_palette_covers_joint_max_k_without_grey():
    pytest.importorskip("matplotlib")
    from cda.divergence.figure import PALETTE, UNASSIGNED_COLOUR

    joint_max_k = load_config().get_int("clustering.joint_max_k")
    colours = PALETTE[:joint_max_k]
    assert len(set(colours)) == joint_max_k
    assert PALETTE[:7] == [
        "#0072b2",
        "#d55e00",
        "#009e73",
        "#cc79a7",
        "#e69f00",
        "#56b4e9",
        "#8c564b",
    ]
    for colour in colours:
        rgb = _rgb(colour)
        assert rgb.max() - rgb.min() >= 0.1, f"{colour} is grey"
        assert np.linalg.norm(rgb - _rgb(UNASSIGNED_COLOUR)) > 0.25


def test_independent_researcher_palette_covers_a_fixed_k_up_to_literature_max_k():
    pytest.importorskip("matplotlib")
    from cda.divergence.figure import LITERATURE_ALPHA, _independent_palettes

    literature, researcher = _independent_palettes()
    needed = load_config().get_int("clustering.literature_max_k")
    assert len(set(researcher[:needed])) == needed
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    assert researcher[:8] == [to_hex(c) for c in colormaps["Dark2"].colors]  # unchanged for k <= 8
    drawn = [LITERATURE_ALPHA * _rgb(c) + (1 - LITERATURE_ALPHA) for c in literature]
    closest = min(np.linalg.norm(lit - _rgb(res)) for lit in drawn for res in researcher)
    assert closest > 0.25


# ---- silhouette n/a in the prompt, stability progress -----------------------


def _summary(used: bool) -> ClusterSummary:
    return ClusterSummary(used, 4, 1 if not used else 2, 0.0 if not used else 0.25, "note")


def _saturation() -> dict:
    return {"enabled": False, "n_candidates": 4, "n_selected": 4}


def test_prompt_says_silhouette_na_without_a_partition():
    from cda.registers.assemble import build_user_prompt

    prompt = build_user_prompt(
        "novelty", _summary(False), _summary(False), _saturation(), load_config()
    )
    assert "Researcher clustering: k=1, silhouette=n/a, N=4" in prompt
    assert "Literature clustering: k=1, silhouette=n/a, N=4" in prompt
    joint = JointClustering(False, 1, 0.0, "none", 2, 2)
    joint_prompt = build_user_prompt(
        "novelty", _summary(False), _summary(False), _saturation(), load_config(), joint=joint
    )
    assert "(joint k=1, silhouette=n/a on the pooled passages)" in joint_prompt


def test_prompt_keeps_three_decimal_silhouette_for_a_real_partition():
    from cda.registers.assemble import build_user_prompt

    prompt = build_user_prompt(
        "novelty", _summary(True), _summary(True), _saturation(), load_config()
    )
    assert "Researcher clustering: k=2, silhouette=0.250, N=4" in prompt


def _stability(monkeypatch, summary: dict, mode: str) -> list[str]:
    import cda.clustering.order_stability as order_stability
    from cda.pipeline import run_stability_check

    monkeypatch.setattr(order_stability, "run_trials", lambda *args, **kwargs: [])
    monkeypatch.setattr(order_stability, "summarize", lambda *args, **kwargs: dict(summary))
    messages: list[str] = []
    run_stability_check([], [], load_config(), mode=mode, embedder=None, progress=messages.append)
    return messages


def test_stability_progress_says_no_valid_partition(monkeypatch):
    joint = {
        "canonical_k": 1,
        "k_match_rate": 0.0,
        "shuffled_trials": 30,
        "canonical_partition_found": False,
    }
    (line,) = _stability(monkeypatch, joint, "joint")
    assert "no valid partition" in line and "matched" not in line

    corpus = {"k_match_rate": 0.5, "shuffled_trials": 30}
    independent = {
        "researcher": {**corpus, "canonical_partition_found": False},
        "literature": corpus,  # recorded before the key existed: counts as found
    }
    (line,) = _stability(monkeypatch, independent, "independent")
    assert "researcher no valid partition" in line
    assert "literature k matched 50%" in line


def test_stability_progress_is_unchanged_for_found_partitions(monkeypatch):
    joint = {"canonical_k": 3, "k_match_rate": 0.4, "shuffled_trials": 30}
    assert _stability(monkeypatch, joint, "joint") == [
        "      stability: k=3 matched 40% of 30 shuffled orderings"
    ]


def test_stability_refusal_reason_comes_from_order_stability(monkeypatch):
    import cda.clustering.order_stability as order_stability
    from cda.pipeline import run_stability_check

    monkeypatch.setattr(order_stability, "resolution_override_reason", lambda params: "sentinel")
    result = run_stability_check([], [], load_config(), mode="joint", embedder=None)
    assert result == {"enabled": False, "trials": result["trials"], "reason": "sentinel"}
    monkeypatch.undo()
    pinned = run_stability_check(
        [], [], load_config(overrides={"clustering": {"k": 3}}), mode="joint", embedder=None
    )
    assert pinned["reason"] == (
        "clustering.method/clustering.k override the paper's "
        "silhouette k-selection, which is what this check measures"
    )
