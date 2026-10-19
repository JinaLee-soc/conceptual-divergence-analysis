"""Configuration loading, precedence, and copy-drift guards."""

import sys
import types
from pathlib import Path

import pytest
import yaml

from cda.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_defaults_load_and_carry_paper_values():
    config = load_config()
    assert config.get("embedding.model") == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.get("llm.model") == "claude-sonnet-4-6"
    assert config.get("clustering.representatives_per_cluster") == 5
    # Joint clustering is the default; independent reproduces the paper's runs.
    assert config.get("clustering.mode") == "joint"
    assert config.get("clustering.joint_max_k") == 15
    # No acquisition-date window unless one is asked for.
    assert config.get("timeframe.added_from") is None
    assert config.get("timeframe.added_to") is None
    assert config.get("timeframe.include_undated") is False
    # The default clusters every retrieved literature passage: the sweep is
    # available but scoped to the prompt rather than the geometry. Its own
    # parameters still carry the paper's values for `scope: sample`.
    assert config.get("saturation.enabled") is True
    assert config.get("saturation.scope") == "prompt"
    assert config.get("saturation.max_passages") == 1200
    assert config.get("saturation.ari_threshold") == 0.8
    assert config.get("saturation.consecutive_increments") == 2
    assert config.get("flags.researcher_share_threshold") == 0.80
    assert config.get("flags.literature_share_threshold") == 0.10
    assert config.get("clustering.kmeans_restarts") == 5
    assert config.get("clustering.seed") == 42


def test_packaged_defaults_identical_to_repo_copy():
    """config/defaults.yaml (documented) must equal src/cda/defaults.yaml (packaged)."""
    repo_copy = (REPO_ROOT / "config" / "defaults.yaml").read_text(encoding="utf-8")
    packaged_copy = (REPO_ROOT / "src" / "cda" / "defaults.yaml").read_text(encoding="utf-8")
    assert repo_copy == packaged_copy, (
        "config/defaults.yaml and src/cda/defaults.yaml have drifted apart; "
        "copy one over the other (they must stay identical)."
    )


@pytest.mark.parametrize(
    "name",
    [
        "register_assignment_v1.md",
        "register_assignment_v2.md",
        "register_assignment_v3.md",
        "register_assignment_v4.md",
        "register_assignment_v5.md",
        "validation_tasks_v1.md",
    ],
)
def test_packaged_prompt_identical_to_repo_copy(name):
    # Active and archived prompts each keep a repository source and a
    # packaged copy; both must match for versioned replay.
    repo_copy = (REPO_ROOT / "prompts" / name).read_text(encoding="utf-8")
    packaged_copy = (REPO_ROOT / "src" / "cda" / "prompts" / name).read_text(encoding="utf-8")
    assert repo_copy == packaged_copy


@pytest.mark.parametrize("name", ["register_assignment_v1.md", "register_assignment_v2.md"])
def test_prior_prompt_versions_stay_frozen(name):
    """A superseded prompt is never edited in place — a report's provenance is
    fixed by the version it names. v1 and v2 must not mention joint clustering,
    which arrived with v3."""
    text = (REPO_ROOT / "prompts" / name).read_text(encoding="utf-8")
    assert "JOINT" not in text


def test_example_config_parses_and_documents_every_default_section():
    example = yaml.safe_load(
        (REPO_ROOT / "config" / "config.example.yaml").read_text(encoding="utf-8")
    )
    defaults = yaml.safe_load((REPO_ROOT / "config" / "defaults.yaml").read_text(encoding="utf-8"))
    missing = [key for key in defaults if key not in example]
    assert not missing, f"config.example.yaml is missing documented sections: {missing}"


def test_user_config_overrides_defaults(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text("clustering:\n  seed: 7\n", encoding="utf-8")
    config = load_config(user)
    assert config.get("clustering.seed") == 7
    assert config.get("clustering.kmeans_restarts") == 5  # untouched default


def test_cli_overrides_beat_user_config(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text("paths:\n  output_dir: from_file\n", encoding="utf-8")
    config = load_config(user, overrides={"paths": {"output_dir": "from_cli"}})
    assert config.get("paths.output_dir") == "from_cli"


def test_verification_model_env(monkeypatch):
    monkeypatch.setenv("CDA_VERIFICATION_MODEL", "claude-haiku-4-5-20251001")
    config = load_config()
    assert config.get("llm.verification_model") == "claude-haiku-4-5-20251001"


def test_missing_key_error_names_the_key():
    config = load_config()
    with pytest.raises(ConfigError, match="no.such.key"):
        config.get("no.such.key")


def test_missing_config_file_says_what_to_do(tmp_path):
    with pytest.raises(ConfigError, match="config.example.yaml"):
        load_config(tmp_path / "absent.yaml")


# --- null handling ------------------------------------------------------


def test_null_values_fall_back_to_defaults(tmp_path):
    """A key left empty in a user YAML means "use the default", never None.

    `seed:` (empty value) used to override the default with None and crash
    as `int(None)` deep in the pipeline; `output_dir: ~` (YAML null) sent
    reports to a directory literally named "None".
    """
    user = tmp_path / "user.yaml"
    user.write_text("clustering:\n  seed:\npaths:\n  output_dir: ~\n", encoding="utf-8")
    config = load_config(user)
    assert config.get("clustering.seed") == 42
    assert config.path("paths.output_dir") == Path("./outputs")


def test_path_getter_refuses_none():
    from cda.config import Config

    config = Config({"paths": {"output_dir": None}})
    with pytest.raises(ConfigError, match="paths.output_dir"):
        config.path("paths.output_dir")


# --- unknown keys and wrong shapes --------------------------------------


def test_misspelled_key_is_rejected_with_a_suggestion(tmp_path):
    """A typo'd KEY must fail fast, not silently leave the default in force."""
    user = tmp_path / "user.yaml"
    user.write_text("clustering:\n  metod: hierarchical\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="clustering.metod") as excinfo:
        load_config(user)
    assert "clustering.method" in str(excinfo.value)  # the suggestion


def test_misspelled_top_level_section_is_rejected(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text("clusterin:\n  seed: 7\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="clusterin"):
        load_config(user)


def test_scalar_where_a_section_is_expected_is_rejected(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text("clustering: independent\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(user)


def test_mapping_where_a_scalar_is_expected_is_rejected(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text("clustering:\n  seed:\n    value: 7\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="clustering.seed"):
        load_config(user)


def test_misspelled_key_inside_a_list_entry_is_rejected(tmp_path):
    """`themes: [{tga: ...}]` used to load fine and drop the tag from the prompt."""
    user = tmp_path / "user.yaml"
    user.write_text("themes:\n  - tga: evaluation\n    description: x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"themes\[0\].tga") as excinfo:
        load_config(user)
    assert "themes[0].tag" in str(excinfo.value)  # the suggestion


def test_well_formed_list_entries_still_load(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text(
        "themes:\n  - tag: evaluation\n    description: x\n  - plain-string-tag\n",
        encoding="utf-8",
    )
    config = load_config(user)
    assert config.get_list("themes")[0]["tag"] == "evaluation"


def test_quoted_numeric_key_matches_an_integer_default(tmp_path):
    """distance_scale is keyed by integers; `"1":` used to be "unknown" with a
    suggestion identical to what the user wrote."""
    user = tmp_path / "user.yaml"
    user.write_text('distance_scale:\n  "1": near-synonyms\n', encoding="utf-8")
    config = load_config(user)
    scale = config.section("distance_scale")
    assert scale[1] == "near-synonyms"
    assert "1" not in scale  # not merged in twice under two key types


def test_empty_config_path_is_rejected():
    with pytest.raises(ConfigError, match="empty path"):
        load_config("")


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze", "--term", "novelty", "--fixtures-dir", ""],
        ["verify", "--term", "novelty", "--fixtures-dir", ""],
        ["ingest", "--pdf-dir", "somewhere", "--zotero-csv", ""],
        ["demo", "--demo-dir", ""],
    ],
)
def test_empty_path_flags_are_rejected_rather_than_ignored(argv, capsys):
    """`--fixtures-dir "$DIR"` with $DIR unset used to fall back silently, or
    become Path("") — the current directory."""
    from cda.cli import main

    assert main(argv) == 1
    assert "empty path" in capsys.readouterr().err


def test_non_utf8_config_names_the_file(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_bytes("clustering:\n  seed: 7\n".encode("utf-16"))
    with pytest.raises(ConfigError, match="UTF-8"):
        load_config(user)


# --- typed getters ------------------------------------------------------


def test_get_int_names_the_key_on_a_non_number(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text("clustering:\n  kmeans_restarts: five\n", encoding="utf-8")
    config = load_config(user)
    with pytest.raises(ConfigError, match="clustering.kmeans_restarts"):
        config.get_int("clustering.kmeans_restarts")


def test_get_int_enforces_minimum():
    config = load_config(overrides={"clustering": {"joint_max_k": 1}})
    with pytest.raises(ConfigError, match="joint_max_k.*>= 2"):
        config.get_int("clustering.joint_max_k", minimum=2)


def test_get_bool_parses_quoted_strings():
    config = load_config(overrides={"corpus": {"recursive": "false"}})
    assert config.get_bool("corpus.recursive") is False
    config = load_config(overrides={"corpus": {"recursive": "true"}})
    assert config.get_bool("corpus.recursive") is True
    config = load_config(overrides={"corpus": {"recursive": "banana"}})
    with pytest.raises(ConfigError, match="corpus.recursive"):
        config.get_bool("corpus.recursive")


def test_get_int_refuses_a_fractional_float():
    """`chunk_size_words: 499.7` used to truncate to 499 without a word."""
    config = load_config(overrides={"corpus": {"chunk_size_words": 499.7}})
    with pytest.raises(ConfigError, match="corpus.chunk_size_words"):
        config.get_int("corpus.chunk_size_words")
    # A float that IS a whole number is still fine (`500.0` means 500).
    config = load_config(overrides={"corpus": {"chunk_size_words": 500.0}})
    assert config.get_int("corpus.chunk_size_words") == 500


def test_get_bool_accepts_the_integers_zero_and_one():
    """`figure.enabled: 1` worked before these settings were typed."""
    config = load_config(overrides={"figure": {"enabled": 1}})
    assert config.get_bool("figure.enabled") is True
    config = load_config(overrides={"figure": {"enabled": 0}})
    assert config.get_bool("figure.enabled") is False
    config = load_config(overrides={"figure": {"enabled": 2}})
    with pytest.raises(ConfigError, match="figure.enabled"):
        config.get_bool("figure.enabled")


def test_get_float_refuses_nan_and_infinity():
    """NaN compares False against every bound, so it disabled the threshold."""
    config = load_config(overrides={"saturation": {"ari_threshold": float("nan")}})
    with pytest.raises(ConfigError, match="finite"):
        config.get_float("saturation.ari_threshold", minimum=0.0, maximum=1.0)
    config = load_config(overrides={"flags": {"researcher_share_threshold": float("inf")}})
    with pytest.raises(ConfigError, match="finite"):
        config.get_float("flags.researcher_share_threshold", minimum=0.0, maximum=1.0)


def test_quoted_boolean_is_read_by_meaning_not_by_truthiness():
    """`include_undated: "false"` is True under a bare bool() cast."""
    from cda.corpus.timeframe import timeframe_from_config

    config = load_config(
        overrides={"timeframe": {"added_to": "2026-05-01", "include_undated": "false"}}
    )
    assert timeframe_from_config(config)["include_undated"] is False


def test_get_list_wraps_a_bare_scalar():
    config = load_config(overrides={"validation": {"readers": "claude-sonnet-4-6"}})
    assert config.get_list("validation.readers") == ["claude-sonnet-4-6"]
    config = load_config()
    assert config.get_list("validation.readers") == []


# --- up-front pipeline validation ---------------------------------------


def test_percent_style_flag_threshold_is_rejected():
    """`researcher_share_threshold: 80` (meaning 80%) can never match a 0-1
    share, so every run would quietly report "no flags" as if it were a real
    finding of no divergence."""
    from cda.pipeline import flag_params

    config = load_config(overrides={"flags": {"researcher_share_threshold": 80}})
    with pytest.raises(ConfigError, match="researcher_share_threshold"):
        flag_params(config)


def test_window_mode_typo_is_rejected_up_front():
    from cda.pipeline import retrieval_params

    config = load_config(overrides={"retrieval": {"window_mode": "sentance"}})
    with pytest.raises(ConfigError, match="window_mode"):
        retrieval_params(config)


def test_saturation_cap_below_start_is_rejected():
    """max_passages < start used to silently disable the cap entirely.

    The cross-field check applies when the sweep can actually run, which
    needs scope "sample" (the shipped default scope is "prompt", where the
    sweep never executes — see saturation_params)."""
    from cda.pipeline import saturation_params

    config = load_config(
        overrides={"saturation": {"scope": "sample", "max_passages": 30, "start": 50}}
    )
    with pytest.raises(ConfigError, match="max_passages"):
        saturation_params(config)


def test_saturation_cap_below_start_is_ignored_when_saturation_is_off():
    """Unused knobs must not block a run that never reads them: the sweep
    runs only with enabled: true AND scope: sample, so either way out means
    the inconsistent pair is never read."""
    from cda.pipeline import saturation_params

    config = load_config(
        overrides={
            "saturation": {
                "enabled": False,
                "scope": "sample",
                "max_passages": 30,
                "start": 50,
            }
        }
    )
    assert saturation_params(config)["enabled"] is False
    # Under the shipped default scope ("prompt") the sweep never executes, so
    # the same inconsistent pair must not refuse to start the run either.
    config = load_config(overrides={"saturation": {"max_passages": 30, "start": 50}})
    assert saturation_params(config)["scope"] == "prompt"
    # Types and spellings are still checked, so a typo surfaces the day the
    # sweep is switched on.
    config = load_config(overrides={"saturation": {"enabled": False, "step": "two"}})
    with pytest.raises(ConfigError, match="saturation.step"):
        saturation_params(config)


def test_saturation_ranking_typo_is_rejected_up_front():
    """Anything but 'similarity' used to be silently treated as term mode."""
    from cda.pipeline import saturation_params

    config = load_config(overrides={"saturation": {"ranking": "trem"}})
    with pytest.raises(ConfigError, match="saturation.ranking"):
        saturation_params(config)


def test_zero_representatives_per_cluster_is_rejected():
    """0 sent the LLM clusters with no sample passages to read at all."""
    from cda.pipeline import representatives_per_cluster

    config = load_config(overrides={"clustering": {"representatives_per_cluster": 0}})
    with pytest.raises(ConfigError, match="representatives_per_cluster"):
        representatives_per_cluster(config)
    config = load_config(overrides={"clustering": {"representatives_per_cluster": -2}})
    with pytest.raises(ConfigError, match="representatives_per_cluster"):
        representatives_per_cluster(config)


def test_bare_string_themes_container_does_not_become_letter_tags():
    """`themes: evaluation` used to put "#e", "#v", "#a"… into the prompt."""
    from cda.registers.assemble import _researcher_context

    config = load_config(overrides={"themes": "evaluation-theory"})
    context = _researcher_context(config)
    assert context == "Theme tags:\n  #evaluation-theory"


def test_max_k_below_two_is_rejected_up_front():
    from cda.pipeline import resolution_params

    config = load_config(overrides={"clustering": {"joint_max_k": 1}})
    with pytest.raises(ConfigError, match="joint_max_k"):
        resolution_params(config)


def test_unknown_model_provider_prefix_is_refused():
    """`ollama:llama3:8b` used to be sent to Anthropic as a model of that name."""
    from cda.llm.base import LLMError
    from cda.llm.providers import resolve_model_spec

    with pytest.raises(LLMError, match="Unknown model provider 'ollama:'"):
        resolve_model_spec("ollama:llama3:8b")


def test_known_provider_prefixes_still_route(monkeypatch):
    from cda.llm.providers import (
        AnthropicProvider,
        OpenAICompatibleProvider,
        resolve_model_spec,
    )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # The real `openai` package is optional (verification-hook only) and is
    # not part of the dev install, so stub it: this test checks routing, and
    # must pass on a fresh offline clone.
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    assert isinstance(resolve_model_spec("openai:gpt-4o"), OpenAICompatibleProvider)
    assert isinstance(resolve_model_spec("anthropic:claude-sonnet-4-6"), AnthropicProvider)
    assert resolve_model_spec("claude-sonnet-4-6").model == "claude-sonnet-4-6"
