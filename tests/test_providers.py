"""Provider seam: fixture replay, loud misses, recording, fingerprints."""

import json

import pytest

from cda.llm.base import LLMError, prompt_fingerprint
from cda.llm.mock import MockProvider, RecordingProvider
from cda.llm.providers import AnthropicProvider


def test_fingerprint_stability_and_sensitivity():
    a = prompt_fingerprint("model-x", "system", "user")
    assert a == prompt_fingerprint("model-x", "system", "user")
    assert len(a) == 16
    assert a != prompt_fingerprint("model-y", "system", "user")
    assert a != prompt_fingerprint("model-x", "system", "user ")
    # The separator prevents boundary-shift collisions.
    assert prompt_fingerprint("m", "ab", "c") != prompt_fingerprint("m", "a", "bc")


def test_mock_replays_fixture(tmp_path):
    key = prompt_fingerprint("claude-sonnet-4-6", "sys", "usr")
    (tmp_path / f"{key}.json").write_text(json.dumps({"response": "the report"}))
    provider = MockProvider(fixtures_dir=tmp_path, model="claude-sonnet-4-6")
    assert (
        provider.complete(system="sys", user="usr", max_tokens=10, temperature=0.0) == "the report"
    )


def test_mock_miss_fails_loudly_with_instructions(tmp_path):
    provider = MockProvider(fixtures_dir=tmp_path, model="claude-sonnet-4-6")
    with pytest.raises(LLMError) as excinfo:
        provider.complete(system="sys", user="usr", max_tokens=10, temperature=0.0)
    message = str(excinfo.value)
    assert prompt_fingerprint("claude-sonnet-4-6", "sys", "usr") in message
    assert "--record-fixtures" in message
    assert "cda demo --term novelty --mock-llm" in message


def test_mock_rejects_corrupt_fixture(tmp_path):
    key = prompt_fingerprint("m", "s", "u")
    (tmp_path / f"{key}.json").write_text("{not json")
    with pytest.raises(LLMError, match="corrupt"):
        MockProvider(tmp_path, "m").complete(system="s", user="u", max_tokens=1, temperature=0.0)


class StubLive:
    name = "stub"
    model = "claude-sonnet-4-6"

    def complete(self, *, system, user, max_tokens, temperature):
        return "live answer"


def test_recording_provider_round_trip(tmp_path):
    recorder = RecordingProvider(StubLive(), fixtures_dir=tmp_path)
    answer = recorder.complete(system="s", user="u", max_tokens=5, temperature=0.0)
    assert answer == "live answer"
    # The recorded fixture must replay identically through MockProvider.
    replay = MockProvider(tmp_path, "claude-sonnet-4-6")
    assert replay.complete(system="s", user="u", max_tokens=5, temperature=0.0) == "live answer"
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert payload["model"] == "claude-sonnet-4-6"
    assert "_note" in payload


def test_anthropic_provider_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="--mock-llm"):
        AnthropicProvider(model="claude-sonnet-4-6")


def test_get_provider_routes_a_provider_model_spec(monkeypatch):
    """`llm.model` accepts the same "provider:model" form as the verification
    hook, so the analysis call can run against a non-Anthropic endpoint."""
    from cda.config import load_config
    from cda.llm.providers import get_provider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    plain = get_provider(load_config(), mock=False)
    assert plain.name == "anthropic"
    assert plain.model == "claude-sonnet-4-6"  # the paper's pinned default

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pytest.importorskip("openai")
    routed = get_provider(load_config(overrides={"llm": {"model": "openai:gpt-4o"}}), mock=False)
    assert routed.name == "openai-compatible"
    assert routed.model == "gpt-4o"


def test_get_provider_without_a_key_says_which_one(monkeypatch):
    from cda.config import load_config
    from cda.llm.base import LLMError
    from cda.llm.providers import get_provider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pytest.importorskip("openai")
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        get_provider(load_config(overrides={"llm": {"model": "openai:gpt-4o"}}), mock=False)


def test_fixture_fingerprint_is_model_scoped():
    """A fixture recorded under one model must not replay under another —
    a report may never be replayed as if a different model had written it."""
    from cda.llm.base import prompt_fingerprint

    system, user = "sys", "usr"
    assert prompt_fingerprint("claude-sonnet-4-6", system, user) != prompt_fingerprint(
        "openai:gpt-4o", system, user
    )


def test_mock_provider_keys_fixtures_by_the_provider_qualified_model(tmp_path):
    """Fixtures recorded under a 'provider:model' spec must replay with --mock-llm.

    Both sides key the fingerprint by ``fixture_model(spec)``, which keeps the
    non-Anthropic prefix, so ``openai:gpt-4o`` replays its own recording.
    """
    from cda.config import load_config
    from cda.llm.providers import fixture_model, get_provider

    class StubOpenAI:
        name = "openai-compatible"
        model = "gpt-4o"  # what OpenAICompatibleProvider carries: prefix stripped

        def complete(self, *, system, user, max_tokens, temperature):
            return "recorded answer"

    RecordingProvider(
        StubOpenAI(), fixtures_dir=tmp_path, fingerprint_model=fixture_model("openai:gpt-4o")
    ).complete(system="s", user="u", max_tokens=5, temperature=0.0)

    config = load_config(overrides={"llm": {"model": "openai:gpt-4o"}})
    replay = get_provider(config, mock=True, fixtures_dir=tmp_path)
    assert replay.complete(system="s", user="u", max_tokens=5, temperature=0.0) == (
        "recorded answer"
    )


def test_mock_provider_refuses_an_unknown_provider_prefix(tmp_path):
    """Parity with the live path: a spec that cannot run live must not
    silently replay either."""
    from cda.config import load_config
    from cda.llm.providers import get_provider

    config = load_config(overrides={"llm": {"model": "ollama:llama3"}})
    with pytest.raises(LLMError, match="Unknown model provider 'ollama:'"):
        get_provider(config, mock=True, fixtures_dir=tmp_path)


def test_live_api_failures_become_instructive_llm_errors(monkeypatch):
    """A mistyped key, a rate limit, or a network drop is the most common
    first live-mode failure; it must say what to do, not dump an SDK
    traceback (the CLI only catches LLMError)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider(model="claude-sonnet-4-6")

    class Boom:
        def create(self, **_kwargs):
            raise RuntimeError("401 authentication_error: invalid x-api-key")

    provider._client.messages = Boom()
    with pytest.raises(LLMError) as excinfo:
        provider.complete(system="s", user="u", max_tokens=5, temperature=0.0)
    message = str(excinfo.value)
    assert "ANTHROPIC_API_KEY" in message and "--mock-llm" in message


def test_validation_replay_then_live_pays_only_for_missing_readings(tmp_path):
    """A rerun after an interrupted live validation must replay the readings
    that already succeeded (recorded as fixtures) and call live only for the
    rest."""
    from cda.validation.agents import _ReplayThenLive

    calls = []

    class CountingLive:
        name = "stub"
        model = "claude-sonnet-4-6"

        def complete(self, *, system, user, max_tokens, temperature):
            calls.append(user)
            return f"reading of {user}"

    def build():
        return _ReplayThenLive(
            MockProvider(fixtures_dir=tmp_path, model="claude-sonnet-4-6"),
            RecordingProvider(CountingLive(), fixtures_dir=tmp_path),
        )

    first = build().complete(system="s", user="cluster-1", max_tokens=5, temperature=0.0)
    assert first == "reading of cluster-1" and len(calls) == 1

    # The rerun replays the recorded reading for free and pays only for the
    # reading that never completed.
    rerun = build()
    assert (
        rerun.complete(system="s", user="cluster-1", max_tokens=5, temperature=0.0)
        == "reading of cluster-1"
    )
    assert len(calls) == 1  # no second payment for the completed reading
    rerun.complete(system="s", user="cluster-2", max_tokens=5, temperature=0.0)
    assert calls == ["cluster-1", "cluster-2"]


def test_anthropic_truncated_reply_is_refused(monkeypatch):
    """A max_tokens-truncated reply must never pass as a complete report:
    the committed demo fixtures already sit near the 6000-token default cap,
    so one more cluster would otherwise write a silently incomplete report."""
    from types import SimpleNamespace

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider(model="claude-sonnet-4-6")

    class Truncating:
        def create(self, **_kwargs):
            return SimpleNamespace(
                stop_reason="max_tokens",
                content=[SimpleNamespace(text="cut off mid-sec")],
            )

    provider._client.messages = Truncating()
    with pytest.raises(LLMError, match="max_tokens"):
        provider.complete(system="s", user="u", max_tokens=5, temperature=0.0)


def test_replay_then_live_heals_a_corrupt_recorded_reading(tmp_path):
    """A user-recorded fixture truncated mid-write used to wedge every
    validation resume: replay hit the same corrupt file each rerun, and the
    error's remedies (re-record / restore from git) never applied to
    git-ignored recordings. Corrupt now falls through to live, which
    re-records over the bad file."""
    from cda.llm.base import prompt_fingerprint
    from cda.validation.agents import _ReplayThenLive

    key = prompt_fingerprint("claude-sonnet-4-6", "s", "u")
    (tmp_path / f"{key}.json").write_text('{"trunc', encoding="utf-8")

    healed = _ReplayThenLive(
        MockProvider(tmp_path, "claude-sonnet-4-6"),
        RecordingProvider(StubLive(), tmp_path),
    )
    assert healed.complete(system="s", user="u", max_tokens=5, temperature=0.0) == "live answer"
    # The live call re-recorded a good fixture over the corrupt one.
    replay = MockProvider(tmp_path, "claude-sonnet-4-6")
    assert replay.complete(system="s", user="u", max_tokens=5, temperature=0.0) == "live answer"


def test_json_array_fixture_is_a_corrupt_fixture_not_a_traceback(tmp_path):
    from cda.llm.base import FixtureCorruptError, prompt_fingerprint

    key = prompt_fingerprint("claude-sonnet-4-6", "s", "u")
    (tmp_path / f"{key}.json").write_text('["not", "an", "object"]', encoding="utf-8")
    replay = MockProvider(tmp_path, "claude-sonnet-4-6")
    with pytest.raises(FixtureCorruptError, match="corrupt"):
        replay.complete(system="s", user="u", max_tokens=5, temperature=0.0)


def test_mock_heals_a_fixture_truncated_mid_multibyte_character(tmp_path):
    """A recording interrupted inside a multi-byte character is exactly the
    wedged resume the corrupt-fixture path exists for: the *read* fails
    before any JSON parsing, and the raw codec error used to escape the
    provider seam entirely."""
    from cda.llm.base import FixtureCorruptError, prompt_fingerprint

    key = prompt_fingerprint("claude-sonnet-4-6", "s", "u")
    (tmp_path / f"{key}.json").write_bytes(b'{"response": "caf\xc3')
    replay = MockProvider(tmp_path, "claude-sonnet-4-6")
    with pytest.raises(FixtureCorruptError, match="corrupt"):
        replay.complete(system="s", user="u", max_tokens=5, temperature=0.0)


def test_replay_then_live_heals_a_multibyte_truncated_recording(tmp_path):
    """Because it is a FixtureCorruptError and not a raw UnicodeDecodeError,
    replay-then-live re-records over the bad file instead of dying on it
    every rerun."""
    from cda.llm.base import prompt_fingerprint
    from cda.validation.agents import _ReplayThenLive

    key = prompt_fingerprint("claude-sonnet-4-6", "s", "u")
    (tmp_path / f"{key}.json").write_bytes(b'{"response": "caf\xc3')

    healed = _ReplayThenLive(
        MockProvider(tmp_path, "claude-sonnet-4-6"),
        RecordingProvider(StubLive(), tmp_path),
    )
    assert healed.complete(system="s", user="u", max_tokens=5, temperature=0.0) == "live answer"
    replay = MockProvider(tmp_path, "claude-sonnet-4-6")
    assert replay.complete(system="s", user="u", max_tokens=5, temperature=0.0) == "live answer"


def test_truncation_message_does_not_prescribe_a_key_that_may_not_apply():
    """The validation reader and split calls pass their own caps (300 / 1200
    tokens) that no config key controls, so an unconditional "raise
    llm.max_tokens and rerun" sent those users round a loop. The message
    names the cap that actually bound the call."""
    from cda.llm.providers import _truncated_reply_error
    from cda.validation.agents import READING_MAX_TOKENS

    message = str(_truncated_reply_error("model's", READING_MAX_TOKENS))
    assert str(READING_MAX_TOKENS) in message
    assert "if llm.max_tokens in your config is" in message
    # And it no longer promises that a partial run left no files behind:
    # the analyze path writes the prompt and diagnostics before the call.
    assert "nothing was written" not in message


def test_both_providers_refuse_truncation_with_the_same_message(monkeypatch):
    """Provider parity: two different sentinels (Anthropic stop_reason
    "max_tokens", OpenAI finish_reason "length"), one refusal."""
    import sys
    import types
    from types import SimpleNamespace

    from cda.llm.providers import OpenAICompatibleProvider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    anthropic_provider = AnthropicProvider(model="claude-sonnet-4-6")
    anthropic_provider._client.messages = SimpleNamespace(
        create=lambda **_kw: SimpleNamespace(
            stop_reason="max_tokens", content=[SimpleNamespace(text="cut off")]
        )
    )
    with pytest.raises(LLMError) as anthropic_error:
        anthropic_provider.complete(system="s", user="u", max_tokens=1234, temperature=0.0)

    # The real `openai` package is optional (verification-hook only).
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **_kw: SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kw: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="length",
                            message=SimpleNamespace(content="cut off"),
                        )
                    ]
                )
            )
        )
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(LLMError) as openai_error:
        OpenAICompatibleProvider(model="gpt-4o").complete(
            system="s", user="u", max_tokens=1234, temperature=0.0
        )

    for message in (str(anthropic_error.value), str(openai_error.value)):
        assert "1234" in message
        assert "if llm.max_tokens in your config is" in message
        assert "no report was written" in message


class TruncatingLive:
    """A live provider whose reply always hits the token cap."""

    name = "stub"
    model = "claude-sonnet-4-6"

    def complete(self, *, system, user, max_tokens, temperature):
        from cda.llm.providers import _truncated_reply_error

        raise _truncated_reply_error("model's", max_tokens)


def test_recording_a_truncated_live_reply_writes_a_marker_fixture(tmp_path):
    """A truncated reading is an outcome, not a non-event: recording only
    successes left the cluster a permanent fixture hole that aborted every
    later --mock-llm replay with advice the user had already followed."""
    from cda.llm.base import TruncatedReplyError

    recorder = RecordingProvider(TruncatingLive(), fixtures_dir=tmp_path)
    with pytest.raises(TruncatedReplyError):
        recorder.complete(system="s", user="u", max_tokens=300, temperature=0.0)

    key = prompt_fingerprint("claude-sonnet-4-6", "s", "u")
    payload = json.loads((tmp_path / f"{key}.json").read_text(encoding="utf-8"))
    assert payload["truncated"] is True
    assert "response" not in payload
    assert "token cap" in payload["error"]


def test_replaying_a_truncated_marker_fixture_raises_the_same_refusal(tmp_path):
    """Mock replay must reproduce live behavior: the validation loop then
    tolerates the reading as unusable, exactly as the live session did —
    and the fixture is VALID, never healed away as corrupt."""
    from cda.llm.base import TruncatedReplyError

    recorder = RecordingProvider(TruncatingLive(), fixtures_dir=tmp_path)
    with pytest.raises(TruncatedReplyError):
        recorder.complete(system="s", user="u", max_tokens=300, temperature=0.0)

    replay = MockProvider(tmp_path, "claude-sonnet-4-6")
    with pytest.raises(TruncatedReplyError, match="300"):
        replay.complete(system="s", user="u", max_tokens=300, temperature=0.0)


def test_replay_then_live_reproduces_a_recorded_truncation_without_paying(tmp_path):
    """The prompts are deterministic and temperature is 0, so an unchanged
    rerun would truncate again live; replaying the recorded refusal gives the
    same outcome for free. (Shortening the input changes the fingerprint, so
    a genuinely different retry still goes live.)"""
    from cda.llm.base import TruncatedReplyError
    from cda.validation.agents import _ReplayThenLive

    recorder = RecordingProvider(TruncatingLive(), fixtures_dir=tmp_path)
    with pytest.raises(TruncatedReplyError):
        recorder.complete(system="s", user="u", max_tokens=300, temperature=0.0)

    calls = []

    class CountingLive:
        name = "stub"
        model = "claude-sonnet-4-6"

        def complete(self, *, system, user, max_tokens, temperature):
            calls.append(user)
            return "paid answer"

    provider = _ReplayThenLive(
        MockProvider(tmp_path, "claude-sonnet-4-6"),
        RecordingProvider(CountingLive(), tmp_path),
    )
    with pytest.raises(TruncatedReplyError):
        provider.complete(system="s", user="u", max_tokens=300, temperature=0.0)
    assert calls == []  # the recorded refusal replayed; nothing was re-paid


def test_half_written_truncation_marker_still_heals_as_corrupt(tmp_path):
    """Only a complete marker fixture is valid: one interrupted mid-write is
    ordinary corruption, so replay-then-live re-records over it as before."""
    from cda.llm.base import FixtureCorruptError

    key = prompt_fingerprint("claude-sonnet-4-6", "s", "u")
    (tmp_path / f"{key}.json").write_text('{"truncated": true, "err', encoding="utf-8")
    with pytest.raises(FixtureCorruptError, match="corrupt"):
        MockProvider(tmp_path, "claude-sonnet-4-6").complete(
            system="s", user="u", max_tokens=5, temperature=0.0
        )


def test_old_format_fixture_replays_unchanged_and_silently(tmp_path, capsys):
    """Committed demo fixtures predate the max_tokens/temperature fields and
    the truncation marker; they must replay byte-identically with no notes."""
    key = prompt_fingerprint("claude-sonnet-4-6", "sys", "usr")
    (tmp_path / f"{key}.json").write_text(json.dumps({"response": "the report"}))
    provider = MockProvider(fixtures_dir=tmp_path, model="claude-sonnet-4-6")
    assert (
        provider.complete(system="sys", user="usr", max_tokens=10, temperature=0.0) == "the report"
    )
    assert capsys.readouterr().out == ""


def test_recording_stores_the_call_cap_and_temperature(tmp_path):
    """Additive payload fields so replay can notice record/replay config
    drift; the fixture KEY (model + prompts) is unchanged."""
    RecordingProvider(StubLive(), fixtures_dir=tmp_path).complete(
        system="s", user="u", max_tokens=6000, temperature=0.0
    )
    payload = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["max_tokens"] == 6000
    assert payload["temperature"] == 0.0
    assert payload["response"] == "live answer"


def test_replay_under_a_smaller_cap_prints_a_drift_note(tmp_path, capsys):
    """A fixture recorded under a larger cap can replay a reply the same live
    config would refuse as truncated — worth one line, never an error."""
    RecordingProvider(StubLive(), fixtures_dir=tmp_path).complete(
        system="s", user="u", max_tokens=6000, temperature=0.0
    )
    capsys.readouterr()  # drop the "[fixture] recorded" line

    replay = MockProvider(tmp_path, "claude-sonnet-4-6")
    assert replay.complete(system="s", user="u", max_tokens=100, temperature=0.0) == "live answer"
    out = capsys.readouterr().out
    assert "max_tokens=6000" in out and "100" in out

    # An equal (or larger) current cap replays silently.
    assert replay.complete(system="s", user="u", max_tokens=6000, temperature=0.0) == "live answer"
    assert capsys.readouterr().out == ""


def test_require_recorded_fixtures_refuses_an_unplayable_mock_run(tmp_path):
    """The cheap precheck a --mock-llm command runs BEFORE its expensive
    stages: the provider raises the same refusal, but only after extraction
    and clustering — minutes on a real corpus."""
    from cda.llm.mock import has_recorded_fixtures, require_recorded_fixtures

    assert has_recorded_fixtures(tmp_path) is False
    assert has_recorded_fixtures(None) is False
    with pytest.raises(LLMError, match="nothing to replay for cda analyze"):
        require_recorded_fixtures(tmp_path, what="cda analyze")

    (tmp_path / "abc123.json").write_text('{"response": "x"}', encoding="utf-8")
    assert has_recorded_fixtures(tmp_path) is True
    require_recorded_fixtures(tmp_path, what="cda analyze")  # no raise
