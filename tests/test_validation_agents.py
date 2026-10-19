"""Agent roles: blind reading prompts, strict parsing, fixture replay."""

import json
from datetime import datetime

import pytest

from cda.config import load_config
from cda.validation.agents import (
    estimate_cost,
    mechanical_split,
    parse_json_reply,
    read_clusters,
    reading_user_prompt,
    resolve_readers,
    spec_model,
    suggest_edit,
)
from cda.validation.prompt_loader import load_validation_prompts


class ScriptedProvider:
    """LLMProvider stub replying from a fixed queue (repeats the last)."""

    def __init__(self, replies, model="stub-model"):
        self.replies = list(replies)
        self.model = model
        self.calls = []

    def complete(self, *, system, user, max_tokens, temperature):
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def reading_reply(judgment="coherent"):
    return json.dumps({"gloss": "one usage", "judgment": judgment, "rationale": "shared framing"})


@pytest.fixture
def prompts():
    return load_validation_prompts(load_config())


def test_reading_prompt_is_blind():
    user = reading_user_prompt("novelty", ["Alpha passage one.", "Alpha passage two."])
    assert "novelty" in user and "Alpha passage one." in user
    # No labels, sources, years, or verdict vocabulary reach the reader.
    for forbidden in ("Cluster", "cluster 0", ".pdf", ".abstract", "verdict", "coherent"):
        assert forbidden not in user


def test_read_clusters_parses_and_flags_invalid(prompts):
    config = load_config()
    provider = ScriptedProvider([reading_reply("mixed"), "I think it is coherent, thanks!"])
    shown = {"literature": {0: ["text a", "text b"], 1: ["text c", "text d"]}}
    readings = read_clusters(
        "novelty",
        shown,
        prompts,
        config,
        mock=False,
        fixtures_dir=None,
        providers={spec: provider for spec in resolve_readers(config)},
        progress=lambda _m: None,
    )
    assert [r["cluster_id"] for r in readings] == [0, 1]
    assert readings[0]["valid"] is True and readings[0]["judgment"] == "mixed"
    # Prose reply: recorded raw, marked invalid, judgment left empty.
    assert readings[1]["valid"] is False and readings[1]["judgment"] == ""
    assert "coherent" in readings[1]["raw"]


def test_read_clusters_rejects_unknown_judgment(prompts):
    config = load_config()
    provider = ScriptedProvider(
        [json.dumps({"gloss": "x", "judgment": "excellent", "rationale": "y"})]
    )
    readings = read_clusters(
        "novelty",
        {"literature": {0: ["a"]}},
        prompts,
        config,
        mock=False,
        fixtures_dir=None,
        providers={spec: provider for spec in resolve_readers(config)},
        progress=lambda _m: None,
    )
    assert readings[0]["valid"] is False


def test_read_clusters_normalizes_judgment_case_and_whitespace(prompts):
    """An alternate reader answering " Coherent " means coherent: case and
    whitespace are presentation, not meaning, and demoting the reading to
    invalid silently dropped it from the agreement statistics."""
    config = load_config()
    provider = ScriptedProvider(
        [json.dumps({"gloss": "one usage", "judgment": " Coherent ", "rationale": "shared"})]
    )
    readings = read_clusters(
        "novelty",
        {"literature": {0: ["text a"]}},
        prompts,
        config,
        mock=False,
        fixtures_dir=None,
        providers={spec: provider for spec in resolve_readers(config)},
        progress=lambda _m: None,
    )
    assert readings[0]["valid"] is True
    # Stored in the normalized form, so downstream comparisons see one spelling.
    assert readings[0]["judgment"] == "coherent"


def test_resolve_readers_derivation(monkeypatch):
    config = load_config()
    assert resolve_readers(config) == ["claude-sonnet-4-6"]
    monkeypatch.setenv("CDA_VERIFICATION_MODEL", "openai:gpt-4o")
    config = load_config()
    assert resolve_readers(config) == ["claude-sonnet-4-6", "openai:gpt-4o"]
    config = load_config(overrides={"validation": {"readers": ["anthropic:claude-x"]}})
    assert resolve_readers(config) == ["anthropic:claude-x"]
    assert spec_model("openai:gpt-4o") == "gpt-4o"
    assert spec_model("claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_parse_json_reply_strictness():
    assert parse_json_reply("no json here", ("a",)) is None
    assert parse_json_reply('{"a": 1}', ("a", "b")) is None
    assert parse_json_reply('noise {"a": 1, "b": 2} trailing', ("a", "b")) == {"a": 1, "b": 2}
    assert parse_json_reply('{"a": [1,2,}', ("a",)) is None


def test_mock_replay_via_fixture_chain(tmp_path, prompts):
    """Fixtures written by fingerprint replay through read_clusters end to end."""
    from cda.llm.base import prompt_fingerprint

    config = load_config()
    model = "claude-sonnet-4-6"
    shown = {"literature": {0: ["fixture passage one", "fixture passage two"]}}
    user = reading_user_prompt("novelty", shown["literature"][0])
    key = prompt_fingerprint(model, prompts["cluster_reading_system"], user)
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"model": model, "response": reading_reply("artifact")}), encoding="utf-8"
    )
    readings = read_clusters(
        "novelty",
        shown,
        prompts,
        config,
        mock=True,
        fixtures_dir=tmp_path,
        progress=lambda _m: None,
    )
    assert readings[0]["valid"] is True and readings[0]["judgment"] == "artifact"


def test_truncated_marker_fixture_replays_as_one_unusable_reading(tmp_path, prompts):
    """A recorded truncation must replay as the live session behaved: one
    unusable reading, not an aborted --mock-llm run. RecordingProvider writes
    a "truncated": true marker when the live call hit its cap; MockProvider
    replays it as the same TruncatedReplyError, and the reading loop records
    the cluster unusable while every other cluster's reading survives."""
    from cda.llm.base import prompt_fingerprint

    config = load_config()
    model = "claude-sonnet-4-6"
    shown = {"literature": {0: ["cut-off passage"], 1: ["healthy passage"]}}
    system = prompts["cluster_reading_system"]
    truncated_key = prompt_fingerprint(
        model, system, reading_user_prompt("novelty", ["cut-off passage"])
    )
    (tmp_path / f"{truncated_key}.json").write_text(
        json.dumps(
            {
                "model": model,
                "truncated": True,
                "error": "The model's reply was cut off at this call's token cap (300).",
            }
        ),
        encoding="utf-8",
    )
    healthy_key = prompt_fingerprint(
        model, system, reading_user_prompt("novelty", ["healthy passage"])
    )
    (tmp_path / f"{healthy_key}.json").write_text(
        json.dumps({"model": model, "response": reading_reply("coherent")}), encoding="utf-8"
    )
    readings = read_clusters(
        "novelty",
        shown,
        prompts,
        config,
        mock=True,
        fixtures_dir=tmp_path,
        progress=lambda _m: None,
    )
    assert [r["cluster_id"] for r in readings] == [0, 1]
    assert readings[0]["valid"] is False
    assert "300-token cap" in readings[0]["raw"]  # the standard truncation record
    assert readings[1]["valid"] is True and readings[1]["judgment"] == "coherent"


def test_mock_missing_fixture_fails_loudly_with_validate_guidance(tmp_path, prompts):
    """The miss must give validate-specific next steps (record fixtures /
    run live), not the analyze demo's remediation, while keeping the
    fingerprint details for debugging."""
    from cda.llm.base import LLMError

    config = load_config()
    with pytest.raises(LLMError, match="record-fixtures") as excinfo:
        read_clusters(
            "novelty",
            {"literature": {0: ["unrecorded text"]}},
            prompts,
            config,
            mock=True,
            fixtures_dir=tmp_path,
            progress=lambda _m: None,
        )
    message = str(excinfo.value)
    assert "fingerprint" in message
    assert "demo/fixtures/validation/README.md" in message
    assert "cda demo --term novelty" not in message  # the analyze remediation


def _session_for_edit(tmp_path, with_verdicts=True):

    from cda.validation.session import create_session, record_verdict

    partitions = [
        {
            "revision": 0,
            "side": "literature",
            "kind": "original",
            "assignments": {"a": 0, "b": 0, "c": 1, "d": 1},
            "clusters": [
                {"cluster_id": 0, "n_passages": 2, "silhouette": 0.2, "shown": ["a", "b"]},
                {"cluster_id": 1, "n_passages": 2, "silhouette": 0.3, "shown": ["c", "d"]},
            ],
        }
    ]
    passages = {
        "a": {"text": "alpha usage one", "side": "literature"},
        "b": {"text": "alpha usage two", "side": "literature"},
        "c": {"text": "beta usage one", "side": "literature"},
        "d": {"text": "beta usage two", "side": "literature"},
    }
    _dir, session = create_session(
        "novelty",
        params={"seed": 42, "embedding_model": str(load_config().get("embedding.model"))},
        sides={"literature": {"validated": True, "note": None}},
        partitions=partitions,
        passages=passages,
        readings=[],
        output_dir=tmp_path,
        now=datetime(2026, 7, 14, 10, 0, 0),
    )
    if with_verdicts:
        # Blind verdicts on both clusters: suggest_edit refuses to discuss a
        # cluster the researcher has not yet judged.
        for cluster_id in (0, 1):
            record_verdict(
                session,
                side="literature",
                cluster_id=cluster_id,
                revision=0,
                verdict="coherent",
                now=datetime(2026, 7, 14, 10, 1, 0),
            )
    return session


def test_suggest_edit_refuses_before_a_blind_verdict(tmp_path, prompts, anchored_embedder):
    """The advice channel must not unblind a verdict (docs/validation_protocol.md).

    The agent's merge/split argument is a reading of the same clusters'
    passages, so it is gated exactly like reveal_readings: no verdict on the
    involved clusters, no advice — otherwise a researcher could read the
    agent's opinion first and still have the later verdict counted as blind.
    """
    session = _session_for_edit(tmp_path, with_verdicts=False)
    provider = ScriptedProvider(['{"recommendation": "merge", "argument": "same"}'])
    with pytest.raises(ValueError, match="blind verdict"):
        suggest_edit(
            session,
            kind="merge",
            side="literature",
            clusters=[0, 1],
            config=load_config(),
            embedder=anchored_embedder,
            prompts=prompts,
            now=datetime(2026, 7, 14, 10, 2, 0),
            provider=provider,
            progress=lambda _m: None,
        )
    assert provider.calls == []  # refused before any model call
    assert session["pending_edit"] is None


def test_suggest_split_refuses_a_single_passage_cluster(tmp_path, prompts, anchored_embedder):
    """k > n makes the mechanical split a no-op; refuse before paying for advice."""

    from cda.validation.session import create_session, record_verdict

    _dir, session = create_session(
        "novelty",
        params={"seed": 42, "embedding_model": str(load_config().get("embedding.model"))},
        sides={"literature": {"validated": True, "note": None}},
        partitions=[
            {
                "revision": 0,
                "side": "literature",
                "kind": "original",
                "assignments": {"a": 0},
                "clusters": [
                    {"cluster_id": 0, "n_passages": 1, "silhouette": None, "shown": ["a"]}
                ],
            }
        ],
        passages={"a": {"text": "alpha usage one", "side": "literature"}},
        readings=[],
        output_dir=tmp_path,
        now=datetime(2026, 7, 14, 10, 0, 0),
    )
    record_verdict(
        session,
        side="literature",
        cluster_id=0,
        revision=0,
        verdict="coherent",
        now=datetime(2026, 7, 14, 10, 1, 0),
    )
    provider = ScriptedProvider(['{"recommendation": "split", "argument": "x"}'])
    with pytest.raises(ValueError, match="nothing to split"):
        suggest_edit(
            session,
            kind="split",
            side="literature",
            clusters=[0],
            config=load_config(),
            embedder=anchored_embedder,
            prompts=prompts,
            now=datetime(2026, 7, 14, 10, 2, 0),
            provider=provider,
            progress=lambda _m: None,
        )
    assert provider.calls == []


def test_semantic_split_using_one_letter_is_recorded_as_unusable(
    tmp_path, prompts, anchored_embedder
):
    """An all-A "split" is not a split; it must not become an applyable proposal."""
    session = _session_for_edit(tmp_path)
    advice = json.dumps({"recommendation": "split", "argument": "two usages"})
    degenerate = json.dumps({"assignments": ["A", "A"], "a_gloss": "ga", "b_gloss": "gb"})
    edit = suggest_edit(
        session,
        kind="split",
        side="literature",
        clusters=[0],
        config=load_config(),
        embedder=anchored_embedder,
        prompts=prompts,
        now=datetime(2026, 7, 14, 10, 2, 0),
        provider=ScriptedProvider([advice, degenerate]),
        progress=lambda _m: None,
    )
    assert edit["split_proposals"]["semantic"] is None


def test_suggest_split_records_both_proposals(tmp_path, prompts, anchored_embedder):
    session = _session_for_edit(tmp_path)
    advice = json.dumps({"recommendation": "keep-together", "argument": "one usage"})
    split = json.dumps({"assignments": ["A", "B"], "a_gloss": "ga", "b_gloss": "gb"})
    provider = ScriptedProvider([advice, split])
    edit = suggest_edit(
        session,
        kind="split",
        side="literature",
        clusters=[0],
        config=load_config(),
        embedder=anchored_embedder,
        prompts=prompts,
        now=datetime(2026, 7, 14, 10, 2, 0),
        provider=provider,
        progress=lambda _m: None,
    )
    assert edit["status"] == "pending" and session["pending_edit"] is edit
    assert edit["agent_advice"]["recommendation"] == "keep-together"
    # Semantic proposal mapped back by position over sorted chunk ids (a, b).
    assert edit["split_proposals"]["semantic"]["assignments"] == {"a": "A", "b": "B"}
    mechanical = edit["split_proposals"]["mechanical"]
    assert set(mechanical) == {"a", "b"} and set(mechanical.values()) <= {"A", "B"}


def test_semantic_split_letters_are_normalized(tmp_path, prompts, anchored_embedder):
    """ "a" means A: lowercase or padded letters from an alternate reader were
    silently voiding the whole semantic proposal (mechanical-only split)."""
    session = _session_for_edit(tmp_path)
    advice = json.dumps({"recommendation": "split", "argument": "two usages"})
    split = json.dumps({"assignments": ["a", " B "], "a_gloss": "ga", "b_gloss": "gb"})
    edit = suggest_edit(
        session,
        kind="split",
        side="literature",
        clusters=[0],
        config=load_config(),
        embedder=anchored_embedder,
        prompts=prompts,
        now=datetime(2026, 7, 14, 10, 2, 0),
        provider=ScriptedProvider([advice, split]),
        progress=lambda _m: None,
    )
    # Stored normalized: apply_pending_edit compares against "A"/"B" literally.
    assert edit["split_proposals"]["semantic"]["assignments"] == {"a": "A", "b": "B"}


def test_split_second_call_failure_keeps_the_paid_advice(tmp_path, prompts, anchored_embedder):
    """A rate limit on the semantic-split call must not discard the advice
    call that already succeeded: curation calls are never fixture-recorded,
    so propagating would re-bill the advice on every retry. The suggestion is
    recorded with the semantic proposal unusable and the mechanical split
    intact."""
    from cda.llm.base import LLMError

    class AdviceThenFailure:
        model = "stub-model"

        def __init__(self):
            self.calls = 0

        def complete(self, *, system, user, max_tokens, temperature):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({"recommendation": "split", "argument": "two usages"})
            raise LLMError("The Anthropic API call failed (RateLimitError)")

    session = _session_for_edit(tmp_path)
    lines = []
    edit = suggest_edit(
        session,
        kind="split",
        side="literature",
        clusters=[0],
        config=load_config(),
        embedder=anchored_embedder,
        prompts=prompts,
        now=datetime(2026, 7, 14, 10, 2, 0),
        provider=AdviceThenFailure(),
        progress=lines.append,
    )
    # The paid advice survives, attached to a pending suggestion.
    assert session["pending_edit"] is edit
    assert edit["agent_advice"]["valid"] is True
    assert edit["agent_advice"]["recommendation"] == "split"
    # The semantic proposal is the ordinary unusable case; mechanical stands.
    assert edit["split_proposals"]["semantic"] is None
    assert "RateLimitError" in edit["split_proposals"]["semantic_raw"]
    assert set(edit["split_proposals"]["mechanical"]) == {"a", "b"}
    assert any("keeping the paid advice" in line for line in lines)


def test_advice_call_failure_still_propagates(tmp_path, prompts, anchored_embedder):
    """Nothing has been paid when the FIRST call fails, so a real error there
    must still surface instead of producing an advice-less suggestion."""
    from cda.llm.base import LLMError

    class FailingProvider:
        model = "stub-model"

        def complete(self, **kwargs):
            raise LLMError("The Anthropic API call failed (AuthenticationError)")

    session = _session_for_edit(tmp_path)
    with pytest.raises(LLMError, match="AuthenticationError"):
        suggest_edit(
            session,
            kind="split",
            side="literature",
            clusters=[0],
            config=load_config(),
            embedder=anchored_embedder,
            prompts=prompts,
            now=datetime(2026, 7, 14, 10, 2, 0),
            provider=FailingProvider(),
            progress=lambda _m: None,
        )
    assert session["pending_edit"] is None


def test_suggest_edit_guards(tmp_path, prompts, anchored_embedder):
    session = _session_for_edit(tmp_path)
    common = dict(
        config=load_config(),
        embedder=anchored_embedder,
        prompts=prompts,
        now=datetime(2026, 7, 14, 10, 2, 0),
        provider=ScriptedProvider(['{"recommendation": "merge", "argument": "same"}']),
        progress=lambda _m: None,
    )
    with pytest.raises(ValueError, match="exactly two"):
        suggest_edit(session, kind="merge", side="literature", clusters=[0], **common)
    with pytest.raises(ValueError, match="exactly one"):
        suggest_edit(session, kind="split", side="literature", clusters=[0, 1], **common)
    suggest_edit(session, kind="merge", side="literature", clusters=[0, 1], **common)
    with pytest.raises(ValueError, match="pending suggestion"):
        suggest_edit(session, kind="merge", side="literature", clusters=[0, 1], **common)


def test_mechanical_split_is_seeded_deterministic(anchored_embedder):
    texts = [f"alpha case {i}" for i in range(4)] + [f"beta case {i}" for i in range(4)]
    first = mechanical_split(texts, anchored_embedder, seed=42)
    second = mechanical_split(texts, anchored_embedder, seed=42)
    assert first == second
    # The two anchored topics separate.
    alpha_letters = {first[i] for i in range(4)}
    beta_letters = {first[i] for i in range(4, 8)}
    assert alpha_letters != beta_letters and len(alpha_letters) == 1


class BrokenEmbedder:
    """Embedder whose second vector is unusable (a broken model, not a bad store)."""

    def __init__(self, bad_value):
        self.bad_value = bad_value

    def embed(self, texts):
        return [[self.bad_value] * 4 if i == 1 else [1.0, 0.0, 0.0, 0.0] for i in range(len(texts))]


@pytest.mark.parametrize("bad_value", [float("nan"), 0.0])
def test_mechanical_split_refuses_unusable_embeddings_by_name(bad_value):
    """A zero vector used to be kept silently and land on side A; a NaN row hit
    the kmeans backstop with a message that named no passage at all."""
    with pytest.raises(ValueError, match="alpha::1"):
        mechanical_split(
            ["one", "two", "three"],
            BrokenEmbedder(bad_value),
            seed=42,
            names=["alpha::0", "alpha::1", "alpha::2"],
        )


class TruncatingProvider:
    """Stub whose reply always hits the provider's max_tokens refusal."""

    model = "stub-model"

    def complete(self, *, system, user, max_tokens, temperature):
        from cda.llm.base import TruncatedReplyError

        # The type, not the wording, is what marks a truncation: the
        # validation loop must keep treating this as one unusable reading
        # even after the provider message is reworded.
        raise TruncatedReplyError(f"The model's reply was cut off at the cap ({max_tokens}).")


def test_truncated_reading_is_recorded_unusable_not_fatal(prompts):
    """Pre-fix behavior: one over-long reply is an invalid reading, not a reason
    to discard every other cluster's (paid) reading in the session."""
    config = load_config()
    lines = []
    readings = read_clusters(
        "novelty",
        {"literature": {0: ["text a"], 1: ["text b"]}},
        prompts,
        config,
        mock=False,
        fixtures_dir=None,
        providers={spec: TruncatingProvider() for spec in resolve_readers(config)},
        progress=lines.append,
    )
    assert len(readings) == 2
    assert not any(r["valid"] for r in readings)
    # The record says which call was cut off, its real cap, and what to change.
    assert "literature cluster 0" in readings[0]["raw"]
    assert "300-token cap" in readings[0]["raw"]
    assert "passages_shown_per_cluster" in readings[0]["raw"]
    assert "llm.max_tokens does not change it" in readings[0]["raw"]
    assert any("cut off" in line for line in lines)


def test_non_truncation_llm_errors_still_propagate(prompts):
    class FailingProvider:
        model = "stub-model"

        def complete(self, **kwargs):
            from cda.llm.base import LLMError

            raise LLMError("The Anthropic API call failed (AuthenticationError)")

    config = load_config()
    with pytest.raises(Exception, match="AuthenticationError"):
        read_clusters(
            "novelty",
            {"literature": {0: ["text a"]}},
            prompts,
            config,
            mock=False,
            fixtures_dir=None,
            providers={spec: FailingProvider() for spec in resolve_readers(config)},
            progress=lambda _m: None,
        )


def test_estimate_cost_line():
    line = estimate_cost(3, 4200)
    assert "3 model call(s)" in line and "4,200" in line
