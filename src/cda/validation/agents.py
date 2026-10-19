"""The two agent roles in a validation session.

Reader agents produce the independent *cluster readings* the researcher's
blind verdicts are compared against: one short call per cluster per reader,
seeing only the term and the cluster's passages (no labels, no sources, no
one else's opinion). The *revision agent* argues about a specific merge or
split the researcher is considering and, for splits, proposes a semantic
two-way reading alongside the mechanical k=2 split.

Replies must be minimal JSON; anything unparseable is recorded as invalid
and excluded from agreement statistics, never guessed at.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from cda.clustering.spherical import kmeans_labels, normalize_rows
from cda.config import Config
from cda.llm.base import FixtureCorruptError, FixtureMissError, LLMError, TruncatedReplyError

# Re-exported so every consumer strips (and validates) a 'provider:model'
# spec the same way; fixture fingerprints depend on all paths agreeing on
# the bare model name.
from cda.llm.providers import spec_model  # noqa: F401
from cda.validation.curate import require_original_verdicts, require_session_embedder
from cda.validation.session import current_revision, has_verdict, log_reveal, partition_for

# Validation's own caps, deliberately small: a cluster reading is three short
# JSON fields, and a split is one letter per passage. They are NOT llm.max_tokens
# (which sizes the analysis report), so provider advice to raise that config key
# does not apply here — see _truncation_note.
READING_MAX_TOKENS = 300
SPLIT_MAX_TOKENS = 1200
JUDGMENTS = ("coherent", "mixed", "artifact")


def _is_truncation(exc: LLMError) -> bool:
    """Whether a provider refused a reply because it hit the token cap.

    Decided by type, not by words in the message: a rewording of the refusal
    must not silently turn one unusable reading back into a dead session.
    """
    return isinstance(exc, TruncatedReplyError)


def _truncation_note(what: str, cap: int, remedy: str, exc: LLMError) -> str:
    """The record left behind when a validation call hits its cap.

    Says which call was cut off, what its cap actually is, and what the
    researcher can change — the provider's own advice ("raise llm.max_tokens")
    cannot help, because these caps live in this module, not in the config.
    """
    return (
        f"[unusable: the {what} was cut off at validation's {cap}-token cap; "
        f"this cap is fixed in cda/validation/agents.py, so raising "
        f"llm.max_tokens does not change it. {remedy}]\n{exc}"
    )


def resolve_readers(config: Config) -> list[str]:
    """Reader model specs: ``validation.readers``, or the derived default.

    An empty list means the pinned analysis model, plus the verification
    model when one is configured (the same derivation rule ``cda verify``
    uses for its second opinion).
    """
    # get_list wraps a bare `readers: claude-sonnet-4-6` string — it would
    # otherwise iterate into one reader per character, each an LLM call with
    # a one-letter model name.
    configured = [str(s) for s in config.get_list("validation.readers", [])]
    if configured:
        return configured
    readers = [str(config.get("llm.model"))]
    verification = config.get("llm.verification_model", default=None)
    if verification and str(verification) not in readers:
        readers.append(str(verification))
    return readers


class _ReplayThenLive:
    """Replay a recorded reading when one exists; otherwise call live and record.

    With recording on, every completed call lands on disk as a fixture keyed
    by its prompt fingerprint, so a rerun after an interruption pays only for
    the readings that are missing. A corrupt fixture is re-recorded by the
    live call.
    """

    def __init__(self, replay, live):
        self._replay = replay
        self._live = live
        self.model = live.model

    def complete(self, *, system: str, user: str, max_tokens: int, temperature: float) -> str:
        try:
            return self._replay.complete(
                system=system, user=user, max_tokens=max_tokens, temperature=temperature
            )
        except (FixtureMissError, FixtureCorruptError):
            return self._live.complete(
                system=system, user=user, max_tokens=max_tokens, temperature=temperature
            )


def reader_provider(spec: str, *, mock: bool, fixtures_dir: Path | None, record: bool = False):
    from cda.llm.mock import MockProvider, RecordingProvider
    from cda.llm.providers import fixture_model, resolve_model_spec

    if mock:
        if fixtures_dir is None:
            raise LLMError(
                "--mock-llm needs a fixtures directory with recorded validation "
                "readings. The demo passes demo/fixtures/validation automatically."
            )
        return MockProvider(
            fixtures_dir=Path(fixtures_dir),
            model=spec_model(spec),
            fingerprint_model=fixture_model(spec),
        )
    provider = resolve_model_spec(spec)
    if record:
        if fixtures_dir is None:
            raise LLMError("--record-fixtures needs a fixtures directory.")
        provider = _ReplayThenLive(
            MockProvider(
                fixtures_dir=Path(fixtures_dir),
                model=spec_model(spec),
                fingerprint_model=fixture_model(spec),
            ),
            RecordingProvider(provider, Path(fixtures_dir), fingerprint_model=fixture_model(spec)),
        )
    return provider


def _numbered(texts: list[str]) -> str:
    return "\n".join(f'{i + 1}. "{text}"' for i, text in enumerate(texts))


def reading_user_prompt(term: str, texts: list[str]) -> str:
    """The reading payload: the term and numbered passages, nothing else.

    Blinding lives here: no cluster label, no sources, no years, no verdicts.
    """
    return (
        f'Target term: "{term}"\n\n'
        f"The {len(texts)} passages grouped together:\n\n{_numbered(texts)}\n"
    )


def parse_json_reply(raw: str, required: tuple[str, ...]) -> dict | None:
    """First-JSON-object parse; None unless every required key is present."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or any(key not in data for key in required):
        return None
    return data


def read_clusters(
    term: str,
    shown: dict[str, dict[int, list[str]]],
    prompts: dict[str, str],
    config: Config,
    *,
    mock: bool,
    fixtures_dir: Path | None,
    record: bool = False,
    providers: dict[str, object] | None = None,
    revision: int = 0,
    progress: Callable[[str], None] = print,
) -> list[dict]:
    """One independent reading per cluster per reader.

    ``shown`` maps side -> cluster id -> the passage texts the researcher
    will also see; ``revision`` stamps which partition revision the readings
    describe (fresh readings after a curation edit pass r+1). ``providers``
    overrides provider construction (tests). Iteration order is fully
    deterministic: sorted sides, sorted cluster ids, readers in configured
    order.
    """
    readers = resolve_readers(config)
    n_clusters = sum(len(clusters) for clusters in shown.values())
    readings: list[dict] = []
    for spec in readers:
        provider = (
            providers[spec]
            if providers is not None
            else reader_provider(spec, mock=mock, fixtures_dir=fixtures_dir, record=record)
        )
        for side in sorted(shown):
            for cluster_id in sorted(shown[side]):
                try:
                    raw = provider.complete(
                        system=prompts["cluster_reading_system"],
                        user=reading_user_prompt(term, shown[side][cluster_id]),
                        max_tokens=READING_MAX_TOKENS,
                        temperature=0.0,
                    )
                except FixtureMissError as exc:
                    # Only a genuine miss gets the validate-specific rewrap: a
                    # corrupt or truncated fixture file raises a plain LLMError
                    # whose own message names the file, and re-labeling it as
                    # "no recorded readings" would send the user to re-record
                    # (and re-pay) when the fix is restoring the file.
                    # The generic fixture-miss message points at the analyze
                    # demo; give the validate-specific next steps instead
                    # (keeping only the lookup details from the original).
                    lookup = str(exc).split("\nWhy this happens", 1)[0]
                    raise LLMError(
                        "No recorded validation readings for this corpus and "
                        "term.\n"
                        "What to do next:\n"
                        "  - Record them once on a live run (a few cents): add\n"
                        "    --record-fixtures and drop --mock-llm (needs\n"
                        "    ANTHROPIC_API_KEY). After that, --mock-llm replays\n"
                        "    offline forever.\n"
                        "  - Or simply run live (drop --mock-llm).\n"
                        "  - Recording notes: demo/fixtures/validation/README.md\n"
                        f"{lookup}"
                    ) from exc
                except LLMError as exc:
                    if not _is_truncation(exc):
                        raise
                    # A reading that ran past its cap is an unusable reading,
                    # not a reason to throw away the session: it is recorded
                    # like any other unparseable reply (valid=False, excluded
                    # from the agreement stats) so the other clusters' paid
                    # readings survive. The researcher sees it here and in the
                    # report's reader-agreement counts.
                    raw = _truncation_note(
                        f"reading of {side} cluster {cluster_id} by {spec}",
                        READING_MAX_TOKENS,
                        "Show fewer or shorter passages per cluster "
                        "(validation.passages_shown_per_cluster) and rerun; "
                        "this cluster has no usable reading until you do.",
                        exc,
                    )
                    progress(
                        f"[validate] {spec}: reading of {side} cluster "
                        f"{cluster_id} was cut off at the "
                        f"{READING_MAX_TOKENS}-token cap and is recorded as "
                        "unusable"
                    )
                parsed = parse_json_reply(raw, required=("gloss", "judgment", "rationale"))
                # Case and whitespace are presentation, not meaning: an
                # alternate reader (validation.readers takes OpenAI-compatible
                # models too) answering "Coherent" must not be demoted to an
                # invalid reading and dropped from the agreement statistics.
                # The normalized form is what gets stored, so every downstream
                # comparison sees one spelling. Anything not in JUDGMENTS
                # after normalizing stays invalid.
                judgment = str(parsed.get("judgment", "")).strip().casefold() if parsed else ""
                valid = parsed is not None and judgment in JUDGMENTS
                readings.append(
                    {
                        "side": side,
                        "cluster_id": cluster_id,
                        "revision": revision,
                        "model": spec,
                        "gloss": parsed.get("gloss", "") if valid else "",
                        "judgment": judgment if valid else "",
                        "rationale": parsed.get("rationale", "") if valid else "",
                        "valid": valid,
                        "raw": raw,
                    }
                )
        progress(f"[validate] {spec}: read {n_clusters} cluster(s)")
    return readings


def _members(session: dict, side: str, cluster_id: int, revision: int) -> list[tuple[str, str]]:
    """(chunk_id, text) members of a cluster, in sorted-chunk-id order.

    Sorted order is the total order everything downstream relies on: the
    semantic split maps letters back to passages by position.
    """
    partition = partition_for(session, side, revision)
    return [
        (chunk_id, session["passages"][chunk_id]["text"])
        for chunk_id in sorted(partition["assignments"])
        if partition["assignments"][chunk_id] == cluster_id
    ]


def mechanical_split(
    texts: list[str], embedder, seed: int, names: list[str] | None = None
) -> dict[int, str]:
    """Seeded k=2 spherical k-means over member embeddings; position -> A/B.

    ``names`` labels the passages for error messages (chunk ids at the real
    call site); without the shared guard a zero vector would be silently kept
    and land on side "A", and a NaN row would trip kmeans_labels' backstop
    with a message that names no passage at all.
    """
    X = normalize_rows(
        np.asarray(embedder.embed(texts), dtype=float),
        names if names is not None else [f"passage {i + 1}" for i in range(len(texts))],
    )
    labels, _inertia = kmeans_labels(X, 2, seed=seed)
    return {i: ("A" if int(label) == 0 else "B") for i, label in enumerate(labels)}


def suggest_edit(
    session: dict,
    *,
    kind: str,
    side: str,
    clusters: list[int],
    config: Config,
    embedder,
    prompts: dict[str, str],
    now: datetime,
    provider=None,
    progress: Callable[[str], None] = print,
) -> dict:
    """Ask the revision agent about a merge or split; record a pending edit.

    Applies nothing: the researcher decides via ``curate.apply_pending_edit``.
    Live-only in normal use (arbitrary edits cannot be pre-recorded);
    ``provider`` is injectable for tests.
    """
    if kind == "merge" and not len(clusters) == len(set(clusters)) == 2:
        raise ValueError("a merge suggestion needs exactly two different cluster ids")
    if kind == "split" and len(clusters) != 1:
        raise ValueError("a split suggestion needs exactly one cluster id")
    if kind not in ("merge", "split"):
        raise ValueError(f"kind must be 'merge' or 'split', got {kind!r}")
    if session.get("pending_edit"):
        raise ValueError(
            "There is already a pending suggestion; decide it first "
            "(apply or reject) before suggesting another edit."
        )

    revision = current_revision(session, side)
    require_original_verdicts(session, [side])
    # apply_pending_edit refuses a mismatched embedder; checking here too keeps
    # the paid advice call from being billed for an edit that cannot be applied.
    require_session_embedder(session, config)
    # The agent's argument is a reading of these clusters' passages: it needs
    # a verdict first, and showing it is logged as a reveal (docs/validation_protocol.md).
    without_verdict = [c for c in clusters if not has_verdict(session, side, c, revision)]
    if without_verdict:
        cluster_word = "cluster" if len(without_verdict) == 1 else "clusters"
        raise ValueError(
            f"Record your blind verdict on {side} {cluster_word} "
            f"{', '.join(str(c) for c in without_verdict)} before asking the agent "
            "about this edit: the agent's argument is a reading of those passages, "
            "and seeing it first would unblind the verdict."
        )
    term = session["term"]
    if provider is None:
        provider = reader_provider(
            resolve_readers(config)[0], mock=False, fixtures_dir=None, record=False
        )

    if kind == "merge":
        group_a = _members(session, side, clusters[0], revision)
        group_b = _members(session, side, clusters[1], revision)
        user = (
            f'Target term: "{term}"\n\n'
            f"Edit under consideration: MERGE Group 1 and Group 2 into one.\n\n"
            f"Group 1 passages:\n\n{_numbered([t for _, t in group_a])}\n\n"
            f"Group 2 passages:\n\n{_numbered([t for _, t in group_b])}\n"
        )
    else:
        members = _members(session, side, clusters[0], revision)
        if len(members) < 2:
            raise ValueError(
                f"{side} cluster {clusters[0]} has only {len(members)} passage; "
                "there is nothing to split. Pick a cluster with at least two."
            )
        user = (
            f'Target term: "{term}"\n\n'
            f"Edit under consideration: SPLIT this group in two.\n\n"
            f"Passages:\n\n{_numbered([t for _, t in members])}\n"
        )

    progress(f"[validate] asking the revision agent about a {kind} on {side} {clusters}")
    try:
        raw = provider.complete(
            system=prompts["revision_advice_system"],
            user=user,
            max_tokens=READING_MAX_TOKENS,
            temperature=0.0,
        )
    except LLMError as exc:
        if not _is_truncation(exc):
            raise
        # Same choice as read_clusters: an argument that was cut off is an
        # unusable argument (agent_advice.valid = False), and the researcher
        # can still decide the edit themselves. For a split, the mechanical
        # proposal below is computed locally and is unaffected.
        raw = _truncation_note(
            f"revision agent's argument about a {kind} on {side} {clusters}",
            READING_MAX_TOKENS,
            f"The {kind} is still yours to decide without it; asking again "
            "about smaller clusters usually produces a shorter argument.",
            exc,
        )
        progress(f"[validate] the agent's argument was cut off: {raw.splitlines()[0]}")
    advice = parse_json_reply(raw, required=("recommendation", "argument"))

    edit = {
        "edit_id": len(session["edits"]) + 1,
        "kind": kind,
        "side": side,
        "clusters": clusters,
        "revision": revision,
        "status": "pending",
        "agent_advice": {
            "model": getattr(provider, "model", "unknown"),
            "recommendation": advice.get("recommendation") if advice else None,
            "argument": advice.get("argument") if advice else None,
            "valid": advice is not None,
            "raw": raw,
        },
    }

    if kind == "split":
        members = _members(session, side, clusters[0], revision)
        texts = [text for _, text in members]
        try:
            split_raw = provider.complete(
                system=prompts["semantic_split_system"],
                user=f'Target term: "{term}"\n\nPassages:\n\n{_numbered(texts)}\n',
                max_tokens=SPLIT_MAX_TOKENS,
                temperature=0.0,
            )
        except LLMError as exc:
            if _is_truncation(exc):
                # A cut-off split reply lists letters for only some passages,
                # which is exactly the "unusable proposal" case
                # apply_pending_edit already handles (semantic stays None). The
                # mechanical split still gives the researcher something to
                # apply or reject.
                split_raw = _truncation_note(
                    f"semantic split proposal for {side} cluster {clusters[0]} "
                    f"({len(members)} passages)",
                    SPLIT_MAX_TOKENS,
                    "Only the mechanical (k=2) split is offered for this cluster; "
                    "splitting a smaller cluster first keeps the reply within the cap.",
                    exc,
                )
                progress(
                    "[validate] the semantic split proposal was cut off: "
                    f"{split_raw.splitlines()[0]}"
                )
            else:
                # Any other failure here (a rate limit, a dropped connection)
                # must not propagate: the advice call above already succeeded
                # and was already paid for, and curation calls are never
                # fixture-recorded — so raising would discard the advice and a
                # retry would re-bill it. Recording the suggestion with the
                # semantic proposal marked unusable is the lightest way to
                # keep the money: it reuses the exact unusable-proposal path
                # the truncation branch and apply_pending_edit already handle
                # (semantic stays None; the locally computed mechanical split
                # below is unaffected), instead of teaching the recording/
                # replay machinery about curation calls it was never meant to
                # serve.
                split_raw = (
                    f"[unusable: the semantic split call for {side} cluster "
                    f"{clusters[0]} failed after the advice call had already "
                    "been paid for. Only the mechanical (k=2) split is offered "
                    "for this suggestion; to try for a semantic proposal again, "
                    "reject this suggestion and ask about the split once "
                    f"more.]\n{exc}"
                )
                progress(
                    "[validate] the semantic split call failed; keeping the "
                    "paid advice and offering the mechanical split only "
                    f"({type(exc).__name__})"
                )
        parsed = parse_json_reply(split_raw, required=("assignments", "a_gloss", "b_gloss"))
        semantic = None
        # Same tolerance as the judgment parse above: "a" means A, and the
        # normalized letters are what get stored (apply_pending_edit compares
        # against "A"/"B" literally). Anything that is not A or B after
        # normalizing still voids the proposal.
        letters = (
            [str(letter).strip().upper() for letter in parsed["assignments"]]
            if parsed and isinstance(parsed["assignments"], list)
            else []
        )
        if (
            len(letters) == len(members)
            and all(letter in ("A", "B") for letter in letters)
            # Both sides must be used: an all-A "split" is not a split, and
            # applying it would retire the cluster's verdict for a no-op.
            and set(letters) == {"A", "B"}
        ):
            semantic = {
                "assignments": {chunk_id: letters[i] for i, (chunk_id, _) in enumerate(members)},
                "a_gloss": str(parsed["a_gloss"]),
                "b_gloss": str(parsed["b_gloss"]),
            }
        seed = config.get_int("clustering.seed", 42, minimum=0)
        positions = mechanical_split(texts, embedder, seed, [c for c, _ in members])
        edit["split_proposals"] = {
            "semantic": semantic,  # None when the agent's reply was unusable
            "semantic_raw": split_raw,
            "mechanical": {chunk_id: positions[i] for i, (chunk_id, _) in enumerate(members)},
        }

    for cluster_id in clusters:
        log_reveal(
            session,
            side=side,
            cluster_id=cluster_id,
            revision=revision,
            via="revision_advice",
            now=now,
        )
    session["pending_edit"] = edit
    return edit


def estimate_cost(n_calls: int, approx_input_tokens: int) -> str:
    """One plain line shown before any live agent action."""
    return (
        f"~{n_calls} model call(s), ~{approx_input_tokens:,} input tokens; "
        "typically a few cents at pinned-model prices."
    )
