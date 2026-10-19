"""Fixture replay (MockProvider) and fixture recording (RecordingProvider).

Fixtures are JSON files named by the prompt fingerprint. They are committed
under demo/fixtures/ so the demo, tests, and CI run with zero network and
zero cost. On a cache miss the MockProvider fails loudly with instructions
rather than silently degrading — a wrong-but-plausible reply would be worse
than an error.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from cda.llm.base import (
    FixtureCorruptError,
    FixtureMissError,
    LLMError,
    TruncatedReplyError,
    prompt_fingerprint,
)
from cda.outputs import write_text_atomic, write_text_immutable


def has_recorded_fixtures(fixtures_dir: Path | str | None) -> bool:
    """Could ``--mock-llm`` replay anything at all from this directory?

    Cheap enough to call before any work starts, which is the point: the
    provider only discovers an empty fixtures directory when the first
    prompt is assembled, i.e. after extraction and clustering have already
    spent minutes on a real corpus.
    """
    if fixtures_dir is None:
        return False
    directory = Path(fixtures_dir)
    return directory.is_dir() and any(directory.glob("*.json"))


def require_recorded_fixtures(fixtures_dir: Path | str | None, *, what: str) -> None:
    """Refuse an unplayable ``--mock-llm`` run before the expensive stages.

    ``what`` names the command in the message ("cda analyze", "cda verify")
    so the remedy the user is given is the one for the run they started.
    """
    if has_recorded_fixtures(fixtures_dir):
        return
    raise LLMError(
        f"--mock-llm has nothing to replay for {what}: no recorded responses "
        f"under {fixtures_dir}.\n"
        "What to do next:\n"
        "  - run once live (set ANTHROPIC_API_KEY, drop --mock-llm) and add\n"
        "    --record-fixtures, which saves replayable responses for later\n"
        "  - or point at an existing set with --fixtures-dir PATH\n"
        "  - or try the committed demo: cda demo --term novelty --mock-llm"
    )


class MockProvider:
    """Replays a recorded response for an identical assembled prompt."""

    name = "mock"

    def __init__(
        self,
        fixtures_dir: Path,
        model: str,
        *,
        fingerprint_model: str | None = None,
        fallbacks: Sequence[tuple[str, str]] = (),
    ):
        """``fingerprint_model`` is the provider-qualified key (see
        ``providers.fixture_model``); it equals ``model`` for Anthropic specs.

        ``fallbacks`` are further (model, fingerprint_model) pairs tried in
        order when nothing was recorded under ``model``. After a replay,
        ``model`` names the model whose fixture was replayed, so the report
        header never credits a reply to a model that did not write it.
        """
        self.model = model
        self.fingerprint_model = fingerprint_model or model
        self._candidates = [(self.model, self.fingerprint_model)] + [
            pair for pair in fallbacks if pair[1] != self.fingerprint_model
        ]
        self._dir = Path(fixtures_dir)

    def _locate(self, system: str, user: str) -> tuple[str, str, Path] | None:
        """(model, fingerprint model, path) of the first recorded candidate."""
        for model, fingerprint_model in self._candidates:
            path = self._dir / f"{prompt_fingerprint(fingerprint_model, system, user)}.json"
            if path.is_file():
                return model, fingerprint_model, path
        return None

    def complete(self, *, system: str, user: str, max_tokens: int, temperature: float) -> str:
        located = self._locate(system, user)
        if located is None:
            key = prompt_fingerprint(self.fingerprint_model, system, user)
            path = self._dir / f"{key}.json"
            fallback_note = "".join(
                f"  also tried : {self._dir / (prompt_fingerprint(fp, system, user) + '.json')}"
                f" (model {model})\n"
                for model, fp in self._candidates[1:]
            )
            available = (
                sorted(p.stem for p in self._dir.glob("*.json")) if self._dir.is_dir() else []
            )
            raise FixtureMissError(
                "Mock fixture not found for this prompt.\n"
                f"  looked for : {path}\n"
                f"{fallback_note}"
                f"  fingerprint: {key} (sha256 of model + system + user prompt)\n"
                f"  available  : {', '.join(available) if available else '(none)'}\n"
                "Why this happens: the assembled prompt differs from the one the\n"
                "fixture was recorded for — a different term, corpus, config\n"
                "value, or prompt version all change the fingerprint.\n"
                "What to do next:\n"
                "  - For the demo, run exactly: cda demo --term novelty --mock-llm\n"
                "  - For your own corpus/terms, run live (set ANTHROPIC_API_KEY,\n"
                "    drop --mock-llm), optionally adding --record-fixtures to\n"
                "    save a fixture for future offline replay."
            )
        model, fingerprint_model, path = located
        if fingerprint_model != self.fingerprint_model:
            print(
                f"[fixture] nothing recorded under {self.model} for this prompt; "
                f"replaying the {model} fixture {path.name}"
            )
        self.model, self.fingerprint_model = model, fingerprint_model
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("truncated"):
                # The live call this fixture was recorded from hit its token
                # cap, so replay reproduces the same refusal instead of a
                # response. This keeps a --mock-llm rerun on the same path the
                # live session took (validation tolerates it as one unusable
                # reading) — without the marker, that cluster would be a
                # permanent fixture hole that aborts every replay with advice
                # ("record on a live run") the user has already followed.
                stored = data.get("error")
                detail = (
                    stored
                    if isinstance(stored, str) and stored.strip()
                    else "The recorded live reply was cut off at its token cap."
                )
                raise TruncatedReplyError(
                    f"Replaying a recorded truncation ({path.name}): the live "
                    "call this fixture was recorded from was cut off at its "
                    "token cap, so the same configuration would truncate again "
                    f"live.\n{detail}"
                )
            response = data["response"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            # TypeError covers a fixture whose top level is valid JSON but
            # not an object (a hand-mangled list, say) — data["response"]
            # raises it, and it would otherwise escape as a raw traceback.
            # UnicodeDecodeError covers a recording interrupted *inside* a
            # multi-byte character: the read itself fails before JSON is ever
            # parsed, so it would escape as a raw codec error that
            # replay-then-live cannot recognize as healable — every resume
            # would die on the same file with no next step.
            raise FixtureCorruptError(
                f"Fixture file {path} is corrupt ({exc}). Restore it from git "
                "(committed demo fixtures), re-record it with --record-fixtures "
                "on a live run, or delete the file so the next live run "
                "records a fresh one."
            ) from exc
        if not isinstance(response, str) or not response.strip():
            raise FixtureCorruptError(
                f"Fixture file {path} has an empty 'response' field. Delete the "
                "file (a live run records a fresh one) or restore it from git."
            )
        # Config-drift note, not an error: the fingerprint ignores max_tokens
        # on purpose (fixtures recorded before the field existed must keep
        # replaying), but a fixture recorded under a larger cap can replay a
        # reply the same live configuration would refuse as truncated.
        recorded_cap = data.get("max_tokens")
        if isinstance(recorded_cap, int) and recorded_cap > max_tokens:
            print(
                f"[fixture] note: {path.name} was recorded with "
                f"max_tokens={recorded_cap} but this run allows {max_tokens} — "
                "a live run at this setting might truncate where the replay "
                "does not."
            )
        return response


class RecordingProvider:
    """Wraps a live provider and saves each response as a replayable fixture."""

    name = "recording"

    def __init__(
        self,
        inner,
        fixtures_dir: Path,
        *,
        fingerprint_model: str | None = None,
        overwrite: bool = False,
    ):
        """Record ``inner``'s replies under ``fixtures_dir``.

        A fixture that already holds a replayable response is never replaced
        unless ``overwrite`` is set; the refusal comes before the paid call.
        Corrupt files and recorded truncations are re-recordable, which is how
        replay-then-live heals them.
        """
        self._inner = inner
        self.model = inner.model
        self.fingerprint_model = fingerprint_model or inner.model
        self._dir = Path(fixtures_dir)
        self._overwrite = overwrite

    def complete(self, *, system: str, user: str, max_tokens: int, temperature: float) -> str:
        path = self._dir / f"{prompt_fingerprint(self.fingerprint_model, system, user)}.json"
        if not self._overwrite and _is_replayable(path):
            raise LLMError(
                f"A replayable fixture is already recorded at {path}; refusing to "
                "replace it. Replay it with --mock-llm, delete it, or record with "
                "overwrite enabled."
            )
        payload = {
            "_note": (
                "Cached LLM response for --mock-llm replay. Recorded from a live "
                "run; keyed by sha256(model + system + user prompt)."
            ),
            "model": self.model,
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "user_prompt_preview": user[:400],
            "user_prompt_chars": len(user),
            # The call's cap and temperature travel with the recording so
            # replay can notice config drift (see MockProvider). Additive:
            # fixtures recorded before these fields existed replay unchanged.
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            response = self._inner.complete(
                system=system, user=user, max_tokens=max_tokens, temperature=temperature
            )
        except TruncatedReplyError as exc:
            # A cut-off reply is still an outcome worth recording. The live
            # session tolerates it (validation counts the reading as unusable
            # and keeps going), but recording only successes would leave that
            # cluster a permanent fixture hole: every later --mock-llm replay
            # would abort there with advice — "record on a live run" — the
            # user has already followed. Recording the refusal (marker instead of a
            # "response") lets replay reproduce the live behavior exactly.
            payload["_note"] = (
                "Recorded truncation for --mock-llm replay: the live call was "
                "cut off at its token cap, so replay raises the same refusal "
                "instead of inventing a response. Keyed like any fixture; a "
                "successful re-recording replaces this file."
            )
            payload["truncated"] = True
            payload["error"] = str(exc)
            _write_fixture(path, payload, label="recorded truncation")
            raise
        payload["response"] = response
        _write_fixture(path, payload, label="recorded")
        return response


def _is_replayable(path: Path) -> bool:
    """True when ``path`` holds a usable recorded response (not corrupt, not a truncation)."""
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(data, dict)
        and not data.get("truncated")
        and isinstance(data.get("response"), str)
        and bool(data["response"].strip())
    )


def _write_fixture(path: Path, payload: dict, *, label: str) -> None:
    """Create the fixture, or replace one the caller has been allowed to replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if path.exists():
        write_text_atomic(path, text)
    else:
        write_text_immutable(path, text)
    print(f"[fixture] {label} {path}")
