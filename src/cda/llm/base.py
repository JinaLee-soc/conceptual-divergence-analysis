"""Provider protocol and fixture fingerprinting."""

from __future__ import annotations

import hashlib
from typing import Protocol


class LLMError(RuntimeError):
    """Raised for provider problems; the message always says what to do next."""


class FixtureMissError(LLMError):
    """No recorded response matches this prompt's fingerprint.

    Its own type because callers need to recognize it without matching on
    message text: the GUI explains this case differently from a real failure
    (a fixture recorded on another CPU architecture is nobody's mistake), and
    a reworded message must not silently turn that explanation back into a
    raw error dump.
    """


class FixtureCorruptError(LLMError):
    """A fixture file exists but cannot be replayed (truncated/invalid JSON).

    Distinct from a miss so replay-then-live can treat a user-recorded
    fixture that was interrupted mid-write as re-recordable: falling through
    to the live call (which re-records over the bad file) un-wedges a
    validation resume that would otherwise fail on every rerun with advice
    that never applied to git-ignored, user-recorded fixtures.
    """


class TruncatedReplyError(LLMError):
    """The model stopped because it hit this call's token cap.

    Its own type for the same reason as the two above: the validation loop
    treats a cut-off reading as one unusable reading rather than a dead
    session, and deciding that by matching words in the message would break
    the next time the wording is improved.
    """


class LLMProvider(Protocol):
    """The one interface every provider implements.

    ``complete`` takes the fully assembled system and user prompts and returns
    the model's text. Nothing else about a provider is visible to the
    pipeline, which is what makes mock replay and multi-model verification
    possible.
    """

    name: str
    model: str

    def complete(self, *, system: str, user: str, max_tokens: int, temperature: float) -> str: ...


def prompt_fingerprint(model: str, system: str, user: str) -> str:
    """Stable key for a fully assembled prompt.

    Used to name fixture files: the same corpus, config, and term always
    assemble the same prompt, so mock replay is deterministic. Sixteen hex
    characters keep filenames short while making collisions implausible.
    """
    digest = hashlib.sha256()
    for part in (model, system, user):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]
