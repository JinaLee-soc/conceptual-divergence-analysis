"""Live providers and the provider factory."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from cda.config import Config, ConfigError
from cda.llm.base import LLMError, TruncatedReplyError


def _truncated_reply_error(what: str, max_tokens: int) -> TruncatedReplyError:
    """The refusal both providers raise for a reply cut off at the token cap.

    One function so the two providers cannot drift apart, and worded around
    the *number* rather than around ``llm.max_tokens``: that config key does
    not govern every call. The validation reader and split calls pass their
    own caps in code, so a user told to raise ``llm.max_tokens`` would change
    a setting that never applied and hit the identical error on every rerun.
    """
    return TruncatedReplyError(
        f"The {what} reply was cut off at this call's token cap "
        f"({max_tokens}), so it is incomplete and was discarded.\n"
        f"  - if llm.max_tokens in your config is {max_tokens}, that is the "
        "cap that bound this call: raise it (for example to 8000) and rerun\n"
        "  - if it is some other number, this call's cap was chosen in code "
        "(validation readings and split proposals do this), so shorten the "
        "input instead — fewer or shorter passages\n"
        "  - no report was written. Any files this run wrote before the call "
        "(the saved prompt, the cluster diagnostics) are harmless: a rerun "
        "writes a fresh, separately named set."
    )


# Stop reasons that mean the model finished a complete text reply.
_ANTHROPIC_COMPLETE_STOPS = ("end_turn", "stop_sequence")

# OpenAI-compatible servers (vLLM, llama.cpp, Ollama, LM Studio, ...) do not
# agree on the normal-finish value: besides OpenAI's "stop", they send None,
# "eos"/"eos_token", or the Anthropic-style names. All mean a complete reply.
_OPENAI_COMPLETE_STOPS = (None, "stop", "eos", "eos_token", "end_turn", "stop_sequence")


class AnthropicProvider:
    """Live calls to the Anthropic API.

    The model string defaults to the paper's reported model
    (``claude-sonnet-4-6``, pinned in config/defaults.yaml). The API key is
    read from the ANTHROPIC_API_KEY environment variable only — never from a
    config file — so it can't end up in version control.
    """

    name = "anthropic"

    def __init__(self, model: str, timeout: float = 300.0):
        self.model = model
        self._timeout = timeout
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set.\n"
                "  - For a live run: copy .env.example to .env and add your key\n"
                "    (get one at https://console.anthropic.com/), or export it in\n"
                "    your shell.\n"
                "  - For a free, offline run: add --mock-llm (works for the demo\n"
                "    and anything with recorded fixtures)."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - installation problem
            raise LLMError(
                "The 'anthropic' package is not installed. Run: pip install -e ."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def complete(self, *, system: str, user: str, max_tokens: int, temperature: float) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            # The most common live-mode failures — a mistyped or expired key,
            # a rate limit, a dropped connection, a misspelled model name —
            # otherwise surface as a raw SDK traceback, which tells the
            # researcher nothing about what to do.
            raise LLMError(
                f"The Anthropic API call failed ({type(exc).__name__}): {exc}\n"
                "  - authentication errors: check ANTHROPIC_API_KEY in your .env "
                "(typo, or an expired/revoked key)\n"
                "  - rate limits and overloads: wait a moment and rerun\n"
                f"  - not-found errors: check the model name ('{self.model}', "
                "set by llm.model)\n"
                "  - connection errors: check your network — a campus or "
                "corporate proxy, VPN, or firewall between you and "
                "api.anthropic.com produces these too\n"
                "  - free offline alternative: --mock-llm with recorded fixtures"
            ) from exc
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "max_tokens":
            # A truncated reply must never pass as a complete report: the cut
            # happens mid-section, and the downstream audit/grounding would
            # quietly run on the fragment.
            raise _truncated_reply_error("model's", max_tokens)
        if stop_reason not in _ANTHROPIC_COMPLETE_STOPS:
            raise LLMError(
                f"The model ({self.model}) stopped with stop_reason={stop_reason!r} "
                "instead of finishing its reply, so the reply was discarded. "
                "A 'refusal' means the model declined the request; rerun or "
                "inspect the saved prompt."
            )
        parts = [block.text for block in response.content if getattr(block, "text", None)]
        text = "\n".join(parts).strip()
        if not text:
            raise LLMError(
                f"The model ({self.model}) returned an empty response. "
                "Rerun the command; if this persists, check the Anthropic status page."
            )
        return text


class OpenAICompatibleProvider:
    """EXPERIMENTAL: any OpenAI-compatible chat-completions endpoint.

    Exists so the multi-model verification hook (`cda verify`) can use a
    second provider through a config block alone. Configure via:

        llm:
          verification_model: "openai:gpt-4o"        # provider:model
        # plus OPENAI_API_KEY and optionally OPENAI_BASE_URL in the environment
    """

    name = "openai-compatible"

    def __init__(self, model: str, timeout: float = 300.0):
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise LLMError(
                "OPENAI_API_KEY is not set (required for an OpenAI-compatible "
                "verification model). Add it to your .env, or unset "
                "llm.verification_model to skip verification."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "The 'openai' package is not installed. It is only needed for an "
                "OpenAI-compatible verification model: pip install openai"
            ) from exc
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def complete(self, *, system: str, user: str, max_tokens: int, temperature: float) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise LLMError(
                f"The OpenAI-compatible API call failed ({type(exc).__name__}): {exc}\n"
                "  - authentication errors: check OPENAI_API_KEY (a locally served "
                "model still needs some placeholder value)\n"
                f"  - not-found errors: check the model name ('{self.model}') and, "
                "for a local server, OPENAI_BASE_URL\n"
                "  - connection errors: check the endpoint is running and reachable"
            ) from exc
        # Mirrors the Anthropic branch's empty-content guard: an endpoint that
        # returns no choices at all would otherwise raise a bare IndexError.
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError(
                f"The verification model ({self.model}) returned no choices. "
                "Rerun the command; if this persists, check that the endpoint "
                "(OPENAI_BASE_URL) is serving the model you configured."
            )
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            # Same refusal as the Anthropic branch, different sentinel: this
            # API reports the cap hit as finish_reason "length".
            raise _truncated_reply_error("verification model's", max_tokens)
        if finish_reason not in _OPENAI_COMPLETE_STOPS:
            raise LLMError(
                f"The model ({self.model}) stopped with finish_reason={finish_reason!r} "
                "instead of finishing its reply, so the reply was discarded."
            )
        text = (choice.message.content or "").strip()
        if not text:
            raise LLMError(
                f"The verification model ({self.model}) returned an empty "
                "response. Rerun the command; if this persists, check that the "
                "endpoint is healthy (and OPENAI_BASE_URL, for a local server)."
            )
        return text


def spec_model(spec: str) -> str:
    """The bare model name inside a 'model' or 'provider:model' spec.

    The live providers send this name to their API (fixture fingerprints use
    ``fixture_model`` instead). Unknown prefixes are refused here, so a spec
    that cannot run live cannot silently replay either.

    A bare name is Anthropic (the paper's pinned default). Anything that
    looks prefixed but names an unknown provider is a mistake worth
    stopping on: "ollama:llama3:8b" would otherwise be sent to Anthropic as
    a model called "ollama:llama3:8b", which fails far from its cause.
    Local models reach CDA through the OpenAI-compatible endpoint, so
    "openai:" plus OPENAI_BASE_URL is the supported spelling.
    """
    if ":" not in spec:
        return spec
    prefix, name = spec.split(":", 1)
    if prefix not in ("openai", "anthropic"):
        raise LLMError(
            f"Unknown model provider '{prefix}:' in model spec '{spec}'. "
            "Use 'openai:MODEL' (any OpenAI-compatible endpoint, including a "
            "locally served model — set OPENAI_BASE_URL), 'anthropic:MODEL', "
            "or a bare Anthropic model name."
        )
    return name


def fixture_model(spec: str) -> str:
    """The model key used in fixture fingerprints.

    Bare and ``anthropic:`` names fingerprint as the bare model name, so
    existing Anthropic fixtures keep replaying; any other provider keeps its
    prefix so ``openai:X`` can never replay a fixture recorded for Anthropic ``X``.
    """
    model = spec_model(spec)
    return model if ":" not in spec or spec.startswith("anthropic:") else spec


def max_temperature(spec: str) -> float:
    """The largest temperature the spec's API accepts: 2 for OpenAI, 1 for Anthropic."""
    spec_model(spec)  # refuses unknown provider prefixes
    return 2.0 if spec.strip().startswith("openai:") else 1.0


def llm_temperature(config: Config, spec: str) -> float:
    """``llm.temperature``, checked against the range the model's API accepts."""
    maximum = max_temperature(spec)
    temperature = config.get_float("llm.temperature", minimum=0.0)
    if temperature > maximum:
        raise ConfigError(
            f"llm.temperature must be between 0 and {maximum:g} for '{spec}' "
            f"(the API refuses larger values), got {temperature:g}."
        )
    return temperature


def resolve_model_spec(spec: str) -> AnthropicProvider | OpenAICompatibleProvider:
    """Turn 'model' or 'provider:model' into a provider instance.

    Public because the verification hook (`cda verify`) builds its second
    provider from a config string through this function.
    """
    model = spec_model(spec)  # refuses unknown provider prefixes
    if model == "claude-code-session":
        raise LLMError(
            "claude-code-session is an authored replay, not an API model. Use --mock-llm."
        )
    if spec.startswith("openai:"):
        return OpenAICompatibleProvider(model=model)
    return AnthropicProvider(model=model)


def get_provider(
    config: Config,
    *,
    mock: bool,
    fixtures_dir: str | Path | None = None,
    fallback_fixture_models: Sequence[str] = (),
):
    """Build the analysis provider for a run.

    ``mock=True`` returns the fixture-replay provider (no network); otherwise
    a live provider pinned to ``llm.model``. A bare model name is Anthropic
    (the paper's pinned default); a ``provider:model`` spec — the same form
    ``llm.verification_model`` already takes — routes elsewhere, so
    ``llm.model: "openai:gpt-4o"`` runs the analysis against an
    OpenAI-compatible endpoint.

    The model name is part of the fixture fingerprint, so fixtures recorded
    under one model replay only under that same model. That is deliberate: a
    report must never be replayed as if a different model had written it.
    ``fallback_fixture_models`` (mock only) are further model specs whose
    fixtures replay, in order, when none was recorded under ``llm.model``;
    the provider then reports the model it actually replayed.
    """
    model = str(config.get("llm.model"))
    if mock:
        from cda.llm.mock import MockProvider

        if fixtures_dir is None:
            raise LLMError(
                "--mock-llm needs a fixtures directory. The demo passes "
                "demo/fixtures automatically; for your own runs use "
                "--fixtures-dir PATH."
            )
        return MockProvider(
            fixtures_dir=Path(fixtures_dir),
            model=spec_model(model),
            fingerprint_model=fixture_model(model),
            fallbacks=[(spec_model(spec), fixture_model(spec)) for spec in fallback_fixture_models],
        )
    return resolve_model_spec(model)
