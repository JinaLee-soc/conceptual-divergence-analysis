"""LLM provider interface.

All model access goes through one small seam so that every command can run
in two modes: live (AnthropicProvider, needs ANTHROPIC_API_KEY) and mock
(MockProvider, replays committed fixtures — zero network, zero cost, used by
tests and CI). A second, OpenAI-compatible provider can slot in for the
experimental verification hook without changes outside this package.
"""

from cda.llm.base import LLMError, LLMProvider, prompt_fingerprint
from cda.llm.mock import MockProvider, RecordingProvider
from cda.llm.providers import AnthropicProvider, get_provider

__all__ = [
    "AnthropicProvider",
    "LLMError",
    "LLMProvider",
    "MockProvider",
    "RecordingProvider",
    "get_provider",
    "prompt_fingerprint",
]
