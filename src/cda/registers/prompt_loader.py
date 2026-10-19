"""Loading the versioned register-assignment prompt.

The system prompt lives in prompts/register_assignment_v5.md (the paper's
appendix template plus the published analysis instructions) — never inline in
code, so the prompt that shaped a report is always inspectable and versioned.
The packaged copy is used unless config points at another file. (Earlier
versions are retained alongside it as frozen prior versions; each version's
deviations are listed at the top of its own file.)
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

from cda.config import Config, ConfigError

_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)


def _prompt_text(config: Config) -> tuple[str, str]:
    """Return (prompt file text, source description)."""
    configured = config.get("paths.prompt_file", default=None)
    if configured:
        path = Path(str(configured)).expanduser()
        if not path.is_file():
            raise ConfigError(
                f"paths.prompt_file points at {path}, which does not exist. "
                "Unset it to use the packaged prompt, or fix the path."
            )
        return path.read_text(encoding="utf-8"), str(path)
    packaged = resources.files("cda").joinpath("prompts/register_assignment_v5.md")
    return packaged.read_text(encoding="utf-8"), "packaged register_assignment_v5.md"


def load_system_prompt(config: Config) -> str:
    """The system prompt is the first fenced block of the prompt file.

    (The second fenced block is the paper's user-prompt template, which
    documents the payload the pipeline assembles programmatically.)
    """
    text, source = _prompt_text(config)
    blocks = _FENCE.findall(text)
    if not blocks:
        raise ConfigError(
            f"No fenced system prompt found in {source}. The prompt file must "
            "contain the system prompt as its first ``` block."
        )
    system = blocks[0].strip()
    if not system:
        raise ConfigError(f"The system prompt block in {source} is empty.")
    return system
