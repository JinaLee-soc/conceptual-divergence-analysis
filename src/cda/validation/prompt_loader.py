"""Loading the versioned validation-task prompts.

Parallel to ``registers/prompt_loader.py`` but deliberately separate: that
loader's first-fenced-block semantic is part of the paper's provenance and
must not grow options. Validation prompts live in
``prompts/validation_tasks_v1.md`` as named blocks (a ``## block: <name>``
heading directly above each fence), so every agent utterance in a validation
session is versioned and inspectable.
"""

from __future__ import annotations

import re
from importlib import resources

from cda.config import Config, ConfigError

PROMPT_FILE = "validation_tasks_v1.md"
REQUIRED_BLOCKS = ("cluster_reading_system", "revision_advice_system", "semantic_split_system")

_NAMED_FENCE = re.compile(
    r"^## block: (?P<name>[a-z_]+)\s*\n+```[^\n]*\n(?P<body>.*?)^```",
    re.DOTALL | re.MULTILINE,
)


def load_validation_prompts(config: Config) -> dict[str, str]:
    """Return {block name: prompt text} for every required block.

    ``config`` is accepted for symmetry with ``load_system_prompt`` and future
    overrides; today the packaged file is always used.
    """
    del config  # no override path yet; the packaged versioned file is the truth
    text = resources.files("cda").joinpath(f"prompts/{PROMPT_FILE}").read_text(encoding="utf-8")
    blocks = {m.group("name"): m.group("body").strip() for m in _NAMED_FENCE.finditer(text)}
    missing = [name for name in REQUIRED_BLOCKS if not blocks.get(name)]
    if missing:
        raise ConfigError(
            f"prompts/{PROMPT_FILE} is missing required block(s): {', '.join(missing)}. "
            "Each block needs a '## block: <name>' heading directly above a fenced "
            "prompt. Restore the file from git."
        )
    return blocks
