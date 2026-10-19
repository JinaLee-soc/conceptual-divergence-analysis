"""Configuration loading.

Defaults ship inside the package (``cda/defaults.yaml``, identical to the
repository's ``config/defaults.yaml`` — a test enforces this). A user config
file overrides defaults key-by-key; CLI flags override both. Secrets never
live in config files: API keys come from the environment (see .env.example).
"""

from __future__ import annotations

import copy
import datetime as dt
import difflib
import math
import os
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed, with a next step."""


_MISSING = object()

_TRUE_WORDS = {"true", "yes", "on", "1"}
_FALSE_WORDS = {"false", "no", "off", "0"}

# Config lists whose entries are mappings with a known shape. The defaults
# cannot describe the shape themselves (`themes: []` is an empty list), so it
# is named here — otherwise `themes: [{tga: x}]` loads without complaint and
# the theme tag is silently dropped from the prompt.
_LIST_ITEM_KEYS: dict[str, tuple[str, ...]] = {"themes": ("tag", "description")}


def _packaged_defaults() -> dict[str, Any]:
    text = resources.files("cda").joinpath("defaults.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError(
            "Packaged defaults.yaml did not parse to a mapping — the installation "
            "is broken. Reinstall with: pip install -e ."
        )
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base`` recursively; override wins.

    A ``None`` override means "use the default", never "replace the default
    with None": otherwise a key left empty in a user YAML (``seed:``,
    ``output_dir: ~``) would override a real default with ``None`` and crash
    far from the config file (``int(None)``) or write output to a directory
    literally named ``None``.
    """
    merged = dict(base)
    for key, value in override.items():
        if value is None and key in merged:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _align_key_types(user: dict, defaults: dict) -> dict:
    """Rewrite user keys that differ from a default key only in type.

    YAML makes ``1:`` an int and ``"1":`` a string, and the ``distance_scale``
    section is keyed by integers. Without this, a quoted numeric key would be
    rejected as unknown with a suggestion identical to what the user wrote
    ("Did you mean 'distance_scale.1'?"), and merging it would produce two keys
    for the same rung. Aligning the type first makes the quoted form work.
    """
    if not isinstance(user, dict) or not isinstance(defaults, dict):
        return user
    by_text = {str(key): key for key in defaults}
    aligned: dict = {}
    for key, value in user.items():
        canonical = key if key in defaults else by_text.get(str(key), key)
        default_value = defaults.get(canonical)
        if isinstance(value, dict) and isinstance(default_value, dict):
            value = _align_key_types(value, default_value)
        aligned[canonical] = value
    return aligned


def _list_item_keys(dotted: str, default_value: Any) -> tuple[str, ...] | None:
    """The allowed keys for mapping entries of a list setting, if any are known."""
    if dotted in _LIST_ITEM_KEYS:
        return _LIST_ITEM_KEYS[dotted]
    if isinstance(default_value, list):
        for entry in default_value:
            if isinstance(entry, dict):
                return tuple(str(k) for k in entry)
    return None


def _unknown_key_problems(user: dict, defaults: dict, prefix: str = "") -> list[str]:
    """Human-readable problems in a user config: unknown keys, wrong shapes.

    Only keys are checked here (values are validated where they are used);
    catching a misspelled *key* matters because a typo like ``metod:`` would
    otherwise silently leave the paper's default in force while the user
    believes they changed it.
    """
    problems: list[str] = []
    for key, value in user.items():
        dotted = f"{prefix}{key}"
        if key not in defaults:
            known = [str(k) for k in defaults]
            hint = ""
            close = difflib.get_close_matches(str(key), known, n=1)
            if close:
                hint = f" Did you mean '{prefix}{close[0]}'?"
            problems.append(
                f"Unknown configuration key '{dotted}'.{hint} "
                f"Known keys at this level: {', '.join(sorted(known))}."
            )
            continue
        default_value = defaults[key]
        if isinstance(default_value, dict) and value is not None and not isinstance(value, dict):
            problems.append(
                f"'{dotted}' must be a mapping (indented `key: value` lines), "
                f"got {value!r}. See config/config.example.yaml."
            )
        elif isinstance(value, dict) and not isinstance(default_value, dict):
            problems.append(
                f"'{dotted}' must be a single value, got a mapping. See config/config.example.yaml."
            )
        elif isinstance(value, dict) and isinstance(default_value, dict):
            problems.extend(_unknown_key_problems(value, default_value, prefix=f"{dotted}."))
        elif isinstance(value, list):
            problems.extend(_unknown_list_item_problems(dotted, value, default_value))
    return problems


def _unknown_list_item_problems(dotted: str, entries: list, default_value: Any) -> list[str]:
    """Check the mapping entries of a list setting against their known shape.

    A list of mappings is a second place a key typo can hide: ``themes:
    - tga: evaluation`` parses as valid YAML, so without this the tag is
    dropped and the prompt says "(no theme tags provided)" instead.
    """
    allowed = _list_item_keys(dotted, default_value)
    if allowed is None:
        return []
    # A one-key-per-entry template stands in for the real defaults: only the
    # key names are known here, and only key names are being checked.
    template = dict.fromkeys(allowed)
    problems: list[str] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            problems.extend(_unknown_key_problems(entry, template, prefix=f"{dotted}[{index}]."))
    return problems


def _iso_date_text(value: dt.date) -> str:
    """A YAML date or datetime as the text the config documents.

    Aware datetimes are converted to UTC first, because every stored
    acquisition time is UTC and the zone would otherwise be dropped.
    """
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(dt.UTC)
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value.isoformat()


def _dates_to_text(value: Any) -> Any:
    """Replace every date/datetime in a config tree with its ISO text.

    YAML loads an unquoted ``2026-05-01`` as a ``datetime.date``; left as is,
    it cannot be hashed into the run manifest (JSON has no date type), and the
    run would fail only after extraction and clustering.
    """
    if isinstance(value, dt.date):
        return _iso_date_text(value)
    if isinstance(value, dict):
        return {key: _dates_to_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dates_to_text(item) for item in value]
    return value


class Config:
    """Read-only view over the merged configuration tree."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        """Fetch ``"section.key"``; raise a helpful error when absent."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                if default is not _MISSING:
                    return default
                raise ConfigError(
                    f"Missing configuration key '{dotted_key}'. "
                    "Check your config file against config/config.example.yaml."
                )
        return node

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name, default={})
        if not isinstance(value, dict):
            raise ConfigError(f"Configuration section '{name}' must be a mapping.")
        return value

    def path(self, dotted_key: str) -> Path:
        value = self.get(dotted_key)
        if value is None or not str(value).strip():
            raise ConfigError(
                f"Configuration key '{dotted_key}' must be a path, got {value!r}. "
                "Set a path, or remove the key to use the default."
            )
        return Path(str(value)).expanduser()

    def get_int(
        self, dotted_key: str, default: Any = _MISSING, *, minimum: int | None = None
    ) -> int:
        """An integer setting, with an error that names the key when it isn't one."""
        value = self.get(dotted_key, default)
        try:
            if isinstance(value, bool):
                raise ValueError
            if isinstance(value, float) and not value.is_integer():
                # A bare int() would truncate `chunk_size_words: 499.7` to 499
                # in silence while the quoted "499.7" raises — the same
                # mistake, two different outcomes. Both raise.
                raise ValueError
            number = int(value)
        except (TypeError, ValueError):
            raise ConfigError(
                f"Configuration key '{dotted_key}' must be an integer, got {value!r}. "
                "See config/config.example.yaml."
            ) from None
        if minimum is not None and number < minimum:
            raise ConfigError(
                f"Configuration key '{dotted_key}' must be >= {minimum}, got {number}."
            )
        return number

    def get_float(
        self,
        dotted_key: str,
        default: Any = _MISSING,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        """A numeric setting, with an error that names the key when it isn't one."""
        value = self.get(dotted_key, default)
        try:
            if isinstance(value, bool):
                raise ValueError
            number = float(value)
        except (TypeError, ValueError):
            raise ConfigError(
                f"Configuration key '{dotted_key}' must be a number, got {value!r}. "
                "See config/config.example.yaml."
            ) from None
        if not math.isfinite(number):
            # Every comparison with NaN is False, so `.nan` slipped past both
            # bound checks below and disabled the threshold it was meant to
            # set — saturation never declared, or a permanent "no flags" that
            # reads as a real finding. Infinities are equally meaningless here.
            raise ConfigError(
                f"Configuration key '{dotted_key}' must be a finite number, got {value!r}. "
                "See config/config.example.yaml."
            )
        if minimum is not None and number < minimum:
            raise ConfigError(
                f"Configuration key '{dotted_key}' must be >= {minimum}, got {number}."
            )
        if maximum is not None and number > maximum:
            raise ConfigError(
                f"Configuration key '{dotted_key}' must be <= {maximum}, got {number}."
            )
        return number

    def get_bool(self, dotted_key: str, default: Any = _MISSING) -> bool:
        """A true/false setting; quoted strings like 'false' parse by meaning.

        ``bool("false")`` is True in Python, so a quoted YAML boolean would
        silently invert the setting if read with a bare ``bool(...)`` cast.
        The integers 0 and 1 are accepted too: ``figure.enabled: 1`` is
        unambiguous about what it means, and rejecting it would break such
        configs.
        """
        value = self.get(dotted_key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            word = value.strip().lower()
            if word in _TRUE_WORDS:
                return True
            if word in _FALSE_WORDS:
                return False
        raise ConfigError(f"Configuration key '{dotted_key}' must be true or false, got {value!r}.")

    def get_list(self, dotted_key: str, default: Any = _MISSING) -> list:
        """A list setting; a single scalar wraps into a one-element list.

        ``readers: claude-sonnet-4-6`` (a YAML string where a list was meant)
        would otherwise iterate into single characters downstream.
        """
        value = self.get(dotted_key, default)
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, dict):
            raise ConfigError(
                f"Configuration key '{dotted_key}' must be a list, got a mapping. "
                "See config/config.example.yaml."
            )
        return [value]


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Build the effective configuration.

    Precedence (lowest to highest): packaged defaults → user config file →
    environment (CDA_VERIFICATION_MODEL) → explicit overrides (CLI flags).
    """
    data = copy.deepcopy(_packaged_defaults())

    if config_path is not None:
        if not str(config_path).strip():
            raise ConfigError(
                "--config got an empty path (did an unset shell variable expand "
                "to nothing?). Pass a real file, or drop the flag to use defaults."
            )
        path = Path(config_path).expanduser()
        if not path.is_file():
            raise ConfigError(
                f"Config file not found: {path}\n"
                "Copy config/config.example.yaml, edit it, and pass it with --config."
            )
        try:
            text = path.read_text(encoding="utf-8")
            if "\x00" in text:
                # UTF-16 of ASCII is byte-valid UTF-8 with a NUL between every
                # character, so it never raises below — it reaches the YAML
                # parser as gibberish and the error blames the syntax.
                raise ConfigError(
                    f"{path} looks like UTF-16, not UTF-8 (it contains NUL bytes). "
                    "Re-save it as UTF-8 (plain text) and rerun."
                )
            user_data = yaml.safe_load(text) or {}
        except UnicodeDecodeError as exc:
            raise ConfigError(
                f"{path} is not UTF-8 encoded ({exc}). "
                "Re-save the file as UTF-8 (plain text) and rerun."
            ) from exc
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Could not parse {path} as YAML: {exc}\n"
                "Fix the syntax (straight quotes only) and rerun."
            ) from exc
        if not isinstance(user_data, dict):
            raise ConfigError(f"{path} must contain a YAML mapping at the top level.")
        user_data = _align_key_types(user_data, data)
        problems = _unknown_key_problems(user_data, data)
        if problems:
            raise ConfigError(f"{path} has configuration problems:\n- " + "\n- ".join(problems))
        data = _deep_merge(data, user_data)

    env_verification = os.environ.get("CDA_VERIFICATION_MODEL", "").strip()
    if env_verification:
        data = _deep_merge(data, {"llm": {"verification_model": env_verification}})

    if overrides:
        data = _deep_merge(data, overrides)

    return Config(_dates_to_text(data))
