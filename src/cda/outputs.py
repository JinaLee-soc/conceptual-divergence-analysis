"""Shared helpers for writing result files on disk.

Two behaviors, shared with validation sessions (see
``validation/session.py``): ``-1``, ``-2`` collision suffixes when two runs
start within the same second, and atomic (write-beside-then-rename)
persistence. Without them, same-second runs would overwrite each other's
reports, prompts, diagnostics, and fixtures, and a crash mid-write would
leave a truncated file that downstream consumers (verify, the GUI) read as a
complete run.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from pathlib import Path

# Suffix of the empty marker file that claims a run basename. Hidden (leading
# dot) and not ``.md``/``.json``/``.png``, so it can never match the
# ``drift_*.md`` globs the GUI and ``cda verify`` scan.
_CLAIM_SUFFIX = ".claim"

# New files are opened with this mode and the kernel applies the process
# umask, so artifacts get the permissions a plain ``open(path, "w")`` would
# give them (mkstemp's 0600 would hide reports from a docker host user).
_NEW_FILE_MODE = 0o666


def create_temp(directory: Path, name: str) -> tuple[int, str]:
    """Open a new hidden temp file beside ``name`` in ``directory``; returns (fd, path)."""
    for _ in range(100):
        temporary = os.path.join(directory, f".{name}.{uuid.uuid4().hex}.tmp")
        try:
            return os.open(
                temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, _NEW_FILE_MODE
            ), temporary
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a temporary file beside {name} in {directory}.")


def unique_run_basename(out_dir: Path, base: str, tails: Sequence[str]) -> str:
    """Claim a basename under which one run may write ``base + tail`` files.

    Returns ``base`` itself when free, else ``base-1``, ``base-2``, … so the
    files of one run (report, prompt, diagnostics, figure) always share one
    basename and never replace an earlier run's files.

    Merely *checking* that the names are free is not enough. The timestamp in
    ``base`` resolves to the second, and the GUI and the CLI can both start an
    analysis in the same second: both would see the bare name free, and the
    later ``os.replace`` in :func:`write_text_atomic` would then silently take
    over the earlier run's files. So the name is *claimed*, in the same spirit
    as ``create_session``'s ``mkdir(..., exist_ok=False)``: an empty marker
    file is created with ``O_CREAT | O_EXCL``, which exactly one process can
    win. The loser moves on to the next suffix.

    The marker is left behind on purpose — it is what keeps the name claimed
    while the run is still writing its first file. A run that then dies leaves
    only that empty hidden file: it blocks nothing later, because the next run
    carries a later timestamp and so a different ``base``.
    """
    out_dir = Path(out_dir)
    for n in range(1000):
        candidate = base if n == 0 else f"{base}-{n}"
        if any((out_dir / f"{candidate}{tail}").exists() for tail in tails):
            continue
        marker = out_dir / f".{candidate}{_CLAIM_SUFFIX}"
        try:
            handle = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, _NEW_FILE_MODE)
        except FileExistsError:
            continue  # another process claimed this name a moment ago
        os.close(handle)
        return candidate
    raise RuntimeError(
        f"Could not find a free filename for '{base}' in {out_dir} after 1000 "
        "attempts — clean up the output directory."
    )


def _fsync(path: str) -> None:
    handle = os.open(path, os.O_WRONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def replace_atomic(temporary: str | Path, path: Path) -> None:
    """Install a fully-written temp file as ``path``, replacing any old file.

    The bytes reach the disk before the rename, so even a power loss cannot
    leave the final name pointing at an empty or half-written file.
    """
    _fsync(str(temporary))
    os.replace(temporary, path)  # atomic within a filesystem


def install_new(temporary: str | Path, path: Path) -> None:
    """Install a fully-written temp file as ``path``; refuse if ``path`` exists."""
    _fsync(str(temporary))
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}") from None


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` so a reader sees the old file or the new one, never a part.

    A direct ``write_text`` truncates first: a crash or full disk mid-write
    leaves a valid-looking fragment, and nothing downstream validates report
    or fixture completeness.
    """
    path = Path(path)
    handle, temporary = create_temp(path.parent, path.name)
    try:
        with os.fdopen(handle, "w", encoding=encoding) as fh:
            fh.write(text)
        replace_atomic(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_text_immutable(path: Path, text: str) -> None:
    """Atomically create an artifact, or accept only an identical replay."""
    path = Path(path)
    payload = text.encode("utf-8")
    handle, temporary = create_temp(path.parent, path.name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        try:
            install_new(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(f"Refusing to change existing run artifact: {path}") from None
    finally:
        os.unlink(temporary)
