"""PDF-directory → ChromaDB ingestion.

Mirrors the corpus construction described in the paper (§3.1): full-text PDFs
are chunked into overlapping word windows, embedded with the pinned
sentence-embedding model, and stored in a local persistent ChromaDB
collection with cosine distance. Zotero metadata (from a CSV export) is
resolved per file at ingest time, so analysis runs never need Zotero.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from cda.config import Config, ConfigError
from cda.corpus.embeddings import Embedder
from cda.corpus.zotero import (
    build_doi_index,
    build_filename_index,
    build_title_index,
    count_ambiguous,
    load_zotero_csv,
    match_by_content,
    match_by_filename,
    normalize_key,
    record_fields,
)
from cda.extraction.windows import SENTENCE_SEGMENTER, sentence_spans
from cda.provenance import digest, file_digest

# Toy-corpus support: `cda demo` ingests JSONL abstracts instead of PDFs, via
# ingest_records() below. Real corpora go through ingest_pdf_dir().

# Stored fields that must equal the current inputs for a source to be reused.
_STALE_REASONS = {
    "chunk_size_words": "the chunk size changed or was not recorded",
    "chunk_overlap_words": "the chunk overlap changed or was not recorded",
    "record_metadata_sha256": "the record metadata changed or was not recorded",
    "source_file_sha256": "the PDF changed since it was ingested",
    "source_text_sha256": "the record text changed since it was ingested",
    "sentence_segmenter": f"it was split into sentences by another segmenter (now {SENTENCE_SEGMENTER})",
}


class IngestRollbackError(RuntimeError):
    """A failed store mutation could not restore the previous source."""


def _refuse_source_kind(collection, sources: list[str], kind: str) -> None:
    """PDFs and supplied records share a store, but cannot take over each other's identity."""
    for source in sources:
        existing = collection.get(where={"pdf_relpath": source}, include=["metadatas"])
        for metadata in existing.get("metadatas") or []:
            metadata = metadata or {}
            other = (
                "pdf"
                if "source_file_sha256" in metadata
                else ("record" if "record_metadata_sha256" in metadata else None)
            )
            if other and other != kind:
                raise ValueError(
                    f"Source {source!r} already belongs to a {other} input; a {kind} input "
                    "cannot replace it. Rename the source or use a separate corpus directory."
                )


def _store_directory(config: Config) -> Path:
    """Resolve both supported corpus layouts to the same physical store."""
    corpus_dir = config.path("paths.corpus_dir")
    # Standard layout: the store lives at <corpus_dir>/chroma. But when
    # corpus_dir itself IS a ChromaDB store (it contains chroma.sqlite3 —
    # e.g. pointing CDA at a pre-existing store), use it directly.
    if (corpus_dir / "chroma.sqlite3").is_file():
        return corpus_dir.resolve()
    return (corpus_dir / "chroma").resolve()


def get_collection(config: Config, *, create: bool = False):
    """Open (or create) the CDA collection, with a clear next step on failure."""
    import chromadb
    from chromadb.config import Settings

    chroma_dir = _store_directory(config)
    collection_name = str(config.get("corpus.collection"))

    # The store file, not the directory: a refused lock leaves an empty
    # directory holding only .ingest.guard, which is not a corpus.
    if not create and not (chroma_dir / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"No corpus found at {chroma_dir}.\n"
            "Build one first:  cda ingest --pdf-dir PATH [--zotero-csv EXPORT.csv]\n"
            "Or try the demo:  cda demo --term novelty --mock-llm"
        )
    chroma_dir.mkdir(parents=True, exist_ok=True)
    # Telemetry off: CDA's privacy contract is that nothing leaves the machine
    # except prompts to the Anthropic API and queries to OpenAlex.
    client = chromadb.PersistentClient(
        path=str(chroma_dir), settings=Settings(anonymized_telemetry=False)
    )
    if create:
        return client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
    try:
        return client.get_collection(name=collection_name)
    except Exception as exc:
        raise FileNotFoundError(
            f"Collection '{collection_name}' not found in {chroma_dir} ({exc}).\n"
            "Run `cda ingest` first, or point paths.corpus_dir at the right place."
        ) from exc


def extract_pdf_text(path: Path) -> str:
    """Plain-text extraction with PyMuPDF.

    An unreadable file raises: ``RuntimeError`` for a damaged one, ``ValueError``
    for an encrypted one, and possibly other types; callers treat any
    exception as "this file is unreadable".
    """
    import fitz  # pymupdf

    with fitz.open(str(path)) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def file_acquisition_date(path: Path) -> str:
    """When this file arrived on disk, as UTC ``YYYY-MM-DD HH:MM:SS``.

    Stored as ``date_added_fs``: the acquisition clock that needs no Zotero
    export and survives a rebuilt Zotero library. Birth time where the
    platform records one, modification time otherwise.
    """
    stat = path.stat()
    stamp = getattr(stat, "st_birthtime", None) or stat.st_mtime
    return datetime.fromtimestamp(stamp, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


_MIN_TAIL_WORDS = 20  # a shorter final window is mostly the previous chunk's overlap


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Overlapping word windows — the ingestion chunking from the paper.

    Invariant: every word is in at least one window. A final window shorter
    than the tail floor is dropped only when the previous window already
    covers all of its words (it would repeat the overlap and nothing else).
    """
    words = text.split()
    if not words:
        return []
    if size <= overlap:
        raise ValueError("corpus.chunk_size_words must be larger than chunk_overlap_words")
    chunks = []
    step = size - overlap
    tail_floor = min(_MIN_TAIL_WORDS, size)
    for start in range(0, len(words), step):
        window = words[start : start + size]
        covered_by_previous = start + overlap >= len(words)
        if chunks and covered_by_previous and len(window) < tail_floor:
            break
        chunks.append(" ".join(window))
    return chunks


def sentence_context_records(
    text: str, size: int, overlap: int, n_chunks: int
) -> list[tuple[int, str, list[int]]]:
    """(first sentence position, complete sentences, their lengths) per word window.

    The source is segmented before chunking; word offsets then map each
    window back to its complete sentences, including sentences longer than
    the window. Each sentence covers the words its characters touch; two
    sentences share a word only when a CJK full stop ends one inside it. The
    context is the normalized text from the first sentence to the last, so
    neighbours are separated by one space or, after a CJK full stop, by
    nothing. Their character lengths are stored so analysis can recover
    exactly these sentences: re-segmenting a chunk's context alone gives
    different boundaries than segmenting the whole document.
    """
    normalized, spans = sentence_spans(text)
    spaces = [i for i, ch in enumerate(normalized) if ch == " "]
    # Word index of a character = the spaces before it (words are space-separated).
    starts = [bisect_left(spaces, a) for a, _ in spans]
    ends = [bisect_left(spaces, b - 1) + 1 for _, b in spans]
    n_words = len(spaces) + 1 if normalized else 0
    contexts = []
    for i in range(n_chunks):
        start = i * (size - overlap)
        end = min(start + size, n_words)
        first = bisect_right(ends, start)
        last = bisect_left(starts, end)
        window = spans[first:last]
        context = normalized[window[0][0] : window[-1][1]] if window else ""
        contexts.append((first, context, [b - a for a, b in window]))
    return contexts


def _chunk_settings(config: Config) -> tuple[int, int]:
    """Chunk size and overlap from config; sizes below the tail floor are refused."""
    size = config.get_int("corpus.chunk_size_words", minimum=1)
    if size < _MIN_TAIL_WORDS:
        raise ConfigError(
            f"corpus.chunk_size_words is {size}, but windows shorter than "
            f"{_MIN_TAIL_WORDS} words are dropped as overlap tails, so a size "
            f"this small cannot chunk documents at all — each document would "
            f"keep only its first {size} words. Set chunk_size_words to at "
            f"least {_MIN_TAIL_WORDS} (the default is 500)."
        )
    return size, config.get_int("corpus.chunk_overlap_words", minimum=0)


_INGEST_LOCK_TIMEOUT_SECONDS = 120
_INGEST_LOCK_POLL_SECONDS = 0.5


class CorpusLockError(RuntimeError):
    """Another ingest or retrieval held the corpus lock past the timeout."""


def _process_alive(pid: int) -> bool:
    """Whether a process with this id exists; one we may not signal counts as alive."""
    if sys.platform == "win32":
        # os.kill on Windows terminates the target for any signal but the two
        # console events, so liveness is read from a query-only handle.
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: exists, not ours
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_holder(lock_path: Path) -> tuple[int | None, str, bool]:
    """(pid, content, stale) read from a lock file: the holder's pid, and whether it is gone.

    A file without a readable pid is stale once it is older than the timeout:
    a live holder writes its pid right after creating the file.
    """
    try:
        content = lock_path.read_text(encoding="utf-8", errors="replace").strip()
        modified = lock_path.stat().st_mtime
    except OSError:
        return None, "", False
    if content.isascii() and content.isdecimal() and len(content) <= 10:
        pid = int(content)
        if pid > 0:
            try:
                return pid, content, not _process_alive(pid)
            except OverflowError:
                pass  # outside this platform's PID range: treat as a malformed record
    return None, content, time.time() - modified > _INGEST_LOCK_TIMEOUT_SECONDS


def _unreadable_lock_age(lock_path: Path) -> float | None:
    """Seconds since a lock entry that cannot be read as a file was last changed.

    ``None`` when the entry is readable (a holder's PID record, however
    malformed, is judged by ``_lock_holder``) or no longer exists. A
    directory, a dangling link, or a file this user may not read is never a
    holder's record: a holder writes a plain file that it created itself.
    """
    try:
        lock_path.read_bytes()
        return None
    except OSError:
        pass
    try:
        return time.time() - os.lstat(lock_path).st_mtime
    except OSError:
        return None


def _discard_unreadable_lock(lock_path: Path) -> str:
    """Move an unreadable lock entry aside and remove it; returns what was done.

    The caller must hold the OS guard, so no holder can be writing the
    entry. A directory is moved aside before removal so the pathname is
    free at once; one that cannot be removed stays aside under its new
    name, which the note reports. An entry that cannot even be moved raises
    ``CorpusLockError`` naming it.
    """
    aside = lock_path.with_name(f"{lock_path.name}.stale-{os.getpid()}-{time.monotonic_ns()}")
    try:
        os.rename(lock_path, aside)
    except OSError as exc:
        raise CorpusLockError(
            f"{lock_path} is not a lock file CDA can read (a directory, a dangling link, "
            f"or a file without read permission) and could not be moved aside "
            f"({type(exc).__name__}: {exc}). Remove it by hand and retry; leave "
            ".ingest.guard in place."
        ) from exc
    try:
        if aside.is_dir() and not aside.is_symlink():
            os.rmdir(aside)
        else:
            os.unlink(aside)
    except OSError:
        return f"moved it aside to {aside.name}; remove that entry by hand"
    return "removed it"


def _reclaim_stale_lock(lock_path: Path, expected: str) -> bool:
    """Remove a lock file whose holder is gone; returns whether this caller removed it.

    The caller must hold the persistent OS guard across the stale check,
    this rename, acquisition, and release. Renaming alone is insufficient:
    another waiter could otherwise acquire the temporarily absent pathname.
    """
    claimed = lock_path.with_name(f"{lock_path.name}.stale-{os.getpid()}-{time.monotonic_ns()}")
    try:
        os.rename(lock_path, claimed)
    except OSError:
        return False
    try:
        content = claimed.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        content = expected
    if content != expected:
        os.replace(claimed, lock_path)
        return False
    claimed.unlink(missing_ok=True)
    return True


def _wait_for_corpus_lock(lock_path, deadline, progress, announced):
    """Wait one poll interval for the holder; at the deadline, name what is actually held.

    A holder of the PID file is named by that file. When no PID file exists,
    a live process holds only the OS lock on ``.ingest.guard`` (a holder
    between taking the guard and writing its record, or another program
    locking the guard); there is nothing on disk to remove, so none is
    advised.
    """
    if not announced:
        progress(
            "[corpus] another ingest or retrieval is running on this corpus "
            f"(lock file: {lock_path}); waiting for it to finish…"
        )
    if time.monotonic() >= deadline:
        if not lock_path.exists():
            guard_path = lock_path.with_name(".ingest.guard")
            raise CorpusLockError(
                f"Another process still holds the corpus guard {guard_path} after "
                f"{_INGEST_LOCK_TIMEOUT_SECONDS}s, and no PID file ({lock_path.name}) names "
                "it. Let the other corpus operation finish and retry. There is nothing to "
                "remove: leave .ingest.guard in place."
            )
        raise CorpusLockError(
            f"Another ingest or retrieval still holds {lock_path} after "
            f"{_INGEST_LOCK_TIMEOUT_SECONDS}s. Let the other corpus operation finish and retry. "
            "If no operation is running, a legacy lock file may remain; remove only "
            f"{lock_path.name} and retry. Leave .ingest.guard in place."
        )
    time.sleep(_INGEST_LOCK_POLL_SECONDS)
    return True


# The guard lock is busy (another holder) for these; any other error is a
# filesystem that cannot take the lock at all (see ``_GUARD_UNSUPPORTED``).
_GUARD_BUSY = (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK)
# These failures cannot safely fall back to a PID file: stale-record
# recovery needs the OS guard across its check, rename, and acquisition.
_GUARD_UNSUPPORTED = frozenset(
    {errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL, errno.ENOSYS}
)


def _guard_owner(guard_path: Path) -> str:
    try:
        status = os.stat(guard_path)
        return f"owned by uid {status.st_uid}, mode {status.st_mode & 0o777:04o}"
    except OSError:
        return "its owner and mode could not be read"


def _open_guard(guard_path: Path) -> tuple[int, bool]:
    """Open (creating if needed) the guard file for locking; returns (fd, writable).

    The guard is created world-readable and -writable (the umask applies) and
    opened read-write when this user may; a group-shared corpus, or a
    container whose uid differs from the host's, can then lock it too. On
    POSIX a guard this user may only read is opened read-only, which local
    ``flock`` accepts; a network filesystem (Linux NFS emulates ``flock``
    with byte-range locks) refuses an exclusive lock on such a descriptor
    with ``EBADF``, which ``_corpus_guard`` reports with the ``chmod`` to
    run. ``msvcrt.locking`` needs a writable handle, so Windows never falls
    back. A guard this user may not open at all is reported with the fix
    instead of a bare ``PermissionError``.
    """
    try:
        return os.open(guard_path, os.O_CREAT | os.O_RDWR, 0o666), True
    except PermissionError as exc:
        if os.name != "nt":
            try:
                return os.open(guard_path, os.O_CREAT | os.O_RDONLY, 0o666), False
            except PermissionError:
                pass
        access = "read and write access" if os.name == "nt" else "read access"
        raise CorpusLockError(
            f"Cannot open the corpus lock guard {guard_path} ({_guard_owner(guard_path)}): "
            f"{exc.strerror or exc}. Every user who runs CDA on this corpus needs "
            f"{access} to this permanent coordination file (write access on a network "
            f"filesystem). Ask its owner to run `chmod a+r {guard_path}` (or `chmod 666` "
            "for a shared corpus) and retry."
        ) from exc


def _guard_lock_refused_read_only(guard_path: Path, writable: bool, exc: OSError) -> str:
    """Message for ``EBADF`` from the guard lock: the descriptor lacks the needed access."""
    opened = (
        "The guard was opened read-write, yet the lock was refused"
        if writable
        else "The guard was opened without write access (read-only, because this user may "
        "not write it)"
    )
    return (
        f"Cannot lock the corpus guard {guard_path} ({_guard_owner(guard_path)}): "
        f"EBADF: {exc.strerror or exc}. {opened}; on a network filesystem (NFS) an "
        "exclusive lock needs write access to the file. Ask its owner to run "
        f"`chmod a+w {guard_path}` (or `chmod 666` for a shared corpus) and retry. "
        "Nothing in the corpus was read or written. Leave .ingest.guard in place."
    )


def _guard_unsupported(guard_path: Path, store_existed: bool, guard_existed: bool) -> str:
    """Message for a filesystem that cannot take the guard lock, naming what now exists."""
    created = []
    if not store_existed:
        created.append(f"the store directory {guard_path.parent}")
    if not guard_existed:
        created.append(f"the empty guard file {guard_path.name}")
    state = (
        f"CDA created only {' and '.join(created)}; nothing else was written."
        if created
        else f"Nothing was written; {guard_path.parent} and {guard_path.name} already existed."
    )
    return (
        "The filesystem could not provide the required OS lock. No corpus data was read "
        f"or written. {state} Use a corpus directory on a filesystem that supports OS file "
        "locks, or restore locking support and retry. Leave .ingest.guard in place."
    )


def _lock_guard(fd: int) -> None:
    """Take the OS lock on the guard descriptor without waiting; raises ``OSError``."""
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


@contextmanager
def _corpus_guard(lock_path, deadline, progress, *, store_existed: bool = True):
    """Hold an OS lock on a persistent inode, including during stale PID-file recovery.

    Closing the descriptor releases the guard even after a process crash.
    Its file must never be unlinked: all waiters must lock the same inode.
    A filesystem that refuses the lock stops the operation before the PID
    file or corpus is touched; the refusal names what the attempt created
    (``store_existed`` says whether the store directory predates it). PID-file
    recovery alone cannot serialize callers, even when the stale file's
    contents are checked after rename.
    """
    guard_path = lock_path.with_name(".ingest.guard")
    guard_existed = guard_path.exists()
    fd, writable = _open_guard(guard_path)
    try:
        announced = False
        while True:
            try:
                _lock_guard(fd)
                break
            except OSError as exc:
                if exc.errno in _GUARD_UNSUPPORTED:
                    raise CorpusLockError(
                        f"Cannot safely lock the corpus at {guard_path} "
                        f"({errno.errorcode.get(exc.errno, exc.errno)}: {exc.strerror or exc}). "
                        + _guard_unsupported(guard_path, store_existed, guard_existed)
                    ) from exc
                if exc.errno == errno.EBADF:
                    raise CorpusLockError(
                        _guard_lock_refused_read_only(guard_path, writable, exc)
                    ) from exc
                if exc.errno not in _GUARD_BUSY:
                    raise
                announced = _wait_for_corpus_lock(lock_path, deadline, progress, announced)
        yield
    finally:
        os.close(fd)


@contextmanager
def corpus_lock(config: Config, progress: Callable[[str], None]):
    """Serialize ingest and complete retrievals across threads and processes.

    A retrieval holds this lock across every page, so no reader can observe
    the delete/add gap in a source replacement. It can release the lock once
    the records are in memory; clustering and model calls do not need it.
    The lock lives inside the resolved store so both supported path layouts
    and symlink aliases coordinate on the same file. The file records the
    holder's pid: a lock whose process no longer exists (a crashed or killed
    run) is removed and acquisition retried at once, so it never stalls the
    next run; a lock held by a live process is waited on until the timeout,
    which raises ``CorpusLockError``. A persistent OS lock on ``.ingest.guard``
    (see ``_corpus_guard``) covers that recovery; while it is held, a record
    of this very pid, or an entry that is not a readable file at all, is
    known to be stale as well.
    """
    store_dir = _store_directory(config)
    store_existed = store_dir.is_dir()
    store_dir.mkdir(parents=True, exist_ok=True)
    lock_path = store_dir / ".ingest.lock"
    deadline = time.monotonic() + _INGEST_LOCK_TIMEOUT_SECONDS
    with _corpus_guard(lock_path, deadline, progress, store_existed=store_existed):
        announced = False
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                pid, content, stale = _lock_holder(lock_path)
                if not stale and pid == os.getpid():
                    # Every holder in this process holds the guard, which this
                    # thread now holds alone, so no thread here owns this record:
                    # it is this PID reused after a crash (a container's PID 1,
                    # or a recycled pid), never a live holder. A record of another
                    # live PID is waited on: with the guard held it is not a
                    # current-protocol holder, but it may be a legacy holder (a
                    # version from before the guard existed, which coordinated
                    # on the PID file alone), which the PID file alone cannot
                    # tell from a stale record.
                    stale = True
                if not stale and pid is None and not content:
                    age = _unreadable_lock_age(lock_path)
                    if age is not None and age > _INGEST_LOCK_POLL_SECONDS:
                        progress(
                            f"[corpus] {lock_path} is not a lock file CDA can read (a "
                            f"directory, a dangling link, or a file without read "
                            f"permission); {_discard_unreadable_lock(lock_path)}; continuing"
                        )
                        continue
                if stale:
                    if _reclaim_stale_lock(lock_path, content):
                        progress(
                            "[corpus] removed a stale lock file left by "
                            + (
                                f"an earlier run that had this process id ({pid})"
                                if pid == os.getpid()
                                else f"process {pid}, which is no longer running"
                                if pid
                                else "a crashed run"
                            )
                            + f" ({lock_path}); continuing"
                        )
                        continue
                announced = _wait_for_corpus_lock(lock_path, deadline, progress, announced)
        try:
            try:
                os.write(fd, f"{os.getpid()}\n".encode())
            finally:
                os.close(fd)
            yield
        finally:
            lock_path.unlink(missing_ok=True)


def _ingest_state(collection, identity: str, current: dict | None = None) -> tuple[str, str]:
    """(state, reason) for the chunks stored under this ``pdf_relpath``.

    States: "absent"; "partial" (inconsistent chunk counts or positions); "legacy" (no original sentence boundaries); "stale"
    (a stored field differs from ``current``, see ``_STALE_REASONS``; a
    ``None`` in ``current`` is not compared); "complete".
    """
    result = collection.get(where={"pdf_relpath": identity}, include=["metadatas"])
    metadatas = [m or {} for m in result.get("metadatas") or []]
    if not metadatas:
        return "absent", ""
    expected = metadatas[0].get("total_chunks")
    if expected is not None and (
        type(expected) is not int
        or expected < 1
        or len(metadatas) != expected
        or any(m.get("total_chunks") != expected for m in metadatas)
    ):
        return "partial", "stored chunk counts are incomplete or inconsistent"
    indices = [m.get("chunk_index") for m in metadatas]
    if expected is not None and (
        any(type(index) is not int for index in indices) or set(indices) != set(range(expected))
    ):
        return "partial", "stored chunk positions are incomplete or inconsistent"
    if any(
        not m.get("sentence_context") or "sentence_start" not in m or "sentence_lengths" not in m
        for m in metadatas
    ):
        return "legacy", "its chunks lack original sentence boundaries"
    for key, value in (current or {}).items():
        if value is not None and any(m.get(key) != value for m in metadatas):
            return "stale", _STALE_REASONS[key]
    return "complete", ""


def _alternate_spellings(collection, rel_name: str, disk_rel: str) -> list[str]:
    """Stored relpaths that are other Unicode spellings of this file's path.

    Relpaths are stored NFC, but older stores and synced folders can hold the
    NFD spelling (or both), which would duplicate every chunk of the file.
    For an ASCII path there is no alternate spelling and nothing is queried.
    """
    alternates = {unicodedata.normalize("NFD", rel_name), disk_rel} - {rel_name}
    return sorted(a for a in alternates if _ingest_state(collection, a)[0] != "absent")


def _delete_sources(
    collection, alternates: list[str], rel_name: str, progress: Callable[[str], None]
) -> None:
    """Remove the chunks stored under alternate spellings of ``rel_name``."""
    for alternate in alternates:
        progress(
            f"  [ingest] {rel_name}: removing chunks stored under a "
            "different Unicode spelling of the same filename, so the "
            "document cannot exist twice in the corpus"
        )
        collection.delete(where={"pdf_relpath": alternate})


def _same_file_elsewhere(
    stored_sources: dict[str, dict], rel_name: str, sha256: str, alternates: list[str]
) -> list[tuple[str, str | None]]:
    """(relpath, pdf_root) of stored sources holding this exact file under another name."""
    return [
        (relpath, info["pdf_root"])
        for relpath, info in sorted(stored_sources.items())
        if info["source_file_sha256"] == sha256
        and unicodedata.normalize("NFC", relpath) != rel_name
        and relpath not in alternates
    ]


def _copy_still_matches(
    root: Path, relpath: str, sha256: str, unchecked: set[str], index: _DiskIndex | None = None
) -> bool:
    """A missing copy can be renamed; a live copy must still hold the recorded bytes.

    A copy whose folder cannot be listed is assumed to still match, so it is
    neither renamed away nor ignored.
    """
    if not _exists_or_unchecked(root, relpath, unchecked, index):
        return True
    if relpath in unchecked:
        return True
    try:
        return file_digest(_path_exactly(root, relpath, index) or root / relpath) == sha256
    except OSError:
        # An unreadable candidate cannot justify dropping the readable source.
        return False


def _exists_or_unchecked(
    root: Path, relpath: str, unchecked: set[str], index: _DiskIndex | None = None
) -> bool:
    """``_exists_exactly``, except that a folder that cannot be listed counts as present.

    Such a relpath is added to ``unchecked`` and reported at the end of the
    run; it is never pruned or renamed away.
    """
    try:
        return _exists_exactly(root, relpath, index)
    except OSError:
        unchecked.add(relpath)
        return True


def _adopt_root(collection, relpath: str, pdf_root: str, progress: Callable[[str], None]) -> None:
    """Record ``pdf_root`` on every chunk of a source stored before roots were recorded."""
    ids = collection.get(where={"pdf_relpath": relpath}, include=[])["ids"]
    for start in range(0, len(ids), 1000):
        batch = ids[start : start + 1000]
        collection.update(ids=batch, metadatas=[{"pdf_root": pdf_root}] * len(batch))
    progress(
        f"  [ingest] {relpath}: stored without a PDF root by an older version; the same "
        f"file is at this path under {pdf_root}, so it now belongs to this root"
    )


# Chunk fields that come from a Zotero export; kept when a source is refreshed
# without one, so a re-ingest never silently strips titles, dates, and tags.
_ZOTERO_FIELDS = (
    "title",
    "year",
    "doi",
    "date_added",
    "zotero_tags",
    "zotero_match",
    "zotero_unmatched_reason",
    "zotero_csv_sha256",
    "zotero_match_filename",
)


def _stored_zotero_fields(collection, identities: list[str]) -> dict:
    """The Zotero fields stored for the first of ``identities`` that has any."""
    for identity in identities:
        result = collection.get(where={"pdf_relpath": identity}, include=["metadatas"], limit=1)
        metadata = (result.get("metadatas") or [None])[0] or {}
        fields = {key: metadata[key] for key in _ZOTERO_FIELDS if key in metadata}
        if fields:
            return fields
    return {}


def _failed_refresh(collection, rel_name: str, state: str, stats: dict) -> str:
    """Settle the stored copy of a file that failed to re-ingest; returns a listing suffix.

    A complete earlier version stays (a failure must never destroy stored
    data). An incomplete one ("partial") is removed: its truncated chunks
    would misrepresent the document in every analysis.
    """
    if state == "partial":
        collection.delete(where={"pdf_relpath": rel_name})
        return " (incomplete stored chunks removed)"
    if state != "absent":
        stats["pdfs_kept_previous"] += 1
        return " (previous version kept)"
    return ""


def _failed_refresh_note(state: str) -> str:
    if state == "partial":
        return "its incomplete stored chunks were removed, so it will not appear in any analysis"
    if state != "absent":
        return (
            "the previously stored version was kept and stays in analyses until "
            "this file can be read"
        )
    return "it will not appear in any analysis"


def select_pdfs(pdf_dir: Path, *, recursive: bool = True, exclude: list[str] | None = None):
    """The PDFs to ingest under ``pdf_dir``, in a stable order; returns (kept, n_excluded).

    ``recursive=False`` takes only files directly in ``pdf_dir``. An
    ``exclude`` pattern matching a directory name skips that sub-tree; a
    wildcard pattern is matched against the path relative to ``pdf_dir``. A
    trailing slash is stripped, and the ``.pdf`` check ignores case. Patterns
    and paths are compared in NFC, so a composed pattern excludes a folder
    whose name is stored decomposed (as macOS often does). Symlinked folders
    are followed, and each physical file is selected once, under its
    canonical route: its real path relative to ``pdf_dir`` when that lies
    under ``pdf_dir``; for a file outside it (a link to another folder), the
    first route found in sorted order. Exclusion is decided on the canonical
    route alone, so a link elsewhere in the tree can neither hide a PDF whose
    real folder is included nor keep one whose real folder is excluded.
    ``n_excluded`` counts physical files, not routes.
    """
    kept, excluded, _ = _select_pdf_routes(pdf_dir, recursive, exclude)
    return kept, excluded


def _select_pdf_routes(pdf_dir: Path, recursive: bool, exclude: list[str] | None):
    """Select PDFs and retain canonical routes relative to the PDF root."""
    found = (p for p in _walk_files(pdf_dir, recursive) if p.suffix.lower() == ".pdf")
    routes = _canonical_routes(pdf_dir, (p for p in found if p.is_file()))
    kept, excluded = [], 0
    for path in sorted(routes.values()):
        if _is_excluded(path.relative_to(pdf_dir), exclude, recursive=recursive):
            excluded += 1
        else:
            kept.append(path)
    return kept, excluded, {real: path.relative_to(pdf_dir) for real, path in routes.items()}


def _canonical_routes(pdf_dir: Path, paths) -> dict[str, Path]:
    """{real path: canonical route} of the files reached by ``paths`` (see ``select_pdfs``)."""
    real_root = os.path.realpath(pdf_dir)
    routes: dict[str, Path] = {}
    for path in sorted(paths):
        real = os.path.realpath(path)
        if real in routes:
            continue
        try:
            routes[real] = pdf_dir / Path(real).relative_to(real_root)
        except ValueError:  # outside pdf_dir: the first route in sorted order is canonical
            routes[real] = path
    return routes


def _canonical_relpath(
    root: Path,
    relpath: str,
    routes: dict[str, Path] | None = None,
    index: _DiskIndex | None = None,
) -> str:
    """The route ``select_pdfs`` lists ``root/relpath`` under, in NFC.

    Use the selection's route for external files too. Without a mapped route,
    resolve internal links and preserve external or missing paths as recorded.
    ``index`` supplies cached folder listings; ``_DiskIndex.canonical`` is
    this computed once per stored relpath.
    """
    try:
        disk_path = _path_exactly(root, relpath, index)
    except OSError:
        disk_path = None  # absence is checked separately; unreadable folders are retained
    real = os.path.realpath(disk_path or root / relpath)
    if routes is not None and real in routes:
        return _nfc(str(routes[real]))
    try:
        return _nfc(str(Path(real).relative_to(os.path.realpath(root))))
    except ValueError:
        return _nfc(relpath)


def _is_alias_route(
    root: Path,
    relpath: str,
    routes: dict[str, Path] | None = None,
    index: _DiskIndex | None = None,
) -> bool:
    """Whether a stored relpath is a link route to a file stored under its real path instead."""
    if index is not None and root == index.root and routes is index.routes:
        canonical = index.canonical(relpath)
    else:
        canonical = _canonical_relpath(root, relpath, routes, index)
    return canonical != _nfc(relpath)


def _walk_files(pdf_dir: Path, recursive: bool) -> list[Path]:
    """Files under ``pdf_dir`` (only its top level unless ``recursive``), following folder symlinks.

    Depth first with siblings in sorted order, so the first route to a folder
    in sorted order is the one walked. Every physical folder is walked once:
    a symlink cycle, or a second link to a folder already walked, is skipped.
    A folder that cannot be listed is skipped, as ``os.walk`` does.
    """
    if not recursive:
        return list(pdf_dir.iterdir())
    files: list[Path] = []
    walked: set[str] = set()
    pending = [pdf_dir]
    while pending:
        folder = pending.pop()
        real = os.path.realpath(folder)
        if real in walked:
            continue
        walked.add(real)
        try:
            with os.scandir(folder) as entries:
                listing = sorted(entries, key=lambda entry: entry.name)
        except OSError:
            continue
        subfolders = []
        for entry in listing:
            path = folder / entry.name
            try:
                is_dir = entry.is_dir()  # follows links; a broken link is not a directory
            except OSError:
                is_dir = False
            (subfolders if is_dir else files).append(path)
        pending.extend(reversed(subfolders))
    return files


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _is_excluded(relative: Path, exclude: list[str] | None, *, recursive: bool = True) -> bool:
    """Whether a path relative to the PDF root is outside the selection of ``select_pdfs``.

    Both the path and the patterns are compared in NFC, so the files walked on
    disk and the relpaths stored in the corpus (always NFC) are judged alike.
    """
    from fnmatch import fnmatch

    relative = Path(_nfc(str(relative)))
    parts = relative.parts[:-1]  # directories only; the file is matched whole
    if parts and not recursive:
        return True
    patterns = [_nfc(pattern.rstrip("/")) for pattern in (exclude or [])]
    return any(
        fnmatch(str(relative), pattern) or any(fnmatch(part, pattern) for part in parts)
        for pattern in patterns
    )


def _load_zotero(zotero_csv, progress: Callable[[str], None]) -> dict:
    """Zotero lookup indexes plus the export's digest; empty when no CSV is given."""
    if not zotero_csv:
        return {}
    records = load_zotero_csv(zotero_csv, progress=progress)
    zotero = {
        "sha256": file_digest(Path(zotero_csv).expanduser()),
        "filename": build_filename_index(records),
        "doi": build_doi_index(records),
        "title": build_title_index(records),
    }
    counts = {
        kind: (len(zotero[kind]), count_ambiguous(zotero[kind]))
        for kind in ("filename", "doi", "title")
    }
    progress(
        f"[zotero] loaded {len(records)} records: "
        + ", ".join(
            f"{n - ambiguous} unambiguous {kind} keys ({ambiguous} shared by several records)"
            for kind, (n, ambiguous) in counts.items()
        )
    )
    return zotero


_AMBIGUITY = {
    "shared_basename": "it shares a basename with another corpus PDF",
    "ambiguous_filename": "its attachment name belongs to more than one Zotero item",
    "ambiguous_doi": (
        "its DOI belongs to more than one Zotero item, or its opening text "
        "names several library DOIs"
    ),
    "ambiguous_title": (
        "its title matches more than one Zotero item, or its opening text "
        "quotes several library titles"
    ),
}


def _zotero_fields(
    pdf_path: Path,
    rel_name: str,
    text: str,
    zotero: dict,
    shared_basename: bool,
    stats: dict,
    progress: Callable[[str], None],
) -> dict:
    """Chunk-metadata fields from the Zotero export for one PDF, with how they matched.

    Ambiguity never resolves to an arbitrary record: a basename shared by
    several corpus PDFs attaches nothing, and a filename, DOI, or title key
    shared by several Zotero records attaches nothing for that key. The
    reason is stored as ``zotero_unmatched_reason`` and reported.
    """
    if shared_basename:
        record, outcome = None, "shared_basename"
    else:
        record, outcome = match_by_filename(pdf_path, zotero["filename"])
        if record is None:
            record, content_outcome = match_by_content(text, zotero["doi"], zotero["title"])
            outcome = content_outcome or outcome
    if record is not None:
        stats[f"zotero_by_{outcome}"] += 1
        return dict(record_fields(record), zotero_match=outcome)
    stats["zotero_unmatched"] += 1
    if outcome in _AMBIGUITY:
        stats["zotero_ambiguous"] += 1
        progress(
            f"  [zotero] {rel_name}: no Zotero metadata attached because "
            f"{_AMBIGUITY[outcome]}. Rename the file or merge the duplicate Zotero "
            "items to disambiguate; MyWork falls back to filename/title rules."
        )
    return {"zotero_unmatched_reason": outcome or "no_match"}


def _refresh_zotero_metadata(
    collection,
    pdf_path: Path,
    rel_name: str,
    zotero: dict,
    shared_basename: bool,
    stats: dict,
    progress: Callable[[str], None],
) -> None:
    """Bring a reused source's Zotero fields up to date with this export, in place.

    The chunks, sentences, and embeddings depend only on the file, so a new
    export changes metadata alone: a field that changed is updated on every
    chunk under the same ids, and when nothing changed only the export's
    digest is recorded. The PDF text is read only when the file name does
    not identify the record (content matching). A read failure raises and
    leaves the stored chunks untouched.
    """
    stored = collection.get(where={"pdf_relpath": rel_name}, include=["metadatas"])
    first = (stored.get("metadatas") or [None])[0] or {}
    old_metadata = stored["metadatas"]
    same_fields = all(
        all(m.get(key) == first.get(key) for key in _ZOTERO_FIELDS) for m in old_metadata
    )
    was_shared = first.get("zotero_unmatched_reason") == "shared_basename"
    if (
        same_fields
        and first.get("zotero_csv_sha256") == zotero["sha256"]
        and first.get("zotero_match_filename") == _nfc(pdf_path.name)
        and was_shared == shared_basename
    ):
        stats["zotero_unchanged"] += 1
        return
    needs_text = not shared_basename and match_by_filename(pdf_path, zotero["filename"])[0] is None
    text = unicodedata.normalize("NFC", extract_pdf_text(pdf_path)) if needs_text else ""
    fields = _zotero_fields(pdf_path, rel_name, text, zotero, shared_basename, stats, progress)
    fields["zotero_csv_sha256"] = zotero["sha256"]
    fields["zotero_match_filename"] = _nfc(pdf_path.name)
    # Chroma merges metadata on update; None removes a field no longer supplied.
    patches = [
        {key: fields.get(key) for key in _ZOTERO_FIELDS if m.get(key) != fields.get(key)}
        for m in old_metadata
    ]
    ids = stored["ids"]
    changed = sorted(
        {key for patch in patches for key in patch} - {"zotero_csv_sha256", "zotero_match_filename"}
    )
    try:
        for start in range(0, len(ids), 1000):
            updates = [
                (i, patch)
                for i, patch in zip(
                    ids[start : start + 1000], patches[start : start + 1000], strict=True
                )
                if patch
            ]
            if updates:
                collection.update(
                    ids=[i for i, _ in updates], metadatas=[patch for _, patch in updates]
                )
    except (Exception, KeyboardInterrupt) as original:
        # None removes newly introduced fields; each chunk gets its own old values.
        try:
            for start in range(0, len(ids), 1000):
                collection.update(
                    ids=ids[start : start + 1000],
                    metadatas=[
                        {key: m.get(key) for key in _ZOTERO_FIELDS}
                        for m in old_metadata[start : start + 1000]
                    ],
                )
        except Exception as rollback:
            raise IngestRollbackError(
                f"Updating metadata for {rel_name!r} failed ({original}); restoring it "
                f"also failed ({rollback}). Metadata may be incomplete; repair storage "
                "and rerun ingestion before analysis."
            ) from original
        raise
    if changed:
        stats["metadata_updated"] += 1
        progress(
            f"  [zotero] {rel_name}: updated {', '.join(changed)} from the new export; "
            "text and embeddings reused"
        )


def _stored_pdf_sources(collection, batch_size: int = 5000) -> dict[str, dict]:
    """{pdf_relpath: {"pdf_root", "source_file_sha256"}} for every PDF-ingested source.

    ``pdf_root`` is ``None`` for sources stored before roots were recorded.
    """
    sources: dict[str, dict] = {}
    offset = 0
    while True:
        page = collection.get(include=["metadatas"], limit=batch_size, offset=offset)
        metadatas = page.get("metadatas") or []
        for metadata in metadatas:
            metadata = metadata or {}
            if "source_file_sha256" in metadata and metadata.get("pdf_relpath"):
                sources[str(metadata["pdf_relpath"])] = {
                    "pdf_root": metadata.get("pdf_root"),
                    "source_file_sha256": metadata["source_file_sha256"],
                }
        if len(metadatas) < batch_size:
            return sources
        offset += len(metadatas)


class _DiskIndex:
    """Per-run cache of folder listings and of stored sources' canonical routes.

    One ingest run neither renames nor removes library files, so a folder is
    listed once ({NFC name: name on disk}) however many stored relpaths pass
    through it, and a stored relpath's canonical route under ``root`` is
    computed once however many checks ask for it. Without the cache every
    check re-listed every ancestor folder of every stored source, which made
    an unchanged rerun quadratic in the library size. Each run builds its
    own index and nothing is kept at module level, so no run can see a
    listing another run made.
    """

    def __init__(self, root: Path, routes: dict[str, Path] | None = None):
        self.root = root
        self.routes = routes
        self._listings: dict[str, dict[str, str] | None] = {}
        self._canonical: dict[str, str] = {}

    def listing(self, directory: Path) -> dict[str, str] | None:
        """{NFC name: disk name} of ``directory``, or ``None`` when it is not a directory.

        Of two disk names with one NFC form, the first listed wins. Permission
        and I/O errors propagate and are not cached: they do not prove
        absence, so pruning or rename detection cannot act on them.
        """
        key = os.fspath(directory)
        if key not in self._listings:
            try:
                names = os.listdir(directory)
            except (FileNotFoundError, NotADirectoryError):
                self._listings[key] = None
            else:
                table: dict[str, str] = {}
                for name in names:
                    table.setdefault(_nfc(name), name)
                self._listings[key] = table
        return self._listings[key]

    def canonical(self, relpath: str) -> str:
        """``_canonical_relpath`` of a stored relpath under this root, computed once."""
        if relpath not in self._canonical:
            self._canonical[relpath] = _canonical_relpath(self.root, relpath, self.routes, self)
        return self._canonical[relpath]


def _exists_exactly(root: Path, relpath: str, index: _DiskIndex | None = None) -> bool:
    """Whether ``root/relpath`` is a file under exactly this spelling, letter case included.

    ``Path.is_file`` ignores case on case-insensitive filesystems (the macOS
    default), so a renamed file would still look present under its old name.
    NFC and NFD spellings of one name count as the same name.
    """
    path = _path_exactly(root, relpath, index)
    return path is not None and path.is_file()


def _path_exactly(root: Path, relpath: str, index: _DiskIndex | None = None) -> Path | None:
    """Resolve a stored NFC relpath to its disk spelling without ignoring letter case.

    ``index`` caches the folder listings across calls; without one each
    ancestor folder is listed for this call alone.
    """
    index = _DiskIndex(root) if index is None else index
    current = root
    for part in Path(relpath).parts:
        names = index.listing(current)
        if names is None:
            return None
        match = names.get(_nfc(part))
        if match is None:
            return None
        current = current / match
    return current


def _missing_sources(
    collection,
    pdf_dir: Path,
    pdf_root: str,
    *,
    exclude: list[str] | None = None,
    recursive: bool = True,
    unchecked: set[str] | None = None,
    routes: dict[str, Path] | None = None,
    stored_sources: dict[str, dict] | None = None,
    index: _DiskIndex | None = None,
) -> tuple[list[str], list[str]]:
    """(missing, root_unknown) among the stored sources.

    ``missing``: sources of this PDF root whose file is no longer on disk
    under its exact recorded name (sources now excluded, or stored under a
    link route, are listed by ``_excluded_sources`` and ``_alias_sources``
    instead). ``root_unknown``: sources stored before ``pdf_root`` was
    recorded; they may belong to any library, so they are never attributed
    to this root or pruned. A source whose folder cannot be listed is not
    missing; it is added to ``unchecked``. ``stored_sources`` (a scan of the
    store) and ``index`` (the run's folder listings and canonical routes)
    are shared with the other listings so each is built once per run.
    """
    unchecked = set() if unchecked is None else unchecked
    stored_sources = _stored_pdf_sources(collection) if stored_sources is None else stored_sources
    index = _DiskIndex(pdf_dir, routes) if index is None else index
    missing, root_unknown = [], []
    for relpath, info in sorted(stored_sources.items()):
        if info["pdf_root"] is None:
            root_unknown.append(relpath)
        elif info["pdf_root"] == pdf_root:
            canonical = index.canonical(relpath)
            if (
                not _is_excluded(Path(canonical), exclude, recursive=recursive)
                and canonical == _nfc(relpath)  # not an alias route
                and not _exists_or_unchecked(pdf_dir, relpath, unchecked, index)
            ):
                missing.append(relpath)
    return missing, root_unknown


def _excluded_sources(
    collection,
    pdf_dir: Path,
    pdf_root: str,
    exclude: list[str] | None,
    recursive: bool,
    routes: dict[str, Path] | None = None,
    *,
    stored_sources: dict[str, dict] | None = None,
    index: _DiskIndex | None = None,
) -> list[str]:
    """Stored sources of this PDF root whose canonical route the current ``exclude``/``recursive`` leave out."""
    stored_sources = _stored_pdf_sources(collection) if stored_sources is None else stored_sources
    index = _DiskIndex(pdf_dir, routes) if index is None else index
    return [
        relpath
        for relpath, info in sorted(stored_sources.items())
        if info["pdf_root"] == pdf_root
        and _is_excluded(Path(index.canonical(relpath)), exclude, recursive=recursive)
    ]


def _alias_sources(
    collection,
    pdf_dir: Path,
    pdf_root: str,
    exclude: list[str] | None,
    recursive: bool,
    routes: dict[str, Path] | None = None,
    *,
    stored_sources: dict[str, dict] | None = None,
    index: _DiskIndex | None = None,
) -> list[str]:
    """Stored sources of this PDF root recorded under a link route to a selected file.

    The file's own path is its identity (see ``select_pdfs``); such a source
    is normally replaced by that path during the run, so only one whose file
    was skipped or unreadable remains to be listed here.
    """
    stored_sources = _stored_pdf_sources(collection) if stored_sources is None else stored_sources
    index = _DiskIndex(pdf_dir, routes) if index is None else index
    aliased = []
    for relpath, info in sorted(stored_sources.items()):
        if info["pdf_root"] != pdf_root:
            continue
        canonical = index.canonical(relpath)
        if canonical != _nfc(relpath) and not _is_excluded(
            Path(canonical), exclude, recursive=recursive
        ):
            aliased.append(relpath)
    return aliased


def _handle_missing(
    collection,
    pdf_dir: Path,
    pdf_root: str,
    prune_missing: bool,
    stats: dict,
    progress: Callable[[str], None],
    *,
    exclude: list[str] | None = None,
    recursive: bool = True,
    unchecked: set[str] | None = None,
    stored_sources: dict[str, dict] | None = None,
    file_digests: dict[str, str] | None = None,
    routes: dict[str, Path] | None = None,
    index: _DiskIndex | None = None,
    failed_renames: set[str] | None = None,
) -> None:
    """Report, and with ``prune_missing`` remove, stored sources the library no longer backs.

    ``stored_sources`` is the ingest loop's view of the store after this run
    and ``file_digests`` the digests it computed ({NFC relpath: sha256} of
    every selected file it could read), so no file is hashed again here. A
    source stored under a symlink route (an older store's identity) is a
    missing source when its canonical file is gone; a duplicate, or an older
    version, when a complete source holds the canonical file's current bytes;
    and retained, as the document's only stored version, when the canonical
    file exists but could not be stored this run (unreadable, no text, or a
    failed refresh). Retention does not depend on ``prune_missing``.
    """
    unchecked = set() if unchecked is None else unchecked
    file_digests = {} if file_digests is None else file_digests
    index = _DiskIndex(pdf_dir, routes) if index is None else index
    # One fresh scan of the store feeds the three listings below. The loop's
    # ``stored_sources`` is not used for them: it does not track every
    # deletion the loop made (alternate Unicode spellings removed by
    # ``_delete_sources``, incomplete chunks removed by ``_failed_refresh``),
    # and a deleted name listed here would be miscounted as missing or aliased.
    current = _stored_pdf_sources(collection)
    stored_sources = current if stored_sources is None else stored_sources
    missing, root_unknown = _missing_sources(
        collection,
        pdf_dir,
        pdf_root,
        exclude=exclude,
        recursive=recursive,
        unchecked=unchecked,
        stored_sources=current,
        index=index,
    )
    retained_renames = sorted(set(missing) & (failed_renames or set()))
    if retained_renames:
        missing = [name for name in missing if name not in failed_renames]
        stats["pdfs_kept_previous"] += len(retained_renames)
        progress(
            "[warn] renamed files could not be read; their previous stored versions "
            "are retained until the replacements can be stored:\n"
            + "\n".join(f"    {name}" for name in retained_renames)
        )
    excluded = _excluded_sources(
        collection, pdf_dir, pdf_root, exclude, recursive, stored_sources=current, index=index
    )
    aliased = _alias_sources(
        collection, pdf_dir, pdf_root, exclude, recursive, stored_sources=current, index=index
    )
    # A replacement must survive cleanup. In particular, aliases must not
    # certify each other and then both be deleted after a failed refresh.
    cleanup_candidates = set(missing) | set(excluded) | set(aliased)
    retained, duplicates, older = [], [], []
    for relpath in aliased:
        canonical = index.canonical(relpath)
        try:
            exists = _exists_exactly(pdf_dir, canonical, index)
        except OSError:
            unchecked.add(relpath)
            continue
        if not exists:
            missing.append(relpath)
            continue
        digest = file_digests.get(canonical)
        stored_elsewhere = digest is not None and any(
            name not in cleanup_candidates
            and info["source_file_sha256"] == digest
            and _ingest_state(collection, name, {"source_file_sha256": digest})[0] == "complete"
            for name, info in stored_sources.items()
        )
        if not stored_elsewhere:
            retained.append(relpath)
        elif (stored_sources.get(relpath) or {}).get("source_file_sha256") == digest:
            duplicates.append(relpath)
        else:
            older.append(relpath)
    missing.sort()
    if retained:
        stats["pdfs_kept_previous"] += len(retained)
        progress(
            "[warn] canonical replacement could not be stored (the file exists but could "
            "not be read, has no text, or failed to refresh); the previous versions stored "
            "under their symlink routes are retained because no complete replacement "
            "will remain after cleanup, whether or not --prune-missing is given:\n"
            + "\n".join(f"    {relpath}" for relpath in retained)
        )
    stats["pdfs_missing"] = len(missing)
    stats["pdfs_aliased"] = len(duplicates) + len(older)
    stats["pdfs_stored_excluded"] = len(excluded)
    stats["pdfs_root_unknown"] = len(root_unknown)
    stats["pdfs_unchecked"] = len(unchecked)
    if unchecked:
        progress(
            f"[warn] {len(unchecked)} stored source(s) are in folders that could not be "
            "read (permission or I/O error); could not check; not pruned:\n"
            + "\n".join(f"    {relpath}" for relpath in sorted(unchecked))
        )
    if root_unknown:
        progress(
            f"[note] {len(root_unknown)} stored source(s) have no recorded PDF root "
            "(ingested by an older version), so CDA cannot tell which library they "
            "belong to; root unknown; not pruned:\n"
            + "\n".join(f"    {relpath}" for relpath in root_unknown)
        )
    for relpaths, what in (
        (missing, "no longer on disk"),
        (
            duplicates,
            "stored under a symlink route to a file stored under its own path (the file's "
            "own path is its identity now; the document is stored twice)",
        ),
        (
            older,
            "stored under a symlink route as an older version of a file re-ingested under "
            "its own path (both versions of the document are counted until the route is "
            "pruned)",
        ),
        (excluded, "now excluded by --exclude or --no-recursive"),
    ):
        if not relpaths:
            continue
        listing = "\n".join(f"    {relpath}" for relpath in relpaths)
        if not prune_missing:
            progress(
                f"[warn] {len(relpaths)} source(s) stored for {pdf_dir} are {what} "
                f"and stay in the corpus and in every analysis:\n{listing}\n"
                "  Rerun cda ingest with --prune-missing to remove them."
            )
            continue
        for relpath in relpaths:
            collection.delete(where={"pdf_relpath": relpath})
        stats["pdfs_pruned"] += len(relpaths)
        progress(f"[ingest] removed {len(relpaths)} source(s) {what}:\n{listing}")


def _reusable_copy(collection, copies: list[str], current: dict) -> str | None:
    """The first stored copy whose chunks are complete and current, so they can be re-keyed.

    Its bytes already equal the file's (``current`` carries the digest), so
    when the chunk settings and segmenter are current too, the stored text,
    sentence boundaries, and embeddings are exactly what a rebuild would
    produce; only the identity changes. A copy that is stale, partial, or
    legacy is rebuilt as before.
    """
    for relpath in copies:
        if _ingest_state(collection, relpath, current)[0] == "complete":
            return relpath
    return None


def _rekeyed_chunks(
    collection, relpath: str, id_key: str, identity: dict, zotero_fields: dict | None
) -> dict:
    """The stored chunks of ``relpath`` prepared under a new identity (see ``_prepare_chunks``).

    Text, sentence boundaries, and embeddings are taken as stored; the
    ``identity`` fields replace the old ones. ``zotero_fields`` replaces the
    stored Zotero fields (the caller takes them from the stored sources, as
    a rebuild without an export would); ``None`` keeps them as stored for a
    later in-place refresh. Nothing is extracted or embedded.
    """
    rows = collection.get(
        where={"pdf_relpath": relpath}, include=["documents", "metadatas", "embeddings"]
    )
    order = sorted(range(len(rows["ids"])), key=lambda i: rows["metadatas"][i]["chunk_index"])
    metadatas = []
    for i in order:
        metadata = dict(rows["metadatas"][i])
        if zotero_fields is not None:
            metadata = {key: value for key, value in metadata.items() if key not in _ZOTERO_FIELDS}
            metadata.update(zotero_fields)
        metadata.update(identity)
        metadatas.append(metadata)
    return {
        "id_key": id_key,
        "source": identity["pdf_relpath"],
        "ids": [f"{id_key}::c{rows['metadatas'][i]['chunk_index']}" for i in order],
        "documents": [rows["documents"][i] for i in order],
        "embeddings": [[float(x) for x in rows["embeddings"][i]] for i in order],
        "metadatas": metadatas,
    }


def _route_moves_summary(
    moves: list[tuple[str, str | None, str]], pdf_dir: Path, pdf_root: str, index: _DiskIndex
) -> list[str]:
    """One line per route change among the sources re-keyed this run.

    A route change is a move between folders that keeps the trailing path
    (``b/x.pdf`` to ``a/x.pdf``): the canonical route of an external folder
    is its first link in sorted order, so adding or removing a link moves
    every source in that folder. Plain renames and copies that came from
    another PDF root are summarized separately.
    """

    def route(parts: tuple[str, ...]) -> str:
        return f"route {'/'.join(parts)}/" if parts else "the top level"

    changes: Counter[tuple[tuple[str, ...], tuple[str, ...], str]] = Counter()
    renamed = other_root = 0
    for old, root, new in moves:
        if root not in (None, pdf_root):
            other_root += 1
            continue
        old_parts, new_parts = Path(old).parts, Path(new).parts
        shared = 0
        while (
            shared < min(len(old_parts), len(new_parts))
            and old_parts[-1 - shared] == new_parts[-1 - shared]
        ):
            shared += 1
        if shared == 0:
            renamed += 1
            continue
        try:
            old_disk = _path_exactly(pdf_dir, old, index)
        except OSError:
            old_disk = pdf_dir / old
        if old_disk is None:
            reason = "the old route is no longer on disk"
        elif os.path.realpath(old_disk) == os.path.realpath(pdf_dir / new):
            reason = "a link that sorts earlier now reaches the same folder"
        else:
            reason = "the file is now selected under this route"
        changes[(old_parts[:-shared], new_parts[:-shared], reason)] += 1
    lines = [
        f"[ingest] {n} source(s) moved from {route(old)} to {route(new)} because {reason}; "
        "their text and embeddings were re-keyed, not rebuilt"
        for (old, new, reason), n in sorted(changes.items())
    ]
    if renamed:
        lines.append(
            f"[ingest] {renamed} source(s) renamed on disk were re-keyed; their text and "
            "embeddings were not rebuilt"
        )
    if other_root:
        lines.append(
            f"[ingest] {other_root} source(s) stored under another PDF root were re-keyed to "
            "this root; their text and embeddings were not rebuilt"
        )
    return lines


def ingest_pdf_dir(
    config: Config,
    embedder: Embedder,
    pdf_dir: str | Path,
    zotero_csv: str | Path | None = None,
    *,
    reingest: bool = False,
    recursive: bool = True,
    exclude: list[str] | None = None,
    prune_missing: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, int]:
    """Ingest the PDFs under ``pdf_dir`` (see ``select_pdfs``). Returns counters.

    A stored source is reused only when it is complete and its file digest,
    chunk settings, and sentence segmenter all match the current inputs;
    otherwise it is rebuilt. A reused source whose Zotero export digest
    differs gets its Zotero fields updated in place (``metadata_updated``
    counts the sources whose fields changed); it is not re-embedded. New chunks are fully
    extracted, segmented, and embedded before the stored ones are replaced,
    so a file that fails (unreadable, encrypted, no text) keeps its previous
    version, and the failure is counted and listed. Without a Zotero CSV, a
    refreshed source keeps its stored Zotero fields. A file whose exact bytes
    are already stored under another name replaces that name when it is gone
    from disk, and is otherwise skipped as a duplicate. A file reached
    through a folder symlink is stored under its canonical route (see
    ``select_pdfs``); a source an older version stored under the link route
    is replaced the same way (same digest, new relpath). Stored sources whose
    file is gone, or whose canonical route the current ``exclude``/``recursive``
    leave out, are reported, and removed only with ``prune_missing``; sources
    without a recorded root are never pruned.
    """
    pdf_dir = Path(pdf_dir).expanduser()
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    pdf_root = unicodedata.normalize("NFC", str(pdf_dir.resolve()))
    zotero = _load_zotero(zotero_csv, progress)

    stats = {
        "pdfs_seen": 0,
        "pdfs_ingested": 0,
        "pdfs_skipped": 0,
        "pdfs_stale": 0,
        "pdfs_refreshed": 0,
        "pdfs_no_text": 0,
        "pdfs_unreadable": 0,
        "pdfs_missing": 0,
        "pdfs_aliased": 0,
        "pdfs_pruned": 0,
        "pdfs_stored_excluded": 0,
        "metadata_updated": 0,
        "chunks_added": 0,
        "zotero_by_filename": 0,
        "zotero_by_doi": 0,
        "zotero_by_title": 0,
        "zotero_unmatched": 0,
        "zotero_ambiguous": 0,
        "pdfs_excluded": 0,
        "pdfs_kept_previous": 0,
        "pdfs_duplicates_skipped": 0,
        "pdfs_renamed": 0,
        "pdfs_moved": 0,
        "pdfs_root_unknown": 0,
        "pdfs_root_adopted": 0,
        "pdfs_unchecked": 0,
        "zotero_kept": 0,
        "zotero_unchanged": 0,
    }
    pdf_paths, n_excluded, routes = _select_pdf_routes(pdf_dir, recursive, exclude)
    index = _DiskIndex(pdf_dir, routes)  # this run's folder listings and canonical routes
    stats["pdfs_excluded"] = n_excluded
    if not recursive:
        progress(f"[ingest] {pdf_dir}: top level only (sub-folders not searched)")
    if n_excluded:
        progress(f"[ingest] {n_excluded} file(s) skipped by exclude patterns: {exclude}")
    if not pdf_paths and not prune_missing:
        detail = (
            f"all {n_excluded} PDFs matched the exclude patterns {exclude}"
            if n_excluded
            else "no PDF files found"
            + ("" if recursive else " at the top level (sub-folders not searched)")
        )
        raise FileNotFoundError(
            f"Nothing to ingest under {pdf_dir}: {detail}. "
            "Check the --pdf-dir path"
            + (", loosen --exclude," if n_excluded else "")
            + " or drop --no-recursive if the PDFs live in sub-folders."
        )
    # Zotero joins by normalized basename, which cannot tell same-named files
    # in different folders apart. An empty key (fully non-ASCII name) is not a
    # shared basename: such files keep content-based matching.
    shared_basenames = {
        key for key, n in Counter(normalize_key(p.name) for p in pdf_paths).items() if key and n > 1
    }
    unreadable: list[str] = []
    failed_renames: set[str] = set()
    unchecked: set[str] = set()  # stored relpaths whose folder could not be listed
    file_digests: dict[str, str] = {}  # NFC relpath -> sha256 of every file read this run
    moves: list[tuple[str, str | None, str]] = []  # (old relpath, its root, new relpath) re-keyed
    with corpus_lock(config, progress):
        collection = get_collection(config, create=True)
        chunk_size, chunk_overlap = _chunk_settings(config)
        stored_sources = _stored_pdf_sources(collection)
        for pdf_path in pdf_paths:
            identity = unicodedata.normalize("NFC", str(pdf_path.relative_to(pdf_dir)))
            for spelling in {identity, unicodedata.normalize("NFD", identity)}:
                _refuse_source_kind(collection, [spelling], "pdf")
                root = (stored_sources.get(spelling) or {}).get("pdf_root")
                if root is not None and root != pdf_root:
                    raise ValueError(
                        f"Source {identity!r} already belongs to PDF root {root!r}, "
                        f"not {pdf_root!r}. Use a separate corpus directory for this root "
                        "or organize both libraries beneath one shared PDF root."
                    )
        for pdf_path in pdf_paths:
            stats["pdfs_seen"] += 1
            # Identity is the NFC relpath: macOS and Linux spell accented names
            # differently, and both spellings must resolve to one source.
            disk_rel = str(pdf_path.relative_to(pdf_dir))
            rel_name = unicodedata.normalize("NFC", disk_rel)
            alternates = _alternate_spellings(collection, rel_name, disk_rel)
            try:
                source_sha256 = file_digest(pdf_path)
            except OSError as exc:
                state, _ = _ingest_state(collection, rel_name)
                stats["pdfs_unreadable"] += 1
                unreadable.append(rel_name + _failed_refresh(collection, rel_name, state, stats))
                progress(
                    f"  [ingest] {rel_name}: could not read the file ({type(exc).__name__}: {exc}); "
                    + _failed_refresh_note(state)
                )
                continue
            file_digests[rel_name] = source_sha256
            current = {
                "source_file_sha256": source_sha256,
                "chunk_size_words": chunk_size,
                "chunk_overlap_words": chunk_overlap,
                "sentence_segmenter": SENTENCE_SEGMENTER,
            }
            stored = stored_sources.get(rel_name)
            if (
                stored
                and stored["pdf_root"] is None
                and stored["source_file_sha256"] == source_sha256
            ):
                _adopt_root(collection, rel_name, pdf_root, progress)
                stored["pdf_root"] = pdf_root
                stats["pdfs_root_adopted"] += 1
            state, reason = _ingest_state(collection, rel_name, current)
            if state == "stale":
                stats["pdfs_stale"] += 1
            if state == "complete" and not reingest:
                _delete_sources(collection, alternates, rel_name, progress)
                if zotero:
                    shared = normalize_key(pdf_path.name) in shared_basenames
                    try:
                        _refresh_zotero_metadata(
                            collection, pdf_path, rel_name, zotero, shared, stats, progress
                        )
                    except IngestRollbackError:
                        raise
                    except Exception as exc:  # extraction failed or the update was rolled back
                        stats["pdfs_unreadable"] += 1
                        unreadable.append(
                            rel_name + _failed_refresh(collection, rel_name, state, stats)
                        )
                        progress(
                            f"  [zotero] {rel_name}: could not be read to match the new Zotero "
                            f"export ({type(exc).__name__}: {exc}); " + _failed_refresh_note(state)
                        )
                        continue
                stats["pdfs_skipped"] += 1
                continue
            copies = _same_file_elsewhere(stored_sources, rel_name, source_sha256, alternates)
            # A stored digest can describe an earlier version of a live file.
            # That old version must not suppress a new source containing those bytes.
            # A copy without a recorded root is looked for under this root:
            # gone from here, it is this file renamed (an older store's source).
            copies = [
                (name, root)
                for name, root in copies
                if _copy_still_matches(
                    pdf_dir if root in (None, pdf_root) else Path(root),
                    name,
                    source_sha256,
                    unchecked,
                    index,
                )
            ]
            # A copy stored under a link route to this very file is not a
            # second copy: the canonical route replaces it (a rename). A copy
            # that is not an alias route is its own canonical route, so the
            # exclusion check needs no further resolution.
            present = []
            for relpath, root in copies:
                copy_root = pdf_dir if root in (None, pdf_root) else Path(root)
                copy_routes = routes if copy_root is pdf_dir else None
                if not _exists_or_unchecked(copy_root, relpath, unchecked, index):
                    continue
                if _is_alias_route(copy_root, relpath, copy_routes, index):
                    continue
                if (
                    prune_missing
                    and root == pdf_root
                    and _is_excluded(Path(_nfc(relpath)), exclude, recursive=recursive)
                ):
                    continue
                present.append((relpath, root))
            if present:
                stats["pdfs_duplicates_skipped"] += 1
                if state != "absent":
                    # This path used to hold different content. Keeping that old
                    # version would retain a document that no longer exists.
                    collection.delete(where={"pdf_relpath": rel_name})
                    stored_sources.pop(rel_name, None)
                    progress(
                        f"  [ingest] {rel_name}: removed its obsolete stored version; the current content is retained under the duplicate source"
                    )
                progress(
                    f"  [warn] {rel_name} (under {pdf_root}) is the same file as the stored "
                    + ", ".join(
                        f"{relpath!r} (under {root or 'an unrecorded root'})"
                        for relpath, root in present
                    )
                    + "; skipping this copy so the document is not counted twice. Remove one "
                    "copy, or ingest the libraries from one shared PDF root."
                )
                continue
            renamed = [relpath for relpath, _ in copies]
            replaced = {rel_name, *alternates, *renamed}
            reusable = None if reingest else _reusable_copy(collection, renamed, current)
            if reusable is not None:
                # The same bytes are already stored, complete and current, under
                # another name (a file renamed on disk, or a route change: an
                # earlier-sorting link now reaches the folder). Re-key those
                # chunks under this identity instead of extracting and
                # embedding the file again.
                try:
                    acquisition_date = file_acquisition_date(pdf_path)
                except Exception as exc:
                    stats["pdfs_unreadable"] += 1
                    failed_renames.update(renamed)
                    unreadable.append(
                        rel_name + _failed_refresh(collection, rel_name, state, stats)
                    )
                    progress(
                        f"  [ingest] {rel_name}: could not read its file properties "
                        f"({type(exc).__name__}: {exc}); stored copies kept for retry"
                    )
                    continue
                identity = {
                    "pdf_filename": _nfc(pdf_path.name),
                    "pdf_relpath": rel_name,
                    "pdf_root": pdf_root,
                    "date_added_fs": acquisition_date,
                }
                kept = None if zotero else _stored_zotero_fields(collection, [rel_name, *renamed])
                prepared = _rekeyed_chunks(
                    collection, reusable, _relpath_key(rel_name), identity, kept
                )
                _refuse_foreign_ids(collection, prepared, replaced)
                _replace_sources(collection, prepared, replaced)
                for relpath, root in copies:
                    progress(
                        f"  [ingest] {rel_name}: replacing the stored copy {relpath!r} "
                        "(same bytes; its text and embeddings were re-keyed, not rebuilt)"
                    )
                    stored_sources.pop(relpath, None)
                    moves.append((relpath, root, rel_name))
                stored_sources[rel_name] = {
                    "pdf_root": pdf_root,
                    "source_file_sha256": source_sha256,
                }
                if kept:
                    stats["zotero_kept"] += 1
                    progress(
                        f"  [zotero] {rel_name}: no --zotero-csv given; kept its existing "
                        "Zotero metadata"
                    )
                if zotero:
                    shared = normalize_key(pdf_path.name) in shared_basenames
                    try:
                        _refresh_zotero_metadata(
                            collection, pdf_path, rel_name, zotero, shared, stats, progress
                        )
                    except IngestRollbackError:
                        raise
                    except Exception as exc:  # extraction failed or the update was rolled back
                        stats["pdfs_unreadable"] += 1
                        unreadable.append(
                            rel_name + _failed_refresh(collection, rel_name, "complete", stats)
                        )
                        progress(
                            f"  [zotero] {rel_name}: could not be read to match the new Zotero "
                            f"export ({type(exc).__name__}: {exc}); "
                            + _failed_refresh_note("complete")
                        )
                        continue
                # Counted only once the refresh succeeds: a failed refresh is
                # reported, as in the complete path, as unreadable with its
                # stored version (now under this name) kept, not also ingested.
                stats["pdfs_renamed"] += len(renamed)
                stats["pdfs_moved"] += 1
                stats["pdfs_ingested"] += 1
                stats["pdfs_refreshed"] += state != "absent"
                progress(
                    f"  [ingest] {rel_name}: {len(prepared['ids'])} chunks re-keyed from {reusable!r}"
                )
                continue
            if reason:
                progress(f"  [ingest] {rel_name}: {reason}; re-ingesting it whole")
            # A renamed file's old copies are the document's only stored version
            # when nothing remains under this name or a link route to it. If the
            # rebuild below fails, they are then kept out of the missing sources
            # --prune-missing removes, as the re-key path keeps them; a stored
            # link route is retained by the symlink-route check instead.
            sole_copies = (
                renamed
                if renamed
                and state in ("absent", "partial")
                and not any(
                    name != rel_name and index.canonical(name) == rel_name
                    for name, info in stored_sources.items()
                    if info["pdf_root"] == pdf_root
                )
                else []
            )
            try:
                raw_text = extract_pdf_text(pdf_path)
                acquisition_date = file_acquisition_date(pdf_path)
            except Exception as exc:  # any extractor failure makes this one file unreadable
                stats["pdfs_unreadable"] += 1
                failed_renames.update(sole_copies)
                unreadable.append(rel_name + _failed_refresh(collection, rel_name, state, stats))
                progress(
                    f"  [ingest] {rel_name}: could not be read ({type(exc).__name__}: {exc}); "
                    + _failed_refresh_note(state)
                )
                continue
            # NFC: extractors often emit decomposed accents; terms are typed composed.
            text = unicodedata.normalize("NFC", raw_text)
            chunks = chunk_words(text, chunk_size, chunk_overlap)
            if not chunks:
                stats["pdfs_no_text"] += 1
                failed_renames.update(sole_copies)
                _failed_refresh(collection, rel_name, state, stats)
                progress(
                    f"  [ingest] {rel_name}: no extractable text (a scan without OCR?); "
                    + _failed_refresh_note(state)
                )
                continue
            try:
                contexts = sentence_context_records(text, chunk_size, chunk_overlap, len(chunks))
            except ValueError as exc:
                stats["pdfs_unreadable"] += 1
                failed_renames.update(sole_copies)
                unreadable.append(rel_name + _failed_refresh(collection, rel_name, state, stats))
                progress(f"  [ingest] {rel_name}: {exc} " + _failed_refresh_note(state))
                continue

            base_meta = {
                "pdf_filename": _nfc(pdf_path.name),
                "pdf_relpath": rel_name,
                "pdf_root": pdf_root,
                "chunk_size_words": chunk_size,
                "chunk_overlap_words": chunk_overlap,
                "total_chunks": len(chunks),
                "source_file_sha256": source_sha256,
                "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "sentence_segmenter": SENTENCE_SEGMENTER,
                "date_added_fs": acquisition_date,
            }
            if zotero:
                base_meta["zotero_csv_sha256"] = zotero["sha256"]
                base_meta["zotero_match_filename"] = _nfc(pdf_path.name)
                shared = normalize_key(pdf_path.name) in shared_basenames
                base_meta.update(
                    _zotero_fields(pdf_path, rel_name, text, zotero, shared, stats, progress)
                )
            else:
                kept = _stored_zotero_fields(collection, [rel_name, *alternates, *renamed])
                if kept:
                    stats["zotero_kept"] += 1
                    base_meta.update(kept)
                    progress(
                        f"  [zotero] {rel_name}: no --zotero-csv given; kept its existing "
                        "Zotero metadata"
                    )
            prepared = _prepare_chunks(
                embedder, _relpath_key(rel_name), chunks, base_meta, contexts=contexts
            )
            _refuse_foreign_ids(collection, prepared, replaced)
            _replace_sources(collection, prepared, replaced)
            if alternates:
                progress(
                    f"  [ingest] {rel_name}: replaced chunks stored under a different Unicode spelling"
                )
            for relpath in renamed:
                progress(f"  [ingest] {rel_name}: replacing the stored copy {relpath!r}")
                stored_sources.pop(relpath, None)
            stats["pdfs_renamed"] += len(renamed)
            stored_sources[rel_name] = {"pdf_root": pdf_root, "source_file_sha256": source_sha256}
            stats["pdfs_ingested"] += 1
            stats["pdfs_refreshed"] += state != "absent"
            stats["chunks_added"] += len(chunks)
            progress(f"  [ingest] {rel_name}: {len(chunks)} chunks")
        for line in _route_moves_summary(moves, pdf_dir, pdf_root, index):
            progress(line)
        _handle_missing(
            collection,
            pdf_dir,
            pdf_root,
            prune_missing,
            stats,
            progress,
            exclude=exclude,
            recursive=recursive,
            unchecked=unchecked,
            stored_sources=stored_sources,
            file_digests=file_digests,
            routes=routes,
            index=index,
            failed_renames=failed_renames,
        )

    progress(
        f"[done] {stats['pdfs_ingested']} ingested ({stats['pdfs_refreshed']} refreshed, "
        f"{stats['pdfs_stale']} of them stale, {stats['pdfs_renamed']} renamed), "
        f"{stats['pdfs_skipped']} unchanged and skipped "
        f"({stats['metadata_updated']} with Zotero metadata updated in place), "
        f"{stats['pdfs_duplicates_skipped']} duplicate copies skipped, "
        f"{stats['pdfs_no_text']} with no extractable text, "
        f"{stats['pdfs_unreadable']} unreadable ({stats['pdfs_kept_previous']} previous "
        f"versions kept), {stats['pdfs_missing']} missing from disk, "
        f"{stats['pdfs_aliased']} stored under a symlink route, "
        f"{stats['pdfs_stored_excluded']} stored but now excluded "
        f"({stats['pdfs_pruned']} removed); {stats['chunks_added']} chunks added"
    )
    if unreadable:
        progress("[warn] unreadable PDFs:\n" + "\n".join(f"    {r}" for r in unreadable))
    if zotero:
        progress(
            f"[zotero] matched {stats['zotero_by_filename']} by filename, "
            f"{stats['zotero_by_doi']} by DOI, {stats['zotero_by_title']} by title; "
            f"{stats['zotero_unmatched']} unmatched, {stats['zotero_ambiguous']} of them "
            f"ambiguous; {stats['zotero_unchanged']} unchanged since this export was last "
            "applied (unmatched files keep the filesystem acquisition date only)"
        )
    return stats


def ingest_records(
    config: Config,
    embedder: Embedder,
    records: list[dict],
    *,
    text_field: str = "abstract",
    source_field: str = "source",
    progress: Callable[[str], None] = print,
) -> dict[str, int]:
    """Ingest pre-extracted text records (used by the toy-corpus demo).

    Each record needs a non-empty ``text_field`` (the document text) and
    ``source_field`` (a file-name-like identifier); a record lacking either
    raises ``ValueError``. title/year/doi/zotero_tags/self_authored/date_added
    are copied into chunk metadata when present. A stored record is reused
    only when its text, metadata, chunk settings and sentence segmenter are unchanged.
    """
    _refuse_incomplete_records(records, text_field=text_field, source_field=source_field)
    _refuse_colliding_record_keys(records, source_field=source_field)

    stats = {"records_seen": 0, "records_ingested": 0, "records_stale": 0, "chunks_added": 0}
    with corpus_lock(config, progress):
        collection = get_collection(config, create=True)
        chunk_size, chunk_overlap = _chunk_settings(config)
        _refuse_source_kind(
            collection, [str(record[source_field]).strip() for record in records], "record"
        )
        for record in records:
            stats["records_seen"] += 1
            # NFC for the same reason as PDF text (see ingest_pdf_dir).
            text = unicodedata.normalize("NFC", str(record[text_field]).strip())
            source = str(record[source_field]).strip()
            text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            fields = {
                key: str(record[key])
                for key in ("title", "year", "doi", "zotero_tags", "self_authored", "date_added")
                if record.get(key)
            }
            current = {
                "source_text_sha256": text_sha256,
                "sentence_segmenter": SENTENCE_SEGMENTER,
                "chunk_size_words": chunk_size,
                "chunk_overlap_words": chunk_overlap,
                "record_metadata_sha256": digest(fields),
            }
            state, reason = _ingest_state(collection, source, current)
            if state == "complete":
                continue
            if state == "stale":
                stats["records_stale"] += 1
            if reason:
                progress(f"  [ingest] {source}: {reason}; re-ingesting it")
            chunks = chunk_words(text, chunk_size, chunk_overlap)
            # Records carry a unique ``source``; use it as both the display
            # filename and the identity (pdf_relpath), matching ingest_pdf_dir.
            base_meta = {
                "pdf_filename": source,
                "pdf_relpath": source,
                "total_chunks": len(chunks),
                "source_text_sha256": text_sha256,
                "sentence_segmenter": SENTENCE_SEGMENTER,
            }
            base_meta.update(current)
            base_meta.update(fields)
            # normalize_key of the unique source keeps demo chunk ids stable.
            # The new chunks are built and embedded before the old ones go.
            prepared = _prepare_chunks(
                embedder,
                normalize_key(source),
                chunks,
                base_meta,
                contexts=sentence_context_records(text, chunk_size, chunk_overlap, len(chunks)),
            )
            _refuse_foreign_ids(collection, prepared, {source})
            _replace_sources(collection, prepared, {source})
            stats["records_ingested"] += 1
            stats["chunks_added"] += len(chunks)
    progress(
        f"[done] {stats['records_ingested']}/{stats['records_seen']} records ingested "
        f"({stats['records_stale']} stale), {stats['chunks_added']} chunks"
    )
    return stats


def _refuse_incomplete_records(records: list[dict], *, text_field: str, source_field: str):
    """A record without text or source would otherwise vanish from the corpus unreported."""
    incomplete = [
        i
        for i, record in enumerate(records)
        if not str(record.get(text_field) or "").strip()
        or not str(record.get(source_field) or "").strip()
    ]
    if incomplete:
        raise ValueError(
            f"Records at positions {incomplete} lack a non-empty '{text_field}' or "
            f"'{source_field}' field; every record must have both."
        )


def _refuse_colliding_record_keys(records: list[dict], *, source_field: str):
    """Refuse records whose distinct sources reduce to the same chunk-id key.

    ``normalize_key`` drops punctuation and case, and Chroma's ``add`` keeps
    an existing id without an error, so a collision would silently lose one
    record. Records keep this key (not ``_relpath_key``) so demo chunk ids,
    which recorded fixtures depend on, stay stable.
    """
    by_key: dict[str, set[str]] = {}
    seen_sources: set[str] = set()
    for record in records:
        source = str(record[source_field]).strip()
        if source in seen_sources:
            raise ValueError(
                f"Duplicate record source {source!r}; supply each source exactly once."
            )
        seen_sources.add(source)
        by_key.setdefault(normalize_key(source), set()).add(source)
    collisions = {key: names for key, names in by_key.items() if len(names) > 1}
    if not collisions:
        return
    detail = "; ".join(
        f"{', '.join(sorted(names))} → '{key}'" for key, names in sorted(collisions.items())
    )
    raise ValueError(
        "These records have different names that reduce to the same chunk-id "
        f"key, so one would silently overwrite the other: {detail}. "
        "Chunk ids ignore case and punctuation, so rename the sources to differ "
        "by a letter or digit (not only by '_' vs '-', or by case), then re-run."
    )


def _relpath_key(relpath: str) -> str:
    """Unique chunk-id key: readable slug plus a short hash of the relpath.

    The slug alone is not injective (case, punctuation, non-ASCII names), and
    Chroma's ``add`` silently keeps an existing id.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", relpath.lower()).strip("_") or "source"
    digest = hashlib.sha256(relpath.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def _prepare_chunks(embedder, id_key, chunks, base_meta, batch_size=64, *, contexts=None) -> dict:
    """Ids, texts, metadata, and embeddings for one source, ready for ``_store_chunks``.

    Everything that can fail per source (embedding included) happens here,
    before the store is touched. ``contexts`` come from
    ``sentence_context_records``; Chroma metadata must be scalar, so sentence
    lengths are stored as a JSON list.
    """
    metadatas = [dict(base_meta, chunk_index=i) for i in range(len(chunks))]
    for metadata, (first, context, lengths) in zip(metadatas, contexts or [], strict=False):
        metadata["sentence_start"] = first
        metadata["sentence_context"] = context
        metadata["sentence_lengths"] = json.dumps(lengths)
    embeddings = []
    for start in range(0, len(chunks), batch_size):
        embeddings.extend(embedder.embed(chunks[start : start + batch_size]))
    return {
        "id_key": id_key,
        "source": str(base_meta.get("pdf_relpath") or id_key),
        "ids": [f"{id_key}::c{i}" for i in range(len(chunks))],
        "documents": list(chunks),
        "embeddings": embeddings,
        "metadatas": metadatas,
    }


def _collision_error(prepared: dict, stored: int, expected: int) -> RuntimeError:
    return RuntimeError(
        f"Only {stored} of {expected} chunks from '{prepared['source']}' were "
        f"stored: the corpus already holds chunk ids beginning "
        f"'{prepared['id_key']}::', and ChromaDB keeps the existing rows without "
        "reporting a conflict.\n"
        "Another source in this corpus produced the same chunk ids. "
        "Rename that file so the two paths differ, or delete the "
        "ChromaDB store and rebuild it with `cda ingest`."
    )


def _refuse_foreign_ids(collection, prepared: dict, replaced: set[str]) -> None:
    """Refuse before any deletion when another source already holds these chunk ids."""
    existing = collection.get(ids=prepared["ids"], include=["metadatas"])
    foreign = [
        m for m in existing.get("metadatas") or [] if (m or {}).get("pdf_relpath") not in replaced
    ]
    if foreign:
        raise _collision_error(prepared, len(prepared["ids"]) - len(foreign), len(prepared["ids"]))


def _store_chunks(collection, prepared: dict, batch_size: int = 64) -> None:
    """Add prepared chunks in batches, checking that every batch landed.

    Chroma's ``add`` is a silent no-op for an existing id, so each batch is
    checked by comparing the collection count.
    """
    keys = ("ids", "documents", "embeddings", "metadatas")
    for start in range(0, len(prepared["ids"]), batch_size):
        batch = {key: prepared[key][start : start + batch_size] for key in keys}
        before = collection.count()
        collection.add(**batch)
        stored = collection.count() - before
        if stored != len(batch["ids"]):
            raise _collision_error(prepared, stored, len(batch["ids"]))


def _replace_sources(collection, prepared: dict, replaced: set[str]) -> None:
    """Replace sources, restoring their exact rows after a handled write failure.

    This protects failed batch writes, including a write that persisted before
    raising. It is not a database transaction across process or machine crashes.
    """
    where = {"pdf_relpath": {"$in": sorted(replaced)}}
    backup = collection.get(where=where, include=["documents", "metadatas", "embeddings"])
    try:
        collection.delete(where=where)
        _store_chunks(collection, prepared)
    except (Exception, KeyboardInterrupt) as original:
        try:
            collection.delete(where=where)
            if backup["ids"]:
                restore = {
                    key: backup[key] for key in ("ids", "documents", "metadatas", "embeddings")
                }
                _store_chunks(
                    collection, dict(restore, id_key=prepared["id_key"], source=prepared["source"])
                )
        except Exception as rollback:
            raise IngestRollbackError(
                f"Writing {prepared['source']!r} failed ({original}); restoring its "
                f"previous chunks also failed ({rollback}). The corpus may be incomplete; "
                "repair the storage and re-ingest before analysis."
            ) from original
        raise


def _add_chunks(collection, embedder, id_key, chunks, base_meta, batch_size=64, *, contexts=None):
    """Embed and store chunks; ``id_key`` must be unique per source."""
    prepared = _prepare_chunks(
        embedder, id_key, chunks, base_meta, batch_size=batch_size, contexts=contexts
    )
    _store_chunks(collection, prepared, batch_size=batch_size)
