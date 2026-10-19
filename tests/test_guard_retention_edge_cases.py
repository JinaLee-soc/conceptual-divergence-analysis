"""The corpus guard on hostile filesystems, and alias-route retention accounting.

Regressions for: a filesystem that refuses ``flock`` stopping corpus
operations safely, a guard another OS user cannot open, a lock record carrying this
very pid (a container's PID 1) that was never stale, a lock entry that is not
a readable file stalling every caller, and a source stored under a symlink
route being retained forever (counted as a kept previous version, never
prunable) once its canonical file was deleted or stored elsewhere.
"""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import cda.corpus.ingest as ing
from cda.cli import main
from cda.config import load_config
from cda.corpus.ingest import CorpusLockError, corpus_lock, get_collection

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="flock and POSIX modes")
not_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file modes"
)


def _quiet(_message: str) -> None:
    pass


def _config(tmp_path):
    return load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})


def _stored(config) -> set[str]:
    metadatas = get_collection(config).get(include=["metadatas"])["metadatas"]
    return {m["pdf_relpath"] for m in metadatas}


def _lock_file(config) -> Path:
    store = ing._store_directory(config)
    store.mkdir(parents=True, exist_ok=True)
    return store / ".ingest.lock"


def _guard_file(config) -> Path:
    return _lock_file(config).with_name(".ingest.guard")


@pytest.fixture
def fast_lock(monkeypatch):
    monkeypatch.setattr(ing, "_INGEST_LOCK_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(ing, "_INGEST_LOCK_POLL_SECONDS", 0.01)


@pytest.fixture
def live_holder():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        yield child.pid
    finally:
        child.kill()
        child.wait()


def _failing_flock(code):
    def flock(fd, operation):
        raise OSError(code, os.strerror(code))

    return flock


# --- A1: a filesystem without flock stops before touching the corpus --------


@posix_only
@pytest.mark.parametrize("code", sorted(ing._GUARD_UNSUPPORTED))
def test_unsupported_flock_refuses_the_operation_with_recovery_advice(
    tmp_path, fast_lock, monkeypatch, code
):
    import fcntl

    config = _config(tmp_path)
    lock = _lock_file(config)
    monkeypatch.setattr(fcntl, "flock", _failing_flock(code))
    with pytest.raises(CorpusLockError, match="Cannot safely lock") as excinfo:
        with corpus_lock(config, _quiet):
            pytest.fail("corpus operation started without the OS guard")
    message = str(excinfo.value)
    assert errno.errorcode[code] in message and ".ingest.guard" in message
    assert "filesystem that supports OS file locks" in message
    assert not lock.exists()


@posix_only
def test_unsupported_guard_leaves_a_record_of_this_pid_untouched(tmp_path, fast_lock, monkeypatch):
    """Do not reclaim any PID record when the guard cannot be acquired."""
    import fcntl

    config = _config(tmp_path)
    lock = _lock_file(config)
    lock.write_text(f"{os.getpid()}\n")
    monkeypatch.setattr(fcntl, "flock", _failing_flock(errno.ENOLCK))
    with pytest.raises(CorpusLockError):
        with corpus_lock(config, _quiet):
            pass
    assert lock.read_text().strip() == str(os.getpid())


@posix_only
def test_unsupported_guard_does_not_leave_its_descriptor_open(tmp_path, fast_lock, monkeypatch):
    import fcntl

    config = _config(tmp_path)
    real_flock = fcntl.flock
    with monkeypatch.context() as patch:
        patch.setattr(fcntl, "flock", _failing_flock(errno.ENOLCK))
        with pytest.raises(CorpusLockError):
            with corpus_lock(config, _quiet):
                pytest.fail("corpus operation started without the OS guard")
    # The failed acquisition closed its descriptor: a real flock succeeds now.
    fd = os.open(_guard_file(config), os.O_RDONLY)
    try:
        real_flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)
    with corpus_lock(config, _quiet):
        pass


@posix_only
def test_other_flock_errors_still_raise(tmp_path, fast_lock, monkeypatch):
    import fcntl

    config = _config(tmp_path)
    monkeypatch.setattr(fcntl, "flock", _failing_flock(errno.EBADF))
    with pytest.raises(OSError):
        with corpus_lock(config, _quiet):
            pass


@posix_only
def test_cli_reports_unsupported_locking_before_creating_a_store(
    tmp_path, fast_lock, fake_embedder, monkeypatch, capsys
):
    import fcntl

    monkeypatch.setattr(ing, "extract_pdf_text", lambda path: Path(path).read_text())
    monkeypatch.setattr("cda.corpus.embeddings.get_embedder", lambda config: fake_embedder)
    monkeypatch.setattr(fcntl, "flock", _failing_flock(errno.ENOLCK))
    library = tmp_path / "pdfs"
    library.mkdir()
    (library / "paper.pdf").write_text("Novelty survives a network share.")
    code = main(["ingest", "--pdf-dir", str(library), "--corpus-dir", str(tmp_path / "corpus")])
    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert "Cannot safely lock" in captured.err
    assert not (ing._store_directory(_config(tmp_path)) / "chroma.sqlite3").exists()


# --- A2: the guard is shareable between OS users ------------------------------


@posix_only
def test_guard_is_created_readable_by_other_users(tmp_path):
    config = _config(tmp_path)
    with corpus_lock(config, _quiet):
        pass
    mask = os.umask(0)
    os.umask(mask)
    assert _guard_file(config).stat().st_mode & 0o777 == 0o666 & ~mask


@posix_only
@not_root
def test_read_only_guard_can_still_be_locked(tmp_path, fast_lock):
    config = _config(tmp_path)
    with corpus_lock(config, _quiet):
        pass
    guard = _guard_file(config)
    os.chmod(guard, 0o400)
    try:
        with corpus_lock(config, _quiet):
            pass
    finally:
        os.chmod(guard, 0o600)


@posix_only
@not_root
def test_unopenable_guard_is_reported_with_the_fix(tmp_path, fast_lock):
    config = _config(tmp_path)
    with corpus_lock(config, _quiet):
        pass
    guard = _guard_file(config)
    os.chmod(guard, 0o000)
    try:
        with pytest.raises(CorpusLockError) as excinfo:
            with corpus_lock(config, _quiet):
                pass
    finally:
        os.chmod(guard, 0o600)
    message = str(excinfo.value)
    assert str(guard) in message and f"uid {os.getuid()}" in message and "chmod a+r" in message
    assert "0000" in message


# --- A3: a record of this very pid is stale once the guard is held -----------


def test_record_of_this_pid_is_reclaimed_at_once(tmp_path, fast_lock):
    """Docker: streamlit is PID 1, `docker stop` leaves "1", the next container is PID 1 again."""
    config = _config(tmp_path)
    lock = _lock_file(config)
    lock.write_text(f"{os.getpid()}\n")
    old = time.time() - 100_000
    os.utime(lock, (old, old))
    messages: list[str] = []
    started = time.monotonic()
    with corpus_lock(config, messages.append):
        assert lock.read_text().strip() == str(os.getpid())
    assert time.monotonic() - started < 0.25
    assert any("stale lock" in m and "this process id" in m for m in messages)
    assert not lock.exists()


def test_record_of_another_live_pid_is_still_waited_on(tmp_path, fast_lock, live_holder):
    """A live pid that holds no guard may be a legacy holder: the PID file cannot tell."""
    config = _config(tmp_path)
    lock = _lock_file(config)
    lock.write_text(f"{live_holder}\n")
    messages: list[str] = []
    with pytest.raises(CorpusLockError, match="ingest.lock"):
        with corpus_lock(config, messages.append):
            pass
    assert any("waiting for it to finish" in m for m in messages)
    assert lock.read_text().strip() == str(live_holder)


# --- B1: a lock entry that is not a readable file ----------------------------


def _age(path: Path, seconds: float) -> None:
    old = time.time() - seconds
    os.utime(path, (old, old), follow_symlinks=False)


def test_directory_at_the_lock_path_is_removed_once_older_than_a_poll(tmp_path, fast_lock):
    config = _config(tmp_path)
    lock = _lock_file(config)
    lock.mkdir()
    _age(lock, 10)
    messages: list[str] = []
    started = time.monotonic()
    with corpus_lock(config, messages.append):
        assert lock.is_file() and lock.read_text().strip() == str(os.getpid())
    assert time.monotonic() - started < 0.25
    assert any("not a lock file CDA can read" in m and "removed it" in m for m in messages)
    assert not lock.exists()
    assert not [p for p in lock.parent.iterdir() if p.name.startswith(".ingest.lock")]


def test_non_empty_directory_at_the_lock_path_is_moved_aside(tmp_path, fast_lock):
    config = _config(tmp_path)
    lock = _lock_file(config)
    lock.mkdir()
    (lock / "note.txt").write_text("not a lock")
    _age(lock, 10)
    messages: list[str] = []
    with corpus_lock(config, messages.append):
        assert lock.is_file()
    aside = [p for p in lock.parent.iterdir() if p.name.startswith(".ingest.lock.stale-")]
    assert len(aside) == 1 and (aside[0] / "note.txt").read_text() == "not a lock"
    assert any("moved it aside to " + aside[0].name in m for m in messages)


@posix_only
@not_root
def test_unreadable_lock_file_is_removed_once_older_than_a_poll(tmp_path, fast_lock):
    config = _config(tmp_path)
    lock = _lock_file(config)
    lock.write_text("secret holder\n")
    os.chmod(lock, 0o000)
    _age(lock, 10)
    with corpus_lock(config, _quiet):
        assert lock.read_text().strip() == str(os.getpid())
    assert not lock.exists()


def test_fresh_directory_at_the_lock_path_is_waited_on_first(tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "_INGEST_LOCK_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(ing, "_INGEST_LOCK_POLL_SECONDS", 0.2)
    config = _config(tmp_path)
    lock = _lock_file(config)
    lock.mkdir()
    started = time.monotonic()
    with corpus_lock(config, _quiet):
        pass
    assert 0.2 <= time.monotonic() - started < 2


def test_unmovable_lock_directory_is_reported_by_name(tmp_path, fast_lock, monkeypatch):
    config = _config(tmp_path)
    lock = _lock_file(config)
    lock.mkdir()
    _age(lock, 10)
    rename = os.rename

    def refuse(source, destination, *args, **kwargs):
        if Path(source) == lock:
            raise PermissionError(errno.EACCES, "Operation not permitted")
        return rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(ing.os, "rename", refuse)
    with pytest.raises(CorpusLockError, match="could not be moved aside") as excinfo:
        with corpus_lock(config, _quiet):
            pass
    assert str(lock) in str(excinfo.value)
    assert lock.is_dir()


@posix_only
def test_unreadable_lock_entry_is_untouched_without_the_guard(tmp_path, fast_lock, monkeypatch):
    import fcntl

    config = _config(tmp_path)
    lock = _lock_file(config)
    lock.mkdir()
    _age(lock, 10)
    monkeypatch.setattr(fcntl, "flock", _failing_flock(errno.ENOLCK))
    with pytest.raises(CorpusLockError):
        with corpus_lock(config, _quiet):
            pass
    assert lock.is_dir()


# --- A4/A5/B3: sources stored under a symlink route ---------------------------


@pytest.fixture
def text_pdfs(monkeypatch):
    monkeypatch.setattr(ing, "extract_pdf_text", lambda path: Path(path).read_text())


@pytest.fixture
def linked_library(tmp_path):
    """pdfs/real/x.pdf, pdfs/shortcuts/c -> ../real, pdfs/y.pdf."""
    root = tmp_path / "pdfs"
    (root / "real").mkdir(parents=True)
    (root / "shortcuts").mkdir()
    (root / "real" / "x.pdf").write_text("Novelty is stored under a link route. " * 30)
    (root / "y.pdf").write_text("Novelty is stored plainly. " * 30)
    os.symlink("../real", root / "shortcuts" / "c")
    return root


def _store_under(config, canonical: str, relpath: str) -> None:
    """Re-key the stored chunks of ``canonical`` to ``relpath``, as an older store would hold them."""
    collection = get_collection(config)
    rows = collection.get(
        where={"pdf_relpath": canonical}, include=["metadatas", "documents", "embeddings"]
    )
    collection.delete(ids=rows["ids"])
    key = ing._relpath_key(relpath)
    collection.add(
        ids=[f"{key}::c{m['chunk_index']}" for m in rows["metadatas"]],
        documents=rows["documents"],
        embeddings=rows["embeddings"],
        metadatas=[dict(m, pdf_relpath=relpath) for m in rows["metadatas"]],
    )


def _aliased_store(config, fake_embedder, library):
    ing.ingest_pdf_dir(config, fake_embedder, library, progress=_quiet)
    _store_under(config, "real/x.pdf", "shortcuts/c/x.pdf")
    assert _stored(config) == {"shortcuts/c/x.pdf", "y.pdf"}


def _done(messages: list[str]) -> str:
    return next(m for m in messages if m.startswith("[done]"))


@pytest.mark.parametrize("prune", [False, True])
def test_alias_whose_canonical_file_is_gone_is_a_missing_source(
    tmp_path, fake_embedder, text_pdfs, linked_library, prune
):
    config = _config(tmp_path)
    _aliased_store(config, fake_embedder, linked_library)
    (linked_library / "real" / "x.pdf").unlink()
    messages: list[str] = []
    stats = ing.ingest_pdf_dir(
        config, fake_embedder, linked_library, prune_missing=prune, progress=messages.append
    )
    assert stats["pdfs_missing"] == 1 and stats["pdfs_kept_previous"] == 0
    assert stats["pdfs_pruned"] == (1 if prune else 0)
    assert not any("retained" in m for m in messages)
    assert "1 missing from disk" in _done(messages)
    if prune:
        assert _stored(config) == {"y.pdf"}
        assert any("removed 1 source(s) no longer on disk" in m for m in messages)
    else:
        assert _stored(config) == {"shortcuts/c/x.pdf", "y.pdf"}
        assert any("shortcuts/c/x.pdf" in m and "--prune-missing" in m for m in messages)


@pytest.mark.parametrize("prune", [False, True])
def test_alias_of_a_file_stored_elsewhere_is_a_duplicate(
    tmp_path, fake_embedder, text_pdfs, linked_library, prune
):
    """The canonical file is skipped as a duplicate of a stored copy: the alias is removable."""
    config = _config(tmp_path)
    _aliased_store(config, fake_embedder, linked_library)
    (linked_library / "z").mkdir()
    (linked_library / "z" / "x.pdf").write_bytes((linked_library / "real" / "x.pdf").read_bytes())
    collection = get_collection(config)
    rows = collection.get(
        where={"pdf_relpath": "shortcuts/c/x.pdf"}, include=["metadatas", "documents", "embeddings"]
    )
    key = ing._relpath_key("z/x.pdf")
    collection.add(
        ids=[f"{key}::c{m['chunk_index']}" for m in rows["metadatas"]],
        documents=rows["documents"],
        embeddings=rows["embeddings"],
        metadatas=[dict(m, pdf_relpath="z/x.pdf") for m in rows["metadatas"]],
    )
    messages: list[str] = []
    stats = ing.ingest_pdf_dir(
        config, fake_embedder, linked_library, prune_missing=prune, progress=messages.append
    )
    assert stats["pdfs_duplicates_skipped"] == 1 and stats["pdfs_kept_previous"] == 0
    assert stats["pdfs_aliased"] == 1 and stats["pdfs_missing"] == 0
    assert "0 missing from disk, 1 stored under a symlink route" in _done(messages)
    if prune:
        assert stats["pdfs_pruned"] == 1 and _stored(config) == {"z/x.pdf", "y.pdf"}
    else:
        assert _stored(config) == {"shortcuts/c/x.pdf", "z/x.pdf", "y.pdf"}
        assert any("stored twice" in m and "shortcuts/c/x.pdf" in m for m in messages)


def test_alias_of_a_changed_file_is_reported_as_an_older_version(
    tmp_path, fake_embedder, text_pdfs, linked_library
):
    config = _config(tmp_path)
    _aliased_store(config, fake_embedder, linked_library)
    (linked_library / "real" / "x.pdf").write_text("Novelty was revised under the link. " * 40)
    messages: list[str] = []
    stats = ing.ingest_pdf_dir(config, fake_embedder, linked_library, progress=messages.append)
    assert _stored(config) == {"shortcuts/c/x.pdf", "real/x.pdf", "y.pdf"}
    assert stats["pdfs_aliased"] == 1 and stats["pdfs_missing"] == 0
    warning = next(m for m in messages if "shortcuts/c/x.pdf" in m and m.startswith("[warn]"))
    assert "older version" in warning and "counted" in warning and "--prune-missing" in warning
    assert "0 missing from disk, 1 stored under a symlink route" in _done(messages)
    stats = ing.ingest_pdf_dir(
        config, fake_embedder, linked_library, prune_missing=True, progress=_quiet
    )
    assert stats["pdfs_pruned"] == 1 and _stored(config) == {"real/x.pdf", "y.pdf"}


def test_alias_canonical_is_hashed_once_per_run_from_its_disk_path(
    tmp_path, fake_embedder, text_pdfs, monkeypatch
):
    """The loop's digest is reused; the NFD name on disk is never re-opened by its NFC relpath."""
    root = tmp_path / "pdfs"
    folder = "réal"  # NFD on disk, as macOS and synced folders spell it
    (root / folder).mkdir(parents=True)
    (root / "shortcuts").mkdir()
    (root / folder / "x.pdf").write_text("Novelty under an accented folder. " * 30)
    os.symlink(f"../{folder}", root / "shortcuts" / "c")
    config = _config(tmp_path)
    ing.ingest_pdf_dir(config, fake_embedder, root, progress=_quiet)
    _store_under(config, "réal/x.pdf", "shortcuts/c/x.pdf")
    (root / folder / "x.pdf").write_text("Novelty under an accented folder, revised. " * 30)
    opened: list[str] = []
    digest = ing.file_digest

    def counting(path):
        opened.append(os.fspath(path))
        return digest(path)

    monkeypatch.setattr(ing, "file_digest", counting)
    messages: list[str] = []
    stats = ing.ingest_pdf_dir(config, fake_embedder, root, progress=messages.append)
    assert stats["pdfs_aliased"] == 1 and stats["pdfs_kept_previous"] == 0
    assert len([p for p in opened if p.endswith("x.pdf")]) == 1
    assert all(folder in p for p in opened if p.endswith("x.pdf"))  # the on-disk spelling


def test_alias_is_retained_only_while_its_canonical_exists_but_cannot_be_stored(
    tmp_path, fake_embedder, text_pdfs, linked_library, monkeypatch
):
    """Retention does not depend on --prune-missing and ends once the file is stored."""
    config = _config(tmp_path)
    _aliased_store(config, fake_embedder, linked_library)
    extract = ing.extract_pdf_text

    def damaged_x(path):
        if path.name == "x.pdf":
            raise ValueError("damaged")
        return extract(path)

    monkeypatch.setattr(ing, "extract_pdf_text", damaged_x)
    messages: list[str] = []
    stats = ing.ingest_pdf_dir(config, fake_embedder, linked_library, progress=messages.append)
    assert _stored(config) == {"shortcuts/c/x.pdf", "y.pdf"}
    assert stats["pdfs_kept_previous"] == 1 and stats["pdfs_missing"] == 0
    assert stats["pdfs_aliased"] == 0
    warning = next(m for m in messages if "retained" in m)
    assert "replacement" in warning and "--prune-missing" in warning
    assert "shortcuts/c/x.pdf" in warning
    monkeypatch.setattr(ing, "extract_pdf_text", extract)
    stats = ing.ingest_pdf_dir(config, fake_embedder, linked_library, progress=_quiet)
    assert stats["pdfs_renamed"] == 1 and _stored(config) == {"real/x.pdf", "y.pdf"}
