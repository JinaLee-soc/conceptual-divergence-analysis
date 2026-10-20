"""The guard on network filesystems, rerun cost, route moves, and lock/report wording.

Regressions for: a guard opened read-only failing with a raw ``EBADF`` on
NFS; an unchanged rerun listing every ancestor folder of every stored source
(quadratic in the library size); an earlier-sorting link to an external
folder re-extracting and re-embedding every file it reaches; a timeout
message naming a PID file that does not exist and advising its removal; a
refused OS lock claiming nothing was started while leaving an empty store
that later passed for a corpus; and the ``[done]`` line counting retained
routes as "failed files".
"""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest

import cda.corpus.ingest as ing
from cda.config import load_config
from cda.corpus.ingest import CorpusLockError, corpus_lock, get_collection
from tests.test_guard_retention_edge_cases import _mark_stale, _store_under

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="flock and POSIX modes")
not_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file modes"
)


def _quiet(_message: str) -> None:
    pass


def _config(tmp_path):
    return load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})


def _guard_file(config) -> Path:
    store = ing._store_directory(config)
    store.mkdir(parents=True, exist_ok=True)
    return store / ".ingest.guard"


def _stored(config) -> set[str]:
    metadatas = get_collection(config).get(include=["metadatas"])["metadatas"]
    return {m["pdf_relpath"] for m in metadatas}


def _rows(config, relpath: str) -> list[tuple[str, list[float]]]:
    """(document, embedding) per chunk of ``relpath`` in chunk order."""
    result = get_collection(config).get(
        where={"pdf_relpath": relpath}, include=["documents", "metadatas", "embeddings"]
    )
    order = sorted(range(len(result["ids"])), key=lambda i: result["metadatas"][i]["chunk_index"])
    return [(result["documents"][i], [float(x) for x in result["embeddings"][i]]) for i in order]


def _assert_same_rows(actual, expected) -> None:
    """Same documents in the same chunk order; embeddings equal at float32 store precision."""
    assert [doc for doc, _ in actual] == [doc for doc, _ in expected]
    for (_, got), (_, want) in zip(actual, expected, strict=True):
        assert np.allclose(got, want, rtol=1e-6, atol=1e-7)


def _done(messages: list[str]) -> str:
    return next(m for m in messages if m.startswith("[done]"))


def _failing_flock(code: int):
    def flock(fd, operation):
        raise OSError(code, os.strerror(code))

    return flock


@pytest.fixture
def fast_lock(monkeypatch):
    monkeypatch.setattr(ing, "_INGEST_LOCK_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(ing, "_INGEST_LOCK_POLL_SECONDS", 0.01)


@pytest.fixture
def text_pdfs(monkeypatch):
    monkeypatch.setattr(ing, "extract_pdf_text", lambda path: Path(path).read_text())


class _CountingEmbedder:
    """Counts the texts embedded; same vectors as the hash embedder for equal texts."""

    name = "fake-counting-embedder"

    def __init__(self, inner):
        self.inner = inner
        self.texts = 0

    def embed(self, texts):
        self.texts += len(texts)
        return self.inner.embed(texts)


def _guard_open_flags(monkeypatch) -> list[int]:
    """Record the flags of every ``os.open`` of a ``.ingest.guard`` file."""
    real_open = os.open
    flags_seen: list[int] = []

    def recording_open(path, flags, *args, **kwargs):
        if os.path.basename(os.fspath(path)) == ".ingest.guard":
            flags_seen.append(flags)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)
    return flags_seen


# --- A1: a guard that NFS refuses to lock read-only -------------------------


@posix_only
def test_ebadf_from_the_guard_lock_is_reported_with_the_chmod(tmp_path, fast_lock, monkeypatch):
    """Linux NFS emulates flock with byte-range locks: LOCK_EX on a read-only descriptor is EBADF."""
    import fcntl

    config = _config(tmp_path)
    guard = _guard_file(config)
    monkeypatch.setattr(fcntl, "flock", _failing_flock(errno.EBADF))
    with pytest.raises(CorpusLockError) as excinfo:
        with corpus_lock(config, _quiet):
            pytest.fail("corpus operation started although the guard was not locked")
    message = str(excinfo.value)
    assert str(guard) in message and f"uid {os.getuid()}" in message
    assert f"chmod a+w {guard}" in message and "write access" in message
    assert "Traceback" not in message


@posix_only
def test_guard_is_opened_read_write_first(tmp_path, fast_lock, monkeypatch):
    config = _config(tmp_path)
    flags_seen = _guard_open_flags(monkeypatch)
    with corpus_lock(config, _quiet):
        pass
    assert flags_seen and flags_seen[0] & os.O_ACCMODE == os.O_RDWR
    assert flags_seen[0] & os.O_CREAT


@posix_only
@not_root
def test_guard_without_write_permission_falls_back_to_read_only_and_locks(
    tmp_path, fast_lock, monkeypatch
):
    import fcntl

    config = _config(tmp_path)
    with corpus_lock(config, _quiet):
        pass
    guard = _guard_file(config)
    os.chmod(guard, 0o400)
    try:
        flags_seen = _guard_open_flags(monkeypatch)
        with corpus_lock(config, _quiet):  # a local flock accepts a read-only descriptor
            pass
        assert [f & os.O_ACCMODE for f in flags_seen] == [os.O_RDWR, os.O_RDONLY]
        # On NFS the same descriptor is refused; the message says why.
        monkeypatch.setattr(fcntl, "flock", _failing_flock(errno.EBADF))
        with pytest.raises(CorpusLockError) as excinfo:
            with corpus_lock(config, _quiet):
                pass
    finally:
        os.chmod(guard, 0o600)
    message = str(excinfo.value)
    assert "without write access" in message and "0400" in message
    assert f"chmod a+w {guard}" in message


# --- A2: an unchanged rerun is linear in the number of folders --------------


def _zotero_style_library(root: Path, n: int) -> int:
    """storage/<KEY>/Paper i.pdf for i < n; returns the number of distinct folders."""
    storage = root / "storage"
    storage.mkdir(parents=True)
    for i in range(n):
        folder = storage / f"K{i:05d}AB"
        folder.mkdir()
        (folder / f"Paper {i} - Novelty in science.pdf").write_text(
            " ".join(f"Sentence {i} number {j} talks about novelty in science." for j in range(12))
        )
    return 2 + n  # root, storage, one folder per key


def test_unchanged_rerun_lists_each_folder_once(tmp_path, fake_embedder, text_pdfs, monkeypatch):
    config = _config(tmp_path)
    root = tmp_path / "library"
    folders = _zotero_style_library(root, 300)
    ing.ingest_pdf_dir(config, fake_embedder, root, progress=_quiet)
    real_listdir = os.listdir
    calls = {"n": 0}

    def counting_listdir(path="."):
        calls["n"] += 1
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", counting_listdir)
    stats = ing.ingest_pdf_dir(config, fake_embedder, root, progress=_quiet)
    assert stats["pdfs_skipped"] == 300 and stats["pdfs_missing"] == 0
    # Every check of a stored source goes through the run's listing cache:
    # O(folders), not O(sources x path depth x checks).
    assert calls["n"] <= folders + 8, calls


def test_exact_matching_holds_through_the_listing_cache(tmp_path, monkeypatch):
    root = tmp_path / "library"
    (root / "sub").mkdir(parents=True)
    nfd_name = unicodedata.normalize("NFD", "Café.pdf")
    (root / "sub" / nfd_name).write_text("x")
    (root / "sub" / "Smith2020.pdf").write_text("y")
    real_listdir = os.listdir
    calls = {"n": 0}

    def counting_listdir(path="."):
        calls["n"] += 1
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", counting_listdir)
    index = ing._DiskIndex(root)
    # NFC and NFD spellings resolve to the one file, which keeps its disk spelling.
    assert ing._exists_exactly(root, "sub/Café.pdf", index)
    assert ing._path_exactly(root, f"sub/{nfd_name}", index) == root / "sub" / nfd_name
    # Letter case is not ignored, even where the filesystem would ignore it.
    assert not ing._exists_exactly(root, "sub/smith2020.pdf", index)
    assert ing._exists_exactly(root, "sub/Smith2020.pdf", index)
    # A missing folder is not an error, and nothing was listed twice.
    assert ing._exists_exactly(root, "gone/x.pdf", index) is False
    assert calls["n"] == 2, calls  # root and sub, once each
    assert index.canonical("sub/Café.pdf") == "sub/Café.pdf"
    assert calls["n"] == 2


# --- A3: a route change moves sources without re-extraction or re-embedding --


def _external_library(tmp_path: Path) -> tuple[Path, Path]:
    """pdfs/b -> ../E (x, y, z) and pdfs/z_local.pdf; returns (root, E)."""
    external = tmp_path / "E"
    external.mkdir()
    for name in "xyz":
        (external / f"{name}.pdf").write_text(
            " ".join(f"Sentence {name} number {i} talks about novelty." for i in range(30))
        )
    root = tmp_path / "pdfs"
    root.mkdir()
    (root / "z_local.pdf").write_text("Local novelty. " * 30)
    os.symlink("../E", root / "b")
    return root, external


def test_earlier_link_moves_sources_without_re_extraction_or_re_embedding(
    tmp_path, fake_embedder, text_pdfs, monkeypatch
):
    config = _config(tmp_path)
    root, _ = _external_library(tmp_path)
    ing.ingest_pdf_dir(config, fake_embedder, root, progress=_quiet)
    assert _stored(config) == {"b/x.pdf", "b/y.pdf", "b/z.pdf", "z_local.pdf"}
    collection = get_collection(config)
    ids = collection.get(where={"pdf_relpath": "b/x.pdf"}, include=[])["ids"]
    collection.update(ids=ids, metadatas=[{"title": "X paper"}] * len(ids))
    before = {name: _rows(config, f"b/{name}.pdf") for name in "xyz"}
    extractions: list[Path] = []
    monkeypatch.setattr(
        ing, "extract_pdf_text", lambda path: extractions.append(path) or Path(path).read_text()
    )
    embedder = _CountingEmbedder(fake_embedder)

    os.symlink("../E", root / "a")
    messages: list[str] = []
    stats = ing.ingest_pdf_dir(config, embedder, root, progress=messages.append)
    assert stats["pdfs_renamed"] == 3 and stats["pdfs_moved"] == 3
    assert extractions == [] and embedder.texts == 0
    assert stats["chunks_added"] == 0
    assert _stored(config) == {"a/x.pdf", "a/y.pdf", "a/z.pdf", "z_local.pdf"}
    for name in "xyz":
        _assert_same_rows(_rows(config, f"a/{name}.pdf"), before[name])
    rows = get_collection(config).get(where={"pdf_relpath": "a/x.pdf"}, include=["metadatas"])
    assert all(i.startswith(ing._relpath_key("a/x.pdf") + "::") for i in rows["ids"])
    assert all(m["title"] == "X paper" for m in rows["metadatas"])
    assert all(m["pdf_filename"] == "x.pdf" for m in rows["metadatas"])
    assert any(
        "3 source(s) moved from route b/ to route a/ because a link that sorts earlier "
        "now reaches the same folder" in m
        for m in messages
    )
    assert "3 renamed" in _done(messages) and "0 chunks added" in _done(messages)

    # Removing the link moves them back, again without any work.
    os.unlink(root / "a")
    messages.clear()
    stats = ing.ingest_pdf_dir(config, embedder, root, progress=messages.append)
    assert stats["pdfs_renamed"] == 3 and extractions == [] and embedder.texts == 0
    assert _stored(config) == {"b/x.pdf", "b/y.pdf", "b/z.pdf", "z_local.pdf"}
    assert any(
        "moved from route a/ to route b/" in m and "no longer on disk" in m for m in messages
    )
    for name in "xyz":
        _assert_same_rows(_rows(config, f"b/{name}.pdf"), before[name])


def test_unchanged_file_stored_under_a_link_route_is_rekeyed_without_reading_it(
    tmp_path, fake_embedder, text_pdfs, monkeypatch
):
    """An alias whose bytes are current needs no refresh, so an unreadable file cannot block it."""
    config = _config(tmp_path)
    root = tmp_path / "library"
    (root / "real").mkdir(parents=True)
    (root / "shortcuts").mkdir()
    os.symlink("../real", root / "shortcuts" / "c")
    (root / "real" / "x.pdf").write_text("Novelty stays as stored. " * 30)
    (root / "y.pdf").write_text("Other novelty. " * 30)
    ing.ingest_pdf_dir(config, fake_embedder, root, progress=_quiet)
    _store_under(config, "real/x.pdf", "shortcuts/c/x.pdf")
    before = _rows(config, "shortcuts/c/x.pdf")

    def damaged(path):
        raise ValueError("damaged")

    monkeypatch.setattr(ing, "extract_pdf_text", damaged)
    messages: list[str] = []
    stats = ing.ingest_pdf_dir(config, fake_embedder, root, progress=messages.append)
    assert _stored(config) == {"real/x.pdf", "y.pdf"}
    _assert_same_rows(_rows(config, "real/x.pdf"), before)
    assert stats["pdfs_renamed"] == 1 and stats["pdfs_moved"] == 1
    assert stats["pdfs_unreadable"] == 0 and stats["pdfs_kept_previous"] == 0
    assert not any("retained" in m for m in messages)


def test_changed_file_under_a_link_route_is_still_rebuilt(tmp_path, fake_embedder, text_pdfs):
    config = _config(tmp_path)
    root, external = _external_library(tmp_path)
    ing.ingest_pdf_dir(config, fake_embedder, root, progress=_quiet)
    (external / "x.pdf").write_text("Revised novelty. " * 30)
    os.symlink("../E", root / "a")
    embedder = _CountingEmbedder(fake_embedder)
    stats = ing.ingest_pdf_dir(config, embedder, root, progress=_quiet)
    assert stats["pdfs_moved"] == 2 and stats["pdfs_renamed"] == 2
    assert stats["pdfs_ingested"] == 3 and embedder.texts > 0
    assert _stored(config) == {"a/x.pdf", "a/y.pdf", "a/z.pdf", "b/x.pdf", "z_local.pdf"}
    assert stats["pdfs_aliased"] == 1  # b/x.pdf: an older version under the old route


# --- B1: a timeout with no PID file names the guard and advises nothing -------

_HOLD_GUARD = """
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o666)
fcntl.flock(fd, fcntl.LOCK_EX)
print("held", flush=True)
time.sleep(60)
"""


@posix_only
def test_timeout_behind_a_guard_only_holder_names_the_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "_INGEST_LOCK_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(ing, "_INGEST_LOCK_POLL_SECONDS", 0.05)
    config = _config(tmp_path)
    guard = _guard_file(config)
    lock = guard.with_name(".ingest.lock")
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_GUARD, str(guard)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().startswith("held")
        messages: list[str] = []
        with pytest.raises(CorpusLockError) as excinfo:
            with corpus_lock(config, messages.append):
                pytest.fail("entered the corpus while another process held the guard")
    finally:
        holder.kill()
        holder.wait()
    message = str(excinfo.value)
    assert not lock.exists()
    assert str(guard) in message and "PID file" in message and ".ingest.lock" in message
    assert "remove only" not in message and "legacy lock file" not in message
    assert "nothing to remove" in message


# --- B2: a refused lock says what exists and leaves nothing that passes for a corpus --


@posix_only
def test_refused_lock_states_what_exists_and_leaves_no_corpus(tmp_path, fast_lock, monkeypatch):
    import fcntl

    config = _config(tmp_path)
    store = ing._store_directory(config)
    assert not store.exists()
    monkeypatch.setattr(fcntl, "flock", _failing_flock(errno.ENOLCK))
    with pytest.raises(CorpusLockError) as excinfo:
        with corpus_lock(config, _quiet):
            pass
    message = str(excinfo.value)
    assert "No corpus operation was started" not in message
    assert f"created only the store directory {store} and the empty guard file .ingest.guard" in (
        message
    )
    assert sorted(p.name for p in store.iterdir()) == [".ingest.guard"]
    with pytest.raises(FileNotFoundError, match="No corpus found"):
        get_collection(config)


def test_an_existing_store_is_opened_by_its_store_file(tmp_path, fake_embedder):
    config = _config(tmp_path)
    ing.ingest_records(
        config, fake_embedder, [{"source": "a.txt", "abstract": "Novelty. " * 30}], progress=_quiet
    )
    assert get_collection(config).count() > 0


# --- C2: the [done] line counts previous versions kept, not "failed files" ---


def test_done_line_counts_previous_versions_kept(tmp_path, fake_embedder, text_pdfs, monkeypatch):
    config = _config(tmp_path)
    root = tmp_path / "library"
    (root / "real").mkdir(parents=True)
    (root / "shortcuts").mkdir()
    for route in "cd":
        os.symlink("../real", root / "shortcuts" / route)
    (root / "real" / "x.pdf").write_text("Novelty under two routes. " * 30)
    ing.ingest_pdf_dir(config, fake_embedder, root, progress=_quiet)
    collection = get_collection(config)
    rows = collection.get(
        where={"pdf_relpath": "real/x.pdf"}, include=["metadatas", "documents", "embeddings"]
    )
    collection.delete(ids=rows["ids"])
    for alias in ("shortcuts/c/x.pdf", "shortcuts/d/x.pdf"):
        key = ing._relpath_key(alias)
        collection.add(
            ids=[f"{key}::c{m['chunk_index']}" for m in rows["metadatas"]],
            documents=rows["documents"],
            embeddings=rows["embeddings"],
            metadatas=[dict(m, pdf_relpath=alias) for m in rows["metadatas"]],
        )
        _mark_stale(config, alias)  # not reusable: the canonical file must be read

    def damaged(path):
        raise ValueError("damaged")

    monkeypatch.setattr(ing, "extract_pdf_text", damaged)
    messages: list[str] = []
    stats = ing.ingest_pdf_dir(config, fake_embedder, root, progress=messages.append)
    assert stats["pdfs_unreadable"] == 1 and stats["pdfs_kept_previous"] == 2
    assert _stored(config) == {"shortcuts/c/x.pdf", "shortcuts/d/x.pdf"}
    assert "1 unreadable (2 previous versions kept)" in _done(messages)
    assert "failed files" not in _done(messages)
