"""Symlink routes under --exclude, stale corpus locks, and the demo's use of both.

Regressions for: a link route defeating ``--exclude`` (selection by
canonical route), a lock file left by a dead process stalling every corpus
operation, the demo creating its store outside the lock, the demo hiding its
progress and its lock error, ``--session`` term comparison, and the capped
keyword read naming one legacy source instead of all.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import cda.corpus.ingest as ing
from cda.cli import main
from cda.config import ConfigError, load_config
from cda.corpus.ingest import CorpusLockError, corpus_lock, get_collection, select_pdfs


def _quiet(_message: str) -> None:
    pass


def _stored(config) -> set[str]:
    metadatas = get_collection(config).get(include=["metadatas"])["metadatas"]
    return {m["pdf_relpath"] for m in metadatas}


def _config(tmp_path):
    return load_config(overrides={"paths": {"corpus_dir": str(tmp_path / "corpus")}})


@pytest.fixture
def text_pdfs(monkeypatch):
    monkeypatch.setattr(ing, "extract_pdf_text", lambda path: Path(path).read_text())


@pytest.fixture
def linked_library(tmp_path):
    """pdfs/confidential/secret.pdf, pdfs/shortcuts/c -> ../confidential, pdfs/public.pdf."""
    root = tmp_path / "pdfs"
    (root / "confidential").mkdir(parents=True)
    (root / "shortcuts").mkdir()
    (root / "confidential" / "secret.pdf").write_text("Novelty is kept secret here. " * 30)
    (root / "public.pdf").write_text("Novelty is public here. " * 30)
    os.symlink("../confidential", root / "shortcuts" / "c")
    return root


# --- A1: exclusion is decided on the canonical route -------------------------


def test_link_route_cannot_keep_an_excluded_pdf(linked_library):
    kept, excluded = select_pdfs(linked_library, exclude=["confidential/"])
    assert kept == [linked_library / "public.pdf"]
    assert excluded == 1  # one physical file, however many routes reach it


def test_link_route_is_stored_under_the_canonical_relpath(linked_library):
    kept, _ = select_pdfs(linked_library)
    assert [str(p.relative_to(linked_library)) for p in kept] == [
        "confidential/secret.pdf",
        "public.pdf",
    ]


def test_link_route_to_an_external_folder_keeps_the_first_route(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "paper.pdf").write_bytes(b"%PDF-")
    root = tmp_path / "pdfs"
    (root / "b").mkdir(parents=True)
    os.symlink(outside, root / "b" / "link", target_is_directory=True)
    os.symlink(outside, root / "zz", target_is_directory=True)
    kept, _ = select_pdfs(root)
    assert kept == [root / "b" / "link" / "paper.pdf"]
    kept, excluded = select_pdfs(root, exclude=["b"])
    assert kept == [] and excluded == 1  # the second route does not keep it either


def test_each_physical_folder_is_walked_once(tmp_path):
    """A DAG of folder links walked per route grows exponentially with depth."""
    root = tmp_path / "pdfs"
    root.mkdir()
    folders = [root] + [tmp_path / f"level{i}" for i in range(1, 5)]
    for folder in folders[1:]:
        folder.mkdir()
    for upper, lower in zip(folders, folders[1:], strict=False):
        for k in range(5):
            os.symlink(lower, upper / f"link{k}", target_is_directory=True)
    (folders[-1] / "paper.pdf").write_bytes(b"%PDF-")
    assert len(ing._walk_files(root, True)) == 1  # 5**4 == 625 routes
    kept, _ = select_pdfs(root)
    assert kept == [root / "link0" / "link0" / "link0" / "link0" / "paper.pdf"]


def test_prune_removes_an_excluded_source_instead_of_renaming_it_to_a_link_route(
    tmp_path, fake_embedder, text_pdfs, linked_library
):
    config = _config(tmp_path)
    ing.ingest_pdf_dir(config, fake_embedder, linked_library, progress=_quiet)
    assert _stored(config) == {"confidential/secret.pdf", "public.pdf"}
    messages: list[str] = []
    stats = ing.ingest_pdf_dir(
        config,
        fake_embedder,
        linked_library,
        exclude=["confidential/"],
        prune_missing=True,
        progress=messages.append,
    )
    assert _stored(config) == {"public.pdf"}
    assert stats["pdfs_pruned"] == 1 and stats["pdfs_renamed"] == 0
    assert stats["pdfs_stored_excluded"] == 1 and stats["pdfs_excluded"] == 1
    assert any("now excluded" in m and "confidential/secret.pdf" in m for m in messages)
    # Without pruning the exclusion holds as well: nothing comes back.
    ing.ingest_pdf_dir(
        config, fake_embedder, linked_library, exclude=["confidential/"], progress=_quiet
    )
    assert _stored(config) == {"public.pdf"}


def _store_under_link_route(config, embedder, library: Path) -> None:
    """Ingest the library as an older version stored it: secret.pdf under its link route.

    Older versions selected a file by whichever route reached it first, so a
    real legacy store holds ``shortcuts/c/secret.pdf`` as the relpath AND
    derives the chunk ids from that route. Rewriting ``pdf_relpath`` on rows
    whose ids were keyed to ``confidential/secret.pdf`` would invent a store
    no version ever wrote (and hide every id-collision path from the tests).
    The link route is made the only route for the ingest by moving the real
    folder out of the library, then the layout is restored as the tests
    expect it: ``confidential/secret.pdf`` with ``shortcuts/c`` linking to it.
    """
    confidential = library / "confidential"
    hidden = library.parent / "confidential_elsewhere"
    link = library / "shortcuts" / "c"
    confidential.rename(hidden)
    link.unlink()
    os.symlink(hidden, link, target_is_directory=True)
    ing.ingest_pdf_dir(config, embedder, library, progress=_quiet)
    link.unlink()
    hidden.rename(confidential)
    os.symlink("../confidential", link)
    assert _stored(config) == {"shortcuts/c/secret.pdf", "public.pdf"}


def test_legacy_fixture_keys_chunk_ids_to_the_link_route(
    tmp_path, fake_embedder, text_pdfs, linked_library
):
    """The fixture is a faithful legacy store: ids and relpath both derive from the route."""
    config = _config(tmp_path)
    _store_under_link_route(config, fake_embedder, linked_library)
    result = get_collection(config).get(
        where={"pdf_relpath": "shortcuts/c/secret.pdf"}, include=["metadatas"]
    )
    assert result["ids"]
    assert all(
        i.startswith(ing._relpath_key("shortcuts/c/secret.pdf") + "::") for i in result["ids"]
    )
    assert not any(i.startswith(ing._relpath_key("confidential/secret.pdf")) for i in result["ids"])
    assert all(m["pdf_root"] == str(linked_library.resolve()) for m in result["metadatas"])


def test_source_stored_under_a_link_route_is_renamed_to_its_canonical_relpath(
    tmp_path, fake_embedder, text_pdfs, linked_library
):
    config = _config(tmp_path)
    _store_under_link_route(config, fake_embedder, linked_library)
    assert _stored(config) == {"shortcuts/c/secret.pdf", "public.pdf"}
    messages: list[str] = []
    stats = ing.ingest_pdf_dir(config, fake_embedder, linked_library, progress=messages.append)
    assert _stored(config) == {"confidential/secret.pdf", "public.pdf"}
    assert stats["pdfs_renamed"] == 1 and stats["pdfs_duplicates_skipped"] == 0
    assert any("replacing the stored copy 'shortcuts/c/secret.pdf'" in m for m in messages)


def test_source_stored_under_a_link_route_is_pruned_when_its_file_is_excluded(
    tmp_path, fake_embedder, text_pdfs, linked_library
):
    config = _config(tmp_path)
    _store_under_link_route(config, fake_embedder, linked_library)
    stats = ing.ingest_pdf_dir(
        config,
        fake_embedder,
        linked_library,
        exclude=["confidential/"],
        prune_missing=True,
        progress=_quiet,
    )
    assert _stored(config) == {"public.pdf"}
    assert stats["pdfs_pruned"] == 1 and stats["pdfs_stored_excluded"] == 1


# --- A2: a lock left by a dead process is reclaimed; a live one is waited on --


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


@pytest.fixture
def fast_lock(monkeypatch):
    monkeypatch.setattr(ing, "_INGEST_LOCK_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(ing, "_INGEST_LOCK_POLL_SECONDS", 0.01)


def _lock_file(config) -> Path:
    store = ing._store_directory(config)
    store.mkdir(parents=True, exist_ok=True)
    return store / ".ingest.lock"


def test_lock_of_a_dead_process_is_reclaimed_at_once(tmp_path, fast_lock):
    config = _config(tmp_path)
    lock = _lock_file(config)
    pid = _dead_pid()
    lock.write_text(f"{pid}\n")
    messages: list[str] = []
    started = time.monotonic()
    with corpus_lock(config, messages.append):
        assert lock.read_text().strip() == str(os.getpid())
    assert time.monotonic() - started < 0.25  # no wait for the timeout
    assert any("stale lock" in m and str(pid) in m for m in messages)
    assert not lock.exists()


def test_empty_lock_older_than_the_timeout_is_stale(tmp_path, fast_lock):
    config = _config(tmp_path)
    lock = _lock_file(config)
    lock.write_text("")
    old = time.time() - 10
    os.utime(lock, (old, old))
    messages: list[str] = []
    with corpus_lock(config, messages.append):
        pass
    assert any("stale lock" in m for m in messages)


def test_fresh_empty_lock_is_waited_on_before_it_counts_as_stale(tmp_path, fast_lock):
    """A holder writes its pid right after creating the file: do not steal that moment.

    An empty file is waited on like a held lock, and reclaimed only once it is
    older than the timeout (no pid ever arrived, so no holder can be checked).
    """
    config = _config(tmp_path)
    lock = _lock_file(config)
    lock.write_text("")
    messages: list[str] = []
    started = time.monotonic()
    with corpus_lock(config, messages.append):
        pass
    assert time.monotonic() - started >= 0.3
    assert messages[0].startswith("[corpus] another ingest or retrieval is running")
    assert any("stale lock" in m and "crashed run" in m for m in messages)


def test_lock_recording_this_process_is_reclaimed_under_the_guard(tmp_path, fast_lock):
    """A record of this very pid cannot be a live holder once the guard is held:
    a process is never inside the region and entering it at once. Docker leaves
    such records ("1") when the container is stopped mid-run."""
    config = _config(tmp_path)
    _lock_file(config).write_text(f"{os.getpid()}\n")
    messages: list[str] = []
    with corpus_lock(config, messages.append):
        pass
    assert any("stale lock" in m for m in messages)
    assert not _lock_file(config).exists()


def test_lock_of_a_live_process_is_waited_on_then_reported(tmp_path, fast_lock):
    config = _config(tmp_path)
    lock = _lock_file(config)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    pid = child.pid
    lock.write_text(f"{pid}\n")
    messages: list[str] = []
    try:
        started = time.monotonic()
        with pytest.raises(CorpusLockError, match="ingest.lock") as excinfo:
            with corpus_lock(config, messages.append):
                pass
        assert time.monotonic() - started >= 0.3
        assert isinstance(excinfo.value, RuntimeError)
        assert any("waiting for it to finish" in m for m in messages)
        assert lock.read_text().strip() == str(pid)  # the holder's file is left alone
    finally:
        if child is not None:
            child.kill()
            child.wait()


def test_cli_reports_a_held_lock_without_a_traceback(
    tmp_path, fast_lock, text_pdfs, fake_embedder, monkeypatch, capsys
):
    config = _config(tmp_path)
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    _lock_file(config).write_text(f"{holder.pid}\n")
    library = tmp_path / "pdfs"
    library.mkdir()
    (library / "paper.pdf").write_text("Novelty is waited on.")
    monkeypatch.setattr("cda.corpus.embeddings.get_embedder", lambda config: fake_embedder)
    try:
        code = main(["ingest", "--pdf-dir", str(library), "--corpus-dir", str(tmp_path / "corpus")])
    finally:
        holder.kill()
    assert code == 1
    err = capsys.readouterr().err
    assert "error:" in err and "ingest.lock" in err and "Traceback" not in err


# --- B2: the demo creates its store under the lock ---------------------------


def _toy_demo_dir(tmp_path, n_records=4) -> Path:
    toy = tmp_path / "demo" / "toy_corpus"
    toy.mkdir(parents=True)
    with (toy / "abstracts.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(n_records):
            record = {"source": f"2020_paper_{i}.abstract", "abstract": f"Novelty text {i}. " * 20}
            handle.write(json.dumps(record) + "\n")
    (toy / "drafts").mkdir()
    return tmp_path / "demo"


def test_concurrent_first_demo_runs_create_the_store_under_the_lock(
    tmp_path, fake_embedder, monkeypatch
):
    from cda.demo import demo_config, ensure_toy_corpus_ingested, load_toy_records

    demo_dir = _toy_demo_dir(tmp_path)
    config = demo_config(demo_dir)
    lock = ing._store_directory(config) / ".ingest.lock"
    real_get_collection = ing.get_collection
    outside_lock: list[bool] = []

    def guarded(config, *, create=False):
        if create:
            outside_lock.append(not lock.exists())
        return real_get_collection(config, create=create)

    monkeypatch.setattr("cda.demo.get_collection", guarded)
    monkeypatch.setattr(ing, "_INGEST_LOCK_POLL_SECONDS", 0.01)
    outcomes: list[str] = []
    barrier = threading.Barrier(6)

    def first_visit():
        barrier.wait()
        try:
            ensure_toy_corpus_ingested(config, demo_dir, fake_embedder, progress=_quiet)
            outcomes.append("ok")
        except Exception as exc:  # noqa: BLE001 - every failure is the finding
            outcomes.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=first_visit) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes == ["ok"] * 6
    assert not any(outside_lock), "the store was opened or created outside the corpus lock"
    expected = {r["source"] for r in load_toy_records(demo_dir)}
    assert _stored(config) == expected
    assert not lock.exists()


# --- B1: the demo streams its progress and surfaces a lock timeout -----------


def test_run_demo_streams_every_line_to_the_progress_callback(
    tmp_path, fake_embedder, monkeypatch, capsys
):
    import cda.demo
    import cda.pipeline

    demo_dir = _toy_demo_dir(tmp_path)
    monkeypatch.setattr(cda.demo, "get_embedder", lambda config: fake_embedder)
    seen = {}

    def fake_analyze_term(term, config, **kwargs):
        seen.update(kwargs)
        kwargs["progress"]("[1/3] pretending")
        return {"status": "skipped", "reason": "captured by the test"}

    monkeypatch.setattr(cda.pipeline, "analyze_term", fake_analyze_term)
    lines: list[str] = []
    result = cda.demo.run_demo(term="novelty", mock=True, demo_dir=demo_dir, progress=lines.append)
    assert result["status"] == "skipped"
    assert any(m.startswith("[demo] toy corpus:") for m in lines)
    assert any(m.startswith("[demo] first run: indexing 4") for m in lines)
    assert any("4/4 records ingested" in m for m in lines)  # the ingest is no longer silent
    assert "[1/3] pretending" in lines
    assert any(m.startswith("[demo] skipped:") for m in lines)
    assert capsys.readouterr().out == ""  # nothing bypasses the callback


def test_run_demo_surfaces_a_lock_timeout(tmp_path, fake_embedder, monkeypatch, fast_lock):
    import cda.demo

    demo_dir = _toy_demo_dir(tmp_path)
    monkeypatch.setattr(cda.demo, "get_embedder", lambda config: fake_embedder)
    config = cda.demo.demo_config(demo_dir)
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    _lock_file(config).write_text(f"{holder.pid}\n")
    lines: list[str] = []
    try:
        with pytest.raises(CorpusLockError):
            cda.demo.run_demo(term="novelty", mock=True, demo_dir=demo_dir, progress=lines.append)
    finally:
        holder.kill()
    assert any("waiting for it to finish" in m for m in lines)


APP_PATH = str(Path(__file__).resolve().parents[1] / "gui" / "app.py")
HOST_LOCK = "/srv/private/host/demo/_runtime/corpus/chroma/.ingest.lock"


def _click_demo(app):
    [button] = [b for b in app.button if "Try the demo" in (b.label or "")]
    button.click().run()


def _lock_timeout(**kwargs):
    kwargs["progress"]("[corpus] another ingest or retrieval is running on this corpus")
    raise CorpusLockError(f"Another ingest or retrieval still holds {HOST_LOCK} after 120s.")


def test_gui_demo_button_streams_progress_and_names_a_lock_timeout(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import cda.demo

    monkeypatch.delenv("CDA_DEMO_MODE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cda.demo, "run_demo", _lock_timeout)
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _click_demo(app)
    assert not app.exception, f"GUI raised: {app.exception}"
    errors = " ".join(e.value for e in app.error)
    assert HOST_LOCK in errors  # locally the path is the actionable detail
    assert "The demo failed" not in errors
    streamed = " ".join(m.value for m in app.markdown)
    assert "another ingest or retrieval is running" in streamed


def test_gui_public_demo_names_a_lock_timeout_without_the_host_path(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import cda.demo

    monkeypatch.setenv("CDA_DEMO_MODE", "1")
    monkeypatch.setattr(cda.demo, "run_demo", _lock_timeout)
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _click_demo(app)
    assert not app.exception, f"GUI raised: {app.exception}"
    errors = " ".join(e.value for e in app.error)
    assert "The demo cannot reach its corpus right now; please retry." in errors
    assert HOST_LOCK not in errors and "/srv/" not in errors
    assert "The demo failed on this server" not in errors


# --- B3: --session compares terms the way result lookup does -----------------


def test_validate_session_matches_its_term_by_identity(tmp_path, capsys, monkeypatch):
    from datetime import datetime

    from cda.validation.session import create_session

    monkeypatch.chdir(tmp_path)
    session_dir, _ = create_session(
        "ai",
        params={"seed": 42},
        sides={"literature": {"validated": True, "note": None}},
        partitions=[
            {
                "revision": 0,
                "side": "literature",
                "kind": "original",
                "assignments": {"a": 0, "b": 0},
                "clusters": [
                    {"cluster_id": 0, "n_passages": 2, "silhouette": None, "shown": ["a", "b"]}
                ],
            }
        ],
        passages={key: {"text": f"text {key}", "side": "literature"} for key in "ab"},
        readings=[],
        output_dir=tmp_path,
        now=datetime(2026, 7, 30, 12, 0, 0),
    )
    assert main(["validate", "--term", "AI", "--session", str(session_dir)]) == 0
    assert "reopened session" in capsys.readouterr().out
    assert main(["validate", "--term", "A.I.", "--session", str(session_dir)]) == 1
    err = capsys.readouterr().err
    assert "is for term 'ai', not 'A.I.'" in err and "--term 'ai'" in err


# --- C1: a capped keyword read names every legacy source ---------------------


def test_capped_keyword_retrieval_names_every_legacy_source(tmp_path, fake_embedder, text_pdfs):
    from cda.extraction.retrieval import keyword_hits

    config = _config(tmp_path)
    library = tmp_path / "pdfs"
    library.mkdir()
    for name in ("alpha", "beta", "gamma"):
        (library / f"{name}.pdf").write_text(f"Sentence about novelty in {name}. " * 8)
    ing.ingest_pdf_dir(config, fake_embedder, library, progress=_quiet)
    collection = get_collection(config)
    result = collection.get(include=["metadatas"])
    legacy_ids = [
        i
        for i, m in zip(result["ids"], result["metadatas"], strict=True)
        if m["pdf_relpath"] in ("alpha.pdf", "beta.pdf")
    ]
    collection.update(
        ids=legacy_ids,
        metadatas=[
            {"sentence_context": None, "sentence_start": None, "sentence_lengths": None}
            for _ in legacy_ids
        ],
    )
    with pytest.raises(ConfigError, match="2 source\\(s\\)") as excinfo:
        keyword_hits(collection, "novelty", max_chunks=10, progress=_quiet)
    assert "alpha.pdf" in str(excinfo.value) and "beta.pdf" in str(excinfo.value)
    assert "gamma.pdf" not in str(excinfo.value)


def test_stale_recovery_cannot_expose_a_live_holders_lock(tmp_path, monkeypatch):
    """Pause a stale waiter while a second holder and then a third contender arrive."""
    from concurrent.futures import ThreadPoolExecutor

    config = _config(tmp_path)
    lock = _lock_file(config)
    lock.write_text(f"{_dead_pid()}\n")
    scanned, resume_scan, renamed, resume_rename, release = [threading.Event() for _ in range(5)]
    observed = {name: threading.Event() for name in ("B", "C")}
    local = threading.local()
    state_lock = threading.Lock()
    active, maximum = 0, 0
    holder, rename = ing._lock_holder, os.rename

    def pause_holder(path):
        result = holder(path)
        if getattr(local, "name", None) == "A" and result[2] and not scanned.is_set():
            scanned.set()
            assert resume_scan.wait(10)
        return result

    def pause_rename(source, destination):
        result = rename(source, destination)
        if getattr(local, "name", None) == "A" and Path(source) == lock:
            renamed.set()
            assert resume_rename.wait(10)
        return result

    def worker(name):
        nonlocal active, maximum
        local.name = name

        def progress(message):
            if name in observed and "waiting for it to finish" in message:
                observed[name].set()

        with corpus_lock(config, progress):
            with state_lock:
                active += 1
                maximum = max(maximum, active)
            if name in observed:
                observed[name].set()
            try:
                assert release.wait(10)
            finally:
                with state_lock:
                    active -= 1

    monkeypatch.setattr(ing, "_lock_holder", pause_holder)
    monkeypatch.setattr(ing.os, "rename", pause_rename)
    monkeypatch.setattr(ing, "_INGEST_LOCK_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(ing, "_INGEST_LOCK_POLL_SECONDS", 0.01)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, "A")]
        try:
            assert scanned.wait(10)
            futures.append(pool.submit(worker, "B"))
            assert observed["B"].wait(10)
            resume_scan.set()
            assert renamed.wait(10)
            futures.append(pool.submit(worker, "C"))
            assert observed["C"].wait(10)
        finally:
            resume_scan.set()
            resume_rename.set()
            release.set()
        errors = []
        for future in futures:
            try:
                future.result(timeout=10)
            except CorpusLockError as exc:
                errors.append(exc)
    assert maximum == 1, "stale recovery admitted overlapping corpus operations"
    assert not errors


@pytest.mark.parametrize(
    "content",
    ["999999999999999999999", "²", "9" * 5000],
    ids=["out-of-range", "unicode-digit", "oversized"],
)
def test_malformed_pid_is_recovered_after_the_stale_timeout(tmp_path, fast_lock, content):
    config = _config(tmp_path)
    lock = _lock_file(config)
    lock.write_text(content)
    old = time.time() - 10
    os.utime(lock, (old, old))
    with corpus_lock(config, _quiet):
        assert lock.read_text().strip() == str(os.getpid())


def test_failed_stale_recovery_obeys_the_lock_timeout(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _lock_file(config).write_text(f"{_dead_pid()}\n")
    calls = 0

    def cannot_rename(*args):
        nonlocal calls
        calls += 1
        assert calls < 3, "failed stale recovery retried without checking the deadline"
        return False

    monkeypatch.setattr(ing, "_reclaim_stale_lock", cannot_rename)
    monkeypatch.setattr(ing, "_INGEST_LOCK_TIMEOUT_SECONDS", 0)
    with pytest.raises(CorpusLockError):
        with corpus_lock(config, _quiet):
            pass


def test_process_crash_releases_guard_and_recovers_pid_file(tmp_path):
    config = _config(tmp_path)
    code = """
import os, sys
from cda.config import load_config
from cda.corpus.ingest import corpus_lock
config = load_config(overrides={"paths": {"corpus_dir": sys.argv[1]}})
with corpus_lock(config, lambda _: None):
    os._exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(config.path("paths.corpus_dir"))], check=True, timeout=15
    )
    assert _lock_file(config).exists()
    with corpus_lock(config, _quiet):
        pass
    assert not _lock_file(config).exists()


def test_separate_process_cannot_enter_a_held_corpus(tmp_path):
    config = _config(tmp_path)
    code = """
import sys
from cda.config import load_config
import cda.corpus.ingest as ing
ing._INGEST_LOCK_TIMEOUT_SECONDS = 0.05
ing._INGEST_LOCK_POLL_SECONDS = 0.01
config = load_config(overrides={"paths": {"corpus_dir": sys.argv[1]}})
try:
    with ing.corpus_lock(config, lambda _: None):
        sys.exit(1)
except ing.CorpusLockError:
    pass
"""
    with corpus_lock(config, _quiet):
        subprocess.run(
            [sys.executable, "-c", code, str(config.path("paths.corpus_dir"))],
            check=True,
            timeout=15,
        )


def test_windows_access_denied_means_the_holder_is_alive(monkeypatch):
    import ctypes
    from types import SimpleNamespace

    class Function:
        def __init__(self, result):
            self.result = result

        def __call__(self, *args):
            return self.result

    kernel = SimpleNamespace(
        OpenProcess=Function(0), GetExitCodeProcess=Function(0), CloseHandle=Function(1)
    )
    last_error_enabled = False

    def dll(name, *, use_last_error):
        nonlocal last_error_enabled
        last_error_enabled = use_last_error
        return kernel

    with monkeypatch.context() as patch:
        patch.setattr(ing.sys, "platform", "win32")
        patch.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel), raising=False)
        patch.setattr(ctypes, "WinDLL", dll, raising=False)
        patch.setattr(
            ctypes, "get_last_error", lambda: 5 if last_error_enabled else 0, raising=False
        )
        assert ing._process_alive(1234)


@pytest.mark.parametrize("failure", ["unreadable", "no_text"])
def test_prune_keeps_alias_source_when_canonical_refresh_fails(
    tmp_path, fake_embedder, text_pdfs, linked_library, monkeypatch, failure
):
    config = _config(tmp_path)
    _store_under_link_route(config, fake_embedder, linked_library)
    before = get_collection(config).get(
        where={"pdf_relpath": "shortcuts/c/secret.pdf"}, include=["documents", "metadatas"]
    )
    # The file must have changed: an unchanged file is re-keyed from the
    # stored alias without being extracted, so its refresh cannot fail.
    (linked_library / "confidential" / "secret.pdf").write_text("Novelty, revised. " * 30)
    extract = ing.extract_pdf_text

    def failed_extract(path):
        if path.name == "secret.pdf":
            if failure == "unreadable":
                raise ValueError("damaged PDF")
            return ""
        return extract(path)

    monkeypatch.setattr(ing, "extract_pdf_text", failed_extract)
    messages = []
    stats = ing.ingest_pdf_dir(
        config, fake_embedder, linked_library, prune_missing=True, progress=messages.append
    )
    after = get_collection(config).get(
        where={"pdf_relpath": "shortcuts/c/secret.pdf"}, include=["documents", "metadatas"]
    )
    assert after == before
    assert stats["pdfs_pruned"] == 0
    assert any("replacement" in m and "retained" in m for m in messages)


@pytest.mark.parametrize("mode", ["demo", "own_text"])
def test_public_progress_never_exposes_server_paths(monkeypatch, mode):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import cda.demo

    def failed_demo(**kwargs):
        kwargs["progress"]("[demo] toy corpus: /srv/private/host/demo/toy_corpus")
        kwargs["progress"](
            f"[corpus] another ingest or retrieval is running (lock file: {HOST_LOCK}); waiting for it to finish…"
        )
        # The real timeout wording: only a held corpus gets the retry notice
        # (see tests/test_public_lock_wording.py for the permanent cases).
        raise CorpusLockError(f"Another ingest or retrieval still holds {HOST_LOCK} after 300s")

    monkeypatch.setenv("CDA_DEMO_MODE", "1")
    monkeypatch.setattr(cda.demo, "run_demo", failed_demo)
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    if mode == "demo":
        _click_demo(app)
    else:
        next(t for t in app.text_area if t.label == "Your paragraphs").set_value(
            "Novelty means a new approach."
        )
        app.run()
        next(b for b in app.button if "no model API" in b.label).click().run()
    assert not app.exception
    page = " ".join(m.value for m in app.markdown) + " ".join(e.value for e in app.error)
    assert "/srv/private" not in page
    assert "The demo cannot reach its corpus right now; please retry." in page
