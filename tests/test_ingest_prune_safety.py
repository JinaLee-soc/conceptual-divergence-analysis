"""Keep the last stored version and apply exclusions across external link routes."""

import errno
import os
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import cda.corpus.ingest as ing
from cda.corpus.ingest import CorpusLockError
from tests.test_guard_retention_edge_cases import _store_under


def _sources(config):
    rows = ing.get_collection(config).get(include=["metadatas"])
    return {m["pdf_relpath"] for m in rows["metadatas"]}


def test_unsupported_guard_cannot_admit_concurrent_operations(tmp_config, monkeypatch):
    store = ing._store_directory(tmp_config)
    store.mkdir(parents=True)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    lock = store / ".ingest.lock"
    record = f"{child.pid}\n"
    lock.write_text(record)

    def unsupported(fd):
        raise OSError(errno.ENOLCK, "filesystem cannot lock")

    monkeypatch.setattr(ing, "_lock_guard", unsupported)
    monkeypatch.setattr(ing, "_INGEST_LOCK_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(ing, "_INGEST_LOCK_POLL_SECONDS", 0.01)

    def attempt(_):
        try:
            with ing.corpus_lock(tmp_config, lambda _: None):
                return "entered"
        except CorpusLockError:
            return "refused"

    with ThreadPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(attempt, range(3)))
    assert outcomes == ["refused"] * 3
    assert lock.read_text() == record


@pytest.mark.parametrize("prune", [False, True])
@pytest.mark.parametrize("other_copy", ["excluded", "missing"])
def test_prune_cannot_use_a_removed_copy_to_drop_a_failed_alias_refresh(
    tmp_path, tmp_config, fake_embedder, monkeypatch, prune, other_copy
):
    root = tmp_path / "library"
    (root / "real").mkdir(parents=True)
    (root / "excluded").mkdir()
    (root / "shortcut").symlink_to("real", target_is_directory=True)
    canonical = root / "real" / "x.pdf"
    canonical.write_text("Novelty has the original interpretation.")
    replacement = "Novelty has the replacement interpretation."
    copy = root / "excluded" / "copy.pdf"
    copy.write_text(replacement)
    monkeypatch.setattr(ing, "extract_pdf_text", lambda path: Path(path).read_text())
    ing.ingest_pdf_dir(tmp_config, fake_embedder, root, progress=lambda _: None)
    _store_under(tmp_config, "real/x.pdf", "shortcut/x.pdf")
    canonical.write_text(replacement)
    if other_copy == "missing":
        copy.unlink()

    def extraction_failed(path):
        if path == canonical:
            raise ValueError("extraction failed")
        return Path(path).read_text()

    monkeypatch.setattr(ing, "extract_pdf_text", extraction_failed)
    stats = ing.ingest_pdf_dir(
        tmp_config,
        fake_embedder,
        root,
        exclude=["excluded/"] if other_copy == "excluded" else None,
        prune_missing=prune,
        progress=lambda _: None,
    )
    expected = {"shortcut/x.pdf"} if prune else {"shortcut/x.pdf", "excluded/copy.pdf"}
    assert _sources(tmp_config) == expected
    assert stats["pdfs_kept_previous"] == 1
    assert stats["pdfs_pruned"] == int(prune)


@pytest.mark.parametrize("external", [False, True])
def test_exclusion_prunes_an_older_route_to_the_same_file(
    tmp_path, tmp_config, fake_embedder, monkeypatch, external
):
    root = tmp_path / "library"
    root.mkdir()
    physical = tmp_path / "outside" if external else root / "real"
    physical.mkdir()
    (physical / "paper.pdf").write_text("Novelty should be excluded from this corpus.")
    (root / "zz").symlink_to(physical, target_is_directory=True)
    monkeypatch.setattr(ing, "extract_pdf_text", lambda path: Path(path).read_text())
    ing.ingest_pdf_dir(tmp_config, fake_embedder, root, progress=lambda _: None)
    if external:
        (root / "aa").symlink_to(physical, target_is_directory=True)
        exclude = ["aa/"]
    else:
        exclude = ["real/"]
    selected, excluded = ing.select_pdfs(root, exclude=exclude)
    assert selected == [] and excluded == 1
    stats = ing.ingest_pdf_dir(
        tmp_config,
        fake_embedder,
        root,
        exclude=exclude,
        prune_missing=True,
        progress=lambda _: None,
    )
    assert _sources(tmp_config) == set()
    assert stats["pdfs_stored_excluded"] == stats["pdfs_pruned"] == 1


@pytest.mark.parametrize("spelling", ["plain", "nfd"])
@pytest.mark.parametrize("root_kind", ["absolute", "relative", "symlink"])
def test_external_alias_is_migrated_when_an_earlier_route_is_added(
    tmp_path, tmp_config, fake_embedder, monkeypatch, spelling, root_kind
):
    root = tmp_path / "library"
    if root_kind == "symlink":
        physical_root = tmp_path / "physical_library"
        physical_root.mkdir()
        root.symlink_to(physical_root, target_is_directory=True)
    else:
        root.mkdir()
    if root_kind == "relative":
        monkeypatch.chdir(tmp_path)
        root = Path("library")
    physical = tmp_path / "outside"
    physical.mkdir()
    filename = "paper.pdf" if spelling == "plain" else "cafe\u0301.pdf"
    old_route = "zz" if spelling == "plain" else "zz-e\u0301"
    new_route = "aa" if spelling == "plain" else "aa-e\u0301"
    (physical / filename).write_text("Novelty has one source despite several routes.")
    os.symlink(physical, root / old_route, target_is_directory=True)
    monkeypatch.setattr(ing, "extract_pdf_text", lambda path: Path(path).read_text())
    ing.ingest_pdf_dir(tmp_config, fake_embedder, root, progress=lambda _: None)
    assert _sources(tmp_config) == {unicodedata.normalize("NFC", f"{old_route}/{filename}")}
    os.symlink(physical, root / new_route, target_is_directory=True)
    stats = ing.ingest_pdf_dir(tmp_config, fake_embedder, root, progress=lambda _: None)
    assert _sources(tmp_config) == {unicodedata.normalize("NFC", f"{new_route}/{filename}")}
    assert stats["pdfs_renamed"] == 1


def test_aliases_cannot_certify_each_other_for_deletion_after_a_failed_refresh(
    tmp_path, tmp_config, fake_embedder, monkeypatch
):
    root = tmp_path / "library"
    (root / "real").mkdir(parents=True)
    (root / "real" / "paper.pdf").write_text("Novelty has a stored interpretation.")
    for route in ("alias1", "alias2"):
        (root / route).symlink_to("real", target_is_directory=True)
    monkeypatch.setattr(ing, "extract_pdf_text", lambda path: Path(path).read_text())
    ing.ingest_pdf_dir(tmp_config, fake_embedder, root, progress=lambda _: None)
    _store_under(tmp_config, "real/paper.pdf", "alias1/paper.pdf")
    collection = ing.get_collection(tmp_config)
    rows = collection.get(include=["metadatas", "documents", "embeddings"])
    key = ing._relpath_key("alias2/paper.pdf")
    collection.add(
        ids=[f"{key}::c{m['chunk_index']}" for m in rows["metadatas"]],
        documents=rows["documents"],
        embeddings=rows["embeddings"],
        metadatas=[dict(m, pdf_relpath="alias2/paper.pdf") for m in rows["metadatas"]],
    )
    monkeypatch.setattr(ing, "extract_pdf_text", lambda path: "")
    stats = ing.ingest_pdf_dir(
        tmp_config,
        fake_embedder,
        root,
        prune_missing=True,
        progress=lambda _: None,
    )
    assert _sources(tmp_config) == {"alias1/paper.pdf", "alias2/paper.pdf"}
    assert stats["pdfs_kept_previous"] == 2 and stats["pdfs_pruned"] == 0
