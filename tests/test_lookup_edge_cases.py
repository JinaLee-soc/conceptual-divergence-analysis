"""Latest-run / latest-session lookup when an OLDER sibling is damaged.

`cda verify` and the GUI's "Resume latest session" both scan every same-slug
entry in the output directory. A damaged older entry (invalid UTF-8, a
truncated or other-schema session.json, an unreadable file) must be skipped
so the intact newer one is found, and must be named when nothing intact
remains — never raised as a bare decode or permission error about a file the
researcher did not ask about.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

from cda.extraction.windows import safe_term_slug
from cda.registers import verify
from cda.registers.verify import find_latest_run
from cda.validation import session as session_module
from cda.validation.session import create_session, find_latest_session, load_session

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "gui" / "app.py")
HEADER = "<!-- generated: 2026-08-01T09:00:00 | model: claude-sonnet-4-6 | cda: v1 -->\nreport"
OLD, NEW = "20260801_090000", "20260801_100000"

not_root = pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0, reason="file permissions are not enforced for root"
)


# --- cda verify: find_latest_run ------------------------------------------


def _write_run(out_dir: Path, term: str, ts: str, prompt: bytes | None = None) -> Path:
    base = f"drift_{safe_term_slug(term)}_{ts}"
    (out_dir / f"{base}.md").write_text(HEADER, encoding="utf-8")
    path = out_dir / f"{base}_prompt.md"
    if prompt is None:
        prompt = f'TARGET TERM: "{term}"\n\nRESEARCHER CONTEXT:\n...'.encode()
    path.write_bytes(prompt)
    return path


def test_find_latest_run_skips_a_damaged_older_prompt(tmp_path):
    """Invalid UTF-8 in an OLDER prompt used to abort `cda verify` for the
    good newer run, with advice naming the older file."""
    _write_run(tmp_path, "novelty", OLD, prompt=b'TARGET TERM: "novelty"\n\nabc\xff')
    _write_run(tmp_path, "novelty", NEW)
    prompt, report = find_latest_run(tmp_path, "novelty")
    assert report.name == f"drift_novelty_{NEW}.md"
    assert prompt.name == f"drift_novelty_{NEW}_prompt.md"


def test_find_latest_run_skips_an_older_prompt_that_cannot_be_opened(tmp_path):
    """An OSError on an older prompt (here: the name is a directory) is skipped."""
    (tmp_path / f"drift_novelty_{OLD}.md").write_text(HEADER, encoding="utf-8")
    (tmp_path / f"drift_novelty_{OLD}_prompt.md").mkdir()
    _write_run(tmp_path, "novelty", NEW)
    assert find_latest_run(tmp_path, "novelty")[1].name == f"drift_novelty_{NEW}.md"


@not_root
def test_find_latest_run_skips_an_older_prompt_without_read_permission(tmp_path):
    older = _write_run(tmp_path, "novelty", OLD)
    _write_run(tmp_path, "novelty", NEW)
    older.chmod(0)
    try:
        assert find_latest_run(tmp_path, "novelty")[1].name == f"drift_novelty_{NEW}.md"
    finally:
        older.chmod(0o644)


def test_find_latest_run_tolerates_a_utf8_bom_before_the_target_term(tmp_path):
    """A prompt re-saved by an editor that adds a BOM is still identified."""
    _write_run(tmp_path, "novelty", OLD, prompt=b'\xef\xbb\xbfTARGET TERM: "novelty"\r\n\r\nx')
    assert find_latest_run(tmp_path, "novelty")[1].name == f"drift_novelty_{OLD}.md"


def test_find_latest_run_head_boundary_inside_a_multibyte_char_is_not_damage(tmp_path):
    """A prompt longer than the head whose cut falls inside a character."""
    head = b'TARGET TERM: "novelty"\n\n'
    filler = "é" * (verify._PROMPT_HEAD_BYTES * 2)  # 2 bytes each
    _write_run(tmp_path, "novelty", OLD, prompt=head + filler.encode("utf-8"))
    assert find_latest_run(tmp_path, "novelty")[1].name == f"drift_novelty_{OLD}.md"


def test_find_latest_run_names_skipped_runs_when_none_remain(tmp_path):
    _write_run(tmp_path, "C#", OLD, prompt=b'TARGET TERM: "C#"\n\nabc\xff')
    _write_run(tmp_path, "C#", NEW, prompt=b"no target term line\n")
    with pytest.raises(FileNotFoundError) as excinfo:
        find_latest_run(tmp_path, "C#")
    message = str(excinfo.value)
    assert "2 same-slug run(s) were skipped" in message
    assert f"drift_c_{OLD}_prompt.md" in message and "not UTF-8" in message
    assert f"drift_c_{NEW}_prompt.md" in message and "no TARGET TERM line" in message
    # The advice is copy-safe for a shell: C# is quoted.
    assert "cda analyze --term 'C#'" in message


def test_find_latest_run_error_without_damage_does_not_mention_skips(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        find_latest_run(tmp_path, "novelty")
    assert "skipped" not in str(excinfo.value)
    assert "cda analyze --term novelty" in str(excinfo.value)


def test_find_latest_run_reads_only_same_slug_prompt_heads(tmp_path, monkeypatch):
    """Other-slug files and the reports are never opened; the same-slug
    prompts are read as a bounded head, and the full read is left to the
    caller for the one chosen run."""
    _write_run(tmp_path, "novelty", OLD)
    big = b'TARGET TERM: "novelty"\n\n' + b"x" * (verify._PROMPT_HEAD_BYTES * 4)
    _write_run(tmp_path, "novelty", NEW, prompt=big)
    _write_run(tmp_path, "novelty claims", "20260801_110000", prompt=b"\xff\xfe damaged")

    opened: list[Path] = []
    original_open = Path.open

    def recording_open(self, *args, **kwargs):
        opened.append(Path(self))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)

    def never(path):
        raise AssertionError(f"_read_run_file called inside find_latest_run for {path}")

    monkeypatch.setattr(verify, "_read_run_file", never)
    _, report = find_latest_run(tmp_path, "novelty")
    assert report.name == f"drift_novelty_{NEW}.md"
    assert sorted(p.name for p in opened) == [
        f"drift_novelty_{OLD}_prompt.md",
        f"drift_novelty_{NEW}_prompt.md",
    ]


def test_find_latest_run_is_independent_of_directory_listing_order(tmp_path, monkeypatch):
    _write_run(tmp_path, "novelty", OLD, prompt=b'TARGET TERM: "novelty"\n\nabc\xff')
    _write_run(tmp_path, "novelty", NEW)
    _write_run(tmp_path, "novelty", f"{NEW}-1")
    original_iterdir = Path.iterdir
    results = []
    for reverse in (False, True):
        monkeypatch.setattr(
            Path,
            "iterdir",
            lambda self, _r=reverse: iter(sorted(original_iterdir(self), reverse=_r)),
        )
        results.append(find_latest_run(tmp_path, "novelty"))
    assert results[0] == results[1]
    assert results[0][1].name == f"drift_novelty_{NEW}-1.md"


# --- validation: find_latest_session --------------------------------------


def _make_session(out_dir: Path, term: str, now: datetime) -> Path:
    passages = {key: {"text": f"visible passage {key}", "side": "literature"} for key in "abcd"}
    session_dir, _ = create_session(
        term,
        params={"seed": 42, "passages_shown_per_cluster": 8, "readers": ["m1"]},
        sides={"literature": {"validated": True, "note": None}},
        partitions=[
            {
                "revision": 0,
                "side": "literature",
                "kind": "original",
                "assignments": {"a": 0, "b": 0, "c": 1, "d": 1},
                "clusters": [
                    {"cluster_id": 0, "n_passages": 2, "silhouette": None, "shown": ["a", "b"]},
                    {"cluster_id": 1, "n_passages": 2, "silhouette": None, "shown": ["c", "d"]},
                ],
            }
        ],
        passages=passages,
        readings=[],
        output_dir=out_dir,
        now=now,
    )
    return session_dir


DAMAGE = {
    "truncated": b'{"schema": "cda-validation-session/1", "term": "nov',
    "other_schema": json.dumps({"schema": "cda-validation-session/0", "term": "novelty"}).encode(),
    "empty": b"",
    "non_utf8": b'{"schema": "cda-validation-session/1", "term": "nov\xff"}',
}


@pytest.mark.parametrize("damage", sorted(DAMAGE))
def test_find_latest_session_skips_a_damaged_older_session(tmp_path, damage):
    """A damaged OLDER session.json used to raise ValueError out of the
    finder, so the intact newer session could not be resumed."""
    older = _make_session(tmp_path, "novelty", datetime(2026, 10, 19, 9, 0, 0))
    newer = _make_session(tmp_path, "novelty", datetime(2026, 10, 19, 10, 0, 0))
    (older / "session.json").write_bytes(DAMAGE[damage])
    assert load_session(newer)["term"] == "novelty"
    assert find_latest_session(tmp_path, "novelty") == newer


def test_find_latest_session_skips_a_damaged_newer_session_for_an_intact_older_one(tmp_path):
    older = _make_session(tmp_path, "novelty", datetime(2026, 10, 19, 9, 0, 0))
    newer = _make_session(tmp_path, "novelty", datetime(2026, 10, 19, 10, 0, 0))
    (newer / "session.json").write_bytes(DAMAGE["truncated"])
    assert find_latest_session(tmp_path, "novelty") == older


@not_root
def test_find_latest_session_skips_an_older_session_without_read_permission(tmp_path):
    older = _make_session(tmp_path, "novelty", datetime(2026, 10, 19, 9, 0, 0))
    newer = _make_session(tmp_path, "novelty", datetime(2026, 10, 19, 10, 0, 0))
    (older / "session.json").chmod(0)
    try:
        assert find_latest_session(tmp_path, "novelty") == newer
    finally:
        (older / "session.json").chmod(0o644)


def test_find_latest_session_lists_skipped_directories_when_none_remain(tmp_path):
    first = _make_session(tmp_path, "C#", datetime(2026, 10, 19, 9, 0, 0))
    second = _make_session(tmp_path, "C#", datetime(2026, 10, 19, 10, 0, 0))
    (first / "session.json").write_bytes(DAMAGE["truncated"])
    (second / "session.json").write_bytes(DAMAGE["other_schema"])
    with pytest.raises(FileNotFoundError) as excinfo:
        find_latest_session(tmp_path, "C#")
    message = str(excinfo.value)
    assert "2 session directories for this term could not be loaded" in message
    assert first.name in message and "not valid JSON" in message
    assert second.name in message and "schema" in message
    assert "cda validate --term 'C#'" in message


def test_find_latest_session_error_without_damage_does_not_mention_skips(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        find_latest_session(tmp_path, "novelty")
    assert "skipped" not in str(excinfo.value)
    assert "cda validate --term novelty" in str(excinfo.value)


def test_find_latest_session_keeps_the_stored_term_check_while_skipping(tmp_path):
    """A damaged C# session must not let a C++ request resume a C# one."""
    wanted = _make_session(tmp_path, "C++", datetime(2026, 10, 19, 9, 0, 0))
    damaged = _make_session(tmp_path, "C#", datetime(2026, 10, 19, 10, 0, 0))
    other = _make_session(tmp_path, "C#", datetime(2026, 10, 19, 11, 0, 0))
    (damaged / "session.json").write_bytes(DAMAGE["truncated"])
    assert find_latest_session(tmp_path, "C++") == wanted
    assert find_latest_session(tmp_path, "C#") == other


def test_find_latest_session_never_loads_other_slug_directories(tmp_path, monkeypatch):
    _make_session(tmp_path, "novelty", datetime(2026, 10, 19, 9, 0, 0))
    other = _make_session(tmp_path, "novelty claims", datetime(2026, 10, 19, 10, 0, 0))
    (other / "session.json").write_bytes(DAMAGE["non_utf8"])
    loaded: list[str] = []
    original = session_module.load_session

    def recording_load(session_dir):
        loaded.append(Path(session_dir).name)
        return original(session_dir)

    monkeypatch.setattr(session_module, "load_session", recording_load)
    assert find_latest_session(tmp_path, "novelty").name == "validation_novelty_20261019_090000"
    assert loaded == ["validation_novelty_20261019_090000"]


def test_find_latest_session_is_independent_of_directory_listing_order(tmp_path, monkeypatch):
    older = _make_session(tmp_path, "novelty", datetime(2026, 10, 19, 9, 0, 0))
    _make_session(tmp_path, "novelty", datetime(2026, 10, 19, 10, 0, 0))
    newest = _make_session(tmp_path, "novelty", datetime(2026, 10, 19, 10, 0, 0))  # "-1"
    (older / "session.json").write_bytes(DAMAGE["empty"])
    original_iterdir = Path.iterdir
    results = []
    for reverse in (False, True):
        monkeypatch.setattr(
            Path,
            "iterdir",
            lambda self, _r=reverse: iter(sorted(original_iterdir(self), reverse=_r)),
        )
        results.append(find_latest_session(tmp_path, "novelty"))
    assert results == [newest, newest]


# --- GUI: "Resume latest session for this term" ---------------------------


def _resume_in_gui(output_dir: Path, term: str):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    if str(ROOT / "gui") not in sys.path:
        sys.path.insert(0, str(ROOT / "gui"))
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    output_input = next(t for t in app.get("text_input") if t.label == "Output directory")
    output_input.set_value(str(output_dir))
    app.text_input(key="validate_term").set_value(term)
    app.run()
    assert not app.exception, f"GUI raised: {app.exception}"
    app.button(key="validate_resume").click().run()
    return app


def test_gui_resume_latest_skips_a_damaged_older_session(tmp_path):
    """The button used to surface an uncaught ValueError for the older,
    damaged session instead of resuming the intact newer one."""
    older = _make_session(tmp_path, "novelty", datetime(2026, 10, 19, 9, 0, 0))
    newer = _make_session(tmp_path, "novelty", datetime(2026, 10, 19, 10, 0, 0))
    (older / "session.json").write_bytes(DAMAGE["truncated"])
    app = _resume_in_gui(tmp_path, "novelty")
    assert not app.exception, f"GUI raised: {app.exception}"
    assert app.session_state["validation_session_dir"] == str(newer)
    assert any(newer.name in c.value for c in app.caption)


def test_gui_resume_latest_shows_an_error_when_only_damaged_sessions_exist(tmp_path):
    only = _make_session(tmp_path, "novelty", datetime(2026, 10, 19, 9, 0, 0))
    (only / "session.json").write_bytes(DAMAGE["non_utf8"])
    app = _resume_in_gui(tmp_path, "novelty")
    assert not app.exception, f"GUI raised: {app.exception}"
    assert "validation_session_dir" not in app.session_state
    errors = " ".join(e.value for e in app.error)
    assert only.name in errors and "could not be loaded" in errors
