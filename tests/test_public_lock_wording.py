"""The public demo tells a visitor to retry only when a retry can help.

Unsupported OS locks, unreadable guards and lock entries that cannot be
moved require operator action. Public messages omit host paths; local mode
keeps the full exception text. ``CDA_GUI_APP_PATH`` selects the app to test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cda.corpus.ingest import CorpusLockError

pytest.importorskip("streamlit")

APP_PATH = os.environ.get(
    "CDA_GUI_APP_PATH", str(Path(__file__).resolve().parents[1] / "gui" / "app.py")
)
HOST = "/srv/x"
RETRY_NOTICE = "The demo cannot reach its corpus right now; please retry."
PERMANENT_NOTICE = (
    "The demo's corpus storage cannot be locked on this server; "
    "the operator needs to fix the installation."
)

UNSUPPORTED = (
    f"Cannot safely lock the corpus at {HOST}/.ingest.guard (ENOLCK: No locks available). "
    "The filesystem could not provide the required OS lock. No corpus operation was "
    "started. Use a corpus directory on a filesystem that supports OS file locks, or "
    "restore locking support and retry. Leave .ingest.guard in place."
)
HELD = (
    f"Another ingest or retrieval still holds {HOST}/.ingest.lock after 120s. Let the "
    "other corpus operation finish and retry. If no operation is running, a legacy lock "
    f"file may remain; remove only .ingest.lock and retry. Leave .ingest.guard in place."
)
GUARD_PERMISSION = (
    f"Cannot open the corpus lock guard {HOST}/.ingest.guard (owned by uid 1000, mode 0600): "
    "Permission denied. Every user who runs CDA on this corpus needs read access to this "
    f"permanent coordination file. Ask its owner to run `chmod a+r {HOST}/.ingest.guard` "
    "(or `chmod 666` for a shared corpus) and retry."
)
UNMOVABLE = (
    f"{HOST}/.ingest.lock is not a lock file CDA can read (a directory, a dangling link, "
    "or a file without read permission) and could not be moved aside (PermissionError: "
    "denied). Remove it by hand and retry; leave .ingest.guard in place."
)

PERMANENT_CASES = pytest.mark.parametrize(
    "detail",
    [UNSUPPORTED, GUARD_PERMISSION, UNMOVABLE],
    ids=["unsupported-filesystem", "guard-permission", "unmovable-lock-entry"],
)


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP_PATH, default_timeout=60).run()


def _click(app, label_part: str):
    [button] = [b for b in app.button if label_part in (b.label or "")]
    button.click().run()


def _page_text(app) -> str:
    parts = []
    for kind in ("markdown", "warning", "error", "info", "success", "caption", "code", "text"):
        parts.extend(str(getattr(e, "value", "")) for e in getattr(app, kind))
    return " ".join(parts)


def _errors(app) -> str:
    assert not app.exception, f"GUI raised: {app.exception}"
    return " ".join(e.value for e in app.error)


def _raising(monkeypatch, detail: str):
    import cda.demo

    def failing_demo(**_kwargs):
        raise CorpusLockError(detail)

    monkeypatch.setattr(cda.demo, "run_demo", failing_demo)


def _run_own_text_preview(monkeypatch, detail: str):
    _raising(monkeypatch, detail)
    app = _app()
    [area] = [a for a in app.text_area if a.label == "Your paragraphs"]
    area.set_value("Novelty is claimed in every abstract; novelty is rarely defined.").run()
    _click(app, "Preview the prompt (free, no model API)")
    return app


# --- the toy-corpus demo button --------------------------------------------


@PERMANENT_CASES
def test_demo_button_shows_the_permanent_notice_for_an_unlockable_corpus(monkeypatch, detail):
    monkeypatch.setenv("CDA_DEMO_MODE", "1")
    _raising(monkeypatch, detail)
    app = _app()
    _click(app, "Try the demo")
    errors = _errors(app)
    assert PERMANENT_NOTICE in errors
    assert "retry" not in errors.lower()  # a retry cannot help here
    page = _page_text(app)
    assert HOST not in page and "/srv/" not in page
    assert "chmod" not in page and "ENOLCK" not in page and "ingest.guard" not in page


def test_demo_button_keeps_the_retry_notice_for_a_held_corpus(monkeypatch):
    monkeypatch.setenv("CDA_DEMO_MODE", "1")
    _raising(monkeypatch, HELD)
    app = _app()
    _click(app, "Try the demo")
    errors = _errors(app)
    assert RETRY_NOTICE in errors
    assert PERMANENT_NOTICE not in errors
    assert "/srv/" not in _page_text(app)


def test_demo_button_keeps_the_retry_notice_while_a_lock_wait_is_announced(monkeypatch):
    """The announced wait wording ("waiting for it to finish") is transient too."""
    monkeypatch.setenv("CDA_DEMO_MODE", "1")
    _raising(
        monkeypatch,
        f"[corpus] another ingest or retrieval is running on this corpus (lock file: "
        f"{HOST}/.ingest.lock); waiting for it to finish…",
    )
    app = _app()
    _click(app, "Try the demo")
    errors = _errors(app)
    assert RETRY_NOTICE in errors
    assert "/srv/" not in _page_text(app)


# --- the paste-your-own-text preview ----------------------------------------


@PERMANENT_CASES
def test_own_text_preview_shows_the_permanent_notice_for_an_unlockable_corpus(monkeypatch, detail):
    monkeypatch.setenv("CDA_DEMO_MODE", "1")
    app = _run_own_text_preview(monkeypatch, detail)
    errors = _errors(app)
    assert PERMANENT_NOTICE in errors
    assert "retry" not in errors.lower()
    page = _page_text(app)
    assert "/srv/" not in page and "chmod" not in page


def test_own_text_preview_keeps_the_retry_notice_for_a_held_corpus(monkeypatch):
    monkeypatch.setenv("CDA_DEMO_MODE", "1")
    app = _run_own_text_preview(monkeypatch, HELD)
    errors = _errors(app)
    assert RETRY_NOTICE in errors
    assert PERMANENT_NOTICE not in errors
    assert "/srv/" not in _page_text(app)


# --- local mode is unchanged: the researcher gets the full text --------------


@pytest.mark.parametrize("detail", [UNSUPPORTED, HELD], ids=["unsupported", "held"])
def test_local_mode_shows_the_exception_text_with_its_path(monkeypatch, detail):
    monkeypatch.delenv("CDA_DEMO_MODE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _raising(monkeypatch, detail)
    app = _app()
    _click(app, "Try the demo")
    errors = _errors(app)
    assert detail in errors
    assert RETRY_NOTICE not in errors and PERMANENT_NOTICE not in errors
