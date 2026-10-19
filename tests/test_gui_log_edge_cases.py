"""The public demo's notices never name the host; the local log streams every line.

Regressions for: the demo tab rendering the pipeline's skip reason (which
embeds the host's drafts directory) on the public demo, the public lock
message claiming a cause it cannot know, and the live log swallowing
legitimate repeated lines in local mode.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cda.corpus.ingest import CorpusLockError

pytest.importorskip("streamlit")

APP_PATH = str(Path(__file__).resolve().parents[1] / "gui" / "app.py")
HOST = "/srv/private/host"
PUBLIC_LOCK_MESSAGE = "The demo cannot reach its corpus right now; please retry."
PASSAGE_LINE = "  [ingest] secret.pdf: removing chunks stored under a different Unicode spelling"


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP_PATH, default_timeout=60).run()


def _click_demo(app):
    [button] = [b for b in app.button if "Try the demo" in (b.label or "")]
    button.click().run()


def _page_text(app) -> str:
    """Every rendered text element the demo and own-text panels can write to."""
    parts = []
    for kind in ("markdown", "warning", "error", "info", "success", "caption", "code", "text"):
        parts.extend(str(getattr(e, "value", "")) for e in getattr(app, kind))
    return "\n".join(parts)


def _skipped_demo(**kwargs):
    kwargs["progress"](f"[demo] skipped: nothing under {HOST}/drafts")
    return {
        "term": kwargs.get("term", "novelty"),
        "status": "skipped",
        "reason": f"nothing under {HOST}/drafts",
        "stats": {"researcher_passages": 0},
    }


# --- B2: a skipped demo run never names the host --------------------------


def test_public_demo_skip_notice_names_no_host_path(monkeypatch):
    import cda.demo

    monkeypatch.setenv("CDA_DEMO_MODE", "1")
    monkeypatch.setattr(cda.demo, "run_demo", _skipped_demo)
    app = _app()
    _click_demo(app)
    assert not app.exception, f"GUI raised: {app.exception}"
    page = _page_text(app)
    assert "/srv/" not in page and HOST not in page
    warnings = " ".join(w.value for w in app.warning)
    assert "too few passages for *novelty*" in warnings
    assert "(passages found: 0)" in warnings


def test_local_demo_skip_notice_keeps_the_pipelines_reason(monkeypatch):
    import cda.demo

    monkeypatch.delenv("CDA_DEMO_MODE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cda.demo, "run_demo", _skipped_demo)
    app = _app()
    _click_demo(app)
    assert not app.exception, f"GUI raised: {app.exception}"
    warnings = " ".join(w.value for w in app.warning)
    assert f"nothing under {HOST}/drafts" in warnings  # locally the path is the detail


@pytest.mark.parametrize(
    "detail",
    [
        f"Another ingest or retrieval still holds {HOST}/.ingest.lock after 300s",
        f"Could not reclaim the stale lock {HOST}/.ingest.lock (PermissionError)",
    ],
    ids=["held", "unreclaimable"],
)
def test_public_lock_failure_is_generic_whatever_its_cause(monkeypatch, detail):
    import cda.demo

    def failing_demo(**kwargs):
        raise CorpusLockError(detail)

    monkeypatch.setenv("CDA_DEMO_MODE", "1")
    monkeypatch.setattr(cda.demo, "run_demo", failing_demo)
    app = _app()
    _click_demo(app)
    assert not app.exception, f"GUI raised: {app.exception}"
    errors = " ".join(e.value for e in app.error)
    assert PUBLIC_LOCK_MESSAGE in errors
    assert "Another run holds" not in errors  # a cause the GUI cannot know
    assert "/srv/" not in _page_text(app)


# --- C1: the local log streams repeats; the public log collapses labels ----


def _repeating_demo(**kwargs):
    progress = kwargs["progress"]
    for _ in range(3):
        progress(PASSAGE_LINE)  # one identical sentence per alternate spelling
    progress("[1/3] extracting passages for 'novelty'")
    progress("[1/3] extracting passages for 'novelty'")
    raise RuntimeError("stopped after logging")


def test_local_log_streams_every_repeated_line(monkeypatch):
    import cda.demo

    monkeypatch.delenv("CDA_DEMO_MODE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cda.demo, "run_demo", _repeating_demo)
    app = _app()
    _click_demo(app)
    assert not app.exception, f"GUI raised: {app.exception}"
    # Markdown rendering drops the log line's leading indent; compare stripped.
    streamed = [m.value.strip() for m in app.markdown]
    assert streamed.count(PASSAGE_LINE.strip()) == 3
    assert streamed.count("[1/3] extracting passages for 'novelty'") == 2


def test_public_log_shows_each_consecutive_stage_label_once(monkeypatch):
    import cda.demo

    monkeypatch.setenv("CDA_DEMO_MODE", "1")
    monkeypatch.setattr(cda.demo, "run_demo", _repeating_demo)
    app = _app()
    _click_demo(app)
    assert not app.exception, f"GUI raised: {app.exception}"
    streamed = [m.value.strip() for m in app.markdown]
    assert streamed.count("Extracting matching passages…") == 1
    assert PASSAGE_LINE.strip() not in streamed
