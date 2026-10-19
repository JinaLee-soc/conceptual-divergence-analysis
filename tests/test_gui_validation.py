"""GUI and validation regressions: usage-statistics
config, embedder check before paid advice, alert severity styling, PCA with
no variance, silhouette formatting, and idle fixture flags."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from cda.cli import main
from cda.config import load_config
from cda.validation import agents
from cda.validation.agents import suggest_edit
from cda.validation.prompt_loader import load_validation_prompts
from cda.validation.report import render_validation_report
from cda.validation.session import create_session, record_verdict, save_session

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "gui"
APP_PATH = str(GUI / "app.py")
NOW = datetime(2026, 10, 19, 9, 0, 0)
OTHER_REVISION = "0" * 40
ALERT_KINDS = ("Error", "Warning", "Info", "Success")


def _gui_module(name):
    pytest.importorskip("streamlit")
    if str(GUI) not in sys.path:
        sys.path.insert(0, str(GUI))
    return __import__(name)


def _judged_session(tmp_path, **params):
    """Two judged literature clusters (alpha, beta), two passages each."""
    passages = {
        f"{topic}::{i}": {"text": f"{topic} usage case {i}", "side": "literature"}
        for topic in ("alpha", "beta")
        for i in range(2)
    }
    assignments = {chunk: (0 if chunk.startswith("alpha") else 1) for chunk in passages}
    session_dir, session = create_session(
        "novelty",
        params={"seed": 42, "embedding_model": str(load_config().get("embedding.model")), **params},
        sides={"literature": {"validated": True, "note": None}},
        partitions=[
            {
                "revision": 0,
                "side": "literature",
                "kind": "original",
                "assignments": assignments,
                "clusters": [
                    {
                        "cluster_id": c,
                        "n_passages": 2,
                        "silhouette": 0.2,
                        "shown": sorted(k for k, v in assignments.items() if v == c),
                    }
                    for c in (0, 1)
                ],
            }
        ],
        passages=passages,
        readings=[],
        output_dir=tmp_path,
        now=NOW,
    )
    for cluster in (0, 1):
        record_verdict(
            session,
            side="literature",
            cluster_id=cluster,
            revision=0,
            verdict="coherent",
            now=NOW,
        )
    save_session(session_dir, session)
    return session_dir, session


class _CountingProvider:
    model = "stub"

    def __init__(self):
        self.calls = 0

    def complete(self, **_kwargs):
        self.calls += 1
        return '{"recommendation": "merge", "argument": "same usage"}'


# --- usage statistics ---------------------------------------------------


def test_project_streamlit_config_turns_usage_statistics_off():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["browser"]["gatherUsageStats"] is False


def test_streamlit_reads_usage_statistics_off_under_a_clean_home(tmp_path):
    """No launcher flag and no user-level config: the project file alone decides."""
    pytest.importorskip("streamlit")
    env = {**os.environ, "HOME": str(tmp_path)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import streamlit.config as c; print(c.get_option('browser.gatherUsageStats'))",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize(("gathering", "expected"), [(False, "turned off"), (True, "ON")])
def test_demo_footer_states_the_live_usage_statistics_setting(monkeypatch, gathering, expected):
    shared = _gui_module("shared")
    monkeypatch.setattr(shared.st, "get_option", lambda name: gathering)
    assert expected in shared.usage_statistics_note()
    app_source = (GUI / "app.py").read_text(encoding="utf-8")
    assert "The operator disables Streamlit's usage statistics" not in app_source
    assert "usage_statistics_note()" in app_source


# --- embedder check before the paid revision call --------------------------


def test_suggest_edit_refuses_a_mismatched_embedder_before_any_paid_call(
    tmp_path, anchored_embedder
):
    _dir, session = _judged_session(tmp_path, embedding_revision=OTHER_REVISION)
    provider = _CountingProvider()
    with pytest.raises(ValueError, match="embedding revision"):
        suggest_edit(
            session,
            kind="merge",
            side="literature",
            clusters=[0, 1],
            config=load_config(),
            embedder=anchored_embedder,
            prompts=load_validation_prompts(load_config()),
            now=NOW,
            provider=provider,
            progress=lambda _m: None,
        )
    assert provider.calls == 0
    assert session["pending_edit"] is None


def test_suggest_split_refuses_a_mismatched_embedder_before_any_paid_call(
    tmp_path, anchored_embedder
):
    _dir, session = _judged_session(tmp_path, embedding_model="some/other-model")
    provider = _CountingProvider()
    with pytest.raises(ValueError, match="embedding model"):
        suggest_edit(
            session,
            kind="split",
            side="literature",
            clusters=[0],
            config=load_config(),
            embedder=anchored_embedder,
            prompts=load_validation_prompts(load_config()),
            now=NOW,
            provider=provider,
            progress=lambda _m: None,
        )
    assert provider.calls == 0


def test_gui_suggest_button_shows_the_embedder_error_without_calling_the_model(
    tmp_path, monkeypatch
):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    session_dir, _session = _judged_session(tmp_path, embedding_revision=OTHER_REVISION)
    built = []
    monkeypatch.setattr(
        agents, "reader_provider", lambda *a, **k: built.append(1) or _CountingProvider()
    )
    monkeypatch.delenv("CDA_DEMO_MODE", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.session_state["validation_session_dir"] = str(session_dir)
    app.run()
    assert not app.exception
    app.multiselect(key="validate_merge").set_value([0, 1]).run()
    button = app.button(key="validate_suggest")
    assert not button.disabled
    button.click().run()
    assert not app.exception
    assert built == []
    assert any("embedding revision" in e.value for e in app.error)


# --- alert severity -------------------------------------------------------


def _css() -> str:
    return (GUI / "styles.css").read_text(encoding="utf-8")


def test_each_alert_kind_renders_as_its_own_element():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    def script():
        import streamlit as st

        st.error("e")
        st.warning("w")
        st.info("i")
        st.success("s")

    app = AppTest.from_function(script).run()
    assert [len(app.error), len(app.warning), len(app.info), len(app.success)] == [1, 1, 1, 1]
    # The browser testid is `stAlertContent` + capitalised kind; the CSS keys on it.
    import streamlit

    bundle = "".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in (Path(streamlit.__file__).parent / "static" / "static" / "js").glob("index.*.js")
    )
    assert "stAlertContent${" in bundle


def test_css_gives_each_alert_kind_its_own_colour():
    css = _css()
    colours = {}
    for kind in ALERT_KINDS:
        border = re.search(
            rf'\[data-testid="stAlert"\]:has\(\[data-testid="stAlertContent{kind}"\]\)'
            r"\s*\{[^}]*border-left:[^;}]*?(#[0-9a-fA-F]{3,6})",
            css,
        )
        icon = re.search(
            rf'\[data-testid="stAlertContent{kind}"\] \[data-testid="stAlertDynamicIcon"\]'
            r"\s*\{[^}]*color:\s*(#[0-9a-fA-F]{3,6})",
            css,
        )
        assert border and icon, f"no per-kind rule for {kind}"
        colours[kind] = border.group(1).lower()
    assert len({colours["Error"], colours["Warning"], colours["Success"]}) == 3
    assert "grayscale" not in css


def test_unstable_k_warning_keeps_its_marker():
    visuals = _gui_module("visuals")
    seen = {}

    class _St:
        def warning(self, message, icon=None):
            seen["icon"] = icon

        def caption(self, message):
            pass

    visuals.render_stability_note(
        _St(),
        {
            "enabled": True,
            "mode": "joint",
            "canonical_k": 3,
            "trials": 5,
            "k_match_rate": 0.5,
            "k_seen": [2, 3],
        },
    )
    assert seen["icon"] == ":material/warning:"


# --- PCA with no variance ------------------------------------------------


def test_pca_share_is_zero_when_all_embeddings_are_identical():
    visuals = _gui_module("visuals")
    identical = np.tile(np.array([[0.6, 0.8, 0.0]]), (4, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coordinates, share = visuals._pca_2d(identical)
    assert share == 0.0
    assert np.isfinite(coordinates).all()
    assert f"{share:.0%}" == "0%"


def test_pca_share_is_unchanged_for_ordinary_embeddings():
    visuals = _gui_module("visuals")
    _coords, share = visuals._pca_2d(np.random.default_rng(0).normal(size=(6, 8)))
    assert 0.0 < share <= 1.0


# --- silhouettes in the curation history -------------------------------------


def test_report_formats_per_cluster_silhouettes_and_missing_ones(tmp_path):
    _dir, session = _judged_session(tmp_path)
    session["edits"].append(
        {
            "edit_id": 1,
            "kind": "merge",
            "side": "literature",
            "clusters": [0, 1],
            "revision": 0,
            "status": "applied",
            "applied_proposal": None,
            "agent_advice": {"recommendation": "merge", "argument": "same"},
            "silhouette_before": {"0": 0.21, "1": 0.3456},
            "silhouette_after": {"0": None},
        }
    )
    report = render_validation_report(session, load_config(), NOW)
    line = next(row for row in report.splitlines() if "silhouette before" in row)
    assert line == (
        "  - silhouette before: cluster 0: 0.2100, cluster 1: 0.3456 | after: cluster 0: n/a"
    )
    assert "{" not in line and "None" not in line


# --- idle --overwrite-fixtures -----------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze", "--term", "x", "--overwrite-fixtures"],
        ["analyze", "--term", "x", "--dry-run", "--overwrite-fixtures"],
        ["analyze", "--term", "x", "--mock-llm", "--overwrite-fixtures"],
        ["demo", "--overwrite-fixtures"],
        ["demo", "--dry-run", "--overwrite-fixtures"],
        ["demo", "--mock-llm", "--overwrite-fixtures"],
    ],
)
def test_overwrite_fixtures_without_recording_is_refused(argv, capsys, monkeypatch):
    def _must_not_run(*_a, **_k):
        raise AssertionError("the command ran despite an idle flag")

    monkeypatch.setattr("cda.pipeline.analyze_term", _must_not_run)
    monkeypatch.setattr("cda.demo.run_demo", _must_not_run)
    code = main(argv)
    assert code != 0
    err = capsys.readouterr().err
    assert "--overwrite-fixtures" in err and "--record-fixtures" in err


def test_report_view_hides_metadata_comments_but_download_keeps_them(tmp_path, monkeypatch):
    shared = _gui_module("shared")
    path = tmp_path / "report.md"
    header = "<!-- generated: 2026-10-19 | model: mock -->\n<!-- chunks_total: 4 -->\n\n"
    body = "# CDA Report\n\nThe quoted usage remains unchanged.\n"
    path.write_text(header + body)
    rendered, downloads = [], []
    monkeypatch.setattr(shared.st, "caption", lambda *_: None)
    monkeypatch.setattr(shared.st, "markdown", rendered.append)
    monkeypatch.setattr(
        shared.st, "download_button", lambda *args, **kwargs: downloads.append(kwargs["data"])
    )
    shared.render_report_file(path)
    assert rendered == [body]
    assert downloads == [header + body]
    assert path.read_text() == header + body
