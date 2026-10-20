"""Edge cases across the GUI, the CLI, CI workflows, and their docs.

Each test pins one reviewed defect: an endorsement that survived the edit it
vouched for, alert severities that looked identical, host paths on the public
demo, a `--demo` command that failed on a fresh clone, and workflow or doc
claims the code did not honour.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from cda.config import load_config
from cda.validation.curate import apply_pending_edit
from cda.validation.report import render_validation_report
from cda.validation.session import endorse, load_session, record_verdict, save_session
from tests.test_validation_curate import (
    alpha_beta_session,
    hierarchical_joint_session,
    judge_all_original_clusters,
    pending_merge,
)

REPO = Path(__file__).resolve().parents[1]
APP_PATH = str(REPO / "gui" / "app.py")
NOW = datetime(2026, 7, 14, 11, 0, 0)
LATER = datetime(2026, 7, 14, 12, 0, 0)


def _quiet(_message: str) -> None:
    pass


def _endorse_then_merge(tmp_path, embedder):
    session_dir, session = alpha_beta_session(tmp_path)
    judge_all_original_clusters(session)
    endorse(session, note="looks right", now=NOW)
    pending_merge(session, clusters=(0, 1))
    apply_pending_edit(
        session,
        "apply-merge",
        load_config(),
        embedder=embedder,
        session_dir=session_dir,
        now=LATER,
        progress=_quiet,
    )
    return session_dir, session


# --- an applied edit withdraws the endorsement ------------------------


def test_applied_merge_withdraws_the_endorsement(tmp_path, anchored_embedder):
    session_dir, session = _endorse_then_merge(tmp_path, anchored_embedder)

    assert session["endorsement"] is None
    [withdrawal] = session["endorsement_withdrawals"]
    assert withdrawal["revisions"] == {"literature": 0}
    assert withdrawal["withdrawn_at_revisions"] == {"literature": 1}
    assert withdrawal["withdrawn_by_edit"] == 1
    assert withdrawal["note"] == "looks right"
    # Persisted, not only in memory: the session file is the audit trail.
    assert load_session(session_dir)["endorsement_withdrawals"] == [withdrawal]


def test_report_shows_the_withdrawal_instead_of_endorsed(tmp_path, anchored_embedder):
    _session_dir, session = _endorse_then_merge(tmp_path, anchored_embedder)
    report = render_validation_report(session, load_config(), LATER)
    status = report.split("## Status", 1)[1].split("## Run Parameters", 1)[0]

    assert "**Endorsed**" not in status
    assert "**withdrawn** at revision(s) {'literature': 1}" in status
    assert "Session open: not yet endorsed." in status


def test_rejected_suggestion_keeps_the_endorsement(tmp_path, anchored_embedder):
    session_dir, session = alpha_beta_session(tmp_path)
    judge_all_original_clusters(session)
    endorse(session, now=NOW)
    pending_merge(session)
    apply_pending_edit(
        session,
        "reject",
        load_config(),
        embedder=anchored_embedder,
        session_dir=session_dir,
        now=LATER,
        progress=_quiet,
    )
    assert session["endorsement"] is not None
    assert not session.get("endorsement_withdrawals")


def test_applied_recut_withdraws_the_endorsement(tmp_path, anchored_embedder):
    from cda.validation.curate import apply_recut

    session_dir, session = hierarchical_joint_session(tmp_path, anchored_embedder)
    judge_all_original_clusters(session)
    endorse(session, now=NOW)
    apply_recut(
        session,
        2,
        load_config(),
        embedder=anchored_embedder,
        session_dir=session_dir,
        now=LATER,
        progress=_quiet,
    )
    assert session["endorsement"] is None
    [withdrawal] = session["endorsement_withdrawals"]
    assert withdrawal["withdrawn_at_revisions"] == {"researcher": 1, "literature": 1}


def test_report_flags_a_legacy_endorsement_older_than_the_structure(tmp_path, anchored_embedder):
    """Sessions saved before withdrawal existed can hold an endorsement whose
    revisions an edit has already moved past; it must not read as current."""
    _session_dir, session = _endorse_then_merge(tmp_path, anchored_embedder)
    legacy = session["endorsement_withdrawals"].pop()
    session["endorsement"] = {k: legacy[k] for k in ("revisions", "note", "at")}

    report = render_validation_report(session, load_config(), LATER)
    status = report.split("## Status", 1)[1].split("## Run Parameters", 1)[0]
    assert "**Endorsed**" not in status
    assert "Endorsement out of date" in status


def _judge_current_revision(session_dir, session):
    for partition in session["partitions"]:
        if partition["revision"] != 1:
            continue
        for cluster in partition["clusters"]:
            if not any(
                v["revision"] == 1 and v["cluster_id"] == cluster["cluster_id"]
                for v in session["verdicts"]
            ):
                record_verdict(
                    session,
                    side=partition["side"],
                    cluster_id=cluster["cluster_id"],
                    revision=1,
                    verdict="coherent",
                    now=LATER,
                )
    save_session(session_dir, session)


def test_gui_offers_endorse_again_after_an_edit(tmp_path, anchored_embedder):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    session_dir, session = _endorse_then_merge(tmp_path, anchored_embedder)
    _judge_current_revision(session_dir, session)

    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["validation_session_dir"] = str(session_dir)
    app.run()
    assert not app.exception, f"GUI raised: {app.exception}"

    endorse_buttons = [b for b in app.button if b.label.startswith("Endorse this structure")]
    assert len(endorse_buttons) == 1 and not endorse_buttons[0].disabled
    assert not any(s.value.startswith("Endorsed at") for s in app.success)
    assert any("was withdrawn" in w.value for w in app.warning)


# --- alert severities are distinguishable without colour ---------------


def _alert_rules(css: str) -> dict[str, str]:
    """Declarations of each stAlert:has(stAlertContent<Kind>) rule, by kind."""
    pattern = re.compile(
        r'\[data-testid="stAlert"\]:has\(\[data-testid="stAlertContent(\w+)"\]\)\s*\{([^}]*)\}'
    )
    return {kind: " ".join(body.split()) for kind, body in pattern.findall(css)}


def test_alert_kinds_have_distinct_borders_and_text_labels():
    css = (REPO / "gui" / "styles.css").read_text(encoding="utf-8")
    rules = _alert_rules(css)
    kinds = ("Error", "Warning", "Info", "Success")
    assert set(kinds) <= set(rules)
    assert len({rules[kind] for kind in kinds}) == len(kinds), rules
    # Weight alone is not enough for every reader: the border STYLES differ too.
    styles = {kind: re.search(r"\b(solid|dashed|dotted|double)\b", rules[kind]) for kind in kinds}
    assert all(styles.values()), rules
    assert len({m.group(1) for m in styles.values()}) >= 3, rules
    for kind, label in (("Error", "Error"), ("Warning", "Warning")):
        assert re.search(
            rf'\[data-testid="stAlertContent{kind}"\][^{{]*::before\s*\{{\s*content:\s*"{label}"',
            css,
        ), f"no text label for {kind} alerts"


# --- the public demo never shows host paths --------------------------


@pytest.fixture
def demo_mode(monkeypatch):
    pytest.importorskip("streamlit")
    monkeypatch.setenv("CDA_DEMO_MODE", "1")


def test_demo_mode_paste_without_the_term_gets_a_demo_message(demo_mode):
    from streamlit.testing.v1 import AppTest

    host_path = "/srv/private/host/demo/toy_corpus/drafts"
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["own_text_result"] = {
        "term": "novelty",
        "status": "skipped",
        "reason": (
            f"Only 0 researcher passages found; minimum is 3. Add drafts to {host_path}, "
            "tag your papers 'mywork' in Zotero and re-ingest."
        ),
        "stats": {"researcher_passages": 0},
    }
    app.run()
    assert not app.exception, f"GUI raised: {app.exception}"
    warnings = " ".join(w.value for w in app.warning)
    assert "Your pasted text does not contain the term *novelty*" in warnings
    assert host_path not in warnings and "Zotero" not in warnings


def _click(app, label_start: str):
    [button] = [b for b in app.button if b.label.startswith(label_start)]
    button.click().run()


def test_demo_mode_fixture_miss_hides_the_technical_path(demo_mode, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import cda.demo
    from cda.llm.base import FixtureMissError

    host_path = "/srv/private/host/demo/fixtures/0123456789abcdef.json"

    def missing_fixture(**_kwargs):
        raise FixtureMissError(f"Mock fixture not found.\n  looked for : {host_path}")

    monkeypatch.setattr(cda.demo, "run_demo", missing_fixture)
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _click(app, "▶ Try the demo")
    errors = " ".join(e.value for e in app.error)
    assert "could not be replayed" in errors
    assert host_path not in errors and "Technical detail" not in errors


def test_local_fixture_miss_keeps_the_technical_detail(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import cda.demo
    from cda.llm.base import FixtureMissError

    monkeypatch.delenv("CDA_DEMO_MODE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def missing_fixture(**_kwargs):
        raise FixtureMissError("looked for : /local/demo/fixtures/x.json")

    monkeypatch.setattr(cda.demo, "run_demo", missing_fixture)
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _click(app, "▶ Try the demo")
    errors = " ".join(e.value for e in app.error)
    assert "Technical detail" in errors and "/local/demo/fixtures/x.json" in errors
    # The advice matches what replay does: a live recording under the
    # configured model is preferred over the committed fixture.
    assert "recorded under your configured model" in errors


def test_live_demo_rerun_replaces_this_machines_recording(monkeypatch):
    """A second live demo click must not be refused for the recording the
    first click made (the Analyze tab already overwrites the same way)."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import cda.demo

    monkeypatch.delenv("CDA_DEMO_MODE", raising=False)
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return {"term": kwargs["term"], "status": "skipped", "reason": "captured"}

    monkeypatch.setattr(cda.demo, "run_demo", capture)
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["anthropic_session_key"] = "sk-ant-test-not-a-real-key"
    app.run()
    _click(app, "▶ Try the demo (live API call")
    assert not app.exception, f"GUI raised: {app.exception}"
    [kwargs] = calls
    assert kwargs["mock"] is False
    assert kwargs["record_fixtures"] is True and kwargs["overwrite_fixtures"] is True


# --- `cda diagnose --demo` on a fresh clone --------------------------


def test_diagnose_demo_builds_the_toy_corpus_first(tmp_path, monkeypatch, capsys, fake_embedder):
    """Before the fix the command died with "No corpus found" because only
    `cda demo`, `emergence --demo`, and `validate --demo` indexed the toy
    corpus. With an empty OPENALEX_MAILTO the run now gets past the corpus
    and stops, offline, at the OpenAlex address check."""
    import cda.corpus.embeddings
    from cda.cli import main

    shutil.copytree(REPO / "demo" / "toy_corpus", tmp_path / "demo" / "toy_corpus")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cda.corpus.embeddings, "get_embedder", lambda _config: fake_embedder)
    monkeypatch.setenv("OPENALEX_MAILTO", "")

    code = main(["diagnose", "--demo"])
    err = capsys.readouterr().err
    assert "No corpus found" not in err
    assert code == 1 and "OPENALEX_MAILTO" in err
    assert (tmp_path / "demo" / "_runtime" / "corpus").is_dir()


# --- CI workflows ----------------------------------------------------


def _workflow(name: str) -> dict:
    return yaml.safe_load((REPO / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def _step(workflow: dict, job: str, name_start: str) -> str:
    [step] = [s for s in workflow["jobs"][job]["steps"] if s.get("name", "").startswith(name_start)]
    return step["run"]


def test_ci_tests_every_python_the_readme_claims():
    matrix = _workflow("ci.yml")["jobs"]["test-offline"]["strategy"]["matrix"]
    assert matrix["python-version"] == ["3.11", "3.12", "3.13"]
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "3.11, 3.12, or 3.13" in readme


def test_record_workflow_stages_new_fixtures_before_checking_for_changes():
    script = _step(_workflow("record-fixtures.yml"), "record", "Commit the new fixture")
    add = script.index("git add demo/fixtures")
    check = script.index("git diff --cached --quiet")
    assert add < check
    # The unstaged check misses untracked files, i.e. every new recording.
    assert "git diff --quiet" not in script


def test_record_workflow_proves_the_new_recording_replays():
    """A plain mock run would pass on the committed fallback fixture, so the
    check must assert the report names the configured model."""
    script = _step(_workflow("record-fixtures.yml"), "record", "Prove the NEW recording")
    assert 'get("llm.model")' in script
    assert "| model: {model} |" in script


# --- docs say what the code does ------------------------------------------


def test_docs_state_the_actual_recut_and_overwrite_rules():
    guide = " ".join((REPO / "docs" / "interpretation_guide.md").read_text().split())
    assert "the recut is GUI-only" in guide
    assert "`cda validate` lets you move the cutoff" not in guide

    privacy = " ".join((REPO / "docs" / "privacy_and_cost.md").read_text().split())
    assert "Recording never replaces a fixture that already replays" not in privacy
    assert "GUI's live Analyze and demo buttons replace it" in privacy

    fixtures = " ".join((REPO / "demo" / "fixtures" / "README.md").read_text().split())
    assert "configured `llm.model`" in fixtures
    assert "to replace the committed file for a term, add `--overwrite-fixtures`" not in fixtures
