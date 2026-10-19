"""Headless GUI smoke test via Streamlit's AppTest.

Runs the whole app script without a browser: no clicks are simulated, so no
embedding model loads and nothing touches the network. Skipped automatically
when streamlit isn't installed (it lives in the [gui] extra; dev environments
install it for this test)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

APP_PATH = str(Path(__file__).resolve().parents[1] / "gui" / "app.py")


def test_gui_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception, f"GUI raised: {app.exception}"
    assert app.title and "Conceptual Divergence Analysis" in app.title[0].value


def test_gui_offers_drag_and_drop():
    """Drag-and-drop uploaders must be present."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    markdown_text = " ".join(m.value for m in app.markdown)
    assert "Drag and drop" in markdown_text
    uploader_labels = " ".join(u.label for u in app.get("file_uploader"))
    assert "Drop PDFs here" in uploader_labels
    assert "Zotero CSV" in uploader_labels
    assert "working drafts" in uploader_labels


def test_gui_offers_paste_your_own_text_mode():
    """The GUI offers a text preview without requiring corpus setup."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    assert any("your own text" in (e.label or "") for e in app.expander)
    area_labels = " ".join(a.label for a in app.text_area)
    assert "Your paragraphs" in area_labels
    button_labels = " ".join(b.label for b in app.button)
    assert "Preview the prompt (free, offline)" in button_labels
    assert "Analyze live" in button_labels


def test_gui_offers_emergence_expander():
    """The experimental emergence timeline lives on the Analyze tab."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    assert any("Cluster emergence timeline" in (e.label or "") for e in app.expander)
    button_labels = " ".join(b.label for b in app.button)
    assert "Track emergence" in button_labels
    # The framing must state the axis: acquisition, not publication.
    markdown_text = " ".join(m.value for m in app.markdown)
    assert "not the field's publication history" in markdown_text


def test_gui_offers_validate_tab():
    """The researcher-led validation flow lives in its own tab."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    button_labels = " ".join(b.label for b in app.button)
    assert "Start validation" in button_labels
    assert "Resume latest session" in button_labels
    markdown_text = " ".join(m.value for m in app.markdown)
    assert "blind verdict" in markdown_text
    # The demo corpus is reachable from the GUI, mirroring `cda validate --demo`
    # (the CLI directs users here to record verdicts, so the GUI must be able
    # to open the same session).
    checkbox_labels = " ".join(c.label for c in app.checkbox)
    assert "toy demo corpus" in checkbox_labels


def test_gui_demo_validation_explains_missing_recordings(tmp_path, monkeypatch):
    """Demo mode + mock toggle + no recorded readings: the tab must say what
    to do BEFORE the click (one live run records and enables offline replay),
    in GUI terms, not CLI flags. The empty-fixtures state is reconstructed in
    tmp_path so this holds regardless of what demo/fixtures/validation holds."""
    from streamlit.testing.v1 import AppTest

    import cda.demo

    fake_demo = tmp_path / "demo"
    (fake_demo / "toy_corpus").mkdir(parents=True)
    monkeypatch.setattr(cda.demo, "find_demo_dir", lambda explicit=None: fake_demo)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # mock toggle defaults on
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    demo_box = next(c for c in app.checkbox if "toy demo corpus" in c.label)
    demo_box.check()
    app.run(timeout=60)
    assert not app.exception, f"GUI raised: {app.exception}"
    warnings = " ".join(str(w.value) for w in app.warning)
    assert "No recorded demo readings" in warnings
    assert "mock toggle" in warnings
    assert "offline forever" in warnings


def test_gui_never_shows_readings_before_a_verdict(tmp_path, monkeypatch):
    """Structural blinding: with a fresh session (readings recorded, zero
    verdicts), the agent's gloss/judgment text must be absent from the
    rendered app, while the cluster's passages are present."""
    import json
    from datetime import datetime

    from streamlit.testing.v1 import AppTest

    from cda.validation.session import create_session

    secret_gloss = "SECRET-AGENT-GLOSS-DO-NOT-SHOW"  # pragma: allowlist secret
    session_dir, _session = create_session(
        "novelty",
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
        passages={key: {"text": f"visible passage {key}", "side": "literature"} for key in "abcd"},
        readings=[
            {
                "side": "literature",
                "cluster_id": 0,
                "revision": 0,
                "model": "m1",
                "gloss": secret_gloss,
                "judgment": "coherent",
                "rationale": secret_gloss,
                "valid": True,
            }
        ],
        output_dir=tmp_path,
        now=datetime(2026, 7, 14, 12, 0, 0),
    )
    # Sanity: the secret is on disk (necessarily), in the session file only.
    assert secret_gloss in (session_dir / "session.json").read_text(encoding="utf-8")

    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["validation_session_dir"] = str(session_dir)
    app.run()
    assert not app.exception, f"GUI raised: {app.exception}"

    rendered = json.dumps(
        [m.value for m in app.markdown]
        + [c.value for c in app.caption]
        + [str(getattr(w, "value", "")) for w in app.get("info")]
    )
    assert "visible passage a" in rendered
    assert secret_gloss not in rendered


def test_gui_same_report_in_two_tabs_does_not_crash(tmp_path, monkeypatch):
    """Regression: a validation session's report is rendered by the Validate
    tab AND listed (and auto-selected) in the Reports tab in the same script
    pass; the two download buttons must not collide on their widget key
    (StreamlitDuplicateElementKey)."""
    from datetime import datetime

    from streamlit.testing.v1 import AppTest

    from cda.validation.session import create_session

    outputs = tmp_path / "outputs"
    session_dir, _session = create_session(
        "temporality",
        params={"seed": 42, "passages_shown_per_cluster": 8, "readers": ["m1"]},
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
        passages={key: {"text": f"passage {key}", "side": "literature"} for key in "ab"},
        readings=[],
        output_dir=outputs,
        now=datetime(2026, 7, 18, 12, 31, 17),
    )
    # The session's report file, exactly where the Validate tab looks for it.
    (session_dir / f"{session_dir.name}.md").write_text("# report body", encoding="utf-8")

    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["validation_session_dir"] = str(session_dir)
    app.run()
    # Point the sidebar's output dir at the same folder so the Reports tab
    # lists (and auto-selects) the very same report file, then rerun.
    output_input = next(t for t in app.get("text_input") if t.label == "Output directory")
    output_input.set_value(str(outputs)).run()
    assert not app.exception, f"GUI raised: {app.exception}"


def test_gui_offers_session_only_api_key_entry(monkeypatch):
    """The sidebar takes the key as a password field, session-memory only."""
    import os

    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    key_inputs = [t for t in app.get("text_input") if "API key" in (t.label or "")]
    assert len(key_inputs) == 1
    from streamlit.proto.TextInput_pb2 import TextInput as TextInputProto

    assert key_inputs[0].proto.type == TextInputProto.PASSWORD

    # Entering a key updates the status message without the secret reaching
    # the process environment (it is injected only around live runs and
    # scrubbed after; load_dotenv may leave the variable present but empty).
    key_inputs[0].set_value("sk-ant-test-not-a-real-key").run()
    assert not app.exception
    assert os.environ.get("ANTHROPIC_API_KEY", "") != "sk-ant-test-not-a-real-key"
    button_labels = " ".join(b.label for b in app.button)
    assert "Forget the key now" in button_labels


def test_gui_session_key_never_rendered(monkeypatch):
    """The key value must not appear anywhere in the rendered text."""
    import json

    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    secret = "sk-ant-SESSION-SECRET-VALUE-123456"  # pragma: allowlist secret
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["anthropic_session_key"] = secret
    app.run()
    assert not app.exception
    rendered = json.dumps(
        [m.value for m in app.markdown]
        + [c.value for c in app.caption]
        + [t.value for t in app.text]
    )
    assert secret not in rendered  # only the 7-char prefix + length may show


def test_gui_declares_its_network_scope():
    """The footer promise is user-facing, and it must be precise rather
    than flattering: a live run DOES send the extracted passages to the model
    provider, so the footer says so instead of implying nothing ever leaves."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    captions = " ".join(c.value for c in app.caption)
    assert "Anthropic API" in captions
    assert "OpenAlex" in captions
    # What genuinely stays local...
    assert "never leave" in captions
    # ...and what does not, on a live run.
    assert "term-sentences" in captions
    # The escape hatch for embargoed/NDA work.
    assert "local OpenAI-compatible endpoint" in captions


@pytest.mark.integration
def test_demo_button_click_renders_visual_summary(monkeypatch):
    """Click the demo button in the real Streamlit runtime (mock LLM).

    Integration tier: the demo loads the real embedding model. Verifies the
    click-to-run path end-to-end: button → pipeline → metrics + passage map.
    """
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # mock toggle defaults on
    app = AppTest.from_file(APP_PATH, default_timeout=240).run()
    # Select replay explicitly: another GUI test may leave a session key in
    # Streamlit's context, and integration tests must never spend an API key.
    next(t for t in app.toggle if t.label.startswith("Mock LLM")).set_value(True)
    app.run()
    demo_button = app.button[0]  # the prominent "Try the demo" button
    demo_button.click()
    app.run(timeout=240)

    assert not app.exception, f"demo click raised: {app.exception}"
    assert not app.error, [e.value for e in app.error]
    metric_labels = [m.label for m in app.metric]
    # Joint clustering is the default, so the headline metrics describe the
    # shared partition. (Under --independent they are the per-corpus k and
    # silhouette instead; see visuals.render_result_visuals.)
    assert "Shared clusters (joint k)" in metric_labels
    assert "…containing your passages" in metric_labels
    all_tabs = [t.label for t in app.tabs]
    assert "Visual summary" in all_tabs and "Full report" in all_tabs
    result_captions = " ".join(c.value for c in app.caption)
    assert "one term-sentence" in result_captions  # the passage map
    assert "coloured by shared cluster" in result_captions  # the joint map
    # The order-stability check reaches the GUI, as a caption when every
    # ordering agreed or a warning when they did not (toy corpus: unstable).
    stability_text = result_captions + " ".join(w.value for w in app.warning)
    assert "independently shuffled passage orderings" in stability_text


def test_openalex_email_is_never_prefilled_into_the_sidebar():
    """The address is personal data: an environment value must not be rendered
    into the input, the way the API key never is. A screenshot or screen share
    of a tool whose privacy claim is 'your material stays local' should not
    display anyone's email."""
    source = (Path(__file__).resolve().parents[1] / "gui" / "app.py").read_text(encoding="utf-8")
    sidebar_call = source.split('"OpenAlex email (for Diagnose)"')[1].split(")")[0]
    assert 'value=""' in sidebar_call
    assert "OPENALEX_MAILTO" not in sidebar_call


def test_setup_tab_describes_the_openalex_email_without_printing_it():
    """The Setup tab may confirm an address is configured and where it came
    from, but must never interpolate the address itself."""
    source = (Path(__file__).resolve().parents[1] / "gui" / "app.py").read_text(encoding="utf-8")
    status_fn = source.split("def openalex_mailto_status()")[1].split("\ndef ")[0]
    # It branches on the env var, but never renders its value.
    assert 'os.environ.get("OPENALEX_MAILTO", "").strip()' in status_fn
    assert "{mailto}" not in status_fn
    # And the Setup tab calls the describing helper, not the raw value.
    assert "openalex_mailto_status()" in source.split("def render_setup_tab")[1]


def test_gui_exposes_all_six_workflow_tabs():
    """Every tab in the documented workflow must render, in order. A tab that
    silently disappears (e.g. an exception while building it) is otherwise
    easy to miss, because Streamlit renders the rest of the page fine."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    labels = [t.label for t in app.tabs]
    expected = ["Setup", "Ingest", "Analyze", "Validate", "Reports", "Diagnose"]
    # The result tabs ("Visual summary"/"Full report") only exist after a run,
    # so at load time the top-level workflow tabs are the first six.
    assert labels[: len(expected)] == expected


def test_setup_tab_reports_environment_without_leaking_it(monkeypatch):
    """Setup is the 'is my install healthy' tab: it must state key/corpus
    status. It must do so without printing the secrets themselves."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SETUPTAB-SECRET-123456")
    monkeypatch.setenv("OPENALEX_MAILTO", "private@example.org")
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    rendered = " ".join([m.value for m in app.markdown] + [c.value for c in app.caption])
    assert "Anthropic API key" in rendered and "OpenAlex email" in rendered
    assert "Embedding model" in rendered and "Corpus" in rendered
    # Neither secret appears in full anywhere on the page.
    assert "sk-ant-SETUPTAB-SECRET-123456" not in rendered
    assert "private@example.org" not in rendered


def test_ingest_tab_offers_pdf_and_zotero_inputs():
    """Ingest is the corpus-building tab: it needs a PDF drop zone, a Zotero
    CSV input, and the re-ingest control the date_added backfill depends on."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    uploader_labels = " ".join(u.label for u in app.get("file_uploader")).lower()
    assert "pdf" in uploader_labels
    assert "zotero" in uploader_labels or "csv" in uploader_labels


def test_diagnose_tab_states_what_it_sends_where():
    """Diagnose is the only tab that talks to OpenAlex; the page must say so,
    since the privacy contract is that nothing else leaves the machine."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    page = " ".join(
        [m.value for m in app.markdown]
        + [c.value for c in app.caption]
        + [t.label for t in app.get("button")]
    )
    assert "OpenAlex" in page


def test_reports_tab_handles_an_empty_output_directory(tmp_path, monkeypatch):
    """A fresh install has no reports yet. The Reports tab must say so rather
    than raising on an empty/missing directory."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception, f"empty output dir raised: {app.exception}"


# --- public demo mode (CDA_DEMO_MODE) -----------------------------------


def test_demo_mode_withholds_the_local_workflow_tabs(monkeypatch):
    """A public URL must not offer ingestion, live analysis, validation,
    reports, or OpenAlex diagnostics: those read a private corpus, write to
    the host's filesystem, and spend a key. They ship in the installable
    release instead."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CDA_DEMO_MODE", "1")
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception, f"demo mode raised: {app.exception}"
    assert [t.label for t in app.tabs] == []
    # No path inputs (host filesystem), no API key field (no paid calls here).
    labels = " ".join(i.label for i in app.text_input)
    assert "directory" not in labels.lower()
    assert "API key" not in labels
    # ...and the page tells visitors how to get the full thing.
    page = " ".join(m.value for m in app.markdown)
    assert "git clone" in page
    assert "Lock the app down" not in page


def test_demo_mode_keeps_exactly_the_two_free_paths(monkeypatch):
    """The toy-corpus demo and a dry-run preview of pasted text — nothing
    that sends anything anywhere."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CDA_DEMO_MODE", "1")
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    button_labels = [b.label for b in app.button]
    assert any("Try the demo" in label for label in button_labels)
    assert any("no model API" in label for label in button_labels)
    # The live, paid path is absent entirely (not merely disabled).
    assert not any("Analyze live" in label for label in button_labels)


def test_normal_mode_still_offers_every_tab(monkeypatch):
    """The lockdown must be opt-in: a local install is unaffected."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("CDA_DEMO_MODE", raising=False)
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    assert [t.label for t in app.tabs][:6] == [
        "Setup",
        "Ingest",
        "Analyze",
        "Validate",
        "Reports",
        "Diagnose",
    ]


def test_pasted_text_is_scoped_to_the_session_not_a_shared_path():
    """Regression: the own-text mode used to write every visitor's paragraphs
    to one fixed path under demo/_runtime, so concurrent users on a hosted
    instance would overwrite each other — and one person's unpublished text
    could be analyzed into another's report."""
    source = (Path(__file__).resolve().parents[1] / "gui" / "app.py").read_text(encoding="utf-8")
    panel = source.split("def render_own_text_panel")[1].split("\ndef ")[0]
    assert "session_runtime_dir()" in panel
    assert '_runtime" / "your_text_drafts"' not in panel


def test_session_runtime_dir_is_uuid_scoped_and_session_backed():
    """Uniqueness comes from a uuid4 stored in session_state, so two browser
    sessions can never resolve to the same directory (and one session is
    stable across reruns)."""
    source = (Path(__file__).resolve().parents[1] / "gui" / "app.py").read_text(encoding="utf-8")
    fn = source.split("def session_runtime_dir()")[1].split("\ndef ")[0]
    assert "uuid.uuid4().hex" in fn  # unique per session
    assert "st.session_state" in fn  # stable within one session


def test_demo_mode_does_not_overclaim_local_confidentiality(monkeypatch):
    """The install pitch must not promise more than the design delivers.
    'Every file stays on your machine' / 'nothing is uploaded' read as a
    guarantee that nothing is transmitted, which is false for a live run:
    the sampled passages go to the model provider. The page states that and
    names the local-model configuration that closes it."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CDA_DEMO_MODE", "1")
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    page = " ".join([m.value for m in app.markdown] + [c.value for c in app.caption])
    assert "every file stays" not in page.lower()
    assert "nothing is uploaded" not in page.lower()
    # The disclosure and the escape hatch are both present.
    assert "term-sentences" in page
    assert "locally served" in page


# --- session hygiene for a public deployment --------------------------------


def test_sweep_removes_only_stale_session_directories(tmp_path):
    """Browsers never announce that a session ended, so old directories are
    swept on the next visit. A live session must survive that sweep."""
    import os
    import time

    from gui.app import SESSION_TTL_HOURS, _sweep_stale_sessions

    root = tmp_path / "cda_gui_sessions"
    fresh, stale = root / "fresh", root / "stale"
    for directory in (fresh, stale):
        directory.mkdir(parents=True)
        (directory / "your_text.md").write_text("a visitor's unpublished paragraph")
    old = time.time() - (SESSION_TTL_HOURS + 1) * 3600
    os.utime(stale, (old, old))

    assert _sweep_stale_sessions(root) == 1
    assert fresh.is_dir() and not stale.exists()


def test_sweep_is_silent_when_there_is_nothing_to_sweep(tmp_path):
    from gui.app import _sweep_stale_sessions

    assert _sweep_stale_sessions(tmp_path / "never_created") == 0


def test_a_live_session_older_than_the_ttl_survives_because_access_touches_it(tmp_path):
    """The sweep judges liveness by the root's mtime, which file writes inside
    subdirectories never update — so a session must be touched on access, or a
    tab left open past the TTL is deleted out from under its owner."""
    import os
    import time

    from gui.app import SESSION_TTL_HOURS, _claim_session_dir, _sweep_stale_sessions

    root = tmp_path / "cda_gui_sessions"
    session = root / "longlived"
    session.mkdir(parents=True)
    (session / "drafts").mkdir()
    old = time.time() - (SESSION_TTL_HOURS + 1) * 3600
    os.utime(session, (old, old))

    # What session_runtime_dir now does on every access for the session owner:
    _claim_session_dir(session)
    os.utime(session)

    assert _sweep_stale_sessions(root) == 0
    assert session.is_dir()


def test_reclaiming_a_session_dir_restores_private_permissions(tmp_path):
    """Every recreation path goes through _claim_session_dir, so a session
    swept out from under its owner never comes back world-readable."""
    import os
    import stat
    import sys

    from gui.app import _claim_session_dir

    root = tmp_path / "cda_gui_sessions" / "recreated"
    _claim_session_dir(root)  # first creation
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(root).st_mode) == 0o700

    # Simulate the sweep deleting it, then a later access re-claiming it.
    os.chmod(root, 0o755)
    _claim_session_dir(root)
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(root).st_mode) == 0o700


def test_pasted_text_is_capped():
    """The box writes to the host's disk on every paste, so it is bounded."""
    from gui.app import PASTE_MAX_CHARS

    assert 0 < PASTE_MAX_CHARS <= 50_000


def test_paste_cap_is_enforced_on_the_server_not_only_in_the_browser(monkeypatch):
    """An over-cap paste never reaches the host's disk, whichever layer stops it.

    Two layers exist. The pinned streamlit (1.58) truncates text_area values
    to max_chars in its server-side deserializer (TextAreaSerde), so even a
    client speaking the websocket protocol directly cannot deliver more than
    PASTE_MAX_CHARS characters to the app. Other streamlit versions (a dev
    environment without the [gui] pin) deliver the full paste — and then the
    in-app length check must refuse it before anything is written. The test
    asserts whichever layer this environment exercises, so it does not
    hard-code one version's deserializer behavior."""
    from streamlit.testing.v1 import AppTest

    from gui.app import PASTE_MAX_CHARS

    monkeypatch.setenv("CDA_DEMO_MODE", "1")
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    boxes = [t for t in app.text_area if "paragraph" in (t.label or "").lower()]
    assert boxes, "own-text paste box not found"
    boxes[0].set_value("x" * (PASTE_MAX_CHARS + 1))
    app.run()
    boxes = [t for t in app.text_area if "paragraph" in (t.label or "").lower()]
    assert boxes, "own-text paste box not found after rerun"
    delivered = boxes[0].value or ""
    if len(delivered) <= PASTE_MAX_CHARS:
        # Layer 1 (pinned streamlit): the deserializer already cut the paste.
        assert len(delivered) == PASTE_MAX_CHARS
        return
    # Layer 2: this streamlit delivered the over-cap paste, so pressing the
    # demo-mode preview button must hit the in-app refusal (before the text
    # is written to the session directory), not run the analysis.
    buttons = [b for b in app.button if "Preview the prompt" in (b.label or "")]
    assert buttons, "own-text preview button not found"
    buttons[0].click()
    app.run()
    errors = " ".join(e.value for e in app.error)
    assert f"up to {PASTE_MAX_CHARS:,}" in errors, (
        "the over-cap paste was neither truncated by streamlit nor refused by "
        "the app — nothing bounded what gets written to this host's disk"
    )


def test_demo_retention_notice_does_not_overclaim():
    """The notice used to say the text was 'not stored beyond this session'
    while it stayed in /tmp indefinitely with no cleanup path."""
    source = Path(APP_PATH).read_text(encoding="utf-8")
    assert 'not stored\n                "beyond this session' not in source
    assert "not stored beyond this session" not in source
    # The sweep runs when the *next* visitor arrives, so the copy must not
    # promise a deletion that happens on a timer.
    assert "deleted within" not in source
    assert "server for an in-memory preview" in source


def test_fixture_miss_is_explained_not_dumped():
    """A visitor cannot act on a raw fingerprint error, and the cause — the
    fixtures were recorded on a different CPU architecture — is not their
    fault or their doing."""
    from cda.llm.base import FixtureMissError, LLMError
    from gui.app import _demo_failure_message

    message = _demo_failure_message(
        FixtureMissError("Mock fixture not found for this prompt.\n  fingerprint: abc123"),
        "consensus",
    )
    assert "consensus" in message
    assert "CPU" in message or "architecture" in message
    assert "abc123" in message  # the detail is kept, just not led with

    # Unrelated failures are passed through untouched — including an LLMError
    # that merely mentions fixtures, since detection is by type, not wording.
    assert _demo_failure_message(RuntimeError("disk on fire"), "novelty") == "disk on fire"
    other = LLMError("the fixture directory is unreadable")
    assert _demo_failure_message(other, "novelty") == str(other)


def test_two_persisted_results_with_figures_render_together(tmp_path):
    """Run the demo, then run an analysis: both results stay on screen.

    Every result view (demo, own-text, Analyze) persists in session state and
    renders on every script pass, so their widget keys must differ per result
    slot. The passage-map download button used one fixed key, and the second
    result with a figure stopped the whole page with
    StreamlitDuplicateElementKey — on every rerun, since both results survive
    in session state."""
    from PIL import Image
    from streamlit.testing.v1 import AppTest

    def result(name: str) -> dict:
        report = tmp_path / f"drift_{name}.md"
        report.write_text(f"# {name} report", encoding="utf-8")
        figure = tmp_path / f"drift_{name}.png"
        Image.new("RGB", (2, 2), "white").save(figure)
        return {
            "status": "ok",
            "report_file": str(report),
            "figure_file": str(figure),
            "researcher_cluster": {"k": 2, "silhouette": 0.1, "n": 10},
            "literature_cluster": {"k": 3, "silhouette": 0.1, "n": 50},
            "joint": None,
            "stability": None,
            "grounding": {},
            "passages": {},
        }

    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["demo_result"] = result("demo")
    app.session_state["analyze_result"] = result("analyze")
    app.run()
    assert not app.exception, f"GUI raised: {app.exception[0].message}"


def test_failed_rediagnosis_clears_the_previous_report(tmp_path, monkeypatch):
    """A Diagnose run that fails must not leave the previous report on screen.

    Persisting the result only on success left the earlier report rendered
    directly under the "Failed" error box, with nothing saying it described a
    different (older) run — while the demo, own-text, and Analyze panels all
    replace their result unconditionally."""
    from streamlit.testing.v1 import AppTest

    stale = tmp_path / "diagnose_20260101.md"
    stale.write_text("# STALE-DIAGNOSE-REPORT", encoding="utf-8")

    def boom(*args, **kwargs):
        raise RuntimeError("OpenAlex is unreachable from this network")

    monkeypatch.setattr("cda.diagnose.report.run_diagnose", boom)

    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["diagnose_result"] = {"report_file": str(stale)}
    app.run()
    assert not app.exception, f"GUI raised: {app.exception}"
    assert "# STALE-DIAGNOSE-REPORT" in [m.value for m in app.markdown]

    diagnose_button = [b for b in app.button if b.label == "Run diagnosis"][0]
    diagnose_button.click()
    app.run()

    assert not app.exception, f"failed diagnosis raised: {app.exception}"
    assert app.session_state["diagnose_result"] is None
    assert "# STALE-DIAGNOSE-REPORT" not in [m.value for m in app.markdown]
    assert any("unreachable" in e.value for e in app.error)


def test_analyze_tab_refuses_an_unplayable_mock_run_before_the_heavy_stages(tmp_path, monkeypatch):
    """Regression: mock toggle ON + no recorded fixtures used to fail only
    AFTER embedding and clustering (minutes on a real corpus), with a
    CLI-worded FixtureMissError. The tab must refuse up front, in GUI terms,
    and without touching the embedding model."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)  # ./outputs/fixtures does not exist here
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # mock toggle defaults ON

    class Boom:
        def __init__(self, *args, **kwargs):
            raise AssertionError("the embedding model must not load for a refused run")

    # If the precheck were missing, the run would reach the embedder first;
    # this turns that regression into a different, failing error message.
    monkeypatch.setattr("cda.corpus.embeddings.SentenceTransformerEmbedder", Boom)

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    term_box = next(t for t in app.get("text_input") if t.label == "Target term")
    term_box.set_value("novelty")
    app.run()
    run_button = next(b for b in app.button if b.label == "Run analysis")
    run_button.click()
    app.run()
    assert not app.exception, f"GUI raised: {app.exception}"
    errors = " ".join(e.value for e in app.error)
    assert "mock toggle" in errors
    # Both ways out are named: one live run from this tab, or the CLI.
    assert "records replayable responses" in errors
    assert "cda analyze --record-fixtures" in errors


def test_gui_own_corpus_mock_validation_explains_missing_recordings(tmp_path, monkeypatch):
    """Regression: the empty-fixtures warning fired only for the DEMO
    checkbox; an own-corpus mock validation started anyway and died after the
    heavy stages. The tab must warn with own-corpus wording (a live run
    records the readings) and refuse the click before any work."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)  # ./outputs/fixtures/validation is empty here
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # mock toggle defaults ON
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception, f"GUI raised: {app.exception}"
    warnings = " ".join(str(w.value) for w in app.warning)
    assert "No recorded readings to replay for this corpus" in warnings
    assert "live validation run records" in warnings

    # And the start button refuses instead of spending the clustering time.
    term_box = next(t for t in app.get("text_input") if getattr(t, "key", None) == "validate_term")
    term_box.set_value("novelty")
    app.run()
    start = next(b for b in app.button if (b.label or "").startswith("Start validation"))
    start.click()
    app.run()
    assert not app.exception, f"GUI raised: {app.exception}"
    errors = " ".join(e.value for e in app.error)
    assert "Nothing to replay yet" in errors
    assert "mock toggle" in errors


def test_an_empty_sidebar_path_field_is_refused_before_any_tab_renders(monkeypatch):
    """Regression: a blanked Output directory reached config.path() during
    a render pass (a ConfigError blanking the whole page) and sent the
    own-text panel's paid report to the process cwd via Path("").expanduser().
    One guard now names the field and stops before any tab renders."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert len(app.tabs) >= 6  # sanity: tabs exist before the field is blanked
    output_input = next(t for t in app.get("text_input") if t.label == "Output directory")
    output_input.set_value("   ")
    app.run()
    assert not app.exception, f"GUI raised: {app.exception}"
    errors = " ".join(e.value for e in app.error)
    assert "Output directory" in errors  # the error names the field
    assert len(app.tabs) == 0  # st.stop() fired before the workflow tabs


def test_live_demo_report_lands_in_the_reports_tab_directory(tmp_path, monkeypatch):
    """Regression: the paid live demo wrote its report to
    demo/_runtime/outputs, which the Reports tab never lists — violating the
    'paid must stay findable' rule the own-text panel states. A live demo run
    must target the sidebar's output directory and record its fixtures."""
    from streamlit.testing.v1 import AppTest

    import cda.demo

    calls = {}

    def fake_run_demo(**kwargs):
        calls.update(kwargs)
        return {"status": "skipped", "reason": "captured by the test"}

    monkeypatch.setattr(cda.demo, "run_demo", fake_run_demo)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")  # mock defaults OFF
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    output_input = next(t for t in app.get("text_input") if t.label == "Output directory")
    output_input.set_value(str(tmp_path / "myreports"))
    app.run()
    demo_button = next(b for b in app.button if "Try the demo" in (b.label or ""))
    assert "live API call" in demo_button.label  # key set + mock off = paid
    demo_button.click()
    app.run()
    assert not app.exception, f"GUI raised: {app.exception}"
    assert calls["mock"] is False
    assert calls["output_dir"] == tmp_path / "myreports"  # where Reports looks
    assert calls["record_fixtures"] is True  # one paid run enables replay


def test_free_demo_replay_stays_in_the_runtime_directory(monkeypatch):
    """The free replay counterpart of the live-demo routing: nothing was
    paid, so the report stays in demo/_runtime (out of the way of real
    reports) and nothing is recorded."""
    from streamlit.testing.v1 import AppTest

    import cda.demo

    calls = {}

    def fake_run_demo(**kwargs):
        calls.update(kwargs)
        return {"status": "skipped", "reason": "captured by the test"}

    monkeypatch.setattr(cda.demo, "run_demo", fake_run_demo)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # mock toggle defaults ON
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    demo_button = next(b for b in app.button if "Try the demo" in (b.label or ""))
    assert "free replay" in demo_button.label
    demo_button.click()
    app.run()
    assert not app.exception, f"GUI raised: {app.exception}"
    assert calls["mock"] is True
    assert calls["output_dir"] is None
    assert calls["record_fixtures"] is False


# --- session locking: two tabs must not lose each other's writes -----


def _single_cluster_session(output_dir):
    """A minimal one-cluster session a commit click can act on."""
    from datetime import datetime

    from cda.validation.session import create_session

    return create_session(
        "novelty",
        params={"seed": 42, "passages_shown_per_cluster": 8, "readers": ["m1"]},
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
        passages={key: {"text": f"passage {key}", "side": "literature"} for key in "ab"},
        readings=[],
        output_dir=output_dir,
        now=datetime(2026, 8, 1, 9, 0, 0),
    )


def test_verdict_commit_records_under_the_lock_and_releases_it(tmp_path):
    """M7: the commit handler's load→record→save cycle runs under
    session_lock. A normal commit must write the verdict and leave no
    lockfile behind for the next writer to trip on."""
    from streamlit.testing.v1 import AppTest

    from cda.validation.session import load_session

    session_dir, _session = _single_cluster_session(tmp_path / "outputs")
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["validation_session_dir"] = str(session_dir)
    app.run()
    commit = next(b for b in app.button if b.label == "Commit verdict")
    commit.click()
    app.run()
    assert not app.exception, f"GUI raised: {app.exception}"
    session = load_session(session_dir)
    assert [v["verdict"] for v in session["verdicts"]] == ["coherent"]  # radio default
    assert not (session_dir / "session.lock").exists()


def test_a_held_session_lock_surfaces_its_own_error_and_drops_no_verdict(tmp_path, monkeypatch):
    """M7: when another tab (or a crashed process) holds session.lock past
    the timeout, the commit must show the lock's message — it names the
    lockfile and says when deleting it is safe — rather than a traceback,
    and must not write anything."""
    from streamlit.testing.v1 import AppTest

    from cda.validation.session import load_session

    monkeypatch.setattr("cda.validation.session._LOCK_TIMEOUT_SECONDS", 0.2)
    session_dir, _session = _single_cluster_session(tmp_path / "outputs")
    (session_dir / "session.lock").write_text("pid 99999\n", encoding="utf-8")

    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["validation_session_dir"] = str(session_dir)
    app.run()
    commit = next(b for b in app.button if b.label == "Commit verdict")
    commit.click()
    app.run()
    assert not app.exception, f"GUI raised: {app.exception}"
    errors = " ".join(e.value for e in app.error)
    assert "session.lock" in errors  # the lock's own message, verbatim
    assert load_session(session_dir)["verdicts"] == []  # nothing was written


def test_emergence_labels_a_retained_result_after_a_failed_run():
    """The emergence panel deliberately keeps its earlier result when a run
    fails (the clustering pass is slow), so it must say that the result on
    screen is not the run the user just asked for."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["emergence_result"] = {
        "status": "no-dates",
        "report_file": "/nonexistent/emergence_novelty.md",
        "emergence": {"term": "novelty", "reason": "no acquisition dates"},
    }
    app.session_state["emergence_result_stale"] = True
    app.run()
    assert not app.exception, f"GUI raised: {app.exception}"
    warnings = " ".join(w.value for w in app.warning)
    assert "the last run that did" in warnings
