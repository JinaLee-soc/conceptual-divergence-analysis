"""CDA Streamlit GUI.

A thin visual layer over the same functions the CLI uses; no analysis logic
lives here. Tabs follow the workflow: Setup → Ingest → Analyze → Reports →
Diagnose, plus a one-click demo on the committed toy corpus.

Run locally:   streamlit run gui/app.py        (pip install -e ".[gui]")
Run in Docker: docker compose up → http://localhost:8501
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import streamlit as st

from cda import __version__
from cda.config import Config, load_config
from cda.corpus.ingest import CorpusLockError, get_collection, ingest_pdf_dir
from cda.registers.prompt_loader import load_system_prompt

# The GUI is a plain script directory, not a package; make sibling modules
# importable regardless of how Streamlit was launched.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import visuals  # noqa: E402
from shared import (  # noqa: E402
    demo_mode_enabled,
    embedder_or_error,
    render_report_file,
    run_with_live_log,
    session_api_key,
    session_mailto_in_env,
    usage_statistics_note,
)
from validate import render_validate_tab  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="CDA: Conceptual Divergence Analysis", layout="wide")
st.html(f"<style>{Path(__file__).with_name('styles.css').read_text(encoding='utf-8')}</style>")


# --- public demo mode -------------------------------------------------------

DEMO_MODE = demo_mode_enabled()
# Public mode exposes only toy replay and an in-memory prompt preview.
# Corpus access, credentials, live calls, and local workflow tabs stay hidden.


SESSION_TTL_HOURS = 6  # a visitor's pasted text does not outlive their visit by long
PASTE_MAX_CHARS = 20_000  # ~3,000 words: several paragraphs, not a manuscript


def _sweep_stale_sessions(sessions_root: Path, ttl_hours: int = SESSION_TTL_HOURS) -> int:
    """Delete session directories older than ``ttl_hours``. Returns the count.

    Browsers do not announce that a session ended, so there is no moment at
    which one visitor's directory can be removed on the spot. Sweeping on each
    new session keeps a public deployment from accumulating strangers' pasted
    text indefinitely, and makes the retention notice in the UI something the
    code actually does rather than something it asserts. Failures are ignored:
    a directory another process is mid-write on will be caught next sweep, and
    losing a cleanup pass must never take down the app.
    """
    import shutil
    import time

    if not sessions_root.is_dir():
        return 0
    cutoff = time.time() - ttl_hours * 3600
    removed = 0
    for directory in sessions_root.iterdir():
        try:
            if directory.is_dir() and directory.stat().st_mtime < cutoff:
                shutil.rmtree(directory, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def _claim_session_dir(root: Path) -> None:
    """Create (or re-create) a session directory with private permissions.

    One helper for both the first creation and any later re-creation, so no
    path can recreate the directory with the default world-readable mode: a
    visitor's unpublished paragraphs land here, and on a shared host 0700 is
    what keeps other local accounts out of them.
    """
    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:  # Windows has no POSIX mode bits
        pass


def session_runtime_dir() -> Path:
    """A directory private to this browser session.

    The paste-your-own-text mode writes the visitor's paragraphs to disk for
    the pipeline to read. A single shared path is fine for the single-user
    local app this GUI was written as, but on a multi-visitor deployment two
    people pasting at once would overwrite each other — and one visitor's
    unpublished text could be analyzed into another's report. Scoping the
    drafts and outputs to a per-session directory removes that entirely.
    """
    import tempfile
    import uuid

    if "session_runtime_dir" not in st.session_state:
        sessions = Path(tempfile.gettempdir()) / "cda_gui_sessions"
        _sweep_stale_sessions(sessions)
        root = sessions / uuid.uuid4().hex
        _claim_session_dir(root)
        st.session_state["session_runtime_dir"] = str(root)
    root = Path(st.session_state["session_runtime_dir"])
    # Re-claimed and touched on every pass: the sweep judges liveness by this
    # directory's own mtime, and a re-created directory must stay 0700.
    _claim_session_dir(root)
    try:
        os.utime(root)
    except OSError:
        pass
    return root


# --- shared helpers ---------------------------------------------------------


def build_config(corpus_dir: str, drafts_dir: str, output_dir: str):
    return load_config(
        overrides={
            "paths": {
                "corpus_dir": corpus_dir,
                "drafts_dir": drafts_dir,
                "output_dir": output_dir,
            }
        }
    )


def api_key_status() -> tuple[bool, str]:
    session_key = session_api_key()
    if session_key:
        return True, (
            f"entered for this session ({session_key[:7]}…, {len(session_key)} "
            "chars); kept in memory only, never written to disk"
        )
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return True, f"set from the environment ({key[:7]}…, {len(key)} chars)"
    return False, (
        "not set: enter a key in the sidebar for live runs; mock/demo mode works without one"
    )


def corpus_status(config) -> str:
    try:
        collection = get_collection(config)
        return f"ready: {collection.count()} chunks in '{config.get('corpus.collection')}'"
    except FileNotFoundError:
        return "no corpus yet; use the Ingest tab or the demo button"
    except Exception as exc:
        # A corrupted or version-incompatible Chroma store raises its own
        # error types; this line renders on every pass of every tab, so an
        # unguarded raise here takes down the whole page.
        return f"corpus store could not be opened ({type(exc).__name__}: {exc})"


# --- sidebar: paths and mode -------------------------------------------------

st.sidebar.title("CDA")
st.sidebar.caption(f"v{__version__}")


def _forget_session_key() -> None:
    st.session_state.pop("anthropic_session_key", None)


if DEMO_MODE:
    # No paths (they would expose the host's filesystem), no key field (this
    # instance never makes a paid call), no mock toggle (replay is the only
    # option), no OpenAlex address (Diagnose is withheld). The defaults below
    # are inert: every path a demo run uses is set per session instead.
    corpus_dir, drafts_dir, output_dir = "./corpus", "./drafts", "./outputs"
    openalex_mailto = ""
    mock_llm = True
    key_ok, key_message = False, "not used on this public demo"
    st.sidebar.info(
        "**Public demo.** Replay recorded responses for the toy corpus or "
        "preview a prompt with your own text. Pasted text is processed in memory "
        "on this server and is not sent to a model API. No API key or payment is required.",
    )
    st.sidebar.markdown(
        "To analyze **your own** corpus and drafts — ingest, live analysis, "
        "validation, reports, and corpus diagnostics — install CDA locally:\n\n"
        "```\ngit clone https://github.com/JinaLee-soc/conceptual-divergence-analysis\ncd conceptual-divergence-analysis\n"
        'pip install -e ".[gui]"\nstreamlit run gui/app.py\n```\n\n'
        "Your corpus is then indexed and analyzed on your own machine; a live "
        "run sends only the sampled passages to the model provider, and a "
        "locally served model keeps even those in-house."
    )
else:
    corpus_dir = st.sidebar.text_input("Corpus directory", value="./corpus")
    drafts_dir = st.sidebar.text_input("Drafts directory", value="./drafts")
    output_dir = st.sidebar.text_input("Output directory", value="./outputs")

    st.sidebar.text_input(
        "Anthropic API key (this session only)",
        type="password",
        key="anthropic_session_key",
        help=(
            "Used for live runs. Kept only in this browser session's memory: "
            "never written to disk, exposed to the API client only while a run "
            "is executing, and gone when you close the tab or press the button "
            "below. (Alternative for CLI use: the ANTHROPIC_API_KEY environment "
            "variable.)"
        ),
    )
    if session_api_key():
        st.sidebar.button("Forget the key now", on_click=_forget_session_key)

    key_ok, key_message = api_key_status()
    mock_llm = st.sidebar.toggle(
        "Mock LLM (replay fixtures, no API cost)",
        value=not key_ok,
        help=(
            "Replays recorded responses instead of calling the Anthropic API. "
            "Works for the demo out of the box; your own corpora need a live run "
            "first — live runs from the Analyze tab record automatically, as "
            "does `cda analyze --record-fixtures`."
        ),
    )
    openalex_mailto = st.sidebar.text_input(
        "OpenAlex email (for Diagnose)",
        value="",
        placeholder="you@example.org",
        help=(
            "OpenAlex asks for an email so it can offer its faster 'polite pool'. "
            "Enter your own: it is used only for Diagnose and corpus-build requests, "
            "kept in this session's environment, and never saved or committed."
        ),
    )
# Never pre-filled from OPENALEX_MAILTO: an address from the environment is
# never rendered on screen (the environment value still serves as fallback).
if openalex_mailto.strip():
    # Kept in session state, not os.environ: the environment is process-global,
    # so writing it there would hand one visitor's address to every other
    # visitor's OpenAlex requests on a shared host, and it would outlive the
    # session that entered it. session_mailto_in_env() scopes it to one call.
    st.session_state["openalex_session_mailto"] = openalex_mailto.strip()
else:
    # Clearing the field must forget the address, like the API key's "forget"
    # path: otherwise the status line would say "not set" while the previous
    # address is still attached to every Diagnose request.
    st.session_state.pop("openalex_session_mailto", None)

# A blank path field stops the app before any tab renders: Path("") would
# silently resolve to the process cwd (the CLI's _reject_empty_path_flags).
_empty_path_fields = [
    label
    for label, value in (
        ("Corpus directory", corpus_dir),
        ("Drafts directory", drafts_dir),
        ("Output directory", output_dir),
    )
    if not str(value).strip()
]
if _empty_path_fields:
    st.error(
        f"Empty sidebar path field: {', '.join(_empty_path_fields)}. "
        "Enter a path in the sidebar (the defaults are ./corpus, ./drafts, "
        "./outputs) and the app will continue."
    )
    st.stop()

config = build_config(corpus_dir, drafts_dir, output_dir)

# --- panels (one function per screen; the main flow at the bottom calls them
#     in order; each is small enough to read on its own) ----------------------


# The public wording for a corpus-lock failure. It must stay generic: the
# error's own text names the lock file on the host, and the condition is not
# only "another run holds the corpus" (see _demo_failure_message).
DEMO_CORPUS_UNAVAILABLE = "The demo cannot reach its corpus right now; please retry."


def _demo_failure_message(exc: Exception, term: str) -> str:
    """Explain a demo failure in terms a visitor can act on.

    The common failure off the maintainer's machine is a fixture miss
    (recognized by its type, not its wording, so a reworded message cannot
    silently turn this explanation back into an error dump). A fixture is
    keyed by a fingerprint over the model name and the prompt, and the prompt
    embeds the clustering, which can differ between CPU architectures and
    library builds (see demo/fixtures/README.md). On the public demo the
    technical detail is withheld: it names paths on the host's filesystem.
    """
    from cda.llm.base import FixtureMissError

    if isinstance(exc, CorpusLockError):
        # A lock error is transient and the visitor can act on it (retry),
        # so it is explained in both modes; locally with its text, which
        # names the lock file. Publicly the wording stays generic: the error
        # is raised for more than another run holding the corpus (a lock
        # that could not be reclaimed, a store directory that cannot be
        # written), and none of those causes is the visitor's to diagnose.
        if DEMO_MODE:
            return DEMO_CORPUS_UNAVAILABLE
        return str(exc)
    if not isinstance(exc, FixtureMissError):
        if DEMO_MODE:
            return f"The demo failed on this server ({type(exc).__name__})."
        return str(exc)
    cause = (
        f"The recorded response for **{term}** could not be replayed here.\n\n"
        "Demo fixtures are keyed to the exact analysis prompt, and the "
        "clustering that prompt describes can differ between CPU "
        "architectures, so a fixture made on one machine may not replay on "
        "another. Nothing is "
        "wrong with your input.\n\n"
    )
    if DEMO_MODE:
        return cause + "Try the other term, or install CDA and run the demo locally."
    return (
        cause + "Try the other term. With an API key, switch the mock toggle off and "
        "run the demo once live (~$0.10–0.15): the response is recorded under "
        "your configured model, and later free replays on this machine use that "
        "recording before the committed one.\n\n"
        f"Technical detail:\n\n```\n{exc}\n```"
    )


def _own_text_failure_message(exc: Exception) -> str:
    """The error for a failed own-text run.

    The public demo never prints exception text: it can carry absolute paths
    on the host. A lock wait that ran out is the one failure a visitor can
    act on, so it is named (without the path).
    """
    if isinstance(exc, CorpusLockError):
        return DEMO_CORPUS_UNAVAILABLE if DEMO_MODE else str(exc)
    return f"The preview failed ({type(exc).__name__})." if DEMO_MODE else str(exc)


def _passages_found(result: dict) -> str:
    """`` (passages found: N)`` when the skipped result counted them, else empty."""
    found = (result.get("stats") or {}).get("researcher_passages")
    return f" (passages found: {found})" if found is not None else ""


def _own_text_skip_message(result: dict, term: str) -> str:
    """The notice for a paste the pipeline skipped.

    The pipeline's own reason is written for a researcher with a corpus: it
    names the host's drafts directory and suggests Zotero tags. On the public
    demo the only researcher input is the pasted text, so the actionable
    cause is that the paste does not use the term often enough.
    """
    if not DEMO_MODE:
        return result.get("reason", str(result))
    return (
        f"Your pasted text does not contain the term *{term}* often enough to "
        f"analyze{_passages_found(result)}. Paste paragraphs that use *{term}*, or "
        "pick the other term."
    )


def _demo_skip_message(result: dict, term: str) -> str:
    """The notice for a toy-corpus demo run that produced no analysis.

    Same rule as ``_own_text_skip_message``: the pipeline's reason names the
    host's drafts directory, so the public demo shows a fixed sentence and
    only the local app shows the reason itself (there the path is the
    actionable detail).
    """
    if not DEMO_MODE:
        return result.get("reason", str(result))
    return (
        f"The demo found too few passages for *{term}* in the toy drafts to "
        f"analyze{_passages_found(result)}. Try the other term, or install CDA "
        "and run the demo locally."
    )


def _show_result_tabs(result: dict, key_prefix: str) -> None:
    """The two-tab result view (charts + full report) shared by demo/analyze.

    ``key_prefix`` names the CALL SITE, the same rule render_report_file
    states: the demo, own-text, and Analyze results all persist in session
    state and are all rendered on every script pass, so any fixed widget key
    below would collide the moment a second result exists — and Streamlit
    answers a duplicate key by stopping the whole page, on every rerun,
    until the session is cleared.
    """
    visual_tab, report_tab = st.tabs(["Visual summary", "Full report"])
    with visual_tab:
        visuals.render_result_visuals(st, result)
        _render_passage_map(result, key_prefix)
    with report_tab:
        render_report_file(Path(result["report_file"]), key_prefix=key_prefix)


def _render_passage_map(result: dict, key_prefix: str) -> None:
    """The print-quality passage map the run wrote next to its report.

    The charts above are for reading on screen; this is the one to put in a
    paper, so it is offered as a file rather than only rendered.
    """
    figure_file = result.get("figure_file")
    if not figure_file:
        return
    path = Path(figure_file)
    if not path.is_file():
        return
    st.markdown("#### Passage map (print quality)")
    st.image(str(path), width="stretch")
    st.caption(
        "Every clustered passage in one PCA plane: literature as dots, yours as "
        "stars, colour = shared cluster. A cluster where your stars sit alone is "
        "usage the literature has no counterpart for; one with dots but no stars "
        "is literature you have not taken up."
    )
    st.download_button(
        "Download the passage map (PNG)",
        data=path.read_bytes(),
        file_name=path.name,
        mime="image/png",
        # Per call site, not fixed: two results with figures on one page
        # (run the demo, then run an analysis) must not share a key.
        key=f"passage_map_{key_prefix}",
    )


def render_demo_button(mock_llm: bool, output_dir: str) -> None:
    """The one-click demo on the committed toy corpus."""
    # The label says when this button costs money: with a key configured and
    # the mock toggle off, "try the demo" is a live API call, and every other
    # paid button on this page states its price.
    live_demo = not DEMO_MODE and not mock_llm
    label = (
        "▶ Try the demo (live API call, ~$0.10–0.15)"
        if live_demo
        else "▶ Try the demo (toy corpus, free replay, no setup needed)"
    )
    demo_col, term_col = st.columns([1, 1])
    with demo_col:
        clicked = st.button(label, type="primary", width="stretch")
    with term_col:
        demo_term = st.selectbox(
            "Demo term", ["novelty", "consensus"], label_visibility="collapsed"
        )
    if clicked:
        from cda.demo import run_demo

        if DEMO_MODE:
            # Per-visitor directory on the hosted demo (see session_runtime_dir).
            demo_output_dir = session_runtime_dir() / "outputs"
        elif live_demo:
            # A live demo spends the researcher's money, and paid reports must
            # stay findable (the own-text panel states the same rule): write it
            # to the sidebar's output directory — the one the Reports tab
            # lists — not demo/_runtime, which Reports never shows.
            demo_output_dir = Path(output_dir).expanduser()
        else:
            # Free replay: keep the default demo/_runtime/outputs, out of the
            # way of the researcher's real reports.
            demo_output_dir = None
        # Streamed like the Analyze tab, so a wait for the corpus lock (the
        # toy store is shared by every visitor) and the ingest's own lines
        # are visible instead of a silent spinner.
        result = run_with_live_log(
            lambda progress: run_demo(
                term=demo_term,
                # Replay is forced on the public demo: this host has no
                # key to spend.
                mock=True if DEMO_MODE else mock_llm,
                # A live demo records its response under the
                # configured model (same rule as the Analyze and
                # Validate tabs); mock replay prefers that recording
                # and falls back to the committed claude-code-session
                # fixture, so one paid run makes replay work on this
                # machine even where the committed fixture does not.
                record_fixtures=live_demo,
                # A deliberate live rerun replaces this machine's
                # earlier recording, as on the Analyze tab; without
                # it the second paid click is refused.
                overwrite_fixtures=live_demo,
                output_dir=demo_output_dir,
                return_passages=True,
                progress=progress,
            ),
            status_label="Running the demo on the toy corpus…",
            failure_message=lambda exc: _demo_failure_message(exc, demo_term),
        )
        # Kept in session state so the result survives the rerun every later
        # widget interaction (the download button included) triggers — the
        # same rule render_emergence_expander states: a blank panel after a
        # completed run is not what we owe the user.
        st.session_state["demo_result"] = result

    result = st.session_state.get("demo_result")
    if result and result.get("status") == "ok":
        st.success("Demo complete. Results show the divergence built into the toy corpus.")
        _show_result_tabs(result, key_prefix="demo_result")
    elif result:
        st.warning(_demo_skip_message(result, result.get("term", demo_term)))


def render_own_text_panel(key_ok: bool) -> None:
    """Paste-your-own-text mode: your paragraphs vs the toy literature corpus."""
    with st.expander("…or try it on **your own text** (no corpus setup needed)"):
        st.write(
            "Paste a few paragraphs of your own writing that use *novelty* or "
            "*consensus*. CDA compares your usage with the toy literature corpus. "
            "You can preview the analysis prompt before building your own corpus."
        )
        own_text = st.text_area(
            "Your paragraphs",
            height=180,
            placeholder="Paste 1–3 paragraphs here…",
            # Without a cap this box accepts whatever a browser will send, and
            # every paste is written to the host's disk. The analysis reads a
            # few paragraphs; a manuscript-sized paste would not improve it.
            max_chars=PASTE_MAX_CHARS,
            help=f"Up to {PASTE_MAX_CHARS:,} characters (roughly 3,000 words).",
        )
        own_term = st.selectbox(
            "Term to analyze",
            ["novelty", "consensus"],
            key="own_text_term",
            help="The toy literature corpus is built around these two terms.",
        )
        run_own = None
        if DEMO_MODE:
            # Public demo: dry run only. There is no key to spend and no
            # reason for a visitor's text to reach any API from this host.
            if st.button("Preview the prompt (free, no model API)", disabled=not own_text.strip()):
                run_own = "dry"
            st.caption(
                "Your text is sent to this server for an in-memory preview. "
                "It is not written to a draft, prompt, report, or diagnostic file, "
                "and is not sent to any model API. It remains in your active browser "
                "session. For confidential drafts, install CDA and run it locally."
            )
        else:
            preview_col, live_col = st.columns(2)
            with preview_col:
                if st.button("Preview the prompt (free, offline)", disabled=not own_text.strip()):
                    run_own = "dry"
            with live_col:
                if st.button(
                    "Analyze live (~$0.10–0.15, uses your API key)",
                    disabled=not (own_text.strip() and key_ok),
                    help=None
                    if key_ok
                    else "Enter your API key in the sidebar to enable live analysis.",
                ):
                    run_own = "live"
            st.caption(
                "Your text never leaves this machine on preview. A live run sends it "
                "(inside the analysis prompt) to the Anthropic API, exactly what the "
                "preview shows. Recorded fixtures can't cover your text, so this mode "
                "doesn't use the mock toggle."
            )
        if run_own:
            # The cap bounds what is written to this host's disk, so it is
            # enforced here, not only by the widget's max_chars.
            if len(own_text) > PASTE_MAX_CHARS:
                st.error(
                    f"That paste is {len(own_text):,} characters; this box accepts "
                    f"up to {PASTE_MAX_CHARS:,} (roughly 3,000 words). The analysis "
                    "reads a few paragraphs — trim the text and try again."
                )
                return

            from cda.demo import run_demo

            # Per-session, so concurrent visitors can never read or overwrite
            # each other's pasted text (see session_runtime_dir).
            if DEMO_MODE:
                own_drafts_dir = own_output_dir = None
            else:
                session_root = session_runtime_dir()
                own_drafts_dir = session_root / "your_text_drafts"
                if own_drafts_dir.exists():
                    for old in own_drafts_dir.glob("*"):
                        if old.is_file():
                            old.unlink()
                own_drafts_dir.mkdir(parents=True, exist_ok=True)
                (own_drafts_dir / "your_text.md").write_text(own_text, encoding="utf-8")

                # A live run's report goes to the sidebar output directory — the
                # one the Reports tab lists — because it cost money and must stay
                # findable. Previews stay in the session directory: they exist to
                # be read once, and they contain the visitor's unpublished text.
                own_output_dir = (
                    Path(output_dir).expanduser() if run_own == "live" else session_root / "outputs"
                )

            # run_with_live_log exposes the sidebar key to the provider for
            # this one call and streams the run's lines (lock waits included).
            result = run_with_live_log(
                lambda progress: run_demo(
                    term=own_term,
                    mock=False,
                    dry_run=(run_own == "dry"),
                    drafts_dir=own_drafts_dir,
                    draft_texts={"your_text.md": own_text} if DEMO_MODE else None,
                    persist=not DEMO_MODE,
                    output_dir=own_output_dir,
                    return_passages=True,
                    progress=progress,
                ),
                status_label="Analyzing your text against the toy literature…",
                failure_message=_own_text_failure_message,
            )
            # Session state, not a local: the paid result must survive the
            # rerun that any later widget interaction (its own download button
            # included) triggers.
            st.session_state["own_text_result"] = result

        result = st.session_state.get("own_text_result")
        if result and result["status"] == "dry-run":
            st.info(
                "Dry run: nothing was sent. This is the exact prompt a live "
                "run would send; your paragraphs appear under RESEARCHER "
                "CLUSTER STRUCTURE."
            )
            with st.expander("User prompt (with your text)", expanded=True):
                if "user_prompt" in result:
                    st.code(result["user_prompt"], language="markdown")
                elif Path(result["prompt_file"]).is_file():
                    st.code(
                        Path(result["prompt_file"]).read_text(encoding="utf-8"), language="markdown"
                    )
                else:
                    st.info("This preview has expired; run the preview again to rebuild it.")
        elif result and result["status"] == "ok":
            _show_result_tabs(result, key_prefix="own_text_result")
        elif result:
            st.warning(_own_text_skip_message(result, result.get("term", own_term)))


def openalex_mailto_status() -> str:
    """Whether an OpenAlex address is available, without printing it.

    Mirrors ``api_key_status``: an address is personal data, so the Setup tab
    confirms one is set and where it came from rather than rendering it.
    """
    entered = str(openalex_mailto or "").strip()
    if entered:
        return "entered in the sidebar for this session (used only for Diagnose)"
    if os.environ.get("OPENALEX_MAILTO", "").strip():
        return "set from the environment; enter one in the sidebar to override it for this session"
    return "not set (enter yours in the sidebar for Diagnose)"


def render_setup_tab(config, key_message: str) -> None:
    st.subheader("Environment")
    st.write(f"**Anthropic API key:** {key_message}")
    st.write(f"**OpenAlex email:** {openalex_mailto_status()}")
    st.write(f"**Embedding model:** `{config.get('embedding.model')}` (runs locally)")
    st.write(f"**LLM model:** `{config.get('llm.model')}`")
    st.write(f"**Corpus:** {corpus_status(config)}")
    st.caption(
        "Set keys in a `.env` file next to this app (copy `.env.example`). "
        "Keys are read from the environment only and never stored elsewhere."
    )


def render_ingest_tab(config, corpus_dir: str, drafts_dir: str) -> None:
    st.subheader("Build your corpus")
    st.write(
        "Drag PDFs (and optionally a Zotero CSV export: File → Export "
        "Library… → Format: CSV) into the boxes below, or point CDA at a "
        "folder on disk. Zotero metadata gives passages title/year/DOI and "
        "recognizes your `mywork`-tagged items as your own papers."
    )

    st.markdown("**Drag and drop**")
    dropped_pdfs = st.file_uploader(
        "Drop PDFs here", type=["pdf"], accept_multiple_files=True, key="pdf_drop"
    )
    dropped_csv = st.file_uploader(
        "Drop a Zotero CSV export here (optional)", type=["csv"], key="csv_drop"
    )
    st.caption(
        "Dropped files never leave this machine; the browser hands them to "
        "the local CDA process, which saves them under your corpus directory."
    )
    if st.button("Ingest dropped files", disabled=not dropped_pdfs):
        uploads_dir = Path(corpus_dir).expanduser() / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        for uploaded in dropped_pdfs:
            (uploads_dir / Path(uploaded.name).name).write_bytes(uploaded.getvalue())
        dropped_csv_path = None
        if dropped_csv is not None:
            dropped_csv_path = uploads_dir / "zotero_export_dropped.csv"
            dropped_csv_path.write_bytes(dropped_csv.getvalue())
        st.caption(f"Saved {len(dropped_pdfs)} PDF(s) to {uploads_dir}")
        embedder = embedder_or_error(config)
        if embedder is None:
            return
        run_with_live_log(
            lambda progress: ingest_pdf_dir(
                config, embedder, uploads_dir, zotero_csv=dropped_csv_path, progress=progress
            )
        )

    st.markdown("**Or point at a folder on disk** (better for large libraries)")
    pdf_dir = st.text_input("PDF directory (searched recursively)")
    zotero_csv = st.text_input("Zotero CSV export (optional path)")
    reingest = st.checkbox("Re-ingest files already in the corpus", value=False)
    if st.button("Ingest", disabled=not pdf_dir):
        embedder = embedder_or_error(config)
        if embedder is None:
            return
        run_with_live_log(
            lambda progress: ingest_pdf_dir(
                config,
                embedder,
                pdf_dir,
                zotero_csv=zotero_csv or None,
                reingest=reingest,
                progress=progress,
            )
        )
    st.caption(f"Drafts are read at analysis time from: {drafts_dir}")


def _analysis_options(config):
    """Clustering-mode and corpus-timeframe controls; returns the run's config.

    Both live in an expander that starts closed: the defaults are the right
    answer for a first run, and the controls only matter once you want to
    compare against the paper's pipeline or pin the corpus to a date.
    """
    overrides: dict = {}
    with st.expander("Analysis options (clustering mode, corpus timeframe)"):
        joint = st.toggle(
            "Cluster your passages together with the literature (joint clustering)",
            value=str(config.get("clustering.mode", default="joint")) == "joint",
            help="On (default): one k-means partition over both corpora, so the "
            "report shows where your passages sit inside each shared cluster. "
            "Off: each corpus is clustered separately and your passages are "
            "located by cross-assignment — the pipeline described in the paper.",
        )
        overrides["clustering"] = {"mode": "joint" if joint else "independent"}

        st.caption(
            "Corpus timeframe — restrict the literature to works you added to your "
            'library in a window (Zotero "Date Added", not publication year). '
            "Leave both empty for the whole corpus."
        )
        columns = st.columns(3)
        added_from = columns[0].text_input("Added from", placeholder="YYYY-MM-DD")
        added_to = columns[1].text_input(
            "Added to", placeholder="YYYY-MM-DD", help="Inclusive of the whole day"
        )
        include_undated = columns[2].checkbox(
            "Keep undated works",
            value=False,
            help='Works with no "Date Added" cannot be placed in time and are '
            "dropped by default. Your drafts are never filtered.",
        )
        timeframe: dict = {}
        if added_from.strip():
            timeframe["added_from"] = added_from.strip()
        if added_to.strip():
            timeframe["added_to"] = added_to.strip()
        if include_undated:
            timeframe["include_undated"] = True
        if timeframe:
            overrides["timeframe"] = timeframe
            st.caption(f"This run: {_timeframe_preview(timeframe)}")
    # Section-wise merge onto the already-loaded config: these overrides only
    # ever touch keys one level deep, and rebuilding from load_config would
    # discard the sidebar's paths.
    data = copy.deepcopy(config.data)
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return Config(data)


def _timeframe_preview(timeframe: dict) -> str:
    """Describe the window the user just typed, or the parse error it has."""
    from cda.corpus.timeframe import describe_window, parse_bound

    try:
        window = {
            "active": True,
            "added_from": (
                parse_bound(timeframe["added_from"], end_of_day=False)
                if timeframe.get("added_from")
                else None
            ),
            "added_to": (
                parse_bound(timeframe["added_to"], end_of_day=True)
                if timeframe.get("added_to")
                else None
            ),
            "include_undated": timeframe.get("include_undated", False),
        }
    except ValueError as exc:
        return f"Warning: {exc}"
    return describe_window(window)


def render_analyze_tab(config, drafts_dir: str, mock_llm: bool) -> None:
    st.subheader("Analyze a term")
    dropped_drafts = st.file_uploader(
        "Drop your working drafts here (.docx / .md / .txt / .pdf), always "
        "treated as your own writing",
        type=["docx", "md", "txt", "pdf"],
        accept_multiple_files=True,
        key="draft_drop",
    )
    if dropped_drafts:
        drafts_target = Path(drafts_dir).expanduser()
        drafts_target.mkdir(parents=True, exist_ok=True)
        for uploaded in dropped_drafts:
            (drafts_target / Path(uploaded.name).name).write_bytes(uploaded.getvalue())
        st.caption(
            f"Saved {len(dropped_drafts)} draft(s) to {drafts_target}, they'll be "
            "included in the next analysis (files stay on this machine)."
        )
    term = st.text_input("Target term", placeholder="e.g. novelty")
    preview_only = st.toggle(
        "Prompt preview only (dry run: assemble everything, send nothing)",
        value=False,
        help="The human-in-the-loop mechanism: inspect the exact prompt before any live run.",
    )
    run_config = _analysis_options(config)
    if st.button("Run analysis", disabled=not term):
        from cda.pipeline import analyze_term

        config = run_config
        # Same location the CLI reads and records (cda analyze
        # --record-fixtures), so fixtures recorded by either replay in the
        # other.
        fixtures_dir = config.path("paths.output_dir") / "fixtures"
        if mock_llm and not preview_only:
            # Same precheck the CLI runs (require_recorded_fixtures): the
            # provider layer raises the same refusal, but only after
            # extraction and clustering — minutes on a real corpus. A dry
            # run makes no LLM call, so it needs no fixtures.
            from cda.llm.mock import has_recorded_fixtures

            if not has_recorded_fixtures(fixtures_dir):
                st.error(
                    "The mock toggle is ON but there are no recorded "
                    f"responses to replay under {fixtures_dir}, so this run "
                    "would spend the embedding and clustering time and then "
                    "fail. Two ways out: switch OFF the sidebar's mock "
                    "toggle (enter your API key) and run once live — that "
                    "run records replayable responses here automatically — "
                    "or record them from the terminal with "
                    "`cda analyze --record-fixtures`."
                )
                return
        embedder = embedder_or_error(config)
        if embedder is None:
            return
        result = run_with_live_log(
            lambda progress: analyze_term(
                term,
                config,
                mock=mock_llm,
                dry_run=preview_only,
                # Live runs record each response as a replayable fixture in the
                # user's own fixtures directory; a deliberate live rerun of the
                # same prompt replaces the earlier recording.
                record_fixtures=not mock_llm,
                overwrite_fixtures=not mock_llm,
                fixtures_dir=fixtures_dir,
                embedder=embedder,
                return_passages=True,
                progress=progress,
            )
        )
        # Session state, not a local: this is the path that spends the
        # researcher's own money, and its result must survive the rerun any
        # later widget interaction (its own download button included)
        # triggers — the rule render_emergence_expander states.
        st.session_state["analyze_result"] = result

    result = st.session_state.get("analyze_result")
    if result and result["status"] == "dry-run":
        st.info("Dry run: nothing was sent. The fully assembled prompt:")
        with st.expander("System prompt", expanded=False):
            st.code(load_system_prompt(config), language="markdown")
        prompt_path = Path(result["prompt_file"])
        with st.expander("User prompt (your passages)", expanded=True):
            if prompt_path.is_file():
                st.code(prompt_path.read_text(encoding="utf-8"), language="markdown")
            else:
                st.info("This preview has expired; run the preview again to rebuild it.")
    elif result and result["status"] == "ok":
        _show_result_tabs(result, key_prefix="analyze_result")
    elif result and result["status"] == "skipped":
        st.warning(result["reason"])


def render_emergence_expander(config) -> None:
    """EXPERIMENTAL, LLM-free: cluster emergence over library-acquisition time."""
    with st.expander("Cluster emergence timeline (experimental, no LLM call)"):
        st.write(
            "How the literature clusters for a term emerged as you expanded "
            'your library, ordered by Zotero\'s "Date Added": your reading '
            "trajectory, not the field's publication history. Needs "
            "`date_added` in the corpus (re-ingest with your Zotero CSV on "
            "older corpora)."
        )
        term = st.text_input("Term", placeholder="e.g. novelty", key="emergence_term")
        if st.button("Track emergence", disabled=not term, key="emergence_run"):
            from cda.divergence.emergence import run_emergence

            embedder = embedder_or_error(config)
            if embedder is None:
                return
            result = run_with_live_log(
                lambda progress: run_emergence(term, config, embedder=embedder, progress=progress)
            )
            if result:
                # Keep the result across reruns: Streamlit re-executes this
                # script on every widget interaction, and a blank panel after
                # a completed run is exactly what we owe the user not to show.
                st.session_state["emergence_result"] = result
                st.session_state.pop("emergence_result_stale", None)
            elif "emergence_result" in st.session_state:
                # The earlier result is kept (it can be a long run) and
                # labelled as not the one just requested.
                st.session_state["emergence_result_stale"] = True

        result = st.session_state.get("emergence_result")
        if result:
            if st.session_state.get("emergence_result_stale"):
                st.warning(
                    "The most recent run did not finish, so what follows is "
                    'still the last run that did. Press "Track emergence" '
                    "again to retry."
                )
            _show_emergence_result(result)


def _show_emergence_result(result: dict) -> None:
    """Persistent emergence result view: timeline chart + births + full report."""
    emergence = result["emergence"]
    st.caption(f'Last run: "{emergence["term"]}"')
    if result["status"] != "ok":
        st.warning(emergence.get("reason", result["status"]))
        report_path = Path(result["report_file"])
        if report_path.is_file():
            st.caption(f"Details: {report_path}")
        return

    coverage = emergence["coverage"]
    if coverage["n_undated"]:
        st.caption(
            f"Warning: {coverage['n_undated']} of {coverage['n_literature']} passages "
            f"have no acquisition date and are excluded (coverage "
            f"{coverage['rate']:.0%}); "
            + (
                "re-ingest the PDFs from disk to raise it."
                if emergence["date_source"] == "filesystem"
                else "re-ingest with your Zotero CSV to raise it."
            )
        )

    timeline_tab, report_tab = st.tabs(["Timeline", "Full report"])
    with timeline_tab:
        chart = visuals.emergence_chart(emergence)
        if chart is None:
            st.info("Too few snapshots to draw a timeline; see the full report.")
        else:
            st.altair_chart(chart, width="stretch")
            st.caption(
                "Each line is one final cluster's size as your library grew; "
                "the diamond marks its birth: the first snapshot whose "
                "structure already contained it. The x-axis is when works "
                "entered your library (the run's date source), not when they "
                "were published."
            )
        st.dataframe(
            [
                {
                    "cluster": f"Cluster {cluster['final_cluster']}",
                    "passages": cluster["n_passages"],
                    "born": cluster["birth_date"],
                    "born at N": cluster["birth_n"],
                    "persistence": cluster["persistence"],
                    "mean match cosine": cluster["mean_match_cosine"],
                }
                for cluster in emergence["final"]["clusters"]
            ],
            hide_index=True,
            width="stretch",
        )
    with report_tab:
        report_path = Path(result["report_file"])
        if report_path.is_file():
            render_report_file(report_path, key_prefix="emergence_result")
        else:
            st.info(f"The report file has moved or been deleted: {report_path}")


def render_reports_tab(output_dir: str) -> None:
    st.subheader("Past reports")
    reports_root = Path(output_dir).expanduser()
    report_paths = (
        sorted(
            [
                *reports_root.glob("drift_*.md"),
                *reports_root.glob("emergence_*.md"),
                # Diagnose writes here too; without this its reports would be
                # findable only through the filesystem after a restart.
                *reports_root.glob("corpus_diagnosis_*.md"),
                *reports_root.glob("validation_*/validation_*.md"),
                *reports_root.glob("validation_*/curated_drift_*.md"),
            ],
            reverse=True,
        )
        if reports_root.is_dir()
        else []
    )
    report_paths = [p for p in report_paths if not p.name.endswith(("_prompt.md", "_system.md"))]
    if not report_paths:
        st.info("No reports yet in this output directory. Run an analysis or the demo.")
    else:
        chosen = st.selectbox("Report", report_paths, format_func=lambda p: p.name)
        if chosen:
            render_report_file(chosen)


def render_diagnose_tab(config) -> None:
    st.subheader("Corpus composition (OpenAlex)")
    st.write(
        "A reflexive diagnostic that makes your curated corpus's structure "
        "visible: venue concentration, field coverage, temporal skew. "
        "Curation is constitutive of the method; this view supports informed "
        "curation, it does not treat curation as a defect."
    )
    declared_fields = st.text_input(
        "Declared fields (comma-separated OpenAlex field names, optional)",
        placeholder="Social Sciences",
        help="Adds a coverage baseline: which large topics of these fields the corpus doesn't touch.",
    )
    st.caption(
        "Needs an OpenAlex address — the sidebar field, or OPENALEX_MAILTO in "
        "your environment. Sends only DOIs and field queries; responses are "
        "cached locally."
    )
    if st.button("Run diagnosis"):
        from cda.diagnose.report import run_diagnose

        with session_mailto_in_env():
            result = run_with_live_log(
                lambda progress: run_diagnose(
                    config,
                    declared_fields=[f.strip() for f in declared_fields.split(",") if f.strip()],
                    progress=progress,
                )
            )
        # Assigned unconditionally: a failed re-diagnosis must not leave the
        # previous report on screen (re-running is free; responses are cached).
        st.session_state["diagnose_result"] = result
    result = st.session_state.get("diagnose_result")
    if result:
        # "diagnose_tab", not a name borrowed from another panel: two call
        # sites sharing a prefix would collide on their widget keys.
        render_report_file(Path(result["report_file"]), key_prefix="diagnose_tab")


# --- main flow (reads as a table of contents) --------------------------------

st.title("Conceptual Divergence Analysis")
st.write(
    "Compare how theoretical terms are used in your manuscripts and your curated literature corpus."
)

render_demo_button(mock_llm, output_dir)
render_own_text_panel(key_ok)

# The workflow tabs operate on a local corpus: they ingest PDFs, write to
# configured paths, spend an API key, and query OpenAlex. That is the right
# behavior on a researcher's own machine and the wrong behavior on a shared
# public URL, so the public demo withholds them and points at the install
# instructions instead (see DEMO_MODE).
if DEMO_MODE:
    st.divider()
    st.subheader("Analyzing your own corpus")
    st.write(
        "Install CDA locally to analyze your curated literature corpus and "
        "unpublished drafts. The local interface includes ingestion, live "
        "analysis, validation, reports, and corpus diagnostics."
    )
    st.code(
        "git clone https://github.com/JinaLee-soc/conceptual-divergence-analysis\n"
        "cd conceptual-divergence-analysis\n"
        'pip install -e ".[gui]"\n'
        "streamlit run gui/app.py",
        language="bash",
    )
    st.caption(
        "Your PDFs, vector store, embeddings, and reports stay on your machine. A "
        "**live** run sends the assembled prompt — the extracted term-sentences "
        "and cluster statistics, not the corpus — to the Anthropic API, and the "
        "dry-run mode shows you that prompt before anything is sent. For work "
        "under embargo or NDA, point `llm.model` at a locally served "
        "OpenAI-compatible model and nothing leaves at all. Docker "
        "(`docker compose up`) and a macOS launcher are also included; see the "
        "repository README and docs/privacy_and_cost.md."
    )
else:
    setup_tab, ingest_tab, analyze_tab, validate_tab, reports_tab, diagnose_tab = st.tabs(
        ["Setup", "Ingest", "Analyze", "Validate", "Reports", "Diagnose"]
    )
    with setup_tab:
        render_setup_tab(config, key_message)
    with ingest_tab:
        render_ingest_tab(config, corpus_dir, drafts_dir)
    with analyze_tab:
        render_analyze_tab(config, drafts_dir, mock_llm)
        render_emergence_expander(config)
    with validate_tab:
        render_validate_tab(config, mock_llm, key_ok)
    with reports_tab:
        render_reports_tab(output_dir)
    with diagnose_tab:
        render_diagnose_tab(config)

st.divider()
st.caption(
    "Network use: the toy-corpus analysis replays recorded responses. Pasted "
    "text is sent to this server for an in-memory prompt preview; no model API "
    "is contacted. The first analysis downloads the embedding model from "
    "Hugging Face unless it is already cached on this host (the Docker image "
    f"includes it). {usage_statistics_note()}"
    if DEMO_MODE
    else "Network use: PDFs, the vector store, embeddings, and reports never leave "
    "this machine. On a **live** run the assembled prompt does — the extracted "
    "term-sentences, their filenames, and cluster statistics go to the Anthropic "
    "API (preview it first with the dry-run toggle). Diagnose additionally sends "
    "DOIs/queries to OpenAlex. For work under embargo or NDA, point `llm.model` at "
    "a local OpenAI-compatible endpoint and nothing leaves at all — see "
    "docs/privacy_and_cost.md."
)
