"""Helpers shared by the GUI's tab modules (app.py, validate.py)."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

from cda.corpus import embeddings


def demo_mode_enabled() -> bool:
    """Whether the app runs as the public demo (``CDA_DEMO_MODE``); see app.py."""
    return os.environ.get("CDA_DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def usage_statistics_note() -> str:
    """The footer's sentence about Streamlit usage statistics.

    Read from the live option rather than asserted, so the public demo never
    claims statistics are off when a launch ignored .streamlit/config.toml.
    """
    if st.get_option("browser.gatherUsageStats"):
        return "Streamlit's usage statistics are ON for this deployment (browser.gatherUsageStats)."
    return "Streamlit's usage statistics are turned off for this app."


@st.cache_resource(show_spinner="Loading the embedding model (first time ~90 MB)…")
def cached_embedder(model_name: str, revision: str | None):
    return embeddings.SentenceTransformerEmbedder(model_name, revision=revision)


def embedder_or_error(config):
    """The embedder, or None after an st.error explaining the failure.

    The first load downloads the model, and every button handler needs it —
    so an offline first run would otherwise fill whichever tab was clicked
    with a Hugging Face traceback. Callers return on None.
    """
    try:
        return cached_embedder(
            str(config.get("embedding.model")), config.get("embedding.revision", None)
        )
    except Exception as exc:
        st.error(
            "Could not load the embedding model "
            f"({config.get('embedding.model')}): {exc}\n\n"
            "The first run downloads it (~90 MB) from Hugging Face — check "
            "your network, then click again. Afterwards it loads from the "
            "local cache and works offline."
        )
        return None


def session_api_key() -> str:
    """The key entered in the sidebar; lives only in this browser session's
    memory (never on disk) and vanishes when the tab closes."""
    return str(st.session_state.get("anthropic_session_key", "") or "").strip()


@contextmanager
def session_mailto_in_env():
    """Expose the session-entered OpenAlex address for ONE request.

    Same reasoning as ``session_key_in_env``: the diagnostic reads
    OPENALEX_MAILTO from the environment, but the environment is
    process-global, so a value written there would be attached to every other
    visitor's OpenAlex requests and would outlive the session that entered it.
    An address already in the environment is left alone as the fallback.
    """
    mailto = str(st.session_state.get("openalex_session_mailto", "") or "").strip()
    if not mailto:
        yield
        return
    previous = os.environ.get("OPENALEX_MAILTO")
    os.environ["OPENALEX_MAILTO"] = mailto
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OPENALEX_MAILTO", None)
        else:
            os.environ["OPENALEX_MAILTO"] = previous


@contextmanager
def session_key_in_env():
    """Expose the session-entered key to the provider layer for ONE run.

    Providers read ANTHROPIC_API_KEY from the environment at construction;
    this injects the session key just for the wrapped call and scrubs it in
    a finally block, so the key never outlives the run outside session
    memory. (Single-user app assumption, like the rest of this GUI: on a
    shared multi-user server the environment is process-global during the
    run itself.)
    """
    key = session_api_key()
    if not key:
        yield
        return
    previous = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = previous


def render_report_file(path: Path, key_prefix: str = "reports_tab") -> None:
    """Render a report with a download button.

    ``key_prefix`` names the CALL SITE: Streamlit renders every tab's widgets
    on each pass, so the same file shown in two places (e.g. the Validate
    tab's live report and the Reports tab's selection) needs two distinct
    widget keys or the app crashes with StreamlitDuplicateElementKey.
    """
    # The path may come from session state and outlive the file: a demo
    # session directory swept by the TTL, or a deleted demo/_runtime. A raw
    # FileNotFoundError traceback would fill the page; say what happened.
    if not path.is_file():
        st.info(
            f"This report is no longer on disk ({path.name}). It may have been "
            "cleaned up — rerun the analysis to regenerate it."
        )
        return
    text = path.read_text(encoding="utf-8")
    # On the hosted demo the full path is the host's private filesystem
    # layout; a visitor only needs the filename.
    st.caption(path.name if demo_mode_enabled() else str(path))
    # Streamlit displays HTML comments as text with safe Markdown rendering.
    # Hide only the generated leading metadata; downloads retain the original.
    display_text = re.sub(r"\A(?:\s*<!--.*?-->\s*)+", "", text, flags=re.DOTALL)
    st.markdown(display_text)
    st.download_button(
        "Download this report",
        data=text,
        file_name=path.name,
        key=f"download_{key_prefix}_{path.name}",
    )


def _public_progress(message: str) -> str | None:
    """Public status text contains no raw paths, source names, or exception details."""
    stages = {
        "[corpus] another ingest or retrieval": "Waiting for another corpus operation to finish…",
        "[corpus] removed a stale lock": "Recovered an interrupted corpus operation; continuing…",
        "[demo] toy corpus:": "Preparing the toy literature corpus…",
        "[demo] first run:": "Indexing the toy literature corpus…",
        "[1/3]": "Extracting matching passages…",
        "[2/3]": "Clustering passages…",
        "[3/3]": "Assigning registers…",
    }
    return next((label for prefix, label in stages.items() if message.startswith(prefix)), None)


def run_with_live_log(fn, *, status_label="Running…", failure_message=None, **kwargs):
    """Run a pipeline function, streaming its progress lines into the page.

    The session-entered API key (if any) is exposed to the provider layer
    only for the duration of this call. ``failure_message(exc)`` renders the
    error shown on failure; by default it is the exception's text. In public
    demo mode only fixed stage labels are streamed, never the raw log text.
    """
    log_box = st.status(status_label, expanded=True)
    lines: list[str] = []

    def progress(message: str) -> None:
        if demo_mode_enabled():
            message = _public_progress(message)
            if message is None:
                return
        if lines and lines[-1] == message:
            return
        lines.append(message)
        log_box.write(message)

    try:
        with session_key_in_env():
            result = fn(progress=progress, **kwargs)
        log_box.update(label="Finished", state="complete")
        return result
    except Exception as exc:
        log_box.update(label="Failed", state="error")
        st.error(failure_message(exc) if failure_message else str(exc))
        return None
