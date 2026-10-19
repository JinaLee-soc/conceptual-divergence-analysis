from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "CDA GUI Demo.command"
COMPOSE = ROOT / "docker-compose.yml"


def test_macos_gui_demo_launcher_is_double_clickable() -> None:
    assert LAUNCHER.is_file()
    assert LAUNCHER.stat().st_mode & stat.S_IXUSR


def test_macos_gui_demo_launcher_starts_streamlit_demo() -> None:
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    assert "CDA_PYTHON" in launcher_text
    assert "python3.13 python3.12 python3.11 python3" in launcher_text
    assert '"$PYTHON_BIN" -m venv .venv' in launcher_text
    assert '-m pip install -e ".[gui]"' in launcher_text
    assert "-m streamlit run gui/app.py" in launcher_text
    assert 'open "$URL"' in launcher_text


def test_gui_is_served_on_localhost_only() -> None:
    """Both launch paths must bind/publish on loopback, not the whole LAN.

    Anyone who can reach the GUI port can run live analyses on the user's
    ANTHROPIC_API_KEY and read the corpus, drafts, and reports — so the
    macOS launcher binds 127.0.0.1, and docker-compose publishes the
    container port on 127.0.0.1 for the same reason. (The 0.0.0.0 inside
    the Dockerfile is container-internal and stays.)
    """
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    assert "--server.address 127.0.0.1" in launcher_text

    compose_text = COMPOSE.read_text(encoding="utf-8")
    # Comments legitimately mention "8501:8501" (they tell the user how to
    # expose the GUI deliberately), so only non-comment lines count.
    active = "\n".join(
        line for line in compose_text.splitlines() if not line.lstrip().startswith("#")
    )
    assert '"127.0.0.1:8501:8501"' in active
    assert '"8501:8501"' not in active.replace('"127.0.0.1:8501:8501"', "")


@pytest.mark.skipif(
    shutil.which("zsh") is None,
    reason="zsh not installed; the launcher is a macOS-only .command file",
)
def test_macos_gui_demo_launcher_has_valid_zsh_syntax() -> None:
    result = subprocess.run(
        ["zsh", "-n", str(LAUNCHER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
