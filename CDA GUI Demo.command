#!/bin/zsh
set -euo pipefail

APP_DIR="${0:a:h}"
cd "$APP_DIR"

PORT="${CDA_GUI_PORT:-8501}"
URL="http://127.0.0.1:${PORT}"
PYTHON="$APP_DIR/.venv/bin/python"
PYTHON_BIN="${CDA_PYTHON:-}"

python_version_supported() {
  "$1" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)' >/dev/null 2>&1
}

python_version_label() {
  "$1" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>/dev/null || echo "unknown"
}

if [[ -n "$PYTHON_BIN" ]]; then
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "CDA_PYTHON is set to '$PYTHON_BIN', but that command was not found."
    echo "Set CDA_PYTHON to Python 3.11, 3.12, or 3.13, then try again."
    exit 1
  fi
  if ! python_version_supported "$PYTHON_BIN"; then
    echo "CDA_PYTHON points to Python $(python_version_label "$PYTHON_BIN"), but CDA needs Python 3.11-3.13."
    echo "Set CDA_PYTHON to a supported interpreter, then try again."
    exit 1
  fi
else
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && python_version_supported "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Could not find Python 3.11, 3.12, or 3.13."
  echo "Install one of those versions, or set CDA_PYTHON to its path, then double-click this file again."
  exit 1
fi

echo "CDA GUI demo"
echo "Repository: $APP_DIR"
echo "URL:        $URL"
echo "Python:     $PYTHON_BIN ($(python_version_label "$PYTHON_BIN"))"
echo ""

if [[ ! -d ".venv" ]]; then
  echo "Creating a local Python environment in .venv ..."
  "$PYTHON_BIN" -m venv .venv
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Could not find $PYTHON after creating .venv."
  echo "Install Python 3.11-3.13, or remove .venv and double-click this file again."
  exit 1
fi

if ! python_version_supported "$PYTHON"; then
  echo "The existing .venv uses Python $(python_version_label "$PYTHON"), but CDA needs Python 3.11-3.13."
  echo "Remove .venv, or set CDA_PYTHON to a supported interpreter, then double-click this file again."
  exit 1
fi

if ! "$PYTHON" -c "import cda, streamlit" >/dev/null 2>&1; then
  echo "Installing CDA GUI dependencies into .venv ..."
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -e ".[gui]"
fi

if nc -z 127.0.0.1 "$PORT" >/dev/null 2>&1; then
  echo "Port $PORT is already in use by another app, so the demo cannot start."
  echo "Close that app, or set CDA_GUI_PORT to a free port and double-click again."
  exit 1
fi

echo "Opening the CDA GUI demo in your browser ..."
echo "Leave this Terminal window open while using the demo."
echo "Press Ctrl-C here to stop the server."
echo ""

# Open the browser only once the server actually answers on the port — a
# fixed timer either raced a slow first boot (connection-refused page) or,
# with the port conflict above, framed some other app as the demo.
#
# The waiter also stops the moment the server process is gone. `exec` below
# replaces this shell with streamlit, so this shell's pid *is* the server's
# pid; without the check, a streamlit that died at boot left the waiter
# polling for another minute, and a relaunch inside that window opened two
# browser tabs (the stale waiter plus the new one).
SERVER_PID="$$"
(
  i=0
  while [ "$i" -lt 120 ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      exit 0
    fi
    if nc -z 127.0.0.1 "$PORT" >/dev/null 2>&1; then
      open "$URL" >/dev/null 2>&1 || true
      exit 0
    fi
    sleep 0.5
    i=$((i + 1))
  done
) &

exec "$PYTHON" -m streamlit run gui/app.py \
  --server.address 127.0.0.1 \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false
