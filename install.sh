#!/usr/bin/env bash
# Creates .venv and installs the Python dependencies. Model weights download on first run into ./models.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10+ is required (found: $("$PY" --version 2>&1))." >&2; exit 1
fi
if ! "$PY" -m venv --help >/dev/null 2>&1; then
  echo "python3-venv is missing. On Debian/Ubuntu: sudo apt-get install -y python3-venv" >&2; exit 1
fi
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip -q
python -m pip install -q -r requirements.txt
if ! command -v gifsicle >/dev/null 2>&1; then
  echo "Note: optional 'gifsicle' not found. GIFs will be larger. Install with: sudo apt-get install -y gifsicle"
fi
if ! python -c "import ctypes; ctypes.CDLL('libEGL.so.1')" 2>/dev/null; then
  echo "Note: libEGL is missing, so face detection (blinking on photos of people) will be skipped."
  echo "      Install with: sudo apt-get install -y libegl1 libgles2"
fi
echo "Installed. Try: .venv/bin/python animate.py test-images/pet-cat.jpg --style pet -o out/pet"
