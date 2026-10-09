#!/usr/bin/env bash
# mjpython (needed for the MuJoCo viewer on macOS) can't find libpython in uv's
# standalone Python builds; point it at the interpreter's LIBDIR.
set -euo pipefail
cd "$(dirname "$0")/.."
LIBDIR="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
DYLD_LIBRARY_PATH="$LIBDIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" exec .venv/bin/mjpython "$@"
