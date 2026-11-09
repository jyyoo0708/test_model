#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
unset LD_LIBRARY_PATH
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/tmp"
export XDG_CACHE_HOME="$PWD/.cache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TORCH_HOME="$PWD/.cache/torch"
exec "$PWD/.venv/bin/python" "$@"
