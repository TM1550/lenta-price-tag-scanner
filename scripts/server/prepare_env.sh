#!/usr/bin/env bash
set -euo pipefail
source /home/ozon_hack/lenta-oof-20261104/environment.sh
cd "$LENTA_WORKSPACE/project"
if [[ ! -x "$LENTA_WORKSPACE/.venv/bin/python" ]]; then
    python3 -m venv --copies --without-pip "$LENTA_WORKSPACE/.venv"
fi
if ! "$LENTA_WORKSPACE/.venv/bin/python" -m pip --version >/dev/null 2>&1; then
    curl --fail --location --connect-timeout 20 --max-time 120 https://bootstrap.pypa.io/get-pip.py -o "$TMPDIR/get-pip.py"
    "$LENTA_WORKSPACE/.venv/bin/python" "$TMPDIR/get-pip.py" --no-cache-dir
fi
"$LENTA_WORKSPACE/.venv/bin/python" -m pip install --no-cache-dir -r scripts/server/requirements-prepare.txt
"$LENTA_WORKSPACE/.venv/bin/python" -m lenta_price_tag_scanner.server_video_oof
"$LENTA_WORKSPACE/.venv/bin/python" -m pip freeze > "$LENTA_WORKSPACE/artifacts/requirements-prepare-installed.txt"
