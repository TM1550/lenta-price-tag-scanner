#!/usr/bin/env bash
set -euo pipefail
source /home/ozon_hack/lenta-oof-20261104/environment.sh
cd "$LENTA_WORKSPACE/project"
if [[ "${1:-}" != --execute ]]; then
    exec "$LENTA_WORKSPACE/.venv/bin/python" -m lenta_price_tag_scanner.server_video_oof
fi
# Run only after a new user authorization and allocation of an idle GPU0.
# The Python preflight rejects other GPU processes and never uses GPU1.
export CUDA_VISIBLE_DEVICES=0
exec "$LENTA_WORKSPACE/.venv/bin/python" -m lenta_price_tag_scanner.server_video_oof execute=true
