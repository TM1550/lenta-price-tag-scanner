#!/usr/bin/env bash
set -euo pipefail
source /home/ozon_hack/lenta-oof-20261104/environment.sh
cd "$LENTA_WORKSPACE/project"
if [[ "${1:-}" != --install ]]; then
    echo 'Explicit --install required; no dependencies or model weights downloaded.'
    exit 0
fi
python3 - <<'PY'
import os, shutil
free=shutil.disk_usage(os.environ['LENTA_WORKSPACE']).free
if free < 35 * 1024**3:
    raise SystemExit(f'Need at least 35 GiB free before full CUDA/Qwen installation; available {free/1024**3:.2f} GiB. No installation started.')
PY
bash scripts/server/prepare_env.sh
"$LENTA_WORKSPACE/.venv/bin/python" -m pip install --no-cache-dir 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' --index-url https://download.pytorch.org/whl/cu128
"$LENTA_WORKSPACE/.venv/bin/python" -m pip install --no-cache-dir -r scripts/server/requirements-runtime.txt
"$LENTA_WORKSPACE/.venv/bin/python" -m pip check
"$LENTA_WORKSPACE/.venv/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-4B-Instruct', revision='ebb281ec70b05090aa6165b016eac8ec08e71b17',
                  local_dir=os.environ['LENTA_WORKSPACE']+'/models/qwen3_vl_4b',
                  allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])
PY
"$LENTA_WORKSPACE/.venv/bin/python" -m pip freeze > "$LENTA_WORKSPACE/artifacts/requirements-runtime-installed.txt"
