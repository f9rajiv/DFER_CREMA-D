#!/bin/bash
# One-time setup. Run from the repo root on BOTH login nodes:
#   ssh <user>@roihu-gpu.csc.fi   ->  bash roihu/setup.sh
#   ssh <user>@roihu-cpu.csc.fi   ->  bash roihu/setup.sh
set -euo pipefail
source roihu/env.sh

if [ ! -d "${DFER_VENV}" ]; then
    python3 -m venv --system-site-packages "${DFER_VENV}"
fi
source "${DFER_VENV}/bin/activate"
pip install --upgrade pip
pip install -r requirements_roihu.txt

mkdir -p "${DFER_DATA_DIR}"/{annotations,frames,results,checkpoints,clip_weights,raw}

python - <<'PY'
import cv2, torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| opencv", cv2.__version__)
print("Haar face detector available:", hasattr(cv2, "CascadeClassifier"))
PY
echo "venv ready: ${DFER_VENV}"
echo "data dir:   ${DFER_DATA_DIR}"