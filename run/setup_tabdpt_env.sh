#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="/data/${USER}/repos/aimers_9th"
MODEL_DIR="/data/${USER}/models/tabdpt"
MODEL_PATH="${MODEL_DIR}/tabdpt1_2.safetensors"
EXPECTED_BYTES="254098072"
EXPECTED_SHA256="06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd"

CONDA_SH="/data/${USER}/anaconda3/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[ERROR] Missing conda initialization: ${CONDA_SH}"
    exit 1
fi
source "${CONDA_SH}"
conda activate aimers

echo "[SETUP] Installing TabDPT-Turbo interface without replacing CUDA PyTorch"
python -m pip install \
    "faiss-cpu>=1.11.0,<1.13.0" \
    "huggingface-hub>=0.33.2,<2.0" \
    "omegaconf>=2.1.1,<3.0" \
    "safetensors>=0.5.3,<1.0"
python -m pip install --no-deps "tabdpt==1.2.0"

mkdir -p "${MODEL_DIR}"
if [[ -f "${MODEL_PATH}" ]] && \
   [[ "$(stat -c '%s' "${MODEL_PATH}")" == "${EXPECTED_BYTES}" ]] && \
   [[ "$(sha256sum "${MODEL_PATH}" | awk '{print $1}')" == "${EXPECTED_SHA256}" ]]; then
    echo "[SETUP] Reusing verified checkpoint: ${MODEL_PATH}"
else
    echo "[SETUP] Downloading the public Apache-2.0 TabDPT-Turbo checkpoint"
    TABDPT_DESTINATION="${MODEL_PATH}" python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

source = Path(
    hf_hub_download(repo_id="Layer6/TabDPT", filename="tabdpt1_2.safetensors")
)
destination = Path(os.environ["TABDPT_DESTINATION"])
temporary = destination.with_suffix(".safetensors.tmp")
shutil.copy2(source, temporary)
temporary.replace(destination)
print(f"[SETUP] Saved {destination}")
PY
fi

actual_bytes="$(stat -c '%s' "${MODEL_PATH}")"
actual_sha256="$(sha256sum "${MODEL_PATH}" | awk '{print $1}')"
if [[ "${actual_bytes}" != "${EXPECTED_BYTES}" || \
      "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
    echo "[ERROR] Checkpoint verification failed"
    echo "[ERROR] bytes=${actual_bytes} sha256=${actual_sha256}"
    exit 2
fi

python - <<'PY'
from importlib.metadata import version
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch

from tabdpt import TabDPTClassifier

assert version("tabdpt") == "1.2.0"
print(f"[SETUP] tabdpt={version('tabdpt')} torch={torch.__version__}")
print("[SETUP] torch SDPA API PASS")
PY

echo "[DONE] ${MODEL_PATH}"
echo "[NEXT] cd ${PROJECT_DIR} && sbatch run/run_tabdpt_submit.sh"
