#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="/data/${USER}/repos/aimers_9th"
MODEL_DIR="/data/${USER}/models/tabdpt"
MODEL_PATH="${MODEL_DIR}/tabdpt1_2.safetensors"
CONDA_ENV="aimers"
EXPECTED_BYTES="254098072"
EXPECTED_SHA256="06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd"

CONDA_SH="/data/${USER}/anaconda3/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[ERROR] Missing conda initialization: ${CONDA_SH}"
    exit 1
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

echo "[SETUP] Installing a mutually compatible PyTorch stack in ${CONDA_ENV}"
python -m pip install \
    --force-reinstall \
    --no-cache-dir \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu118

echo "[SETUP] Installing the complete TabDPT 1.2 runtime contract"
python -m pip install \
    "faiss-cpu>=1.11.0,<1.13.0" \
    "huggingface-hub>=0.33.2,<2.0" \
    "numpy>=1.25.0,<3.0" \
    "omegaconf>=2.1.1,<3.0" \
    "safetensors>=0.5.3,<1.0" \
    "scikit-learn>=1.4.0,<2.0" \
    "scipy>=1.9.0,<2.0" \
    "tqdm>=4.38.0,<5.0"
python -m pip install --no-deps "tabdpt==1.2.0"
python -m pip check

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

print(f"[SETUP] tabdpt={version('tabdpt')}")
print(f"[SETUP] torch={torch.__version__} cuda_build={torch.version.cuda}")
if version("tabdpt") != "1.2.0":
    raise RuntimeError(f"Expected tabdpt==1.2.0, got {version('tabdpt')}")
if not torch.__version__.startswith("2.6.0"):
    raise RuntimeError(f"Expected torch 2.6.0, got {torch.__version__}")
if torch.version.cuda != "11.8":
    raise RuntimeError(
        f"Expected the CUDA 11.8 PyTorch build, got {torch.version.cuda}"
    )
print("[SETUP] torch SDPA API PASS")
PY

echo "[DONE] ${MODEL_PATH}"
echo "[DONE] conda_env=${CONDA_ENV}"
echo "[NEXT] cd ${PROJECT_DIR} && sbatch run/run_tabdpt_submit.sh"