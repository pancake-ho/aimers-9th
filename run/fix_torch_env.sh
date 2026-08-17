#!/usr/bin/env bash

set -Eeuo pipefail

CONDA_SH="/data/${USER}/anaconda3/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[ERROR] Conda initialization file not found: ${CONDA_SH}"
    exit 1
fi

source "${CONDA_SH}"
conda activate aimers

echo "[ENV] python=$(which python)"
python --version

echo "[FIX] Removing PyTorch packages that can pin an incompatible CUDA build"
python -m pip uninstall -y torch torchvision torchaudio

echo "[FIX] Installing the CUDA 12.1 build used by this project"
python -m pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[VERIFY] Checking the installed build"
python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")

assert torch.__version__.split("+", 1)[0] == "2.5.1", torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
print("[VERIFY] PASS: torch 2.5.1+cu121")
PY

echo "[DONE] Submit the training job with: sbatch run/run.sh"