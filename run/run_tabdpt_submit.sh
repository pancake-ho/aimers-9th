#!/usr/bin/env bash
#SBATCH -J tabdpt-xgb-v1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH --exclude=moana-y5
#SBATCH -t 1-0
#SBATCH -o /data/surt321/repos/aimers_9th/logs/slurm-%A.out

set -Eeuo pipefail

PROJECT_DIR="/data/${USER}/repos/aimers_9th"
JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
LOCAL_JOB_ROOT="/local_datasets/${USER}/aimers_9th/tabdpt-${JOB_ID}"
LOCAL_PROJECT="${LOCAL_JOB_ROOT}/project"
DATA_ARCHIVE="/data/${USER}/datasets/aimers_9th/open.zip"
TABDPT_WEIGHT_SOURCE="${TABDPT_WEIGHT_PATH:-/data/${USER}/models/tabdpt/tabdpt1_2.safetensors}"
TABDPT_WEIGHT_LOCAL="${LOCAL_JOB_ROOT}/tabdpt1_2.safetensors"

preserve_diagnostics() {
    report="${LOCAL_PROJECT}/tabdpt_submission_build/model/validation_report.json"
    if [[ -f "${report}" ]]; then
        destination="${PROJECT_DIR}/artifacts/tabdpt-${JOB_ID}"
        mkdir -p "${destination}"
        cp -- "${report}" "${destination}/validation_report.json"
        sha256sum "${destination}/validation_report.json"
    fi
}

cleanup() {
    exit_code=$?
    preserve_diagnostics || true
    if [[ -n "${LOCAL_JOB_ROOT:-}" && \
          "${LOCAL_JOB_ROOT}" == "/local_datasets/${USER}/aimers_9th/tabdpt-"* && \
          -d "${LOCAL_JOB_ROOT}" ]]; then
        echo "[CLEANUP] Removing ${LOCAL_JOB_ROOT}"
        rm -rf -- "${LOCAL_JOB_ROOT}"
    fi
    exit "${exit_code}"
}
trap cleanup EXIT

echo "============================================================"
echo "[JOB] id=${JOB_ID} host=$(hostname)"
echo "[JOB] experiment=recent_context_tabdpt_turbo_xgb_v1"
echo "============================================================"

if [[ "$(hostname -s)" == *-master ]]; then
    echo "[ERROR] Submit this script with sbatch; do not run on the login node."
    exit 1
fi
if [[ ! -f "${TABDPT_WEIGHT_SOURCE}" ]]; then
    echo "[ERROR] Missing ${TABDPT_WEIGHT_SOURCE}"
    echo "[FIX] Run bash run/setup_tabdpt_env.sh on the login node first."
    exit 2
fi
if [[ "$(stat -c '%s' "${TABDPT_WEIGHT_SOURCE}")" != "254098072" ]]; then
    echo "[ERROR] TabDPT checkpoint size mismatch."
    exit 2
fi

mkdir -p \
    "${PROJECT_DIR}/logs" \
    "${PROJECT_DIR}/artifacts" \
    "${PROJECT_DIR}/dist" \
    "${LOCAL_PROJECT}" \
    "${LOCAL_JOB_ROOT}/tmp" \
    "${LOCAL_JOB_ROOT}/cache"

export TMPDIR="${LOCAL_JOB_ROOT}/tmp"
export XDG_CACHE_HOME="${LOCAL_JOB_ROOT}/cache"
export JOBLIB_TEMP_FOLDER="${LOCAL_JOB_ROOT}/tmp"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export TOKENIZERS_PARALLELISM=false

CONDA_SH="/data/${USER}/anaconda3/etc/profile.d/conda.sh"
source "${CONDA_SH}"
conda activate aimers
python --version
python - <<'PY'
from importlib.metadata import version
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy
import torch
import xgboost

from tabdpt import TabDPTClassifier

assert version("tabdpt") == "1.2.0"
assert torch.cuda.is_available()
x = torch.ones((128, 128), device="cuda")
assert torch.isfinite(x @ x).all()
print(f"[ENV] numpy={numpy.__version__}")
print(f"[ENV] xgboost={xgboost.__version__}")
print(f"[ENV] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[ENV] tabdpt={version('tabdpt')}")
print(f"[GPU] {torch.cuda.get_device_name(0)}")
PY
nvidia-smi
free -h
df -h /local_datasets

echo "[STAGE] Copying source to ${LOCAL_PROJECT}"
rsync -a \
    --exclude=".git/" \
    --exclude="baseline/data/" \
    --exclude="logs/" \
    --exclude="outputs/" \
    --exclude="submission_build/" \
    --exclude="tabdpt_submission_build/" \
    --exclude="dist/" \
    --exclude="artifacts/" \
    --exclude="__pycache__/" \
    "${PROJECT_DIR}/" \
    "${LOCAL_PROJECT}/"
cp -- "${TABDPT_WEIGHT_SOURCE}" "${TABDPT_WEIGHT_LOCAL}"

if [[ ! -f "${DATA_ARCHIVE}" ]]; then
    echo "[ERROR] Missing ${DATA_ARCHIVE}"
    exit 3
fi
mkdir -p "${LOCAL_JOB_ROOT}/open" "${LOCAL_PROJECT}/baseline/data"
unzip -q "${DATA_ARCHIVE}" -d "${LOCAL_JOB_ROOT}/open"
rsync -a "${LOCAL_JOB_ROOT}/open/data/" "${LOCAL_PROJECT}/baseline/data/"
export AIMERS_DATA_DIR="${LOCAL_PROJECT}/baseline/data"
du -sh "${AIMERS_DATA_DIR}"/*

cd "${LOCAL_PROJECT}"
echo "[PREFLIGHT] Running TabDPT source and contract tests"
python -m unittest \
    tests.test_tabdpt_context \
    tests.test_tabdpt_submission_contract \
    tests.test_feature_config_alignment \
    -q

echo "[TRAIN] Forward validation, quality gate and final fit"
python scripts/train_tabdpt_submit.py \
    --clean \
    --model-weight-path "${TABDPT_WEIGHT_LOCAL}" \
    --xgb-device cuda \
    --tabdpt-device cuda \
    --context-size 32768 \
    --n-ensembles 2 \
    --inference-batch-size 65536

echo "[BUILD] Building and GPU-smoke-testing submit.zip"
export AIMERS_TABDPT_SMOKE=1
python scripts/build_tabdpt_submit.py

source_zip="${LOCAL_PROJECT}/dist/submit-tabdpt.zip"
named_zip="${PROJECT_DIR}/dist/submit-tabdpt-${JOB_ID}.zip"
temporary_zip="${PROJECT_DIR}/dist/submit.zip.tmp-${JOB_ID}"
cp -- "${source_zip}" "${named_zip}"
cp -- "${source_zip}" "${temporary_zip}"
mv -f -- "${temporary_zip}" "${PROJECT_DIR}/dist/submit.zip"
sha256sum "${named_zip}" "${PROJECT_DIR}/dist/submit.zip"
echo "[DONE] submit=${PROJECT_DIR}/dist/submit.zip"
