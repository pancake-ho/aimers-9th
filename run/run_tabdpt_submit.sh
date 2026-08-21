#!/usr/bin/env bash
#SBATCH -J tabdpt-xgb-v1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH --exclude=moana-y5,moana-u8
#SBATCH -t 1-0
#SBATCH -o /data/surt321/repos/aimers_9th/logs/slurm-%A.out

set -Eeuo pipefail

PROJECT_DIR="/data/${USER}/repos/aimers_9th"
JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
CONDA_ENV="aimers"
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
echo "[JOB] experiment=representative_context_tabdpt_turbo_xgb_v1"
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
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[ERROR] Missing conda initialization: ${CONDA_SH}"
    exit 4
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
python --version
python - <<'PY'
from importlib.metadata import version

print(f"[ENV] numpy_package={version('numpy')}", flush=True)
print(f"[ENV] xgboost_package={version('xgboost')}", flush=True)
print(f"[ENV] torch_package={version('torch')}", flush=True)
print(f"[ENV] tabdpt_package={version('tabdpt')}", flush=True)

import numpy
import torch
import torch.nn.functional as F
import xgboost
from torch.nn.attention import SDPBackend, sdpa_kernel

from tabdpt import TabDPTClassifier

print(f"[ENV] torch={torch.__version__} cuda_build={torch.version.cuda}", flush=True)
if version("tabdpt") != "1.2.0":
    raise RuntimeError(f"Expected tabdpt==1.2.0, got {version('tabdpt')}")
if not torch.__version__.startswith("2.6.0"):
    raise RuntimeError(f"Expected torch 2.6.0, got {torch.__version__}")
if torch.version.cuda != "11.8":
    raise RuntimeError(
        f"Expected the CUDA 11.8 PyTorch build, got {torch.version.cuda}"
    )
if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch CUDA initialization failed inside the allocated Slurm job. "
        "Re-run run/setup_tabdpt_env.sh and confirm this job requests --gres=gpu:1."
    )

device = torch.device("cuda:0")

gpu_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
major, minor = capability

print(
    f"[GPU] device={gpu_name}",
    flush=True,
)
print(
    f"[GPU] compute_capability=sm{major}{minor}",
    flush=True,
)

x = torch.ones(
    (128, 128),
    device=device,
)

product = x @ x

if not torch.isfinite(product).all():
    raise RuntimeError(
        "CUDA matrix multiplication "
        "produced non-finite values."
    )

print(
    "[GPU] CUDA matmul PASS",
    flush=True,
)

# This experiment intentionally keeps the same Flash-Attention
# inference path as the existing TabDPT temporal baseline.
#
# PyTorch's native Flash SDPA used by this environment requires
# Ampere-or-newer CUDA GPUs (sm80+).  Running the representative
# context experiment through a non-Flash path would change more
# than the context-selection variable and weaken the ablation.
if major < 8:
    raise RuntimeError(
        "This TabDPT temporal ablation requires an "
        "Ampere-or-newer GPU (compute capability >= 8.0) "
        "to preserve the validated Flash-Attention path. "
        f"Allocated GPU: {gpu_name}, "
        f"compute capability sm{major}{minor}."
    )

if not torch.backends.cuda.is_flash_attention_available():
    raise RuntimeError(
        "PyTorch reports that native Flash Attention "
        "is unavailable on this CUDA build/GPU."
    )

query = torch.randn(
    (1, 4, 128, 64),
    device=device,
    dtype=torch.float16,
)

with sdpa_kernel(
    SDPBackend.FLASH_ATTENTION
):
    attention = (
        F.scaled_dot_product_attention(
            query,
            query,
            query,
        )
    )

if not torch.isfinite(
    attention
).all():
    raise RuntimeError(
        "Flash SDPA produced "
        "non-finite values."
    )

print(
    "[GPU] Flash SDPA PASS",
    flush=True,
)

del attention
del product
del query
del x

torch.cuda.empty_cache()

print(f"[GPU] device={torch.cuda.get_device_name(0)}", flush=True)
print("[GPU] CUDA matmul PASS", flush=True)
print("[GPU] Flash SDPA PASS", flush=True)
del attention, product, query, x
torch.cuda.empty_cache()
PY
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
PERSISTENT_DATA_DIR="/data/${USER}/datasets/aimers_9th/extracted_v1/data"

for required in \
    train.csv \
    test.csv \
    sample_submission.csv \
    trackman_history.csv
do
    if [[ ! -s "${PERSISTENT_DATA_DIR}/${required}" ]]; then
        echo "[ERROR] Missing persistent dataset cache:"
        echo "        ${PERSISTENT_DATA_DIR}/${required}"
        exit 3
    fi
done

export AIMERS_DATA_DIR="${PERSISTENT_DATA_DIR}"

echo "[DATA] Using validated persistent cache:"
echo "       ${AIMERS_DATA_DIR}"

du -sh \
    "${AIMERS_DATA_DIR}/train.csv" \
    "${AIMERS_DATA_DIR}/test.csv" \
    "${AIMERS_DATA_DIR}/sample_submission.csv" \
    "${AIMERS_DATA_DIR}/trackman_history.csv"

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
    --context-strategy representative_v1 \
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