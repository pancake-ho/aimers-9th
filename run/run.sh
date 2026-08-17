#!/usr/bin/env bash
#SBATCH -J second-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH -o /data/surt321/repos/aimers_9th/logs/slurm-%A.out

set -Eeuo pipefail

PROJECT_DIR="/data/surt321/repos/aimers_9th"
JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
LOCAL_JOB_ROOT="/local_datasets/${USER}/aimers_9th/job-${JOB_ID}"
LOCAL_PROJECT="${LOCAL_JOB_ROOT}/project"
DATA_ARCHIVE="/data/${USER}/datasets/aimers_9th/open.zip"

cleanup() {
    exit_code=$?
    if [[ -n "${LOCAL_JOB_ROOT:-}" &&
          "${LOCAL_JOB_ROOT}" == "/local_datasets/${USER}/aimers_9th/job-"* &&
          -d "${LOCAL_JOB_ROOT}" ]]; then
        echo "[CLEANUP] Removing job-local files: ${LOCAL_JOB_ROOT}"
        rm -rf -- "${LOCAL_JOB_ROOT}"
    fi
    exit "${exit_code}"
}
trap cleanup EXIT

echo "============================================================"
echo "[JOB] id=${JOB_ID}"
echo "[JOB] host=$(hostname)"
echo "[JOB] started=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[JOB] submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
echo "[JOB] project=${PROJECT_DIR}"
echo "============================================================"

if [[ "$(hostname -s)" == *-master ]]; then
    echo "[ERROR] Training must not run on moana-master. Submit with sbatch."
    exit 1
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
    echo "[ERROR] Conda initialization file not found: ${CONDA_SH}"
    exit 1
fi
source "${CONDA_SH}"
conda activate aimers

echo "[ENV] conda=${CONDA_DEFAULT_ENV:-unknown}"
echo "[ENV] python=$(which python)"
python --version
python - <<'PY'
import catboost
import numpy
import pandas
import torch
import xgboost

print(f"[ENV] numpy={numpy.__version__}")
print(f"[ENV] pandas={pandas.__version__}")
print(f"[ENV] xgboost={xgboost.__version__}")
print(f"[ENV] catboost={catboost.__version__}")
print(f"[ENV] torch={torch.__version__}")
print(f"[ENV] torch_cuda_build={torch.version.cuda}")
PY

echo "[RESOURCE] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
free -h
df -h /local_datasets

echo "[GPU-PROBE] Testing PyTorch CUDA initialization"
python - <<'PY'
import torch

assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
device = torch.device("cuda")
x = torch.randn(1024, 1024, device=device)
y = x @ x.T
assert torch.isfinite(y).all()
torch.cuda.synchronize()
print(f"[GPU-PROBE] PASS device={torch.cuda.get_device_name(0)}")
PY

echo "[STAGE] Copying source code to ${LOCAL_PROJECT}"
rsync -a \
    --exclude=".git/" \
    --exclude="baseline/data/" \
    --exclude="logs/" \
    --exclude="outputs/" \
    --exclude="submission_build/" \
    --exclude="dist/" \
    --exclude="artifacts/" \
    --exclude="__pycache__/" \
    "${PROJECT_DIR}/" \
    "${LOCAL_PROJECT}/"

mkdir -p "${LOCAL_PROJECT}/baseline/data"
REQUIRED_DATA_FILES=(
    "train.csv"
    "test.csv"
    "sample_submission.csv"
    "trackman_history.csv"
)

if [[ -f "${DATA_ARCHIVE}" ]]; then
    echo "[STAGE] Using compressed dataset: ${DATA_ARCHIVE}"
    cp -- "${DATA_ARCHIVE}" "${LOCAL_JOB_ROOT}/open.zip"
    mkdir -p "${LOCAL_JOB_ROOT}/open"
    unzip -q "${LOCAL_JOB_ROOT}/open.zip" -d "${LOCAL_JOB_ROOT}/open"
    if [[ ! -d "${LOCAL_JOB_ROOT}/open/data" ]]; then
        echo "[ERROR] open.zip does not contain data/."
        exit 1
    fi
    rsync -a "${LOCAL_JOB_ROOT}/open/data/" "${LOCAL_PROJECT}/baseline/data/"
else
    echo "[STAGE] ${DATA_ARCHIVE} not found; copying repository CSV files."
    for filename in "${REQUIRED_DATA_FILES[@]}"; do
        source_path="${PROJECT_DIR}/baseline/data/${filename}"
        if [[ ! -f "${source_path}" ]]; then
            echo "[ERROR] Missing dataset file: ${source_path}"
            echo "[ERROR] Upload open.zip to ${DATA_ARCHIVE}."
            exit 1
        fi
        cp -- "${source_path}" "${LOCAL_PROJECT}/baseline/data/${filename}"
    done
fi

for filename in "${REQUIRED_DATA_FILES[@]}"; do
    data_path="${LOCAL_PROJECT}/baseline/data/${filename}"
    if [[ ! -s "${data_path}" ]]; then
        echo "[ERROR] Missing or empty staged file: ${data_path}"
        exit 1
    fi
    echo "[DATA] $(du -h "${data_path}")"
done

cd "${LOCAL_PROJECT}"
echo "[STAGE] Local working directory: $(pwd)"
echo "[TRAIN] XGBoost=CPU, CatBoost=CPU, ResNet=CUDA, FT-Transformer=CUDA"

TRAIN_ARGS=(
    "--clean"
    "--cat-task-type" "CPU"
    "--xgb-device" "cpu"
    "--nn-device" "cuda"
)
echo "[TRAIN] command: python scripts/train_submit.py ${TRAIN_ARGS[*]}"
python scripts/train_submit.py "${TRAIN_ARGS[@]}"

echo "[BUILD] Building submit.zip"
python scripts/build_submit.py
SUBMIT_PATH="${LOCAL_PROJECT}/dist/submit.zip"
if [[ ! -s "${SUBMIT_PATH}" ]]; then
    echo "[ERROR] submit.zip was not created: ${SUBMIT_PATH}"
    exit 1
fi

RESULT_DIR="${PROJECT_DIR}/artifacts/job-${JOB_ID}"
mkdir -p "${RESULT_DIR}" "${PROJECT_DIR}/dist"
cp -- "${SUBMIT_PATH}" "${PROJECT_DIR}/artifacts/submit-${JOB_ID}.zip"
cp -- "${SUBMIT_PATH}" "${PROJECT_DIR}/dist/submit.zip"
rsync -a "${LOCAL_PROJECT}/submission_build/" "${RESULT_DIR}/submission_build/"

sha256sum "${PROJECT_DIR}/artifacts/submit-${JOB_ID}.zip"
echo "============================================================"
echo "[DONE] finished=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[DONE] submit=${PROJECT_DIR}/artifacts/submit-${JOB_ID}.zip"
echo "[DONE] latest=${PROJECT_DIR}/dist/submit.zip"
echo "[DONE] report=${RESULT_DIR}/submission_build/model/validation_report.json"
echo "============================================================"
