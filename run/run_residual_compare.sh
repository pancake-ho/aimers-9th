#!/usr/bin/env bash
#SBATCH -J compare-baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH --exclude=moana-y5
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH -o /data/surt321/repos/aimers_9th/logs/residual-%A.out

set -Eeuo pipefail

PROJECT_DIR="/data/surt321/repos/aimers_9th"
JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
LOCAL_JOB_ROOT="/local_datasets/${USER}/aimers_9th/residual-${JOB_ID}"
LOCAL_PROJECT="${LOCAL_JOB_ROOT}/project"
DATA_ARCHIVE="/data/${USER}/datasets/aimers_9th/open.zip"
REPORT_NAME="pitcher_residual_comparison-${JOB_ID}.json"

cleanup() {
    exit_code=$?
    if [[ -n "${LOCAL_JOB_ROOT:-}" &&
          "${LOCAL_JOB_ROOT}" == "/local_datasets/${USER}/aimers_9th/residual-"* &&
          -d "${LOCAL_JOB_ROOT}" ]]; then
        echo "[CLEANUP] Removing ${LOCAL_JOB_ROOT}"
        rm -rf -- "${LOCAL_JOB_ROOT}"
    fi
    exit "${exit_code}"
}
trap cleanup EXIT

echo "============================================================"
echo "[JOB] id=${JOB_ID} host=$(hostname)"
echo "[JOB] experiment=pitcher_logit_offset_residual_xgboost_v1"
echo "============================================================"

if [[ "$(hostname -s)" == *-master ]]; then
    echo "[ERROR] Submit this script with sbatch; do not train on moana-master."
    exit 1
fi
if [[ ! -f "${DATA_ARCHIVE}" ]]; then
    echo "[ERROR] Missing official archive: ${DATA_ARCHIVE}"
    exit 1
fi

mkdir -p \
    "${PROJECT_DIR}/logs" \
    "${PROJECT_DIR}/artifacts" \
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

source "/data/${USER}/anaconda3/etc/profile.d/conda.sh"
conda activate aimers
python --version
python - <<'PY'
import xgboost
print(f"[ENV] xgboost={xgboost.__version__}")
PY

echo "[GPU] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

echo "[STAGE] Copying experiment source"
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

cp -- "${DATA_ARCHIVE}" "${LOCAL_JOB_ROOT}/open.zip"
mkdir -p "${LOCAL_JOB_ROOT}/open"
unzip -q "${LOCAL_JOB_ROOT}/open.zip" -d "${LOCAL_JOB_ROOT}/open"
if [[ ! -d "${LOCAL_JOB_ROOT}/open/data" ]]; then
    echo "[ERROR] open.zip does not contain data/."
    exit 1
fi
mkdir -p "${LOCAL_PROJECT}/baseline/data"
rsync -a "${LOCAL_JOB_ROOT}/open/data/" "${LOCAL_PROJECT}/baseline/data/"

cd "${LOCAL_PROJECT}"
python -m unittest \
    tests.test_pitcher_residual_baseline \
    tests.test_pitcher_residual_xgboost \
    -v

REPORT_PATH="${LOCAL_PROJECT}/outputs/${REPORT_NAME}"
python scripts/compare_pitcher_residual.py \
    --xgb-device cuda \
    --context-strengths 25 100 500 \
    --full-strength 100 \
    --bootstrap-samples 2000 \
    --output "${REPORT_PATH}"

if [[ ! -s "${REPORT_PATH}" ]]; then
    echo "[ERROR] Comparison report was not created: ${REPORT_PATH}"
    exit 1
fi
cp -- "${REPORT_PATH}" "${PROJECT_DIR}/artifacts/${REPORT_NAME}"
sha256sum "${PROJECT_DIR}/artifacts/${REPORT_NAME}"
echo "[DONE] report=${PROJECT_DIR}/artifacts/${REPORT_NAME}"
