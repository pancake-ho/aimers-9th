#!/usr/bin/env bash
#SBATCH -J first-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH -o /data/surt321/repos/aimers_9th/logs/slurm-%A.out

set -Eeuo pipefail

# ============================================================
# 1. 기본 경로
# ============================================================
PROJECT_DIR="/data/surt321/repos/aimers_9th"
JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
LOCAL_JOB_ROOT="/local_datasets/${USER}/aimers_9th/job-${JOB_ID}"
LOCAL_PROJECT="${LOCAL_JOB_ROOT}/project"

# open.zip을 별도로 올려뒀다면 이 경로를 우선 사용한다.
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
    echo "[ERROR] Training must not run on moana-master."
    exit 1
fi

mkdir -p \
    "${PROJECT_DIR}/logs" \
    "${PROJECT_DIR}/artifacts" \
    "${PROJECT_DIR}/dist" \
    "${LOCAL_PROJECT}" \
    "${LOCAL_JOB_ROOT}/tmp" \
    "${LOCAL_JOB_ROOT}/cache"

# Python 및 라이브러리 임시 파일을 compute node 로컬 디스크에 저장
export TMPDIR="${LOCAL_JOB_ROOT}/tmp"
export XDG_CACHE_HOME="${LOCAL_JOB_ROOT}/cache"
export JOBLIB_TEMP_FOLDER="${LOCAL_JOB_ROOT}/tmp"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_GPU:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_GPU:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_GPU:-8}"

# ============================================================
# 2. Conda aimers 환경 활성화
# ============================================================
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
import xgboost

print(f"[ENV] numpy={numpy.__version__}")
print(f"[ENV] pandas={pandas.__version__}")
print(f"[ENV] xgboost={xgboost.__version__}")
print(f"[ENV] catboost={catboost.__version__}")
PY

echo "[RESOURCE] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
free -h
df -h /local_datasets

# ============================================================
# 3. 코드를 NAS에서 compute node 로컬 디스크로 복사
# ============================================================
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

# ============================================================
# 4. 공식 데이터도 compute node 로컬 디스크로 복사
# ============================================================
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
        echo "[ERROR] open.zip does not contain the data/ directory."
        exit 1
    fi

    rsync -a \
        "${LOCAL_JOB_ROOT}/open/data/" \
        "${LOCAL_PROJECT}/baseline/data/"
else
    echo "[STAGE] ${DATA_ARCHIVE} not found."
    echo "[STAGE] Copying the four official CSV files from repository storage once."

    for filename in "${REQUIRED_DATA_FILES[@]}"; do
        source_path="${PROJECT_DIR}/baseline/data/${filename}"

        if [[ ! -f "${source_path}" ]]; then
            echo "[ERROR] Missing dataset file: ${source_path}"
            echo "[ERROR] Upload open.zip to ${DATA_ARCHIVE}, or place the CSV files under baseline/data."
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
echo "[STAGE] Local data directory: ${LOCAL_PROJECT}/baseline/data"

# ============================================================
# 5. CatBoost GPU 사전검사 후 학습 실행
# ============================================================
echo "[GPU-PROBE] Testing CatBoost CUDA initialization"

if python - <<'PY'
import numpy as np
from catboost import CatBoostClassifier

X = np.asarray(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [3.0, 0.0],
        [3.0, 1.0],
    ],
    dtype=np.float32,
)
y = np.asarray([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.int32)

model = CatBoostClassifier(
    iterations=2,
    depth=2,
    learning_rate=0.1,
    loss_function="Logloss",
    task_type="GPU",
    devices="0",
    verbose=False,
    allow_writing_files=False,
)
model.fit(X, y)

prediction = model.predict_proba(X)[:, 1]
assert prediction.shape == (8,)
assert np.isfinite(prediction).all()

print("[GPU-PROBE] CatBoost GPU PASS")
PY
then
    CAT_TASK_TYPE="GPU"
    echo "[GPU-PROBE] CatBoost will use GPU"
else
    CAT_TASK_TYPE="CPU"
    echo "[GPU-PROBE] CatBoost GPU unavailable on this node"
    echo "[GPU-PROBE] Falling back to CatBoost CPU"
fi

TRAIN_ARGS=(
    "--clean"
    "--cat-task-type"
    "${CAT_TASK_TYPE}"
)

# 현재 GitHub 코드가 --xgb-device를 지원할 때만 XGBoost CUDA를 사용한다.
# CatBoost GPU probe까지 통과한 노드에서만 CUDA를 활성화한다.
XGB_TASK_TYPE="CPU"

if [[ "${CAT_TASK_TYPE}" == "GPU" ]] &&
   python scripts/train_submit.py --help | grep -q -- "--xgb-device"; then
    TRAIN_ARGS+=("--xgb-device" "cuda")
    XGB_TASK_TYPE="CUDA"
fi

echo "[TRAIN] XGBoost=${XGB_TASK_TYPE}, CatBoost=${CAT_TASK_TYPE}"
echo "[TRAIN] command: python scripts/train_submit.py ${TRAIN_ARGS[*]}"

python scripts/train_submit.py "${TRAIN_ARGS[@]}"

# ============================================================
# 6. 실제 제출용 submit.zip 생성 및 smoke test
# ============================================================
echo "[BUILD] Building submit.zip"
python scripts/build_submit.py

SUBMIT_PATH="${LOCAL_PROJECT}/dist/submit.zip"

if [[ ! -s "${SUBMIT_PATH}" ]]; then
    echo "[ERROR] submit.zip was not created: ${SUBMIT_PATH}"
    exit 1
fi

# ============================================================
# 7. 결과물을 NAS의 원본 프로젝트로 회수
# ============================================================
RESULT_DIR="${PROJECT_DIR}/artifacts/job-${JOB_ID}"
mkdir -p "${RESULT_DIR}" "${PROJECT_DIR}/dist"

cp -- "${SUBMIT_PATH}" \
    "${PROJECT_DIR}/artifacts/submit-${JOB_ID}.zip"

cp -- "${SUBMIT_PATH}" \
    "${PROJECT_DIR}/dist/submit.zip"

rsync -a \
    "${LOCAL_PROJECT}/submission_build/" \
    "${RESULT_DIR}/submission_build/"

if [[ -d "${LOCAL_PROJECT}/outputs" ]]; then
    rsync -a \
        "${LOCAL_PROJECT}/outputs/" \
        "${RESULT_DIR}/outputs/"
fi

sha256sum "${PROJECT_DIR}/artifacts/submit-${JOB_ID}.zip"

echo "============================================================"
echo "[DONE] finished=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[DONE] submit=${PROJECT_DIR}/artifacts/submit-${JOB_ID}.zip"
echo "[DONE] latest=${PROJECT_DIR}/dist/submit.zip"
echo "[DONE] report=${RESULT_DIR}/submission_build/model/validation_report.json"
echo "============================================================"