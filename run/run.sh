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

# A CUDA toolkit's stub libcuda is only for linking. If a shell startup file
# accidentally adds a */stubs directory to LD_LIBRARY_PATH, CUDA applications
# can load the stub instead of the compute-node driver and fail at cuInit().
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    clean_ld_library_path=""
    IFS=':' read -r -a ld_paths <<< "${LD_LIBRARY_PATH}"
    for ld_path in "${ld_paths[@]}"; do
        [[ -z "${ld_path}" ]] && continue
        if [[ "${ld_path}" == */stubs || "${ld_path}" == */stubs/ ]]; then
            echo "[ENV] Removing CUDA stub path from LD_LIBRARY_PATH: ${ld_path}"
            continue
        fi
        if [[ -z "${clean_ld_library_path}" ]]; then
            clean_ld_library_path="${ld_path}"
        else
            clean_ld_library_path="${clean_ld_library_path}:${ld_path}"
        fi
    done
    export LD_LIBRARY_PATH="${clean_ld_library_path}"
fi

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
TORCH_BUILD_OK=1
if ! python - <<'PY'
import sys

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

torch_base_version = torch.__version__.split("+", 1)[0]
if torch_base_version != "2.5.1" or torch.version.cuda != "12.1":
    print(
        "[ERROR] Incompatible PyTorch build. "
        "Expected torch=2.5.1+cu121 and torch.version.cuda=12.1.",
        file=sys.stderr,
    )
    print(
        "[FIX] Run from the login node: bash run/fix_torch_env.sh",
        file=sys.stderr,
    )
    raise SystemExit(65)
PY
then
    TORCH_BUILD_OK=0
    echo "[WARN] Preferred PyTorch CUDA build is unavailable."
    echo "[FALLBACK] Try ResNet on CPU; if PyTorch itself is unusable, retry GBDT-only."
fi

echo "[RESOURCE] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
GPU_READY=1
if ! nvidia-smi; then
    GPU_READY=0
    echo "[WARN] NVIDIA driver could not open the allocated GPU on $(hostname -s)."
    echo "[FALLBACK] Continue with leakage-safe ResNet and GBDTs on CPU."
fi

if [[ "${GPU_READY}" -eq 1 ]]; then
    nvidia-smi -L || true

    echo "[GPU-DIAG] NVIDIA device nodes"
    for device_node in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
        if [[ -e "${device_node}" ]]; then
            ls -l "${device_node}"
        else
            echo "[GPU-DIAG] MISSING ${device_node}"
        fi
    done
    for device_node in /dev/nvidia[0-9]*; do
        [[ -e "${device_node}" ]] && ls -l "${device_node}"
    done

    echo "[GPU-DIAG] Direct CUDA Driver API initialization"
    if ! python - <<'PY'
import ctypes
import ctypes.util
import os
import socket
import sys

library = ctypes.util.find_library("cuda") or "libcuda.so.1"
print(f"[GPU-DIAG] libcuda={library}")
print(f"[GPU-DIAG] LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")

try:
    cuda = ctypes.CDLL(library)
except OSError as exc:
    print(f"[ERROR] Cannot load NVIDIA libcuda.so.1: {exc}", file=sys.stderr)
    raise SystemExit(71)

cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int
result = int(cuda.cuInit(0))

name = ctypes.c_char_p()
description = ctypes.c_char_p()
if hasattr(cuda, "cuGetErrorName"):
    cuda.cuGetErrorName(result, ctypes.byref(name))
if hasattr(cuda, "cuGetErrorString"):
    cuda.cuGetErrorString(result, ctypes.byref(description))

error_name = name.value.decode() if name.value else "UNKNOWN"
error_description = description.value.decode() if description.value else "no description"
print(
    f"[GPU-DIAG] cuInit_result={result} "
    f"name={error_name} description={error_description}"
)

if result != 0:
    print(
        "[ERROR] CUDA Driver API initialization failed before PyTorch. "
        f"host={socket.gethostname()} job={os.environ.get('SLURM_JOB_ID', 'unknown')}",
        file=sys.stderr,
    )
    print(
        "[ERROR] nvidia-smi success does not prove that CUDA compute/UVM is usable. "
        "This node requires administrator repair or a different allocation.",
        file=sys.stderr,
    )
    raise SystemExit(71)
PY
    then
        GPU_READY=0
        echo "[FALLBACK] CUDA Driver API is unavailable; ResNet will use CPU."
    fi
fi

free -h
df -h /local_datasets

if [[ "${TORCH_BUILD_OK}" -eq 0 ]]; then
    GPU_READY=0
fi

if [[ "${GPU_READY}" -eq 1 ]]; then
    echo "[GPU-PROBE] Testing PyTorch CUDA initialization"
    if ! python - <<'PY'
import os
import socket
import sys

import torch

if not torch.cuda.is_available():
    print(
        "[ERROR] torch.cuda.is_available() is false after a successful cuInit. "
        f"host={socket.gethostname()} job={os.environ.get('SLURM_JOB_ID', 'unknown')}",
        file=sys.stderr,
    )
    raise SystemExit(72)
device = torch.device("cuda")
x = torch.randn(1024, 1024, device=device)
y = x @ x.T
assert torch.isfinite(y).all()
torch.cuda.synchronize()
print(f"[GPU-PROBE] PASS device={torch.cuda.get_device_name(0)}")
PY
    then
        GPU_READY=0
        echo "[FALLBACK] PyTorch CUDA probe failed; ResNet will use CPU."
    fi
fi

if [[ "${GPU_READY}" -eq 1 ]]; then
    echo "[MODE] three_model_resnet_cuda"
else
    echo "[MODE] three_model_resnet_cpu"
fi

echo "[PREFLIGHT] Validating DACON submission requirements before training"
(
    cd "${PROJECT_DIR}"
    python - <<'PY'
from scripts.build_submit import (
    SUBMISSION_DIR,
    _validate_submission_requirements,
)

path = SUBMISSION_DIR / "requirements.txt"
_validate_submission_requirements(path)
print(f"[PREFLIGHT] PASS requirements={path}")
PY
)

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

TRAIN_ARGS=(
    "--clean"
    "--cat-task-type" "CPU"
    "--xgb-device" "cpu"
)
if [[ "${GPU_READY}" -eq 1 ]]; then
    TRAIN_ARGS+=("--nn-device" "cuda")
    echo "[TRAIN] XGBoost=CPU, CatBoost=CPU, ResNet=CUDA"
else
    TRAIN_ARGS+=("--nn-device" "cpu")
    echo "[TRAIN] XGBoost=CPU, CatBoost=CPU, ResNet=CPU"
fi
echo "[TRAIN] command: python scripts/train_submit.py ${TRAIN_ARGS[*]}"
if ! python scripts/train_submit.py "${TRAIN_ARGS[@]}"; then
    echo "[FALLBACK] Three-model training failed."
    echo "[FALLBACK] Restarting clean XGBoost+CatBoost CPU training in the same job."
    TRAIN_ARGS=(
        "--clean"
        "--cat-task-type" "CPU"
        "--xgb-device" "cpu"
        "--disable-neural"
    )
    echo "[TRAIN] retry command: python scripts/train_submit.py ${TRAIN_ARGS[*]}"
    python scripts/train_submit.py "${TRAIN_ARGS[@]}"
fi

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
