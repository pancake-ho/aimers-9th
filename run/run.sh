#!/usr/bin/env bash
#SBATCH -J second-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH --exclude=moana-y5
#SBATCH -t 1-0
#SBATCH -o /data/surt321/repos/aimers_9th/logs/slurm-%A.out

set -Eeuo pipefail

PROJECT_DIR="/data/surt321/repos/aimers_9th"
JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
LOCAL_JOB_ROOT="/local_datasets/${USER}/aimers_9th/job-${JOB_ID}"
LOCAL_PROJECT="${LOCAL_JOB_ROOT}/project"
DATA_ARCHIVE="/data/${USER}/datasets/aimers_9th/open.zip"

preserve_diagnostics() {
    report_path="${LOCAL_PROJECT}/submission_build/model/validation_report.json"
    if [[ -f "${report_path}" ]]; then
        diagnostic_dir="${PROJECT_DIR}/artifacts/job-${JOB_ID}/submission_build/model"
        mkdir -p "${diagnostic_dir}"
        cp -- "${report_path}" "${diagnostic_dir}/validation_report.json"
        echo "[DIAGNOSTIC] Preserved ${diagnostic_dir}/validation_report.json"
    fi
}

cleanup() {
    exit_code=$?
    preserve_diagnostics || true
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

echo "[ENV] package versions"
if ! python - <<'PY'
import sys

import catboost
import lightgbm
import numpy
import pandas
import torch
import xgboost

print(f"[ENV] numpy={numpy.__version__}")
print(f"[ENV] pandas={pandas.__version__}")
print(f"[ENV] xgboost={xgboost.__version__}")
print(f"[ENV] lightgbm={lightgbm.__version__}")
print(f"[ENV] catboost={catboost.__version__}")
print(f"[ENV] torch={torch.__version__}")
print(f"[ENV] torch_cuda_build={torch.version.cuda}")

# Do NOT require one exact PyTorch/CUDA wheel.
# The compute-node functional probe below is the actual compatibility test.
if torch.version.cuda is None:
    print(
        "[ERROR] Installed PyTorch is a CPU-only build. "
        "Neural training requires CUDA-enabled PyTorch.",
        file=sys.stderr,
    )
    raise SystemExit(65)
PY
then
    echo "[ERROR] Required Python runtime packages are unavailable."
    exit 65
fi


echo "[RESOURCE] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

GPU_READY=1

# ------------------------------------------------------------
# 1. NVIDIA driver visibility
# ------------------------------------------------------------
if ! nvidia-smi; then
    GPU_READY=0
    echo "[ERROR] nvidia-smi cannot access the allocated GPU on $(hostname -s)."
fi

if [[ "${GPU_READY}" -eq 1 ]]; then
    nvidia-smi -L || true

    echo "[GPU-DIAG] NVIDIA device nodes"

    for device_node in \
        /dev/nvidiactl \
        /dev/nvidia-uvm \
        /dev/nvidia-uvm-tools
    do
        if [[ -e "${device_node}" ]]; then
            ls -l "${device_node}"
        else
            echo "[GPU-DIAG] MISSING ${device_node}"
        fi
    done

    for device_node in /dev/nvidia[0-9]*; do
        [[ -e "${device_node}" ]] && ls -l "${device_node}"
    done
fi


# ------------------------------------------------------------
# 2. CUDA Driver API
# ------------------------------------------------------------
if [[ "${GPU_READY}" -eq 1 ]]; then
    echo "[GPU-DIAG] Direct CUDA Driver API initialization"

    if ! python - <<'PY'
import ctypes
import ctypes.util
import os
import socket
import sys

library = ctypes.util.find_library("cuda") or "libcuda.so.1"

print(f"[GPU-DIAG] libcuda={library}")
print(
    f"[GPU-DIAG] LD_LIBRARY_PATH="
    f"{os.environ.get('LD_LIBRARY_PATH', '')}"
)

try:
    cuda = ctypes.CDLL(library)
except OSError as exc:
    print(
        f"[ERROR] Cannot load NVIDIA libcuda.so.1: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(71)

cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int

result = int(cuda.cuInit(0))

name = ctypes.c_char_p()
description = ctypes.c_char_p()

if hasattr(cuda, "cuGetErrorName"):
    cuda.cuGetErrorName(
        result,
        ctypes.byref(name),
    )

if hasattr(cuda, "cuGetErrorString"):
    cuda.cuGetErrorString(
        result,
        ctypes.byref(description),
    )

error_name = (
    name.value.decode()
    if name.value
    else "UNKNOWN"
)

error_description = (
    description.value.decode()
    if description.value
    else "no description"
)

print(
    f"[GPU-DIAG] cuInit_result={result} "
    f"name={error_name} "
    f"description={error_description}"
)

if result != 0:
    print(
        "[ERROR] CUDA Driver API initialization failed. "
        f"host={socket.gethostname()} "
        f"job={os.environ.get('SLURM_JOB_ID', 'unknown')}",
        file=sys.stderr,
    )
    raise SystemExit(71)
PY
    then
        GPU_READY=0
        echo "[ERROR] CUDA Driver API is unavailable."
    fi
fi


# ------------------------------------------------------------
# 3. Actual PyTorch CUDA compute test
#
# This, not an exact wheel version, is the compatibility contract.
# ------------------------------------------------------------
if [[ "${GPU_READY}" -eq 1 ]]; then
    echo "[GPU-PROBE] Testing PyTorch CUDA forward/backward"

    if ! python - <<'PY'
import os
import socket
import sys

import torch

print(f"[GPU-PROBE] torch={torch.__version__}")
print(f"[GPU-PROBE] torch_cuda_build={torch.version.cuda}")

if not torch.cuda.is_available():
    print(
        "[ERROR] torch.cuda.is_available() is false. "
        f"host={socket.gethostname()} "
        f"job={os.environ.get('SLURM_JOB_ID', 'unknown')}",
        file=sys.stderr,
    )
    raise SystemExit(72)

device = torch.device("cuda")

print(
    f"[GPU-PROBE] device="
    f"{torch.cuda.get_device_name(0)}"
)
print(
    f"[GPU-PROBE] capability="
    f"{torch.cuda.get_device_capability(0)}"
)

# Exercise the operations used by the neural learners:
# CUDA allocation + GEMM + autocast + backward.
x = torch.randn(
    1024,
    1024,
    device=device,
    requires_grad=True,
)

with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
):
    y = (x @ x.T).square().mean()

if not torch.isfinite(y):
    raise RuntimeError(
        "PyTorch CUDA forward produced a non-finite value."
    )

y.backward()

if x.grad is None:
    raise RuntimeError(
        "PyTorch CUDA backward did not produce gradients."
    )

if not torch.isfinite(x.grad).all():
    raise RuntimeError(
        "PyTorch CUDA backward produced non-finite gradients."
    )

torch.cuda.synchronize()

allocated_mb = (
    torch.cuda.max_memory_allocated()
    / (1024 ** 2)
)

print(
    "[GPU-PROBE] PASS "
    f"device={torch.cuda.get_device_name(0)} "
    f"peak_allocated={allocated_mb:.1f}MiB"
)
PY
    then
        GPU_READY=0
        echo "[ERROR] PyTorch CUDA compute probe failed."
        echo "[ERROR] Only now should the PyTorch environment be repaired."
    fi
fi


# ------------------------------------------------------------
# Resource diagnostics
# ------------------------------------------------------------
free -h
df -h /local_datasets

if [[ -r /sys/fs/cgroup/memory.max ]]; then
    echo "[RESOURCE] cgroup memory.max=$(cat /sys/fs/cgroup/memory.max)"
elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    echo "[RESOURCE] cgroup memory.limit_in_bytes=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
fi


if [[ "${GPU_READY}" -ne 1 ]]; then
    echo "[ERROR] No valid CUDA backend; refusing neural training."
    exit 73
fi

echo "[GPU] Functional CUDA backend confirmed."
echo "[MODE] xgb+lgb+cat+resnet+ft_transformer temporal_v9"

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

echo "[PREFLIGHT] Exercising feature and neural source contracts"
(
    cd "${PROJECT_DIR}"
    python -m unittest \
        tests.test_config_contract \
        tests.test_feature_config_alignment \
        tests.test_four_model_ensemble \
        tests.test_optional_neural_submission \
        tests.test_run_script_contract \
        tests.test_runtime_contract \
        tests.test_trackman_entity_contract \
        tests.test_neural_config_alignment \
        tests.test_neural_contract \
        -q
)
echo "[PREFLIGHT] PASS feature/neural source contracts"

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
    "--cat-task-type" "GPU"
    "--xgb-device" "cuda"
    "--nn-device" "cuda"
)
echo "[TRAIN] XGBoost=CUDA, LightGBM=CPU, CatBoost=GPU, ResNet/FT-Transformer=CUDA"
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
