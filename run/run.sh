#!/usr/bin/env bash
#SBATCH -J ensemble-v9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH --exclude=moana-y5
#SBATCH -t 1-0
#SBATCH -o /data/surt321/repos/aimers_9th/logs/slurm-%A.out

set -Eeuo pipefail


# ============================================================
# Project / job paths
# ============================================================

PROJECT_DIR="/data/surt321/repos/aimers_9th"
JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"

DATA_ARCHIVE="/data/${USER}/datasets/aimers_9th/open.zip"

LOCAL_WORK_BASE="/local_datasets/${USER}/aimers_9th"
SHARED_WORK_BASE="/data/${USER}/tmp/aimers_9th"

# Persistent, immutable extracted cache.
#
# The competition dataset is fixed.  Do not unzip ~700 MB into every
# job-local workspace and then copy it again.
DATA_CACHE_ROOT="/data/${USER}/datasets/aimers_9th/extracted_v1"
DATA_CACHE_DIR="${DATA_CACHE_ROOT}/data"

# Keep a generous node-local safety margin for:
# model checkpoints, XGBoost/CatBoost temporary files, PyTorch caches,
# submit.zip smoke tests, etc.
MIN_LOCAL_FREE_BYTES=$((6 * 1024 * 1024 * 1024))


# ============================================================
# Helpers
# ============================================================

free_bytes() {
    local path="$1"

    df -B1 --output=avail "${path}" \
        | tail -n 1 \
        | tr -d '[:space:]'
}


validate_data_dir() {
    local directory="$1"
    local filename

    for filename in \
        train.csv \
        test.csv \
        sample_submission.csv \
        trackman_history.csv
    do
        if [[ ! -s "${directory}/${filename}" ]]; then
            echo "[ERROR] Missing or empty dataset file:"
            echo "        ${directory}/${filename}"
            return 1
        fi
    done

    return 0
}


# ============================================================
# Choose workspace
# ============================================================

LOCAL_FREE_BYTES="$(free_bytes /local_datasets)"

if [[ ! "${LOCAL_FREE_BYTES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not determine /local_datasets free space."
    exit 60
fi

if (( LOCAL_FREE_BYTES >= MIN_LOCAL_FREE_BYTES )); then
    WORK_MODE="node-local"
    JOB_ROOT="${LOCAL_WORK_BASE}/job-${JOB_ID}"
else
    WORK_MODE="shared-fallback"
    JOB_ROOT="${SHARED_WORK_BASE}/job-${JOB_ID}"
fi

WORK_PROJECT="${JOB_ROOT}/project"


# ============================================================
# Diagnostics preservation / cleanup
# ============================================================

preserve_diagnostics() {
    local report_path
    local diagnostic_dir

    report_path="${WORK_PROJECT}/submission_build/model/validation_report.json"

    if [[ -f "${report_path}" ]]; then
        diagnostic_dir="${PROJECT_DIR}/artifacts/job-${JOB_ID}/submission_build/model"

        mkdir -p "${diagnostic_dir}"

        cp -- \
            "${report_path}" \
            "${diagnostic_dir}/validation_report.json"

        echo \
            "[DIAGNOSTIC] Preserved " \
            "${diagnostic_dir}/validation_report.json"
    fi
}


cleanup() {
    local exit_code=$?

    # Prevent recursive EXIT trap handling.
    trap - EXIT

    preserve_diagnostics || true

    if [[ -n "${JOB_ROOT:-}" ]] && {
        [[ "${JOB_ROOT}" == "${LOCAL_WORK_BASE}"/job-* ]] ||
        [[ "${JOB_ROOT}" == "${SHARED_WORK_BASE}"/job-* ]]
    }; then
        if [[ -d "${JOB_ROOT}" ]]; then
            echo "[CLEANUP] Removing job workspace: ${JOB_ROOT}"
            rm -rf -- "${JOB_ROOT}"
        fi
    else
        echo \
            "[WARN] Refusing cleanup for unexpected JOB_ROOT=" \
            "${JOB_ROOT:-unset}"
    fi

    exit "${exit_code}"
}

trap cleanup EXIT


# ============================================================
# Job metadata
# ============================================================

echo "============================================================"
echo "[JOB] id=${JOB_ID}"
echo "[JOB] host=$(hostname)"
echo "[JOB] started=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[JOB] submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
echo "[JOB] project=${PROJECT_DIR}"
echo "[JOB] work_mode=${WORK_MODE}"
echo "[JOB] job_root=${JOB_ROOT}"
echo "============================================================"

if [[ "$(hostname -s)" == *-master ]]; then
    echo "[ERROR] Training must not run on moana-master."
    echo "[ERROR] Submit with sbatch."
    exit 1
fi


mkdir -p \
    "${PROJECT_DIR}/logs" \
    "${PROJECT_DIR}/artifacts" \
    "${PROJECT_DIR}/dist" \
    "${JOB_ROOT}" \
    "${WORK_PROJECT}" \
    "${JOB_ROOT}/tmp" \
    "${JOB_ROOT}/cache"


# ============================================================
# Runtime temporary directories
#
# If local disk is nearly full, these automatically live on /data.
# ============================================================

export TMPDIR="${JOB_ROOT}/tmp"
export XDG_CACHE_HOME="${JOB_ROOT}/cache"
export JOBLIB_TEMP_FOLDER="${JOB_ROOT}/tmp"

export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"

export TOKENIZERS_PARALLELISM=false


# ============================================================
# Remove accidental CUDA stub path
# ============================================================

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    clean_ld_library_path=""

    IFS=':' read -r -a ld_paths <<< "${LD_LIBRARY_PATH}"

    for ld_path in "${ld_paths[@]}"; do
        [[ -z "${ld_path}" ]] && continue

        if [[ "${ld_path}" == */stubs || "${ld_path}" == */stubs/ ]]; then
            echo \
                "[ENV] Removing CUDA stub path from LD_LIBRARY_PATH: " \
                "${ld_path}"
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


# ============================================================
# Conda environment
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


# ============================================================
# Package validation
# ============================================================

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

if torch.version.cuda is None:
    print(
        "[ERROR] Installed PyTorch is CPU-only.",
        file=sys.stderr,
    )
    raise SystemExit(65)
PY
then
    echo "[ERROR] Required Python runtime packages are unavailable."
    exit 65
fi


# ============================================================
# GPU diagnostics
# ============================================================

echo "[RESOURCE] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

GPU_READY=1

if ! nvidia-smi; then
    GPU_READY=0

    echo \
        "[ERROR] nvidia-smi cannot access the allocated GPU on " \
        "$(hostname -s)."
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


# ============================================================
# CUDA Driver API probe
# ============================================================

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


# ============================================================
# Functional PyTorch CUDA probe
# ============================================================

if [[ "${GPU_READY}" -eq 1 ]]; then
    echo "[GPU-PROBE] Testing PyTorch CUDA forward/backward"

    if ! python - <<'PY'
import socket
import sys

import torch

print(f"[GPU-PROBE] torch={torch.__version__}")
print(f"[GPU-PROBE] torch_cuda_build={torch.version.cuda}")

if not torch.cuda.is_available():
    print(
        "[ERROR] torch.cuda.is_available() is false. "
        f"host={socket.gethostname()}",
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
        "PyTorch CUDA forward produced non-finite output."
    )

y.backward()

if x.grad is None:
    raise RuntimeError(
        "PyTorch CUDA backward produced no gradient."
    )

if not torch.isfinite(x.grad).all():
    raise RuntimeError(
        "PyTorch CUDA backward produced non-finite gradients."
    )

torch.cuda.synchronize()

peak_mb = (
    torch.cuda.max_memory_allocated()
    / (1024 ** 2)
)

print(
    "[GPU-PROBE] PASS "
    f"device={torch.cuda.get_device_name(0)} "
    f"peak_allocated={peak_mb:.1f}MiB"
)
PY
    then
        GPU_READY=0
        echo "[ERROR] PyTorch CUDA compute probe failed."
    fi
fi


if [[ "${GPU_READY}" -ne 1 ]]; then
    echo "[ERROR] No valid CUDA backend; refusing neural training."
    exit 73
fi

echo "[GPU] Functional CUDA backend confirmed."
echo "[MODE] xgb+lgb+cat+resnet+ft_transformer temporal_v9"


# ============================================================
# Storage diagnostics
# ============================================================

echo "[RESOURCE] Memory"
free -h

echo "[RESOURCE] Filesystems"
df -h /local_datasets /data || true

echo \
    "[RESOURCE] local_free_bytes=" \
    "${LOCAL_FREE_BYTES}"

echo \
    "[RESOURCE] workspace_mode=" \
    "${WORK_MODE}"

if [[ -r /sys/fs/cgroup/memory.max ]]; then
    echo \
        "[RESOURCE] cgroup memory.max=" \
        "$(cat /sys/fs/cgroup/memory.max)"
elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    echo \
        "[RESOURCE] cgroup memory.limit_in_bytes=" \
        "$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
fi


# ============================================================
# Submission preflight
# ============================================================

echo "[PREFLIGHT] Validating DACON submission requirements"

(
    cd "${PROJECT_DIR}"

    python - <<'PY'
from scripts.build_submit import (
    SUBMISSION_DIR,
    _validate_submission_requirements,
)

path = SUBMISSION_DIR / "requirements.txt"

_validate_submission_requirements(path)

print(
    f"[PREFLIGHT] PASS requirements={path}"
)
PY
)


echo "[PREFLIGHT] Exercising source contracts"

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

echo "[PREFLIGHT] PASS source contracts"


# ============================================================
# Stage source code
#
# Dataset is deliberately excluded.  It will be referenced through
# AIMERS_DATA_DIR rather than copied into every job workspace.
# ============================================================

echo \
    "[STAGE] Copying source code to " \
    "${WORK_PROJECT}"

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
    "${WORK_PROJECT}/"


# ============================================================
# Persistent dataset cache
# ============================================================

if [[ ! -s "${DATA_ARCHIVE}" ]]; then
    echo "[ERROR] Dataset archive not found: ${DATA_ARCHIVE}"
    exit 74
fi

ARCHIVE_SHA256="$(
    sha256sum "${DATA_ARCHIVE}" \
        | awk '{print $1}'
)"

dataset_cache_ready() {
    if [[ ! -s "${DATA_CACHE_ROOT}/.archive_sha256" ]]; then
        return 1
    fi

    if [[ "$(
        cat "${DATA_CACHE_ROOT}/.archive_sha256"
    )" != "${ARCHIVE_SHA256}" ]]; then
        return 1
    fi

    validate_data_dir "${DATA_CACHE_DIR}"
}


if ! dataset_cache_ready; then
    echo "[DATA-CACHE] Cache missing/stale."

    mkdir -p "$(dirname "${DATA_CACHE_ROOT}")"

    # Avoid two concurrent jobs rebuilding the same cache.
    if ! command -v flock >/dev/null 2>&1; then
        echo "[ERROR] flock is required for safe dataset cache creation."
        exit 75
    fi

    exec 9>"${DATA_CACHE_ROOT}.lock"
    flock 9

    # Another job may have created it while this job waited for the lock.
    if ! dataset_cache_ready; then
        TEMP_CACHE="${DATA_CACHE_ROOT}.tmp-${JOB_ID}"

        rm -rf -- "${TEMP_CACHE}"
        mkdir -p "${TEMP_CACHE}"

        UNCOMPRESSED_BYTES="$(
            unzip -l "${DATA_ARCHIVE}" \
            | awk '
                $4 == "data/train.csv" ||
                $4 == "data/test.csv" ||
                $4 == "data/sample_submission.csv" ||
                $4 == "data/trackman_history.csv"
                {
                    total += $1
                }
                END {
                    printf "%.0f\n", total
                }
            '
        )"

        CACHE_PARENT="$(
            dirname "${DATA_CACHE_ROOT}"
        )"

        SHARED_FREE_BYTES="$(
            free_bytes "${CACHE_PARENT}"
        )"

        CACHE_SAFETY_BYTES=$((1024 * 1024 * 1024))

        REQUIRED_CACHE_BYTES=$(
            (
                UNCOMPRESSED_BYTES
                + CACHE_SAFETY_BYTES
            )
        )

        echo \
            "[DATA-CACHE] uncompressed_bytes=" \
            "${UNCOMPRESSED_BYTES}"

        echo \
            "[DATA-CACHE] shared_free_bytes=" \
            "${SHARED_FREE_BYTES}"

        if (
            SHARED_FREE_BYTES
            < REQUIRED_CACHE_BYTES
        ); then
            echo \
                "[ERROR] Not enough /data space to create dataset cache."
            echo \
                "[ERROR] required=${REQUIRED_CACHE_BYTES} " \
                "available=${SHARED_FREE_BYTES}"
            exit 76
        fi

        echo \
            "[DATA-CACHE] Extracting official files directly from:" \
            "${DATA_ARCHIVE}"

        unzip -q \
            "${DATA_ARCHIVE}" \
            "data/train.csv" \
            "data/test.csv" \
            "data/sample_submission.csv" \
            "data/trackman_history.csv" \
            -d "${TEMP_CACHE}"

        if ! validate_data_dir "${TEMP_CACHE}/data"; then
            echo "[ERROR] Extracted dataset cache is incomplete."
            exit 77
        fi

        printf '%s\n' \
            "${ARCHIVE_SHA256}" \
            > "${TEMP_CACHE}/.archive_sha256"

        # Replace only after the new cache is complete.
        rm -rf -- "${DATA_CACHE_ROOT}"
        mv -- \
            "${TEMP_CACHE}" \
            "${DATA_CACHE_ROOT}"

        echo "[DATA-CACHE] Cache created successfully."
    else
        echo "[DATA-CACHE] Cache was created by another job."
    fi

    flock -u 9
    exec 9>&-
else
    echo "[DATA-CACHE] Reusing validated persistent cache."
fi


if ! validate_data_dir "${DATA_CACHE_DIR}"; then
    echo "[ERROR] Final dataset cache validation failed."
    exit 78
fi

export AIMERS_DATA_DIR="${DATA_CACHE_DIR}"

echo "[DATA] AIMERS_DATA_DIR=${AIMERS_DATA_DIR}"

for filename in \
    train.csv \
    test.csv \
    sample_submission.csv \
    trackman_history.csv
do
    data_path="${AIMERS_DATA_DIR}/${filename}"

    echo \
        "[DATA] " \
        "$(du -h "${data_path}")"
done


# ============================================================
# Compatibility symlink
#
# Main code uses AIMERS_DATA_DIR, but keep baseline/data available for
# any utility still using the repository-default path.
# ============================================================

mkdir -p "${WORK_PROJECT}/baseline"

rm -rf -- "${WORK_PROJECT}/baseline/data"

ln -s \
    "${AIMERS_DATA_DIR}" \
    "${WORK_PROJECT}/baseline/data"


# ============================================================
# Train
# ============================================================

cd "${WORK_PROJECT}"

echo "[STAGE] Working directory: $(pwd)"
echo "[STAGE] workspace_mode=${WORK_MODE}"
echo "[STAGE] data_dir=${AIMERS_DATA_DIR}"

TRAIN_ARGS=(
    "--clean"
    "--cat-task-type" "GPU"
    "--xgb-device" "cuda"
    "--nn-device" "cuda"
)

echo \
    "[TRAIN] XGBoost=CUDA, LightGBM=CPU, CatBoost=GPU, " \
    "ResNet/FT-Transformer=CUDA"

echo \
    "[TRAIN] command: python scripts/train_submit.py " \
    "${TRAIN_ARGS[*]}"

python scripts/train_submit.py \
    "${TRAIN_ARGS[@]}"


# ============================================================
# Package
# ============================================================

echo "[BUILD] Building submit.zip"

python scripts/build_submit.py

SUBMIT_PATH="${WORK_PROJECT}/dist/submit.zip"

if [[ ! -s "${SUBMIT_PATH}" ]]; then
    echo "[ERROR] submit.zip was not created: ${SUBMIT_PATH}"
    exit 79
fi


# ============================================================
# Preserve outputs
# ============================================================

RESULT_DIR="${PROJECT_DIR}/artifacts/job-${JOB_ID}"

mkdir -p \
    "${RESULT_DIR}" \
    "${PROJECT_DIR}/dist"

cp -- \
    "${SUBMIT_PATH}" \
    "${PROJECT_DIR}/artifacts/submit-${JOB_ID}.zip"

cp -- \
    "${SUBMIT_PATH}" \
    "${PROJECT_DIR}/dist/submit.zip"

rsync -a \
    "${WORK_PROJECT}/submission_build/" \
    "${RESULT_DIR}/submission_build/"


sha256sum \
    "${PROJECT_DIR}/artifacts/submit-${JOB_ID}.zip"

echo "============================================================"
echo "[DONE] finished=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[DONE] workspace_mode=${WORK_MODE}"
echo "[DONE] submit=${PROJECT_DIR}/artifacts/submit-${JOB_ID}.zip"
echo "[DONE] latest=${PROJECT_DIR}/dist/submit.zip"
echo "[DONE] report=${RESULT_DIR}/submission_build/model/validation_report.json"
echo "============================================================"