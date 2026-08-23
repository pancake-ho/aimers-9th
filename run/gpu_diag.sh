#!/usr/bin/env bash
#SBATCH -p batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=4G
#SBATCH --exclude=moana-y4,moana-y5
#SBATCH -o /data/surt321/repos/aimers_9th/logs/gpu-diag-%j.out

set -euo pipefail

source /data/$USER/anaconda3/etc/profile.d/conda.sh
conda activate aimers

echo "=================================================="
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "=================================================="

echo
echo "===== NVIDIA ====="
nvidia-smi

echo
echo "===== BEFORE ====="
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

ldconfig -p | grep -E 'libcuda\.so' || true

python - <<'PY'
import ctypes
import torch

print("torch =", torch.__version__)
print("torch CUDA =", torch.version.cuda)
print("torch available before =", torch.cuda.is_available())

try:
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    print("cuInit before =", cuda.cuInit(0))
except Exception as exc:
    print("libcuda before ERROR =", repr(exc))
PY

echo
echo "===== AFTER unset LD_LIBRARY_PATH ====="

unset LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

python - <<'PY'
import ctypes
import torch

cuda = ctypes.CDLL("libcuda.so.1")
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int

result = cuda.cuInit(0)

print("cuInit after =", result)
print("torch available after =", torch.cuda.is_available())
print("device count =", torch.cuda.device_count())

if result != 0:
    raise SystemExit(
        f"CUDA Driver API still failed after unset: cuInit={result}"
    )

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is still False")

print("GPU =", torch.cuda.get_device_name(0))

x = torch.randn(
    1024,
    1024,
    device="cuda",
)

y = x @ x
torch.cuda.synchronize()

print(
    "CUDA COMPUTE PASS",
    float(y.mean()),
)
PY

echo
echo "===== PASS ====="
