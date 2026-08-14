#!/bin/bash
#SBATCH --job-name=train_v6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH --exclude=moana-y3
#SBATCH -o /data/yuse57625/lg_aimers/logs/slurm-%A.out

echo "Starting job $SLURM_JOB_ID"

# 1. NAS 부하 방지를 위한 로컬 데이터셋 디렉토리 설정 및 데이터 복사
LOCAL_DATA_DIR="/local_datasets/${USER}_${SLURM_JOB_ID}"
echo "Creating local dataset directory at: $LOCAL_DATA_DIR"
mkdir -p "$LOCAL_DATA_DIR"

echo "Copying data from NAS to local dataset directory..."
# /data/yuse57625/lg_aimers/baseline/open 하위에 data 폴더가 있다고 가정
cp -r /data/yuse57625/lg_aimers/baseline/open/data "$LOCAL_DATA_DIR/"

# 환경 변수로 DATA_DIR 전달하여 python 스크립트 내에서 경로 인식
export DATA_DIR="$LOCAL_DATA_DIR"

# 2. 작업 디렉토리 이동 및 학습 실행
cd /data/yuse57625/repos/aimers
echo "Running train_v6.py..."

# conda 환경이나 가상환경이 있다면 여기서 activate 합니다. (기본 환경 사용)
source /data/yuse57625/anaconda3/etc/profile.d/conda.sh
conda activate aimers_gpu
python -u src/train_v6.py

# 3. 작업 완료 후 로컬 데이터셋 정리 (필수 규정)
echo "Cleaning up local dataset directory..."
rm -rf "$LOCAL_DATA_DIR"

echo "Job finished successfully."
