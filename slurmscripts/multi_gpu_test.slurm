#!/bin/bash
#SBATCH --job-name=ragged-pipeline
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --time=0-1:00:00
#SBATCH --output=/scratch/zwu24/Flash-VStream-highres/log/%x-%N-%j.out
#SBATCH --error=/scratch/zwu24/Flash-VStream-highres/log/%x-%N-%j.err

set -euo pipefail

module load cuda/12.2
module load singularity

SIF_PATH="/scratch/zwu24/singularity_sifs/qwen.sif"
WORK_DIR="/scratch/zwu24/Flash-VStream-highres"
LOG_DIR="${WORK_DIR}/log"

mkdir -p "${LOG_DIR}"

echo "[INFO] Hostname: $(hostname)"
echo "[INFO] Work dir: ${WORK_DIR}"
echo "[INFO] Container: ${SIF_PATH}"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID}"

nvidia-smi