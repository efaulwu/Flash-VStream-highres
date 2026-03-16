#!/bin/bash
#SBATCH --job-name=ragged-pipeline
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --time=0-12:00:00
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

# 1) retriever-only quick validation
singularity exec --nv -e --bind /scratch:/scratch --pwd "${WORK_DIR}" "${SIF_PATH}" \
  python pipeline_test_ragged_streaming_dummy.py \
  --mode retriever-only \
  --num_chunks 10 \
  --budget_target 11520 \
  --budget_hard 12000 \
  --r_t 0.3333 \
  --num_anchors 8 \
  --topM_items 10 \
  --log_level INFO

# 2) model-wrapped pipeline validation
singularity exec --nv -e --bind /scratch:/scratch --pwd "${WORK_DIR}" "${SIF_PATH}" \
  python pipeline_test_ragged_streaming_dummy.py \
  --mode model-wrapped \
  --num_chunks 10 \
  --frames_per_chunk 8 \
  --budget_target 11520 \
  --budget_hard 12000 \
  --r_t 0.3333 \
  --num_anchors 8 \
  --topM_items 10 \
  --log_level INFO

echo "[INFO] Job finished successfully."
