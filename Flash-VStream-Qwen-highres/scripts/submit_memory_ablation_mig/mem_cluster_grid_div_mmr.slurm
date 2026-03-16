#!/bin/bash
#SBATCH --job-name=mem_cluster_grid_div_mmr
#SBATCH --output=/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/log/mem_cluster_grid_div_mmr-%j.out
#SBATCH --error=/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/log/mem_cluster_grid_div_mmr-%j.err
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00

set -euo pipefail
set -x
PS4='[${BASH_SOURCE##*/}:${LINENO}] '
trap 'rc=$?; echo "[ERROR] line=${LINENO} exit_code=${rc}"; exit ${rc}' ERR

REPO=/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres
SIF=/scratch/zwu24/singularity_sifs/qwen.sif
LOG_DIR=${REPO}/log
RUN_LOG=${LOG_DIR}/mem_cluster_grid_div_mmr_${SLURM_JOB_ID}.log
mkdir -p "${LOG_DIR}"

echo "[INFO] job_id=${SLURM_JOB_ID} host=$(hostname) start=$(date -Is)" | tee -a "${RUN_LOG}"
module load singularity
module load cuda/12.2
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi | tee -a "${RUN_LOG}"; fi

singularity exec --nv ${SIF} bash -lc "
set -euo pipefail
export PYTHONUNBUFFERED=1
cd ${REPO}
python3 -u scripts/memory_retrieval_ablation.py --mode run --experiment_name mem_cluster_grid_div_mmr --utility_method norm --clustering_method grid --diversity_method mmr --coverage_method none --dataset_json /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/data/train_test_1fps_8_vis.json --frame_root /scratch/zwu24/datasets/LLaVA-Video-178K_fps1 --max_videos 10 --random_sample_videos true --dataset_max_frames 8 --topk 512 --seed 42 --save_vis true --output_dir /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/output/memory_ablation_real10
" 2>&1 | tee -a "${RUN_LOG}"

echo "[INFO] finished=$(date -Is)" | tee -a "${RUN_LOG}"
