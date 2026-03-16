#!/bin/bash
#SBATCH --job-name=m2_mem_cluster_grid_div_mmr
#SBATCH --output=log/m2_mem_cluster_grid_div_mmr-%j.out
#SBATCH --error=log/m2_mem_cluster_grid_div_mmr-%j.err
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00

set -euo pipefail
set -x
PS4='[${BASH_SOURCE##*/}:${LINENO}] '
trap 'rc=$?; echo [ERROR] line=${LINENO} exit_code=${rc}; exit ${rc}' ERR

cd /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres
export SIF=/scratch/zwu24/singularity_sifs/qwen.sif
module load singularity || true
module load cuda/12.2 || true

if [ -n "${SIF:-}" ] && [ -f "${SIF}" ]; then
  singularity exec --nv -e --bind /scratch:/scratch --pwd "$PWD" "${SIF}" bash -lc "
  set -euo pipefail
  export PYTHONUNBUFFERED=1
  python3 \
  -u \
  scripts/memory_retrieval_ablation.py \
  --mode \
  run_experiment_list \
  --experiment_list \
  /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/output/memory_ablation_real10_maskviz_resstrat/ablation_plan.json \
  --run_experiment_name \
  mem_cluster_grid_div_mmr \
  --output_dir \
  /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/output/memory_ablation_real10_maskviz_resstrat \
  --save_vis \
  True \
  --max_videos \
  10 \
  --random_sample_videos \
  True \
  --resolution_stratified_sample \
  True \
  --resolution_probe_first_frame \
  True \
  --resolution_bucket_mode \
  short_side \
  --resolution_bin1 \
  360 \
  --resolution_bin2 \
  540 \
  --resolution_bin3 \
  720 \
  --max_frames_per_video \
  0 \
  --dataset_max_frames \
  8 \
  --raw_grid_h \
  28 \
  --raw_grid_w \
  28 \
  --raw_token_dim \
  256 \
  --topk \
  512 \
  --dataset_json \
  /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/data/train_test_1fps_8_vis.json \
  --frame_root \
  /scratch/zwu24/datasets/LLaVA-Video-178K_fps1
  "
else
  export PYTHONUNBUFFERED=1
  python3 \
  -u \
  scripts/memory_retrieval_ablation.py \
  --mode \
  run_experiment_list \
  --experiment_list \
  /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/output/memory_ablation_real10_maskviz_resstrat/ablation_plan.json \
  --run_experiment_name \
  mem_cluster_grid_div_mmr \
  --output_dir \
  /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/output/memory_ablation_real10_maskviz_resstrat \
  --save_vis \
  True \
  --max_videos \
  10 \
  --random_sample_videos \
  True \
  --resolution_stratified_sample \
  True \
  --resolution_probe_first_frame \
  True \
  --resolution_bucket_mode \
  short_side \
  --resolution_bin1 \
  360 \
  --resolution_bin2 \
  540 \
  --resolution_bin3 \
  720 \
  --max_frames_per_video \
  0 \
  --dataset_max_frames \
  8 \
  --raw_grid_h \
  28 \
  --raw_grid_w \
  28 \
  --raw_token_dim \
  256 \
  --topk \
  512 \
  --dataset_json \
  /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/data/train_test_1fps_8_vis.json \
  --frame_root \
  /scratch/zwu24/datasets/LLaVA-Video-178K_fps1
fi
