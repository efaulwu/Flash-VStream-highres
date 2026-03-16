#!/bin/bash
set -euo pipefail
set -x
PS4='[${BASH_SOURCE##*/}:${LINENO}] '
trap 'rc=$?; echo "[ERROR] line=${LINENO} exit_code=${rc}"; exit ${rc}' ERR
export PYTHONUNBUFFERED=1

REPO=/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres
MODEL_PATH=/scratch/zwu24/hfmodels/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac
FRAME_ROOT=/scratch/zwu24/datasets/LLaVA-Video-178K_fps1
DATA_JSON=${REPO}/data/train_test_1fps_8_vis.json
VIS_DIR=${REPO}/output/ragged_present_vis_${SLURM_JOB_ID:-manual}

cd "$REPO"

python3 -V
python3 -u scripts/ensure_peft.py

python3 -u scripts/prepare_train_test_1fps.py \
  --output-json "$DATA_JSON" \
  --max-samples 8

python3 -u scripts/ragged_memory_present_visualize.py \
  --model-path "$MODEL_PATH" \
  --train-json "$DATA_JSON" \
  --frame-root "$FRAME_ROOT" \
  --out-dir "$VIS_DIR" \
  --sample-index 0 \
  --frames-per-chunk 8 \
  --max-frames 32 \
  --variant dynamic \
  --target-present-tokens 11520 \
  --device cuda

echo "[INFO] short training skipped in memory-only mode"
echo "[INFO] ragged visualization output: $VIS_DIR"