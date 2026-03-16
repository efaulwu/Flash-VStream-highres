#!/bin/bash
set -euo pipefail
set -x
PS4='[${BASH_SOURCE##*/}:${LINENO}] '
trap 'rc=$?; echo "[ERROR] line=${LINENO} exit_code=${rc}"; exit ${rc}' ERR
export PYTHONUNBUFFERED=1

REPO=/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres
cd "$REPO"

python3 -V
python3 -u -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
python3 -u scripts/ensure_peft.py
python3 -u scripts/prepare_train_test_1fps.py --max-samples 1 --output-json data/train_test_1fps_1.json
python3 -u pipeline_train_test_1fps.py \
  --train-json data/train_test_1fps_1.json \
  --frame-root /scratch/zwu24/datasets/LLaVA-Video-178K_fps1 \
  --max-frames 4

python3 -u finetune_flash.py \
  --model_name_or_path /scratch/zwu24/hfmodels/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac \
  --data_path data/train_test_1fps_1.json \
  --video_path /scratch/zwu24/datasets/LLaVA-Video-178K_fps1 \
  --use_flash_attn True \
  --bf16 True \
  --output_dir output/train_test_1fps_manual \
  --num_train_epochs 1 \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy no \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.0 \
  --logging_steps 1 \
  --report_to none \
  --model_max_length 4096 \
  --fps 1.0 \
  --max_frames 4 \
  --lazy_preprocess True \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 8 \
  --gradient_checkpointing \
  --batch_size_shrink_factor 2 \
  --python_gc_interval 50
