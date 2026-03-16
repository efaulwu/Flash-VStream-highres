#!/bin/bash
#SBATCH --job-name=flashvstream-1fps-memsafe
#SBATCH --output=flashvstream-1fps-memsafe-%j.out
#SBATCH --error=flashvstream-1fps-memsafe-%j.err
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zwu24@gmu.edu

set -euo pipefail
set -x
PS4='[${BASH_SOURCE##*/}:${LINENO}] '
trap 'rc=$?; echo "[ERROR] line=${LINENO} exit_code=${rc}"; exit ${rc}' ERR

REPO=/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres
SIF=/scratch/zwu24/singularity_sifs/qwen.sif
MODEL_PATH=/scratch/zwu24/hfmodels/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac
DATA_JSON=${REPO}/data/train_full_1fps_all.json
FRAME_ROOT=/scratch/zwu24/datasets/LLaVA-Video-178K_fps1
OUT_DIR=${REPO}/output/train_test_1fps_memsafe_${SLURM_JOB_ID}
LOG_DIR=${REPO}/log
RUN_LOG=${LOG_DIR}/train_test_1fps_memsafe_${SLURM_JOB_ID}.log

mkdir -p "${LOG_DIR}"
echo "[INFO] job_id=${SLURM_JOB_ID} host=$(hostname) start=$(date -Is)" | tee -a "${RUN_LOG}"
echo "[INFO] repo=${REPO}" | tee -a "${RUN_LOG}"
echo "[INFO] out_dir=${OUT_DIR}" | tee -a "${RUN_LOG}"
echo "[INFO] model_path=${MODEL_PATH}" | tee -a "${RUN_LOG}"

module load singularity
module load cuda/12.2
module list 2>&1 | tee -a "${RUN_LOG}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee -a "${RUN_LOG}"
fi

singularity exec --nv ${SIF} bash -lc "
set -euo pipefail
set -x
PS4='[container:${BASH_SOURCE##*/}:${LINENO}] '
trap 'rc=\$?; echo \"[CONTAINER-ERROR] line=\${LINENO} exit_code=\${rc}\"; exit \${rc}' ERR
export PYTHONUNBUFFERED=1
cd ${REPO}

python3 -V
python3 -u scripts/prepare_train_test_1fps.py --output-json ${DATA_JSON} --max-samples 0
python3 -u pipeline_train_test_1fps.py --train-json ${DATA_JSON} --frame-root ${FRAME_ROOT} --max-frames 8
python3 -u scripts/ensure_peft.py

python3 -u finetune_flash_memsafe.py \
  --model_name_or_path ${MODEL_PATH} \
  --data_path ${DATA_JSON} \
  --video_path ${FRAME_ROOT} \
  --use_flash_attn True \
  --bf16 True \
  --output_dir ${OUT_DIR} \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy epoch \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.0 \
  --logging_steps 1 \
  --report_to none \
  --model_max_length 16384 \
  --fps 1.0 \
  --max_frames 4 \
  --lazy_preprocess True \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 16 \
  --gradient_checkpointing \
  --batch_size_shrink_factor 2 \
  --python_gc_interval 50 \
  --memlog_path ${OUT_DIR}/cpu_mem_trace_memsafe.jsonl \
  --memlog_interval_seconds 15 \
  --step_log_interval 10 \
  --dataset_cache_log_interval 50 \
  --dataset_cache_max_entries 0 \
  --dataset_cache_store_pixels False \
  --enable_tracemalloc False
" 2>&1 | tee -a "${RUN_LOG}"

echo "[INFO] finished=$(date -Is) output_dir=${OUT_DIR}" | tee -a "${RUN_LOG}"
