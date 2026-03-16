#!/bin/bash
#SBATCH --job-name=ragged-fillup-vis
#SBATCH --output=ragged-fillup-vis-%j.out
#SBATCH --error=ragged-fillup-vis-%j.err
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
set -x
PS4='[${BASH_SOURCE##*/}:${LINENO}] '
trap 'rc=$?; echo "[ERROR] line=${LINENO} exit_code=${rc}"; exit ${rc}' ERR

REPO=/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres
SIF=/scratch/zwu24/singularity_sifs/qwen.sif
MODEL_PATH=/scratch/zwu24/hfmodels/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac
TRAIN_JSON=${REPO}/data/train_test_1fps_8_vis.json
FRAME_ROOT=/scratch/zwu24/datasets/LLaVA-Video-178K_fps1
OUT_DIR=${REPO}/output/ragged_fillup_vis_${SLURM_JOB_ID}
LOG_DIR=${REPO}/log
RUN_LOG=${LOG_DIR}/ragged_fillup_vis_${SLURM_JOB_ID}.log

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
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

singularity exec --nv -e --bind /scratch:/scratch --pwd "${REPO}" "${SIF}" bash -lc "
set -euo pipefail
set -x
PS4='[container:${BASH_SOURCE##*/}:${LINENO}] '
trap 'rc=\$?; echo \"[CONTAINER-ERROR] line=\${LINENO} exit_code=\${rc}\"; exit \${rc}' ERR
export PYTHONUNBUFFERED=1
cd ${REPO}

python3 -V
python3 -u scripts/ragged_memory_present_visualize_fillup.py \
  --model-path ${MODEL_PATH} \
  --train-json ${TRAIN_JSON} \
  --frame-root ${FRAME_ROOT} \
  --out-dir ${OUT_DIR} \
  --sample-index 0 \
  --frames-per-chunk 8 \
  --max-frames 32 \
  --variant dynamic \
  --target-present-tokens 12000 \
  --device cuda
" 2>&1 | tee -a "${RUN_LOG}"

echo "[INFO] finished=$(date -Is) out_dir=${OUT_DIR}" | tee -a "${RUN_LOG}"
