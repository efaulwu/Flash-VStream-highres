#!/bin/bash
#SBATCH --job-name=flashvstream-vis-memory
#SBATCH --output=flashvstream-vis-memory-%j.out
#SBATCH --error=flashvstream-vis-memory-%j.err
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
RUN_LOG=${LOG_DIR}/memory_only_ragged_vis_${SLURM_JOB_ID}.log

mkdir -p "${LOG_DIR}"
echo "[INFO] job_id=${SLURM_JOB_ID} host=$(hostname) start=$(date -Is)" | tee -a "${RUN_LOG}"
echo "[INFO] repo=${REPO}" | tee -a "${RUN_LOG}"
echo "[INFO] container=${SIF}" | tee -a "${RUN_LOG}"

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
bash scripts/run_short_train_and_ragged_vis_in_container.sh
" 2>&1 | tee -a "${RUN_LOG}"

echo "[INFO] finished=$(date -Is)" | tee -a "${RUN_LOG}"