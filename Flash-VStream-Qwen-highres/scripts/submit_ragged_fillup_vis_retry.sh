#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/submit_ragged_fillup_vis.slurm"

if [[ ! -f "${SBATCH_FILE}" ]]; then
  echo "[FATAL] missing sbatch file: ${SBATCH_FILE}" >&2
  exit 1
fi

attempt=1
max_attempts="${MAX_ATTEMPTS:-3}"
queue_poll_secs="${QUEUE_POLL_SECS:-30}"
retry_sleep_secs="${RETRY_SLEEP_SECS:-10}"
while true; do
  if [[ "${attempt}" -gt "${max_attempts}" ]]; then
    echo "[FATAL] reached max attempts=${max_attempts}, stop retrying" >&2
    exit 2
  fi
  echo "[INFO] attempt=${attempt} submitting ${SBATCH_FILE}"
  submit_out="$(sbatch "${SBATCH_FILE}")"
  echo "[INFO] ${submit_out}"
  job_id="$(awk '{print $4}' <<<"${submit_out}")"
  if [[ -z "${job_id}" ]]; then
    echo "[ERROR] cannot parse job id from: ${submit_out}" >&2
    exit 1
  fi

  echo "[INFO] waiting job=${job_id} to leave queue"
  while squeue -j "${job_id}" -h >/dev/null 2>&1 && [[ -n "$(squeue -j "${job_id}" -h)" ]]; do
    squeue -j "${job_id}" -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" || true
    sleep "${queue_poll_secs}"
  done

  state_raw="$(sacct -j "${job_id}" --format=State --noheader | head -n 1 | xargs || true)"
  state="${state_raw%% *}"
  echo "[INFO] job=${job_id} state=${state:-UNKNOWN}"

  run_log="/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/log/ragged_fillup_vis_${job_id}.log"
  err_log="/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/ragged-fillup-vis-${job_id}.err"
  out_log="/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/ragged-fillup-vis-${job_id}.out"

  if [[ "${state}" == "COMPLETED" ]]; then
    echo "[SUCCESS] job=${job_id} completed"
    [[ -f "${run_log}" ]] && tail -n 80 "${run_log}" || true
    exit 0
  fi

  echo "[WARN] job=${job_id} failed with state=${state}; printing recent logs and retrying"
  [[ -f "${run_log}" ]] && tail -n 120 "${run_log}" || true
  [[ -f "${err_log}" ]] && tail -n 120 "${err_log}" || true
  [[ -f "${out_log}" ]] && tail -n 120 "${out_log}" || true

  attempt=$((attempt + 1))
  sleep "${retry_sleep_secs}"
done
