# Flash-VStream-highres Agent Handoff (2026-03-09)

## Scope
This document is written for the next coding agent, not for end users.
Primary objective was to build and debug a 1fps training test pipeline on `Flash-VStream-Qwen-highres`, while ensuring the model path uses `models/ragged_flash_memory_retriever.py`.

## Repositories and Key Paths
- Main repo: `/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres`
- Dataset root: `/scratch/zwu24/datasets`
- Selected QA pairs input: `/scratch/zwu24/datasets/LLaVA-Video-178K/selected_10k_pairs_seed42.json`
- 1fps frame root: `/scratch/zwu24/datasets/LLaVA-Video-178K_fps1`
- Singularity image: `/scratch/zwu24/singularity_sifs/qwen.sif`
- Local model snapshot used in tests: `/scratch/zwu24/hfmodels/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac`

## What Was Added
- `Flash-VStream-Qwen-highres/scripts/prepare_train_test_1fps.py`
- `Flash-VStream-Qwen-highres/pipeline_train_test_1fps.py`
- `Flash-VStream-Qwen-highres/scripts/ensure_peft.py`
- `Flash-VStream-Qwen-highres/scripts/run_train_test_1fps_in_container.sh`
- `Flash-VStream-Qwen-highres/scripts/submit_train_test_1fps.slurm`

## What Was Modified
- `Flash-VStream-Qwen-highres/finetune_flash.py`
- `Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime.py`

## Technical Changes in `finetune_flash.py`
- Switched model import backend to realtime variant:
  - now imports `FlashVStreamQwen2VLModel` and `FlashVStreamQwen2VLConfig` from `models.vstream_qwen2vl_realtime`.
  - anchor line (current): `finetune_flash.py:8`
- Added PEFT optional import guard:
  - `_PEFT_AVAILABLE` flag and explicit runtime error when `--use_lora` is requested without PEFT.
  - anchors: `finetune_flash.py:46`, `finetune_flash.py:52`, `finetune_flash.py:607`
- Added robust video/frame input resolver for dataset paths:
  - supports both video files and extracted-frame directories.
  - converts `*.mp4` path to frame directory candidates and scans frames safely.
  - anchor: `finetune_flash.py:233`
- Fixed single-GPU trainer compatibility:
  - `model.module.prepare_inputs_for_training(...)` changed to DDP/non-DDP safe variant.
  - anchor: `finetune_flash.py:508`
- Added retriever backend assertion to enforce ragged retriever usage:
  - checks `model.ragged_retriever`; if absent, checks `model.visual.flash_memory.ragged_retriever` and `model.model.visual.flash_memory.ragged_retriever`.
  - logs active backend class when found.
  - anchors: `finetune_flash.py:581`, `finetune_flash.py:586`

## Technical Change in `models/vstream_qwen2vl_realtime.py`
- Fixed signature mismatch causing runtime `TypeError`:
  - changed
    - `def temporal_compress(self, x, thw, temporal_length, temporal_weights, temporal_indices)`
  - to
    - `def temporal_compress(self, x, thw, temporal_length, temporal_weights=None, temporal_indices=None)`
  - anchor: `models/vstream_qwen2vl_realtime.py:150`

## Added Script Behavior Details

### `scripts/prepare_train_test_1fps.py`
- Input: selected QA pairs JSON.
- Deduplicates by `video`.
- Keeps only entries with non-empty extracted frame dirs in fps1 root.
- Writes a small training JSON for smoke tests.
- Typical outputs used during debugging:
  - `data/train_test_1fps_8.json`
  - `data/train_test_1fps_2.json`
  - `data/train_test_1fps_1.json`

### `pipeline_train_test_1fps.py`
- Hard-checks retriever source file path via `inspect.getsourcefile(RaggedFlashMemoryRetriever)`.
- Expected substring: `ragged_flash_memory_retriever.py`.
- Loads first sample frames and verifies streaming chunk builders:
  - `fullres` and `dynamic` variants from `finetune_streaming_variants.py`.

### `scripts/ensure_peft.py`
- Checks for `peft` module.
- Installs with `pip --user` if missing.

### `scripts/run_train_test_1fps_in_container.sh`
- Meant for direct execution inside Singularity runtime.
- Steps:
  - torch cuda check
  - ensure peft
  - prepare 1-sample JSON
  - pipeline test
  - 1-step training smoke run (`max_steps=1`, `num_train_epochs=1`)
- Current key training args in this script:
  - `--model_max_length 4096`
  - `--max_frames 4`
  - `--use_lora --lora_r 8 --lora_alpha 8`

### `scripts/submit_train_test_1fps.slurm` (CURRENT CONTENT CHANGED BY USER/TOOLS)
- Current scheduler params (as last read):
  - `#SBATCH --qos=gpu`
  - `#SBATCH --partition=gpuq`
  - `#SBATCH --gres=gpu:A100.80gb:1`
  - no explicit `--account` in current version.
- Email notifications now enabled:
  - `#SBATCH --mail-type=END,FAIL`
  - `#SBATCH --mail-user=zwu24@gmu.edu`
- In-script flow:
  - prepare 8-sample JSON
  - run pipeline test with `--max-frames 8` (was 16)
  - ensure peft
  - run `finetune_flash.py` with 1 epoch (no max_steps), `model_max_length=16384`, `max_frames=4`, plus `--batch_size_shrink_factor 2` and `--python_gc_interval 50`

### Memory Mitigation Additions (2026-03-09)
- `finetune_flash.py`
  - Adds `TrainingArguments.batch_size_shrink_factor` (default `1`). The script halves `per_device_train_batch_size` when possible; if already at `1`, it halves `gradient_accumulation_steps`. Rank-0 logging clarifies when no further shrink is possible.
  - Adds `TrainingArguments.python_gc_interval` (default `0`). When >0, a new `MemoryCleanupCallback` triggers `gc.collect()` and `torch.cuda.empty_cache()` every N optimizer steps to cap host RAM.
- Pipeline adjustments for lower per-sample memory:
  - `submit_train_test_1fps.slurm` pipeline check now limits to 8 frames per clip (`--max-frames 8`).
  - The actual finetune invocation trims `--max_frames` to 4 to halve decoded frame payload.
  - `scripts/run_train_test_1fps_in_container.sh` mirrors the 4-frame limit and forwards the new CLI flags for smoke runs.
  - `scripts/train_and_eval.sh` also passes `--batch_size_shrink_factor 2` and `--python_gc_interval 50` so multi-GPU runs inherit the safety nets.

### Memory Debug Runner (2026-03-10)
- Added `Flash-VStream-Qwen-highres/finetune_flash_memdebug.py`.
  - Wraps the standard finetune flow but streams periodic CPU snapshots (`rss`, `vms`, `cpu_percent`, system availability) to JSONL.
  - Stage markers capture memory deltas for tokenizer/model/dataset/trainer creation, plus cache growth inside `LazySupervisedDataset`.
  - CLI extras (parsed via `MemoryDebugArguments`):
    - `--memlog_path` (default: `<output_dir>/cpu_memory_trace.jsonl`)
    - `--memlog_interval_seconds` (default `30s`)
    - `--step_log_interval` (default `50` global steps)
    - `--enable_tracemalloc/--tracemalloc_topk` for optional top allocator dumps.
    - `--dataset_cache_log_interval` controls how frequently cache growth events are logged.
  - To run inside the existing Slurm recipe, swap the train call:
    ```bash
    python3 -u finetune_flash_memdebug.py \
      --model_name_or_path ... (same args as finetune_flash.py) \
      --memlog_path ${OUT_DIR}/cpu_mem_trace.jsonl \
      --memlog_interval_seconds 15 \
      --step_log_interval 10
    ```
  - Look for `dataset_cache_growth` events to pinpoint when LazySupervisedDataset caches enough samples to saturate host RAM.

## Debugging Timeline and Root Causes
- Early failures were mostly environment/scheduler related:
  - wrong partition/qos/account combinations (`Invalid qos specification`, etc.)
  - shell quoting failures when passing long nested commands to `srun + singularity`
  - container missing modules (`torch`, `peft`) in non-singularity path
- Training runtime failures and fixes:
  - `model.module` access on single GPU.
  - retriever attachment lookup mismatch.
  - rope shape mismatch due too-small `model_max_length` in smoke run.
  - realtime model `temporal_compress` signature mismatch.

## Verified Successful Run (Important)
Executed inside an active GPU allocation (`job 6533411`) with:
- `module load singularity`
- `module load cuda/12.2`
- `singularity exec --nv /scratch/zwu24/singularity_sifs/qwen.sif ...`

Observed successful checkpoints in logs:
- `cuda True 1`
- `Ragged retriever source verified: /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/models/ragged_flash_memory_retriever.py`
- `[main] Ragged retriever backend active: RaggedFlashMemoryRetriever`
- training reached step completion and emitted:
  - `{'loss': ...}`
  - `{'train_runtime': ...}`
  - model save completed to `output/train_test_1fps_manual`

Current artifact path from successful smoke test:
- `/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/output/train_test_1fps_manual`

## Submitted Job State
- Submitted with `sbatch scripts/submit_train_test_1fps.slurm`
- Job ID: `6537861`
- Last known status: `PD (Priority)`
- Command used for status check:
  - `squeue -j 6533414 -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'`

## Repro Commands for Next Agent

### Fast sanity in existing GPU allocation
```bash
cd /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres
srun --jobid=<existing_gpu_jobid> bash -lc 'module load singularity && module load cuda/12.2 && singularity exec --nv /scratch/zwu24/singularity_sifs/qwen.sif /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/scripts/run_train_test_1fps_in_container.sh'
```

### Submit batch test
```bash
cd /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres
sbatch scripts/submit_train_test_1fps.slurm
squeue -j <jobid> -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'
```

## Open Risks and TODOs for Next Agent
- Scheduler policy drift:
  - `submit_train_test_1fps.slurm` was modified outside this agent cycle (user/tool). Re-validate account/qos/partition/GRES if queue rejects.
- Kernel warning observed (`4.18 < 5.5`):
  - may cause occasional hangs in long jobs; monitor early runtime.
- Flash-attn warnings:
  - model init currently emits warnings about initialization context; no blocker for completed smoke run, but can be cleaned later.
- Training scale-up validation pending:
  - only smoke run (`max_steps=1`) has been proven fully end-to-end.
  - full 1-epoch on 8 samples in batch script is configured, but still pending scheduler run (`6533414` still PD at last check).
- Optional cleanup:
  - if desired, consolidate duplicated debug JSONs (`train_test_1fps_1.json`, `2.json`, `8.json`) and keep one canonical file.

## Minimal Acceptance Criteria Already Met
- 1fps interface script exists and runs.
- pipeline test runs in singularity+GPU.
- training enters and completes 1 step.
- backend enforced to ragged retriever file (`models/ragged_flash_memory_retriever.py`).
- slurm submission script exists and job submitted.
