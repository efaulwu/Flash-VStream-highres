# 2026-03-14 CPU Memory Safe Pipeline

## Task Summary
Created a CPU-memory-safe training pipeline copy without modifying original source files.
The new pipeline is designed to prevent unbounded RAM growth from dataset sample caching.

## What Was Created
1. New training script copy:
   - `/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/finetune_flash_memsafe.py`
2. New Slurm submission script copy:
   - `/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/scripts/submit_train_test_1fps_memsafe.slurm`

## Safety Changes Applied (in the new copy only)
- Added memory-safe dataset cache controls to CLI:
  - `--dataset_cache_max_entries` (default `0`, means no in-memory sample cache)
  - `--dataset_cache_store_pixels` (default `False`, avoids caching heavy `pixel_values_videos`)
- Replaced unbounded `cached_data_dict` behavior in the patched lazy dataset path with:
  - optional bounded cache (LRU-like eviction when enabled)
  - detailed cache policy and eviction logging into JSONL memory log
- Kept original runtime/training hyperparameters aligned with the original script.

## Validation Performed
- Python syntax compile check:
  - `python3 -m py_compile finetune_flash_memsafe.py`
- Slurm script syntax check:
  - `bash -n scripts/submit_train_test_1fps_memsafe.slurm`
- Both checks passed.

## Slurm Submission
- Command used:
  - `sbatch scripts/submit_train_test_1fps_memsafe.slurm`
- Submitted job id:
  - `6559601`
- Initial queue status after submit:
  - `PENDING (Priority)`

## Runtime Logging
- Slurm stdout/stderr (as configured in new script):
  - `flashvstream-1fps-memsafe-6559601.out`
  - `flashvstream-1fps-memsafe-6559601.err`
- In-job run log:
  - `/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/log/train_test_1fps_memsafe_6559601.log`
- Memory trace JSONL expected path:
  - `/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/output/train_test_1fps_memsafe_6559601/cpu_mem_trace_memsafe.jsonl`

## Notes
- Original files were not modified.
- New run keeps `--lazy_preprocess True` but enforces memory-safe dataset caching behavior in the copied script.
