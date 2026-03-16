# Memory Retrieval Ablation (MIG 40G)

This folder contains one Slurm script per experiment.

## Dataset mode
- dataset_json: /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/data/train_test_1fps_8_vis.json
- frame_root: /scratch/zwu24/datasets/LLaVA-Video-178K_fps1
- sampling: random 10 videos (`--max_videos 10 --random_sample_videos true`)

## Submit all
```bash
cd /scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres
for f in scripts/submit_memory_ablation_mig/*.slurm; do sbatch "$f"; done
```

## Output structure
- output root: `output/memory_ablation_real10`
- each experiment: `output/memory_ablation_real10/<experiment_name>/`
- visualizations: `.../visualizations/<video_id>/`
- logs: `.../logs/frame_logs.jsonl`
- metrics: `.../metrics/frame_metrics.json`, `.../metrics/summary_metrics.json`
