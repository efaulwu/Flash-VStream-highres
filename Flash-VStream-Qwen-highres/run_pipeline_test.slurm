#!/bin/bash
#SBATCH --job-name=flashvstream_qwen_test
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

# ── 环境初始化 ──────────────────────────────────────────────────
source ~/.bashrc
conda activate vstream

# ── 工作目录 ────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"

# ── 模型路径（请根据实际情况修改）──────────────────────────────
MODEL_PATH="/path/to/Qwen2-VL-7B-Instruct"   # 替换为实际模型路径
VIDEO_PATH="/path/to/test_video.mp4"           # 替换为实际视频路径
OUTPUT_DIR="outputs/run_${SLURM_JOB_ID}"

mkdir -p "$OUTPUT_DIR" logs

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Video: $VIDEO_PATH"

# ── 启动测试脚本 ─────────────────────────────────────────────────
python pipeline_test_1gpu.py \
    --model_path "$MODEL_PATH" \
    --video_path "$VIDEO_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --flash_memory_mode ragged \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo "Done. Output at: $OUTPUT_DIR"
