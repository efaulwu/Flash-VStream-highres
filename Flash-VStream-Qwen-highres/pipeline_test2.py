#!/usr/bin/env python3
"""
Standalone streaming pipeline test for Flash-VStream-Qwen-highres (single GPU, with LLM).

Simulates streaming video input:
  - Multiple batches of dummy frames are sequentially processed
  - Visual encoder and flash-memory update for each batch
  - LLM receives visual features and a dummy prompt, generates output

Stages timed and memory-profiled:
  1. Model initialisation (visual encoder + LLM)
  2. Streaming input creation (multiple batches)
  3. Visual encoding + memory update per batch
  4. LLM inference per batch
  5. End-to-end summary

Usage:
    cd Flash-VStream-Qwen-highres
    python pipeline_test2.py [--model-path PATH] [--llm-path PATH] [--attn {sdpa,flash_attention_2,eager}]
"""
import os
import sys
import time
import warnings

import torch
import torch.nn.functional as F
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration
NUM_BATCHES = 4
BATCH_SIZE = 3
FRAME_H = 448
FRAME_W = 448
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
RESIZE_FACTOR = PATCH_SIZE * MERGE_SIZE
RESIZED_H = round(FRAME_H / RESIZE_FACTOR) * RESIZE_FACTOR
RESIZED_W = round(FRAME_W / RESIZE_FACTOR) * RESIZE_FACTOR
GRID_T = BATCH_SIZE // TEMPORAL_PATCH_SIZE
GRID_H = RESIZED_H // PATCH_SIZE
GRID_W = RESIZED_W // PATCH_SIZE
PATCH_DIM = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE
FLASH_MEMORY_CONFIG = dict(
    flash_memory_temporal_length=120,
    flash_memory_temporal_method='kmeans_ordered',
    flash_memory_temporal_poolsize=1,
    flash_memory_temporal_pca_dim=32,
    flash_memory_spatial_length=0,
    flash_memory_spatial_method='klarge_retrieve',
)
VISION_CONFIG_DEFAULTS = dict(
    depth=32,
    embed_dim=1280,
    hidden_size=1536,
    mlp_ratio=4,
    num_heads=16,
    in_channels=3,
    patch_size=PATCH_SIZE,
    spatial_merge_size=MERGE_SIZE,
    temporal_patch_size=TEMPORAL_PATCH_SIZE,
)

# Helpers
def gpu_memory_mb(device=0):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        return alloc, reserved
    return 0, 0

def log_gpu(tag: str, device=0):
    alloc, reserved = gpu_memory_mb(device)
    print(f"  [GPU mem] {tag:<50s}  alloc={alloc:8.1f} MB  reserved={reserved:8.1f} MB")

class StepTimer:
    def __init__(self):
        self.times = {}
        self.start_times = {}
    def start(self, name: str):
        torch.cuda.synchronize()
        self.start_times[name] = time.time()
    def stop(self, name: str) -> float:
        torch.cuda.synchronize()
        t = time.time() - self.start_times[name]
        self.times[name] = self.times.get(name, 0) + t
        return t
    def report(self):
        print("\nTiming summary:")
        for k, v in self.times.items():
            print(f"  {k:<30s}: {v:.3f} s")

def encode_visual_batch(
    visual_model,
    pixel_values: torch.Tensor,
    video_grid_thw: torch.Tensor,
    position_ids: torch.Tensor,
    visual_position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = pixel_values.view(-1, PATCH_DIM)
    hidden_states = visual_model.patch_embed(hidden_states)

    rotary_pos_emb = visual_model.rot_pos_emb(video_grid_thw)
    cu_seqlens = torch.repeat_interleave(
        video_grid_thw[:, 1] * video_grid_thw[:, 2],
        video_grid_thw[:, 0],
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for blk in visual_model.blocks:
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )

    video_embeds, updated_position_ids = visual_model.flash_memory(
        hidden_states,
        video_grid_thw,
        None,
        position_ids,
        visual_position_ids,
    )
    video_embeds = visual_model.merger(video_embeds)
    return video_embeds, updated_position_ids

# Main test
def run_pipeline_test2(args):
    device = torch.device("cuda:0")
    timer = StepTimer()
    print("=" * 70)
    print("Flash-VStream-Qwen-highres  │  pipeline_test2.py (streaming + LLM)")
    print("=" * 70)
    print(f"  Batches         : {NUM_BATCHES} × {BATCH_SIZE} frames")
    print(f"  Frame size      : {FRAME_H}×{FRAME_W} (resized to {RESIZED_H}×{RESIZED_W})")
    print(f"  Grid (T, H, W)  : {GRID_T}, {GRID_H}, {GRID_W}")
    print(f"  Flash-mem cfg   : {FLASH_MEMORY_CONFIG}")
    print(f"  Device          : {device}")
    print(f"  Attn impl       : {args.attn}")
    print()

    # Step 1: Model initialisation
    print("── Step 1: Model initialisation")
    log_gpu("before model init", device.index)
    timer.start("1. Model init")
    if args.model_path:
        from models import FlashVStreamQwen2VLConfig, FlashVStreamQwen2VLModel

        model_config = FlashVStreamQwen2VLConfig.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        model_config.vision_config.flash_memory_config = FLASH_MEMORY_CONFIG

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            visual_model = FlashVStreamQwen2VLModel.from_pretrained(
                args.model_path,
                config=model_config,
                device_map={"": device},
                torch_dtype=torch.bfloat16,
                attn_implementation=args.attn,
                trust_remote_code=True,
            ).visual
        visual_model = visual_model.eval()
        print(f"  Loaded weights from: {args.model_path}")
    else:
        from models.vstream_qwen2vl_model import FlashVStreamQwen2VisionTransformerPretrainedModel
        from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig

        vision_cfg = Qwen2VLVisionConfig(
            **VISION_CONFIG_DEFAULTS,
            flash_memory_config=FLASH_MEMORY_CONFIG,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            visual_model = FlashVStreamQwen2VisionTransformerPretrainedModel._from_config(
                vision_cfg,
                attn_implementation=args.attn,
            )
        visual_model = visual_model.eval()
        print("  Initialised visual encoder with RANDOM weights")
    visual_model = visual_model.to(device=device, dtype=torch.bfloat16)
    # LLM (dummy, replace with actual model if available)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.llm_path:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
        llm = AutoModelForCausalLM.from_pretrained(args.llm_path).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        llm = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    init_time = timer.stop("1. Model init")
    log_gpu("after  model init", device.index)
    print(f"  Visual-encoder parameters : {sum(p.numel() for p in visual_model.parameters()) / 1e6:.1f} M")
    print(f"  LLM parameters            : {sum(p.numel() for p in llm.parameters()) / 1e6:.1f} M")
    print(f"  Time                      : {init_time:.3f} s\n")

    # Step 2: Streaming input creation
    print("── Step 2: Streaming input creation")
    log_gpu("before input creation", device.index)
    timer.start("2. Streaming input creation")
    batches = []
    for batch_idx in range(NUM_BATCHES):
        pixel_values = torch.randn(
            GRID_T * GRID_H * GRID_W, PATCH_DIM,
            dtype=torch.bfloat16, device=device
        )
        batches.append(pixel_values)
    video_grid_thw = torch.tensor(
        [[GRID_T, GRID_H, GRID_W]], dtype=torch.long, device=device
    )
    tem_size = GRID_T * (GRID_H // 2) * (GRID_W // 2)
    position_ids = torch.zeros(3, 1, tem_size, dtype=torch.long, device=device)
    visual_position_ids = torch.arange(
        tem_size, dtype=torch.long, device=device
    ).unsqueeze(0)
    input_time = timer.stop("2. Streaming input creation")
    log_gpu("after  input creation", device.index)
    print(f"  Created {NUM_BATCHES} batches of dummy frames.")
    print(f"  Time                      : {input_time:.3f} s\n")

    # Step 3: Visual encoding + memory update per batch
    print("── Step 3: Visual encoding + memory update per batch")
    for batch_idx, pixel_values in enumerate(batches):
        log_gpu(f"before visual encoding (batch {batch_idx})", device.index)
        timer.start(f"3.{batch_idx}. Visual encoding")
        video_embeds, updated_position_ids = encode_visual_batch(
            visual_model,
            pixel_values,
            video_grid_thw,
            position_ids,
            visual_position_ids,
        )
        encode_time = timer.stop(f"3.{batch_idx}. Visual encoding")
        log_gpu(f"after  visual encoding (batch {batch_idx})", device.index)
        print(f"  Batch {batch_idx}: video embeds shape: {video_embeds.shape}")
        print(f"  Updated position ids shape: {updated_position_ids.shape}")
        print(f"  Time: {encode_time:.3f} s\n")

        # Step 4: LLM inference per batch
        print(f"── Step 4: LLM inference (batch {batch_idx})")
        timer.start(f"4.{batch_idx}. LLM inference")
        embed_mean = video_embeds.mean().item()
        dummy_prompt = (
            f"Describe the visual content of batch {batch_idx}. "
            f"Feature mean={embed_mean:.4f}."
        )
        # Simulate passing visual features to LLM (here, just text prompt)
        inputs = tokenizer(dummy_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = llm.generate(**inputs, max_new_tokens=20)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        llm_time = timer.stop(f"4.{batch_idx}. LLM inference")
        print(f"  Batch {batch_idx}: LLM output: {text}")
        print(f"  Time: {llm_time:.3f} s\n")

    timer.report()

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="zhang9302002/Flash-VStream-Qwen-7b", help="Path to visual encoder checkpoint")
    parser.add_argument("--llm-path", type=str, default=None, help="Path to LLM checkpoint")
    parser.add_argument("--attn", type=str, default="sdpa", help="Attention implementation")
    return parser.parse_args()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires at least one CUDA GPU.")
    args = parse_args()
    run_pipeline_test2(args)
