#!/usr/bin/env python3
"""
Standalone pipeline test for Flash-VStream-Qwen-highres (single GPU, no spatial pooling).

Tests the full visual-encoding pipeline on ONE GPU:
  - Visual encoder WITHOUT spatial pooling  (flash_memory_temporal_poolsize=1)
  - flash_memory_spatial_length=0            (required when temporal_poolsize=1,
                                              because there is no small-resolution
                                              pathway to drive spatial enhancement)
  - 120 dummy frames of 1200×1200 pixels
    → resized to 1204×1204 (nearest multiple of 28) for patch processing

Pipeline stages that are timed and memory-profiled:
  1. Model initialisation
  2. Dummy input creation  (raw pixel-value tensor)
  3. Patch-embedding        (PatchEmbed layer)
  4. Vision-transformer blocks (32 × Qwen2VLVisionBlock)
  5. Flash-memory           (temporal compression → PatchMerger)
  6. End-to-end summary

Usage:
    cd Flash-VStream-Qwen-highres
    python pipeline_test_1gpu.py [--model-path PATH] [--attn {sdpa,flash_attention_2,eager}]

    --model-path  (optional) path to a Qwen2-VL / FlashVStream checkpoint.
                  When omitted the model is initialised with RANDOM weights using
                  the standard Qwen2-VL-2B-Instruct vision-encoder architecture.
    --attn        attention implementation (default: sdpa).
                  Use flash_attention_2 if flash-attn is installed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import warnings

import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

NUM_FRAMES          = 10       # number of input frames
FRAME_H             = 448      # original frame height (pixels)
FRAME_W             = 448      # original frame width  (pixels)

# Qwen2-VL patch / merge constants
PATCH_SIZE          = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE          = 2          # spatial merge size (PatchMerger groups 2×2 patches)

# With temporal_poolsize=1, the resize factor = patch_size * merge_size * 1 = 28
# smart_resize(1200, 28) → round(1200/28)*28 = 43*28 = 1204
RESIZE_FACTOR       = PATCH_SIZE * MERGE_SIZE   # 28
RESIZED_H           = round(FRAME_H / RESIZE_FACTOR) * RESIZE_FACTOR   # 1204
RESIZED_W           = round(FRAME_W / RESIZE_FACTOR) * RESIZE_FACTOR   # 1204

GRID_T  = NUM_FRAMES  // TEMPORAL_PATCH_SIZE    # 60
GRID_H  = RESIZED_H   // PATCH_SIZE             # 86
GRID_W  = RESIZED_W   // PATCH_SIZE             # 86

PATCH_DIM = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE  # 1176

# Flash-memory config – NO spatial pooling
#   temporal_poolsize = 1  → vision transformer skips the "small-resolution pathway"
#   spatial_length    = 0  → spatial enhancement is disabled (it would fail otherwise,
#                             since it expects a 4× spatially down-sampled small_x)
FLASH_MEMORY_CONFIG = dict(
    flash_memory_temporal_length   = 120,
    flash_memory_temporal_method   = 'kmeans_ordered',
    flash_memory_temporal_poolsize = 1,    # ← KEY: no spatial pooling
    flash_memory_temporal_pca_dim  = 32,
    flash_memory_spatial_length    = 0,    # ← required companion setting
    flash_memory_spatial_method    = 'klarge_retrieve',
)

# Qwen2-VL-2B-Instruct vision-encoder architecture defaults
VISION_CONFIG_DEFAULTS = dict(
    depth         = 32,
    embed_dim     = 1280,
    hidden_size   = 1536,   # LLM hidden size (also used by PatchMerger output)
    mlp_ratio     = 4,
    num_heads     = 16,
    in_channels   = 3,
    patch_size    = PATCH_SIZE,
    spatial_merge_size    = MERGE_SIZE,
    temporal_patch_size   = TEMPORAL_PATCH_SIZE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def gpu_memory_mb(device=0):
    """Return (allocated_MB, reserved_MB) for *device*."""
    if torch.cuda.is_available():
        alloc   = torch.cuda.memory_allocated(device)  / 1024 ** 2
        reserved = torch.cuda.memory_reserved(device)  / 1024 ** 2
        return alloc, reserved
    return 0.0, 0.0


def log_gpu(tag: str, device=0):
    alloc, reserved = gpu_memory_mb(device)
    print(f"  [GPU mem] {tag:<50s}  alloc={alloc:8.1f} MB  reserved={reserved:8.1f} MB")


class StepTimer:
    """Accumulate per-step wall-clock times (GPU-synchronised)."""

    def __init__(self):
        self._steps = {}
        self._start  = {}

    def start(self, name: str):
        torch.cuda.synchronize()
        self._start[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start[name]
        self._steps[name] = elapsed
        return elapsed

    def report(self):
        total = sum(self._steps.values())
        print("\n" + "=" * 70)
        print(f"{'Step':<50s}  {'Time (s)':>10s}  {'% total':>8s}")
        print("-" * 70)
        for name, t in self._steps.items():
            pct = 100.0 * t / total if total > 0 else 0.0
            print(f"  {name:<48s}  {t:10.3f}  {pct:7.1f}%")
        print("-" * 70)
        print(f"  {'TOTAL':<48s}  {total:10.3f}")
        print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Main test
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline_test(args):
    device = torch.device("cuda:0")
    timer  = StepTimer()

    print("=" * 70)
    print("Flash-VStream-Qwen-highres  │  pipeline_test_1gpu.py")
    print("=" * 70)
    print(f"  Frames          : {NUM_FRAMES} × {FRAME_H}×{FRAME_W}  "
          f"(resized to {RESIZED_H}×{RESIZED_W} for processing)")
    print(f"  Grid (T, H, W)  : {GRID_T}, {GRID_H}, {GRID_W}")
    print(f"  Total patches   : {GRID_T * GRID_H * GRID_W:,}")
    print(f"  Visual tokens   : {GRID_T * (GRID_H // 2) * (GRID_W // 2):,}  "
          f"(after PatchMerger)")
    print(f"  Flash-mem cfg   : {FLASH_MEMORY_CONFIG}")
    print(f"  Device          : {device}")
    print(f"  Attn impl       : {args.attn}")
    print()

    # ── Step 1 : Model initialisation ─────────────────────────────────────────
    print("── Step 1: Model initialisation")
    log_gpu("before model init", device.index)
    timer.start("1. Model init")

    if args.model_path:
        # Load real weights
        from models import FlashVStreamQwen2VLConfig, FlashVStreamQwen2VLModel
        model_config = FlashVStreamQwen2VLConfig.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        model_config.vision_config.flash_memory_config = FLASH_MEMORY_CONFIG
        visual_model = FlashVStreamQwen2VLModel.from_pretrained(
            args.model_path,
            config=model_config,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn,
            trust_remote_code=True,
        ).visual.eval()
        print(f"  Loaded weights from: {args.model_path}")
    else:
        # Build from config with random weights (no checkpoint required)
        from models.vstream_qwen2vl_model import (
            FlashVStreamQwen2VisionTransformerPretrainedModel,
        )
        from transformers.models.qwen2_vl.configuration_qwen2_vl import (
            Qwen2VLVisionConfig,
        )
        vision_cfg = Qwen2VLVisionConfig(
            **VISION_CONFIG_DEFAULTS,
            flash_memory_config=FLASH_MEMORY_CONFIG,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            visual_model = (
                FlashVStreamQwen2VisionTransformerPretrainedModel._from_config(
                    vision_cfg, attn_implementation=args.attn
                )
                .to(device=device, dtype=torch.bfloat16)
                .eval()
            )
        print("  Initialised visual encoder with RANDOM weights "
              "(Qwen2-VL-2B architecture).")

    init_time = timer.stop("1. Model init")
    log_gpu("after  model init", device.index)
    param_count = sum(p.numel() for p in visual_model.parameters())
    print(f"  Visual-encoder parameters : {param_count / 1e6:.1f} M")
    print(f"  Time                      : {init_time:.3f} s\n")

    # ── Step 2 : Dummy input creation ─────────────────────────────────────────
    print("── Step 2: Dummy input creation")
    log_gpu("before input creation", device.index)
    timer.start("2. Dummy input creation")

    # pixel_values  – shape [T*H_g*W_g, C * t_patch * p * p]
    #   = [443 760, 1 176]  for 120 frames at 1204×1204
    # This is the format produced by FlashVStreamQwen2VLImageProcessor._preprocess
    pixel_values = torch.randn(
        GRID_T * GRID_H * GRID_W, PATCH_DIM,
        dtype=torch.bfloat16, device=device
    )
    video_grid_thw = torch.tensor(
        [[GRID_T, GRID_H, GRID_W]], dtype=torch.long, device=device
    )

    # Prepare position_ids and visual_position_ids for the flash-memory module.
    # FlashMemory stores temporal_length = flash_memory_temporal_length // 2 = 60
    # internally.  With GRID_T=60 == internal temporal_length=60, temporal_compress
    # returns features unchanged (no k-means compression needed).
    # No spatial enhancement since spatial_length=0.
    #   tem_size = GRID_T * (GRID_H//2) * (GRID_W//2) = 60*43*43 = 110 940
    tem_size = GRID_T * (GRID_H // 2) * (GRID_W // 2)
    position_ids       = torch.zeros(3, 1, tem_size, dtype=torch.long, device=device)
    visual_position_ids = torch.arange(
        tem_size, dtype=torch.long, device=device
    ).unsqueeze(0)   # [1, tem_size]

    input_time = timer.stop("2. Dummy input creation")
    log_gpu("after  input creation", device.index)
    print(f"  pixel_values shape         : {list(pixel_values.shape)}")
    print(f"  video_grid_thw             : {video_grid_thw.tolist()}")
    print(f"  Expected visual tokens     : {tem_size:,}")
    print(f"  Time                       : {input_time:.3f} s\n")

    # ── Step 3 : PatchEmbed ────────────────────────────────────────────────────
    print("── Step 3: PatchEmbed  (+ small-resolution pathway check)")
    log_gpu("before PatchEmbed", device.index)

    with torch.inference_mode():
        timer.start("3. PatchEmbed")

        # Replicate the first part of vision-transformer forward:
        hidden_states = pixel_values.view(-1, PATCH_DIM)

        # temporal_poolsize=1  →  NO small-resolution pathway
        assert visual_model.flash_memory.temporal_poolsize == 1, \
            "Expected temporal_poolsize=1 for no-spatial-pooling test!"
        small_hidden_states, small_grid_thw = None, None
        total_grid_thw = video_grid_thw

        hidden_states = visual_model.patch_embed(hidden_states)
        embed_time = timer.stop("3. PatchEmbed")

    log_gpu("after  PatchEmbed", device.index)
    print(f"  hidden_states shape        : {list(hidden_states.shape)}")
    print(f"  Time                       : {embed_time:.3f} s\n")

    # ── Step 4 : Vision-transformer blocks ────────────────────────────────────
    print(f"── Step 4: Vision-transformer blocks  ({len(visual_model.blocks)} blocks)")
    log_gpu("before transformer blocks", device.index)

    with torch.inference_mode():
        # Build rotary position embeddings over total_grid_thw
        rotary_pos_emb = visual_model.rot_pos_emb(total_grid_thw)

        cu_seqlens = torch.repeat_interleave(
            total_grid_thw[:, 1] * total_grid_thw[:, 2],
            total_grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        timer.start("4. Transformer blocks")
        for i, blk in enumerate(visual_model.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )
        blk_time = timer.stop("4. Transformer blocks")

    log_gpu("after  transformer blocks", device.index)
    print(f"  hidden_states shape        : {list(hidden_states.shape)}")
    print(f"  Time                       : {blk_time:.3f} s\n")

    # ── Step 5 : Flash-memory (temporal compression + merger) ─────────────────
    print("── Step 5: Flash-memory  (temporal compression + PatchMerger)")
    log_gpu("before flash-memory", device.index)

    with torch.inference_mode():
        timer.start("5. Flash-memory + merger")

        # flash_memory.forward expects position_ids shape [3, bsz, seq_len]
        # and visual_position_ids shape [bsz, seq_len]
        video_embeds, updated_position_ids = visual_model.flash_memory(
            hidden_states,
            video_grid_thw,
            small_grid_thw,      # None  (no small-resolution pathway)
            position_ids,
            visual_position_ids,
        )
        # PatchMerger: [1, N_patches, embed_dim] → [N_tokens, hidden_size]
        video_embeds = visual_model.merger(video_embeds)
        fm_time = timer.stop("5. Flash-memory + merger")

    log_gpu("after  flash-memory", device.index)
    print(f"  video_embeds shape         : {list(video_embeds.shape)}")
    print(f"  updated_position_ids shape : {list(updated_position_ids.shape)}")
    print(f"  Time                       : {fm_time:.3f} s\n")

    # ── End-to-end summary ────────────────────────────────────────────────────
    print("── GPU memory peak")
    peak_alloc = torch.cuda.max_memory_allocated(device.index) / 1024 ** 2
    peak_reserved = torch.cuda.max_memory_reserved(device.index) / 1024 ** 2
    print(f"  Peak allocated : {peak_alloc:.1f} MB")
    print(f"  Peak reserved  : {peak_reserved:.1f} MB\n")

    timer.report()
    print()
    print("Pipeline test PASSED.")


# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline test: Flash-VStream visual encoder, 1 GPU, no spatial pooling"
    )
    parser.add_argument(
        "--model-path", type=str, default="zhang9302002/Flash-VStream-Qwen-7b",
        help="Path to a Qwen2-VL / FlashVStream checkpoint (optional). "
             "If omitted, random weights are used."
    )
    parser.add_argument(
        "--attn",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
        help="Attention implementation (default: sdpa). "
             "Use flash_attention_2 if flash-attn is installed."
    )
    return parser.parse_args()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires at least one CUDA GPU.")
    args = parse_args()
    run_pipeline_test(args)
