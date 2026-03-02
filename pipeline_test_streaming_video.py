"""
Pipeline test for streaming video input with Flash-VStream memory.

目标：
1) 使用默认 FlashMemory 配置。
2) 随机生成视频 chunk，逐 chunk 更新模型记忆。
3) 每一步打印记忆模块调试信息（shape / 时间索引 / 关键变量状态）。
4) 喂入 10 个 chunk 后，读取模型输出并交给一个 LLM 生成文本。

说明：
- 该测试是“轻量可运行”的 pipeline 验证，不依赖真实大模型权重。
- 通过 dummy 视觉编码器 + dummy LLM，重点验证 streaming memory 数据流。
"""

import os
import sys
import threading
from typing import Any, List

import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(CURRENT_DIR, "Flash-VStream-Qwen-highres")
if MODEL_ROOT not in sys.path:
    sys.path.insert(0, MODEL_ROOT)

from models.flash_memory_constants import DEFAULT_FLASH_MEMORY_CONFIG
from models.vstream_qwen2vl_realtime import FlashMemory, FlashVStreamQwen2VLModel


torch.manual_seed(42)


class ChunkSpec:
    def __init__(self, t=1, h=2, w=2, dim=16):
        self.t = t
        self.h = h
        self.w = w
        self.dim = dim

    def __repr__(self):
        return "ChunkSpec(t={}, h={}, w={}, dim={})".format(self.t, self.h, self.w, self.dim)


class DummyVisionBackbone:
    def __init__(self, flash_memory: FlashMemory):
        self.flash_memory = flash_memory

    def get_dtype(self):
        return torch.float32

    def get_device(self):
        return torch.device("cpu")

    def forward_simple_not_merge(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor):
        thw = video_grid_thw[0]
        t, h, w = [int(v.item()) for v in thw]
        token_count = t * h * w
        dim = pixel_values_videos.shape[-1]

        x = torch.randn(token_count, dim, dtype=torch.float32)
        grid_thw = video_grid_thw.clone()
        small_grid_thw = None
        return x, grid_thw, small_grid_thw

    def merger(self, flash_memory_tokens: torch.Tensor):
        return flash_memory_tokens


class DummyLLM:
    def generate(self, prompt: str, video_embeds: torch.Tensor, position_ids: torch.Tensor) -> str:
        return (
            f"[DummyLLM] prompt={prompt} | "
            f"video_embeds_shape={list(video_embeds.shape)} | "
            f"position_ids_shape={list(position_ids.shape)} | "
            f"video_embeds_mean={video_embeds.float().mean().item():.4f}"
        )


def _shape(x: Any):
    if torch.is_tensor(x):
        return list(x.shape)
    return type(x).__name__


def print_memory_debug(step_idx: int, memory_list: List[Any]):
    (
        tem_x,
        tem_thw,
        tem_weights,
        tem_timestamp,
        spa_x,
        spa_thw,
        spa_positions,
        full_x,
        full_thw,
        small_x,
        small_thw,
        video_embeds,
        video_embeds_shape,
    ) = memory_list

    print("\n" + "=" * 88)
    print(f"[Chunk {step_idx:02d}] Memory Debug")
    print("=" * 88)
    print(f"tem_x.shape={_shape(tem_x)} tem_thw={tem_thw.tolist()} tem_weights.shape={_shape(tem_weights)}")
    print(
        f"tem_timestamp(min,max)=({tem_timestamp.min().item():.2f}, {tem_timestamp.max().item():.2f}) "
        f"tem_positions_unique={torch.unique(tem_timestamp.round().long()).tolist()}"
    )
    print(f"spa_x.shape={_shape(spa_x)} spa_thw={spa_thw.tolist()} spa_positions.shape={_shape(spa_positions)}")
    print(f"full_x.shape={_shape(full_x)} full_thw={full_thw.tolist()} small_x.shape={_shape(small_x)} small_thw={small_thw.tolist()}")
    print(f"video_embeds.shape={_shape(video_embeds)} video_embeds_shape_meta={_shape(video_embeds_shape)}")


def build_streaming_model() -> FlashVStreamQwen2VLModel:
    model = FlashVStreamQwen2VLModel.__new__(FlashVStreamQwen2VLModel)
    flash_memory = FlashMemory(**DEFAULT_FLASH_MEMORY_CONFIG)

    model.visual = DummyVisionBackbone(flash_memory=flash_memory)
    model.use_video_streaming_mode = True
    model.video_embedding_memory = []
    model.video_embedding_mem_lock = threading.Lock()

    return model


def run_pipeline_test(num_chunks: int = 10):
    model = build_streaming_model()
    llm = DummyLLM()

    chunk_spec = ChunkSpec()
    print("\n[Config] Using DEFAULT_FLASH_MEMORY_CONFIG:")
    print(DEFAULT_FLASH_MEMORY_CONFIG)
    print(f"[Config] ChunkSpec={chunk_spec}")

    for chunk_idx in range(1, num_chunks + 1):
        pixel_values_videos = torch.randn(1, chunk_spec.dim, dtype=torch.float32)
        video_grid_thw = torch.tensor([[chunk_spec.t, chunk_spec.h, chunk_spec.w]], dtype=torch.long)
        start_idx = chunk_idx - 1

        timeline = model.embed_new_video_clip(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            start_idx=start_idx,
        )

        print(f"\n[Chunk {chunk_idx:02d}] profiler timestamps len={len(timeline)}")
        print_memory_debug(step_idx=chunk_idx, memory_list=model.video_embedding_memory)

    with torch.no_grad():
        memory = model.video_embedding_memory
        tem_thw = memory[1]
        spa_thw = memory[5]
        visual_token_count = int((tem_thw.prod() // 4 + spa_thw.prod() // 4).item())

        text_prefix = 2
        text_suffix = 3
        seq_len = text_prefix + visual_token_count + text_suffix

        base = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        position_ids = torch.stack([base, base, base], dim=0)
        visual_position_ids = torch.full((1, seq_len), -1, dtype=torch.long)
        visual_position_ids[0, text_prefix : text_prefix + visual_token_count] = 0

        video_embeds, new_position_ids = model.prepare_realtime_inference(
            position_ids=position_ids,
            visual_position_ids=visual_position_ids,
        )

    print("\n" + "-" * 88)
    print("[Post-10-Chunks] Realtime inference preparation")
    print("-" * 88)
    print(f"video_embeds.shape={list(video_embeds.shape)}")
    print(f"new_position_ids.shape={list(new_position_ids.shape)}")

    prompt = "Please summarize the streamed video content."
    generation = llm.generate(prompt=prompt, video_embeds=video_embeds, position_ids=new_position_ids)

    print("\n" + "-" * 88)
    print("[LLM Generation]")
    print("-" * 88)
    print(generation)


if __name__ == "__main__":
    run_pipeline_test(num_chunks=10)
