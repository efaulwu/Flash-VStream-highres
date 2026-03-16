from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import random
import torch
import torch.nn.functional as F


@dataclass
class StreamingVariantConfig:
    variant: str = "fullres"
    frames_per_chunk: int = 8
    align_factor: int = 28
    short_side_candidates: Tuple[int, ...] = (240, 360, 480, 720)
    seed: int = 42


@dataclass
class VideoChunk:
    chunk_id: int
    frames: torch.Tensor
    orig_size: Tuple[int, int]
    used_size: Tuple[int, int]


def _round_to_multiple(value: int, multiple: int, min_value: Optional[int] = None) -> int:
    rounded = int(round(value / multiple) * multiple)
    if min_value is not None:
        rounded = max(rounded, min_value)
    return max(multiple, rounded)


def _choose_res_with_aspect(orig_h: int, orig_w: int, short_side: int, align_factor: int) -> Tuple[int, int]:
    if orig_h <= orig_w:
        new_h = short_side
        new_w = int(round(orig_w * short_side / orig_h))
    else:
        new_w = short_side
        new_h = int(round(orig_h * short_side / orig_w))

    new_h = _round_to_multiple(new_h, align_factor, min_value=align_factor)
    new_w = _round_to_multiple(new_w, align_factor, min_value=align_factor)
    return new_h, new_w


def _resize_chunk(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError("frames must be [T,C,H,W]")
    t, c, h, w = frames.shape
    if h == target_h and w == target_w:
        return frames
    return F.interpolate(frames, size=(target_h, target_w), mode="bilinear", align_corners=False)


def build_fullres_streaming_chunks(video_frames: torch.Tensor, frames_per_chunk: int) -> List[VideoChunk]:
    if video_frames.ndim != 4:
        raise ValueError("video_frames must be [T,C,H,W]")
    t, c, h, w = video_frames.shape
    chunks: List[VideoChunk] = []
    chunk_id = 0
    for start in range(0, t, frames_per_chunk):
        end = min(start + frames_per_chunk, t)
        chunk = video_frames[start:end]
        chunks.append(
            VideoChunk(
                chunk_id=chunk_id,
                frames=chunk,
                orig_size=(h, w),
                used_size=(h, w),
            )
        )
        chunk_id += 1
    return chunks


def build_dynamic_resolution_chunks(
    video_frames: torch.Tensor,
    frames_per_chunk: int,
    align_factor: int,
    short_side_candidates: Sequence[int],
    seed: int = 42,
) -> List[VideoChunk]:
    if video_frames.ndim != 4:
        raise ValueError("video_frames must be [T,C,H,W]")
    t, c, h, w = video_frames.shape
    rng = random.Random(seed)

    chunks: List[VideoChunk] = []
    chunk_id = 0
    for start in range(0, t, frames_per_chunk):
        end = min(start + frames_per_chunk, t)
        chunk = video_frames[start:end]

        short_side = rng.choice(list(short_side_candidates))
        target_h, target_w = _choose_res_with_aspect(h, w, short_side, align_factor)
        chunk_resized = _resize_chunk(chunk, target_h, target_w)

        chunks.append(
            VideoChunk(
                chunk_id=chunk_id,
                frames=chunk_resized,
                orig_size=(h, w),
                used_size=(target_h, target_w),
            )
        )
        chunk_id += 1
    return chunks


def build_streaming_chunks(video_frames: torch.Tensor, cfg: StreamingVariantConfig) -> List[VideoChunk]:
    variant = cfg.variant.lower()
    if variant == "fullres":
        return build_fullres_streaming_chunks(video_frames, frames_per_chunk=cfg.frames_per_chunk)
    if variant == "dynamic":
        return build_dynamic_resolution_chunks(
            video_frames=video_frames,
            frames_per_chunk=cfg.frames_per_chunk,
            align_factor=cfg.align_factor,
            short_side_candidates=cfg.short_side_candidates,
            seed=cfg.seed,
        )
    raise ValueError(f"Unsupported variant: {cfg.variant}")


def chunk_to_processor_video_list(chunk: VideoChunk) -> List[torch.Tensor]:
    frames = chunk.frames
    if frames.dtype != torch.uint8:
        frames = frames.clamp(0, 255).to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).contiguous()
    return [frame.cpu().numpy() for frame in frames]


def run_streaming_memory_updates(model, processor, chunks: List[VideoChunk], flash_memory_config: Dict, start_idx: int = 0):
    stats = []
    temporal_patch_size = int(getattr(processor.image_processor, "temporal_patch_size", 1))
    for i, chunk in enumerate(chunks):
        chunk_frames = chunk.frames
        padded_frames = 0
        if temporal_patch_size > 1:
            remainder = int(chunk_frames.shape[0]) % temporal_patch_size
            if remainder != 0:
                padded_frames = temporal_patch_size - remainder
                last = chunk_frames[-1:].repeat(padded_frames, 1, 1, 1)
                chunk_frames = torch.cat([chunk_frames, last], dim=0)

        chunk_for_processor = VideoChunk(
            chunk_id=chunk.chunk_id,
            frames=chunk_frames,
            orig_size=chunk.orig_size,
            used_size=chunk.used_size,
        )
        videos = [chunk_to_processor_video_list(chunk_for_processor)]
        processed = processor.image_processor(
            images=None,
            videos=videos,
            return_tensors="pt",
            additional_pool_size=flash_memory_config["flash_memory_temporal_poolsize"],
        )
        pixel_values_videos = processed["pixel_values_videos"]
        video_grid_thw = processed["video_grid_thw"]

        timing = model.embed_new_video_clip(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            start_idx=start_idx + i,
        )
        stats.append(
            {
                "chunk_id": chunk.chunk_id,
                "orig_size": chunk.orig_size,
                "used_size": chunk.used_size,
                "original_chunk_frames": int(chunk.frames.shape[0]),
                "padded_chunk_frames": int(chunk_frames.shape[0]),
                "padded_tail_frames": int(padded_frames),
                "video_grid_thw": video_grid_thw.tolist(),
                "timing_len": len(timing),
            }
        )
    return stats
