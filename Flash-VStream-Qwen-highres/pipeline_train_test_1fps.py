#!/usr/bin/env python3
import argparse
import inspect
import json
import os
from typing import List

import torch
from PIL import Image

from finetune_streaming_variants import StreamingVariantConfig, build_streaming_chunks
from models.ragged_flash_memory_retriever import RaggedFlashMemoryRetriever


def load_frames(frame_dir: str, max_frames: int) -> torch.Tensor:
    names = sorted(os.listdir(frame_dir))[:max_frames]
    frames: List[torch.Tensor] = []
    for name in names:
        path = os.path.join(frame_dir, name)
        if not os.path.isfile(path):
            continue
        img = Image.open(path).convert("RGB")
        t = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float()
        frames.append(t)
    if not frames:
        raise RuntimeError(f"No frame files found in {frame_dir}")
    return torch.stack(frames, dim=0)


def video_to_frame_dir(video_rel: str) -> str:
    if video_rel.lower().endswith(".mp4"):
        return video_rel[:-4]
    stem, _ = os.path.splitext(video_rel)
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline test for 1fps dataset + streaming chunk interface.")
    parser.add_argument(
        "--train-json",
        default="/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/data/train_test_1fps_64.json",
    )
    parser.add_argument(
        "--frame-root",
        default="/scratch/zwu24/datasets/LLaVA-Video-178K_fps1",
    )
    parser.add_argument("--max-frames", type=int, default=16)
    args = parser.parse_args()

    source_file = inspect.getsourcefile(RaggedFlashMemoryRetriever) or ""
    expected = "ragged_flash_memory_retriever.py"
    if expected not in source_file:
        raise RuntimeError(f"Ragged retriever source mismatch: {source_file}")
    print(f"Ragged retriever source verified: {source_file}")

    with open(args.train_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise RuntimeError("Empty train_json")

    print(f"Loaded samples: {len(data)}")
    print("First 5 videos:")
    for item in data[:5]:
        print(f"  - {item['video']}")

    v = data[0]["video"]
    frame_dir = os.path.join(args.frame_root, video_to_frame_dir(v))
    frames = load_frames(frame_dir, max_frames=args.max_frames)
    print(f"Loaded first sample frames from: {frame_dir}")
    print(f"Tensor shape [T,C,H,W]: {tuple(frames.shape)}")

    fullres_cfg = StreamingVariantConfig(variant="fullres", frames_per_chunk=8)
    dynamic_cfg = StreamingVariantConfig(variant="dynamic", frames_per_chunk=8, seed=42)
    chunks_fullres = build_streaming_chunks(frames, fullres_cfg)
    chunks_dynamic = build_streaming_chunks(frames, dynamic_cfg)

    print(f"fullres chunks: {len(chunks_fullres)}")
    print(f"dynamic chunks: {len(chunks_dynamic)}")
    if chunks_fullres:
        c0 = chunks_fullres[0]
        print(f"fullres[0] used_size={c0.used_size}, shape={tuple(c0.frames.shape)}")
    if chunks_dynamic:
        c0 = chunks_dynamic[0]
        print(f"dynamic[0] used_size={c0.used_size}, shape={tuple(c0.frames.shape)}")


if __name__ == "__main__":
    main()
