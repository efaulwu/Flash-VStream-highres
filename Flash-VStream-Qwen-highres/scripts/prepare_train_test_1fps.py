#!/usr/bin/env python3
import argparse
import json
import os
from typing import List


def video_to_frame_dir(video_rel: str) -> str:
    if video_rel.lower().endswith(".mp4"):
        return video_rel[:-4]
    stem, _ = os.path.splitext(video_rel)
    return stem


def has_frames(frame_root: str, video_rel: str) -> bool:
    frame_dir = os.path.join(frame_root, video_to_frame_dir(video_rel))
    if not os.path.isdir(frame_dir):
        return False
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    with os.scandir(frame_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            _, ext = os.path.splitext(entry.name)
            if ext.lower() in valid_exts and entry.stat().st_size > 0:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small 1fps training-test subset JSON.")
    parser.add_argument(
        "--input-json",
        default="/scratch/zwu24/datasets/LLaVA-Video-178K/selected_10k_pairs_seed42.json",
    )
    parser.add_argument(
        "--frame-root",
        default="/scratch/zwu24/datasets/LLaVA-Video-178K_fps1",
    )
    parser.add_argument(
        "--output-json",
        default="/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/data/train_test_1fps_64.json",
    )
    parser.add_argument("--max-samples", type=int, default=64)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    selected: List[dict] = []
    seen = set()
    skipped_non_dict = 0
    skipped_bad_video_field = 0
    skipped_duplicate = 0
    skipped_no_decoded_frames = 0
    use_limit = args.max_samples > 0

    for item in data:
        if not isinstance(item, dict):
            skipped_non_dict += 1
            continue
        video = item.get("video")
        if not isinstance(video, str):
            skipped_bad_video_field += 1
            continue
        # Keep only one QA pair per video for a quick smoke train.
        if video in seen:
            skipped_duplicate += 1
            continue
        if not has_frames(args.frame_root, video):
            skipped_no_decoded_frames += 1
            continue
        selected.append(item)
        seen.add(video)
        if use_limit and len(selected) >= args.max_samples:
            break

    if not selected:
        raise RuntimeError("No valid samples with fps1 frames were found.")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False)

    print(f"Wrote {len(selected)} samples to {args.output_json}")
    print(f"Sample limit mode: {'full-dataset' if not use_limit else f'first-{args.max_samples}'}")
    print(f"Frame root: {args.frame_root}")
    print(
        "Skip stats: "
        f"non_dict={skipped_non_dict}, "
        f"bad_video_field={skipped_bad_video_field}, "
        f"duplicate_video={skipped_duplicate}, "
        f"missing_or_invalid_decoded_frames={skipped_no_decoded_frames}"
    )
    print("First 5 videos:")
    for item in selected[:5]:
        print(f"  - {item['video']}")


if __name__ == "__main__":
    main()
