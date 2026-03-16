import os
import sys

import torch

ROOT = "/scratch/zwu24/Flash-VStream-highres"
QWEN_ROOT = os.path.join(ROOT, "Flash-VStream-Qwen-highres")
if QWEN_ROOT not in sys.path:
    sys.path.insert(0, QWEN_ROOT)

from finetune_streaming_variants import (
    StreamingVariantConfig,
    build_streaming_chunks,
    run_streaming_memory_updates,
)


class _DummyImageProcessor:
    def __call__(self, images, videos, return_tensors, additional_pool_size):
        video = videos[0]
        t = len(video)
        h = video[0].shape[0]
        w = video[0].shape[1]
        grid_h = max(2, h // 28)
        grid_w = max(2, w // 28)
        grid_t = max(1, t // 2)
        n = grid_t * grid_h * grid_w
        return {
            "pixel_values_videos": torch.randn(n, 1176),
            "video_grid_thw": torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long),
        }


class _DummyProcessor:
    def __init__(self):
        self.image_processor = _DummyImageProcessor()


class _DummyModel:
    def __init__(self):
        self.calls = []

    def embed_new_video_clip(self, pixel_values_videos, video_grid_thw, start_idx):
        self.calls.append(
            {
                "pixel_values_shape": list(pixel_values_videos.shape),
                "video_grid_thw": video_grid_thw.tolist(),
                "start_idx": int(start_idx),
            }
        )
        return [0.0, 1.0, 2.0, 3.0]


def _make_dummy_video(t=23, h=333, w=517):
    return torch.randint(0, 256, (t, 3, h, w), dtype=torch.uint8)


def test_fullres_variant_chunks_keep_original_resolution():
    video = _make_dummy_video()
    cfg = StreamingVariantConfig(variant="fullres", frames_per_chunk=8)
    chunks = build_streaming_chunks(video, cfg)

    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.orig_size == (333, 517)
        assert chunk.used_size == (333, 517)


def test_dynamic_variant_chunks_are_aligned_and_varied():
    video = _make_dummy_video()
    cfg = StreamingVariantConfig(
        variant="dynamic",
        frames_per_chunk=8,
        align_factor=28,
        short_side_candidates=(240, 360, 480, 720),
        seed=123,
    )
    chunks = build_streaming_chunks(video, cfg)

    assert len(chunks) == 3
    any_changed = False
    for chunk in chunks:
        h, w = chunk.used_size
        assert h % 28 == 0
        assert w % 28 == 0
        if chunk.used_size != chunk.orig_size:
            any_changed = True
    assert any_changed


def test_memory_update_pipeline_runs_for_both_variants():
    video = _make_dummy_video(t=16, h=360, w=640)
    processor = _DummyProcessor()
    model = _DummyModel()
    flash_memory_config = {"flash_memory_temporal_poolsize": 2}

    cfg_full = StreamingVariantConfig(variant="fullres", frames_per_chunk=8)
    full_chunks = build_streaming_chunks(video, cfg_full)
    full_stats = run_streaming_memory_updates(model, processor, full_chunks, flash_memory_config, start_idx=0)

    assert len(full_stats) == len(full_chunks)
    assert len(model.calls) == len(full_chunks)
    assert all(item["timing_len"] == 4 for item in full_stats)

    model2 = _DummyModel()
    cfg_dyn = StreamingVariantConfig(variant="dynamic", frames_per_chunk=8, align_factor=28, seed=1)
    dyn_chunks = build_streaming_chunks(video, cfg_dyn)
    dyn_stats = run_streaming_memory_updates(model2, processor, dyn_chunks, flash_memory_config, start_idx=100)

    assert len(dyn_stats) == len(dyn_chunks)
    assert len(model2.calls) == len(dyn_chunks)
    assert model2.calls[0]["start_idx"] == 100
