#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers.modeling_utils import load_sharded_checkpoint

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_streaming_variants import StreamingVariantConfig, build_streaming_chunks, chunk_to_processor_video_list
from models import FlashVStreamQwen2VLProcessor, DEFAULT_FLASH_MEMORY_CONFIG
from models.ragged_flash_memory_retriever_fillup import RaggedFlashMemoryRetrieverFillup
from models.vstream_qwen2vl_realtime import (
    FlashVStreamQwen2VLConfig,
    FlashVStreamQwen2VisionTransformerPretrainedModel,
)


class _VisualOnlyWrapper(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.visual = FlashVStreamQwen2VisionTransformerPretrainedModel._from_config(
            vision_config,
            attn_implementation="flash_attention_2",
        )


class RaggedMemoryOnlyRunnerFillup:
    def __init__(self, model_path: str, device: str, dtype: torch.dtype):
        model_config = FlashVStreamQwen2VLConfig.from_pretrained(model_path, trust_remote_code=True)
        if getattr(model_config.vision_config, "flash_memory_config", None) is None:
            model_config.vision_config.flash_memory_config = DEFAULT_FLASH_MEMORY_CONFIG

        self.config = model_config
        self.flash_memory_config = dict(model_config.vision_config.flash_memory_config)
        self.device = torch.device(device)

        wrapper = _VisualOnlyWrapper(model_config.vision_config)
        load_sharded_checkpoint(wrapper, model_path, strict=False, prefer_safe=True)

        self.visual = wrapper.visual.to(device=self.device, dtype=dtype).eval()
        for p in self.visual.parameters():
            p.requires_grad_(False)

        self.ragged_retriever = RaggedFlashMemoryRetrieverFillup(
            dim=model_config.hidden_size,
            budget_target=int(self.flash_memory_config.get("flash_memory_budget_target", 11520)),
            budget_hard=int(self.flash_memory_config.get("flash_memory_budget_hard", 12000)),
            r_t=float(self.flash_memory_config.get("flash_memory_rt", 0.3333)),
            xbin=int(self.flash_memory_config.get("flash_memory_xbin", 64)),
            ybin=int(self.flash_memory_config.get("flash_memory_ybin", 64)),
            topM_items=int(self.flash_memory_config.get("flash_memory_topM_items", 10)),
            num_anchors=int(self.flash_memory_config.get("flash_memory_num_anchors", 8)),
        )

        self.use_video_streaming_mode = True
        self.video_embedding_memory = None
        self.logger = logging.getLogger(__name__ + ".RaggedMemoryOnlyRunnerFillup")

    @torch.no_grad()
    def embed_new_video_clip(self, pixel_values_videos, video_grid_thw, start_idx):
        pixel_values_videos = pixel_values_videos.to(self.device, dtype=self.visual.get_dtype())
        video_grid_thw = video_grid_thw.to(self.device)

        x, grid_thw, small_grid_thw = self.visual.forward_simple_not_merge(pixel_values_videos, video_grid_thw)
        thw = video_grid_thw[0]
        t, h, w = thw.tolist()

        if small_grid_thw is not None:
            hi_raw, lo_raw = torch.split(x, [grid_thw.prod(), small_grid_thw.prod()])
        else:
            hi_raw = x
            lo_raw = x

        hi_tokens = self.visual.merger(hi_raw.unsqueeze(0))
        if hi_tokens.ndim == 3 and hi_tokens.shape[0] == 1:
            hi_tokens = hi_tokens[0]
        lo_tokens = self.visual.merger(lo_raw.unsqueeze(0))
        if lo_tokens.ndim == 3 and lo_tokens.shape[0] == 1:
            lo_tokens = lo_tokens[0]

        token_meta = {
            "chunk_id": int(start_idx),
            "t0": int(start_idx),
            "t_len": int(t),
            "src_res_h": int(h * 14),
            "src_res_w": int(w * 14),
            "x_norm": None,
            "y_norm": None,
        }

        update_stats = self.ragged_retriever.update(
            hi_tokens=hi_tokens,
            lo_tokens=lo_tokens,
            token_meta=token_meta,
        )
        self.video_embedding_memory = {
            "mode": "ragged_fillup",
            "latest_update_stats": update_stats,
        }
        return [0.0, 0.0, 0.0]

    @torch.no_grad()
    def prepare_realtime_inference(self, position_ids, visual_position_ids, query_embed: Optional[torch.Tensor] = None):
        selected_tokens, selected_meta, local_position_ids, stats = self.ragged_retriever.retrieve(query_embed=query_embed)

        full_position = position_ids[:, 0].clone()
        visual_mask = visual_position_ids[0] >= 0
        visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(1)
        target_n = int(visual_indices.shape[0])

        if selected_tokens.shape[0] > target_n:
            selected_tokens = selected_tokens[:target_n]
            local_position_ids = local_position_ids[:, :, :target_n]
        elif selected_tokens.shape[0] < target_n:
            pad_n = target_n - selected_tokens.shape[0]
            if pad_n > 0:
                pad_tokens = torch.zeros(
                    pad_n,
                    selected_tokens.shape[1],
                    dtype=selected_tokens.dtype,
                    device=selected_tokens.device,
                )
                selected_tokens = torch.cat([selected_tokens, pad_tokens], dim=0)
                pad_pos = torch.zeros(3, 1, pad_n, dtype=local_position_ids.dtype, device=local_position_ids.device)
                local_position_ids = torch.cat([local_position_ids, pad_pos], dim=2)

        if target_n > 0:
            visual_start_pos = int(visual_indices[0].item())
            visual_start_id = int(full_position[0, visual_start_pos].item())
            full_position[:, visual_indices] = visual_start_id + local_position_ids[:, 0, :target_n]

        return selected_tokens, selected_meta, full_position.unsqueeze(1), stats


def _resolve_frame_dir(frame_root: str, video_rel: str) -> str:
    raw = os.path.join(frame_root, video_rel)
    candidates = [os.path.splitext(raw)[0]]
    if raw.lower().endswith(".mp4"):
        candidates.append(raw[:-4])
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"No decoded frame dir found for: {video_rel}")


def _frame_sort_key(name: str):
    stem = os.path.splitext(name)[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else stem


def _load_frames(frame_dir: str, max_frames: int) -> torch.Tensor:
    names = sorted(os.listdir(frame_dir), key=_frame_sort_key)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    frames: List[torch.Tensor] = []
    for name in names:
        path = os.path.join(frame_dir, name)
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() not in valid_exts:
            continue
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        frames.append(torch.from_numpy(arr).permute(2, 0, 1))
        if max_frames > 0 and len(frames) >= max_frames:
            break
    if not frames:
        raise RuntimeError(f"No valid frames loaded from: {frame_dir}")
    return torch.stack(frames, dim=0)


def _build_alignment_inputs(target_n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    text_prefix = 4
    text_suffix = 8
    seq_len = text_prefix + target_n + text_suffix
    base = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    position_ids = torch.stack([base, base, base], dim=0)
    visual_position_ids = torch.full((1, seq_len), -1, dtype=torch.long, device=device)
    visual_position_ids[0, text_prefix : text_prefix + target_n] = 0
    return position_ids, visual_position_ids


def _build_query_embedding(query: str, tokenizer, dim: int, device: torch.device) -> Tuple[Optional[torch.Tensor], Dict[str, int]]:
    query = (query or "").strip()
    if not query:
        return None, {"query_tokens": 0}

    token_ids = tokenizer(query, add_special_tokens=False).get("input_ids", [])
    if not token_ids:
        return None, {"query_tokens": 0}

    vec = torch.zeros(dim, dtype=torch.float32, device=device)
    scale = 1.0 / float(max(1, len(token_ids)))
    for tid in token_ids:
        tid_int = int(tid)
        idx_a = tid_int % dim
        idx_b = (tid_int * 1103515245 + 12345) % dim
        sign = -1.0 if (tid_int & 1) else 1.0
        vec[idx_a] += scale
        vec[idx_b] += sign * (0.5 * scale)

    norm = torch.norm(vec, p=2)
    if float(norm.item()) > 0.0:
        vec = vec / norm
    return vec, {"query_tokens": int(len(token_ids))}


def _extract_dataset_query(sample: Dict) -> Tuple[str, str]:
    conversations = sample.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            if str(turn.get("from", "")).lower() != "human":
                continue
            value = str(turn.get("value", "")).strip()
            if not value:
                continue
            lines = [ln.strip() for ln in value.splitlines() if ln.strip() and ln.strip() != "<image>"]
            if lines:
                return lines[0], "sample.conversations[human].value"

    for key in ("question", "query", "instruction", "prompt"):
        v = sample.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip(), f"sample.{key}"

    return "", "none"


def _run_streaming_memory_updates_with_stats(model, processor, chunks: List, flash_memory_config: Dict, start_idx: int = 0):
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

        videos = [chunk_to_processor_video_list(type(chunk)(
            chunk_id=chunk.chunk_id,
            frames=chunk_frames,
            orig_size=chunk.orig_size,
            used_size=chunk.used_size,
        ))]

        processed = processor.image_processor(
            images=None,
            videos=videos,
            return_tensors="pt",
            additional_pool_size=flash_memory_config["flash_memory_temporal_poolsize"],
        )
        pixel_values_videos = processed["pixel_values_videos"]
        video_grid_thw = processed["video_grid_thw"]

        model.embed_new_video_clip(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            start_idx=start_idx + i,
        )

        latest_update = (model.video_embedding_memory or {}).get("latest_update_stats", {})
        stats.append(
            {
                "chunk_id": int(chunk.chunk_id),
                "orig_size": tuple(chunk.orig_size),
                "used_size": tuple(chunk.used_size),
                "original_chunk_frames": int(chunk.frames.shape[0]),
                "padded_chunk_frames": int(chunk_frames.shape[0]),
                "padded_tail_frames": int(padded_frames),
                "video_grid_thw": video_grid_thw.tolist(),
                "update_stats": latest_update,
            }
        )
    return stats


def _save_overlay_images(
    out_dir: Path,
    chunks,
    chunk_stats: List[Dict],
    selected_meta: List[Dict],
    xbin: int,
    ybin: int,
) -> Dict:
    chunk_by_id = {c.chunk_id: c for c in chunks}
    grid_t_by_chunk = {}
    for stat in chunk_stats:
        cid = int(stat["chunk_id"])
        grid_t_by_chunk[cid] = int(stat["video_grid_thw"][0][0])

    hit_maps: Dict[Tuple[int, int], np.ndarray] = {}
    val_maps: Dict[Tuple[int, int], np.ndarray] = {}
    counts: Dict[Tuple[int, int], int] = {}
    skipped_temporal_tokens = 0

    for meta in selected_meta:
        chunk_id = int(meta.get("chunk_id", -1))
        if chunk_id < 0:
            skipped_temporal_tokens += 1
            continue
        if chunk_id not in chunk_by_id:
            continue

        chunk = chunk_by_id[chunk_id]
        t_frames = int(chunk.frames.shape[0])
        grid_t = max(1, grid_t_by_chunk.get(chunk_id, t_frames))
        t0 = chunk_id
        patch_t = int(meta.get("t", t0)) - t0
        patch_t = max(0, min(grid_t - 1, patch_t))
        frame_idx = int(round((patch_t + 0.5) * t_frames / grid_t - 0.5))
        frame_idx = max(0, min(t_frames - 1, frame_idx))

        h, w = chunk.used_size
        key = (chunk_id, frame_idx)
        if key not in hit_maps:
            hit_maps[key] = np.zeros((h, w), dtype=np.float32)
            val_maps[key] = np.zeros((h, w), dtype=np.float32)
            counts[key] = 0

        x_bin = int(meta.get("x_bin", xbin // 2))
        y_bin = int(meta.get("y_bin", ybin // 2))
        utility = float(meta.get("utility", 0.0))

        x_bin = max(0, min(xbin - 1, x_bin))
        y_bin = max(0, min(ybin - 1, y_bin))

        x0 = int(np.floor(x_bin * w / xbin))
        x1 = int(np.floor((x_bin + 1) * w / xbin))
        y0 = int(np.floor(y_bin * h / ybin))
        y1 = int(np.floor((y_bin + 1) * h / ybin))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))

        hit_maps[key][y0:y1, x0:x1] += 1.0
        val_maps[key][y0:y1, x0:x1] += utility
        counts[key] += 1

    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for (chunk_id, frame_idx), hits in hit_maps.items():
        chunk = chunk_by_id[chunk_id]
        frame = chunk.frames[frame_idx]
        frame_np = frame.detach().cpu().float().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()

        base = Image.fromarray(frame_np, mode="RGB").convert("RGBA")

        value = val_maps[(chunk_id, frame_idx)]
        avg_value = np.divide(value, np.maximum(hits, 1e-6))

        vmin = float(np.min(avg_value))
        vmax = float(np.max(avg_value))
        if vmax > vmin:
            norm = (avg_value - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(avg_value, dtype=np.float32)

        # Keep overlay transparent and use red depth to encode token value.
        alpha = np.clip(15.0 + norm * 110.0, 0.0, 125.0).astype(np.uint8)
        color_g = np.clip(230.0 - norm * 190.0, 40.0, 230.0).astype(np.uint8)
        color_b = np.clip(230.0 - norm * 205.0, 25.0, 230.0).astype(np.uint8)

        overlay_rgba = np.zeros((hits.shape[0], hits.shape[1], 4), dtype=np.uint8)
        overlay_rgba[..., 0] = 255
        overlay_rgba[..., 1] = color_g
        overlay_rgba[..., 2] = color_b
        overlay_rgba[..., 3] = alpha

        overlay = Image.fromarray(overlay_rgba, mode="RGBA")
        composed = Image.alpha_composite(base, overlay)

        out_name = f"chunk{chunk_id:03d}_frame{frame_idx:03d}_tokens{counts[(chunk_id, frame_idx)]:04d}.png"
        out_path = overlay_dir / out_name
        composed.save(out_path)
        saved.append(str(out_path))

    return {
        "overlay_count": len(saved),
        "overlays": saved,
        "skipped_temporal_tokens": skipped_temporal_tokens,
    }


def _build_selection_report(selected_meta: List[Dict], chunk_stats: List[Dict]) -> Dict:
    chunk_tokens: Dict[int, int] = {}
    frame_tokens: Dict[Tuple[int, int], int] = {}

    for meta in selected_meta:
        chunk_id = int(meta.get("chunk_id", -1))
        if chunk_id < 0:
            continue
        t = int(meta.get("t", 0))
        chunk_tokens[chunk_id] = chunk_tokens.get(chunk_id, 0) + 1
        k = (chunk_id, t)
        frame_tokens[k] = frame_tokens.get(k, 0) + 1

    chunk_available = {}
    for st in chunk_stats:
        cid = int(st["chunk_id"])
        up = st.get("update_stats", {})
        chunk_available[cid] = int(up.get("hi_tokens_n", 0))

    per_chunk = []
    for cid in sorted(chunk_available.keys()):
        per_chunk.append(
            {
                "chunk_id": cid,
                "available_hi_tokens": int(chunk_available.get(cid, 0)),
                "selected_tokens": int(chunk_tokens.get(cid, 0)),
            }
        )

    per_frame = []
    for (cid, t), c in sorted(frame_tokens.items(), key=lambda x: (x[0][0], x[0][1])):
        per_frame.append(
            {
                "chunk_id": int(cid),
                "t": int(t),
                "selected_tokens": int(c),
            }
        )

    return {
        "per_chunk_token_counts": per_chunk,
        "per_frame_token_counts": per_frame,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize fill-up ragged memory tokens presented to LLM.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--frames-per-chunk", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--variant", choices=["fullres", "dynamic"], default="dynamic")
    parser.add_argument("--target-present-tokens", type=int, default=12000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.train_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise RuntimeError("train-json is empty")

    idx = max(0, min(len(data) - 1, args.sample_index))
    sample = data[idx]
    video_rel = sample["video"]
    frame_dir = _resolve_frame_dir(args.frame_root, video_rel)
    frames = _load_frames(frame_dir, args.max_frames)

    cfg = StreamingVariantConfig(
        variant=args.variant,
        frames_per_chunk=args.frames_per_chunk,
        seed=42,
    )
    chunks = build_streaming_chunks(frames, cfg)

    model = RaggedMemoryOnlyRunnerFillup(
        model_path=args.model_path,
        device=args.device,
        dtype=torch.bfloat16,
    )
    processor = FlashVStreamQwen2VLProcessor.from_pretrained(args.model_path)

    # Hard override to enforce user-required fill-up target.
    model.ragged_retriever.budget_target = int(args.target_present_tokens)
    model.ragged_retriever.budget_hard = int(args.target_present_tokens)

    flash_memory_config = model.flash_memory_config
    dataset_query, dataset_query_source = _extract_dataset_query(sample)
    effective_query = dataset_query if dataset_query else (args.query or "")
    effective_query_source = dataset_query_source if dataset_query else ("cli.--query" if args.query else "none")

    query_embed, query_info = _build_query_embedding(
        query="People in the video",
        tokenizer=processor.tokenizer,
        dim=model.config.hidden_size,
        device=model.device,
    )

    chunk_stats = _run_streaming_memory_updates_with_stats(
        model=model,
        processor=processor,
        chunks=chunks,
        flash_memory_config=flash_memory_config,
        start_idx=0,
    )

    selected_tokens, selected_meta, local_position_ids, retrieve_stats = model.ragged_retriever.retrieve(query_embed=query_embed)

    position_ids, visual_position_ids = _build_alignment_inputs(args.target_present_tokens, selected_tokens.device)
    aligned_video_embeds, aligned_meta, aligned_position_ids, aligned_stats = model.prepare_realtime_inference(
        position_ids=position_ids,
        visual_position_ids=visual_position_ids,
        query_embed=query_embed,
    )

    vis_info = _save_overlay_images(
        out_dir=out_dir,
        chunks=chunks,
        chunk_stats=chunk_stats,
        selected_meta=selected_meta,
        xbin=int(flash_memory_config.get("flash_memory_xbin", 64)),
        ybin=int(flash_memory_config.get("flash_memory_ybin", 64)),
    )

    selection_report = _build_selection_report(selected_meta=selected_meta, chunk_stats=chunk_stats)

    summary = {
        "sample_index": idx,
        "video": video_rel,
        "query": effective_query,
        "query_source": effective_query_source,
        "query_info": query_info,
        "query_embed_used": bool(query_embed is not None),
        "frame_dir": frame_dir,
        "chunks": len(chunks),
        "chunk_stats": chunk_stats,
        "retrieve_stats": retrieve_stats,
        "aligned_retrieve_stats": aligned_stats,
        "raw_retrieved_tokens_shape": list(selected_tokens.shape),
        "raw_position_ids_shape": list(local_position_ids.shape),
        "aligned_video_embeds_shape": list(aligned_video_embeds.shape),
        "aligned_position_ids_shape": list(aligned_position_ids.shape),
        "selection_report": selection_report,
        "selected_meta_preview": selected_meta[:50],
        "overlay": vis_info,
    }

    summary_path = out_dir / "ragged_memory_fillup_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[summary] sample={idx} video={video_rel}")
    print(
        f"[summary] query_source={effective_query_source} "
        f"query_embed_used={bool(query_embed is not None)} "
        f"query_tokens={query_info['query_tokens']}"
    )
    print(f"[summary] raw_retrieved_tokens_shape={list(selected_tokens.shape)}")
    print(f"[summary] aligned_video_embeds_shape={list(aligned_video_embeds.shape)}")
    print(f"[summary] retrieve_stats={retrieve_stats}")
    print(f"[summary] overlay_count={vis_info['overlay_count']} skipped_temporal_tokens={vis_info['skipped_temporal_tokens']}")
    print(f"[summary] saved={summary_path}")


if __name__ == "__main__":
    main()
