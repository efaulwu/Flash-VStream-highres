import argparse
import json
import logging
import os
import random
import statistics
import threading
import time
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(PROJECT_ROOT, "Flash-VStream-Qwen-highres")

import sys
if MODEL_ROOT not in sys.path:
    sys.path.insert(0, MODEL_ROOT)

from models.ragged_flash_memory_retriever import RaggedFlashMemoryRetriever
from models.vstream_qwen2vl_realtime import FlashVStreamQwen2VLModel


RES_SET = [(426, 240), (640, 360), (854, 480), (1280, 720), (1920, 1080)]


def gpu_mem_mb():
    if not torch.cuda.is_available():
        return {"cuda": False, "allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
    return {
        "cuda": True,
        "allocated": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved": torch.cuda.memory_reserved() / (1024 ** 2),
        "max_allocated": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def token_estimate_from_res(w, h):
    base = (w * h) / float(16 * 16)
    hi = max(64, int(base * 0.35))
    lo = max(16, hi // 4)
    return hi, lo


def build_logger(level):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("pipeline_test_ragged")


def run_retriever_only(args):
    logger = logging.getLogger("pipeline_test_ragged.retriever_only")
    retriever = RaggedFlashMemoryRetriever(
        dim=args.hidden_size,
        budget_target=args.budget_target,
        budget_hard=args.budget_hard,
        r_t=args.r_t,
        xbin=args.xbin,
        ybin=args.ybin,
        topM_items=args.topM_items,
        num_anchors=args.num_anchors,
    )

    rows = []
    for chunk_id in range(args.num_chunks):
        w, h = random.choice(RES_SET)
        hi_n, lo_n = token_estimate_from_res(w, h)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        hi_tokens = torch.randn(hi_n, args.hidden_size, device=args.device)
        lo_tokens = torch.randn(lo_n, args.hidden_size, device=args.device)
        up_stats = retriever.update(
            hi_tokens=hi_tokens,
            lo_tokens=lo_tokens,
            token_meta={
                "chunk_id": chunk_id,
                "t0": chunk_id * args.frames_per_chunk,
                "t_len": args.frames_per_chunk,
                "src_res_h": h,
                "src_res_w": w,
            },
        )
        t1 = time.perf_counter()

        selected_tokens, selected_meta, position_ids, ret_stats = retriever.retrieve(query_embed=None)
        t2 = time.perf_counter()
        assert selected_tokens.shape[0] <= args.budget_hard, f"N_sel={selected_tokens.shape[0]} > {args.budget_hard}"

        mem = gpu_mem_mb()
        logger.info(
            "[retriever-only] chunk=%d res=%dx%d hi=%d lo=%d update_ms=%.3f retrieve_ms=%.3f N_sel=%d B_target=%d B_hard=%d cuda=%s alloc=%.2fMB reserved=%.2fMB peak=%.2fMB",
            chunk_id,
            w,
            h,
            hi_n,
            lo_n,
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            selected_tokens.shape[0],
            args.budget_target,
            args.budget_hard,
            mem["cuda"],
            mem["allocated"],
            mem["reserved"],
            mem["max_allocated"],
        )
        rows.append(
            {
                "chunk_id": chunk_id,
                "res": f"{w}x{h}",
                "update_ms": (t1 - t0) * 1000,
                "retrieve_ms": (t2 - t1) * 1000,
                "N_sel": int(selected_tokens.shape[0]),
                "gpu_peak_mb": mem["max_allocated"],
            }
        )

    summarize(rows)


class DummyVisual:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = torch.randn(in_dim, out_dim) / (in_dim ** 0.5)

    def get_dtype(self):
        return torch.float32

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward_simple_not_merge(self, pixel_values_videos, video_grid_thw):
        thw = video_grid_thw[0]
        n = int(torch.prod(thw).item())
        x = torch.randn(n, self.in_dim, device=pixel_values_videos.device)
        return x, video_grid_thw, None

    def merger(self, x):
        if x.ndim == 3:
            x = x[0]
        n = x.shape[0]
        n4 = max(1, n // 4)
        x = x[: n4 * 4].reshape(n4, 4, -1).mean(dim=1)
        y = x @ self.proj.to(x.device)
        return y


def build_dummy_wrapped_model(args):
    model = FlashVStreamQwen2VLModel.__new__(FlashVStreamQwen2VLModel)
    model.logger = logging.getLogger("pipeline_test_ragged.model_wrapped")
    model.visual = DummyVisual(in_dim=args.visual_dim, out_dim=args.hidden_size)
    model.use_video_streaming_mode = True
    model.flash_memory_mode = "ragged"
    model.flash_memory_budget_target = args.budget_target
    model.flash_memory_budget_hard = args.budget_hard
    model.flash_memory_rt = args.r_t
    model.video_embedding_memory = {}
    model.video_embedding_mem_lock = threading.Lock()
    model.ragged_retriever = RaggedFlashMemoryRetriever(
        dim=args.hidden_size,
        budget_target=args.budget_target,
        budget_hard=args.budget_hard,
        r_t=args.r_t,
        xbin=args.xbin,
        ybin=args.ybin,
        topM_items=args.topM_items,
        num_anchors=args.num_anchors,
    )
    return model


def run_model_wrapped(args):
    logger = logging.getLogger("pipeline_test_ragged.model_wrapped")
    model = build_dummy_wrapped_model(args)
    rows = []

    for chunk_id in range(args.num_chunks):
        w, h = random.choice(RES_SET)
        grid_h = max(2, h // 28)
        grid_w = max(2, w // 28)
        grid_t = max(1, args.frames_per_chunk // 2)

        pixel_values_videos = torch.randn(1, args.visual_dim, device=args.device)
        video_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long, device=args.device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        u0 = time.perf_counter()
        model.embed_new_video_clip(pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, start_idx=chunk_id)
        u1 = time.perf_counter()

        n_override = min(args.budget_hard, args.budget_target)
        seq_len = 2 + n_override + 2
        base = torch.arange(seq_len, dtype=torch.long, device=args.device).unsqueeze(0)
        position_ids = torch.stack([base, base, base], dim=0)
        visual_position_ids = torch.full((1, seq_len), -1, dtype=torch.long, device=args.device)
        visual_position_ids[0, 2 : 2 + n_override] = 0

        video_embeds, new_position = model.prepare_realtime_inference(position_ids=position_ids, visual_position_ids=visual_position_ids)
        r0 = time.perf_counter()

        assert video_embeds.shape[0] <= args.budget_hard, f"N_sel={video_embeds.shape[0]} > {args.budget_hard}"

        mem = gpu_mem_mb()
        logger.info(
            "[model-wrapped] chunk=%d res=%dx%d update_ms=%.3f retrieve_ms=%.3f N_sel=%d pos_shape=%s cuda=%s alloc=%.2fMB reserved=%.2fMB peak=%.2fMB",
            chunk_id,
            w,
            h,
            (u1 - u0) * 1000,
            (r0 - u1) * 1000,
            int(video_embeds.shape[0]),
            list(new_position.shape),
            mem["cuda"],
            mem["allocated"],
            mem["reserved"],
            mem["max_allocated"],
        )
        rows.append(
            {
                "chunk_id": chunk_id,
                "res": f"{w}x{h}",
                "update_ms": (u1 - u0) * 1000,
                "retrieve_ms": (r0 - u1) * 1000,
                "N_sel": int(video_embeds.shape[0]),
                "gpu_peak_mb": mem["max_allocated"],
            }
        )

    ckpt_dir = os.path.join(PROJECT_ROOT, "tmp_ragged_dummy_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "flash_memory_mode": model.flash_memory_mode,
        "flash_memory_budget_target": model.flash_memory_budget_target,
        "flash_memory_budget_hard": model.flash_memory_budget_hard,
        "flash_memory_rt": model.flash_memory_rt,
        "num_items": len(model.ragged_retriever.items),
    }
    with open(os.path.join(ckpt_dir, "ragged_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(ckpt_dir, "ragged_config.json"), "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    logger.info("[save-reload-dummy] saved_and_reloaded=%s", reloaded)

    summarize(rows)


def summarize(rows):
    print("\n" + "=" * 88)
    print("RAGGED PIPELINE SUMMARY")
    print("=" * 88)
    for row in rows:
        print(
            f"chunk={row['chunk_id']:02d} res={row['res']} "
            f"update_ms={row['update_ms']:.2f} retrieve_ms={row['retrieve_ms']:.2f} "
            f"N_sel={row['N_sel']} gpu_peak_mb={row['gpu_peak_mb']:.2f}"
        )

    update_vals = [r["update_ms"] for r in rows]
    retrieve_vals = [r["retrieve_ms"] for r in rows]
    nsel_vals = [r["N_sel"] for r in rows]

    p95_update = sorted(update_vals)[max(0, int(0.95 * len(update_vals)) - 1)]
    p95_retrieve = sorted(retrieve_vals)[max(0, int(0.95 * len(retrieve_vals)) - 1)]
    print("-" * 88)
    print(
        f"mean_update_ms={statistics.mean(update_vals):.2f} p95_update_ms={p95_update:.2f} | "
        f"mean_retrieve_ms={statistics.mean(retrieve_vals):.2f} p95_retrieve_ms={p95_retrieve:.2f} | "
        f"mean_N_sel={statistics.mean(nsel_vals):.2f} max_N_sel={max(nsel_vals)}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["retriever-only", "model-wrapped"], default="retriever-only")
    parser.add_argument("--num_chunks", type=int, default=10)
    parser.add_argument("--frames_per_chunk", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=3584)
    parser.add_argument("--visual_dim", type=int, default=1280)
    parser.add_argument("--budget_target", type=int, default=11520)
    parser.add_argument("--budget_hard", type=int, default=12000)
    parser.add_argument("--r_t", type=float, default=0.3333)
    parser.add_argument("--xbin", type=int, default=64)
    parser.add_argument("--ybin", type=int, default=64)
    parser.add_argument("--topM_items", type=int, default=10)
    parser.add_argument("--num_anchors", type=int, default=8)
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    build_logger(args.log_level)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    torch.manual_seed(42)

    if args.mode == "retriever-only":
        run_retriever_only(args)
    else:
        run_model_wrapped(args)


if __name__ == "__main__":
    main()
