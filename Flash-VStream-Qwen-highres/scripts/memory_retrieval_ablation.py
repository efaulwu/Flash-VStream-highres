#!/usr/bin/env python3
"""Memory retrieval ablation framework (single-file, highly configurable).

Focus scope for this stage:
- Single-frame spatial token selection.
- Cross-region / cross-feature redundancy handling.
- No motion, no optical flow, no frame-difference temporal saliency.

Main capabilities:
- Utility module: norm / rarity / local_contrast / saliency_proxy / hybrid.
- Clustering module: none / grid / kmeans_xy / kmeans_xyf / density_radius.
- Diversity module: topk / mmr / cluster_mmr / facility_coverage / nms.
- Coverage module: none / per_grid_quota / per_cluster_quota / coverage_reward / region_balancing.
- Unified logging, visualization, metrics, ablation list generation.
- Batch experiments and Slurm script generation/submission (MIG 40G style).

Input data format options:
1) Manifest JSON/JSONL with per-video entries and per-frame token references.
2) Synthetic generator for sanity checks and script generation dry-runs.

Manifest example (JSON list):
[
  {
    "video_id": "vid_001",
    "frames": [
      {
        "frame_id": 0,
        "frame_path": "/abs/path/frame000.jpg",
        "token_path": "/abs/path/frame000_tokens.pt"
      }
    ]
  }
]

Supported token file keys (.pt/.pth/.npz):
- required: tokens [N, D]
- optional: x_norm [N], y_norm [N], token_ids [N]
"""

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


LOGGER = logging.getLogger("memory_retrieval_ablation")


def _now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def _normalize(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(v, p=2, dim=dim, eps=1e-6)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _hash_color(idx: int) -> Tuple[int, int, int]:
    h = hashlib.md5(str(idx).encode("utf-8")).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (55 + r // 2, 55 + g // 2, 55 + b // 2)


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolution_bucket(w: int, h: int, cfg: argparse.Namespace) -> str:
    if w <= 0 or h <= 0:
        return "unknown"
    mode = str(getattr(cfg, "resolution_bucket_mode", "short_side")).lower()
    if mode == "exact":
        return f"{w}x{h}"
    s = min(w, h)
    if s <= int(cfg.resolution_bin1):
        return f"short_le_{int(cfg.resolution_bin1)}"
    if s <= int(cfg.resolution_bin2):
        return f"short_{int(cfg.resolution_bin1)+1}_{int(cfg.resolution_bin2)}"
    if s <= int(cfg.resolution_bin3):
        return f"short_{int(cfg.resolution_bin2)+1}_{int(cfg.resolution_bin3)}"
    return f"short_gt_{int(cfg.resolution_bin3)}"


def _sample_videos_with_resolution(videos: List[Dict[str, Any]], cfg: argparse.Namespace) -> List[Dict[str, Any]]:
    if cfg.max_videos <= 0 or len(videos) <= cfg.max_videos:
        return videos

    if not _as_bool(getattr(cfg, "resolution_stratified_sample", True)):
        rnd = random.Random(cfg.seed)
        return rnd.sample(videos, cfg.max_videos)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for v in videos:
        b = str(v.get("resolution_bucket", "unknown"))
        grouped.setdefault(b, []).append(v)

    rnd = random.Random(cfg.seed)
    buckets = sorted(grouped.keys())
    for b in buckets:
        rnd.shuffle(grouped[b])

    # Round-robin pick to ensure multiple resolution buckets are represented.
    ptr: Dict[str, int] = {b: 0 for b in buckets}
    picked: List[Dict[str, Any]] = []
    while len(picked) < cfg.max_videos:
        progressed = False
        for b in buckets:
            i = ptr[b]
            if i < len(grouped[b]):
                picked.append(grouped[b][i])
                ptr[b] += 1
                progressed = True
                if len(picked) >= cfg.max_videos:
                    break
        if not progressed:
            break
    return picked


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2, ensure_ascii=False)


def _write_status_files(out_root: Path, status: Dict[str, Any]) -> None:
    _write_json(out_root / "run_status.json", status)
    lines = []
    lines.append("# 运行状态")
    lines.append("")
    lines.append(f"- 开始时间: {status.get('start_time', '')}")
    lines.append(f"- 结束时间: {status.get('end_time', '')}")
    lines.append(f"- 停止原因: {status.get('stop_reason', '')}")
    lines.append(f"- 实际处理视频数: {status.get('processed_videos', 0)}")
    lines.append(f"- 计划视频数: {status.get('planned_videos', 0)}")
    lines.append(f"- 最近处理视频: {status.get('last_video_id', '')}")
    lines.append(f"- 是否异常: {bool(status.get('exception'))}")
    if status.get("exception"):
        lines.append(f"- 异常信息: {status.get('exception')}")
    lines.append("- 资源释放风险: 未发现常驻线程/进程；文件写入使用 with 上下文自动关闭。")
    (out_root / "运行状态.md").write_text("\n".join(lines), encoding="utf-8")


def _write_method_cn(path: Path, exp_cfg: Dict[str, Any], cfg: argparse.Namespace) -> None:
    lines = []
    lines.append("# 方法说明")
    lines.append("")
    lines.append(f"- 实验名: {exp_cfg.get('name', '')}")
    lines.append(f"- utility: {exp_cfg.get('utility_method', '')}")
    lines.append(f"- clustering/region: {exp_cfg.get('clustering_method', '')}")
    lines.append(f"- diversity: {exp_cfg.get('diversity_method', '')}")
    lines.append(f"- coverage: {exp_cfg.get('coverage_method', '')}")
    lines.append("")
    lines.append("## 实验目标")
    lines.append("验证在统一预算下，不同 region discovery / diversity / coverage 组合对 token 选择分布与可解释性的影响。")
    lines.append("")
    lines.append("## 公式")
    lines.append("- MMR: $s(i)=u(i)-\\lambda\\max_{j\\in S}\\mathrm{sim}(i,j)$")
    lines.append("- 覆盖奖励(示例): $b(i)\\propto d(i,S)$")
    lines.append("")
    lines.append("## 代码位置")
    lines.append("- Utility: UtilityComputer.compute")
    lines.append("- Region/Clustering: Clusterer.cluster")
    lines.append("- Selection: Selector.select")
    lines.append("- Coverage: Selector._coverage_bonus / _build_coverage_caps")
    lines.append("")
    lines.append("## 关键参数")
    lines.append(f"- topk: {exp_cfg.get('topk', cfg.topk)}")
    lines.append(f"- seed: {exp_cfg.get('seed', cfg.seed)}")
    lines.append(f"- max_videos: {cfg.max_videos}")
    lines.append(f"- max_runtime_minutes: {cfg.max_runtime_minutes}")
    lines.append("")
    lines.append("## 输出文件")
    lines.append("- config.json")
    lines.append("- frame_summary.jsonl")
    lines.append("- frame_importance_summary.json / .csv")
    lines.append("- metrics/frame_metrics.json")
    lines.append("- metrics/summary_metrics.json")
    lines.append("- 运行状态.md")
    lines.append("- visualizations/")
    path.write_text("\n".join(lines), encoding="utf-8")


def _compute_frame_importance(all_frame_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in all_frame_metrics:
        num_tokens = max(1, int(m.get("num_tokens", 0)))
        num_selected = int(m.get("num_selected", 0))
        selected_ratio = float(num_selected / num_tokens)
        mean_sel = float(m.get("utility_after_mean", 0.0))
        max_sel = float(m.get("utility_after_mean", 0.0))
        cluster_cnt = int(m.get("num_clusters", 0))
        score = 0.45 * selected_ratio + 0.35 * mean_sel + 0.2 * float(m.get("coverage_ratio", 0.0))
        rows.append(
            {
                "experiment": m.get("experiment", ""),
                "video_id": m.get("video_id", ""),
                "frame_id": int(m.get("frame_id", -1)),
                "selected_tokens": num_selected,
                "selected_ratio": selected_ratio,
                "mean_selected_utility": mean_sel,
                "max_selected_utility": max_sel,
                "selected_cluster_count": cluster_cnt,
                "frame_score": float(score),
            }
        )
    rows.sort(key=lambda x: x["frame_score"], reverse=True)
    return rows


def _save_frame_importance(out_root: Path, frame_rows: List[Dict[str, Any]]) -> None:
    _write_json(out_root / "frame_importance_summary.json", frame_rows)
    csv_path = out_root / "frame_importance_summary.csv"
    if frame_rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
            writer.writeheader()
            for r in frame_rows:
                writer.writerow(r)


def _write_dataset_top3_check(summary: Dict[str, Any], out_root: Path) -> None:
    lines = ["# 数据集 Top3 检查", ""]
    lines.append("## 分辨率最高 Top3")
    for r in summary.get("groups", {}).get("high_resolution", []):
        lines.append(
            f"- {r.get('video_id', '')} | {r.get('width', 0)}x{r.get('height', 0)} | frames={r.get('num_frames', 0)} | rel={r.get('video_rel', '')}"
        )
    lines.append("")
    lines.append("## 帧数最多 Top3")
    for r in summary.get("groups", {}).get("long_video", []):
        lines.append(
            f"- {r.get('video_id', '')} | {r.get('width', 0)}x{r.get('height', 0)} | frames={r.get('num_frames', 0)} | rel={r.get('video_rel', '')}"
        )
    lines.append("")
    (out_root / "dataset_top3_check.md").write_text("\n".join(lines), encoding="utf-8")


def _write_video_info_file(video_dir: Path, video_entry: Dict[str, Any], loaded_frames: int) -> None:
    _ensure_dir(video_dir)
    w = int(video_entry.get("resolution_w", 0))
    h = int(video_entry.get("resolution_h", 0))
    lines = ["# 视频信息", ""]
    lines.append(f"- video_id: {video_entry.get('video_id', '')}")
    lines.append(f"- video_rel: {video_entry.get('video_rel', '')}")
    lines.append(f"- resolution: {w}x{h}")
    total_frames = int(video_entry.get("num_frames_total", loaded_frames))
    lines.append(f"- frame_count_loaded: {loaded_frames}")
    lines.append(f"- frame_count_total: {total_frames}")
    (video_dir / "video_info.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json(
        video_dir / "video_info.json",
        {
            "video_id": str(video_entry.get("video_id", "")),
            "video_rel": str(video_entry.get("video_rel", "")),
            "resolution_w": w,
            "resolution_h": h,
            "frame_count_loaded": int(loaded_frames),
            "frame_count_total": int(total_frames),
        },
    )


def _summarize_dataset_selection(videos: List[Dict[str, Any]], cfg: argparse.Namespace, out_root: Path) -> Dict[str, Any]:
    rows = []
    for v in videos:
        w = int(v.get("resolution_w", 0))
        h = int(v.get("resolution_h", 0))
        nf = int(v.get("num_frames_total", len(v.get("frames", []))))
        fps = float(getattr(cfg, "dataset_assumed_fps", 1.0))
        dur = float(nf / max(1e-6, fps))
        pressure = int(max(1, w) * max(1, h) * max(1, nf))
        rows.append(
            {
                "video_id": str(v.get("video_id", "")),
                "video_rel": str(v.get("video_rel", "")),
                "width": w,
                "height": h,
                "num_frames": nf,
                "fps": fps,
                "duration_sec": dur,
                "pressure": pressure,
            }
        )

    by_res = sorted(rows, key=lambda r: (r["width"] * r["height"], r["num_frames"]), reverse=True)
    by_len = sorted(rows, key=lambda r: (r["num_frames"], r["width"] * r["height"]), reverse=True)
    by_pressure = sorted(rows, key=lambda r: r["pressure"], reverse=True)
    k_axis = max(1, int(getattr(cfg, "selection_topk_per_axis", 3)))
    k_summary = max(k_axis, int(cfg.dataset_selection_topn))

    summary = {
        "groups": {
            "high_resolution": by_res[:k_axis],
            "long_video": by_len[:k_axis],
            "high_pressure": by_pressure[:k_summary],
        },
        "all_stats": rows,
    }
    _write_json(out_root / "dataset_selection_summary.json", summary)

    lines = ["# 数据集样本筛选摘要", ""]
    lines.append(f"- 总视频数: {len(rows)}")
    lines.append(f"- 每轴TopK(分辨率/帧数): {k_axis}")
    lines.append(f"- 摘要TopN: {k_summary}")
    lines.append("")
    for g in ["high_resolution", "long_video", "high_pressure"]:
        lines.append(f"## {g}")
        for r in summary["groups"][g]:
            lines.append(
                f"- {r['video_id']} | {r['width']}x{r['height']} | frames={r['num_frames']} | pressure={r['pressure']}"
            )
        lines.append("")
    (out_root / "dataset_selection_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def _pick_preferred_videos(videos: List[Dict[str, Any]], summary: Dict[str, Any], cfg: argparse.Namespace) -> List[Dict[str, Any]]:
    if not videos:
        return videos
    wanted = set()
    for g in ["high_resolution", "long_video"]:
        for r in summary.get("groups", {}).get(g, []):
            wanted.add(str(r.get("video_id", "")))
    preferred = [v for v in videos if str(v.get("video_id", "")) in wanted]
    others = [v for v in videos if str(v.get("video_id", "")) not in wanted]
    if _as_bool(getattr(cfg, "top3_priority_only", True)):
        return preferred
    merged = preferred + others
    if cfg.max_videos > 0:
        target = max(int(cfg.max_videos), len(preferred))
        return merged[:target]
    return merged


class FrameTokens:
    def __init__(
        self,
        video_id: str,
        frame_id: int,
        tokens: torch.Tensor,
        x_norm: torch.Tensor,
        y_norm: torch.Tensor,
        frame_path: Optional[str],
        token_ids: torch.Tensor,
    ):
        self.video_id = video_id
        self.frame_id = frame_id
        self.tokens = tokens
        self.x_norm = x_norm
        self.y_norm = y_norm
        self.frame_path = frame_path
        self.token_ids = token_ids


class ClusterResult:
    def __init__(
        self,
        cluster_ids: torch.Tensor,
        centers_xy: torch.Tensor,
        cluster_sizes: Dict[int, int],
        cluster_avg_utility: Dict[int, float],
        cluster_max_utility: Dict[int, float],
        cluster_token_indices: Dict[int, List[int]],
    ):
        self.cluster_ids = cluster_ids
        self.centers_xy = centers_xy
        self.cluster_sizes = cluster_sizes
        self.cluster_avg_utility = cluster_avg_utility
        self.cluster_max_utility = cluster_max_utility
        self.cluster_token_indices = cluster_token_indices


class UtilityComputer:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg

    def compute(self, tokens: torch.Tensor, x: torch.Tensor, y: torch.Tensor, method: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        method = method.lower()
        details: Dict[str, torch.Tensor] = {}
        norm_u = torch.norm(tokens, p=2, dim=-1)
        details["norm"] = norm_u

        rarity_u = self._rarity(tokens)
        details["rarity"] = rarity_u

        local_u = self._local_contrast(tokens, x, y)
        details["local_contrast"] = local_u

        saliency_u = 0.55 * self._zscore(norm_u) + 0.45 * self._zscore(rarity_u)
        details["saliency_proxy"] = saliency_u

        if method == "norm":
            util = norm_u
        elif method == "rarity_within_frame":
            util = rarity_u
        elif method == "local_contrast":
            util = local_u
        elif method == "global_saliency_proxy":
            util = saliency_u
        elif method == "hybrid":
            a = self.cfg.hybrid_norm_w
            b = self.cfg.hybrid_rarity_w
            c = self.cfg.hybrid_local_w
            util = a * self._zscore(norm_u) + b * self._zscore(rarity_u) + c * self._zscore(local_u)
        elif method == "hybrid_norm_rarity":
            util = 0.5 * self._zscore(norm_u) + 0.5 * self._zscore(rarity_u)
        elif method == "hybrid_rarity_localcontrast":
            util = 0.5 * self._zscore(rarity_u) + 0.5 * self._zscore(local_u)
        else:
            raise ValueError(f"Unknown utility method: {method}")

        util = self._zscore(util)
        details["final"] = util
        return util, details

    def _zscore(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        m = x.mean()
        s = x.std(unbiased=False)
        return (x - m) / (s + 1e-6)

    def _rarity(self, tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[0]
        if n <= 1:
            return torch.ones(n, device=tokens.device)
        max_n = max(2, self.cfg.utility_pairwise_max_tokens)
        if n > max_n:
            idx = torch.linspace(0, n - 1, steps=max_n, device=tokens.device).long()
            ref = tokens[idx]
        else:
            ref = tokens
        q = _normalize(tokens)
        r = _normalize(ref)
        sim = torch.matmul(q, r.t())
        mean_sim = sim.mean(dim=1)
        return 1.0 - mean_sim

    def _local_contrast(self, tokens: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[0]
        if n <= 1:
            return torch.zeros(n, device=tokens.device)
        coords = torch.stack([x, y], dim=1)
        dist = torch.cdist(coords, coords)
        radius = max(1e-4, self.cfg.local_radius)
        local_mask = dist <= radius
        tok = _normalize(tokens)
        sim = torch.matmul(tok, tok.t())
        local_mean = []
        for i in range(n):
            m = local_mask[i]
            m[i] = False
            if int(m.sum().item()) == 0:
                local_mean.append(torch.tensor(0.0, device=tokens.device))
            else:
                local_mean.append(sim[i][m].mean())
        lm = torch.stack(local_mean)
        return 1.0 - lm


class Clusterer:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg

    def cluster(self, tokens: torch.Tensor, x: torch.Tensor, y: torch.Tensor, utility: torch.Tensor, method: str) -> ClusterResult:
        method = method.lower()
        aliases = {
            "baseline_none": "none",
            "grid_clustering": "grid",
            "slic_superpixel": "kmeans_xyf",
            "felzenszwalb_segmentation": "density_radius",
            "edge_connected_components": "grid",
            "optical_flow_region": "kmeans_xy",
            "simple_pixel_segmentation": "kmeans_xyf",
        }
        method = aliases.get(method, method)
        n = tokens.shape[0]
        if n == 0:
            return ClusterResult(
                cluster_ids=torch.empty(0, dtype=torch.long),
                centers_xy=torch.empty(0, 2),
                cluster_sizes={},
                cluster_avg_utility={},
                cluster_max_utility={},
                cluster_token_indices={},
            )

        if method == "none":
            cluster_ids = torch.zeros(n, dtype=torch.long)
        elif method == "grid":
            cluster_ids = self._grid_cluster(x, y)
        elif method == "kmeans_xy":
            cluster_ids = self._kmeans_xy(x, y)
        elif method == "kmeans_xyf":
            cluster_ids = self._kmeans_xyf(tokens, x, y)
        elif method == "density_radius":
            cluster_ids = self._density_radius_cluster(x, y)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        return self._build_cluster_result(cluster_ids, x, y, utility)

    def _grid_cluster(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gx = max(1, self.cfg.grid_cluster_x)
        gy = max(1, self.cfg.grid_cluster_y)
        ix = torch.clamp((x * gx).long(), min=0, max=gx - 1)
        iy = torch.clamp((y * gy).long(), min=0, max=gy - 1)
        return iy * gx + ix

    def _kmeans_xy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        data = torch.stack([x, y], dim=1)
        return self._kmeans(data, k=max(1, self.cfg.kmeans_k), iters=max(1, self.cfg.kmeans_iters))

    def _kmeans_xyf(self, tokens: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        feat = _normalize(tokens)
        d = feat.shape[1]
        fd = min(d, max(1, self.cfg.kmeans_feature_dims))
        feat = feat[:, :fd]
        pw = self.cfg.kmeans_pos_weight
        fw = self.cfg.kmeans_feat_weight
        data = torch.cat([pw * torch.stack([x, y], dim=1), fw * feat], dim=1)
        return self._kmeans(data, k=max(1, self.cfg.kmeans_k), iters=max(1, self.cfg.kmeans_iters))

    def _kmeans(self, data: torch.Tensor, k: int, iters: int) -> torch.Tensor:
        n = data.shape[0]
        k = min(k, n)
        if k <= 1:
            return torch.zeros(n, dtype=torch.long)
        g = torch.Generator(device=data.device)
        g.manual_seed(self.cfg.seed)
        init_ids = torch.randperm(n, generator=g, device=data.device)[:k]
        centers = data[init_ids].clone()
        assign = torch.zeros(n, dtype=torch.long, device=data.device)
        for _ in range(iters):
            dist = torch.cdist(data, centers)
            assign = torch.argmin(dist, dim=1)
            for cid in range(k):
                m = assign == cid
                if int(m.sum().item()) > 0:
                    centers[cid] = data[m].mean(dim=0)
        return assign.cpu()

    def _density_radius_cluster(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        coords = torch.stack([x, y], dim=1)
        n = coords.shape[0]
        radius = max(1e-4, self.cfg.density_radius)
        min_pts = max(1, self.cfg.density_min_pts)
        dist = torch.cdist(coords, coords)
        visited = torch.zeros(n, dtype=torch.bool)
        cluster_ids = torch.full((n,), -1, dtype=torch.long)
        cur_cid = 0

        for i in range(n):
            if bool(visited[i]):
                continue
            visited[i] = True
            neigh = torch.where(dist[i] <= radius)[0]
            if int(neigh.numel()) < min_pts:
                continue
            queue = neigh.tolist()
            cluster_ids[i] = cur_cid
            qh = 0
            while qh < len(queue):
                j = int(queue[qh])
                qh += 1
                if not bool(visited[j]):
                    visited[j] = True
                    n2 = torch.where(dist[j] <= radius)[0]
                    if int(n2.numel()) >= min_pts:
                        queue.extend([int(v) for v in n2.tolist()])
                if int(cluster_ids[j].item()) < 0:
                    cluster_ids[j] = cur_cid
            cur_cid += 1

        # Attach noise points to nearest center-like point for stability.
        if cur_cid == 0:
            return torch.zeros(n, dtype=torch.long)
        noise = torch.where(cluster_ids < 0)[0]
        if int(noise.numel()) > 0:
            for idx in noise.tolist():
                d = dist[idx].clone()
                d[idx] = 1e9
                nn = int(torch.argmin(d).item())
                nn_cid = int(cluster_ids[nn].item())
                cluster_ids[idx] = nn_cid if nn_cid >= 0 else 0
        return cluster_ids

    def _build_cluster_result(self, cluster_ids: torch.Tensor, x: torch.Tensor, y: torch.Tensor, utility: torch.Tensor) -> ClusterResult:
        cids = sorted(set(int(v) for v in cluster_ids.tolist()))
        centers = []
        cluster_sizes: Dict[int, int] = {}
        cluster_avg_utility: Dict[int, float] = {}
        cluster_max_utility: Dict[int, float] = {}
        cluster_token_indices: Dict[int, List[int]] = {}

        for cid in cids:
            idx = torch.where(cluster_ids == cid)[0]
            if idx.numel() == 0:
                continue
            cx = float(x[idx].mean().item())
            cy = float(y[idx].mean().item())
            centers.append([cx, cy])
            cluster_sizes[cid] = int(idx.numel())
            cluster_avg_utility[cid] = float(utility[idx].mean().item())
            cluster_max_utility[cid] = float(utility[idx].max().item())
            cluster_token_indices[cid] = [int(v) for v in idx.tolist()]

        centers_xy = torch.tensor(centers, dtype=torch.float32) if centers else torch.empty(0, 2)
        return ClusterResult(
            cluster_ids=cluster_ids,
            centers_xy=centers_xy,
            cluster_sizes=cluster_sizes,
            cluster_avg_utility=cluster_avg_utility,
            cluster_max_utility=cluster_max_utility,
            cluster_token_indices=cluster_token_indices,
        )


class Selector:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg

    def select(
        self,
        tokens: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        utility: torch.Tensor,
        clusters: ClusterResult,
        budget: int,
        diversity_method: str,
        coverage_method: str,
    ) -> Tuple[List[int], Dict[str, Any]]:
        n = tokens.shape[0]
        if n == 0 or budget <= 0:
            return [], {"trace": [], "reason": "empty"}

        diversity_method = diversity_method.lower()
        coverage_method = coverage_method.lower()
        diversity_alias = {
            "topk_utility": "topk",
            "mmr_token": "mmr",
            "mmr_cluster": "cluster_mmr",
            "redundancy_suppression": "nms",
            "greedy_representative_selection": "facility_coverage",
        }
        coverage_alias = {
            "crowding_penalty": "region_balancing",
        }
        diversity_method = diversity_alias.get(diversity_method, diversity_method)
        coverage_method = coverage_alias.get(coverage_method, coverage_method)

        token_norm = _normalize(tokens)
        cluster_ids = clusters.cluster_ids
        all_ids = list(range(n))
        selected: List[int] = []
        trace: List[Dict[str, Any]] = []
        selected_cluster_order: List[int] = []

        grid_caps, cluster_caps = self._build_coverage_caps(coverage_method, x, y, cluster_ids, utility, budget, clusters)
        grid_counts: Dict[Tuple[int, int], int] = {}
        cluster_counts: Dict[int, int] = {}

        # Precompute cluster-level priorities for cluster_mmr.
        cluster_rank_bonus: Dict[int, float] = {}
        if diversity_method == "cluster_mmr":
            cluster_rank_bonus = self._cluster_mmr_priority(tokens, clusters)

        support_ids = all_ids
        if len(support_ids) > self.cfg.facility_support_max:
            support_ids = support_ids[:: max(1, len(support_ids) // self.cfg.facility_support_max)]
        support = token_norm[support_ids]
        covered_gain = torch.zeros(len(support_ids), dtype=torch.float32, device=tokens.device)

        for step in range(min(budget, n)):
            best_idx = -1
            best_score = -1e18
            best_detail: Dict[str, Any] = {}

            for i in all_ids:
                if i in selected:
                    continue
                cid = int(cluster_ids[i].item())
                cell = self._coarse_cell(x[i], y[i])
                if not self._check_hard_caps(i, cid, cell, coverage_method, grid_caps, cluster_caps, grid_counts, cluster_counts):
                    continue

                relevance = float(utility[i].item())
                redundancy = self._redundancy(token_norm[i], token_norm[selected] if selected else None)
                score = relevance
                detail = {
                    "relevance": relevance,
                    "redundancy": redundancy,
                    "coverage_bonus": 0.0,
                    "cluster_bonus": 0.0,
                }

                if diversity_method == "mmr":
                    score = relevance - self.cfg.mmr_lambda * redundancy
                elif diversity_method == "cluster_mmr":
                    cbonus = cluster_rank_bonus.get(cid, 0.0)
                    score = relevance + cbonus - self.cfg.mmr_lambda * redundancy
                    detail["cluster_bonus"] = cbonus
                elif diversity_method == "facility_coverage":
                    # Greedy representative selection: maximize relevance + marginal support coverage.
                    sim_i = torch.matmul(support, token_norm[i])
                    marginal = torch.clamp(sim_i - covered_gain, min=0.0).mean().item()
                    score = relevance + self.cfg.facility_alpha * float(marginal)
                    detail["facility_marginal"] = float(marginal)
                elif diversity_method == "nms":
                    if redundancy > self.cfg.nms_sim_threshold:
                        continue
                    score = relevance
                elif diversity_method == "topk":
                    score = relevance
                else:
                    raise ValueError(f"Unknown diversity method: {diversity_method}")

                cov_bonus = self._coverage_bonus(
                    coverage_method,
                    i,
                    cid,
                    cell,
                    x,
                    y,
                    selected,
                    grid_counts,
                    cluster_counts,
                )
                score += cov_bonus
                detail["coverage_bonus"] = cov_bonus

                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_detail = detail

            if best_idx < 0:
                break

            selected.append(best_idx)
            scid = int(cluster_ids[best_idx].item())
            selected_cluster_order.append(scid)
            cell = self._coarse_cell(x[best_idx], y[best_idx])
            grid_counts[cell] = grid_counts.get(cell, 0) + 1
            cluster_counts[scid] = cluster_counts.get(scid, 0) + 1

            if diversity_method == "facility_coverage":
                sim_sel = torch.matmul(support, token_norm[best_idx])
                covered_gain = torch.maximum(covered_gain, sim_sel)

            trace.append(
                {
                    "step": step,
                    "token_idx": int(best_idx),
                    "cluster_id": int(scid),
                    "x": float(x[best_idx].item()),
                    "y": float(y[best_idx].item()),
                    "score": float(best_score),
                    **{k: _safe_float(v) for k, v in best_detail.items()},
                }
            )

        return selected, {
            "trace": trace,
            "grid_counts": {f"{k[0]}_{k[1]}": int(v) for k, v in grid_counts.items()},
            "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
            "selected_cluster_order": selected_cluster_order,
            "grid_caps": {f"{k[0]}_{k[1]}": _to_jsonable(v) for k, v in grid_caps.items()},
            "cluster_caps": {str(k): _to_jsonable(v) for k, v in cluster_caps.items()},
        }

    def _build_coverage_caps(
        self,
        coverage_method: str,
        x: torch.Tensor,
        y: torch.Tensor,
        cluster_ids: torch.Tensor,
        utility: torch.Tensor,
        budget: int,
        clusters: ClusterResult,
    ) -> Tuple[Dict[Tuple[int, int], Dict[str, int]], Dict[int, Dict[str, int]]]:
        grid_caps: Dict[Tuple[int, int], Dict[str, int]] = {}
        cluster_caps: Dict[int, Dict[str, int]] = {}

        if coverage_method == "per_grid_quota":
            for i in range(x.shape[0]):
                cell = self._coarse_cell(x[i], y[i])
                if cell not in grid_caps:
                    grid_caps[cell] = {
                        "hard_max": max(1, self.cfg.grid_quota_hard_max),
                        "soft_max": max(1, self.cfg.grid_quota_soft_max),
                        "hard_min": max(0, self.cfg.grid_quota_hard_min),
                    }

        if coverage_method == "per_cluster_quota":
            cids = sorted(clusters.cluster_sizes.keys())
            if not cids:
                return grid_caps, cluster_caps
            weights = []
            for cid in cids:
                size = float(clusters.cluster_sizes[cid])
                avg_u = float(clusters.cluster_avg_utility[cid])
                if self.cfg.cluster_quota_strategy == "uniform":
                    w = 1.0
                elif self.cfg.cluster_quota_strategy == "size":
                    w = size
                elif self.cfg.cluster_quota_strategy == "utility":
                    w = max(1e-6, avg_u - min(clusters.cluster_avg_utility.values()) + 1e-3)
                elif self.cfg.cluster_quota_strategy == "sqrt_size":
                    w = math.sqrt(max(1.0, size))
                else:
                    raise ValueError(f"Unknown cluster quota strategy: {self.cfg.cluster_quota_strategy}")
                weights.append(w)
            s = sum(weights)
            alloc = [max(1, int(round(budget * (w / max(1e-6, s))))) for w in weights]
            # Normalize allocation sum.
            while sum(alloc) > budget:
                j = int(np.argmax(alloc))
                if alloc[j] > 1:
                    alloc[j] -= 1
                else:
                    break
            while sum(alloc) < budget:
                j = int(np.argmin(alloc))
                alloc[j] += 1
            for cid, a in zip(cids, alloc):
                cluster_caps[cid] = {
                    "hard_max": int(max(1, a)),
                    "soft_max": int(max(1, int(round(a * self.cfg.cluster_quota_soft_scale)))),
                }

        return grid_caps, cluster_caps

    def _cluster_mmr_priority(self, tokens: torch.Tensor, clusters: ClusterResult) -> Dict[int, float]:
        cluster_ids = sorted(clusters.cluster_token_indices.keys())
        if not cluster_ids:
            return {}
        reps = []
        for cid in cluster_ids:
            idx = clusters.cluster_token_indices[cid]
            t = tokens[idx]
            if self.cfg.cluster_repr == "max":
                reps.append(t[torch.argmax(torch.norm(t, p=2, dim=1)).item()])
            elif self.cfg.cluster_repr == "weighted_mean":
                w = torch.softmax(torch.norm(t, p=2, dim=1), dim=0).unsqueeze(1)
                reps.append((t * w).sum(dim=0))
            else:
                reps.append(t.mean(dim=0))
        reps_t = _normalize(torch.stack(reps, dim=0))

        selected: List[int] = []
        bonus: Dict[int, float] = {}
        for _ in range(len(cluster_ids)):
            best_j = -1
            best_s = -1e9
            for j, cid in enumerate(cluster_ids):
                if j in selected:
                    continue
                rel = clusters.cluster_avg_utility.get(cid, 0.0)
                if selected:
                    red = float(torch.max(torch.matmul(reps_t[j : j + 1], reps_t[selected].t())).item())
                else:
                    red = 0.0
                s = rel - self.cfg.cluster_mmr_lambda * red
                if s > best_s:
                    best_s = s
                    best_j = j
            if best_j < 0:
                break
            selected.append(best_j)
            cid = cluster_ids[best_j]
            # Higher rank gets larger bonus.
            rank = len(selected) - 1
            bonus[cid] = float(self.cfg.cluster_rank_bonus_scale * (1.0 / (1.0 + rank)))
        return bonus

    def _coarse_cell(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[int, int]:
        gx = max(1, self.cfg.coverage_grid_x)
        gy = max(1, self.cfg.coverage_grid_y)
        ix = max(0, min(gx - 1, int(math.floor(float(x.item()) * gx))))
        iy = max(0, min(gy - 1, int(math.floor(float(y.item()) * gy))))
        return (ix, iy)

    def _check_hard_caps(
        self,
        idx: int,
        cid: int,
        cell: Tuple[int, int],
        coverage_method: str,
        grid_caps: Dict[Tuple[int, int], Dict[str, int]],
        cluster_caps: Dict[int, Dict[str, int]],
        grid_counts: Dict[Tuple[int, int], int],
        cluster_counts: Dict[int, int],
    ) -> bool:
        if coverage_method == "per_grid_quota":
            cap = grid_caps.get(cell)
            if cap is not None and grid_counts.get(cell, 0) >= cap.get("hard_max", 10**9):
                return False
        if coverage_method == "per_cluster_quota":
            cap = cluster_caps.get(cid)
            if cap is not None and cluster_counts.get(cid, 0) >= cap.get("hard_max", 10**9):
                return False
        return True

    def _coverage_bonus(
        self,
        coverage_method: str,
        idx: int,
        cid: int,
        cell: Tuple[int, int],
        x: torch.Tensor,
        y: torch.Tensor,
        selected: List[int],
        grid_counts: Dict[Tuple[int, int], int],
        cluster_counts: Dict[int, int],
    ) -> float:
        if coverage_method == "none":
            return 0.0
        if coverage_method == "per_grid_quota":
            c = grid_counts.get(cell, 0)
            soft = max(1, self.cfg.grid_quota_soft_max)
            return float(self.cfg.coverage_bonus_scale * (1.0 - c / soft))
        if coverage_method == "per_cluster_quota":
            c = cluster_counts.get(cid, 0)
            return float(self.cfg.coverage_bonus_scale * (1.0 / (1.0 + c)))
        if coverage_method == "coverage_reward":
            if not selected:
                return self.cfg.coverage_bonus_scale
            sx = x[selected]
            sy = y[selected]
            dx = sx - x[idx]
            dy = sy - y[idx]
            d = torch.sqrt(dx * dx + dy * dy + 1e-8)
            min_d = float(torch.min(d).item())
            return float(self.cfg.coverage_bonus_scale * min(min_d / max(1e-4, self.cfg.coverage_decay_radius), 1.0))
        if coverage_method == "region_balancing":
            c = grid_counts.get(cell, 0)
            return float(self.cfg.coverage_bonus_scale * (1.0 / (1.0 + c)))
        if coverage_method == "crowding_penalty":
            if not selected:
                return 0.0
            sx = x[selected]
            sy = y[selected]
            dx = sx - x[idx]
            dy = sy - y[idx]
            d = torch.sqrt(dx * dx + dy * dy + 1e-8)
            nearest = float(torch.min(d).item()) if d.numel() > 0 else 1.0
            return float(-self.cfg.coverage_bonus_scale * max(0.0, 1.0 - nearest / max(1e-4, self.cfg.coverage_decay_radius)))
        raise ValueError(f"Unknown coverage method: {coverage_method}")

    def _redundancy(self, token: torch.Tensor, selected_tokens: Optional[torch.Tensor]) -> float:
        if selected_tokens is None or selected_tokens.numel() == 0:
            return 0.0
        sim = torch.matmul(selected_tokens, token)
        return float(torch.max(sim).item())


class Visualizer:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg

    def draw_all(
        self,
        frame: FrameTokens,
        utility: torch.Tensor,
        clusters: ClusterResult,
        selected: List[int],
        stage_records: Dict[str, List[int]],
        frame_out_dir: Path,
        exp_name: str,
        video_id: str,
    ) -> Dict[str, str]:
        all_dir = frame_out_dir / "all_frames"
        temporal_dir = frame_out_dir / "temporal_tokens"
        spatial_dir = frame_out_dir / "spatial_tokens"
        _ensure_dir(all_dir)
        _ensure_dir(temporal_dir)
        _ensure_dir(spatial_dir)
        base_img = self._load_base(frame)

        paths: Dict[str, str] = {}
        p_all = all_dir / f"{exp_name}_{video_id}_f{frame.frame_id:05d}_all_grid.png"
        self._draw_selection(base_img.copy(), frame, [], clusters, title="all-frame grid (no temporal filter)").save(p_all)
        paths["all_grid"] = str(p_all)

        p1 = spatial_dir / f"{exp_name}_{video_id}_f{frame.frame_id:05d}_01_raw.png"
        self._draw_raw_distribution(base_img.copy(), frame, utility).save(p1)
        paths["raw_distribution"] = str(p1)

        p2 = spatial_dir / f"{exp_name}_{video_id}_f{frame.frame_id:05d}_02_cluster.png"
        self._draw_clusters(base_img.copy(), frame, utility, clusters).save(p2)
        paths["clustering"] = str(p2)

        p3 = spatial_dir / f"{exp_name}_{video_id}_f{frame.frame_id:05d}_03_selection.png"
        self._draw_selection(base_img.copy(), frame, selected, clusters).save(p3)
        paths["selection"] = str(p3)

        stage_paths = []
        stage_order = ["utility_only", "clustering_only", "cluster_diversity", "full_pipeline"]
        for stage in stage_order:
            ids = stage_records.get(stage, [])
            p = spatial_dir / f"{exp_name}_{video_id}_f{frame.frame_id:05d}_stage_{stage}.png"
            img = self._draw_selection(base_img.copy(), frame, ids, clusters, title=f"stage={stage}")
            img.save(p)
            paths[f"stage_{stage}"] = str(p)
            stage_paths.append(str(p))

        panel = temporal_dir / f"{exp_name}_{video_id}_f{frame.frame_id:05d}_summary_panel.png"
        self._make_summary_panel(stage_paths, panel)
        paths["summary_panel"] = str(panel)

        temporal_view = temporal_dir / f"{exp_name}_{video_id}_f{frame.frame_id:05d}_temporal_view.png"
        self._draw_selection(
            base_img.copy(),
            frame,
            stage_records.get("full_pipeline", []),
            clusters,
            title=f"temporal-token frame={int(frame.frame_id)}",
        ).save(temporal_view)
        paths["temporal_view"] = str(temporal_view)

        trace_img = spatial_dir / f"{exp_name}_{video_id}_f{frame.frame_id:05d}_cluster_trace.png"
        self._draw_cluster_trace(base_img.copy(), frame, selected, clusters).save(trace_img)
        paths["cluster_trace"] = str(trace_img)
        paths["all_dir"] = str(all_dir)
        paths["temporal_dir"] = str(temporal_dir)
        paths["spatial_dir"] = str(spatial_dir)
        return paths

    def _load_base(self, frame: FrameTokens) -> Image.Image:
        if frame.frame_path and os.path.exists(frame.frame_path):
            try:
                img = Image.open(frame.frame_path).convert("RGB")
                return img
            except Exception:
                pass
        return Image.new("RGB", (self.cfg.canvas_w, self.cfg.canvas_h), color=(18, 20, 26))

    def _xy_to_px(self, img: Image.Image, x: torch.Tensor, y: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        w, h = img.size
        px = np.clip((x.detach().cpu().numpy() * (w - 1)).astype(np.int32), 0, w - 1)
        py = np.clip((y.detach().cpu().numpy() * (h - 1)).astype(np.int32), 0, h - 1)
        return px, py

    def _draw_dashed_line(
        self,
        draw: ImageDraw.ImageDraw,
        p0: Tuple[int, int],
        p1: Tuple[int, int],
        fill: Tuple[int, int, int, int],
        width: int = 1,
        dash: int = 2,
        gap: int = 4,
    ) -> None:
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        dist = max(1, int(round(math.sqrt(dx * dx + dy * dy))))
        step = max(1, dash + gap)
        for st in range(0, dist, step):
            ed = min(dist, st + dash)
            t0 = st / float(dist)
            t1 = ed / float(dist)
            xs = int(round(x0 + dx * t0))
            ys = int(round(y0 + dy * t0))
            xe = int(round(x0 + dx * t1))
            ye = int(round(y0 + dy * t1))
            draw.line((xs, ys, xe, ye), fill=fill, width=width)

    def _draw_dashed_rect(
        self,
        draw: ImageDraw.ImageDraw,
        box: Tuple[int, int, int, int],
        fill: Tuple[int, int, int, int],
        width: int = 1,
    ) -> None:
        l, t, r, b = box
        self._draw_dashed_line(draw, (l, t), (r, t), fill=fill, width=width)
        self._draw_dashed_line(draw, (r, t), (r, b), fill=fill, width=width)
        self._draw_dashed_line(draw, (r, b), (l, b), fill=fill, width=width)
        self._draw_dashed_line(draw, (l, b), (l, t), fill=fill, width=width)

    def _unique_sorted(self, arr: np.ndarray, tol: float = 1e-4) -> List[float]:
        vals = sorted(float(v) for v in arr.tolist())
        out: List[float] = []
        for v in vals:
            if not out or abs(v - out[-1]) > tol:
                out.append(v)
        return out

    def _token_boxes(self, img: Image.Image, frame: FrameTokens) -> List[Tuple[int, int, int, int]]:
        w, h = img.size
        x = frame.x_norm.detach().cpu().numpy()
        y = frame.y_norm.detach().cpu().numpy()
        n = len(x)
        if n == 0:
            return []

        ux = self._unique_sorted(x)
        uy = self._unique_sorted(y)

        boxes: List[Tuple[int, int, int, int]] = []
        if len(ux) * len(uy) == n and len(ux) >= 2 and len(uy) >= 2:
            xb = [0.0]
            for i in range(len(ux) - 1):
                xb.append((ux[i] + ux[i + 1]) * 0.5)
            xb.append(1.0)

            yb = [0.0]
            for i in range(len(uy) - 1):
                yb.append((uy[i] + uy[i + 1]) * 0.5)
            yb.append(1.0)

            for i in range(n):
                xi = int(np.argmin(np.abs(np.asarray(ux) - x[i])))
                yi = int(np.argmin(np.abs(np.asarray(uy) - y[i])))
                l = int(round(xb[xi] * (w - 1)))
                r = int(round(xb[xi + 1] * (w - 1)))
                t = int(round(yb[yi] * (h - 1)))
                b = int(round(yb[yi + 1] * (h - 1)))
                boxes.append((max(0, l), max(0, t), min(w - 1, r), min(h - 1, b)))
            return boxes

        sx = np.sort(np.unique(x))
        sy = np.sort(np.unique(y))
        dx = float(np.min(np.diff(sx))) if len(sx) >= 2 else 1.0 / max(1, int(round(math.sqrt(n))))
        dy = float(np.min(np.diff(sy))) if len(sy) >= 2 else 1.0 / max(1, int(round(math.sqrt(n))))
        hw = max(1, int(round(0.5 * dx * (w - 1))))
        hh = max(1, int(round(0.5 * dy * (h - 1))))

        px, py = self._xy_to_px(img, frame.x_norm, frame.y_norm)
        for i in range(n):
            l = max(0, int(px[i]) - hw)
            r = min(w - 1, int(px[i]) + hw)
            t = max(0, int(py[i]) - hh)
            b = min(h - 1, int(py[i]) + hh)
            boxes.append((l, t, r, b))
        return boxes

    def _draw_raw_distribution(self, img: Image.Image, frame: FrameTokens, utility: torch.Tensor) -> Image.Image:
        draw = ImageDraw.Draw(img, "RGBA")
        boxes = self._token_boxes(img, frame)
        u = utility.detach().cpu().numpy()
        umin = float(np.min(u)) if len(u) > 0 else 0.0
        umax = float(np.max(u)) if len(u) > 0 else 1.0
        denom = max(1e-6, umax - umin)
        for i in range(len(boxes)):
            t = (u[i] - umin) / denom
            col = (int(255 * t), int(210 * (1 - t)), 80, int(50 + 120 * t))
            draw.rectangle(boxes[i], fill=col)
            self._draw_dashed_rect(draw, boxes[i], fill=(235, 235, 235, 22), width=1)
        draw.rectangle((4, 4, 260, 28), fill=(0, 0, 0, 130))
        draw.text((8, 8), "Token utility mask", fill=(230, 230, 230, 255))
        return img

    def _draw_clusters(self, img: Image.Image, frame: FrameTokens, utility: torch.Tensor, clusters: ClusterResult) -> Image.Image:
        draw = ImageDraw.Draw(img, "RGBA")
        boxes = self._token_boxes(img, frame)
        cids = clusters.cluster_ids.detach().cpu().numpy()
        for i in range(len(boxes)):
            col = _hash_color(int(cids[i]))
            draw.rectangle(boxes[i], fill=(col[0], col[1], col[2], 96))
            self._draw_dashed_rect(draw, boxes[i], fill=(235, 235, 235, 22), width=1)
        # centers
        w, h = img.size
        for cid, idxs in clusters.cluster_token_indices.items():
            xs = frame.x_norm[idxs]
            ys = frame.y_norm[idxs]
            cx = int(float(xs.mean().item()) * (w - 1))
            cy = int(float(ys.mean().item()) * (h - 1))
            col = _hash_color(int(cid))
            draw.rectangle((cx - 4, cy - 4, cx + 4, cy + 4), outline=(255, 255, 255, 255), fill=(col[0], col[1], col[2], 220))
        draw.rectangle((4, 4, 230, 28), fill=(0, 0, 0, 130))
        draw.text((8, 8), "Cluster mask", fill=(230, 230, 230, 255))
        return img

    def _draw_selection(
        self,
        img: Image.Image,
        frame: FrameTokens,
        selected: List[int],
        clusters: ClusterResult,
        title: str = "Selection visualization",
    ) -> Image.Image:
        draw = ImageDraw.Draw(img, "RGBA")
        sel = set(int(v) for v in selected)
        boxes = self._token_boxes(img, frame)
        for i in range(len(boxes)):
            if i in sel:
                cid = int(clusters.cluster_ids[i].item())
                col = _hash_color(cid)
                draw.rectangle(boxes[i], fill=(col[0], col[1], col[2], 130))
                draw.rectangle(boxes[i], outline=(255, 255, 255, 160), width=1)
            else:
                self._draw_dashed_rect(draw, boxes[i], fill=(235, 235, 235, 20), width=1)
        draw.rectangle((4, 4, 360, 28), fill=(0, 0, 0, 130))
        draw.text((8, 8), f"{title} | selected={len(selected)}", fill=(230, 230, 230, 255))
        return img

    def _draw_cluster_trace(self, img: Image.Image, frame: FrameTokens, selected: List[int], clusters: ClusterResult) -> Image.Image:
        draw = ImageDraw.Draw(img, "RGBA")
        w, _ = img.size
        boxes = self._token_boxes(img, frame)
        cluster_steps: Dict[int, List[int]] = {}
        for s, idx in enumerate(selected):
            cid = int(clusters.cluster_ids[idx].item())
            cluster_steps.setdefault(cid, []).append(s)
            col = _hash_color(cid)
            draw.rectangle(boxes[idx], fill=(col[0], col[1], col[2], 130), outline=(255, 255, 255, 160), width=1)

        for i in range(len(boxes)):
            if i not in set(selected):
                self._draw_dashed_rect(draw, boxes[i], fill=(235, 235, 235, 20), width=1)

        draw.rectangle((4, 4, w - 4, 64), fill=(0, 0, 0, 150))
        text = "Cluster selection trace: " + ", ".join([f"c{cid}:{len(steps)}" for cid, steps in sorted(cluster_steps.items())[:8]])
        draw.text((8, 8), text[:120], fill=(230, 230, 230, 255))
        return img

    def _make_summary_panel(self, image_paths: List[str], out_path: Path) -> None:
        imgs = []
        for p in image_paths:
            if os.path.exists(p):
                imgs.append(Image.open(p).convert("RGB"))
        if not imgs:
            return
        w = min(im.size[0] for im in imgs)
        h = min(im.size[1] for im in imgs)
        imgs = [im.resize((w, h)) for im in imgs]
        panel = Image.new("RGB", (w * len(imgs), h), color=(15, 15, 18))
        for i, im in enumerate(imgs):
            panel.paste(im, (i * w, 0))
        panel.save(out_path)


class Metrics:
    @staticmethod
    def compute(tokens: torch.Tensor, x: torch.Tensor, y: torch.Tensor, selected: List[int], clusters: ClusterResult, utility: torch.Tensor, cfg: argparse.Namespace) -> Dict[str, Any]:
        if not selected:
            return {
                "spatial_dispersion": 0.0,
                "spatial_compactness": 0.0,
                "redundancy_mean_sim": 0.0,
                "redundancy_max_sim": 0.0,
                "coverage_cells": 0,
                "coverage_ratio": 0.0,
                "cluster_concentration_top1": 0.0,
                "cluster_concentration_top3": 0.0,
                "utility_before_mean": float(utility.mean().item()) if utility.numel() > 0 else 0.0,
                "utility_after_mean": 0.0,
            }

        idx = torch.tensor(selected, dtype=torch.long)
        sx = x[idx]
        sy = y[idx]
        coords = torch.stack([sx, sy], dim=1)
        if coords.shape[0] > 1:
            d = torch.cdist(coords, coords)
            spatial_disp = float(d.mean().item())
            spatial_comp = float(1.0 / (d.mean().item() + 1e-6))
        else:
            spatial_disp = 0.0
            spatial_comp = 0.0

        st = _normalize(tokens[idx])
        if st.shape[0] > 1:
            sim = torch.matmul(st, st.t())
            tri = sim[~torch.eye(sim.shape[0], dtype=torch.bool)]
            red_mean = float(tri.mean().item()) if tri.numel() > 0 else 0.0
            red_max = float(tri.max().item()) if tri.numel() > 0 else 0.0
        else:
            red_mean = 0.0
            red_max = 0.0

        gx = max(1, cfg.coverage_grid_x)
        gy = max(1, cfg.coverage_grid_y)
        cells = set()
        for xv, yv in zip(sx.tolist(), sy.tolist()):
            ix = max(0, min(gx - 1, int(math.floor(xv * gx))))
            iy = max(0, min(gy - 1, int(math.floor(yv * gy))))
            cells.add((ix, iy))
        coverage_cells = len(cells)
        coverage_ratio = float(coverage_cells / float(gx * gy))

        cc: Dict[int, int] = {}
        for i in selected:
            cid = int(clusters.cluster_ids[i].item())
            cc[cid] = cc.get(cid, 0) + 1
        vals = sorted(cc.values(), reverse=True)
        top1 = vals[0] if vals else 0
        top3 = sum(vals[:3]) if vals else 0
        cluster_concentration_top1 = float(top1 / len(selected))
        cluster_concentration_top3 = float(top3 / len(selected))

        return {
            "spatial_dispersion": spatial_disp,
            "spatial_compactness": spatial_comp,
            "redundancy_mean_sim": red_mean,
            "redundancy_max_sim": red_max,
            "coverage_cells": coverage_cells,
            "coverage_ratio": coverage_ratio,
            "cluster_concentration_top1": cluster_concentration_top1,
            "cluster_concentration_top3": cluster_concentration_top3,
            "utility_before_mean": float(utility.mean().item()),
            "utility_after_mean": float(utility[idx].mean().item()),
        }


class DataLoader:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self._proj_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def _probe_image_resolution(self, frame_path: str) -> Tuple[int, int]:
        try:
            with Image.open(frame_path) as im:
                return int(im.size[0]), int(im.size[1])
        except Exception:
            return (0, 0)

    def _video_to_frame_dir(self, video_rel: str) -> str:
        if video_rel.lower().endswith(".mp4"):
            return video_rel[:-4]
        stem, _ = os.path.splitext(video_rel)
        return stem

    def load_videos(self) -> List[Dict[str, Any]]:
        if self.cfg.synthetic_videos > 0:
            return self._synthetic_videos()
        if self.cfg.dataset_json:
            return self._videos_from_dataset_json(self.cfg.dataset_json, self.cfg.frame_root)
        if not self.cfg.input_manifest:
            raise ValueError("input_manifest is required unless synthetic_videos > 0 or dataset_json is set")
        path = Path(self.cfg.input_manifest)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        if path.suffix.lower() == ".jsonl":
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "videos" in obj:
            return obj["videos"]
        if isinstance(obj, list):
            return obj
        raise ValueError("Unsupported manifest structure")

    def _videos_from_dataset_json(self, dataset_json: str, frame_root: str) -> List[Dict[str, Any]]:
        path = Path(dataset_json)
        if not path.exists():
            raise FileNotFoundError(f"dataset_json not found: {dataset_json}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("dataset_json must be a JSON list")

        videos: List[Dict[str, Any]] = []
        max_frames = int(self.cfg.dataset_max_frames)
        ext_set = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        for idx, item in enumerate(data):
            video_rel = str(item.get("video", item.get("video_id", "")))
            if not video_rel:
                continue
            frame_dir = Path(frame_root) / self._video_to_frame_dir(video_rel)
            if not frame_dir.exists() or not frame_dir.is_dir():
                continue
            names = sorted([n for n in os.listdir(frame_dir) if Path(n).suffix.lower() in ext_set])
            if not names:
                continue
            if max_frames > 0 and not _as_bool(getattr(self.cfg, "force_full_video_frames", True)):
                names = names[:max_frames]

            rw, rh = (0, 0)
            if _as_bool(getattr(self.cfg, "resolution_probe_first_frame", True)):
                rw, rh = self._probe_image_resolution(str(frame_dir / names[0]))
            rbucket = _resolution_bucket(rw, rh, self.cfg)

            frames = []
            for fid, name in enumerate(names):
                frames.append(
                    {
                        "frame_id": fid,
                        "frame_path": str(frame_dir / name),
                        "token_path": None,
                    }
                )
            video_id = str(item.get("id", video_rel)).replace("/", "_")
            videos.append(
                {
                    "video_id": video_id,
                    "video_rel": video_rel,
                    "source": "dataset_json",
                    "frames": frames,
                    "dataset_index": idx,
                    "resolution_w": int(rw),
                    "resolution_h": int(rh),
                    "resolution_bucket": rbucket,
                    "num_frames_total": int(len(names)),
                }
            )
        return videos

    def load_video_frames(self, video_entry: Dict[str, Any]) -> List[FrameTokens]:
        vid = str(video_entry.get("video_id", video_entry.get("video", "video_unknown")))
        if vid.startswith("synthetic_"):
            out: List[FrameTokens] = []
            for f in range(self.cfg.synthetic_frames_per_video):
                n = self.cfg.synthetic_tokens_per_frame
                d = self.cfg.synthetic_dim
                g = torch.Generator()
                g.manual_seed(self.cfg.seed + f + int(vid.split("_")[-1]))
                tokens = torch.randn(n, d, generator=g)
                x = torch.rand(n, generator=g)
                y = torch.rand(n, generator=g)
                # move synthetic tensors to configured device
                dev = getattr(self.cfg, "device", torch.device("cpu"))
                out.append(
                    FrameTokens(
                        video_id=vid,
                        frame_id=f,
                        tokens=tokens.to(dev),
                        x_norm=x.to(dev),
                        y_norm=y.to(dev),
                        frame_path=None,
                        token_ids=torch.arange(n, dtype=torch.long).to(dev),
                    )
                )
            return out

        frames_meta = video_entry.get("frames", [])
        out: List[FrameTokens] = []
        for i, fm in enumerate(frames_meta):
            frame_id = _safe_int(fm.get("frame_id", i), i)
            frame_path = fm.get("frame_path")
            token_path = fm.get("token_path")
            if token_path is None:
                if not frame_path:
                    continue
                tokens, x_norm, y_norm, token_ids = self._extract_tokens_from_frame(frame_path)
            else:
                tokens, x_norm, y_norm, token_ids = self._load_tokens(token_path)
            # move loaded tensors to configured device (GPU if available)
            dev = getattr(self.cfg, "device", torch.device("cpu"))
            try:
                tokens = tokens.to(dev)
                x_norm = x_norm.to(dev)
                y_norm = y_norm.to(dev)
                token_ids = token_ids.to(dev)
            except Exception:
                # fallback: keep on CPU if move fails
                pass
            out.append(
                FrameTokens(
                    video_id=vid,
                    frame_id=frame_id,
                    tokens=tokens,
                    x_norm=x_norm,
                    y_norm=y_norm,
                    frame_path=frame_path,
                    token_ids=token_ids,
                )
            )
        return out

    def _extract_tokens_from_frame(self, frame_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"frame file not found: {frame_path}")
        img = Image.open(frame_path).convert("RGB")
        gh = max(4, int(self.cfg.raw_grid_h))
        gw = max(4, int(self.cfg.raw_grid_w))
        resized = img.resize((gw, gh), resample=Image.BICUBIC)
        arr = np.asarray(resized).astype(np.float32) / 255.0
        n = gh * gw

        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, gh, dtype=np.float32),
            np.linspace(0.0, 1.0, gw, dtype=np.float32),
            indexing="ij",
        )
        rgb = arr.reshape(n, 3)
        px = xx.reshape(n, 1)
        py = yy.reshape(n, 1)
        feat = np.concatenate([rgb, px, py], axis=1)

        out_dim = max(8, int(self.cfg.raw_token_dim))
        key = (feat.shape[1], out_dim, int(self.cfg.seed))
        if key not in self._proj_cache:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(self.cfg.seed))
            proj = torch.randn(feat.shape[1], out_dim, generator=g, dtype=torch.float32)
            proj = proj / math.sqrt(float(feat.shape[1]))
            self._proj_cache[key] = proj
        proj = self._proj_cache[key]

        feat_t = torch.from_numpy(feat)
        tokens = torch.matmul(feat_t, proj)
        tokens = _normalize(tokens, dim=-1)

        x_norm = torch.from_numpy(xx.reshape(-1).astype(np.float32))
        y_norm = torch.from_numpy(yy.reshape(-1).astype(np.float32))
        token_ids = torch.arange(n, dtype=torch.long)
        return tokens.float(), x_norm, y_norm, token_ids

    def _load_tokens(self, token_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = Path(token_path)
        if not p.exists():
            raise FileNotFoundError(f"token file not found: {token_path}")

        if p.suffix.lower() in {".pt", ".pth"}:
            obj = torch.load(str(p), map_location="cpu")
            if isinstance(obj, torch.Tensor):
                tokens = obj.float()
                x_norm = self._default_coord(tokens.shape[0], "x")
                y_norm = self._default_coord(tokens.shape[0], "y")
                token_ids = torch.arange(tokens.shape[0], dtype=torch.long)
                return tokens, x_norm, y_norm, token_ids
            if isinstance(obj, dict):
                tokens = self._as_tensor(obj.get("tokens", obj.get("hi_tokens")))
                x_norm = self._as_tensor(obj.get("x_norm", None), ndim=1, n=tokens.shape[0], mode="x")
                y_norm = self._as_tensor(obj.get("y_norm", None), ndim=1, n=tokens.shape[0], mode="y")
                token_ids = self._as_tensor(obj.get("token_ids", None), ndim=1, n=tokens.shape[0], mode="id").long()
                return tokens, x_norm, y_norm, token_ids
            raise ValueError(f"Unsupported .pt content: {token_path}")

        if p.suffix.lower() == ".npz":
            data = np.load(str(p))
            keys = set(data.keys())
            tok_key = "tokens" if "tokens" in keys else ("hi_tokens" if "hi_tokens" in keys else None)
            if tok_key is None:
                raise ValueError(f"No tokens key in npz: {token_path}")
            tokens = torch.from_numpy(data[tok_key]).float()
            x_norm = torch.from_numpy(data["x_norm"]).float() if "x_norm" in keys else self._default_coord(tokens.shape[0], "x")
            y_norm = torch.from_numpy(data["y_norm"]).float() if "y_norm" in keys else self._default_coord(tokens.shape[0], "y")
            token_ids = torch.from_numpy(data["token_ids"]).long() if "token_ids" in keys else torch.arange(tokens.shape[0], dtype=torch.long)
            return tokens, _clamp01(x_norm), _clamp01(y_norm), token_ids

        raise ValueError(f"Unsupported token file suffix: {token_path}")

    def _as_tensor(self, value: Any, ndim: Optional[int] = None, n: Optional[int] = None, mode: str = "x") -> torch.Tensor:
        if value is None:
            if mode == "id":
                return torch.arange(max(0, int(n or 0)), dtype=torch.long)
            return self._default_coord(max(0, int(n or 0)), mode)
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().float()
        else:
            t = torch.tensor(value, dtype=torch.float32)
        if ndim is not None and t.ndim != ndim:
            t = t.reshape(-1)
        if mode in {"x", "y"}:
            t = _clamp01(t)
        return t

    def _default_coord(self, n: int, mode: str) -> torch.Tensor:
        if n <= 0:
            return torch.empty(0, dtype=torch.float32)
        side = max(1, int(round(math.sqrt(n))))
        idx = torch.arange(n, dtype=torch.float32)
        if mode == "x":
            return _clamp01((idx % side) / max(1, side - 1))
        if mode == "y":
            return _clamp01((idx // side) / max(1, side - 1))
        return torch.arange(n, dtype=torch.float32)

    def _synthetic_videos(self) -> List[Dict[str, Any]]:
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        vids = []
        for v in range(self.cfg.synthetic_videos):
            frames = []
            for f in range(self.cfg.synthetic_frames_per_video):
                token_path = f"__synthetic__/video{v}_frame{f}"
                frames.append({"frame_id": f, "frame_path": None, "token_path": token_path})
            vids.append({"video_id": f"synthetic_{v:03d}", "frames": frames})
        return vids

    def maybe_patch_synthetic_tokens(self, frame: FrameTokens) -> FrameTokens:
        if not frame.frame_path and frame.tokens.numel() == 0 and frame.video_id.startswith("synthetic_"):
            pass
        if isinstance(frame.frame_path, str):
            return frame
        # token_path not carried in dataclass, infer from frame_id for synthetic mode only.
        if not frame.video_id.startswith("synthetic_"):
            return frame
        n = self.cfg.synthetic_tokens_per_frame
        d = self.cfg.synthetic_dim
        g = torch.Generator()
        g.manual_seed(self.cfg.seed + frame.frame_id + int(frame.video_id.split("_")[-1]))
        tokens = torch.randn(n, d, generator=g)
        x = torch.rand(n, generator=g)
        y = torch.rand(n, generator=g)
        return FrameTokens(
            video_id=frame.video_id,
            frame_id=frame.frame_id,
            tokens=tokens,
            x_norm=x,
            y_norm=y,
            frame_path=None,
            token_ids=torch.arange(n, dtype=torch.long),
        )


def _build_stage_configs(cfg: argparse.Namespace) -> Dict[str, Dict[str, str]]:
    return {
        "utility_only": {
            "clustering_method": "none",
            "diversity_method": "topk",
            "coverage_method": "none",
        },
        "clustering_only": {
            "clustering_method": cfg.clustering_method,
            "diversity_method": "topk",
            "coverage_method": "none",
        },
        "cluster_diversity": {
            "clustering_method": cfg.clustering_method,
            "diversity_method": cfg.diversity_method,
            "coverage_method": "none",
        },
        "full_pipeline": {
            "clustering_method": cfg.clustering_method,
            "diversity_method": cfg.diversity_method,
            "coverage_method": cfg.coverage_method,
        },
    }


def run_single_experiment(cfg: argparse.Namespace, exp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    exp_name = exp_cfg["name"]
    out_root = Path(cfg.output_dir) / exp_name
    vis_dir = out_root / "visualizations"
    key_vis_dir = out_root / "vis" / "key_frames"
    logs_dir = out_root / "logs"
    metrics_dir = out_root / "metrics"
    _ensure_dir(out_root)
    _ensure_dir(vis_dir)
    _ensure_dir(key_vis_dir)
    _ensure_dir(logs_dir)
    _ensure_dir(metrics_dir)

    with open(out_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(exp_cfg), f, indent=2)

    status = {
        "experiment": exp_name,
        "start_time": _now_str(),
        "end_time": "",
        "planned_videos": 0,
        "processed_videos": 0,
        "last_video_id": "",
        "stop_reason": "running",
        "exception": "",
    }
    global_start = time.time()

    data_loader = DataLoader(cfg)
    utility_comp = UtilityComputer(cfg)
    clusterer = Clusterer(cfg)
    selector = Selector(cfg)
    visualizer = Visualizer(cfg)

    videos = data_loader.load_videos()
    status["planned_videos"] = len(videos)
    ds_summary = _summarize_dataset_selection(videos, cfg, out_root)
    _write_dataset_top3_check(ds_summary, out_root)
    if _as_bool(getattr(cfg, "prefer_stress_videos", True)):
        videos = _pick_preferred_videos(videos, ds_summary, cfg)

    if cfg.max_videos > 0:
        if cfg.random_sample_videos and not _as_bool(getattr(cfg, "prefer_stress_videos", True)):
            if len(videos) > cfg.max_videos:
                videos = _sample_videos_with_resolution(videos, cfg)
        elif not _as_bool(getattr(cfg, "prefer_stress_videos", True)):
            videos = videos[: cfg.max_videos]

    all_frame_metrics: List[Dict[str, Any]] = []
    per_video_summary: List[Dict[str, Any]] = []
    keyframe_rows: List[Dict[str, Any]] = []

    frame_log_jsonl = logs_dir / "frame_logs.jsonl"
    frame_summary_jsonl = out_root / "frame_summary.jsonl"
    with open(frame_log_jsonl, "w", encoding="utf-8") as f_jsonl, open(frame_summary_jsonl, "w", encoding="utf-8") as f_frame:
        for video_idx, video_entry in enumerate(videos):
            if cfg.max_runtime_minutes > 0 and (time.time() - global_start) / 60.0 >= cfg.max_runtime_minutes:
                status["stop_reason"] = "达到max_runtime_minutes"
                break
            video_id = str(video_entry.get("video_id", video_entry.get("video", "video_unknown")))
            status["last_video_id"] = video_id
            frames = data_loader.load_video_frames(video_entry)
            if cfg.max_frames_per_video > 0 and not _as_bool(getattr(cfg, "force_full_video_frames", True)):
                frames = frames[: cfg.max_frames_per_video]

            video_vis_dir = vis_dir / video_id
            _write_video_info_file(video_vis_dir, video_entry, len(frames))
            per_video_frame_scores: List[Dict[str, Any]] = []

            vid_selected_total = 0
            vid_tokens_total = 0
            vid_cluster_sizes: List[int] = []
            vid_start = time.time()

            for frame in frames:
                if cfg.max_video_runtime_minutes > 0 and (time.time() - vid_start) / 60.0 >= cfg.max_video_runtime_minutes:
                    status["stop_reason"] = "达到max_video_runtime_minutes"
                    break
                frame = data_loader.maybe_patch_synthetic_tokens(frame)
                tokens = frame.tokens.float()
                x = _clamp01(frame.x_norm.float())
                y = _clamp01(frame.y_norm.float())
                n = tokens.shape[0]
                if n == 0:
                    continue

                utility, utility_details = utility_comp.compute(tokens, x, y, exp_cfg["utility_method"])
                clusters = clusterer.cluster(tokens, x, y, utility, exp_cfg["clustering_method"])

                stage_cfgs = _build_stage_configs(copy.deepcopy(cfg))
                stage_records: Dict[str, List[int]] = {}
                final_trace: Dict[str, Any] = {}

                for stage, methods in stage_cfgs.items():
                    sel_ids, trace = selector.select(
                        tokens=tokens,
                        x=x,
                        y=y,
                        utility=utility,
                        clusters=clusters if methods["clustering_method"] != "none" else clusterer.cluster(tokens, x, y, utility, "none"),
                        budget=min(exp_cfg["topk"], n),
                        diversity_method=methods["diversity_method"],
                        coverage_method=methods["coverage_method"],
                    )
                    stage_records[stage] = sel_ids
                    if stage == "full_pipeline":
                        final_trace = trace

                selected = stage_records.get("full_pipeline", [])
                frame_metrics = Metrics.compute(tokens, x, y, selected, clusters, utility, cfg)
                frame_metrics.update(
                    {
                        "experiment": exp_name,
                        "video_id": video_id,
                        "frame_id": int(frame.frame_id),
                        "num_tokens": int(n),
                        "num_selected": int(len(selected)),
                        "num_clusters": int(len(clusters.cluster_sizes)),
                    }
                )
                frame_metrics["selected_ratio"] = float(len(selected) / max(1, n))
                frame_metrics["mean_selected_utility"] = float(utility[selected].mean().item()) if selected else 0.0
                frame_metrics["max_selected_utility"] = float(utility[selected].max().item()) if selected else 0.0
                frame_metrics["selected_cluster_count"] = int(len(set(int(clusters.cluster_ids[i].item()) for i in selected))) if selected else 0
                frame_metrics["frame_score"] = float(
                    0.45 * frame_metrics["selected_ratio"]
                    + 0.35 * frame_metrics["mean_selected_utility"]
                    + 0.2 * frame_metrics["coverage_ratio"]
                )
                all_frame_metrics.append(frame_metrics)

                vid_selected_total += len(selected)
                vid_tokens_total += n
                vid_cluster_sizes.extend(list(clusters.cluster_sizes.values()))

                if cfg.save_vis:
                    frame_dir = video_vis_dir
                    vis_paths = visualizer.draw_all(
                        frame=frame,
                        utility=utility,
                        clusters=clusters,
                        selected=selected,
                        stage_records=stage_records,
                        frame_out_dir=frame_dir,
                        exp_name=exp_name,
                        video_id=video_id,
                    )

                    sel_detail_path = Path(vis_paths.get("spatial_dir", "")) / f"{exp_name}_{video_id}_f{frame.frame_id:05d}_selected_tokens.json"
                    _write_json(
                        sel_detail_path,
                        {
                            "video_id": video_id,
                            "frame_id": int(frame.frame_id),
                            "selected_indices": [int(v) for v in selected],
                            "selected_count": int(len(selected)),
                            "num_tokens": int(n),
                        },
                    )
                else:
                    vis_paths = {}

                frame_log = {
                    "experiment": exp_name,
                    "video_id": video_id,
                    "frame_id": int(frame.frame_id),
                    "frame_path": frame.frame_path,
                    "utility_method": exp_cfg["utility_method"],
                    "clustering_method": exp_cfg["clustering_method"],
                    "diversity_method": exp_cfg["diversity_method"],
                    "coverage_method": exp_cfg["coverage_method"],
                    "num_tokens": int(n),
                    "token_x_norm": [float(v) for v in x.tolist()],
                    "token_y_norm": [float(v) for v in y.tolist()],
                    "token_utility": [float(v) for v in utility.tolist()],
                    "utility_stats": {
                        "min": float(utility.min().item()),
                        "max": float(utility.max().item()),
                        "mean": float(utility.mean().item()),
                        "std": float(utility.std(unbiased=False).item()),
                    },
                    "clusters": {
                        "count": int(len(clusters.cluster_sizes)),
                        "sizes": {str(k): int(v) for k, v in clusters.cluster_sizes.items()},
                        "centers": {
                            str(cid): [
                                float(x[clusters.cluster_token_indices[cid]].mean().item()),
                                float(y[clusters.cluster_token_indices[cid]].mean().item()),
                            ]
                            for cid in clusters.cluster_token_indices
                        },
                        "avg_utility": {str(k): float(v) for k, v in clusters.cluster_avg_utility.items()},
                        "max_utility": {str(k): float(v) for k, v in clusters.cluster_max_utility.items()},
                    },
                    "selected": {
                        "count": int(len(selected)),
                        "indices": [int(v) for v in selected],
                        "per_cluster_counts": final_trace.get("cluster_counts", {}),
                        "trace": final_trace.get("trace", []),
                    },
                    "stage_selected": {k: [int(v) for v in vals] for k, vals in stage_records.items()},
                    "metrics": frame_metrics,
                    "visualizations": vis_paths,
                }
                f_jsonl.write(json.dumps(_to_jsonable(frame_log), ensure_ascii=False) + "\n")
                f_frame.write(json.dumps(_to_jsonable(frame_log), ensure_ascii=False) + "\n")

                keyframe_rows.append(
                    {
                        "video_id": video_id,
                        "frame_id": int(frame.frame_id),
                        "frame_score": frame_metrics["frame_score"],
                        "selection_vis": vis_paths.get("selection", ""),
                    }
                )
                per_video_frame_scores.append(
                    {
                        "frame_id": int(frame.frame_id),
                        "frame_score": float(frame_metrics["frame_score"]),
                        "temporal_view": vis_paths.get("temporal_view", ""),
                    }
                )

            if status["stop_reason"] == "达到max_video_runtime_minutes":
                break

            per_video_summary.append(
                {
                    "experiment": exp_name,
                    "video_id": video_id,
                    "frames": len(frames),
                    "total_tokens": int(vid_tokens_total),
                    "total_selected_tokens": int(vid_selected_total),
                    "mean_selected_per_frame": float(vid_selected_total / max(1, len(frames))),
                    "cluster_size_mean": float(np.mean(vid_cluster_sizes)) if vid_cluster_sizes else 0.0,
                    "cluster_size_std": float(np.std(vid_cluster_sizes)) if vid_cluster_sizes else 0.0,
                }
            )

            if per_video_frame_scores:
                sorted_rows = sorted(per_video_frame_scores, key=lambda r: r["frame_score"], reverse=True)
                total = len(sorted_rows)
                for i, r in enumerate(sorted_rows):
                    q = int(i * 4 / max(1, total)) + 1
                    if q > 4:
                        q = 4
                    r["temporal_bucket"] = f"Q{q}"
                    r["temporal_rank"] = i + 1

                temporal_dir = video_vis_dir / "temporal_tokens"
                _ensure_dir(temporal_dir)
                _write_json(temporal_dir / "temporal_partition.json", sorted_rows)
                md_lines = ["# 时序token划分", ""]
                for r in sorted_rows:
                    md_lines.append(
                        f"- frame={r['frame_id']} | rank={r['temporal_rank']} | bucket={r['temporal_bucket']} | score={r['frame_score']:.6f}"
                    )
                (temporal_dir / "temporal_partition.md").write_text("\n".join(md_lines), encoding="utf-8")

            status["processed_videos"] = int(len(per_video_summary))

            if cfg.max_videos > 0 and len(per_video_summary) >= cfg.max_videos:
                status["stop_reason"] = "达到max_videos"
                break

    if status["stop_reason"] == "running":
        status["stop_reason"] = "正常完成"
    summary = {
        "experiment": exp_name,
        "run_time": _now_str(),
        "num_videos": len(per_video_summary),
        "num_frames": len(all_frame_metrics),
        "avg_selected_tokens": float(np.mean([m["num_selected"] for m in all_frame_metrics])) if all_frame_metrics else 0.0,
        "avg_num_clusters": float(np.mean([m["num_clusters"] for m in all_frame_metrics])) if all_frame_metrics else 0.0,
        "avg_spatial_dispersion": float(np.mean([m["spatial_dispersion"] for m in all_frame_metrics])) if all_frame_metrics else 0.0,
        "avg_redundancy": float(np.mean([m["redundancy_mean_sim"] for m in all_frame_metrics])) if all_frame_metrics else 0.0,
        "avg_coverage_ratio": float(np.mean([m["coverage_ratio"] for m in all_frame_metrics])) if all_frame_metrics else 0.0,
        "per_video": per_video_summary,
        "stop_reason": status["stop_reason"],
    }

    with open(metrics_dir / "frame_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(all_frame_metrics), f, indent=2)
    with open(metrics_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(summary), f, indent=2)

    frame_importance = _compute_frame_importance(all_frame_metrics)
    _save_frame_importance(out_root, frame_importance)

    topk = max(1, int(cfg.keyframe_topk))
    top_rows = sorted(keyframe_rows, key=lambda r: r["frame_score"], reverse=True)[:topk]
    copied = []
    for r in top_rows:
        src = str(r.get("selection_vis", ""))
        if src and os.path.exists(src):
            dst = key_vis_dir / f"{r['video_id']}_f{int(r['frame_id']):05d}.png"
            shutil.copy2(src, dst)
            copied.append({**r, "copied_path": str(dst)})
    _write_json(out_root / "key_frames.json", copied)

    _write_markdown_report(out_root / "README_EXPERIMENT.md", exp_cfg, summary, per_video_summary)
    _write_method_cn(out_root / "方法说明.md", exp_cfg, cfg)
    status["end_time"] = _now_str()
    _write_status_files(out_root, status)
    return summary


def _write_markdown_report(path: Path, exp_cfg: Dict[str, Any], summary: Dict[str, Any], per_video: List[Dict[str, Any]]) -> None:
    lines = []
    lines.append(f"# Experiment {exp_cfg['name']}")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(_to_jsonable(exp_cfg), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(_to_jsonable(summary), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Per Video")
    lines.append("")
    for v in per_video:
        lines.append(f"- video={v['video_id']} frames={v['frames']} selected={v['total_selected_tokens']} tokens={v['total_tokens']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_ablation_experiments(cfg: argparse.Namespace) -> List[Dict[str, Any]]:
    base = {
        "name": "mem_base_norm",
        "utility_method": "norm",
        "clustering_method": "none",
        "diversity_method": "topk_utility",
        "coverage_method": "none",
        "topk": cfg.topk,
        "seed": cfg.seed,
    }
    exps: List[Dict[str, Any]] = [base]

    # Required module ablations
    exps.append({**base, "name": "mem_only_utility_norm"})
    exps.append({**base, "name": "mem_only_clustering_grid", "clustering_method": "grid_clustering"})
    exps.append({**base, "name": "mem_only_diversity_mmr", "diversity_method": "mmr_token"})
    exps.append({**base, "name": "mem_only_coverage_quota", "coverage_method": "per_grid_quota"})
    exps.append({**base, "name": "mem_cluster_diversity", "clustering_method": "grid_clustering", "diversity_method": "mmr_token"})
    exps.append({**base, "name": "mem_cluster_coverage", "clustering_method": "grid_clustering", "coverage_method": "per_grid_quota"})
    exps.append({**base, "name": "mem_diversity_coverage", "diversity_method": "mmr_token", "coverage_method": "per_grid_quota"})
    exps.append(
        {
            **base,
            "name": "mem_cluster_diversity_coverage",
            "clustering_method": "grid_clustering",
            "diversity_method": "mmr_token",
            "coverage_method": "per_grid_quota",
        }
    )

    # Old-school region methods comparison
    region_methods = [
        ("grid_clustering", "mem_region_grid"),
        ("slic_superpixel", "mem_region_slic"),
        ("felzenszwalb_segmentation", "mem_region_felz"),
        ("edge_connected_components", "mem_region_edge"),
        ("optical_flow_region", "mem_region_flow"),
        ("simple_pixel_segmentation", "mem_region_simplepix"),
    ]
    for cm, nm in region_methods:
        exps.append({**base, "name": nm, "clustering_method": cm})
        exps.append({**base, "name": f"{nm}_mmr", "clustering_method": cm, "diversity_method": "mmr_token"})
        exps.append(
            {
                **base,
                "name": f"{nm}_mmr_quota",
                "clustering_method": cm,
                "diversity_method": "mmr_token",
                "coverage_method": "per_cluster_quota",
            }
        )

    # Utility method ablations
    util_methods = [
        "norm",
        "rarity_within_frame",
        "local_contrast",
        "global_saliency_proxy",
        "hybrid_norm_rarity",
        "hybrid_rarity_localcontrast",
    ]
    for um in util_methods:
        exps.append(
            {
                **base,
                "name": f"mem_util_{um}_slic_mmr",
                "utility_method": um,
                "clustering_method": "slic_superpixel",
                "diversity_method": "mmr_token",
                "coverage_method": "none",
            }
        )

    # Make unique names while preserving order.
    seen = set()
    uniq = []
    for e in exps:
        n = e["name"]
        if n in seen:
            continue
        seen.add(n)
        uniq.append(e)

    if cfg.ablation_limit > 0:
        uniq = uniq[: cfg.ablation_limit]
    return uniq


def generate_slurm_scripts(cfg: argparse.Namespace, experiments: List[Dict[str, Any]], submit: bool) -> Dict[str, Any]:
    slurm_dir = Path(cfg.slurm_out_dir)
    _ensure_dir(slurm_dir)
    log_dir = Path(cfg.slurm_log_dir)
    _ensure_dir(log_dir)

    scripts = []
    submitted = []

    for exp in experiments:
        job_name = exp["name"][:80]
        exp_name = exp["name"]
        if str(cfg.job_name_prefix):
            job_name = f"{str(cfg.job_name_prefix)}{exp_name}"[:80]
        script_path = slurm_dir / f"{job_name}.slurm"
        exp_json = Path(cfg.output_dir) / "ablation_plan.json"

        cmd = [
            "python3",
            "-u",
            cfg.main_script_path,
            "--mode",
            "run_experiment_list",
            "--experiment_list",
            str(exp_json),
            "--run_experiment_name",
            exp_name,
            "--output_dir",
            str(cfg.output_dir),
            "--save_vis",
            str(cfg.save_vis),
            "--max_videos",
            str(cfg.max_videos),
            "--random_sample_videos",
            str(cfg.random_sample_videos),
            "--prefer_stress_videos",
            str(cfg.prefer_stress_videos),
            "--top3_priority_only",
            str(cfg.top3_priority_only),
            "--dataset_selection_topn",
            str(cfg.dataset_selection_topn),
            "--selection_topk_per_axis",
            str(cfg.selection_topk_per_axis),
            "--resolution_stratified_sample",
            str(cfg.resolution_stratified_sample),
            "--resolution_probe_first_frame",
            str(cfg.resolution_probe_first_frame),
            "--resolution_bucket_mode",
            str(cfg.resolution_bucket_mode),
            "--resolution_bin1",
            str(cfg.resolution_bin1),
            "--resolution_bin2",
            str(cfg.resolution_bin2),
            "--resolution_bin3",
            str(cfg.resolution_bin3),
            "--max_frames_per_video",
            str(cfg.max_frames_per_video),
            "--dataset_max_frames",
            str(cfg.dataset_max_frames),
            "--force_full_video_frames",
            str(cfg.force_full_video_frames),
            "--raw_grid_h",
            str(cfg.raw_grid_h),
            "--raw_grid_w",
            str(cfg.raw_grid_w),
            "--raw_token_dim",
            str(cfg.raw_token_dim),
            "--topk",
            str(cfg.topk),
            "--max_runtime_minutes",
            str(cfg.max_runtime_minutes),
            "--max_video_runtime_minutes",
            str(cfg.max_video_runtime_minutes),
            "--keyframe_topk",
            str(cfg.keyframe_topk),
        ]
        if str(cfg.input_manifest or "").strip():
            cmd.extend(["--input_manifest", str(cfg.input_manifest)])
        if str(cfg.dataset_json or "").strip():
            cmd.extend(["--dataset_json", str(cfg.dataset_json)])
        if str(cfg.frame_root or "").strip():
            cmd.extend(["--frame_root", str(cfg.frame_root)])
        if cfg.synthetic_videos > 0:
            cmd.extend(["--synthetic_videos", str(cfg.synthetic_videos), "--synthetic_frames_per_video", str(cfg.synthetic_frames_per_video)])

        bash_cmd = " \\\n  ".join(cmd)

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={str(log_dir / (job_name + '-%j.out'))}",
            f"#SBATCH --error={str(log_dir / (job_name + '-%j.err'))}",
            "#SBATCH --qos=gpu",
            "#SBATCH --partition=gpuq",
            "#SBATCH --nodes=1",
            "#SBATCH --gres=gpu:3g.40gb:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=40G",
            "#SBATCH --time=06:00:00",
            "",
            "set -euo pipefail",
            "set -x",
            "PS4='[${BASH_SOURCE##*/}:${LINENO}] '",
            "trap 'rc=$?; echo [ERROR] line=${LINENO} exit_code=${rc}; exit ${rc}' ERR",
            "",
            f"cd {cfg.repo_root}",
            f"export SIF={cfg.sif_path}",
            "module load singularity || true",
            "module load cuda/12.2 || true",
            "",
            "if [ -n \"${SIF:-}\" ] && [ -f \"${SIF}\" ]; then",
            "  singularity exec --nv -e --bind /scratch:/scratch --pwd \"$PWD\" \"${SIF}\" bash -lc \"",
            "  set -euo pipefail",
            "  export PYTHONUNBUFFERED=1",
            f"  {bash_cmd}",
            "  \"",
            "else",
            "  export PYTHONUNBUFFERED=1",
            f"  {bash_cmd}",
            "fi",
            "",
        ]
        script_path.write_text("\n".join(lines), encoding="utf-8")
        scripts.append(str(script_path))

        if submit:
            try:
                proc = subprocess.run(["sbatch", str(script_path)], check=False, capture_output=True, text=True)
                submitted.append(
                    {
                        "job_name": job_name,
                        "script": str(script_path),
                        "returncode": int(proc.returncode),
                        "stdout": proc.stdout.strip(),
                        "stderr": proc.stderr.strip(),
                    }
                )
            except Exception as exc:
                submitted.append(
                    {
                        "job_name": job_name,
                        "script": str(script_path),
                        "returncode": -1,
                        "stdout": "",
                        "stderr": f"submit_exception: {exc}",
                    }
                )

    return {
        "slurm_dir": str(slurm_dir),
        "scripts": scripts,
        "submitted": submitted,
    }


def _setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="[%(asctime)s] %(levelname)s %(name)s - %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory retrieval ablation framework")

    p.add_argument(
        "--mode",
        default="run",
        choices=["run", "generate_ablation", "run_experiment_list", "generate_slurm", "full_pipeline", "dataset_top3_check"],
    )
    p.add_argument("--input_manifest", default="", help="JSON/JSONL manifest path")
    p.add_argument(
        "--dataset_json",
        default="/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/data/train_test_1fps_8_vis.json",
        help="Existing 1fps dataset JSON (list of entries with video field)",
    )
    p.add_argument(
        "--frame_root",
        default="/scratch/zwu24/datasets/LLaVA-Video-178K_fps1",
        help="Root directory for decoded 1fps frames",
    )
    p.add_argument("--output_dir", default="./output/memory_ablation")
    p.add_argument("--experiment_name", default="mem_exp")
    p.add_argument("--experiment_list", default="", help="ablation plan json")
    p.add_argument("--run_experiment_name", default="", help="run only one experiment from experiment_list")

    p.add_argument(
        "--utility_method",
        default="norm",
        choices=[
            "norm",
            "rarity_within_frame",
            "local_contrast",
            "global_saliency_proxy",
            "hybrid",
            "hybrid_norm_rarity",
            "hybrid_rarity_localcontrast",
        ],
    )
    p.add_argument(
        "--clustering_method",
        default="none",
        choices=[
            "none",
            "baseline_none",
            "grid",
            "grid_clustering",
            "kmeans_xy",
            "kmeans_xyf",
            "density_radius",
            "slic_superpixel",
            "felzenszwalb_segmentation",
            "edge_connected_components",
            "optical_flow_region",
            "simple_pixel_segmentation",
        ],
    )
    p.add_argument(
        "--diversity_method",
        default="topk",
        choices=[
            "topk",
            "topk_utility",
            "mmr",
            "mmr_token",
            "cluster_mmr",
            "mmr_cluster",
            "facility_coverage",
            "greedy_representative_selection",
            "nms",
            "redundancy_suppression",
        ],
    )
    p.add_argument(
        "--coverage_method",
        default="none",
        choices=["none", "per_grid_quota", "per_cluster_quota", "coverage_reward", "region_balancing", "crowding_penalty"],
    )

    p.add_argument("--topk", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)

    # Utility params
    p.add_argument("--local_radius", type=float, default=0.12)
    p.add_argument("--utility_pairwise_max_tokens", type=int, default=1024)
    p.add_argument("--hybrid_norm_w", type=float, default=0.34)
    p.add_argument("--hybrid_rarity_w", type=float, default=0.33)
    p.add_argument("--hybrid_local_w", type=float, default=0.33)

    # Clustering params
    p.add_argument("--grid_cluster_x", type=int, default=8)
    p.add_argument("--grid_cluster_y", type=int, default=8)
    p.add_argument("--kmeans_k", type=int, default=16)
    p.add_argument("--kmeans_iters", type=int, default=15)
    p.add_argument("--kmeans_feature_dims", type=int, default=16)
    p.add_argument("--kmeans_pos_weight", type=float, default=1.0)
    p.add_argument("--kmeans_feat_weight", type=float, default=0.5)
    p.add_argument("--density_radius", type=float, default=0.08)
    p.add_argument("--density_min_pts", type=int, default=4)

    # Diversity params
    p.add_argument("--mmr_lambda", type=float, default=0.4)
    p.add_argument("--cluster_mmr_lambda", type=float, default=0.35)
    p.add_argument("--cluster_repr", default="mean", choices=["mean", "max", "weighted_mean"])
    p.add_argument("--cluster_rank_bonus_scale", type=float, default=0.5)
    p.add_argument("--facility_alpha", type=float, default=0.6)
    p.add_argument("--facility_support_max", type=int, default=256)
    p.add_argument("--nms_sim_threshold", type=float, default=0.92)

    # Coverage params
    p.add_argument("--coverage_grid_x", type=int, default=8)
    p.add_argument("--coverage_grid_y", type=int, default=8)
    p.add_argument("--grid_quota_hard_max", type=int, default=32)
    p.add_argument("--grid_quota_soft_max", type=int, default=20)
    p.add_argument("--grid_quota_hard_min", type=int, default=0)
    p.add_argument("--cluster_quota_strategy", default="uniform", choices=["uniform", "size", "utility", "sqrt_size"])
    p.add_argument("--cluster_quota_soft_scale", type=float, default=1.2)
    p.add_argument("--coverage_bonus_scale", type=float, default=0.4)
    p.add_argument("--coverage_decay_radius", type=float, default=0.15)

    # Data/run controls
    p.add_argument("--max_videos", type=int, default=5)
    p.add_argument("--random_sample_videos", type=lambda s: str(s).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--prefer_stress_videos", type=lambda s: str(s).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--top3_priority_only", type=lambda s: str(s).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--dataset_selection_topn", type=int, default=20)
    p.add_argument("--selection_topk_per_axis", type=int, default=3)
    p.add_argument("--dataset_assumed_fps", type=float, default=1.0)
    p.add_argument("--resolution_stratified_sample", type=lambda s: str(s).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--resolution_probe_first_frame", type=lambda s: str(s).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--resolution_bucket_mode", default="short_side", choices=["short_side", "exact"])
    p.add_argument("--resolution_bin1", type=int, default=360)
    p.add_argument("--resolution_bin2", type=int, default=540)
    p.add_argument("--resolution_bin3", type=int, default=720)
    p.add_argument("--max_frames_per_video", type=int, default=0)
    p.add_argument("--dataset_max_frames", type=int, default=0, help="Frames loaded per video for dataset_json mode; <=0 means full video")
    p.add_argument("--force_full_video_frames", type=lambda s: str(s).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--raw_grid_h", type=int, default=28, help="Grid height for frame->token extraction")
    p.add_argument("--raw_grid_w", type=int, default=28, help="Grid width for frame->token extraction")
    p.add_argument("--raw_token_dim", type=int, default=256, help="Projected token dim for frame->token extraction")
    p.add_argument("--save_vis", type=lambda s: str(s).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--save_logs", type=lambda s: str(s).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--save_cluster_logs", type=lambda s: str(s).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--log_level", default="INFO")
    p.add_argument("--max_runtime_minutes", type=int, default=60)
    p.add_argument("--max_video_runtime_minutes", type=int, default=20)
    p.add_argument("--keyframe_topk", type=int, default=8)

    # Synthetic mode for dry run / dev
    p.add_argument("--synthetic_videos", type=int, default=0)
    p.add_argument("--synthetic_frames_per_video", type=int, default=8)
    p.add_argument("--synthetic_tokens_per_frame", type=int, default=1024)
    p.add_argument("--synthetic_dim", type=int, default=256)
    p.add_argument("--canvas_w", type=int, default=1280)
    p.add_argument("--canvas_h", type=int, default=720)

    # Ablation generation
    p.add_argument("--ablation_limit", type=int, default=0)

    # Slurm
    p.add_argument("--generate_slurm", type=lambda s: str(s).lower() in {"1", "true", "yes"}, default=False)
    p.add_argument("--submit_slurm", type=lambda s: str(s).lower() in {"1", "true", "yes"}, default=False)
    p.add_argument("--slurm_out_dir", default="./scripts/generated_memory_ablation_slurm")
    p.add_argument("--slurm_log_dir", default="./log")
    p.add_argument("--repo_root", default=".")
    p.add_argument("--main_script_path", default="scripts/memory_retrieval_ablation.py")
    p.add_argument("--job_name_prefix", default="")
    p.add_argument("--sif_path", default="/scratch/zwu24/singularity_sifs/qwen.sif")

    return p.parse_args()


def _load_experiment_list(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"experiment list not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "experiments" in obj:
        return obj["experiments"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Unsupported experiment list format")


def main() -> None:
    cfg = parse_args()
    _setup_logging(cfg.log_level)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # Determine runtime device early so tensors/models can be moved to GPU when available.
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] using device={cfg.device} cuda_available={torch.cuda.is_available()} cuda_count={torch.cuda.device_count()}")

    out_dir = Path(cfg.output_dir)
    _ensure_dir(out_dir)

    if cfg.max_videos < 0:
        cfg.max_videos = 5

    if cfg.mode == "generate_ablation":
        exps = generate_ablation_experiments(cfg)
        out = out_dir / "ablation_plan.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"experiments": exps}, f, indent=2)
        print(f"[ablation] generated={len(exps)} path={out}")
        return

    if cfg.mode == "dataset_top3_check":
        data_loader = DataLoader(cfg)
        videos = data_loader.load_videos()
        ds_summary = _summarize_dataset_selection(videos, cfg, out_dir)
        _write_dataset_top3_check(ds_summary, out_dir)
        print(f"[dataset_top3_check] videos={len(videos)} path={str(out_dir / 'dataset_top3_check.md')}")
        return

    if cfg.mode == "run":
        exp = {
            "name": cfg.experiment_name,
            "utility_method": cfg.utility_method,
            "clustering_method": cfg.clustering_method,
            "diversity_method": cfg.diversity_method,
            "coverage_method": cfg.coverage_method,
            "topk": cfg.topk,
            "seed": cfg.seed,
        }
        summary = run_single_experiment(cfg, exp)
        with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(summary), f, indent=2)
        print(f"[run] done experiment={cfg.experiment_name}")
        return

    if cfg.mode == "run_experiment_list":
        exps = _load_experiment_list(cfg.experiment_list)
        if cfg.run_experiment_name:
            exps = [e for e in exps if e.get("name") == cfg.run_experiment_name]
            if not exps:
                raise ValueError(f"No experiment named {cfg.run_experiment_name}")
        all_summary = []
        for exp in exps:
            merged = {
                "name": exp.get("name", cfg.experiment_name),
                "utility_method": exp.get("utility_method", cfg.utility_method),
                "clustering_method": exp.get("clustering_method", cfg.clustering_method),
                "diversity_method": exp.get("diversity_method", cfg.diversity_method),
                "coverage_method": exp.get("coverage_method", cfg.coverage_method),
                "topk": int(exp.get("topk", cfg.topk)),
                "seed": int(exp.get("seed", cfg.seed)),
            }
            LOGGER.info("running experiment=%s", merged["name"])
            s = run_single_experiment(cfg, merged)
            all_summary.append(s)

        with open(out_dir / "batch_summary.json", "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(all_summary), f, indent=2)
        print(f"[batch] done experiments={len(all_summary)}")
        return

    if cfg.mode == "generate_slurm":
        exps = _load_experiment_list(cfg.experiment_list)
        info = generate_slurm_scripts(cfg, exps, submit=cfg.submit_slurm)
        out = out_dir / "slurm_generation_report.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(info), f, indent=2)
        print(f"[slurm] scripts={len(info['scripts'])} report={out}")
        return

    if cfg.mode == "full_pipeline":
        exps = generate_ablation_experiments(cfg)
        ab_path = out_dir / "ablation_plan.json"
        with open(ab_path, "w", encoding="utf-8") as f:
            json.dump({"experiments": exps}, f, indent=2)
        print(f"[full] generated ablation plan path={ab_path} n={len(exps)}")

        all_summary = []
        for exp in exps:
            LOGGER.info("running experiment=%s", exp["name"])
            s = run_single_experiment(cfg, exp)
            all_summary.append(s)

        with open(out_dir / "batch_summary.json", "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(all_summary), f, indent=2)

        if cfg.generate_slurm:
            info = generate_slurm_scripts(cfg, exps, submit=cfg.submit_slurm)
            with open(out_dir / "slurm_generation_report.json", "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(info), f, indent=2)
            print(f"[full] slurm scripts generated={len(info['scripts'])} submit={cfg.submit_slurm}")

        print(f"[full] completed experiments={len(all_summary)}")
        return

    raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()
