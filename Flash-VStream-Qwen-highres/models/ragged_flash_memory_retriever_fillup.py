import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ChunkTokenMeta:
    chunk_id: int
    t0: int
    t_len: int
    src_res_h: int
    src_res_w: int
    x_norm: Optional[torch.Tensor] = None
    y_norm: Optional[torch.Tensor] = None


@dataclass
class ChunkItem:
    hi_tokens: torch.Tensor
    lo_tokens: torch.Tensor
    repr: torch.Tensor
    meta: ChunkTokenMeta


class RaggedFlashMemoryRetrieverFillup:
    """Ragged retriever that tries to fill visual token budget up to budget_target/budget_hard.

    Behavior goals:
    - Always present up to 12k tokens (or less only when total available tokens are fewer).
    - Remove per-frame or per-chunk hard caps in spatial token selection.
    - Keep data interface compatible with the existing ragged retriever.
    """

    def __init__(
        self,
        dim: int,
        budget_target: int = 11520,
        budget_hard: int = 12000,
        r_t: float = 1.0 / 3.0,
        xbin: int = 64,
        ybin: int = 64,
        topM_items: int = 10,
        num_anchors: int = 8,
        alpha: float = 0.05,
        seed: int = 0,
        recency_alpha: float = 0.15,
    ):
        self.logger = logging.getLogger(__name__ + ".RaggedFlashMemoryRetrieverFillup")
        self.dim = dim
        self.budget_target = int(budget_target)
        self.budget_hard = int(budget_hard)
        self.r_t = float(r_t)
        self.xbin = int(xbin)
        self.ybin = int(ybin)
        self.topM_items = int(topM_items)
        self.num_anchors = int(num_anchors)
        self.alpha = float(alpha)
        self.recency_alpha = float(recency_alpha)

        self.items: List[ChunkItem] = []
        self.centroids: List[torch.Tensor] = []
        self.cluster_weight: List[float] = []
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def _as_tensor(self, x: Optional[torch.Tensor], n: int, device: torch.device, default_mode: str):
        if x is not None:
            return x.to(device=device).float().clamp(0.0, 1.0)
        if n <= 0:
            return torch.empty(0, device=device)
        idx = torch.arange(n, device=device, dtype=torch.float32)
        side = max(1, int(math.sqrt(n)))
        if default_mode == "x":
            return ((idx % side) / float(max(1, side - 1))).clamp(0.0, 1.0)
        return ((idx // side) / float(max(1, side - 1))).clamp(0.0, 1.0)

    def _default_time_ids(self, n: int, t0: int, t_len: int, device: torch.device) -> torch.Tensor:
        if n <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        t_len = max(1, int(t_len))
        return int(t0) + (torch.arange(n, device=device, dtype=torch.long) % t_len)

    def _build_position_ids(self, selected_meta: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
        n = len(selected_meta)
        if n == 0:
            return torch.zeros(3, 1, 0, dtype=torch.long, device=device)
        t = torch.tensor([int(m.get("t", 0)) for m in selected_meta], dtype=torch.long, device=device)
        x_bin = torch.tensor([int(m.get("x_bin", self.xbin // 2)) for m in selected_meta], dtype=torch.long, device=device)
        y_bin = torch.tensor([int(m.get("y_bin", self.ybin // 2)) for m in selected_meta], dtype=torch.long, device=device)
        return torch.stack([t, x_bin, y_bin], dim=0).unsqueeze(1)

    def _temporal_candidates(self, query_embed: Optional[torch.Tensor], device: torch.device):
        candidates = []
        if len(self.centroids) == 0:
            return candidates

        centroids = torch.stack([c.to(device=device) for c in self.centroids], dim=0)
        centroids = F.normalize(centroids, dim=-1)
        weights = torch.tensor(self.cluster_weight, device=device, dtype=centroids.dtype)

        q = None
        if query_embed is not None:
            q = query_embed.to(device=device, dtype=centroids.dtype)
            q = q / (q.norm(p=2) + 1e-6)

        utility = torch.log1p(torch.clamp_min(weights, 1e-6))
        if q is not None:
            utility = utility + torch.matmul(centroids, q)

        for idx in range(centroids.shape[0]):
            candidates.append(
                {
                    "type": "temporal",
                    "token": centroids[idx],
                    "utility": float(utility[idx].item()),
                    "cost": 1,
                    "meta": {
                        "chunk_id": -1,
                        "t": idx,
                        "x_bin": self.xbin // 2,
                        "y_bin": self.ybin // 2,
                        "cluster_weight": float(self.cluster_weight[idx]),
                        "utility": float(utility[idx].item()),
                    },
                }
            )
        return candidates

    def _spatial_candidates_all(self, query_embed: Optional[torch.Tensor], device: torch.device):
        candidates = []
        if len(self.items) == 0:
            return candidates

        q = None
        if query_embed is not None:
            q = query_embed.to(device=device)

        max_chunk_id = max(int(it.meta.chunk_id) for it in self.items)
        recency_denom = max(1, max_chunk_id)

        for item in self.items:
            hi_tokens = item.hi_tokens.to(device)
            if hi_tokens.numel() == 0:
                continue

            n = hi_tokens.shape[0]
            x_norm = self._as_tensor(item.meta.x_norm, n, device, default_mode="x")
            y_norm = self._as_tensor(item.meta.y_norm, n, device, default_mode="y")
            t_ids = self._default_time_ids(n=n, t0=item.meta.t0, t_len=item.meta.t_len, device=device)

            hi_norm = F.normalize(hi_tokens, dim=-1)
            if q is None:
                score = torch.norm(hi_tokens, dim=-1)
            else:
                q_local = q.to(dtype=hi_tokens.dtype)
                q_local = q_local / (q_local.norm(p=2) + 1e-6)
                score = torch.matmul(hi_norm, q_local)

            # Mild recency bias keeps latest details without hard-dropping earlier chunks.
            recency = float(item.meta.chunk_id) / float(recency_denom)
            score = score + self.recency_alpha * recency

            score_cpu = score.detach().cpu().tolist()
            x_norm_cpu = x_norm.detach().cpu().tolist()
            y_norm_cpu = y_norm.detach().cpu().tolist()
            t_ids_cpu = t_ids.detach().cpu().tolist()

            for idx in range(n):
                x_bin = int(math.floor(float(x_norm_cpu[idx]) * (self.xbin - 1)))
                y_bin = int(math.floor(float(y_norm_cpu[idx]) * (self.ybin - 1)))
                x_bin = max(0, min(self.xbin - 1, x_bin))
                y_bin = max(0, min(self.ybin - 1, y_bin))
                utility = float(score_cpu[idx])
                candidates.append(
                    {
                        "type": "spatial",
                        "token": hi_tokens[idx],
                        "utility": utility,
                        "cost": 1,
                        "meta": {
                            "chunk_id": int(item.meta.chunk_id),
                            "t": int(t_ids_cpu[idx]),
                            "x_bin": x_bin,
                            "y_bin": y_bin,
                            "src_res_h": int(item.meta.src_res_h),
                            "src_res_w": int(item.meta.src_res_w),
                            "utility": utility,
                        },
                    }
                )

        return candidates

    def update(self, hi_tokens: torch.Tensor, lo_tokens: torch.Tensor, token_meta: Dict[str, Any]):
        t_start = time.perf_counter()
        if hi_tokens.ndim != 2 or lo_tokens.ndim != 2:
            raise ValueError("hi_tokens and lo_tokens must be rank-2 tensors [N, D]")
        if hi_tokens.shape[1] != self.dim or lo_tokens.shape[1] != self.dim:
            raise ValueError(f"token dim mismatch: expected {self.dim}")

        device = hi_tokens.device
        repr_vec = lo_tokens.mean(dim=0)
        meta = ChunkTokenMeta(
            chunk_id=int(token_meta["chunk_id"]),
            t0=int(token_meta["t0"]),
            t_len=int(token_meta["t_len"]),
            src_res_h=int(token_meta.get("src_res_h", 0)),
            src_res_w=int(token_meta.get("src_res_w", 0)),
            x_norm=token_meta.get("x_norm", None),
            y_norm=token_meta.get("y_norm", None),
        )
        self.items.append(ChunkItem(hi_tokens=hi_tokens.detach(), lo_tokens=lo_tokens.detach(), repr=repr_vec.detach(), meta=meta))

        cluster_idx = -1
        if len(self.centroids) < self.num_anchors:
            self.centroids.append(repr_vec.detach().cpu().clone())
            self.cluster_weight.append(1.0)
            cluster_idx = len(self.centroids) - 1
        else:
            centroids = torch.stack([c.to(device=device) for c in self.centroids], dim=0)
            sim = torch.matmul(F.normalize(centroids, dim=-1), F.normalize(repr_vec.unsqueeze(0), dim=-1).t()).squeeze(1)
            cluster_idx = int(torch.argmax(sim).item())
            self.centroids[cluster_idx] = ((1.0 - self.alpha) * self.centroids[cluster_idx].to(device) + self.alpha * repr_vec).detach().cpu()
            self.cluster_weight[cluster_idx] += 1.0

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "chunk_id": meta.chunk_id,
            "cluster_idx": cluster_idx,
            "num_items": len(self.items),
            "hi_tokens_n": int(hi_tokens.shape[0]),
            "lo_tokens_n": int(lo_tokens.shape[0]),
            "cluster_weight_top5": sorted(self.cluster_weight, reverse=True)[:5],
            "update_ms": dt_ms,
        }

    def retrieve(self, query_embed: Optional[torch.Tensor] = None, current_time: Optional[int] = None):
        t_start = time.perf_counter()
        device = self.items[0].hi_tokens.device if self.items else torch.device("cpu")

        if len(self.items) == 0:
            empty = torch.zeros(0, self.dim, device=device)
            pos = torch.zeros(3, 1, 0, dtype=torch.long, device=device)
            return empty, [], pos, {
                "N_tem": 0,
                "N_spa": 0,
                "N_sel": 0,
                "budget_target": self.budget_target,
                "budget_hard": self.budget_hard,
                "rt": self.r_t,
                "retrieve_ms": 0.0,
            }

        b_target = min(self.budget_target, self.budget_hard)
        b_tem_target = int(math.floor(b_target * self.r_t))
        b_tem_target = max(0, min(b_tem_target, b_target))

        anchor_start = time.perf_counter()
        temporal_candidates = self._temporal_candidates(query_embed=query_embed, device=device)
        spatial_candidates = self._spatial_candidates_all(query_embed=query_embed, device=device)
        anchor_ms = (time.perf_counter() - anchor_start) * 1000.0

        temporal_candidates.sort(key=lambda x: x["utility"], reverse=True)
        spatial_candidates.sort(key=lambda x: x["utility"], reverse=True)

        temporal_take = min(len(temporal_candidates), b_tem_target)
        selected_temporal = temporal_candidates[:temporal_take]

        b_spa_target = b_target - temporal_take
        spatial_take = min(len(spatial_candidates), b_spa_target)
        selected_spatial = spatial_candidates[:spatial_take]

        # Backfill from remaining temporal candidates if spatial pool is insufficient.
        missing = b_target - (temporal_take + spatial_take)
        if missing > 0:
            backfill_temporal = temporal_candidates[temporal_take : temporal_take + missing]
            selected_temporal.extend(backfill_temporal)

        selected = selected_temporal + selected_spatial

        # If there are still not enough tokens, we only return available tokens.
        if len(selected) > self.budget_hard:
            selected = selected[: self.budget_hard]

        selected_tokens: List[torch.Tensor] = [cand["token"] for cand in selected]
        selected_meta: List[Dict[str, Any]] = [cand["meta"] for cand in selected]

        if selected_tokens:
            selected_tensor = torch.stack(selected_tokens, dim=0)
        else:
            selected_tensor = torch.zeros(0, self.dim, device=device)

        position_ids = self._build_position_ids(selected_meta, device=device)

        chunk_token_counts: Dict[int, int] = {}
        chunk_frame_counts: Dict[int, Dict[int, int]] = {}
        for meta in selected_meta:
            cid = int(meta.get("chunk_id", -1))
            if cid < 0:
                continue
            t = int(meta.get("t", 0))
            chunk_token_counts[cid] = chunk_token_counts.get(cid, 0) + 1
            if cid not in chunk_frame_counts:
                chunk_frame_counts[cid] = {}
            chunk_frame_counts[cid][t] = chunk_frame_counts[cid].get(t, 0) + 1

        stats = {
            "anchors": len(temporal_candidates),
            "candidate_chunks": len(self.items),
            "available_spatial": len(spatial_candidates),
            "available_temporal": len(temporal_candidates),
            "N_tem": int(sum(1 for s in selected if s["type"] == "temporal")),
            "N_spa": int(sum(1 for s in selected if s["type"] == "spatial")),
            "N_sel": int(selected_tensor.shape[0]),
            "budget_target": int(self.budget_target),
            "budget_hard": int(self.budget_hard),
            "B_t": int(b_tem_target),
            "B_s": int(b_target - b_tem_target),
            "rt": float(self.r_t),
            "fill_ratio_vs_target": float(selected_tensor.shape[0]) / float(max(1, b_target)),
            "chunk_token_counts": {str(k): int(v) for k, v in sorted(chunk_token_counts.items(), key=lambda x: x[0])},
            "chunk_frame_counts": {
                str(k): {str(t): int(c) for t, c in sorted(v.items(), key=lambda x: x[0])}
                for k, v in sorted(chunk_frame_counts.items(), key=lambda x: x[0])
            },
            "anchor_select_ms": anchor_ms,
            "retrieve_ms": (time.perf_counter() - t_start) * 1000.0,
        }

        return selected_tensor, selected_meta, position_ids, stats
