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


class RaggedFlashMemoryRetriever:
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
    ):
        self.logger = logging.getLogger(__name__ + ".RaggedFlashMemoryRetriever")
        self.dim = dim
        self.budget_target = int(budget_target)
        self.budget_hard = int(budget_hard)
        self.r_t = float(r_t)
        self.xbin = int(xbin)
        self.ybin = int(ybin)
        self.topM_items = int(topM_items)
        self.num_anchors = int(num_anchors)
        self.alpha = float(alpha)

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
        if default_mode == "x":
            width = max(1, int(math.sqrt(n)))
            return ((idx % width) / float(max(1, width - 1))).clamp(0.0, 1.0)
        height = max(1, int(math.sqrt(n)))
        return ((idx // height) / float(max(1, height - 1))).clamp(0.0, 1.0)

    def _build_position_ids(self, selected_meta: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
        n = len(selected_meta)
        if n == 0:
            return torch.zeros(3, 1, 0, dtype=torch.long, device=device)
        t = torch.tensor([m["t"] for m in selected_meta], dtype=torch.long, device=device)
        x_bin = torch.tensor([m["x_bin"] for m in selected_meta], dtype=torch.long, device=device)
        y_bin = torch.tensor([m["y_bin"] for m in selected_meta], dtype=torch.long, device=device)
        return torch.stack([t, x_bin, y_bin], dim=0).unsqueeze(1)

    def _temporal_candidates(self, query_embed: Optional[torch.Tensor], device: torch.device):
        candidates = []
        if len(self.centroids) == 0:
            return candidates
        centroids = torch.stack(self.centroids, dim=0).to(device)
        weights = torch.tensor(self.cluster_weight, device=device, dtype=centroids.dtype)
        num_anchors = min(self.num_anchors, centroids.shape[0])
        topk_idx = torch.topk(weights, k=num_anchors).indices.tolist()
        q = None
        if query_embed is not None:
            q = query_embed.to(device=device, dtype=centroids.dtype)
            q = q / (q.norm(p=2) + 1e-6)
        for idx in topk_idx:
            centroid = centroids[idx]
            utility = float(math.log1p(max(self.cluster_weight[idx], 1e-6)))
            if q is not None:
                utility += float(F.cosine_similarity(centroid.unsqueeze(0), q.unsqueeze(0)).item())
            candidates.append(
                {
                    "type": "temporal",
                    "token": centroid,
                    "utility": utility,
                    "cost": 1,
                    "meta": {
                        "chunk_id": -1,
                        "t": idx,
                        "x_bin": self.xbin // 2,
                        "y_bin": self.ybin // 2,
                        "cluster_weight": self.cluster_weight[idx],
                    },
                }
            )
        return candidates

    def _candidate_chunk_ids(self, anchor_vectors: torch.Tensor, device: torch.device):
        if len(self.items) == 0:
            return []
        item_reprs = torch.stack([it.repr.to(device) for it in self.items], dim=0)
        item_reprs = F.normalize(item_reprs, dim=-1)
        cand_ids = set()
        for anchor in anchor_vectors:
            a = F.normalize(anchor.unsqueeze(0), dim=-1)
            sim = torch.matmul(item_reprs, a.t()).squeeze(1)
            k = min(self.topM_items, sim.shape[0])
            top_ids = torch.topk(sim, k=k).indices.tolist()
            cand_ids.update(top_ids)
        return sorted(cand_ids)

    def _spatial_candidates(self, candidate_chunk_ids: List[int], query_embed: Optional[torch.Tensor], device: torch.device):
        candidates = []
        if len(candidate_chunk_ids) == 0:
            return candidates

        q = None
        if query_embed is not None:
            q = query_embed.to(device=device)
            q = q / (q.norm(p=2) + 1e-6)

        for chunk_idx in candidate_chunk_ids:
            item = self.items[chunk_idx]
            hi_tokens = item.hi_tokens.to(device)
            if hi_tokens.numel() == 0:
                continue
            n = hi_tokens.shape[0]
            x_norm = self._as_tensor(item.meta.x_norm, n, device, default_mode="x")
            y_norm = self._as_tensor(item.meta.y_norm, n, device, default_mode="y")

            if q is None:
                score = torch.norm(hi_tokens, dim=-1)
            else:
                score = torch.matmul(F.normalize(hi_tokens, dim=-1), q)

            topk_per_chunk = min(256, n)
            top_idx = torch.topk(score, k=topk_per_chunk).indices
            sel_tokens = hi_tokens[top_idx]
            sel_scores = score[top_idx]

            for token, utility, idx in zip(sel_tokens, sel_scores.tolist(), top_idx.tolist()):
                x_bin = int(torch.floor(x_norm[idx] * (self.xbin - 1)).item())
                y_bin = int(torch.floor(y_norm[idx] * (self.ybin - 1)).item())
                t = int(item.meta.t0 + (idx % max(1, item.meta.t_len)))
                candidates.append(
                    {
                        "type": "spatial",
                        "token": token,
                        "utility": float(utility),
                        "cost": 1,
                        "meta": {
                            "chunk_id": item.meta.chunk_id,
                            "t": t,
                            "x_bin": x_bin,
                            "y_bin": y_bin,
                            "src_res_h": item.meta.src_res_h,
                            "src_res_w": item.meta.src_res_w,
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
            self.centroids.append(repr_vec.detach().clone())
            self.cluster_weight.append(1.0)
            cluster_idx = len(self.centroids) - 1
        else:
            centroids = torch.stack(self.centroids, dim=0).to(device)
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
        b_tem = int(math.floor(b_target * self.r_t))
        b_tem = max(0, min(b_tem, b_target))
        b_spa = b_target - b_tem

        anchor_start = time.perf_counter()
        temporal_candidates = self._temporal_candidates(query_embed=query_embed, device=device)
        anchor_vectors = torch.stack([c["token"] for c in temporal_candidates], dim=0) if temporal_candidates else torch.zeros(0, self.dim, device=device)
        candidate_chunk_ids = self._candidate_chunk_ids(anchor_vectors=anchor_vectors, device=device) if anchor_vectors.numel() > 0 else []
        spatial_candidates = self._spatial_candidates(candidate_chunk_ids, query_embed=query_embed, device=device)
        anchor_ms = (time.perf_counter() - anchor_start) * 1000.0

        pack_start = time.perf_counter()
        temporal_candidates.sort(key=lambda x: x["utility"], reverse=True)
        spatial_candidates.sort(key=lambda x: x["utility"], reverse=True)

        selected_tokens: List[torch.Tensor] = []
        selected_meta: List[Dict[str, Any]] = []

        for cand in temporal_candidates[:b_tem]:
            selected_tokens.append(cand["token"])
            selected_meta.append(cand["meta"])

        for cand in spatial_candidates[:b_spa]:
            selected_tokens.append(cand["token"])
            selected_meta.append(cand["meta"])

        if len(selected_tokens) > self.budget_hard:
            selected_tokens = selected_tokens[: self.budget_hard]
            selected_meta = selected_meta[: self.budget_hard]

        if selected_tokens:
            selected_tensor = torch.stack(selected_tokens, dim=0)
        else:
            selected_tensor = torch.zeros(0, self.dim, device=device)

        position_ids = self._build_position_ids(selected_meta, device=device)
        pack_ms = (time.perf_counter() - pack_start) * 1000.0

        stats = {
            "anchors": len(temporal_candidates),
            "candidate_chunks": len(candidate_chunk_ids),
            "N_tem": min(len(temporal_candidates), b_tem),
            "N_spa": min(len(spatial_candidates), b_spa),
            "N_sel": int(selected_tensor.shape[0]),
            "budget_target": int(self.budget_target),
            "budget_hard": int(self.budget_hard),
            "B_t": int(b_tem),
            "B_s": int(b_spa),
            "rt": float(self.r_t),
            "anchor_select_ms": anchor_ms,
            "packing_ms": pack_ms,
            "retrieve_ms": (time.perf_counter() - t_start) * 1000.0,
        }

        return selected_tensor, selected_meta, position_ids, stats
