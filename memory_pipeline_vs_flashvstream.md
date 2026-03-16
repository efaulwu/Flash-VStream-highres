# 现有记忆 Pipeline vs 原始 Flash-VStream：差异、技术与接口形状

## 1. 范围与代码入口

本文档对比两条实现：

1) 原始 Flash-VStream 风格（Grid Memory，规则网格）
- 入口实现见 [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime.py)
- 主要模块：`FlashMemory`（`temporal_compress + spatial_enhance + calc_am_rope`）

2) 现有新 Pipeline（Ragged Memory，动态分辨率）
- 检索器实现见 [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/ragged_flash_memory_retriever.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/ragged_flash_memory_retriever.py)
- 接入点同样在 [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime.py)
- 通过 `flash_memory_mode='ragged'` 开关切换

---

## 2. 核心差异（概念层）

## 2.1 原始 Flash-VStream（Grid Memory）

特点：
- 假设 memory 里的 token 可以按统一 `thw` 重排。
- 通过规则 reshape 完成：
  - temporal 压缩（CSM）
  - spatial 检索（DAM）
- 位置编码通过网格公式直接推导（`thw -> (t,h,w)`）。

限制：
- 历史 chunk 与新 chunk 在合并时默认共享空间网格结构。
- 对“每个 chunk 分辨率不同、token 数不同”的场景不友好。

## 2.2 现有 Ragged Pipeline（动态分辨率）

特点：
- 不再依赖统一 `thw` 规则重排。
- 以 `ChunkItem` 列表维护历史：每个 chunk 可有不同 token 数。
- 检索拆分成两级：
  - Temporal：基于 centroid/cluster 的 item-level 选择
  - Spatial：在候选 chunk 内 token-level 选择
- 用预算机制控制输出长度：`B_target` 与 `B_hard`。

收益：
- 原生支持动态分辨率与变长 token。
- 能显式做 token budget 调度，便于实时性与质量折中。

---

## 3. 新 Pipeline 使用的关键技术

当前 Ragged 版本主要用了以下技术：

1) **Ragged 存储结构**
- `items: List[ChunkItem]`，每个 chunk 单独存 `hi_tokens/lo_tokens/repr/meta`。

2) **在线聚类更新（CSM 近似）**
- `centroids + cluster_weight`。
- 更新策略：相似度匹配后 EMA 更新 centroid（`alpha`）。

3) **两级检索（DAM 近似）**
- 先在 `item repr` 上做 topM chunk 检索（避免全 token 暴力搜索）。
- 再在候选 chunk 内做 token 打分（norm 或 query cosine）并取 topK。

4) **预算打包（u/c 贪心）**
- `B_t = floor(B_target * r_t)`，`B_s = B_target - B_t`。
- temporal / spatial 分别排序后选取。
- 最终强裁剪到 `B_hard`。

5) **Ragged 位置编码构造**
- 每个 token 生成 `(t, x_bin, y_bin)`。
- 形成 `position_ids: [3,1,N_sel]`。
- 在模型侧再做文本 offset 对齐。

---

## 4. 接口对比：输入 / 输出 / 形状

## 4.1 RaggedFlashMemoryRetriever.update

函数：`update(hi_tokens, lo_tokens, token_meta)`

输入：
- `hi_tokens: Tensor[n_hi, D]`
- `lo_tokens: Tensor[n_lo, D]`
- `token_meta: dict`
  - 至少含 `chunk_id, t0, t_len, src_res_h, src_res_w`
  - 可选 `x_norm, y_norm`

输出：
- `stats: dict`
  - `chunk_id, cluster_idx, num_items, hi_tokens_n, lo_tokens_n, cluster_weight_top5, update_ms`

语义：
- 把当前 chunk 写入 ragged memory，并更新 temporal centroids。

## 4.2 RaggedFlashMemoryRetriever.retrieve

函数：`retrieve(query_embed=None, current_time=None)`

输入：
- `query_embed: Optional[Tensor[D]]`（当前默认可为 `None`）

输出：
- `selected_tokens: Tensor[N_sel, D]`
- `selected_meta: List[dict]`（长度 `N_sel`）
- `position_ids: Tensor[3, 1, N_sel]`
- `stats: dict`
  - 包括 `N_tem, N_spa, N_sel, B_t, B_s, budget_target, budget_hard, retrieve_ms...`

硬约束：
- `N_sel <= budget_hard`

## 4.3 模型流式写入：embed_new_video_clip（ragged 分支）

函数：`embed_new_video_clip(pixel_values_videos, video_grid_thw, start_idx)`

输入：
- `pixel_values_videos: Tensor[N_patch, C*Tp*P*P]`（当前实现里通常第二维是 `3*2*14*14=1176`）
- `video_grid_thw: Tensor[1,3]`，即 `[T,H,W]`
- `start_idx: int`

中间：
- `forward_simple_not_merge` 输出 backbone 特征 `x: Tensor[N_backbone, D_v]`
- 经 `self.visual.merger` 映射到 LLM 维度后形成：
  - `hi_tokens: Tensor[n_hi, D_llm]`
  - `lo_tokens: Tensor[n_lo, D_llm]`

输出：
- 返回 profiling 时间戳列表（长度 4）
- 内部写入 retriever 状态，不直接返回 memory token

## 4.4 模型流式读取：prepare_realtime_inference（ragged 分支）

函数：`prepare_realtime_inference(position_ids, visual_position_ids)`

输入：
- `position_ids: Tensor[3, B, L]`（当前实现按 `B=1` 使用）
- `visual_position_ids: Tensor[B, L]`（视频位非负）

输出：
- `video_embeds: Tensor[N_align, D_llm]`
  - 其中 `N_align` 是对齐后的视频 token 数（按 prompt 中 `<|video_pad|>` 区域对齐，可能 pad/truncate）
- `new_position_ids: Tensor[3, 1, L]`

说明：
- `retrieve` 得到的 `N_sel` 会与目标视频槽位数做对齐，日志中可区分 raw 与 aligned（建议保留两者）。

---

## 5. 原始 Grid Memory 接口与形状（对照）

## 5.1 FlashMemory.forward

函数：`FlashMemory.forward(x, grid_thw, small_grid_thw, position_ids, visual_position_ids)`

输入：
- `x: Tensor[N_total, D_v]`
- `grid_thw: Tensor[B,3]`
- `small_grid_thw: Optional[Tensor[B,3]]`
- `position_ids: Tensor[3,B,L]`
- `visual_position_ids: Tensor[B,L]`

输出：
- `new_x: Tensor[B, L_premerge, D_v]`
- `new_position_ids: Tensor[3,B,L]`

其中：
- `L_premerge = L_spa_premerge + L_tem_premerge`

## 5.2 Grid 路径下模型 memory 缓存结构（13槽）

`video_embedding_memory` 的典型内容：
1. `tem_x: Tensor[tem_thw.prod(), D_v]`
2. `tem_thw: Tensor[3]`
3. `tem_weights: Tensor[tem_thw[0]]`
4. `tem_timestamp: Tensor[tem_thw[0]]`
5. `spa_x: Tensor[spa_thw.prod(), D_v]`
6. `spa_thw: Tensor[3]`
7. `spa_positions: Tensor[spa_thw[0]]`
8. `x: Tensor[thw.prod(), D_v]`
9. `thw: Tensor[3]`
10. `small_x: Tensor[small_thw.prod(), D_v]`
11. `small_thw: Tensor[3]`
12. `video_embeds: Tensor[L_video, D_llm]`（经 merger）
13. `video_embeds.shape`

---

## 6. 现在这套实现的工程状态总结

1) 你已经从“规则网格 memory”迈到了“ragged memory + budget 检索”。
2) 运行日志显示 pipeline 已可稳定跑完，并满足 `N_sel <= 12000`。
3) 当前 `model-wrapped` 路径还存在“对齐到 prompt 槽位”的行为，因此建议日志里同时统计：
- `raw_selected_n`（retrieve 原始输出）
- `aligned_selected_n`（pad/truncate 后）

这样论文分析会更准确：
- 前者用于 memory 调度质量；
- 后者用于接口对齐与推理兼容性。

---

## 7. 一句话结论

与原始 Flash-VStream 相比，你的现有 pipeline 已从“固定网格压缩/检索”升级为“动态分辨率 ragged 检索 + 预算控制 + token级位置构造”，接口上最核心的变化是：memory 输入输出从 `thw` 驱动的规则张量，变成了 `selected_tokens + token_meta + [3,1,N] position_ids` 的变长检索结果。