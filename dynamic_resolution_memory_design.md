# Flash-VStream Realtime 记忆模块动态分辨率改造分析

## 背景与目标

你希望在 streaming 场景下支持“每个视频 chunk 分辨率不一致”。
这会直接导致：

- 每个 chunk 的 `thw` 不同；
- 每帧 visual token 数（P）不同；
- token 数来自 Qwen 原生 visual encoder，不再是固定网格推导后的常量。

目标是：在不破坏 realtime memory 逻辑（temporal compress + spatial retrieve + AM-RoPE）的前提下，支持上述可变 token 流。

---

## 现状：当前实现为什么会卡在动态分辨率

### 1) 数据预处理其实已经支持“每个视频输入不同网格”

- 入口：Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_processor.py
- `FlashVStreamQwen2VLImageProcessor._preprocess` 会根据尺寸得到每个视频的 `(grid_t, grid_h, grid_w)`。
- `FlashVStreamQwen2VLProcessor.__call__` 也会用 `video_grid_thw` 动态计算 `<|video_pad|>` 展开长度。

结论：processor 这一层“单次调用维度变化”没有根本问题。

### 2) realtime memory 强依赖“历史 chunk 与当前 chunk 同 h,w”

关键位置：Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py

- `embed_new_video_clip` 内部 `merge_thw` 有强断言：`thw_1[1:] == thw_2[1:]`。
- 随后 `old_x/old_small_x` 与新 chunk 特征直接 `torch.cat`，默认它们可在时序维拼接（隐含每帧 patch 数一致）。

这一步是动态分辨率的第一处硬阻塞。

### 3) FlashMemory 的 reshape 逻辑默认固定 patch 结构

同文件 `FlashMemory` 里：

- `temporal_compress`：`x.reshape(t, h//2*w//2*2*2, d)`，要求整个序列共享同一个 `(h,w)`；
- `spatial_enhance`：对 `x/small_x/tem_x` 都按统一 `thw` 做 reshape；
- `cat_spa_tem`：按固定 `2*2` merge 结构拼接；
- `calc_am_rope`：根据 `tem_thw/spa_thw` 构造规则网格位置。

这意味着 memory 当前是“规则网格 memory”，不是“ragged token memory”。

### 4) 位置编码与 prompt token 数也基于规则网格长度

- `get_real_grid_thw/get_spatial_real_grid_thw/get_rope_index` 依据 `thw` 公式计算视觉 token 数和 3D 位置。
- 如果 memory 改成原生 ragged token，这套由网格推导长度的方法也要同步调整。

---

## 设计路线建议

## 路线 A（推荐）：进入 memory 前做“统一网格适配”（Canonical Memory Grid）

这是工程成本最低、与现有代码最兼容的方案。

核心思路：

1. 每个 chunk 先走 Qwen 原生 encoder，得到原生 token（可变长度）；
2. 在写入 memory 前，将每帧 token 适配到统一目标网格 `(Hc, Wc)`；
3. memory 内部继续使用当前规则网格逻辑（基本不改 temporal/spatial 算法）；
4. 同时保留原生 token 统计信息用于 debug 和质量分析。

### 需要修改的模块

1) realtime 写入入口

- 文件：Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py
- 函数：`embed_new_video_clip`
- 改造点：
  - 删除/改造 `merge_thw` 的同分辨率断言；
  - 新增 `adapt_chunk_to_canonical_grid(x, thw, target_hw)`；
  - 历史记忆中保存 `canonical_thw`，并将新 chunk 映射后再做拼接。

2) FlashMemory 输入契约

- 文件同上
- 函数：`temporal_compress`, `spatial_enhance`, `cat_spa_tem`, `calc_am_rope`
- 改造点：
  - 这几个函数可以基本保持算法不变；
  - 输入统一变成 canonical 网格（即在入口就归一化）；
  - `calc_am_rope` 继续按 canonical 的 `tem_thw/spa_thw` 计算。

3) 配置层

- 新增 memory 配置项（建议）：
  - `flash_memory_dynamic_resolution_enable: bool`
  - `flash_memory_canonical_h: int`
  - `flash_memory_canonical_w: int`
  - `flash_memory_adapt_method: {avg_pool, bilinear, attention_pool}`

4) debug 观测

- 在 memory 存储结构中增加：
  - `raw_chunk_thw_history`
  - `raw_token_count_history`
  - `canonical_token_count_history`

这样可以对比“原生 token 信息损失”并评估效果。

### 路线 A 的优缺点

优点：
- 改动集中在写入入口，风险可控；
- 复用现有 memory 算法与 AM-RoPE 逻辑；
- 训练/推理一致性更容易保持。

缺点：
- 会对原生可变分辨率 token 做压缩/重采样，细节有损失；
- canonical 目标网格选择不当会影响性能。

---

## 路线 B（完整原生）：memory 全链路改成 Ragged Token Memory

这条路线完全保留 Qwen 原生每 chunk token 数，但工程量明显更大。

核心变化：

1. memory 不再存单个 `thw`，而是存 chunk/frame 级 token 段边界；
2. temporal compress 不再以 `[T, P, D]` 的固定 P 表达，而以 frame-level 聚合表示进行聚类（例如每帧先 pooling 成 `[D]` 或 `[K,D]`）；
3. spatial retrieve 从变长 token 库检索，返回 ragged 索引集合；
4. AM-RoPE 不能只依赖规则网格，要引入显式 `visual_pos_table`（每个 token 的 t/h/w 或替代坐标）。

### 必改点清单

1) 数据结构重写

- `video_embedding_memory` 当前是固定 13 槽位张量列表（默认规则网格）。
- 需要改为结构化字典，至少包括：
  - `tokens`（拼接后的大张量）
  - `segment_ptr`（每 chunk/frame 的起止）
  - `token_meta`（每 token 的 time/frame/chunk/source）
  - `native_thw_history`

2) 压缩函数接口重写

- 文件：Flash-VStream-highres/Flash-VStream-Qwen-highres/models/compress_functions.py
- 当前大部分函数假设输入 `T,P,D`（固定 P）。
- 需要新增 ragged 版本，例如：
  - `weighted_kmeans_ordered_feature_ragged(frame_repr, ...)`
  - 或先做 per-frame global/learned pooling，再沿时间聚类。

3) FlashMemory 主逻辑重写

- 文件：Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py
- 函数：`temporal_compress`, `spatial_enhance`, `cat_spa_tem`, `calc_am_rope`
- 重点：从“规则网格重排”迁移到“显式索引 gather/scatter”。

4) token 长度与位置编码来源重写

- 文件：Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_processor.py
- 以及 `get_rope_index`（realtime 文件中）
- 需从“thw 公式推导长度”改成“由 memory 输出真实 token 数与 token-level 位置表”。

### 路线 B 的优缺点

优点：
- 最大化保留原生动态分辨率信息；
- 理论上上限更高。

缺点：
- 改造面广，回归风险高；
- 训练和部署都需要大量验证；
- 对性能（尤其 GPU gather/scatter）有挑战。

---

## 推荐实施顺序（务实）

第 1 阶段（1~2 天）
- 先落地路线 A，保证 realtime 可稳定处理不等分辨率 chunk。

第 2 阶段（2~5 天）
- 加入可观测性：记录 raw/canonical token 数、每步 memory 命中统计、时延。

第 3 阶段（按收益决定）
- 如果路线 A 的质量下降明显，再启动路线 B 的 ragged memory 原型。

---

## 最小改造补丁建议（路线 A）

1. 在 `embed_new_video_clip` 增加：
- 读取当前 chunk 原生 `thw` 与 token；
- 适配到 canonical 网格；
- 拼接历史 canonical memory。

2. memory 缓存新增字段：
- `canonical_thw`
- `raw_thw_history`
- `raw_token_count_history`

3. `prepare_realtime_inference` 维持接口不变：
- 继续返回 `video_embeds + new_position_ids`；
- 内部基于 canonical memory 计算 AM-RoPE。

4. `processor.__call__` 保持现有动态文本展开逻辑；
- 对于 streaming 模式，建议新增可选参数：
  - `video_visual_token_length_override`
- 由 memory 输出真实长度覆盖文本展开长度，避免公式与真实输出偏差。

---

## 相关代码入口（可直接开始改）

- Realtime memory 主实现：
  - [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py)
- Visual/token 预处理与 `<|video_pad|>` 展开：
  - [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_processor.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_processor.py)
- 压缩算法（当前固定 `T,P,D` 假设）：
  - [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/compress_functions.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/compress_functions.py)
- 默认 memory 配置：
  - [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/flash_memory_constants.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/flash_memory_constants.py)

---

## 一句话结论

如果你想尽快支持“每 chunk 分辨率不一致”，优先走“写入前统一 canonical 网格”的路线；如果你要完全保留 Qwen 原生动态 token，必须把 memory 从规则网格改造成 ragged token 架构。