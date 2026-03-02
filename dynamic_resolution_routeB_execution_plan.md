# Route B: Dynamic-Resolution Ragged Memory with 12K Token Budget

## 1. 目标与硬约束

你的论文目标是动态分辨率 scheduling 提升实时性与空间理解，因此必须采用原生动态 token 路线（Route B）：

- 每个 chunk 的分辨率可变；
- visual token 数由 Qwen native visual encoder 决定，不做统一网格强行对齐；
- memory 总视觉 token 预算必须满足 12000 上限。

硬约束：

- 设视觉 memory token 总数为 N_vis，要求 N_vis <= 12000。
- 若要保留系统余量（建议给文本与对齐误差留 buffer），可设目标预算 B_target=11520，硬上限 B_hard=12000。

---

## 2. 为什么原先 120/60 对应约 11520（而不是 23040）

当前实现中，FlashMemory 内部会把 temporal_length 与 spatial_length 各自除以 2：

- 有效 temporal 帧数 L_t = flash_memory_temporal_length / 2
- 有效 spatial 帧数 L_s = flash_memory_spatial_length / 2

在原先固定网格经验里：

- temporal 分支每帧约 64 token
- spatial 分支每帧约 256 token

所以预算是：

N_vis = L_t * 64 + L_s * 256

当配置是 120/60 时：

L_t=60, L_s=30

N_vis = 60*64 + 30*256 = 3840 + 7680 = 11520 <= 12000

这与当前工程实践一致，也说明 12000 是合理的硬预算目标。

相关代码入口：
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py)
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/flash_memory_constants.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/flash_memory_constants.py)

---

## 3. Route B 总体架构

核心思想：把 memory 从规则网格 tensor 变成 ragged token bank。

### 3.1 新 memory state（结构化）

建议将当前 13 槽位 list 改为字典或 dataclass：

- token_bank: FloatTensor [N_total, D]
- token_meta: 结构化字段（同长度 N_total）
  - global_time
  - chunk_id
  - frame_id_local
  - h_idx_norm, w_idx_norm（归一化空间坐标，范围 0~1）
  - level: temporal 或 spatial
  - source_res_h, source_res_w
- frame_ptr: 每帧 token 起止偏移
- frame_repr: 每帧聚合表示 [T_total, D]
- temporal_select_ids: 当前 temporal 记忆选中帧ID
- spatial_select_ids: 当前 spatial 记忆选中token或帧ID
- budget_stats: 每步预算分配与消耗

说明：
- 这里不再依赖单个 thw 进行 reshape。
- 通过 frame_ptr 与 token_meta 支持变长 token。

### 3.2 两层候选池

1) Temporal候选池（帧级）
- 从 frame_repr 里选择关键时间帧，保证事件连续性。

2) Spatial候选池（token级或帧内子token级）
- 在高价值帧内选择细节 token，保障空间理解。

---

## 4. 12000 预算下的动态时空调度机制

## 4.1 预算模型（统一）

定义：
- B_hard = 12000
- B_target = 11520（建议）
- B_t: temporal 配额
- B_s: spatial 配额

约束：
- B_t + B_s <= B_target
- 实际选中 token 数 N_t + N_s <= B_hard

在 ragged 方案里，N_t 和 N_s 不再是“帧数乘常数”，而是基于选中项真实 token 数累加。

## 4.2 三类动态机制（可组合）

### 机制A：内容驱动比例分配（主推）

为当前滑窗计算两个复杂度指标：

- motion_score: 时序变化强度（帧间特征差）
- detail_score: 空间细节强度（边缘密度/patch方差/注意力熵）

分配规则：

r_t = motion_score / (motion_score + detail_score + eps)

r_s = 1 - r_t

B_t = clamp(round(r_t * B_target), B_t_min, B_t_max)

B_s = B_target - B_t

优点：可解释性强，适合论文呈现。

### 机制B：效用密度贪心（近似背包）

为候选项 k 定义：
- utility u_k
- cost c_k（token 数）

按 u_k / c_k 排序，贪心装入预算。

temporal 候选用帧级，spatial 候选用token级或小块级。

优点：直接约束 token；实现简单；适应动态 token 数。

### 机制C：实时性闭环控制（Latency-aware）

目标：端到端延迟低于阈值 L_target。

使用 PID 或 EMA 控制器调整 B_target：

if latency_ema > L_target: B_target <- B_target - delta
if latency_ema < L_target and quality_gap high: B_target <- B_target + delta

并始终裁剪到 [B_min, 12000]。

优点：直接服务实时性，和论文指标一致。

建议组合：A + B + C
- A 决定时空大比例；
- B 在比例下做精细选取；
- C 跨时间调整整体预算。

---

## 5. AM-RoPE 与 position ids 改造

当前实现在多处通过 thw 推导位置与长度。Route B 需改为 token 级显式位置表：

- 每个保留 token 都有 (t, h_norm, w_norm) 或离散化后的 (t_bin, h_bin, w_bin)
- 按最终拼接顺序生成 position_ids，不再依赖规则网格公式

建议：
- temporal token 的 h/w 可设为 coarse 网格中心
- spatial token 保持真实归一化坐标并量化到固定 bins（如 64x64）

输出接口保持不变：
- prepare_realtime_inference 返回 video_embeds 与 new_position_ids

相关代码入口：
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py)
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_processor.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_processor.py)

---

## 6. 可由 Agent 执行的代码计划（可直接分阶段实施）

## Phase 0: 可观测性与开关（低风险）

目标：不改行为，先加配置与日志。

改动：
1. 在配置中新增
   - flash_memory_mode: grid 或 ragged
   - flash_memory_budget_target: 默认 11520
   - flash_memory_budget_hard: 默认 12000
   - flash_memory_scheduler: heuristic 或 knapsack 或 hybrid
2. 在 memory 更新路径记录
   - raw_tokens_per_chunk
   - selected_temporal_tokens
   - selected_spatial_tokens
   - total_tokens
   - step_latency_ms

文件：
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/flash_memory_constants.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/flash_memory_constants.py)
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py)

验收：
- 默认行为与旧版一致
- 日志能输出每步 token 消耗和预算余量

## Phase 1: Ragged memory state 重构

目标：替换固定 13 槽位 list 为结构化 memory_state。

改动：
1. 新增 RaggedMemoryState 数据结构
2. 新增写入/读取辅助函数
3. 兼容旧接口（过渡期保留 adapter）

文件：
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py)

验收：
- 仅重构存储，不改变输出结果（在 grid 模式下）

## Phase 2: Native token ingest + frame_ptr 构建

目标：每个 chunk 接入原生动态 token，记录 frame 级边界与 meta。

改动：
1. 在 embed_new_video_clip 中新增 native ingest 路径
2. 由 chunk 的 thw 构造 frame_ptr（每帧 token 可变）
3. 建立 token_meta（time/chunk/frame/coords）

文件：
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py)

验收：
- 输入不同分辨率 chunk 时不再因 h,w 不等报错
- memory_state 正确记录变长分段

## Phase 3: Temporal/Spatial scheduler（预算受限）

目标：实现 12000 上限下的动态时空分配。

改动：
1. 新增 frame_repr 计算
2. 实现机制A（内容驱动比例）
3. 实现机制B（效用密度贪心）
4. 可选机制C（延迟闭环）
5. 强制 budget guard：任何情况下 N_vis <= B_hard

文件：
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/compress_functions.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/compress_functions.py)
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py)

验收：
- 随机动态分辨率流输入 100 step，无 budget 超限
- scheduler 统计可解释（motion/detail 与 B_t/B_s 对应）

## Phase 4: AM-RoPE token-level 位置构建

目标：抛弃 thw 推导，改显式 token 位置编码。

改动：
1. 新增 build_position_ids_from_token_meta
2. prepare_realtime_inference 走新路径
3. get_rope_index 增加 ragged 分支

文件：
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime_annotated.py)

验收：
- position_ids 长度与最终 memory token 一致
- 与输入 prompt 的 video token 槽位对齐

## Phase 5: Processor 长度覆盖与端到端对齐

目标：文本侧 video token 展开长度来自 memory 实际输出。

改动：
1. 在 processor 新增可选覆盖参数
   - video_visual_token_length_override
2. streaming 推理时由 memory 返回真实长度并覆盖展开

文件：
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_processor.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_processor.py)

验收：
- 不同分辨率 chunk 下，无长度错位或索引越界

## Phase 6: 测试与基准

目标：证明 Route B 正确性与论文价值。

测试集：
1. 单元测试
   - ragged state 序列化/反序列化
   - budget allocator 不超限
   - position ids 对齐
2. pipeline 测试
   - 10/50/100 chunk 随机分辨率输入
3. 回归测试
   - 固定分辨率输入与旧逻辑一致（在 grid 模式）
4. 性能测试
   - 平均时延、P95 时延、token 使用率、空间问答准确率代理指标

建议测试文件：
- [Flash-VStream-highres/pipeline_test_streaming_video.py](Flash-VStream-highres/pipeline_test_streaming_video.py)
- 新增 tests 目录下 routeB 专项测试

---

## 7. 论文可写的关键论证点

1) 预算一致性
- 在统一 12000 上限下比较固定配额与动态配额，证明动态调度不是靠“更多 token”取胜。

2) 时空权衡可解释性
- 展示 motion/detail 指标如何驱动 B_t/B_s 变化。

3) 实时性收益
- 延迟闭环机制使得复杂场景下仍保持可控时延。

4) 空间理解收益
- 在细节密集场景中，spatial 配额上升带来更高空间问答性能。

---

## 8. 推荐默认参数（Route B 初版）

- flash_memory_mode: ragged
- flash_memory_budget_target: 11520
- flash_memory_budget_hard: 12000
- scheduler: hybrid
- temporal_min_share: 0.30
- temporal_max_share: 0.70
- spatial_min_share: 0.30
- spatial_max_share: 0.70
- pid_enable: true
- pid_latency_target_ms: 按你的部署环境设定（建议先从当前P50+10%开始）

---

## 9. 一句话执行建议

先按 Phase 0-3 做出可运行的 ragged+budget scheduler 主干（确保任何时刻 <=12000），再做 Phase 4-6 完成位置编码对齐与论文级实验闭环。