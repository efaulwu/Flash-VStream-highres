# Flash-VStream Finetuning（结合新的 Ragged Memory 管理）

## 1) 当前 finetune 脚本在做什么

参考主脚本：
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/finetune_flash.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/finetune_flash.py)

当前训练本质是 **SFT（监督微调）**：

1. 用 `preprocess(...)` 把一条样本（对话 + 视频）打包成：
   - `input_ids`
   - `labels`
   - `attention_mask`
   - `pixel_values_videos`
   - `video_grid_thw`
   - `visual_position_ids`
2. `CustomTrainer.compute_loss(...)` 里先调用 `model.module.prepare_inputs_for_training(**inputs)`，再 `model(**inputs)` 算 LM loss。
3. 模型是 `FlashVStreamQwen2VLModel.from_pretrained(...)`，可以 LoRA/QLoRA。

---

## 2) 当前训练接口（你后续改造要保持兼容）

### 2.1 数据侧输出接口

`preprocess(...)` 返回字典字段（训练侧强依赖）：

- `input_ids: Tensor[B, L]`
- `labels: Tensor[B, L]`
- `attention_mask: Tensor[B, L]`
- `pixel_values_videos: List[Tensor]`
- `video_grid_thw: List[Tensor(3)]`
- `visual_position_ids: Tensor[B, L]`

这几个字段最终会进入模型 `forward(...)` 和 `prepare_inputs_for_training(...)`。

### 2.2 模型侧关键接口

参考：
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime.py)

现有接口：

- `embed_new_video_clip(pixel_values_videos, video_grid_thw, start_idx)`
- `prepare_realtime_inference(position_ids, visual_position_ids)`
- `forward(...)`
- `prepare_inputs_for_training(...)`

重要事实：
- **训练主路径默认不是 streaming 路径**（`use_video_streaming_mode` 通常不会在 SFT 里打开）。
- 即使你已接了 `ragged retriever`，如果不改训练流程，它不会成为主训练监督对象。

### 2.3 Processor 接口

参考：
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_processor.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_processor.py)

你已具备关键扩展：
- `video_visual_token_length_override`

这在 ragged streaming 训练里非常关键：
- `<|video_pad|>` 展开长度必须和 `retrieve` 得到的 `N_sel` 对齐。

---

## 3) 新记忆管理（Ragged）在当前代码中的位置

参考：
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/ragged_flash_memory_retriever.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/ragged_flash_memory_retriever.py)
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/flash_memory_constants.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/flash_memory_constants.py)
- [Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime.py](Flash-VStream-highres/Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime.py)

你目前已经有：

- `flash_memory_mode='ragged'`
- 预算参数：`flash_memory_budget_target=11520`, `flash_memory_budget_hard=12000`, `flash_memory_rt`
- `RaggedFlashMemoryRetriever.update(...)` / `retrieve(...)`
- realtime 路径日志（update/retrieve + 显存）

这非常适合做论文里的 dynamic-resolution memory training，但要把它纳入 SFT 主循环还需要额外改造。

---

## 4) 结合 Ragged Memory 的 finetuning 思路（推荐）

## 方案A：两阶段（推荐，工程最稳）

### Stage-1（离线常规 SFT）

目标：先稳住语言能力和视觉基础对齐。

- 继续用现有 `finetune_flash.py` 训练主干（不强开 streaming）
- memory 配置可先保持 `grid` 或 `ragged`，但不依赖 `embed_new_video_clip`

收益：
- 最小改造，训练风险低。

### Stage-2（streaming-aware SFT，核心创新）

目标：显式训练“chunk->memory->retrieve->回答”。

核心流程（每条样本）：
1. 将视频按 chunk 切分（动态分辨率可变）
2. 逐 chunk 调 `embed_new_video_clip(...)`
3. 调 `prepare_realtime_inference(...)` 得到 `selected_tokens`（N_sel）与 position
4. 用 `video_visual_token_length_override=N_sel` 构建 prompt
5. 前向计算 LM loss（仅回答 token 计损）

收益：
- 训练目标与推理目标一致，论文创新点能真正被优化。

---

## 5) 具体改造点（按优先级）

## P0：参数打通（必须）

在 `finetune_flash.py` 的 `ModelArguments` 增加 ragged 参数：

- `flash_memory_mode`
- `flash_memory_budget_target`
- `flash_memory_budget_hard`
- `flash_memory_rt`
- `flash_memory_xbin`
- `flash_memory_ybin`
- `flash_memory_topM_items`
- `flash_memory_num_anchors`

并在 `flash_memory_config` 构建时全部传入 `config.set_flash_memory_config(...)`。

否则训练脚本虽然能跑，但不会按你新的 memory 策略工作。

## P1：数据集增加 streaming 样本形态（必须）

当前 `preprocess(...)` 主要按“整段视频一次编码”处理。

建议新增 `preprocess_streaming(...)` 输出：

- `video_chunks`: List[chunk_payload]
  - 每个 chunk 含 `pixel_values_videos`, `video_grid_thw`, `chunk_id`
- `final_prompt_text`
- `labels`

并新增 `DataArguments`：

- `streaming_train: bool`
- `chunk_seconds` 或 `chunk_frames`
- `max_chunks_per_sample`

## P2：Trainer 中加入 streaming 前处理（必须）

当前 `compute_loss` 只做一次 `prepare_inputs_for_training`。

新增 streaming 分支：

1. `model.use_video_streaming_mode = True`
2. 对每个 chunk 调 `embed_new_video_clip(...)`
3. 调 `prepare_realtime_inference(...)` 获取 `N_sel`
4. 调 processor 时传 `video_visual_token_length_override=N_sel`
5. 用新的 `input_ids/visual_position_ids` 做 forward

注意：
- 一个 batch 内样本的 `N_sel` 可能不同，建议先从 `batch_size=1` 的 streaming SFT 做起。

## P3：损失设计（强烈建议）

在 streaming 阶段可采用：

- 主损失：答案 token 的 CE loss
- 辅助损失（可选）：
  - budget regularization（惩罚接近 hard 上限）
  - temporal/spatial 平衡正则（避免极端偏置）

---

## 6) 预算约束在 finetuning 中如何落地

硬约束：
- `N_sel <= 12000`

建议：

1. 在每个训练 step 里断言：
   - `assert selected_tokens.shape[0] <= flash_memory_budget_hard`
2. 将 `stats` 写日志：
   - `N_tem/N_spa/N_sel`
   - `B_target/B_hard/r_t`
   - `retrieve_ms`
3. 若超限，立即 fallback：
   - 最后一道裁剪（已有 retriever guard）

这样可以保证论文实验“预算公平对比”成立。

---

## 7) 与现有脚本的关系（你现在就能跑到哪一步）

当前可直接用于验证的脚本：
- [Flash-VStream-highres/pipeline_test_ragged_streaming_dummy.py](Flash-VStream-highres/pipeline_test_ragged_streaming_dummy.py)
- [Flash-VStream-highres/slurmscripts/ragged_pipeline_hopper.slurm](Flash-VStream-highres/slurmscripts/ragged_pipeline_hopper.slurm)

它已经验证：
- 动态分辨率 chunk
- ragged update/retrieve
- 预算上限 12000
- 日志与显存统计

但它还是“pipeline 验证”，不是完整训练循环。

---

## 8) 推荐执行路线（最小风险）

1. 先保留现有 `finetune_flash.py` 跑 Stage-1（确认基础 loss 收敛）
2. 新建 `finetune_flash_streaming.py` 做 Stage-2（不要一开始就重写原脚本）
3. 先做 `batch_size=1` streaming SFT，确认稳定后再考虑梯度累积扩展
4. 每个 epoch 导出 memory 统计曲线（N_sel、N_tem/N_spa、latency）用于论文图

---

## 9) 你可以直接复用的接口清单

- 模型 memory 更新：`embed_new_video_clip(...)`
- 模型 memory 读取：`prepare_realtime_inference(...)`
- processor 长度对齐：`video_visual_token_length_override`
- retriever 核心：`RaggedFlashMemoryRetriever.update/retrieve`

这四个接口足够搭建完整的 streaming-aware finetuning。

---

## 10) 一句话结论

当前代码已经具备 ragged memory 推理与验证能力；要把它“训练起来”，关键是把 SFT 数据管线从“整视频一次编码”改成“chunk 逐步写 memory + 按 N_sel 构建 prompt”的 streaming 训练闭环。