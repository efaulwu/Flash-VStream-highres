# Streaming Finetune Variants (Full-Res vs Dynamic-Res)

## Variant A: Full-Resolution Streaming

目标：保留视频原始分辨率，不做人为降采样，按时间 chunk 流式写入 memory。

流程：
1. 读取视频帧序列 `frames: [T,C,H,W]`。
2. 按 `frames_per_chunk` 切分为多个 chunk。
3. 每个 chunk 保持原始 `H,W`。
4. 每个 chunk 通过 processor 变成 `pixel_values_videos + video_grid_thw`。
5. 顺序调用 `model.embed_new_video_clip(...)` 更新记忆。
6. 最后按当前记忆做 `prepare_realtime_inference(...)` / forward。

优点：
- 最大化保留细节。
- 与真实部署“原分辨率输入”更一致。

风险：
- 显存/延迟波动较大，尤其在高分辨率片段。

---

## Variant B: Dynamic-Resolution Chunked Streaming

目标：视频按 chunk 处理，并为每个 chunk 随机采样一个分辨率（保持比例 + 对齐倍数），模拟动态调度训练。

流程：
1. 读取视频帧序列 `frames: [T,C,H,W]`。
2. 按 `frames_per_chunk` 切分。
3. 对每个 chunk：
   - 随机采样短边目标值（如 240/360/480/720）。
   - 保持宽高比。
   - 将 `H,W` 对齐到 `align_factor` 的倍数（通常 28 或 56）。
4. resize 后转 processor 输入。
5. 顺序更新 memory 并记录每步统计。

优点：
- 显式学习分辨率变化鲁棒性。
- 更贴合 dynamic-resolution 论文设定。

风险：
- 若采样策略过激，可能影响语义一致性。

---

## 训练接口建议

统一输出一个 chunk 列表，每个元素包含：
- `chunk_id: int`
- `frames: Tensor[t,C,H,W]`
- `orig_size: (H0,W0)`
- `used_size: (H,W)`

训练时统一调用：
- `run_streaming_memory_updates(model, processor, chunks, flash_memory_config, start_idx)`

这样可在同一 trainer 中切换 variant，仅替换 chunk builder。

---

## Dummy Test 覆盖点

1. Full-Res：
- chunk 数正确。
- 所有 chunk `used_size == orig_size`。

2. Dynamic-Res：
- 每个 chunk `H,W` 都是 `align_factor` 倍数。
- 至少存在一个 chunk 的 `used_size != orig_size`。

3. Memory update 调用：
- 每个 chunk 都调用一次 `embed_new_video_clip`。
- 返回统计长度与 chunk 数一致。
