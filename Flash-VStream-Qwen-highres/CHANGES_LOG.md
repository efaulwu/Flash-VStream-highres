# Flash-VStream Query-Based Scoring 实现日志

**日期**：2026-03-16  
**分支**：`copilot/configure-git-settings`  
**操作者**：Copilot Agent

---

## 一、Git 配置检查

检查了 `Flash-VStream-Qwen-highres` 文件夹的 git 配置。

**结论**：该文件夹是主仓库（`efaulwu/Flash-VStream-highres`）的子目录，  
所有文件均已被父仓库的 git 追踪（可通过 `git ls-files Flash-VStream-Qwen-highres/` 确认）。  
**无需额外配置**——在该目录内执行 `git status`、`git log`、`git add`、`git commit` 等命令均可正常使用。

当前 git 配置：
- `remote.origin.url` = `https://github.com/efaulwu/Flash-VStream-highres`
- `branch` = `copilot/configure-git-settings`
- `user.name` = `copilot-swe-agent[bot]`

---

## 二、Visual Token 与 Query Embedding 对齐情况分析

### 问题

记忆模块（`RaggedFlashMemoryRetriever`）中的 `retrieve()` 调用使用了 `query_embed=None`，导致 query-based scoring 未被启用：

```python
# vstream_qwen2vl_realtime.py, 修改前（第 756 行）
selected_tokens, selected_meta, local_position_ids, stats = \
    self.ragged_retriever.retrieve(query_embed=None)
```

### 对齐分析

| 向量类型 | 来源 | 维度空间 | 是否需要额外映射 |
|---------|------|---------|----------------|
| Visual tokens（hi/lo） | `self.visual.merger(raw_tokens)` | `hidden_size`（LLM 空间） | **否** |
| Query text embeddings | `self.model.embed_tokens(input_ids)` | `hidden_size`（LLM 空间） | **否** |

**结论**：Visual tokens 在存入 `RaggedFlashMemoryRetriever` 前，已经通过 Qwen2VL 的 `PatchMerger` 连接层（`self.visual.merger`）投影到 LLM 的 `hidden_size` 空间（见 `embed_new_video_clip()` 第 642-647 行）。Query text embeddings 也在同一空间。**两者已对齐，无需额外投影。**

---

## 三、Query-Based Scoring 实现

### 修改文件

`Flash-VStream-Qwen-highres/models/vstream_qwen2vl_realtime.py`

### 修改 1：`prepare_realtime_inference()` 方法签名

```python
# 修改前
def prepare_realtime_inference(self, position_ids, visual_position_ids):
    ...
    selected_tokens, selected_meta, local_position_ids, stats = \
        self.ragged_retriever.retrieve(query_embed=None)

# 修改后
def prepare_realtime_inference(self, position_ids, visual_position_ids, query_embed=None):
    ...
    selected_tokens, selected_meta, local_position_ids, stats = \
        self.ragged_retriever.retrieve(query_embed=query_embed)
```

### 修改 2：`forward()` 方法提取 Query Embedding

在 streaming 推理路径中，提取文本 token 的嵌入向量并取平均作为 query embedding：

```python
if self.use_video_streaming_mode:
    video_mask = input_ids == self.config.video_token_id
    if video_mask.sum() > 0:
        # 从文本 token 嵌入（hidden_size 空间）中构建 query embedding
        # visual token 已通过 PatchMerger 投影到同一空间，无需额外映射
        query_embed = None
        if self._is_ragged_mode():
            text_mask = ~video_mask
            if text_mask.any():
                query_embed = inputs_embeds[text_mask].mean(dim=0).detach()
        video_embeds, position_ids = self.prepare_realtime_inference(
            position_ids, visual_position_ids, query_embed=query_embed
        )
```

### 工作原理

`RaggedFlashMemoryRetriever.retrieve()` 中：

1. **Temporal scoring**（`_temporal_candidates()`）：用 query embedding 与各 cluster centroid 做余弦相似度，加权后选出最高 utility 的 anchor。
2. **Spatial scoring**（`_spatial_candidates()`）：对每个候选 chunk 的 hi-resolution tokens，计算与 query embedding 的点积相似度，选 top-K 个 token。

两阶段均使用了 query embedding 驱动的检索，实现了基础的 query-based scoring。

---

## 四、SLURM 脚本

生成了 `run_pipeline_test.slurm`，配置如下：

- 2 块 GPU，8 核 CPU，64G 内存，4 小时时限
- 激活 `vstream` conda 环境
- 调用 `pipeline_test_1gpu.py`，启用 `--flash_memory_mode ragged`
- 日志保存到 `logs/slurm_<JOB_ID>.out`

**使用方法**（在集群上）：
```bash
cd Flash-VStream-Qwen-highres
sbatch run_pipeline_test.slurm
```

> ⚠️ 提交前请修改 `MODEL_PATH` 和 `VIDEO_PATH` 为实际路径，并按集群实际情况调整 `--partition`、`--gres` 等参数。

---

## 五、文件变更总结

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `models/vstream_qwen2vl_realtime.py` | 修改 | 启用 query-based scoring |
| `run_pipeline_test.slurm` | 新增 | SLURM 作业提交脚本 |
| `CHANGES_LOG.md` | 新增 | 本文档 |
