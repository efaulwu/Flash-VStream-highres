#!/usr/bin/env python3

import json
import os
import signal
import sys
import threading
import time
import tracemalloc
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import transformers

try:
    import psutil
except ImportError as exc:  # pragma: no cover - psutil should exist on cluster nodes
    raise RuntimeError(
        "psutil is required for finetune_flash_memdebug.py. Please install it inside the runtime environment."
    ) from exc

import finetune_flash as ft

GenerationConfig = transformers.GenerationConfig
HfArgumentParser = transformers.HfArgumentParser


@dataclass
class MemoryDebugArguments:
    """Extra CLI arguments dedicated to memory instrumentation."""

    memlog_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to JSONL log that records periodic CPU usage snapshots."},
    )
    memlog_interval_seconds: float = field(
        default=30.0,
        metadata={"help": "Sampling interval for periodic process memory snapshots."},
    )
    step_log_interval: int = field(
        default=50,
        metadata={"help": "Trainer logging interval (in global steps) for fine-grained memory markers."},
    )
    enable_tracemalloc: bool = field(
        default=False,
        metadata={"help": "If true, include top Python allocators in periodic samples (can be expensive)."},
    )
    tracemalloc_topk: int = field(
        default=10,
        metadata={"help": "How many top tracemalloc entries to capture when enabled."},
    )
    dataset_cache_log_interval: int = field(
        default=200,
        metadata={"help": "Log LazySupervisedDataset cache growth every N newly cached samples."},
    )
    dataset_cache_max_entries: int = field(
        default=0,
        metadata={
            "help": "Maximum number of dataset samples to cache in RAM. 0 disables sample caching for CPU-memory safety."
        },
    )
    dataset_cache_store_pixels: bool = field(
        default=False,
        metadata={
            "help": "Whether cached samples are allowed to include pixel_values_videos/video_grid_thw. Keep False for CPU-memory safety."
        },
    )


class MemoryLogger:
    """Thread-safe helper that streams JSONL memory snapshots to disk."""

    def __init__(
        self,
        log_path: Path,
        interval_seconds: float,
        enable_tracemalloc: bool,
        tracemalloc_topk: int,
    ) -> None:
        self.process = psutil.Process(os.getpid())
        self.interval = max(1.0, float(interval_seconds))
        self.enable_tracemalloc = enable_tracemalloc
        self.tracemalloc_topk = max(1, tracemalloc_topk)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._file = open(self.log_path, "a", buffering=1)
        self.process.cpu_percent(None)  # prime CPU measurement
        if self.enable_tracemalloc:
            tracemalloc.start()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._periodic_loop, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.enable_tracemalloc:
            tracemalloc.stop()
        with self._lock:
            self._file.flush()
            self._file.close()

    # ------------------------------------------------------------------
    def log_stage(self, stage: str, component: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        self._write_record(
            {
                "event": "stage",
                "stage": stage,
                "component": component,
                "extra": extra or {},
            }
        )

    def log_exception(self, message: str, *, exc_type: str) -> None:
        self._write_record(
            {
                "event": "exception",
                "exc_type": exc_type,
                "message": message,
            }
        )

    def log_metadata(self, payload: Dict[str, Any]) -> None:
        self._write_record({"event": "metadata", "payload": payload})

    # ------------------------------------------------------------------
    def _periodic_loop(self) -> None:
        while not self._stop_event.wait(self.interval):
            self._write_record({"event": "periodic"}, include_traces=self.enable_tracemalloc)

    def _write_record(self, payload: Dict[str, Any], *, include_traces: bool = False) -> None:
        stats = self._collect_stats(include_traces=include_traces)
        record = {
            "ts": time.time(),
            **payload,
            **stats,
        }
        with self._lock:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()

    def _collect_stats(self, *, include_traces: bool) -> Dict[str, Any]:
        info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent(interval=None)
        vmem = psutil.virtual_memory()
        children_rss = 0
        for child in self.process.children(recursive=True):
            try:
                if child.is_running():
                    children_rss += child.memory_info().rss
            except psutil.Error:
                continue
        stats: Dict[str, Any] = {
            "rss_mb": round(info.rss / (1024 ** 2), 2),
            "vms_mb": round(info.vms / (1024 ** 2), 2),
            "shared_mb": round(getattr(info, "shared", 0) / (1024 ** 2), 2),
            "cpu_percent": round(cpu_percent, 2),
            "children_rss_mb": round(children_rss / (1024 ** 2), 2),
            "system_used_gb": round((vmem.total - vmem.available) / (1024 ** 3), 2),
            "system_available_gb": round(vmem.available / (1024 ** 3), 2),
        }
        if include_traces and self.enable_tracemalloc:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")[: self.tracemalloc_topk]
            stats["top_allocators"] = [
                {
                    "trace": str(stat.traceback[0]),
                    "size_kb": round(stat.size / 1024, 2),
                }
                for stat in top_stats
            ]
        return stats


class StepMemoryLogger(ft.TrainerCallback):
    """Hooks into Trainer logs to inject fine-grained memory samples."""

    def __init__(self, mem_logger: MemoryLogger, every_n_steps: int) -> None:
        self.mem_logger = mem_logger
        self.every_n_steps = max(1, every_n_steps)

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if state.global_step == 0:
            return control
        if state.global_step % self.every_n_steps != 0:
            return control
        serializable_logs: Dict[str, float] = {}
        if logs:
            for key in ("loss", "learning_rate", "grad_norm", "train_runtime"):
                value = logs.get(key)
                if isinstance(value, (int, float)):
                    serializable_logs[key] = float(value)
        serializable_logs["global_step"] = state.global_step
        self.mem_logger.log_stage("trainer_log", component="trainer", extra=serializable_logs)
        return control

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        summary = {
            "global_step": state.global_step,
            "train_runtime": getattr(state, "train_runtime", None),
        }
        self.mem_logger.log_stage("train_end", component="trainer", extra=summary)
        return control


def install_lazy_dataset_probe(mem_logger: MemoryLogger, mem_args: MemoryDebugArguments) -> None:
    """Monkey patch LazySupervisedDataset with a memory-safe cache policy and detailed logs."""

    dataset_cls = getattr(ft, "LazySupervisedDataset", None)
    if dataset_cls is None:
        return
    if getattr(dataset_cls, "__memdebug_wrapped", False):
        return

    original_init = dataset_cls.__init__

    def wrapped_init(self, *args, **kwargs):  # type: ignore[override]
        original_init(self, *args, **kwargs)
        self.cached_data_dict = OrderedDict()
        self._memsafe_cache_max_entries = max(0, int(mem_args.dataset_cache_max_entries))
        self._memsafe_cache_store_pixels = bool(mem_args.dataset_cache_store_pixels)
        self._memsafe_cache_log_interval = max(1, int(mem_args.dataset_cache_log_interval))
        mem_logger.log_stage(
            "dataset_cache_policy",
            component="dataset",
            extra={
                "cache_max_entries": self._memsafe_cache_max_entries,
                "cache_store_pixels": self._memsafe_cache_store_pixels,
            },
        )

    def wrapped_getitem(self, index):  # type: ignore[override]
        i = int(index)
        cache = getattr(self, "cached_data_dict", None)
        if cache is not None and i in cache:
            cached = cache.pop(i)
            cache[i] = cached
            return cached

        if "video" not in self.raw_data[i]:
            self.raw_data[i]["video"] = self.raw_data[i]["videos"][0] + ".mp4"

        ret = ft.preprocess(
            [self.raw_data[i]["conversations"]],
            [self.raw_data[i]["video"]],
            self.tokenizer,
            self.processor,
            self.max_len,
            self.data_args,
            self.flash_memory_config,
        )
        ret2 = {
            "input_ids": ret["input_ids"][0],
            "labels": ret["labels"][0],
            "attention_mask": ret["attention_mask"][0],
            "visual_position_ids": ret["visual_position_ids"][0],
        }
        if len(ret["pixel_values_videos"]) > 0:
            ret2["pixel_values_videos"] = ret["pixel_values_videos"][0]
            ret2["video_grid_thw"] = ret["video_grid_thw"][0]
            print(
                f"LazySupervisedDataset[{i}] contains, "
                f"pixel_values_videos={ret2['pixel_values_videos'].shape}, "
                f"video_grid_thw={ret2['video_grid_thw']}"
            )

        cache_max_entries = getattr(self, "_memsafe_cache_max_entries", 0)
        cache_store_pixels = getattr(self, "_memsafe_cache_store_pixels", False)
        log_interval = getattr(self, "_memsafe_cache_log_interval", 200)
        if cache is not None and cache_max_entries > 0:
            if cache_store_pixels:
                cache_value = ret2
            else:
                cache_value = {
                    k: v
                    for k, v in ret2.items()
                    if k not in ("pixel_values_videos", "video_grid_thw")
                }
            cache[i] = cache_value
            evicted = []
            while len(cache) > cache_max_entries:
                evicted_idx, _ = cache.popitem(last=False)
                evicted.append(int(evicted_idx))
            if evicted:
                mem_logger.log_stage(
                    "dataset_cache_eviction",
                    component="dataset",
                    extra={
                        "evicted_count": len(evicted),
                        "last_evicted_index": evicted[-1],
                        "cache_size": len(cache),
                    },
                )
            cache_size = len(cache)
            if cache_size > 0 and cache_size % log_interval == 0:
                mem_logger.log_stage(
                    "dataset_cache_growth",
                    component="dataset",
                    extra={
                        "cached_samples": cache_size,
                        "last_index": i,
                        "cache_store_pixels": cache_store_pixels,
                    },
                )

        return ret2

    dataset_cls.__init__ = wrapped_init  # type: ignore[assignment]
    dataset_cls.__getitem__ = wrapped_getitem  # type: ignore[assignment]
    dataset_cls.__memdebug_wrapped = True  # type: ignore[attr-defined]


def build_callbacks(training_args, mem_logger: MemoryLogger, mem_args: MemoryDebugArguments):
    callbacks = []
    if getattr(training_args, "python_gc_interval", 0) > 0:
        callbacks.append(ft.MemoryCleanupCallback(training_args.python_gc_interval))
    callbacks.append(StepMemoryLogger(mem_logger, mem_args.step_log_interval))
    return callbacks


def log_stage(mem_logger: MemoryLogger, stage: str, component: str, **extra):
    mem_logger.log_stage(stage, component=component, extra=extra or None)


def train_with_memory_safe_debug() -> None:
    parser = HfArgumentParser(
        (
            ft.ModelArguments,
            ft.DataArguments,
            ft.TrainingArguments,
            ft.LoraArguments,
            MemoryDebugArguments,
        )
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        mem_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None) and getattr(lora_args, "q_lora", False):
        training_args.distributed_state.distributed_type = ft.DistributedType.DEEPSPEED

    memlog_path = Path(mem_args.memlog_path or (Path(training_args.output_dir) / "cpu_memory_trace.jsonl"))
    mem_logger = MemoryLogger(
        log_path=memlog_path,
        interval_seconds=mem_args.memlog_interval_seconds,
        enable_tracemalloc=mem_args.enable_tracemalloc,
        tracemalloc_topk=mem_args.tracemalloc_topk,
    )
    mem_logger.start()
    mem_logger.log_metadata({"argv": sys.argv, "output_dir": training_args.output_dir})

    def _signal_handler(signum, frame):  # pragma: no cover - best-effort logging
        mem_logger.log_exception(
            f"Received signal {signum}",
            exc_type="signal",
        )
        mem_logger.shutdown()
        sys.exit(1)

    signal.signal(signal.SIGTERM, _signal_handler)

    install_lazy_dataset_probe(mem_logger, mem_args)

    shrink_factor = max(1, getattr(training_args, "batch_size_shrink_factor", 1))
    if shrink_factor > 1:
        shrink_applied = False
        if training_args.per_device_train_batch_size > 1:
            new_train_batch = max(1, training_args.per_device_train_batch_size // shrink_factor)
            if new_train_batch < training_args.per_device_train_batch_size:
                ft.rank0_print(
                    f"[memdebug] batch_size_shrink_factor={shrink_factor} -> per_device_train_batch_size {training_args.per_device_train_batch_size} -> {new_train_batch}"
                )
                training_args.per_device_train_batch_size = new_train_batch
                if getattr(training_args, "per_device_eval_batch_size", None):
                    training_args.per_device_eval_batch_size = max(
                        1, training_args.per_device_eval_batch_size // shrink_factor
                    )
                shrink_applied = True
        if not shrink_applied and training_args.gradient_accumulation_steps > 1:
            new_grad_acc = max(1, training_args.gradient_accumulation_steps // shrink_factor)
            if new_grad_acc < training_args.gradient_accumulation_steps:
                ft.rank0_print(
                    f"[memdebug] batch_size_shrink_factor={shrink_factor} -> gradient_accumulation_steps {training_args.gradient_accumulation_steps} -> {new_grad_acc}"
                )
                training_args.gradient_accumulation_steps = new_grad_acc
                shrink_applied = True
        if not shrink_applied:
            ft.rank0_print(
                f"[memdebug] batch_size_shrink_factor={shrink_factor} had no effect (already minimal micro-batch)."
            )

    local_rank = training_args.local_rank
    log_stage(mem_logger, "args_ready", "process", local_rank=local_rank)

    try:
        tokenizer = ft.AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        processor = ft.FlashVStreamQwen2VLProcessor.from_pretrained(model_args.model_name_or_path)
        log_stage(mem_logger, "tokenizer_processor_loaded", "io")

        base_config = ft.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        base_model_type = getattr(base_config, "model_type", "unknown")
        if base_model_type != "qwen2_vl":
            raise RuntimeError(
                f"Expected Qwen2-VL checkpoint (model_type=qwen2_vl), but got model_type={base_model_type}."
            )
        config_dict = base_config.to_dict()
        config_dict["model_type"] = "flash_vstream_qwen2_vl"
        vision_cfg = config_dict.get("vision_config", {})
        if isinstance(vision_cfg, dict):
            vision_cfg["model_type"] = "qwen2_vl"
            config_dict["vision_config"] = vision_cfg
        config = ft.FlashVStreamQwen2VLConfig.from_dict(config_dict)
        flash_memory_config = dict(
            flash_memory_temporal_length=model_args.flash_memory_temporal_length,
            flash_memory_temporal_method=model_args.flash_memory_temporal_method,
            flash_memory_temporal_poolsize=model_args.flash_memory_temporal_poolsize,
            flash_memory_temporal_pca_dim=model_args.flash_memory_temporal_pca_dim,
            flash_memory_spatial_length=model_args.flash_memory_spatial_length,
            flash_memory_spatial_method=model_args.flash_memory_spatial_method,
        )
        config.set_flash_memory_config(**flash_memory_config)

        model = ft.FlashVStreamQwen2VLModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            attn_implementation="flash_attention_2" if model_args.use_flash_attn else "eager",
        )
        log_stage(mem_logger, "model_loaded", "model")

        retriever = getattr(model, "ragged_retriever", None)
        if retriever is None:
            visual_module = getattr(model, "visual", None)
            if visual_module is None and hasattr(model, "model"):
                visual_module = getattr(model.model, "visual", None)
            retriever = getattr(getattr(visual_module, "flash_memory", None), "ragged_retriever", None)
        if retriever is None:
            raise RuntimeError("Expected realtime model with RaggedFlashMemoryRetriever, but none was found.")
        if local_rank in (0, -1, None):
            print(f"[memdebug] Ragged retriever backend active: {type(retriever).__name__}")

        target_modules = []
        pattern = ft.re.compile(r"^(?!.*visual).*(?:v_proj|gate_proj|up_proj|k_proj|o_proj|down_proj|q_proj).*")
        module_names = [name for name, _ in model.named_modules()]
        target_modules = [name for name in module_names if pattern.match(name)]
        target_modules.append("visual.merger.mlp.0")
        target_modules.append("visual.merger.mlp.2")
        lora_args.lora_target_modules = target_modules

        if training_args.use_lora:
            if not ft._PEFT_AVAILABLE:
                raise RuntimeError("PEFT is required for --use_lora but is not installed in the current environment.")
            if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
                modules_to_save = None
            else:
                modules_to_save = ["wte", "lm_head"]
            lora_config = ft.LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.lora_target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
                task_type="CAUSAL_LM",
                modules_to_save=modules_to_save,
            )
            if lora_args.q_lora:
                model = ft.prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=training_args.gradient_checkpointing
                )
            model = ft.get_peft_model(model, lora_config)
            if training_args.gradient_checkpointing:
                model.enable_input_require_grads()
            log_stage(mem_logger, "lora_wrapped", "model", modules=len(lora_args.lora_target_modules))

        data_module = ft.make_supervised_data_module(
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
            max_len=training_args.model_max_length,
            flash_memory_config=flash_memory_config,
        )
        train_dataset = data_module["train_dataset"]
        dataset_info = {
            "train_size": len(train_dataset),
            "lazy": bool(data_args.lazy_preprocess),
            "cache_max_entries": int(mem_args.dataset_cache_max_entries),
            "cache_store_pixels": bool(mem_args.dataset_cache_store_pixels),
        }
        cache_attr = getattr(train_dataset, "cached_data_dict", None)
        if cache_attr is not None:
            dataset_info["initial_cache"] = len(cache_attr)
        log_stage(mem_logger, "dataset_ready", "dataset", **dataset_info)

        callbacks = build_callbacks(training_args, mem_logger, mem_args)
        trainer = ft.CustomTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            callbacks=callbacks,
            **data_module,
        )
        log_stage(
            mem_logger,
            "trainer_ready",
            "trainer",
            train_batch_size=training_args.per_device_train_batch_size,
            grad_acc_steps=training_args.gradient_accumulation_steps,
        )

        log_stage(mem_logger, "training_start", "trainer")
        resume = bool(list(Path(training_args.output_dir).glob("checkpoint-*")))
        try:
            if resume:
                trainer.train(resume_from_checkpoint=True)
            else:
                trainer.train()
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover
            mem_logger.log_exception(str(exc), exc_type="cuda_oom")
            raise
        except MemoryError as exc:  # pragma: no cover
            mem_logger.log_exception("Host MemoryError", exc_type="cpu_oom")
            raise
        except RuntimeError as exc:  # pragma: no cover
            if "out of memory" in str(exc).lower():
                mem_logger.log_exception(str(exc), exc_type="runtime_oom")
            raise

        log_stage(mem_logger, "training_finished", "trainer")
        trainer.save_state()
        ft.safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=os.path.join(training_args.output_dir, "lora_model"),
            bias=lora_args.lora_bias,
        )
        if local_rank in (0, -1, None):
            full_model = model.merge_and_unload()
            full_model.generation_config = GenerationConfig.from_pretrained(
                model_args.model_name_or_path, trust_remote_code=True
            )
            ft.save_full_model(training_args.output_dir, full_model, tokenizer, processor)
        log_stage(mem_logger, "artifacts_saved", "trainer")

    finally:
        mem_logger.shutdown()


if __name__ == "__main__":
    train_with_memory_safe_debug()
