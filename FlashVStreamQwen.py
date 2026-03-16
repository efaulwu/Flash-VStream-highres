"""
Flash-VStream Eval Code (Qwen backbone)

Weight examples:
- https://huggingface.co/zhang9302002/Flash-VStream-Qwen-7b

This module mirrors the OVO-Bench wrapper style used by `FlashVStream.py`,
but runs the Qwen-based Flash-VStream pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import os
import sys

import torch
from decord import VideoReader, cpu

_QWEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Flash-VStream-Qwen")
if _QWEN_DIR not in sys.path:
    sys.path.insert(0, _QWEN_DIR)

try:
    from utils.OVOBench import OVOBenchOffline
except ImportError:
    class OVOBenchOffline:
        def __init__(self, args):
            self.args = args


def load_video(video_path: str):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = max(1, round(vr.get_avg_fps()))
    frame_idx = [i for i in range(0, len(vr), fps)]
    return vr.get_batch(frame_idx).asnumpy()


@dataclass
class _RuntimeConfig:
    max_new_tokens: int = 256
    do_sample: bool = False
    top_k: int = 1


class EvalFlashVStreamQwen(OVOBenchOffline):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.runtime = _RuntimeConfig(
            max_new_tokens=getattr(args, "max_new_tokens", 256),
            do_sample=getattr(args, "do_sample", False),
            top_k=getattr(args, "top_k", 1),
        )
        self._model_init()

    def _model_init(self):
        from models import (
            FlashVStreamQwen2VLConfig,
            FlashVStreamQwen2VLModel,
            FlashVStreamQwen2VLProcessor,
            DEFAULT_FLASH_MEMORY_CONFIG,
        )

        model_path = "zhang9302002/Flash-VStream-Qwen-7b"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_config = FlashVStreamQwen2VLConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        model_flash_config = getattr(model_config.vision_config, "flash_memory_config", None)
        if model_flash_config is None:
            model_config.vision_config.flash_memory_config = DEFAULT_FLASH_MEMORY_CONFIG.copy()
        self.flash_memory_config = model_config.vision_config.flash_memory_config

        attn_impl = getattr(self.args, "attn_implementation", None)
        if attn_impl is None:
            attn_impl = "flash_attention_2" if self.device.type == "cuda" else "eager"

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.model = FlashVStreamQwen2VLModel.from_pretrained(
            model_path,
            config=model_config,
            device_map="auto" if self.device.type == "cuda" else {"": "cpu"},
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        ).eval()

        self.processor = FlashVStreamQwen2VLProcessor.from_pretrained(model_path)

    def _build_text_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "<video>"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _move_inputs_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def inference(self, video_file_name: str, prompt: str):
        try:
            video = load_video(video_file_name)
            text_prompt = self._build_text_prompt(prompt)

            inputs = self.processor(
                text=[text_prompt],
                videos=[video],
                padding=True,
                return_tensors="pt",
                flash_memory_config=self.flash_memory_config,
            )
            inputs = self._move_inputs_to_device(dict(inputs))

            generate_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": self.runtime.max_new_tokens,
                "top_k": self.runtime.top_k,
                "do_sample": self.runtime.do_sample,
            }
            for opt_key in ("pixel_values_videos", "video_grid_thw", "visual_position_ids"):
                if opt_key in inputs:
                    generate_kwargs[opt_key] = inputs[opt_key]

            with torch.inference_mode():
                generated_ids = self.model.generate(**generate_kwargs)

            input_ids = inputs["input_ids"]
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]

            outputs = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return outputs[0].strip() if outputs else None
        except Exception as exc:
            print(exc)
            return None
