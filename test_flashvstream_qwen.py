from types import SimpleNamespace
from pprint import pformat

import numpy as np
import torch

from FlashVStreamQwen import EvalFlashVStreamQwen


class _DummyProcessor:
    def __init__(self):
        self.last_messages = None
        self.last_text = None
        self.last_videos = None
        self.last_flash_memory_config = None
        self.last_return_tensors = None
        self.last_padding = None
        self.last_tokenize = None
        self.last_add_generation_prompt = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_messages = messages
        self.last_tokenize = tokenize
        self.last_add_generation_prompt = add_generation_prompt
        return "dummy-template"

    def __call__(self, text, videos, padding, return_tensors, flash_memory_config):
        self.last_text = text
        self.last_videos = videos
        self.last_flash_memory_config = flash_memory_config
        self.last_return_tensors = return_tensors
        self.last_padding = padding
        return {
            "input_ids": torch.tensor([[11, 22, 33]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "pixel_values_videos": torch.randn(8, 1176),
            "video_grid_thw": torch.tensor([[2, 2, 2]], dtype=torch.long),
            "visual_position_ids": torch.arange(2, dtype=torch.long).unsqueeze(0),
        }

    def batch_decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ["dummy answer"]


class _DummyModel:
    def __init__(self):
        self.last_kwargs = None
        self.last_generated_ids = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        appended = torch.tensor([[101, 102]], dtype=input_ids.dtype, device=input_ids.device)
        self.last_generated_ids = torch.cat([input_ids, appended], dim=1)
        return self.last_generated_ids


def _shape_dtype(tensor):
    if not torch.is_tensor(tensor):
        return "<non-tensor>"
    return f"shape={list(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"


def _print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_qwen_wrapper_inference_with_dummy_pipeline(monkeypatch=None):
    _print_section("[Test Start] Qwen wrapper dummy pipeline")

    def fake_model_init(self):
        self.device = torch.device("cpu")
        self.flash_memory_config = {
            "flash_memory_temporal_length": 120,
            "flash_memory_temporal_method": "kmeans_ordered",
            "flash_memory_temporal_poolsize": 1,
            "flash_memory_temporal_pca_dim": 32,
            "flash_memory_spatial_length": 0,
            "flash_memory_spatial_method": "klarge_retrieve",
        }
        self.processor = _DummyProcessor()
        self.model = _DummyModel()

    if monkeypatch is not None:
        monkeypatch.setattr(EvalFlashVStreamQwen, "_model_init", fake_model_init)
        video_shape = (4, 32, 32, 3)
        video_dtype = np.uint8
        monkeypatch.setattr(
            "FlashVStreamQwen.load_video",
            lambda _path: np.zeros(video_shape, dtype=video_dtype),
        )
    else:
        setattr(EvalFlashVStreamQwen, "_model_init", fake_model_init)
        video_shape = (4, 32, 32, 3)
        video_dtype = np.uint8
        import FlashVStreamQwen
        setattr(FlashVStreamQwen, "load_video", lambda _path: np.zeros(video_shape, dtype=video_dtype))

    args = SimpleNamespace(model_path="dummy")
    _print_section("[Config] Runtime arguments")
    print(pformat(vars(args), sort_dicts=True))

    evaluator = EvalFlashVStreamQwen(args)
    _print_section("[Config] Loaded flash memory config")
    print(pformat(evaluator.flash_memory_config, sort_dicts=True))
    print(f"device={evaluator.device}")

    prompt = "What happens in the clip?"
    _print_section("[Input] Dummy video + prompt")
    print(f"video_shape={video_shape}, video_dtype={video_dtype}")
    print(f"prompt={prompt}")

    output = evaluator.inference("dummy.mp4", prompt)

    _print_section("[Processor] Template + call diagnostics")
    print(f"apply_chat_template.tokenize={evaluator.processor.last_tokenize}")
    print(f"apply_chat_template.add_generation_prompt={evaluator.processor.last_add_generation_prompt}")
    print("messages=")
    print(pformat(evaluator.processor.last_messages, sort_dicts=False, width=100))
    print(f"processor.text={evaluator.processor.last_text}")
    video_info = evaluator.processor.last_videos[0] if evaluator.processor.last_videos else None
    if isinstance(video_info, np.ndarray):
        print(f"processor.video.shape={list(video_info.shape)}, dtype={video_info.dtype}")
    print(f"processor.padding={evaluator.processor.last_padding}")
    print(f"processor.return_tensors={evaluator.processor.last_return_tensors}")
    print("processor.flash_memory_config=")
    print(pformat(evaluator.processor.last_flash_memory_config, sort_dicts=True))

    _print_section("[Model] Generation kwargs diagnostics")
    for key, value in evaluator.model.last_kwargs.items():
        if torch.is_tensor(value):
            print(f"{key}: {_shape_dtype(value)}")
        else:
            print(f"{key}: {value}")
    print(f"generated_ids: {_shape_dtype(evaluator.model.last_generated_ids)}")

    _print_section("[Output] Decoded response")
    print(f"output={output}")

    assert output == "dummy answer"
    assert evaluator.model.last_kwargs is not None
    assert "pixel_values_videos" in evaluator.model.last_kwargs
    assert evaluator.model.last_kwargs["pixel_values_videos"].shape[1] == 1176

    _print_section("[Test End] Assertions passed")


if __name__ == "__main__":
    test_qwen_wrapper_inference_with_dummy_pipeline(None)