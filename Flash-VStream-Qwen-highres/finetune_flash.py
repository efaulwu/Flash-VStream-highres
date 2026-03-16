import gc
import json
import os
import pathlib
import re
import torch
from pathlib import Path
from transformers import AutoConfig

from models.vstream_qwen2vl_realtime import (
    FlashVStreamQwen2VLModel,
    FlashVStreamQwen2VLConfig,
    get_real_grid_thw,
    get_spatial_real_grid_thw,
)
from models import FlashVStreamQwen2VLProcessor, DEFAULT_FLASH_MEMORY_CONFIG

import transformers
transformers.FlashVStreamQwen2VLModel = FlashVStreamQwen2VLModel

from torch.utils.data import Dataset, IterableDataset
from safetensors.torch import load_file
from typing import Callable, Dict, Optional, List, Tuple, Union, Any
from transformers import Trainer, GPTQConfig, deepspeed, AutoProcessor, AutoTokenizer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import _is_peft_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
from transformers.training_args import TrainingArguments
from accelerate.utils import DistributedType
try:
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
except Exception:
    # Compatible with deepspeed builds where `zero` is not exported at top level.
    zero = None

    class ZeroParamStatus:
        NOT_AVAILABLE = None
from dataclasses import dataclass, field

from qwen_vl_utils import process_vision_info
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
    _PEFT_AVAILABLE = True
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    AutoPeftModelForCausalLM = None
    _PEFT_AVAILABLE = False

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    use_flash_attn: bool = False
    flash_memory_temporal_length: int = 120
    flash_memory_temporal_method: str = 'kmeans_ordered'
    flash_memory_temporal_poolsize: int = 2
    flash_memory_temporal_pca_dim: int = 32
    flash_memory_spatial_length: int = 60
    flash_memory_spatial_method: str = 'klarge_retrieve'

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    video_path: str = field(
        default=None, metadata={"help": "Path to the training video."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    fps: float = 1.0
    max_frames: int = 1000
    max_pixels: int = -1  # -1 means native resolution (no pixel count cap)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    batch_size_shrink_factor: int = field(
        default=1,
        metadata={
            "help": "Factor to divide the effective training batch size by. Applied to per-device batch size first, then gradient accumulation."
        },
    )
    python_gc_interval: int = field(
        default=0,
        metadata={
            "help": "Number of optimizer steps between explicit gc.collect / torch.cuda.empty_cache calls. 0 disables the periodic collection."
        },
    )


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "attn.c_proj", "w1", "w2"]  ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    longlora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id") and zero is not None:
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None

def rank0_print(*args):
    if local_rank in (0, -1, None):
        print(*args)


def _video_to_frame_dir(video_rel: str) -> str:
    if video_rel.lower().endswith(".mp4"):
        return video_rel[:-4]
    stem, _ = os.path.splitext(video_rel)
    return stem


def _resolve_frame_dir(video_root: str, video_rel: str) -> Optional[str]:
    raw_path = os.path.join(video_root, video_rel)
    candidates = [os.path.splitext(raw_path)[0]]
    if raw_path.lower().endswith(".mp4"):
        candidates.append(raw_path[:-4])

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def _has_decoded_frames(frame_dir: str) -> bool:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    try:
        with os.scandir(frame_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                _, ext = os.path.splitext(entry.name)
                if ext.lower() in valid_exts and entry.stat().st_size > 0:
                    return True
    except FileNotFoundError:
        return False
    return False


def _filter_samples_with_frames(raw_data: List[dict], video_root: str) -> List[dict]:
    kept: List[dict] = []
    skipped_missing_video = 0
    skipped_missing_dir = 0
    skipped_empty_dir = 0

    for item in raw_data:
        if not isinstance(item, dict):
            continue

        video_rel = item.get("video")
        if not isinstance(video_rel, str):
            videos = item.get("videos", [])
            if isinstance(videos, list) and len(videos) > 0 and isinstance(videos[0], str):
                video_rel = videos[0] + ".mp4"
                item["video"] = video_rel
            else:
                skipped_missing_video += 1
                continue

        frame_dir = _resolve_frame_dir(video_root, video_rel)
        if frame_dir is None:
            skipped_missing_dir += 1
            continue

        if not _has_decoded_frames(frame_dir):
            skipped_empty_dir += 1
            continue

        kept.append(item)

    rank0_print(
        f"[dataset] total={len(raw_data)} kept={len(kept)} "
        f"skip_missing_video={skipped_missing_video} "
        f"skip_missing_frame_dir={skipped_missing_dir} "
        f"skip_empty_or_invalid_frame_dir={skipped_empty_dir}"
    )
    return kept


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

"""
{
    "id": "000000",
    "conversations": [
        {
            "value": "<video>When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?\nA. Apples.\nB. Candles.\nC. Berries.\nD. The three kinds are of the same number.",
            "from": "user"
        },
        {
            "value": "C",
            "from": "assistant"
        }
    ],
    "videos": [
        "data/eval_video/videomme/video_frames/fFjv93ACGo8"
    ]
},
{   'id': '101-zukEQgbWdsY-split_2', 
    'conversations': [
        {'from': 'human', 'value': "<image>\nWhat is visible on the person's arm in the video?\nA. A tattoo\nB. A watch\nC. A bracelet\nD. A scar\nPlease respond with only the letter of the correct answer."
        },
        {'from': 'gpt', 'value': 'A.'
        },
        {'from': 'human', 'value': 'What is the color of the skillet in the video?\nA. Black\nB. Silver\nC. Red\nD. Blue\nPlease provide your answer by stating the letter followed by the full option.'
        },
        {'from': 'gpt', 'value': 'A. Black.'
        }
    ], 
    'data_source': '0_30_s_academic_v0_1', 
    'video': 'academic_source/youcook2/101/zukEQgbWdsY/split_2.mp4'
},
"""

def preprocess(
    sources,
    videos,
    tokenizer: transformers.PreTrainedTokenizer,
    processor,
    max_len: int,
    data_args,
    flash_memory_config,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer('<|im_start|>').input_ids
    im_end = tokenizer('<|im_end|>').input_ids
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets, visual_position_ids = [], [], []
    pixel_values_videos, video_grid_thw = [], []

    def _frame_sort_key(frame_name: str):
        stem = os.path.splitext(os.path.basename(frame_name))[0]
        digits = re.findall(r"\d+", stem)
        return int(digits[-1]) if digits else stem

    def _resolve_video_or_frame_input(video_root: str, video_rel: str):
        raw_path = os.path.join(video_root, video_rel)
        candidates = []
        no_ext = os.path.splitext(raw_path)[0]
        candidates.append(no_ext)
        if raw_path.lower().endswith(".mp4"):
            candidates.append(raw_path[:-4])

        frame_dir = None
        for candidate in candidates:
            if os.path.isdir(candidate):
                frame_dir = candidate
                break

        if frame_dir is not None:
            frame_paths = sorted(os.listdir(frame_dir), key=_frame_sort_key)
            frame_paths = [os.path.join(frame_dir, p) for p in frame_paths]
            return "frames", frame_paths

        return "video", raw_path
    
    for i, source in enumerate(sources):
        input_mode, video_input = _resolve_video_or_frame_input(data_args.video_path, videos[i])
        if input_mode == "frames":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_input,
                            "fps": data_args.fps,
                            "max_frames": data_args.max_frames,
                            **({"max_pixels": data_args.max_pixels} if data_args.max_pixels > 0 else {}),
                        },
                        {"type": "text", "text": "dummy_query"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "dummy_answer"},
                    ],
                }
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            video_inputs = processor.image_processor(
                images=None, 
                videos=video_inputs, 
                return_tensors="pt",
                additional_pool_size=flash_memory_config['flash_memory_temporal_poolsize'],
            )
            video_grid_thw_0 = video_inputs["video_grid_thw"][0]
            pixel_values_videos.append(video_inputs["pixel_values_videos"])
            video_grid_thw.append(video_inputs["video_grid_thw"][0])
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_input,
                            "fps": data_args.fps,
                            "max_frames": data_args.max_frames,
                            **({"max_pixels": data_args.max_pixels} if data_args.max_pixels > 0 else {}),
                        },
                        {"type": "text", "text": "dummy_query"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "dummy_answer"},
                    ],
                }
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            video_inputs = processor.image_processor(
                images=None, 
                videos=video_inputs, 
                return_tensors="pt",
                additional_pool_size=flash_memory_config['flash_memory_temporal_poolsize'],
            )
            video_grid_thw_0 = video_inputs["video_grid_thw"][0]
            pixel_values_videos.append(video_inputs["pixel_values_videos"])
            video_grid_thw.append(video_inputs["video_grid_thw"][0])

        input_id, target, visual_position_id = [], [], []
        # system
        # text = "<|im_start|>system\n"
        # text += "You are a helpful assistant.<|im_end|>\n"
        system = im_start + _system + tokenizer(system_message).input_ids + im_end + nl_tokens
        input_id += system
        target += [IGNORE_TOKEN_ID] * (len(system))
        visual_position_id += [-1] * len(system)
        assert len(input_id) == len(target)

        is_first_flag=True
        video_temporal_grid_thw_0 = get_real_grid_thw(video_grid_thw_0, flash_memory_config)
        video_spatial_grid_thw_0 = get_spatial_real_grid_thw(video_grid_thw_0, flash_memory_config)
        visual_embed_length = video_spatial_grid_thw_0.prod() // 4 + video_temporal_grid_thw_0.prod() // 4
        print(f'Generating video, video_grid_thw_0={video_grid_thw_0}, video_spatial_grid_thw_0={video_spatial_grid_thw_0}, video_temporal_grid_thw_0={video_temporal_grid_thw_0}')
        for idx, conv in enumerate(source):
            if idx % 2 == 0:
                assert conv["from"] == "human" or conv["from"] == "user"
                role = "user"
                query = conv["value"]
                if '<image>\n' in query:
                    query = query.replace('<image>\n', '')
                # user
                # text += "<|im_start|>user\n"
                # text += f"<|vision_start|><|video_pad|><|vision_end|>{query}<|im_end|>\n"
                if is_first_flag:  # first query, with vision embedding
                    is_first_flag=False
                    _input_id = im_start + tokenizer('user').input_ids + nl_tokens + \
                                tokenizer('<|vision_start|>').input_ids + tokenizer('<|video_pad|>').input_ids * visual_embed_length + tokenizer('<|vision_end|>').input_ids + \
                                tokenizer(query).input_ids + im_end + nl_tokens
                    input_id += _input_id
                    target += [IGNORE_TOKEN_ID] * (len(_input_id))
                    visual_position_id += [-1] * len(im_start + tokenizer('user').input_ids + nl_tokens + \
                                tokenizer('<|vision_start|>').input_ids) + [x for x in range(visual_embed_length)] + [-1] * len(tokenizer('<|vision_end|>').input_ids + \
                                tokenizer(query).input_ids + im_end + nl_tokens)
                else:  # others not
                    _input_id = im_start + tokenizer('user').input_ids + nl_tokens + \
                                tokenizer(query).input_ids + im_end + nl_tokens
                    input_id += _input_id
                    target += [IGNORE_TOKEN_ID] * (len(_input_id))
                    visual_position_id += [-1] * len(im_start + tokenizer('user').input_ids + nl_tokens + \
                                tokenizer(query).input_ids + im_end + nl_tokens)

            else:
                assert conv["from"] == "gpt" or conv["from"] == "assistant"
                role = "assistant"
                answer = conv["value"]
                # assistant
                # text += "<|im_start|>assistant\n"
                # text += f"{answer}<|im_end|>\n"
                _input_id = im_start + tokenizer('assistant').input_ids + nl_tokens + \
                            tokenizer(answer).input_ids + im_end + nl_tokens
                input_id += _input_id
                target += [IGNORE_TOKEN_ID] * (1 + len(tokenizer('assistant').input_ids)) + \
                                        _input_id[len(tokenizer('assistant').input_ids) + 1:-2] + im_end + [IGNORE_TOKEN_ID]
                visual_position_id += [-1] * len(_input_id)


        assert len(input_id) == len(target)
        assert len(input_id) == len(visual_position_id)
        print(f'padding {len(input_id)} to {max_len}')
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        visual_position_id += [-1] * (max_len - len(visual_position_id))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])  # Warning: this shift need to change Qwen2VL.forward()
        # assert len(visual_position_id) <= max_len, f"visual_position_id {len(visual_position_id)} > {max_len}, visual token is truncated"
        visual_position_ids.append(visual_position_id[:max_len])

    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    visual_position_ids = torch.tensor(visual_position_ids, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        visual_position_ids=visual_position_ids,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, processor, max_len: int, data_args, flash_memory_config):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.data_args = data_args
        self.flash_memory_config = flash_memory_config
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        
        if not 'video' in self.raw_data[i]:
            self.raw_data[i]['video'] = self.raw_data[i]['videos'][0] + '.mp4'
        ret = preprocess([self.raw_data[i]["conversations"]], [self.raw_data[i]['video']], self.tokenizer, self.processor, self.max_len, self.data_args, self.flash_memory_config)
        ret2 = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            visual_position_ids=ret["visual_position_ids"][0]
        )
        if len(ret["pixel_values_videos"]) > 0:
            ret2["pixel_values_videos"] = ret["pixel_values_videos"][0]
            ret2["video_grid_thw"] = ret["video_grid_thw"][0]
            print(f'LazySupervisedDataset[{i}] contains, pixel_values_videos={ret2["pixel_values_videos"].shape}, video_grid_thw={ret2["video_grid_thw"]}')

        self.cached_data_dict[i] = ret2
        return ret2


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, processor, data_args, max_len, flash_memory_config
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_json = _filter_samples_with_frames(train_json, data_args.video_path)
    if len(train_json) == 0:
        raise RuntimeError(
            f"No valid training samples remain after frame filtering. data_path={data_args.data_path}, video_path={data_args.video_path}"
        )
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, processor=processor, max_len=max_len, data_args=data_args, flash_memory_config=flash_memory_config)
    rank0_print(f"Training data loaded, length={len(train_dataset)}")

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, processor=processor, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)




class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # prepare attention_mask, m-rope, etc
        train_model = model.module if hasattr(model, "module") else model
        inputs = train_model.prepare_inputs_for_training(**inputs)

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class MemoryCleanupCallback(TrainerCallback):
    """Periodically run Python GC and release CUDA caches to fight host OOM."""

    def __init__(self, step_interval: int):
        self.step_interval = max(1, step_interval)

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_interval <= 0:
            return control
        if state.global_step == 0:
            return control
        if state.global_step % self.step_interval != 0:
            return control

        freed = gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        local_rank = getattr(args, "local_rank", -1)
        if local_rank in (0, -1, None):
            print(
                f"[memory-gc] step={state.global_step} interval={self.step_interval} freed_objects={freed}"
            )
        return control



def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    shrink_factor = max(1, getattr(training_args, "batch_size_shrink_factor", 1))
    if shrink_factor > 1:
        shrink_applied = False
        if training_args.per_device_train_batch_size > 1:
            new_train_batch = max(1, training_args.per_device_train_batch_size // shrink_factor)
            if new_train_batch < training_args.per_device_train_batch_size:
                rank0_print(
                    f"[main] batch_size_shrink_factor={shrink_factor} -> per_device_train_batch_size {training_args.per_device_train_batch_size} -> {new_train_batch}"
                )
                training_args.per_device_train_batch_size = new_train_batch
                if hasattr(training_args, "per_device_eval_batch_size") and training_args.per_device_eval_batch_size:
                    training_args.per_device_eval_batch_size = max(
                        1, training_args.per_device_eval_batch_size // shrink_factor
                    )
                shrink_applied = True
        if not shrink_applied and training_args.gradient_accumulation_steps > 1:
            new_grad_acc = max(1, training_args.gradient_accumulation_steps // shrink_factor)
            if new_grad_acc < training_args.gradient_accumulation_steps:
                rank0_print(
                    f"[main] batch_size_shrink_factor={shrink_factor} -> gradient_accumulation_steps {training_args.gradient_accumulation_steps} -> {new_grad_acc}"
                )
                training_args.gradient_accumulation_steps = new_grad_acc
                shrink_applied = True
        if not shrink_applied:
            rank0_print(
                f"[main] batch_size_shrink_factor={shrink_factor} had no effect (already at minimal micro-batch)."
            )

    # Load model
    rank0_print(f"[main] model_name_or_path={model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    processor = FlashVStreamQwen2VLProcessor.from_pretrained(model_args.model_name_or_path)
    base_config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    base_model_type = getattr(base_config, "model_type", "unknown")
    if base_model_type != "qwen2_vl":
        raise RuntimeError(
            f"Expected official Qwen2-VL checkpoint (model_type=qwen2_vl), but got model_type={base_model_type}."
        )
    rank0_print(f"[main] Official Qwen checkpoint verified: model_type={base_model_type}")

    config_dict = base_config.to_dict()
    config_dict["model_type"] = "flash_vstream_qwen2_vl"
    vision_cfg = config_dict.get("vision_config", {})
    if isinstance(vision_cfg, dict):
        vision_cfg["model_type"] = "qwen2_vl"
        config_dict["vision_config"] = vision_cfg
    config = FlashVStreamQwen2VLConfig.from_dict(config_dict)
    flash_memory_config = dict(
        flash_memory_temporal_length=model_args.flash_memory_temporal_length,
        flash_memory_temporal_method=model_args.flash_memory_temporal_method,
        flash_memory_temporal_poolsize=model_args.flash_memory_temporal_poolsize,
        flash_memory_temporal_pca_dim=model_args.flash_memory_temporal_pca_dim,
        flash_memory_spatial_length=model_args.flash_memory_spatial_length,
        flash_memory_spatial_method=model_args.flash_memory_spatial_method,
    )
    config.set_flash_memory_config(**flash_memory_config)
    model = FlashVStreamQwen2VLModel.from_pretrained(
        model_args.model_name_or_path, 
        config=config,
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        attn_implementation="flash_attention_2" if model_args.use_flash_attn else "eager",
    )
    rank0_print("[main] FlashVStream model weights loaded from official Qwen checkpoint path.")
    retriever = getattr(model, "ragged_retriever", None)
    if retriever is None:
        visual_module = getattr(model, "visual", None)
        if visual_module is None and hasattr(model, "model"):
            visual_module = getattr(model.model, "visual", None)
        retriever = getattr(getattr(visual_module, "flash_memory", None), "ragged_retriever", None)
    if retriever is None:
        raise RuntimeError(
            "Expected realtime model with RaggedFlashMemoryRetriever, but ragged_retriever was not found."
        )
    if local_rank in (0, -1, None):
        print(f"[main] Ragged retriever backend active: {type(retriever).__name__}")

    target_modules = []

    # only lora llm
    pattern = re.compile(r"^(?!.*visual).*(?:v_proj|gate_proj|up_proj|k_proj|o_proj|down_proj|q_proj).*")
    module_names = [name for name, _ in model.named_modules()]
    target_modules = [name for name in module_names if pattern.match(name)]
    target_modules.append('visual.merger.mlp.0')
    target_modules.append('visual.merger.mlp.2')
    if local_rank == 0:
        print(f'[main] target_modules={target_modules}')
    lora_args.lora_target_modules = target_modules

    if training_args.use_lora:
        if not _PEFT_AVAILABLE:
            raise RuntimeError("PEFT is required for --use_lora but is not installed in the current environment.")
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, processor=processor, data_args=data_args, max_len=training_args.model_max_length, flash_memory_config=flash_memory_config
    )

    callbacks: List[TrainerCallback] = []
    if getattr(training_args, "python_gc_interval", 0) > 0:
        callbacks.append(MemoryCleanupCallback(training_args.python_gc_interval))
        rank0_print(
            f"[main] Enabled periodic Python GC every {training_args.python_gc_interval} steps"
        )

    # Start trainer
    trainer = CustomTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        callbacks=callbacks if callbacks else None,
        **data_module,
    )
    print(f'rank:{local_rank}, start training')
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    print(f'rank:{local_rank}, save state')
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir + '/lora_model', bias=lora_args.lora_bias)
    if local_rank == 0:
        print(f'rank:{local_rank}, save whole model to {training_args.output_dir}')
        full_model = model.merge_and_unload()
        full_model.generation_config = GenerationConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        save_full_model(training_args.output_dir, full_model, tokenizer, processor)


def save_full_model(output_path, model, tokenizer, processor):
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    print(f"Model and tokenizer saved to {output_path}")
    chat_template_path = os.path.join(output_path, "chat_template.json")
    with open(chat_template_path, 'w') as f:
        json.dump({'chat_template': processor.chat_template}, f, indent=4)
    print(f"Chat template saved to {chat_template_path}")

if __name__ == "__main__":
    train()
