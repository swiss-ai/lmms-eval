from typing import List, Optional, Tuple, Union

import PIL
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.media_encoder import encode_image_to_data_url

try:
    import numpy as np
    from decord import VideoReader, cpu
    from torchvision.transforms.functional import to_pil_image

    _VIDEO_SUPPORT = True
except ImportError:
    _VIDEO_SUPPORT = False
    eval_logger.warning("decord or torchvision not available; video evaluation will not work")

try:
    from transformers import Llama4ForConditionalGeneration
except ImportError:
    Llama4ForConditionalGeneration = None
    eval_logger.warning("Failed to import Llama4ForConditionalGeneration. Please install transformers>=4.51.0")


@register_model("llama4_scout")
class Llama4Scout(lmms):
    """
    Llama-4-Scout Model
    https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct

    Requirements:
    - transformers >= 4.51.0
    """

    def __init__(
        self,
        pretrained: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        max_new_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        max_frames_num: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if Llama4ForConditionalGeneration is None:
            raise ImportError("Llama4ForConditionalGeneration not available. Please install transformers>=4.51.0")

        valid_attn_implementations = [None, "flex_attention", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "device_map": self.device_map,
            "torch_dtype": torch.bfloat16,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        eval_logger.info(f"Loading Llama-4-Scout model from {pretrained}")

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = self.processor.tokenizer

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained)
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = self._tokenizer.pad_token_id
        if not hasattr(config, "eos_token_id") or config.eos_token_id is None:
            config.eos_token_id = self._tokenizer.eos_token_id
        model_kwargs["config"] = config

        self._model = Llama4ForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        self._fix_meta_buffers()
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.max_frames_num = max_frames_num

        self._config = self.model.config
        self._max_length = 131072
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP, FSDP, and DEEPSPEED are supported."
            if accelerator.distributed_type in (DistributedType.FSDP, DistributedType.DEEPSPEED):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _fix_meta_buffers(self) -> None:
        """Re-materialize Llama4VisionRotaryEmbedding.freqs_ci left on meta device.

        When loaded with device_map="auto", accelerate intercepts __setattr__ and
        leaves plain tensor attributes (not register_buffer) on meta device.
        We reproduce the exact computation from Llama4VisionRotaryEmbedding.__init__
        using the vision config, putting the result on CPU so that the module's
        forward (.to(hidden_states.device)) moves it to the right device on demand.
        """
        vision_config = self.model.config.vision_config
        idx = vision_config.image_size // vision_config.patch_size
        img_idx = torch.arange(idx**2, dtype=torch.int32).reshape(idx**2, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = -2  # ID_CLS_TOKEN
        frequencies_x = img_idx % idx
        frequencies_y = img_idx // idx
        freq_dim = vision_config.hidden_size // vision_config.num_attention_heads // 2
        rope_theta = vision_config.rope_parameters["rope_theta"]
        rope_freq = 1.0 / (rope_theta ** (torch.arange(0, freq_dim, 2)[: freq_dim // 2].float() / freq_dim))
        freqs_x = ((frequencies_x + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
        freqs_y = ((frequencies_y + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)
        freqs_ci = torch.view_as_complex(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1))

        n_fixed = 0
        for module in self._model.modules():
            if type(module).__name__ == "Llama4VisionRotaryEmbedding":
                if hasattr(module, "freqs_ci") and module.freqs_ci.is_meta:
                    module.freqs_ci = freqs_ci  # CPU tensor; forward does .to(device)
                    n_fixed += 1
        if n_fixed:
            eval_logger.info(f"Fixed {n_fixed} Llama4VisionRotaryEmbedding meta buffer(s)")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Llama4Scout")

    def _image_to_data_url(self, image: Image.Image) -> str:
        return encode_image_to_data_url(image, image_format="JPEG", mime_type="image/jpeg", convert_rgb=True, quality=85)

    def _load_video(self, video_path: str) -> List[Image.Image]:
        if not _VIDEO_SUPPORT:
            raise ImportError("decord and torchvision are required for video evaluation")
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        frame_idx = np.linspace(0, total_frame_num - 1, self.max_frames_num, dtype=int).tolist()
        frames = torch.from_numpy(vr.get_batch(frame_idx).asnumpy()).permute(0, 3, 1, 2)
        return [to_pil_image(frame) for frame in frames]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            visual_list = [doc_to_visual[0](self.task_dict[t][s][ids]) for ids, t, s in zip(doc_id, task, split)]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]

            batched_messages = []
            for i, context in enumerate(contexts):
                context = context.replace("<image>", "")
                message = []
                if self.system_prompt:
                    message.append({"role": "system", "content": self.system_prompt})

                content = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, PIL.Image.Image):
                            content.append({"type": "image", "url": self._image_to_data_url(visual)})
                        elif isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                            for frame in self._load_video(visual):
                                content.append({"type": "image", "url": self._image_to_data_url(frame)})
                        elif isinstance(visual, str):
                            content.append({"type": "image", "url": visual})

                content.append({"type": "text", "text": context})
                message.append({"role": "user", "content": content})
                batched_messages.append(message)

            inputs = self.processor.apply_chat_template(
                batched_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for Llama4Scout")
