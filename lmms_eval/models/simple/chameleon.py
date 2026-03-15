import time
from typing import List, Optional, Tuple, Union

import PIL
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics

IMAGE_PLACEHOLDER = "<image>"


@register_model("chameleon")
class Chameleon(lmms):
    """Meta Chameleon model integration.

    Chameleon is a mixed-modal early-fusion VLM that tokenizes images
    into discrete tokens via VQ-VAE. Uses ChameleonForConditionalGeneration
    directly to avoid Auto class resolution issues.
    """

    def __init__(
        self,
        pretrained: str = "facebook/chameleon-7b",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[str] = "bfloat16",
        debug_samples: bool = False,
        num_debug_samples: int = 5,
        **kwargs,
    ) -> None:
        super().__init__()

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(
                f"attn_implementation must be one of "
                f"{valid_attn_implementations}, "
                f"got {attn_implementation}"
            )

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            torch_dtype = getattr(torch, torch_dtype)

        model_kwargs: dict = {
            "torch_dtype": torch_dtype,
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = ChameleonForConditionalGeneration.from_pretrained(
            pretrained, **model_kwargs
        ).eval()

        self.processor = ChameleonProcessor.from_pretrained(pretrained)
        self.processor.tokenizer.padding_side = "left"
        self._tokenizer = self.processor.tokenizer
        self._config = self._model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.debug_samples = debug_samples
        self.num_debug_samples = num_debug_samples
        self._debug_samples_printed = 0

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        if self.debug_samples and self.rank == 0:
            eval_logger.info(
                f"Debug mode enabled: will print first "
                f"{self.num_debug_samples} samples"
            )

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

    def flatten(self, input: list) -> list:
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Chameleon")

    def _get_stop_token_ids(self, until: list[str]) -> list[int]:
        """Convert stop strings to token IDs where possible."""
        stop_ids = [self.tokenizer.eos_token_id]
        for term in until:
            token_ids = self.tokenizer.encode(term, add_special_tokens=False)
            if len(token_ids) == 1:
                stop_ids.append(token_ids[0])
        return stop_ids

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )
        e2e_latency = 0
        total_tokens = 0

        # Group by gen_kwargs so different generation configs don't mix
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            (
                contexts,
                all_gen_kwargs,
                doc_to_visual,
                doc_id,
                task,
                split,
            ) = zip(*chunk)
            task = task[0]
            split = split[0]
            gen_kwargs = dict(all_gen_kwargs[0])

            visual_list = [
                doc_to_visual[0](self.task_dict[task][split][ids])
                for ids in doc_id
            ]

            # Build prompts and collect images for each sample
            batch_prompts = []
            batch_images = []
            for i, context in enumerate(contexts):
                images = []
                for visual in visual_list[i]:
                    if isinstance(visual, PIL.Image.Image):
                        images.append(visual.convert("RGB"))

                # Strip any existing <image> tokens from context
                clean_context = context.replace(IMAGE_PLACEHOLDER, "")
                # Place <image> at end of prompt (matching Chameleon's
                # expected format)
                prompt = (
                    clean_context
                    + IMAGE_PLACEHOLDER * len(images)
                )
                batch_prompts.append(prompt)
                batch_images.extend(images)

            expected_images = sum(
                prompt.count(IMAGE_PLACEHOLDER) for prompt in batch_prompts
            )
            if expected_images != len(batch_images):
                raise ValueError(
                    "Chameleon batching mismatch: "
                    f"{expected_images} <image> tokens in prompts but "
                    f"{len(batch_images)} images supplied."
                )

            inputs = self.processor(
                text=batch_prompts,
                images=batch_images if batch_images else None,
                padding=True,
                return_tensors="pt",
            )
            target_device = getattr(self.model, "device", self.device)
            for key, value in inputs.items():
                value = value.to(device=target_device)
                if value.is_floating_point():
                    value = value.to(dtype=self.model.dtype)
                inputs[key] = value

            # Build stop token IDs from until strings
            until = gen_kwargs.pop("until", [])
            if isinstance(until, str):
                until = [until]
            stop_token_ids = self._get_stop_token_ids(until)

            default_gen_kwargs = {
                "max_new_tokens": 1024,
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "do_sample": False
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

            start_time = time.time()
            with torch.no_grad():
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=stop_token_ids,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
            end_time = time.time()

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            answers_with_tokens = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            if self.debug_samples:
                prompts_with_tokens = self.processor.batch_decode(
                    inputs.input_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

            for i, (ans, context) in enumerate(zip(answers, contexts)):
                res.append(ans)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), ans
                )
                pbar.update(1)

                if (
                    self.debug_samples
                    and self._debug_samples_printed
                    < self.num_debug_samples
                    and self.rank == 0
                ):
                    self._debug_samples_printed += 1
                    ids = inputs.input_ids[i]
                    attn = inputs["attention_mask"][i]
                    seq_len = attn.shape[0]
                    n_pad = (attn == 0).sum().item()
                    n_real = (attn == 1).sum().item()
                    head = attn[:8].tolist()
                    tail = attn[-8:].tolist()
                    ids_list = ids.tolist()
                    eval_logger.info("=" * 80)
                    eval_logger.info(
                        f"DEBUG SAMPLE {self._debug_samples_printed}/"
                        f"{self.num_debug_samples}"
                    )
                    eval_logger.info("=" * 80)
                    eval_logger.info(
                        f"PROMPT (clean): {batch_prompts[i]}"
                    )
                    eval_logger.info(
                        f"PROMPT (with tokens): "
                        f"{prompts_with_tokens[i]}"
                    )
                    # Verify pixel_values presence and shape
                    eval_logger.info(
                        f"INPUT KEYS: {list(inputs.keys())}"
                    )
                    if "pixel_values" in inputs:
                        pv = inputs["pixel_values"]
                        eval_logger.info(
                            f"PIXEL VALUES: shape={pv.shape} "
                            f"dtype={pv.dtype} "
                            f"min={pv.min().item():.4f} "
                            f"max={pv.max().item():.4f} "
                            f"mean={pv.mean().item():.4f}"
                        )
                        # Run VQ-VAE to get actual image tokens
                        with torch.no_grad():
                            vq_tokens = (
                                self.model.model.get_image_tokens(pv)
                            )
                        vq_list = vq_tokens[i].tolist()
                        eval_logger.info(
                            f"VQ-VAE TOKENS: count={len(vq_list)} "
                            f"unique={len(set(vq_list))} "
                            f"first_20={vq_list[:20]} "
                            f"last_20={vq_list[-20:]}"
                        )
                    else:
                        eval_logger.warning(
                            "NO pixel_values in inputs! "
                            "Images not being processed!"
                        )
                    eval_logger.info(
                        f"INPUT IDS: total={len(ids_list)} "
                        f"unique={len(set(ids_list))} "
                        f"n_placeholder(8711)="
                        f"{ids_list.count(8711)}"
                    )
                    eval_logger.info(
                        f"ATTENTION MASK: len={seq_len} pad={n_pad} "
                        f"real={n_real} head={head} tail={tail}"
                    )
                    eval_logger.info(f"ANSWER (clean): {ans}")
                    eval_logger.info(
                        f"ANSWER (with tokens): {answers_with_tokens[i]}"
                    )
                    eval_logger.info(
                        f"ANSWER IDS: "
                        f"{generated_ids_trimmed[i].tolist()}"
                    )
                    eval_logger.info("=" * 80)

        res = re_ords.get_original(res)

        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        log_metrics(
            total_tokens=total_tokens,
            e2e_latency=e2e_latency,
            avg_speed=avg_speed,
        )

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError(
            "Multi-round generation is not implemented for Chameleon"
        )
