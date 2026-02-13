from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Tuple

from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.apertus_omni_base_model import ApertusOmniBaseModel
from lmms_eval.protocol import ChatMessages

WORKERS = int(os.getenv("WORKERS", "32"))


@register_model("apertus_omni")
class ApertusOmniChat(ApertusOmniBaseModel):
    """
    Apertus Omni adapter (chat), following the VLLMGenerate style.
    """

    is_simple = False

    def __init__(
        self,
        model_descriptor: str,
        tokenizer_path: str | None = None,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_descriptor=model_descriptor, **kwargs)
        tok_path = tokenizer_path or model_descriptor
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.chat_template = None
        if chat_template is not None:
            if os.path.sep in chat_template or chat_template.endswith((".jinja", ".jinja2", ".j2")):
                if not os.path.isfile(chat_template):
                    raise FileNotFoundError(f"Chat template file not found: {chat_template}")
                with open(chat_template, "r") as f:
                    self.chat_template = f.read()
            else:
                self.chat_template = chat_template

        if self.chat_template is not None:
            self.tokenizer.chat_template = self.chat_template

    @staticmethod
    def _chat_transform(hf_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        transformed = []
        for msg in hf_messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                transformed.append(
                    {
                        "role": msg["role"],
                        "content": {"parts": msg["content"]},
                    }
                )
            else:
                transformed.append(msg)
        return transformed

    def _build_chat_prompt(self, chat_messages: ChatMessages, fallback_context: Any) -> str:
        hf_messages = chat_messages.to_hf_messages()
        transformed_messages = self._chat_transform(hf_messages)
        try:
            return self.tokenizer.apply_chat_template(
                transformed_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            eval_logger.warning(f"ApertusOmniChat: apply_chat_template failed; falling back to raw context. Error: {e}")
            return str(fallback_context)

    def make_one_request(self, request: Instance) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, int]]:
        counters = {
            "text_only": 0,
            "multi_image": 0,
            "unsupported_modality": 0,
            "failed": 0,
            "skipped": 0,
        }

        context, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        gen_kwargs = self._normalize_gen_kwargs(gen_kwargs)

        try:
            sample = self.task_dict[task][split][doc_id]
            raw_messages = self._invoke_extractor(doc_to_messages, sample)
            chat_messages = raw_messages if isinstance(raw_messages, ChatMessages) else ChatMessages(messages=raw_messages)
            images, videos, audios = chat_messages.extract_media()
        except Exception as e:
            eval_logger.warning(f"ApertusOmniChat: failed to parse request, returning empty output. Error: {e}")
            counters["failed"] = 1
            counters["skipped"] = 1
            return None, gen_kwargs, counters

        if videos or audios:
            counters["unsupported_modality"] = 1
            counters["skipped"] = 1
            return None, gen_kwargs, counters

        try:
            images = self._normalize_images(images)
        except Exception as e:
            eval_logger.warning(f"ApertusOmniChat: failed to normalize images, returning empty output. Error: {e}")
            counters["failed"] = 1
            counters["skipped"] = 1
            return None, gen_kwargs, counters

        if len(images) == 0:
            counters["text_only"] = 1
            if self.skip_text_only:
                counters["skipped"] = 1
                return None, gen_kwargs, counters

        if len(images) > 1:
            counters["multi_image"] = 1
            if self.skip_multi_image:
                counters["skipped"] = 1
                return None, gen_kwargs, counters

        prompt = self._build_chat_prompt(chat_messages, context)
        prompt_dict = self._build_prompt_dict(prompt, images)
        return prompt_dict, gen_kwargs, counters

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        self.load_cache()
        res, requests = self.get_response_from_cache(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        text_only_count = 0
        multi_image_count = 0
        unsupported_modality_count = 0
        failed_count = 0
        skipped_count = 0

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        for batch_requests in batched_requests:
            batch_outputs = [""] * len(batch_requests)
            batched_inputs: list[tuple[int, dict[str, Any]]] = []
            sampling_params_dict: dict[str, Any] | None = None

            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = [executor.submit(self.make_one_request, request) for request in batch_requests]
                for idx, future in enumerate(futures):
                    prompt_dict, gen_kwargs, counters = future.result()

                    text_only_count += counters["text_only"]
                    multi_image_count += counters["multi_image"]
                    unsupported_modality_count += counters["unsupported_modality"]
                    failed_count += counters["failed"]
                    skipped_count += counters["skipped"]

                    if prompt_dict is None:
                        self.add_request_response_to_cache(batch_requests[idx], "")
                        continue

                    batched_inputs.append((idx, prompt_dict))
                    sampling_params_dict = gen_kwargs

            if batched_inputs and sampling_params_dict is not None:
                sampling_params = self._build_sampling_params(sampling_params_dict)
                prompt_dicts = [entry[1] for entry in batched_inputs]
                response_text = self._generate_batch(prompt_dicts, sampling_params)

                for (idx, _), text in zip(batched_inputs, response_text):
                    batch_outputs[idx] = text
                    self.add_request_response_to_cache(batch_requests[idx], text)

            res.extend(batch_outputs)
            pbar.update(len(batch_requests))

        pbar.close()

        if self.rank == 0:
            eval_logger.warning(
                f"ApertusOmniChat stats: text-only={text_only_count}/{len(requests)} "
                f"(skip_text_only={self.skip_text_only}), "
                f"multi-image={multi_image_count}/{len(requests)} "
                f"(skip_multi_image={self.skip_multi_image}), "
                f"unsupported(video/audio)={unsupported_modality_count}, "
                f"skipped={skipped_count}, failures={failed_count}"
            )

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for ApertusOmniChat.")
