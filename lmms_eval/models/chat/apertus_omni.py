from __future__ import annotations

import os
import time
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
        debug_samples_raw = kwargs.pop("debug_samples", os.getenv("LMMS_APERTUS_DEBUG_SAMPLES", "5"))
        debug_max_chars_raw = kwargs.pop("debug_max_chars", os.getenv("LMMS_APERTUS_DEBUG_MAX_CHARS", "4000"))

        super().__init__(
            model_descriptor=model_descriptor,
            tokenizer_path=tokenizer_path,
            **kwargs,
        )
        tok_path = self.tokenizer_path
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

        try:
            self.debug_samples = max(0, int(debug_samples_raw))
        except (TypeError, ValueError):
            self.debug_samples = 5
        try:
            self.debug_max_chars = max(0, int(debug_max_chars_raw))
        except (TypeError, ValueError):
            self.debug_max_chars = 4000
        self._debug_logged_samples = 0

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

    def _replace_media_with_placeholders(self, hf_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rewritten_messages = []
        for msg in hf_messages:
            content = msg.get("content")
            if isinstance(content, list):
                rewritten_parts = []
                for part in content:
                    part_type = part.get("type")
                    if part_type == "audio":
                        rewritten_parts.append({"type": "text", "text": self.audio_placeholder})
                    elif part_type == "image":
                        rewritten_parts.append({"type": "text", "text": self.image_placeholder})
                    else:
                        rewritten_parts.append(part)
                rewritten_messages.append(
                    {
                        **msg,
                        "content": rewritten_parts,
                    }
                )
            else:
                rewritten_messages.append(msg)
        return rewritten_messages

    def _render_fallback_prompt(self, hf_messages: list[dict[str, Any]], fallback_context: Any) -> str:
        rendered_messages: list[str] = []
        for msg in hf_messages:
            role = str(msg.get("role", "user"))
            content = msg.get("content")

            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                text = "".join(parts)
            else:
                text = ""

            rendered_messages.append(f"{role}: {text}".strip())

        prompt = "\n".join(item for item in rendered_messages if item).strip()
        return prompt or str(fallback_context)

    def _build_chat_prompt(self, chat_messages: ChatMessages, fallback_context: Any) -> str:
        hf_messages = chat_messages.to_hf_messages()
        rewritten_messages = self._replace_media_with_placeholders(hf_messages)
        transformed_messages = self._chat_transform(rewritten_messages)
        try:
            return self.tokenizer.apply_chat_template(
                transformed_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            eval_logger.warning(f"ApertusOmniChat: apply_chat_template failed; falling back to raw context. Error: {e}")
            return self._render_fallback_prompt(rewritten_messages, fallback_context)

    def _truncate_for_debug(self, text: str) -> str:
        if self.debug_max_chars <= 0 or len(text) <= self.debug_max_chars:
            return text
        head = self.debug_max_chars // 2
        tail = self.debug_max_chars - head
        hidden = len(text) - self.debug_max_chars
        return f"{text[:head]}\n...[truncated {hidden} chars]...\n{text[-tail:]}"

    def _maybe_log_debug_sample(
        self,
        request: Instance,
        prompt_dict: dict[str, Any],
        output_text: str,
        gen_kwargs: dict[str, Any],
    ) -> None:
        if self.rank != 0:
            return
        if self.debug_samples <= 0 or self._debug_logged_samples >= self.debug_samples:
            return

        try:
            context, _doc_to_messages, _request_gen_kwargs, doc_id, task, split = request.arguments
            prompt_text = str(prompt_dict.get("prompt", context))
            mm_data = prompt_dict.get("multi_modal_data") or {}
            image_data = mm_data.get("image") if isinstance(mm_data, dict) else None
            audio_data = mm_data.get("audio") if isinstance(mm_data, dict) else None
            image_count = len(image_data) if isinstance(image_data, list) else (1 if image_data is not None else 0)
            audio_count = len(audio_data) if isinstance(audio_data, list) else (1 if audio_data is not None else 0)

            sample_idx = self._debug_logged_samples + 1
            eval_logger.info(
                f"[ApertusOmniChat Debug {sample_idx}/{self.debug_samples}] "
                f"task={task}, split={split}, doc_id={doc_id}, images={image_count}, audios={audio_count}, "
                f"prompt_chars={len(prompt_text)}, output_chars={len(output_text)}"
            )
            eval_logger.info(f"[ApertusOmniChat Debug {sample_idx}] gen_kwargs={gen_kwargs}")
            eval_logger.info(f"[ApertusOmniChat Debug {sample_idx}] input_prompt:\n{self._truncate_for_debug(prompt_text)}")
            eval_logger.info(f"[ApertusOmniChat Debug {sample_idx}] output_text:\n{self._truncate_for_debug(output_text)}")
            self._debug_logged_samples += 1
        except Exception as e:
            eval_logger.warning(f"ApertusOmniChat: debug sample logging failed: {e}")

    def make_one_request(self, request: Instance) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, int]]:
        counters = {
            "text_only": 0,
            "multi_image": 0,
            "audio_present": 0,
            "unsupported_video": 0,
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

        if videos:
            counters["unsupported_video"] = 1
            counters["skipped"] = 1
            return None, gen_kwargs, counters

        try:
            images = self._normalize_images(images)
            audios = self._normalize_audios(audios)
        except Exception as e:
            eval_logger.warning(f"ApertusOmniChat: failed to normalize media, returning empty output. Error: {e}")
            counters["failed"] = 1
            counters["skipped"] = 1
            return None, gen_kwargs, counters

        if len(images) == 0 and len(audios) == 0:
            counters["text_only"] = 1
            if self.skip_text_only:
                counters["skipped"] = 1
                return None, gen_kwargs, counters

        if audios:
            counters["audio_present"] = 1

        if len(images) > 1:
            counters["multi_image"] = 1
            if self.skip_multi_image:
                counters["skipped"] = 1
                return None, gen_kwargs, counters

        prompt = self._build_chat_prompt(chat_messages, context)
        prompt_dict = self._build_prompt_dict(prompt, images=images, audios=audios)
        return prompt_dict, gen_kwargs, counters

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        self.load_cache()
        res, requests = self.get_response_from_cache(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        text_only_count = 0
        multi_image_count = 0
        audio_present_count = 0
        unsupported_video_count = 0
        failed_count = 0
        skipped_count = 0

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        for batch_idx, batch_requests in enumerate(batched_requests, start=1):
            batch_outputs = [""] * len(batch_requests)
            batched_inputs: list[tuple[int, dict[str, Any]]] = []
            sampling_params_dict: dict[str, Any] | None = None

            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = [executor.submit(self.make_one_request, request) for request in batch_requests]
                for idx, future in enumerate(futures):
                    prompt_dict, gen_kwargs, counters = future.result()

                    text_only_count += counters["text_only"]
                    multi_image_count += counters["multi_image"]
                    audio_present_count += counters["audio_present"]
                    unsupported_video_count += counters["unsupported_video"]
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
                if self.rank == 0:
                    eval_logger.info(f"ApertusOmniChat: running batch {batch_idx}/{len(batched_requests)} " f"with {len(prompt_dicts)} requests")
                batch_t0 = time.time()
                response_text = self._generate_batch(prompt_dicts, sampling_params)
                if self.rank == 0:
                    eval_logger.info(f"ApertusOmniChat: finished batch {batch_idx}/{len(batched_requests)} in " f"{time.time() - batch_t0:.2f}s")

                for (idx, prompt_dict), text in zip(batched_inputs, response_text):
                    batch_outputs[idx] = text
                    self._maybe_log_debug_sample(
                        batch_requests[idx],
                        prompt_dict=prompt_dict,
                        output_text=text,
                        gen_kwargs=sampling_params_dict,
                    )
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
                f"audio-present={audio_present_count}/{len(requests)}, "
                f"unsupported-video={unsupported_video_count}, "
                f"skipped={skipped_count}, failures={failed_count}"
            )

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for ApertusOmniChat.")
