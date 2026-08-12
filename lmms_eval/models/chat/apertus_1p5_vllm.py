import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from apertus_image_tokenizer import splice_frames
from PIL import Image as PILImage
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.vllm import VLLM, WORKERS
from lmms_eval.protocol import ChatMessages
from lmms_eval.utils import eval_logger

_INNER_PREFIX = "<|inner_prefix|>"
_INNER_SUFFIX = "<|inner_suffix|>"
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")

DEFAULT_TOKENIZER_PATH = "swiss-ai/Apertus-v1.5-8B"


@register_model("apertus_1p5_vllm")
class Apertus1p5VLLM(VLLM):
    """Apertus 1.5 vLLM chat wrapper with chat-template-safe tokenization.

    The engine is a text-only ApertusForCausalLM: images are spliced into the
    prompt as framed visual-token text (Emu3.5 VQ, via the suite's shared
    apertus_image_tokenizer) before tokenization, so the engine only ever
    sees token ids. The chat renderer (which needs an HF multimodal processor
    the checkpoint does not ship) is bypassed entirely.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("skip_mm_profiling", True)
        self.enable_thinking = kwargs.pop("enable_thinking", False)
        tokenizer_path = kwargs.get("tokenizer") or os.environ.get("APERTUS_TOKENIZER_PATH") or DEFAULT_TOKENIZER_PATH
        super().__init__(*args, **kwargs)
        from transformers import AutoTokenizer

        self._ap_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
        # A local tokenizer dir may carry the template as a file; an HF repo id
        # ships it inside the tokenizer itself (chat_template=None uses that).
        self._ap_chat_template = None
        if os.path.isdir(tokenizer_path):
            template_path = os.path.join(tokenizer_path, "chat_template.jinja")
            if os.path.isfile(template_path):
                with open(template_path, encoding="utf-8") as f:
                    self._ap_chat_template = f.read()
        # A deliberation run that silently loses this flag still completes and
        # looks healthy, so record what the template will actually receive.
        eval_logger.info(f"apertus_1p5_vllm: enable_thinking={self.enable_thinking} tokenizer={tokenizer_path}")

    def _build_sampling_params_dict(self, gen_kwargs):
        params = super()._build_sampling_params_dict(gen_kwargs)
        # Keep the deliberation delimiters so _strip_thinking can split the
        # answer off; otherwise the raw chain-of-thought reaches the scorer.
        params["skip_special_tokens"] = not self.enable_thinking
        return params

    def _render_request(self, request):
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        template_messages, images = [], []
        for message in ChatMessages(messages=raw_messages).messages:
            parts = []
            for content in message.content:
                if content.type == "text":
                    parts.append({"type": "text", "text": content.text})
                elif content.type == "image":
                    img = content.url if isinstance(content.url, PILImage.Image) else PILImage.open(content.url)
                    images.append(img)
                    parts.append({"type": "image"})
                else:
                    raise ValueError(f"apertus_1p5_vllm does not support content type {content.type!r}")
            template_messages.append({"role": message.role, "content": {"parts": parts}})

        prompt = self._ap_tokenizer.apply_chat_template(
            template_messages,
            add_generation_prompt=True,
            tokenize=False,
            chat_template=self._ap_chat_template,
            enable_thinking=self.enable_thinking,
        )
        if images:
            prompt = splice_frames(prompt, images, self._ap_tokenizer)
        token_ids = self._ap_tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        prompt_data = {"prompt_token_ids": token_ids}

        gen = dict(gen_kwargs or {})
        gen["max_new_tokens"] = self._select_max_new_tokens(gen.get("max_new_tokens"))
        gen.setdefault("temperature", 0)
        gen.setdefault("top_p", 0.95)
        return prompt_data, self._build_sampling_params_dict(gen)

    def _run_generate(self, items):
        from vllm import SamplingParams

        response = self.client.generate(
            prompts=[prompt_data for prompt_data, _ in items],
            sampling_params=[SamplingParams(**params) for _, params in items],
        )
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in response]

    def generate_until(self, requests):
        results = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Apertus vLLM generate")
        batch_size = self.batch_size_per_gpu
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                rendered = list(executor.map(self._render_request, batch))
            outputs = self._run_tp_synced(rendered, self._run_generate)
            assert len(outputs) == len(batch)
            results.extend(GenerationResult(text=text, token_counts=TokenCounts(output_tokens=n_out)) for text, n_out in outputs)
            pbar.update(len(batch))
        pbar.close()

        if self.enable_thinking:
            results = [replace(r, text=self._strip_thinking(r.text)) for r in results]
        return results

    @staticmethod
    def _strip_thinking(text):
        if _INNER_SUFFIX in text:
            # Answer is the span after the close of the deliberation block, up
            # to any reopened (unclosed) block.
            answer = text.rsplit(_INNER_SUFFIX, 1)[1].split(_INNER_PREFIX, 1)[0]
        elif _INNER_PREFIX in text:
            # Opened but never closed: no committed answer to extract.
            return ""
        else:
            answer = text
        return _SPECIAL_TOKEN_RE.sub("", answer).strip()
