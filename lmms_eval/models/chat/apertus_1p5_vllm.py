import re
from dataclasses import replace

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.vllm import VLLM

_INNER_PREFIX = "<|inner_prefix|>"
_INNER_SUFFIX = "<|inner_suffix|>"
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


@register_model("apertus_1p5_vllm")
class Apertus1p5VLLM(VLLM):
    """Apertus 1.5 vLLM chat wrapper with chat-template-safe tokenization."""

    _chat_add_special_tokens = False

    def _build_sampling_params_dict(self, gen_kwargs):
        params = super()._build_sampling_params_dict(gen_kwargs)
        # Keep the deliberation delimiters so _strip_thinking can split the
        # answer off; otherwise the raw chain-of-thought reaches the scorer.
        params["skip_special_tokens"] = not self.enable_thinking
        return params

    def generate_until(self, requests):
        results = super().generate_until(requests)
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
