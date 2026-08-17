from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.vllm import VLLM


@register_model("apertus_1p5_vllm")
class Apertus1p5VLLM(VLLM):
    """Simple-mode stub: the generic simple vLLM path would hand raw images to
    the text-only Apertus engine and score text-only answers as if multimodal.
    Apertus runs through the chat wrapper, which splices VQ visual tokens."""

    _chat_add_special_tokens = False

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("apertus_1p5_vllm has no simple-mode path; run without --force_simple so the chat wrapper (VQ token splicing) is used.")
