"""
Llama base model with EMU3.5 vision encoder (simple/non-chat version).

For evaluating base models without instruction tuning.
Uses direct text prompts instead of chat templates.
"""

from typing import Optional, Union

import torch
from loguru import logger as eval_logger
from transformers import AutoTokenizer, LlamaForCausalLM

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.emu_simple_model import EMU3p5SimpleModel


@register_model("llama_emu3p5_simple")
class LlamaEmu3p5Simple(EMU3p5SimpleModel):
    """
    Llama base model with EMU3.5 vision encoder (simple/non-chat version).

    For evaluating base models without instruction tuning.
    Uses direct text prompts instead of chat templates.
    """

    def __init__(
        self,
        model_descriptor: str,
        tokenizer_path: str,
        vq_hub: str = "BAAI/Emu3.5-VisionTokenizer",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        vision_tokenizer_dtype: Optional[torch.dtype] = None,
        use_cache: bool = True,
        emu_min_pixels: int = 256 * 256,
        emu_max_pixels: int = 1400 * 1400,
        skip_text_only: bool = False,
        skip_multi_image: bool = True,
        debug_samples: bool = False,
        num_debug_samples: int = 5,
        prompt_override: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Wire Llama causal-LM weights to the EMU3.5 IBQ vision encoder.

        Thin forwarding constructor; every argument is passed through
        unchanged to :class:`EMU3p5SimpleModel` (see its type-hinted
        signature for accepted parameters).
        """
        super().__init__(
            model_descriptor=model_descriptor,
            tokenizer_path=tokenizer_path,
            vq_hub=vq_hub,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            vision_tokenizer_dtype=vision_tokenizer_dtype,
            use_cache=use_cache,
            emu_min_pixels=emu_min_pixels,
            emu_max_pixels=emu_max_pixels,
            skip_text_only=skip_text_only,
            skip_multi_image=skip_multi_image,
            debug_samples=debug_samples,
            num_debug_samples=num_debug_samples,
            prompt_override=prompt_override,
            **kwargs,
        )

    def _load_tokenizer(self, tokenizer_path: str, **kwargs) -> AutoTokenizer:
        """
        Left padding is required so generated tokens stay contiguous
        across a batch; a missing pad token falls back to eos so
        batched generation does not error.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
        if tokenizer.pad_token is None:
            eval_logger.warning("No pad_token found, setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_llm(self, model_path: str, **kwargs) -> LlamaForCausalLM:
        """Force eval mode at load to disable dropout for deterministic eval."""
        return LlamaForCausalLM.from_pretrained(model_path, **kwargs).eval()

    @property
    def image_placeholder(self) -> str:
        """Sentinel the EMU3.5 processor replaces with encoded vision tokens."""
        return "<|image|>"
