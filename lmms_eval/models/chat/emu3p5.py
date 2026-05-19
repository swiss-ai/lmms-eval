"""
EMU3.5 Chat Model using EMU3p5EncoderBaseModel.

EMU3.5 Multimodal Model (34B parameters).
Uses IBQ vision tokenizer for improved image understanding.
https://github.com/baaivision/Emu3.5

Builds the processor's chat template manually and uses
encode_and_inject_vision_tokens, which naturally supports batches that
mix image and text-only samples.

Generation-config precedence (highest -> lowest):
1. Task gen_kwargs (YAML generation_kwargs or model_specific_generation_kwargs)
   for max_new_tokens, temperature, do_sample, top_k, top_p, num_beams.
2. Wrapper-supplied: pad/bos_token_id from tokenizer, use_cache from __init__.
3. In-code fallback: max_new_tokens=1024 when no task value is provided.
4. Everything else (eos_token_id, repetition_penalty, length_penalty, ...)
   defers to model.generation_config shipped with the checkpoint.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.debug_utils import log_debug_sample
from lmms_eval.models.model_utils.emu3p5.download_utils import ensure_local_weights
from lmms_eval.models.model_utils.emu3p5.emu3p5_tokenizer_loader import (
    load_emu3p5_tokenizer,
)
from lmms_eval.protocol import ChatMessages

# isort: off
# Order matters: importing EMU3p5EncoderBaseModel inserts external/Emu3.5/src
# into sys.path, which must happen BEFORE `from emu3p5 import ...` can resolve.
# Keep this block last; do not let isort/black reorder these two imports.
from lmms_eval.models.emu3p5_encoder_base_model import EMU3p5EncoderBaseModel

from emu3p5 import Emu3Config, Emu3ForCausalLM  # noqa: E402

# isort: on

# Path to EMU3.5 tokenizer directory
_current_file = Path(__file__).resolve()
_repo_root = _current_file.parents[3]  # Go up to lmms-eval root
_txt_tok_path = _repo_root / "external" / "Emu3.5" / "src" / "tokenizer_emu3_ibq"


@register_model("emu3p5")
class EMU3_5(EMU3p5EncoderBaseModel):
    """
    EMU3.5 Multimodal Model (34B parameters).

    Inherits infrastructure from EMU3p5EncoderBaseModel.
    """

    is_simple = False  # Chat model

    def __init__(
        self,
        model_descriptor: str = "BAAI/Emu3.5",
        tokenizer_path: Optional[str] = None,
        vq_hub: str = "BAAI/Emu3.5-VisionTokenizer",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: Optional[str] = "flash_attention_2",
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
        **kwargs,
    ):
        # Store settings for use in abstract methods
        self._trust_remote_code = trust_remote_code
        self._pretrained = model_descriptor

        # Call parent constructor with mapped parameters
        # Native EMU3.5 uses 6-digit zero-padded visual token format
        super().__init__(
            model_descriptor=model_descriptor,
            # Native EMU3.5 always uses the submodule tokenizer (custom tiktoken-based,
            # not available on HF), ignore any tokenizer_path passed from eval scripts.
            tokenizer_path=str(_txt_tok_path),
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
            visual_token_format="padded",
            **kwargs,
        )

    def _load_tokenizer(self, tokenizer_path: str, **kwargs):
        """Load EMU3.5 text tokenizer with fallback handling."""
        return load_emu3p5_tokenizer(
            tokenizer_path,
            trust_remote_code=self._trust_remote_code,
            padding_side="left",
        )

    def _load_llm(self, model_path: str, **kwargs) -> Emu3ForCausalLM:
        """Load EMU3.5 causal language model with Emu3Config."""
        # Ensure main model weights are available locally
        model_path = ensure_local_weights(model_path, "BAAI/Emu3.5", accelerator=self.accelerator)

        # Extract kwargs for model loading
        trust_remote_code = kwargs.pop("trust_remote_code", self._trust_remote_code)

        # Load config
        model_config = Emu3Config.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        # Load model
        return Emu3ForCausalLM.from_pretrained(
            model_path,
            config=model_config,
            **kwargs,
        )

    @property
    def image_placeholder(self) -> str:
        """Sentinel string injected into the chat template; replaced with
        wrapped vision tokens by encode_and_inject_vision_tokens."""
        return "<|image|>"

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate responses using the processor's chat template + vision-token injection."""
        res = []

        # Initialize statistics counters
        text_only_count = 0
        multi_image_count = 0
        total_samples = 0
        skipped_text_only = 0
        skipped_multi_image = 0

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # Group requests by their generation_kwargs
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        # iterate over batches
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            # Get chat messages
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            # Convert to ChatMessages protocol
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]

            # Build per-sample chat-templated prompts with image placeholders.
            # encode_and_inject_vision_tokens replaces placeholders with
            # wrapped vision tokens and naturally handles text-only samples
            # (zero placeholders + empty image list).
            sample_data = []
            chunk_results = [None] * len(chat_messages)
            batch_to_chunk_idx = []

            chat_template = self.processor.chat_template
            image_placeholder = self.image_placeholder

            for idx, messages in enumerate(chat_messages):
                total_samples += 1
                visual, video, audio = messages.extract_media()

                # Extract text for this sample
                text = ""
                for message in messages.messages:
                    for content in message.content:
                        if content.type == "text":
                            text += content.text

                # Text-only samples
                if not visual or len(visual) == 0:
                    text_only_count += 1
                    if self.skip_text_only:
                        skipped_text_only += 1
                        chunk_results[idx] = ""
                        self.cache_hook.add_partial(
                            "generate_until",
                            (ctx[idx], all_gen_kwargs[idx]),
                            "",
                        )
                        continue
                    visual = []

                # Multi-image samples
                if len(visual) > 1:
                    multi_image_count += 1
                    if self.skip_multi_image:
                        skipped_multi_image += 1
                        chunk_results[idx] = ""
                        self.cache_hook.add_partial(
                            "generate_until",
                            (ctx[idx], all_gen_kwargs[idx]),
                            "",
                        )
                        continue
                    # If not skipping, keep only the first image
                    visual = visual[:1]

                # Load images and build the per-sample prompt
                images_for_sample = []
                for img in visual:
                    if isinstance(img, str):
                        img = Image.open(img)
                    images_for_sample.append(img)

                images_slot = image_placeholder if images_for_sample else ""
                # No BOS prepend: the Emu3.5 chat_template already starts with
                # <|extra_203|>, which is the tokenizer's bos_token. Prepending
                # again would duplicate BOS.
                prompt = chat_template.format(question=text, images=images_slot)

                batch_to_chunk_idx.append(idx)
                sample_data.append(
                    {
                        "text": prompt,
                        "images": images_for_sample,
                        "context": ctx[idx],
                    }
                )

            # If all samples in batch were skipped, continue to next batch
            if len(sample_data) == 0:
                # Append skipped results in chunk order
                for result in chunk_results:
                    if result is not None:
                        res.append(result)
                        pbar.update(1)
                continue

            gen_kwargs = all_gen_kwargs[0]

            texts = [item["text"] for item in sample_data]
            images_list = [item["images"] for item in sample_data]

            inputs = self.processor.encode_and_inject_vision_tokens(
                texts=texts,
                images=images_list,
                image_placeholder=image_placeholder,
                return_tensors="pt",
                padding="longest",
            )

            # Move to device
            if self.device_map == "auto":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Build generate kwargs: defer to model.generation_config for any
            # field the task didn't explicitly set (including eos_token_id and
            # BAAI's sampling defaults shipped with the checkpoint).
            generate_kwargs = {
                "pad_token_id": self.tokenizer.pad_token_id,
                "bos_token_id": self.tokenizer.bos_token_id,
                "use_cache": self.use_cache,
            }
            for k in (
                "max_new_tokens",
                "temperature",
                "do_sample",
                "top_k",
                "top_p",
                "num_beams",
            ):
                if k in gen_kwargs:
                    generate_kwargs[k] = gen_kwargs[k]
            generate_kwargs.setdefault("max_new_tokens", 1024)

            # Filter inputs to only include keys accepted by model.generate()
            model_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }

            with torch.inference_mode():
                outputs = self.model.generate(**model_inputs, **generate_kwargs)

            # Trim input_ids from outputs
            outputs_trimmed = outputs[:, model_inputs["input_ids"].shape[-1] :]
            answers = self.processor.batch_decode(outputs_trimmed, skip_special_tokens=True)

            # Decode with special tokens for debugging
            if self.debug_samples:
                prompts_with_tokens = self.processor.batch_decode(model_inputs["input_ids"], skip_special_tokens=False)
                answers_with_tokens = self.processor.batch_decode(outputs_trimmed, skip_special_tokens=False)

            for i, (ans, item, text) in enumerate(zip(answers, sample_data, texts)):
                chunk_idx = batch_to_chunk_idx[i]
                chunk_results[chunk_idx] = ans
                self.cache_hook.add_partial("generate_until", (item["context"], gen_kwargs), ans)

                # Debug sample output (only on rank 0 to avoid duplicates)
                if self.debug_samples and self._debug_samples_printed < self.num_debug_samples and self.rank == 0:
                    self._debug_samples_printed += 1
                    log_debug_sample(
                        sample_num=self._debug_samples_printed,
                        total_samples=self.num_debug_samples,
                        prompt_clean=text,
                        prompt_with_tokens=prompts_with_tokens[i],
                        answer_clean=ans,
                        answer_with_tokens=answers_with_tokens[i],
                        attention_mask=model_inputs["attention_mask"][i],
                    )

                eval_logger.debug(f"Question: {text}")
                eval_logger.debug(f"Model Response: {ans}")

            # Append all chunk results in correct order
            for result in chunk_results:
                if result is not None:
                    res.append(result)
                    pbar.update(1)

        # Reorder results back to original unsorted form
        res = re_ords.get_original(res)
        pbar.close()

        # Print statistics at the end (warning mode)
        if self.rank == 0:  # Only print from main process
            eval_logger.warning(f"EMU3.5 Statistics: Found {text_only_count}/{total_samples} " f"text-only samples (no images). " f"Skipped: {skipped_text_only} " f"(skip_text_only={self.skip_text_only})")
            eval_logger.warning(f"EMU3.5 Statistics: Found {multi_image_count}/{total_samples} " f"multi-image samples (>1 image). " f"Skipped: {skipped_multi_image} " f"(skip_multi_image={self.skip_multi_image})")
            if text_only_count == 0 and multi_image_count == 0:
                eval_logger.info(f"EMU3.5 Statistics: All {total_samples} samples had exactly 1 " "image. No text-only or multi-image samples encountered.")

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented for EMU3.5")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not implemented for EMU3.5")
