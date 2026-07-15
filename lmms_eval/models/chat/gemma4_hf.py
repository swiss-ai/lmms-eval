from typing import List

import torch
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.huggingface import Huggingface
from lmms_eval.protocol import ChatMessages


@register_model("gemma4_hf")
class Gemma4HF(Huggingface):
    """Gemma 4 transformers wrapper.

    The generic backend's two-step input build (template text, then a separate
    processor call) routes images through the wrong branch of gemma4's dual
    understanding/generation pipeline; inputs must come from a single
    apply_chat_template(tokenize=True) call, mirroring the VLMEvalKit wrapper.
    """

    def __init__(self, *args, **kwargs):
        # The auto class resolves the generative entrypoint for gemma4_unified;
        # the concrete classes load but break at the patch projection.
        kwargs.setdefault("model_class", "AutoModelForMultimodalLM")
        super().__init__(*args, **kwargs)

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []

        def _collate(x):
            return x[2], x[2]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            batch = []
            for d2m, ids, tsk, spl in zip(doc_to_messages, doc_id, task, split):
                messages = d2m(self.task_dict[tsk][spl][ids])
                if self.system_prompt:
                    messages = self._apply_system_prompt(messages, self.system_prompt)
                batch.append(ChatMessages(messages=messages).to_hf_messages())

            # apply_chat_template batches conversations itself (it collects the
            # per-conversation images and hands the processor one batched list);
            # left padding keeps the generated span at a common offset.
            self.processor.tokenizer.padding_side = "left"
            inputs = self.processor.apply_chat_template(
                batch if len(batch) > 1 else batch[0],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]

            gen_kwargs = all_gen_kwargs[0]
            max_new_tokens = gen_kwargs.get("max_new_tokens", 4096)
            temperature = gen_kwargs.get("temperature", 0.0)
            do_sample = bool(temperature and temperature > 0)
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=gen_kwargs.get("top_p") if do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            for i in range(len(batch)):
                new_tokens = generation[i][input_len:]
                decoded = self.processor.decode(new_tokens, skip_special_tokens=False)
                if hasattr(self.processor, "parse_response"):
                    parsed = self.processor.parse_response(decoded)
                    if isinstance(parsed, dict):
                        decoded = parsed.get("answer") or parsed.get("response") or parsed.get("content") or str(parsed)
                    elif isinstance(parsed, tuple):
                        decoded = parsed[-1]
                    else:
                        decoded = parsed
                ans = str(decoded).strip()
                self.cache_hook.add_partial("generate_until", (ctx[i], gen_kwargs), ans)
                res.append(GenerationResult(text=ans, token_counts=TokenCounts(output_tokens=len(new_tokens))))
                pbar.update(1)
        pbar.close()
        return re_ords.get_original(res)
