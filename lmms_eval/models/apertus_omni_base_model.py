from __future__ import annotations

import asyncio
import copy
import json
import os
import uuid
from pathlib import Path
from typing import Any, Sequence

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.api.model import lmms

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None

try:
    from vllm_omni.entrypoints.async_omni import AsyncOmni
except ImportError:
    AsyncOmni = None


def _default_apertus_stage_config_path() -> str | None:
    env_path = os.getenv("APERTUS_OMNI_STAGE_CONFIG")
    if env_path:
        return env_path

    # Expected monorepo layout:
    #   swissai/
    #     lmms-eval/
    #     vllm-omni/
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "vllm-omni" / "vllm_omni" / "model_executor" / "stage_configs" / "apertus.yaml"
    if candidate.exists():
        return str(candidate)
    return None


def _to_pil_image(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, str):
        return Image.open(image_obj).convert("RGB")
    raise TypeError(f"Unsupported image type for Apertus Omni adapter: {type(image_obj)}")


class ApertusOmniBaseModel(lmms):
    is_simple = True

    def __init__(
        self,
        model_descriptor: str,
        stage_configs_path: str | None = None,
        batch_size: int | str = 1,
        trust_remote_code: bool = False,
        log_stats: bool = False,
        stage_init_timeout: int = 300,
        skip_text_only: bool = True,
        skip_multi_image: bool = True,
        vq_hub: str = "BAAI/Emu3.5-VisionTokenizer",
        vision_tokenizer_device: str = "cuda:0",
        vision_tokenizer_dtype: str = "bfloat16",
        emu_min_pixels: int = 256 * 256,
        emu_max_pixels: int = 1400 * 1400,
        vq_trust_remote_code: bool = True,
        image_placeholder: str = "<|image|>",
        tokenizer_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if AsyncOmni is None:
            raise ImportError(
                "vllm_omni is required for Apertus Omni adapters. "
                "Install vllm-omni or set PYTHONPATH to include it."
            )
        if SamplingParams is None:
            raise ImportError("vllm is required for Apertus Omni adapters (SamplingParams import failed).")

        resolved_stage_config = stage_configs_path or _default_apertus_stage_config_path()
        if resolved_stage_config is None:
            raise ValueError(
                "Could not resolve Apertus stage config. Provide model_args "
                "'stage_configs_path=/path/to/vllm_omni/model_executor/stage_configs/apertus.yaml' "
                "or set APERTUS_OMNI_STAGE_CONFIG."
            )
        if not Path(resolved_stage_config).exists():
            raise FileNotFoundError(f"Apertus stage config not found: {resolved_stage_config}")

        # Keep parity with VLLM adapter behavior for model-specific kwargs.
        for key, value in kwargs.items():
            if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                try:
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    eval_logger.warning(f"Failed to parse JSON-like string for argument '{key}': {value}")

        # Ensure Omni engine tokenization uses the intended tokenizer.
        # `tokenizer_path` is the public lmms-eval argument, while vLLM expects `tokenizer`.
        if tokenizer_path is None:
            tokenizer_path = kwargs.pop("tokenizer_path", None)
        else:
            # Avoid passing an unused extra kwarg into Omni.
            kwargs.pop("tokenizer_path", None)
        if tokenizer_path is not None:
            kwargs.setdefault("tokenizer", tokenizer_path)

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        # Evaluator gathers per-rank metadata tensors on lm.device in distributed mode.
        self.device = self.accelerator.device

        self.model_descriptor = model_descriptor
        self.stage_configs_path = resolved_stage_config
        self.batch_size_per_gpu = int(batch_size)
        self.skip_text_only = skip_text_only
        self.skip_multi_image = skip_multi_image
        self.vq_hub = vq_hub
        self.vision_tokenizer_device = vision_tokenizer_device
        self.vision_tokenizer_dtype = vision_tokenizer_dtype
        self.emu_min_pixels = int(emu_min_pixels)
        self.emu_max_pixels = int(emu_max_pixels)
        self.vq_trust_remote_code = bool(vq_trust_remote_code)
        self.image_placeholder = image_placeholder
        self.tokenizer_path = str(kwargs.get("tokenizer", model_descriptor))
        eval_logger.info(f"ApertusOmni tokenizer path: {self.tokenizer_path}")

        self._async_loop = asyncio.new_event_loop()
        previous_loop = self._swap_event_loop(self._async_loop)
        try:
            self.client = AsyncOmni(
                model=model_descriptor,
                stage_configs_path=resolved_stage_config,
                trust_remote_code=trust_remote_code,
                log_stats=log_stats,
                stage_init_timeout=stage_init_timeout,
                **kwargs,
            )
        finally:
            self._swap_event_loop(previous_loop)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @staticmethod
    def _invoke_extractor(extractor: Any, sample: Any) -> Any:
        if callable(extractor):
            return extractor(sample)
        if (
            isinstance(extractor, Sequence)
            and not isinstance(extractor, (str, bytes))
            and extractor
            and callable(extractor[0])
        ):
            return extractor[0](sample)
        raise TypeError(f"Unsupported extractor type: {type(extractor)}")

    @staticmethod
    def _flatten_once(data: Any) -> list[Any]:
        if data is None:
            return []
        if isinstance(data, (list, tuple)):
            flattened: list[Any] = []
            for item in data:
                if isinstance(item, (list, tuple)):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            return flattened
        return [data]

    def _normalize_images(self, data: Any) -> list[Image.Image]:
        return [_to_pil_image(item) for item in self._flatten_once(data)]

    def _build_mm_processor_kwargs(self) -> dict[str, Any]:
        return {
            "apertus_vq_hub": self.vq_hub,
            "apertus_vision_tokenizer_device": self.vision_tokenizer_device,
            "apertus_vision_tokenizer_dtype": self.vision_tokenizer_dtype,
            "apertus_min_pixels": self.emu_min_pixels,
            "apertus_max_pixels": self.emu_max_pixels,
            "apertus_vq_trust_remote_code": self.vq_trust_remote_code,
            "apertus_image_placeholder": self.image_placeholder,
        }

    def _build_prompt_dict(self, prompt: str, images: list[Image.Image]) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": prompt}
        if images:
            payload["multi_modal_data"] = {"image": images}
            payload["mm_processor_kwargs"] = self._build_mm_processor_kwargs()
        return payload

    @staticmethod
    def _normalize_gen_kwargs(gen_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
        gen = dict(gen_kwargs or {})
        gen.setdefault("max_new_tokens", 1024)
        gen.setdefault("temperature", 0)
        gen.setdefault("top_p", 0.95)
        return gen

    def _build_sampling_params(self, gen_kwargs: dict[str, Any] | None = None) -> Any:
        gen = self._normalize_gen_kwargs(gen_kwargs)
        max_tokens = int(gen.get("max_new_tokens", gen.get("max_tokens", 4096)))
        temperature = float(gen.get("temperature", 0.0))
        if gen.get("do_sample") is False:
            temperature = 0.0

        top_p = float(gen.get("top_p", 1.0))
        top_k_raw = gen.get("top_k", -1)
        top_k = int(top_k_raw) if top_k_raw is not None else -1
        repetition_penalty = float(gen.get("repetition_penalty", 1.0))
        seed = gen.get("seed")

        stop = gen.get("until")
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, list):
            stop = None

        params: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        if stop:
            params["stop"] = stop
        if seed is not None:
            params["seed"] = int(seed)

        return SamplingParams(**params)

    @staticmethod
    def _extract_text_from_omni_output(output: Any) -> str:
        request_output = getattr(output, "request_output", output)
        if isinstance(request_output, list):
            if not request_output:
                return ""
            request_output = request_output[0]

        completions = getattr(request_output, "outputs", None)
        if completions:
            first = completions[0]
            if isinstance(first, dict):
                text = first.get("text")
            else:
                text = getattr(first, "text", None)
            if isinstance(text, str):
                return text

        text = getattr(request_output, "text", None)
        return text if isinstance(text, str) else ""

    @staticmethod
    def _swap_event_loop(new_loop: asyncio.AbstractEventLoop | None) -> asyncio.AbstractEventLoop | None:
        try:
            previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            previous_loop = None
        asyncio.set_event_loop(new_loop)
        return previous_loop

    def _run_on_internal_loop(self, awaitable: Any) -> Any:
        if getattr(self, "_async_loop", None) is None or self._async_loop.is_closed():
            raise RuntimeError("Apertus Omni async loop is unavailable.")
        if self._async_loop.is_running():
            raise RuntimeError("Apertus Omni async loop is already running.")
        previous_loop = self._swap_event_loop(self._async_loop)
        try:
            return self._async_loop.run_until_complete(awaitable)
        finally:
            self._swap_event_loop(previous_loop)

    async def _generate_batch_async(self, prompt_dicts: list[dict[str, Any]], sampling_params: Any) -> list[str]:
        async def _run_single(prompt_dict: dict[str, Any], index: int) -> str:
            request_id = f"lmms-apertus-{self.rank}-{index}-{uuid.uuid4().hex}"
            final_output: Any = None
            async for output in self.client.generate(
                prompt=prompt_dict,
                request_id=request_id,
                sampling_params_list=[copy.deepcopy(sampling_params)],
                output_modalities=["text"],
            ):
                final_output = output
            if final_output is None:
                return ""
            return self._extract_text_from_omni_output(final_output)

        tasks = [asyncio.create_task(_run_single(prompt_dict, idx)) for idx, prompt_dict in enumerate(prompt_dicts)]
        return await asyncio.gather(*tasks)

    def _generate_batch(self, prompt_dicts: list[dict[str, Any]], sampling_params: Any) -> list[str]:
        if not prompt_dicts:
            return []
        return self._run_on_internal_loop(self._generate_batch_async(prompt_dicts, sampling_params))

    def close(self) -> None:
        if getattr(self, "client", None) is not None:
            try:
                previous_loop = self._swap_event_loop(self._async_loop)
                try:
                    self.client.shutdown()
                finally:
                    self._swap_event_loop(previous_loop)
            except Exception as e:
                eval_logger.warning(f"Failed to close Omni client cleanly: {e}")
            finally:
                self.client = None
        if getattr(self, "_async_loop", None) is not None and not self._async_loop.is_closed():
            previous_loop = self._swap_event_loop(self._async_loop)
            try:
                pending = [task for task in asyncio.all_tasks(loop=self._async_loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    self._async_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:
                eval_logger.warning(f"Failed to drain Apertus async loop cleanly: {e}")
            finally:
                self._swap_event_loop(previous_loop)
                self._async_loop.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def loglikelihood(self, requests):
        raise NotImplementedError("Loglikelihood is not implemented for Apertus Omni adapters.")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Multi-round generation is not implemented for Apertus Omni adapters.")
