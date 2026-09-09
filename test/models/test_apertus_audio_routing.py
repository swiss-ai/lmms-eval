"""CPU regression for the real Apertus audio/image dispatch and constructors.

AST-load production methods to avoid optional GPU imports. Only the engine,
tokenizer, image encoder, accelerator and protocol validation are substituted.
"""
import ast
import base64
import io
from dataclasses import dataclass, replace
from itertools import groupby
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Optional
import unittest
from unittest.mock import patch

import test_apertus_thinking_constructor as constructor

MODELS = Path(__file__).resolve().parents[2] / "lmms_eval" / "models"


def load_class(scope, relative, source_name, name, base, methods):
    path = MODELS / relative
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == source_name)
    body = [node for node in cls.body if
            (isinstance(node, ast.FunctionDef) and node.name in methods)
            or isinstance(node, (ast.Assign, ast.AnnAssign))]
    selected = ast.ClassDef(name=name, bases=[ast.Name(id=base, ctx=ast.Load())],
                            keywords=[], body=body, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)


@dataclass
class TokenCounts:
    output_tokens: int


@dataclass
class GenerationResult:
    text: str
    token_counts: TokenCounts


class ProtocolValidation:
    calls = 0

    def __init__(self, messages):
        type(self).calls += 1
        self.messages = [SimpleNamespace(role=m["role"], content=[SimpleNamespace(**c) for c in m["content"]])
                         for m in messages]


class Image:
    pass


class Waveform:
    ndim = 1

    def __init__(self):
        self.writes = []


class AudioDecoder:
    def __init__(self):
        self.calls = 0
        self.samples = Waveform()

    def get_all_samples(self):
        self.calls += 1
        return SimpleNamespace(samples=self.samples, sample_rate=16000)


def write_audio(buffer, samples, sampling_rate, format):
    samples.writes.append((sampling_rate, format))
    buffer.write(b"waveform")


class Tokenizer:
    def __init__(self):
        self.messages = []
        self.tokenized = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append((messages, kwargs))
        return next(c["text"] for c in messages[0]["content"]["parts"] if c["type"] == "text")

    def __call__(self, text, **kwargs):
        self.tokenized.append((text, kwargs))
        return {"input_ids": [text]}


class Engine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        ids = [next(c["text"] for c in m[0]["content"] if c["type"] == "text") for m in kwargs["messages"]]
        return self.responses(ids)

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return self.responses([p["prompt_token_ids"][0].removesuffix("+image") for p in kwargs["prompts"]])

    @staticmethod
    def responses(ids):
        return [SimpleNamespace(outputs=[SimpleNamespace(text=f"<|inner_prefix|>reason<|inner_suffix|>{name}", token_ids=[1, 2])])
                for name in ids]


def model_class():
    scope = {
        "Any": Any, "Optional": Optional, "os": os, "json": json, "re": re,
        "replace": replace, "groupby": groupby, "base64": base64, "io": io,
        "np": SimpleNamespace(asarray=lambda audio, dtype: audio, float32="float32"),
        "sf": SimpleNamespace(write=write_audio),
        "_NoEngineBase": constructor._NoEngineBase, "ProtocolValidation": ProtocolValidation,
        "Accelerator": lambda: SimpleNamespace(num_processes=1, process_index=0, device="cpu"),
        "LLM": Engine, "eval_logger": SimpleNamespace(info=lambda *a: None),
        "DEFAULT_TOKENIZER_PATH": "fixture/tokenizer", "PILImage": SimpleNamespace(Image=Image),
        "splice_frames": lambda prompt, images, tokenizer: prompt + "+image",
        "tqdm": lambda **k: SimpleNamespace(update=lambda *a: None, close=lambda: None),
        "GenerationResult": GenerationResult, "TokenCounts": TokenCounts,
        "_INNER_PREFIX": "<|inner_prefix|>", "_INNER_SUFFIX": "<|inner_suffix|>",
        "_SPECIAL_TOKEN_RE": re.compile(r"<\|[^|]+\|>"),
    }
    load_class(scope, "../protocol.py", "ChatMessages", "ChatMessages", "ProtocolValidation",
               {"_encode_settings", "_audio_to_data_url", "_audio_object_to_data_url", "_audio_mime_type_from_path",
                "_coerce_audio_array", "_audio_array_to_data_url", "to_openai_messages"})
    load_class(scope, "simple/vllm.py", "VLLM", "VLLMSimple", "_NoEngineBase",
               {"__init__", "_is_qwen_vl_model", "_select_max_new_tokens", "_normalize_top_p_for_vllm",
                "_build_sampling_params_dict", "_chat_template_kwargs", "_chat_tokenization_kwargs", "rank"})
    load_class(scope, "chat/vllm.py", "VLLM", "VLLM", "VLLMSimple", {"__init__", "make_one_request"})
    load_class(scope, "chat/apertus_1p5_vllm.py", "Apertus1p5VLLM", "Apertus1p5VLLM", "VLLM",
               {"__init__", "_render_request", "_build_sampling_params_dict", "_run_generate", "generate_until", "_strip_thinking"})
    return scope["Apertus1p5VLLM"], scope["ChatMessages"]


class TestApertusAudioRouting(unittest.TestCase):
    def test_mixed_audio_image_text_keep_order_and_thinking_without_reloading_documents(self):
        cls, protocol = model_class()
        for thinking in (True, False):
            with self.subTest(thinking=thinking):
                tokenizer = Tokenizer()
                transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer))
                vllm = SimpleNamespace(SamplingParams=lambda **k: k)
                with patch.dict("sys.modules", {"transformers": transformers, "vllm": vllm}):
                    model = cls(model="fixture", tokenizer="fixture/tokenizer", enable_thinking=thinking,
                                batch_size=8, tensor_parallel_size=4, max_num_seqs=8, max_new_tokens=1)
                    model.rank = 0
                    model._supports_chat_template_kwargs = model._supports_chat_tokenization_kwargs = True
                    model._run_tp_synced = lambda items, run: run(items)
                    accesses = []
                    class Documents:
                        def __getitem__(self, index):
                            accesses.append(index)
                            return index
                    model.task_dict = {"task": {"test": Documents()}}
                    callbacks = []
                    decoder = AudioDecoder()
                    requests = []
                    names = ["audio1", "audio2", "image", "text", "audio3"]
                    for index, name in enumerate(names):
                        def messages(doc, name=name):
                            callbacks.append(doc)
                            content = [{"type": "text", "text": name}]
                            if name.startswith("audio"):
                                content.append({"type": "audio", "url": decoder if name == "audio2" else "data:audio/wav;base64,fixture"})
                            elif name == "image":
                                content.append({"type": "image", "url": Image()})
                            return [{"role": "user", "content": content}]
                        requests.append(SimpleNamespace(arguments=("context", messages,
                            {"max_new_tokens": 30 + index, "temperature": 0, "top_p": 1}, index, "task", "test")))
                    protocol.calls = 0
                    results = model.generate_until(requests)
                self.assertEqual(accesses, list(range(5)))
                self.assertEqual(callbacks, list(range(5)))
                self.assertEqual(protocol.calls, 5)
                self.assertEqual(decoder.calls, 1)
                self.assertEqual(decoder.samples.writes, [(16000, "WAV")])
                expected = names if thinking else [f"<|inner_prefix|>reason<|inner_suffix|>{n}" for n in names]
                self.assertEqual([r.text for r in results], expected)
                self.assertEqual([r.token_counts.output_tokens for r in results], [2] * 5)
                self.assertTrue(model.client.kwargs["skip_mm_profiling"])
                self.assertEqual(model.client.kwargs["max_num_seqs"], 8)
                self.assertEqual(model.client.kwargs["tensor_parallel_size"], 4)
                self.assertEqual([kind for kind, _ in model.client.calls], ["chat", "generate", "chat"])
                for kind, kwargs in model.client.calls:
                    self.assertTrue(all(p["skip_special_tokens"] is (not thinking) for p in kwargs["sampling_params"]))
                    if kind == "chat":
                        self.assertEqual(kwargs["chat_template_kwargs"], {"enable_thinking": thinking})
                        self.assertEqual(kwargs["tokenization_kwargs"], {"add_special_tokens": False})
                        for messages in kwargs["messages"]:
                            name = messages[0]["content"][0]["text"]
                            encoded = "d2F2ZWZvcm0=" if name == "audio2" else "fixture"
                            self.assertEqual(messages[0]["content"][1],
                                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{encoded}"}})
                self.assertEqual([p["max_tokens"] for _, kwargs in model.client.calls for p in kwargs["sampling_params"]], [30, 31, 32, 33, 34])
                self.assertEqual([text for text, _ in tokenizer.tokenized], ["image+image", "text"])
                self.assertTrue(all(not kwargs["add_special_tokens"] for _, kwargs in tokenizer.tokenized))


if __name__ == "__main__":
    unittest.main()
