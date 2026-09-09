"""CPU-only constructor/render regression, runnable directly with Python.

Execute the real three constructor bodies and Apertus rendering methods;
replace only engine, accelerator, tokenizer and protocol adapters. AST loading
avoids importing optional GPU packages before those boundaries can be replaced.
"""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
import unittest
from unittest.mock import patch


MODELS = Path(__file__).resolve().parents[2] / "lmms_eval" / "models"


class _NoEngineBase:
    def __init__(self):
        self.task_dict = {}

    def _client_chat_supports_tokenization_kwargs(self):
        return False

    def _client_chat_supports_chat_template_kwargs(self):
        return False

    def _setup_tp_group_for_request_sync(self):
        pass


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "rendered prompt"

    def __call__(self, prompt, **kwargs):
        self.tokenizer_kwargs = kwargs
        return {"input_ids": [11, 12]}


def _load_class(scope, relative, source_name, name, base, methods):
    path = MODELS / relative
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == source_name)
    selected = ast.ClassDef(name=name, bases=[ast.Name(id=base, ctx=ast.Load())], keywords=[],
                            body=[node for node in cls.body if isinstance(node, ast.FunctionDef)
                                  and node.name in methods], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)


def _model_class():
    scope = {
        "Any": Any, "Optional": Optional, "os": os, "json": json,
        "_NoEngineBase": _NoEngineBase,
        "Accelerator": lambda: SimpleNamespace(num_processes=1, process_index=0, device="cpu"),
        "LLM": lambda **kwargs: SimpleNamespace(),
        "eval_logger": SimpleNamespace(info=lambda *args: None),
        "DEFAULT_TOKENIZER_PATH": "fixture/tokenizer",
        "ChatMessages": lambda messages: SimpleNamespace(messages=messages),
    }
    _load_class(scope, "simple/vllm.py", "VLLM", "VLLMSimple", "_NoEngineBase",
                {"__init__", "_is_qwen_vl_model", "_select_max_new_tokens",
                 "_normalize_top_p_for_vllm", "_build_sampling_params_dict"})
    _load_class(scope, "chat/vllm.py", "VLLM", "VLLM", "VLLMSimple", {"__init__"})
    _load_class(scope, "chat/apertus_1p5_vllm.py", "Apertus1p5VLLM", "Apertus1p5VLLM", "VLLM",
                {"__init__", "_render_request", "_build_sampling_params_dict"})
    return scope["Apertus1p5VLLM"]


class TestApertusThinkingConstructor(unittest.TestCase):
    def test_thinking_and_direct_modes_reach_template_and_sampling_after_constructor(self):
        cls = _model_class()
        for supplied, expected in ((True, True), (False, False), (None, False)):
            with self.subTest(supplied=supplied):
                tokenizer = _Tokenizer()
                transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer))
                kwargs = {} if supplied is None else {"enable_thinking": supplied}
                with patch.dict("sys.modules", {"transformers": transformers}):
                    model = cls(model="fixture", tokenizer="fixture/tokenizer", **kwargs)
                model.task_dict = {"task": {"test": [{}]}}
                messages = [SimpleNamespace(role="user", content=[SimpleNamespace(type="text", text="question")])]
                request = SimpleNamespace(arguments=("context", lambda doc: messages, {}, 0, "task", "test"))
                prompt, sampling = model._render_request(request)
                self.assertIs(model.enable_thinking, expected)
                self.assertIs(tokenizer.template_kwargs["enable_thinking"], expected)
                self.assertIs(sampling["skip_special_tokens"], not expected)
                self.assertEqual(tokenizer.messages, [{"role": "user", "content": {"parts": [{"type": "text", "text": "question"}]}}])
                self.assertEqual(prompt, {"prompt_token_ids": [11, 12]})
                self.assertFalse(tokenizer.tokenizer_kwargs["add_special_tokens"])


if __name__ == "__main__":
    unittest.main()
