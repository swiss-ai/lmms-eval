"""Task-context contracts for the simple vLLM generation path.

These tests load the production wrapper modules while replacing only optional
runtime dependencies and the inference client.  This keeps prompt construction,
task dispatch, media encoding, and Molmo prefix selection on their real paths.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch


class _FakeLMMS:
    @property
    def rank(self):
        return self._rank


class _FakeProgress:
    def update(self, _count):
        pass

    def close(self):
        pass


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _CaptureClient:
    def __init__(self):
        self.messages = []

    def chat(self, *, messages, **_kwargs):
        self.messages.extend(messages)
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=f"answer-{idx}")]) for idx, _ in enumerate(messages)]


def _module(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _load_module(name: str, relative_path: str) -> types.ModuleType:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(name, repo_root / relative_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_IMPORT_STUBS = {
    "lmms_eval.api.instance": _module("lmms_eval.api.instance", Instance=object),
    "lmms_eval.api.model": _module("lmms_eval.api.model", lmms=_FakeLMMS),
    "lmms_eval.api.registry": _module("lmms_eval.api.registry", register_model=lambda _name: lambda cls: cls),
    "lmms_eval.imports": _module(
        "lmms_eval.imports",
        optional_import=lambda _module_name, attribute=None: ((_FakeSamplingParams, True) if attribute == "SamplingParams" else (None, False)),
    ),
    "lmms_eval.models.model_utils.load_video": _module("lmms_eval.models.model_utils.load_video", read_video=lambda *_args, **_kwargs: []),
    "lmms_eval.models.model_utils.media_encoder": _module("lmms_eval.models.model_utils.media_encoder", encode_image_to_base64=lambda *_args, **_kwargs: "encoded"),
    "lmms_eval.models.model_utils.progress": _module("lmms_eval.models.model_utils.progress", make_progress=lambda **_kwargs: _FakeProgress()),
    "loguru": _module("loguru", logger=types.SimpleNamespace(warning=lambda *_args, **_kwargs: None)),
}

_LOADED_NAMES = (
    "lmms_eval.models.simple.vllm",
    "lmms_eval.models.simple.molmo_vllm",
)
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in (*_IMPORT_STUBS, *_LOADED_NAMES)}
try:
    sys.modules.update(_IMPORT_STUBS)
    _simple_vllm = _load_module("lmms_eval.models.simple.vllm", "lmms_eval/models/simple/vllm.py")
    _molmo_vllm = _load_module("lmms_eval.models.simple.molmo_vllm", "lmms_eval/models/simple/molmo_vllm.py")
finally:
    for name, module in _ORIGINAL_MODULES.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _model(model_class, task: str, *, image_first: bool = False):
    model = model_class.__new__(model_class)
    model._rank = 0
    model._world_size = 1
    model._tp_world_size = 1
    model._tp_group_cpu = None
    model._tp_rank_in_group = 0
    model._watchdog_path = ""
    model.batch_size_per_gpu = 1
    model.max_new_tokens = 16
    model.enable_thinking = None
    model.chat_template = None
    model.image_first = image_first
    model.task_dict = {task: {"test": [{"question": "Which option is correct?"}]}}
    model.client = _CaptureClient()
    model.encode_image = lambda image: f"encoded-{image}"
    model._logged_prefixes = set()
    return model


def _request(task: str, images=("second.png", "first.png")):
    return types.SimpleNamespace(
        arguments=(
            "Choose A or B.",
            {},
            lambda _doc: list(images),
            0,
            task,
            "test",
        )
    )


def _content(model, task: str):
    logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    with patch.dict(sys.modules, {"lmms_eval.utils": _module("lmms_eval.utils", eval_logger=logger)}):
        outputs = model.generate_until([_request(task)])
    assert outputs == ["answer-0"]
    return model.client.messages[0][0]["content"]


class TestSimpleVLLMTaskContext(unittest.TestCase):
    def test_molmo_preserves_mcq_task_during_image_encoding(self):
        content = _content(_model(_molmo_vllm.MolmoVLLM, "mmvp"), "mmvp")

        self.assertEqual(content[0], {"type": "text", "text": "a_okvqa_mc: Choose A or B."})

    def test_molmo_preserves_specific_task_prefix_during_image_encoding(self):
        content = _content(_model(_molmo_vllm.MolmoVLLM, "chartqa"), "chartqa")

        self.assertEqual(content[0], {"type": "text", "text": "chart_qa: Choose A or B."})

    def test_generic_wrapper_keeps_context_and_image_order(self):
        content = _content(_model(_simple_vllm.VLLM, "mmvp", image_first=True), "mmvp")

        self.assertEqual(
            content,
            [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,encoded-second.png"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,encoded-first.png"}},
                {"type": "text", "text": "Choose A or B."},
            ],
        )


if __name__ == "__main__":
    unittest.main()
