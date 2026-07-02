"""Prompt-integrity and scoring regression tests for the remote-sensing tasks.

Guards the failure classes found in the first Apertus runs:
- rsrcc leaked '**Answer:** X' into every prompt because the dataset text
  field escapes newlines as literal backslash-n sequences
- vrsbench_ref/bigearth_bbox scored raw model coordinates against a fixed
  ground-truth convention, inverting checkpoint rankings
- geobench crashed on docs whose imagery is not redistributed (xBD)

Task utils modules are loaded by file path, matching how lmms-eval's
!function resolver loads them in production.
"""

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest

TASKS_DIR = Path(__file__).resolve().parents[2] / "lmms_eval" / "tasks"

# bigearth_txt imports torch and safetensors at module level but only uses
# them in the LMDB image-loading path, which these tests never touch. Stub
# them when absent so the suite runs without the full GPU stack.
for _heavy in ("torch", "safetensors", "safetensors.numpy"):
    _root = _heavy.split(".")[0]
    if _root not in sys.modules and importlib.util.find_spec(_root) is not None:
        continue
    if _heavy in sys.modules:
        continue
    _mod = types.ModuleType(_heavy)
    _mod.__spec__ = importlib.machinery.ModuleSpec(_heavy, loader=None, is_package="." not in _heavy)
    if "." not in _heavy:
        _mod.__path__ = []
    if _heavy == "safetensors.numpy":
        _mod.load = None
    sys.modules[_heavy] = _mod


def load_task_utils(task):
    spec = importlib.util.spec_from_file_location(f"{task}_utils_under_test", TASKS_DIR / task / "utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rsrcc = load_task_utils("rsrcc")
vrsbench = load_task_utils("vrsbench")
geobench = load_task_utils("geobench")
bigearth = load_task_utils("bigearth_txt")
frieda = load_task_utils("frieda")

ESCAPED_MCQ = r"**Question:** What changed?\n\n**A)** Removed.\n**B)** Built.\n**C)** Widened.\n**D)** Replaced.\n\n**Answer:** B"
ESCAPED_YESNO = r"**Question:** Did the road change?\n\n**Answer:** Yes"


class TestRsrccParser:
    def test_escaped_mcq(self):
        q, a, is_mcq = rsrcc._parse_text(ESCAPED_MCQ)
        assert a == "B"
        assert is_mcq
        assert "**Answer:" not in q
        assert "\\n" not in q
        assert "**B)**" in q

    def test_escaped_yesno(self):
        q, a, is_mcq = rsrcc._parse_text(ESCAPED_YESNO)
        assert a == "Yes"
        assert not is_mcq
        assert "**Answer:" not in q

    def test_real_newlines(self):
        q, a, _ = rsrcc._parse_text("**Question:** Real newlines?\n\n**Answer:** No")
        assert a == "No"
        assert "**Answer:" not in q

    def test_doc_to_text_has_no_answer_marker(self):
        prompt = rsrcc.rsrcc_doc_to_text({"text": ESCAPED_MCQ})
        assert "**Answer:" not in prompt
        assert "**Question:**" in prompt

    def test_doc_to_text_guard_raises_on_leak(self, monkeypatch):
        monkeypatch.setattr(rsrcc, "_parse_text", lambda text: ("leaky **Answer:** B", "B", True))
        with pytest.raises(ValueError, match="answer marker"):
            rsrcc.rsrcc_doc_to_text({"text": ESCAPED_MCQ})

    def test_doc_to_target(self):
        assert rsrcc.rsrcc_doc_to_target({"text": ESCAPED_MCQ}) == "B"
        assert rsrcc.rsrcc_doc_to_target({"text": ESCAPED_YESNO}) == "Yes"


GT = "SENTINEL_GROUND_TRUTH_ZQX"

LEAKAGE_CASES = [
    ("rsrcc", rsrcc.rsrcc_doc_to_text, {"text": ESCAPED_MCQ}, "**Answer:"),
    ("vrsbench_vqa", vrsbench.vrsbench_vqa_doc_to_text, {"question": "What color is the roof?", "ground_truth": GT}, GT),
    ("vrsbench_cap", vrsbench.vrsbench_cap_doc_to_text, {"ground_truth": GT}, GT),
    ("vrsbench_ref", vrsbench.vrsbench_ref_doc_to_text, {"question": "the red car", "ground_truth": GT}, GT),
    ("geobench_single", geobench.geobench_single_doc_to_text, {"prompts": ["Which class?"], "options": "A) x\nB) y", "ground_truth_option": GT}, GT),
    ("geobench_temporal", geobench.geobench_temporal_doc_to_text, {"prompts": ["What changed?"], "options": "A) x\nB) y", "ground_truth_option": GT}, GT),
    ("geobench_cap", geobench.geobench_cap_doc_to_text, {"prompts": ["Describe the image."], "ground_truth": GT}, GT),
    ("geobench_ref", geobench.geobench_ref_doc_to_text, {"prompts": ["the ship"], "ground_truth": GT}, GT),
    ("bigearth_binary", bigearth.bigearth_binary_doc_to_text, {"input": "Is there water?", "output": GT}, GT),
    ("bigearth_mcq", bigearth.bigearth_mcq_doc_to_text, {"input": "Which land cover?", "output": GT}, GT),
    ("bigearth_bbox", bigearth.bigearth_bbox_doc_to_text, {"input": "Box the lake.", "output": GT}, GT),
    ("bigearth_cap", bigearth.bigearth_cap_doc_to_text, {"input": "Describe the patch.", "output": GT}, GT),
    ("frieda", frieda.frieda_doc_to_text, {"question_text": "Where is the station?", "expected_answer": GT, "image_urls": ["a.png", "b.png"]}, GT),
]


@pytest.mark.parametrize("name,fn,doc,needle", LEAKAGE_CASES, ids=[c[0] for c in LEAKAGE_CASES])
def test_prompt_does_not_leak_target(name, fn, doc, needle):
    assert needle not in fn(doc)


class TestBboxScaleNormalization:
    def test_vrsbench_norm_maps_common_conventions_to_0_100(self):
        for box in ([0.25, 0.40, 0.33, 0.60], [25, 40, 33, 60], [250, 400, 330, 600]):
            assert vrsbench._norm_bbox_to_100(box) == pytest.approx([25, 40, 33, 60], rel=1e-6)

    def test_vrsbench_ref_normalized_and_strict_metrics(self):
        doc = {"ground_truth": "{<25><40><33><60>}"}
        res = vrsbench.vrsbench_ref_process_results(doc, ["[0.25, 0.40, 0.33, 0.60]"])
        assert res["ref_acc50"] == 1
        assert res["ref_acc50_strict"] == 0
        res = vrsbench.vrsbench_ref_process_results(doc, ["{<25><40><33><60>}"])
        assert res["ref_acc50"] == 1
        assert res["ref_acc50_strict"] == 1
        res = vrsbench.vrsbench_ref_process_results(doc, ["no box here"])
        assert res["ref_acc50"] == 0
        assert res["ref_acc50_strict"] == 0

    def test_bigearth_norm_maps_common_conventions_to_0_1(self):
        for box in ([0.25, 0.40, 0.33, 0.60], [25, 40, 33, 60], [250, 400, 330, 600]):
            assert bigearth._norm_bbox_to_01(box) == pytest.approx([0.25, 0.40, 0.33, 0.60], rel=1e-6)

    def test_bigearth_bbox_scores_cross_convention(self):
        doc = {"output": "[0.25 0.40, 0.33 0.60]"}
        assert bigearth.bigearth_bbox_process_results(doc, ["[25, 40, 33, 60]"])["bbox_acc50"] == 1
        assert bigearth.bigearth_bbox_process_results(doc, ["[0.7, 0.7, 0.9, 0.9]"])["bbox_acc50"] == 0


class _StubDataset:
    def __init__(self, docs):
        self._docs = docs

    def filter(self, fn):
        return _StubDataset([d for d in self._docs if fn(d)])

    def __len__(self):
        return len(self._docs)


class TestGeobenchMissingImageFilters:
    def test_single_filter_keeps_embedded_and_staged_drops_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEOBENCH_DIR", str(tmp_path))
        images = tmp_path / "Single" / "images"
        images.mkdir(parents=True)
        (images / "ok.png").write_bytes(b"x")
        ds = _StubDataset(
            [
                {"image": {"bytes": b"embedded", "path": None}, "image_path": "Single/images/anything.png"},
                {"image": None, "image_path": "Single/images/ok.png"},
                {"image": None, "image_path": "Single/images/missing.png"},
            ]
        )
        assert len(geobench.geobench_single_filter_docs(ds)) == 2

    def test_temporal_filter_requires_both_images(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEOBENCH_DIR", str(tmp_path))
        images = tmp_path / "Temporal" / "images"
        images.mkdir(parents=True)
        (images / "a.png").write_bytes(b"x")
        (images / "b.png").write_bytes(b"x")
        ds = _StubDataset(
            [
                {"image_path": ["Temporal/images/a.png", "Temporal/images/b.png"]},
                {"image_path": ["Temporal/images/a.png", "Temporal/images/missing.png"]},
            ]
        )
        assert len(geobench.geobench_temporal_filter_docs(ds)) == 1
