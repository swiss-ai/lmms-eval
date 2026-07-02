"""Prompt-integrity and scoring regression tests for the remote-sensing tasks.

Guards the failure classes found in the first Apertus runs: rsrcc leaked
'**Answer:** X' into every prompt and mis-extracted 1.3% of gold answers,
bbox tasks scored raw model coordinates against a fixed ground-truth
convention, and geobench crashed on docs whose imagery is not redistributed.
Task utils modules are loaded by file path, matching how lmms-eval's
!function resolver loads them in production.
"""

import importlib.util
from pathlib import Path

import pytest

TASKS_DIR = Path(__file__).resolve().parents[2] / "lmms_eval" / "tasks"


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


class _StubDataset:
    def __init__(self, docs):
        self._docs = docs

    def filter(self, fn):
        return _StubDataset([d for d in self._docs if fn(d)])

    def __len__(self):
        return len(self._docs)


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

    @pytest.mark.parametrize(
        "tail,want",
        [
            ("**Answer:** B", "B"),
            ("**Answer:** B)", "B"),
            ("**Answer: B**", "B"),
            ("**Answer:** Yes.", "Yes"),
            ("**Answer:** B)Item", "B"),
            (r"**Answer:** A)_\n", "A"),
            ("**Answer:** B)</body></html>", "B"),
        ],
        ids=["plain", "paren", "colon-in-bold", "trailing-dot", "concatenated", "escaped-tail", "html-tail"],
    )
    def test_gold_extraction_observed_formats(self, tail, want):
        _, a, _ = rsrcc._parse_text(rf"**Question:** Q?\n\n**A)** x\n**B)** y\n\n{tail}")
        assert a == want

    def test_filter_drops_unparseable_gold(self):
        ds = _StubDataset([{"text": ESCAPED_MCQ}, {"text": "**Question:** Broken doc without a marker"}])
        assert len(rsrcc.rsrcc_filter_docs(ds)) == 1

    def test_doc_to_text_and_target(self):
        prompt = rsrcc.rsrcc_doc_to_text({"text": ESCAPED_MCQ})
        assert "**Answer:" not in prompt
        assert "**Question:**" in prompt
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


class TestBboxScoring:
    CONVENTIONS_100 = [
        ("0-1-floats", "[0.25, 0.40, 0.33, 0.60]"),
        ("0-100-ints", "{<25><40><33><60>}"),
        ("0-1000-ints", "[250, 400, 330, 600]"),
        ("512px", "[128.0, 204.8, 168.96, 307.2]"),
    ]

    @pytest.mark.parametrize("name,pred", CONVENTIONS_100, ids=[c[0] for c in CONVENTIONS_100])
    def test_vrsbench_ref_scores_every_convention(self, name, pred):
        doc = {"ground_truth": "{<25><40><33><60>}"}
        assert vrsbench.vrsbench_ref_process_results(doc, [pred])["ref_acc50"] == 1

    def test_vrsbench_lenient_never_below_strict(self):
        doc = {"ground_truth": "{<0><0><1><1>}"}
        res = vrsbench.vrsbench_ref_process_results(doc, ["{<0><0><1><1>}"])
        assert res["strict_ref_acc50"] == 1
        assert res["ref_acc50"] >= res["strict_ref_acc50"]

    def test_vrsbench_strict_only_for_prompted_convention(self):
        doc = {"ground_truth": "{<25><40><33><60>}"}
        res = vrsbench.vrsbench_ref_process_results(doc, ["[0.25, 0.40, 0.33, 0.60]"])
        assert res["ref_acc50"] == 1
        assert res["strict_ref_acc50"] == 0

    def test_vrsbench_wrong_box_scores_zero(self):
        doc = {"ground_truth": "{<25><40><33><60>}"}
        res = vrsbench.vrsbench_ref_process_results(doc, ["{<70><70><90><90>}"])
        assert res["ref_acc50"] == 0
        assert res["strict_ref_acc50"] == 0

    def test_vrsbench_unparseable_scores_zero(self):
        doc = {"ground_truth": "{<25><40><33><60>}"}
        assert vrsbench.vrsbench_ref_process_results(doc, ["no box here"]) == {"ref_acc50": 0, "strict_ref_acc50": 0}

    CONVENTIONS_01 = [
        ("0-1-floats", "[0.25 0.40, 0.33 0.60]"),
        ("0-100-ints", "[25, 40, 33, 60]"),
        ("0-1000-ints", "[250, 400, 330, 600]"),
        ("120px", "[30.0, 48.0, 39.6, 72.0]"),
    ]

    @pytest.mark.parametrize("name,pred", CONVENTIONS_01, ids=[c[0] for c in CONVENTIONS_01])
    def test_bigearth_bbox_scores_every_convention(self, name, pred):
        doc = {"output": "[0.25 0.40, 0.33 0.60]"}
        assert bigearth.bigearth_bbox_process_results(doc, [pred])["bbox_acc50"] == 1

    def test_bigearth_wrong_box_scores_zero(self):
        doc = {"output": "[0.25 0.40, 0.33 0.60]"}
        assert bigearth.bigearth_bbox_process_results(doc, ["[0.7, 0.7, 0.9, 0.9]"])["bbox_acc50"] == 0

    def test_geobench_ref_rescues_standard_conventions(self):
        gt = [0.25, 0.40, 0.33, 0.60]
        for pred in ([0.25, 0.40, 0.33, 0.60], [25, 40, 33, 60], [250, 400, 330, 600]):
            assert geobench._best_iou(pred, gt, (1.0, 0.01, 0.001)) == pytest.approx(1.0)

    def test_scale_candidates_equivalent_across_tasks(self):
        pred = [250.0, 400.0, 330.0, 600.0]
        gt100 = [25.0, 40.0, 33.0, 60.0]
        gt01 = [0.25, 0.40, 0.33, 0.60]
        assert vrsbench._best_iou(pred, gt100) == pytest.approx(bigearth._best_iou(pred, gt01))


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
