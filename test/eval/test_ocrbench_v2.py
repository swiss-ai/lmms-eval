from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from lmms_eval.tasks.ocrbench_v2 import spotting_metric, utils, vqa_metric


def test_ocrbench_v2_chart_parsing_uses_ground_truth(monkeypatch):
    captured = {}

    class DummyTEDS:
        def evaluate(self, pred_html, gt_html):
            captured["pred_html"] = pred_html
            captured["gt_html"] = gt_html
            return 0.42

    monkeypatch.setattr(utils, "teds", DummyTEDS())
    monkeypatch.setattr(utils, "convert_str_to_multi_dict", lambda raw: {"source": raw})
    monkeypatch.setattr(utils, "dict_to_html", lambda data: f"html::{data['source']}")

    result = utils.ocrbench_v2_process_results(
        {
            "question": "Parse the chart into a structured representation.",
            "answers": ["ground-truth-chart"],
            "type": "chart parsing en",
        },
        ["predicted-chart"],
    )

    assert captured == {
        "pred_html": "html::predicted-chart",
        "gt_html": "html::ground-truth-chart",
    }
    assert result["ocrbench_v2_accuracy"]["score"] == pytest.approx(0.42)


def test_ocrbench_v2_aggregate_accuracy_is_stateless(tmp_path):
    args = SimpleNamespace(output_path=str(tmp_path))

    first_score = utils.ocrbench_v2_aggregate_accuracy(
        [{"question_type": "text recognition en", "score": 1.0}],
        args,
    )
    second_score = utils.ocrbench_v2_aggregate_accuracy(
        [{"question_type": "text recognition en", "score": 0.0}],
        args,
    )

    assert first_score == pytest.approx(0.0625)
    assert second_score == 0.0


def test_spotting_evaluation_uses_temp_workdir(monkeypatch):
    captured = {}
    module_dir = Path(spotting_metric.__file__).resolve().parent / "spotting_eval"

    def fake_main_evaluation(command, default_params, validate, evaluate):
        captured["command"] = command
        return {"method": {"hmean": 0.9}}

    monkeypatch.setattr(spotting_metric.rrc_evaluation_funcs, "main_evaluation", fake_main_evaluation)

    score = spotting_metric.spotting_evaluation(
        [[0, 0, 10, 10, "hello"]],
        {"bbox_list": [[0, 0, 10, 0, 10, 10, 0, 10]], "content": ["hello"]},
    )

    assert score == pytest.approx(0.9)
    assert module_dir not in Path(captured["command"]["g"]).resolve().parents
    assert module_dir not in Path(captured["command"]["s"]).resolve().parents
    assert not (module_dir / "submit").exists()
    assert not (module_dir / "gt").exists()


def test_cn_vqa_evaluation_supports_scalar_answers():
    assert vqa_metric.cn_vqa_evaluation("答案是北京", "北京") == 1


def test_counting_evaluation_supports_scalar_answers():
    assert vqa_metric.counting_evaluation("There are 3 objects.", "3", "exact match") == 1


@pytest.mark.parametrize(
    ("prediction", "reference", "options", "expected"),
    [
        ("<td>a</td>", "<td>a</td>", {}, 1.0),
        ("<td>a</td>", "<td>b</td>", {}, 0.5),
        # One changed token among 3 (or 5), normalized by 3 HTML nodes.
        ("<td><b>a</b></td>", "<td><b>b</b></td>", {}, 8 / 9),
        ("<td><b>a</b> x</td>", "<td><b>a</b> y</td>", {}, 14 / 15),
        ("<td>a</td>", '<td colspan="2">a</td>', {}, 0.5),
        ("<td>a</td>", "<td>b</td>", {"structure_only": True}, 1.0),
        ("<td><b>a</b> tail</td>", "<td>a tail</td>", {"ignore_nodes": ["b"]}, 1.0),
    ],
)
def test_teds_preserves_serial_scores(prediction, reference, options, expected):
    scorer = utils.TEDS(**options)
    pred = f"<html><body><table><tr>{prediction}</tr></table></body></html>"
    truth = f"<html><body><table><tr>{reference}</tr></table></body></html>"
    assert scorer.evaluate(pred, truth) == pytest.approx(expected)


def test_ocrbench_v2_shared_teds_scores_concurrent_documents(monkeypatch):
    """A second document must not overwrite an unfinished cell's tokens."""
    first_cell_tokenized = Event()
    second_document_scored = Event()
    scorer = utils.TEDS(n_jobs=32)
    monkeypatch.setattr(utils, "teds", scorer)
    tokenize = scorer.tokenize

    def interleaved_tokenize(node):
        tokens = tokenize(node)
        if node.tag == "td" and node.text == "a" and not first_cell_tokenized.is_set():
            # Pause the first document after tokenization but before its cell
            # tree is built. The second document then completes on this same
            # scorer, deterministically exercising the shared-state race.
            first_cell_tokenized.set()
            assert second_document_scored.wait(timeout=10)
        return tokens

    monkeypatch.setattr(scorer, "tokenize", interleaved_tokenize)

    def score(text):
        table = f"<html><body><table><tr><td>{text}</td></tr></table></body></html>"
        doc = {"question": "Return the HTML table.", "answers": [table], "type": "table parsing en"}
        return utils.ocrbench_v2_process_results(doc, [table])["ocrbench_v2_accuracy"]["score"]

    def score_second_document():
        assert first_cell_tokenized.wait(timeout=10)
        try:
            return score("b")
        finally:
            second_document_scored.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(score, "a")
        second = executor.submit(score_second_document)
        scores = [first.result(timeout=15), second.result(timeout=15)]

    assert scores == [1.0, 1.0]
