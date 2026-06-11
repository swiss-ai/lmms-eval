from pathlib import Path

from lmms_eval import utils as lmms_utils
from lmms_eval.tasks.healthbench import utils

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_healthbench_doc_to_visual_is_empty_for_text_only_task():
    assert utils.healthbench_doc_to_visual({}) == []


def test_healthbench_doc_to_messages_preserves_multiturn_roles():
    doc = {
        "prompt": [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "follow up"},
        ],
    }

    messages = utils.healthbench_doc_to_messages(doc)

    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "first question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "first answer"}]},
        {"role": "user", "content": [{"type": "text", "text": "follow up"}]},
    ]


def test_healthbench_declares_grader_failure_rate_metric():
    config = lmms_utils.load_yaml_config(str(REPO_ROOT / "lmms_eval/tasks/healthbench/healthbench.yaml"), mode="full")

    metric_config = {metric["metric"]: metric for metric in config["metric_list"]}

    assert metric_config["healthbench_grader_failure_rate"]["higher_is_better"] is False


def test_grade_rubric_marks_unparseable_outputs_as_grader_failures(monkeypatch):
    class _Message:
        content = "not json"

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _Completions:
        def create(self, **_kwargs):
            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(utils, "_client", lambda: _Client())
    monkeypatch.setattr(utils, "_GRADER_MAX_RETRIES", 0)

    grade = utils._grade_rubric("user: hello\n\nassistant: hi", {"points": 1, "criterion": "responds"})

    assert grade["criteria_met"] is False
    assert grade["healthbench_grader_failed"] is True


def test_healthbench_process_results_records_rubric_grades_and_failure_rate(monkeypatch):
    doc = {
        "prompt": [{"role": "user", "content": "What should I do?"}],
        "rubrics": [
            {"criterion": "mentions follow-up", "points": 1, "tags": ["care"]},
            {"criterion": "contains unsupported diagnosis", "points": -1, "tags": ["safety"]},
        ],
    }

    def _fake_grade(_convo_str, rubric):
        if rubric["points"] > 0:
            return {
                "criteria_met": True,
                "explanation": "The response recommends follow-up.",
                "healthbench_grader_failed": False,
            }
        return {
            "criteria_met": False,
            "explanation": "Failed to parse grader output after retries",
            "healthbench_grader_failed": True,
        }

    monkeypatch.setattr(utils, "_grade_rubric", _fake_grade)
    monkeypatch.setattr(utils, "_GRADER_CONCURRENCY", 1)

    metrics = utils.healthbench_process_results(doc, ["Please arrange a follow-up."])

    assert metrics == {
        "healthbench_score": 1.0,
        "healthbench_grader_failure_rate": 0.5,
    }
    assert doc["healthbench_grader_summary"] == {
        "failed_rubrics": 1,
        "total_rubrics": 2,
    }
    assert doc["healthbench_rubric_grades"] == [
        {
            "criterion": "mentions follow-up",
            "points": 1,
            "tags": ["care"],
            "criteria_met": True,
            "explanation": "The response recommends follow-up.",
            "healthbench_grader_failed": False,
        },
        {
            "criterion": "contains unsupported diagnosis",
            "points": -1,
            "tags": ["safety"],
            "criteria_met": False,
            "explanation": "Failed to parse grader output after retries",
            "healthbench_grader_failed": True,
        },
    ]
