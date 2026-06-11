import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SH = REPO_ROOT / "examples" / "apertus-vllm" / "scripts" / "eval.sh"
SUMMARY_SCRIPT = REPO_ROOT / "examples" / "apertus-vllm" / "scripts" / "summarize_results.py"


def _write_fake_sbatch(fakebin: Path) -> tuple[Path, Path]:
    fakebin.mkdir()
    log_path = fakebin / "sbatch.log"
    counter_path = fakebin / "counter"
    counter_path.write_text("9000\n", encoding="utf-8")
    sbatch = fakebin / "sbatch"
    sbatch.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
counter="${SBATCH_FAKE_COUNTER}"
job_id="$(cat "${counter}")"
job_id="$((job_id + 1))"
printf '%s\n' "${job_id}" > "${counter}"
printf 'CALL %s\n' "$*" >> "${SBATCH_FAKE_LOG}"
printf '%s\n' "${job_id}"
""",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    return log_path, counter_path


def _launcher_env(tmp_path: Path, fakebin: Path, sbatch_log: Path, counter: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fakebin}:{env['PATH']}",
            "SBATCH_FAKE_LOG": str(sbatch_log),
            "SBATCH_FAKE_COUNTER": str(counter),
            "LOG_DIR": str(tmp_path / "logs"),
            "RUNS_ROOT": str(tmp_path / "runs"),
            "CACHE_BASE": str(tmp_path / "cache"),
            "RUN_ID": "test-run",
            "MAX_CONCURRENT": "3",
            "ENABLE_WANDB": "false",
        }
    )
    return env


def test_eval_sh_submits_one_array_and_dependent_summary_job(tmp_path):
    model = tmp_path / "hf-checkpoints" / "Apertus-test"
    model.mkdir(parents=True)
    sbatch_log, counter = _write_fake_sbatch(tmp_path / "fakebin")

    result = subprocess.run(
        ["bash", str(EVAL_SH), str(model), "--tasks", "gqa,pope"],
        cwd=REPO_ROOT,
        env=_launcher_env(tmp_path, tmp_path / "fakebin", sbatch_log, counter),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_root = tmp_path / "runs" / "Apertus-test" / "test-run"
    assert (run_root / "tasks.txt").read_text(encoding="utf-8").splitlines() == ["gqa", "pope"]

    submission = json.loads((run_root / "submission.json").read_text(encoding="utf-8"))
    assert submission["array_job_id"] == "9001"
    assert submission["summary_job_id"] == "9002"
    assert submission["tasks"] == ["gqa", "pope"]
    assert submission["run_root"] == str(run_root)

    calls = sbatch_log.read_text(encoding="utf-8")
    assert "--array=0-1%3" in calls
    assert f"--task-list {run_root / 'tasks.txt'}" in calls
    assert f"--run-output-root {run_root / 'results'}" in calls
    assert "--enable-thinking true" in calls
    assert "--dependency=afterany:9001" in calls
    assert "summarize_job.slurm" in calls
    assert f"--run-root {run_root}" in calls
    assert str(run_root / "summary" / "summary.md") in result.stdout


def test_eval_sh_wires_healthbench_qwen_judge(tmp_path):
    model = tmp_path / "hf-checkpoints" / "Apertus-test"
    model.mkdir(parents=True)
    sbatch_log, counter = _write_fake_sbatch(tmp_path / "fakebin")
    env = _launcher_env(tmp_path, tmp_path / "fakebin", sbatch_log, counter)
    env["HEALTHBENCH_GRADER_ENDPOINT"] = "nid006114:8080"

    result = subprocess.run(
        ["bash", str(EVAL_SH), str(model), "--tasks", "healthbench,pope"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = sbatch_log.read_text(encoding="utf-8")
    assert "name=sglang,model=Qwen/Qwen3-235B-A22B-Instruct-2507" not in calls
    assert "--healthbench-grader-model Qwen/Qwen3-235B-A22B-Instruct-2507-" in calls
    assert "--healthbench-grader-base-url http://nid006114:8080/v1" in calls
    assert "--healthbench-grader-api-key EMPTY" in calls


def test_eval_sh_labels_hf_checkpoint_paths_by_training_run_directory(tmp_path):
    run_a = tmp_path / "ablations" / "run-a" / "checkpoints" / "iter_0003000_hf"
    run_b = tmp_path / "ablations" / "run-b" / "checkpoints" / "iter_0003000_hf"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    sbatch_log, counter = _write_fake_sbatch(tmp_path / "fakebin")

    result = subprocess.run(
        ["bash", str(EVAL_SH), f"{run_a},{run_b}", "--tasks", "gqa"],
        cwd=REPO_ROOT,
        env=_launcher_env(tmp_path, tmp_path / "fakebin", sbatch_log, counter),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    first_root = tmp_path / "runs" / "run-a" / "test-run"
    second_root = tmp_path / "runs" / "run-b" / "test-run"
    assert (first_root / "tasks.txt").is_file()
    assert (second_root / "tasks.txt").is_file()

    first_submission = json.loads((first_root / "submission.json").read_text(encoding="utf-8"))
    second_submission = json.loads((second_root / "submission.json").read_text(encoding="utf-8"))
    assert first_submission["model_label"] == "run-a"
    assert second_submission["model_label"] == "run-b"
    assert first_submission["run_root"] == str(first_root)
    assert second_submission["run_root"] == str(second_root)


def test_eval_sh_fails_cleanly_when_no_model_was_submitted(tmp_path):
    sbatch_log, counter = _write_fake_sbatch(tmp_path / "fakebin")

    result = subprocess.run(
        ["bash", str(EVAL_SH), str(tmp_path / "missing-model"), "--tasks", "gqa"],
        cwd=REPO_ROOT,
        env=_launcher_env(tmp_path, tmp_path / "fakebin", sbatch_log, counter),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "No array jobs submitted" in combined
    assert "unbound variable" not in combined
    assert not sbatch_log.exists()


def test_eval_sh_rejects_invalid_max_concurrent(tmp_path):
    model = tmp_path / "Apertus-test"
    model.mkdir()
    sbatch_log, counter = _write_fake_sbatch(tmp_path / "fakebin")

    result = subprocess.run(
        ["bash", str(EVAL_SH), str(model), "--tasks", "gqa", "--max-concurrent", "0"],
        cwd=REPO_ROOT,
        env=_launcher_env(tmp_path, tmp_path / "fakebin", sbatch_log, counter),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "--max-concurrent must be a positive integer" in result.stderr
    assert not sbatch_log.exists()


def test_summarize_results_writes_user_friendly_artifacts(tmp_path):
    run_root = tmp_path / "runs" / "Apertus-test" / "test-run"
    result_dir = run_root / "results" / "gqa" / "hf-checkpoints__Apertus-test"
    result_dir.mkdir(parents=True)
    (run_root / "logs").mkdir(parents=True)
    (run_root / "tasks.txt").write_text("gqa\npope\n", encoding="utf-8")
    (run_root / "logs" / "eval_Apertus-test_123_0.out").write_text("ok\n", encoding="utf-8")
    (run_root / "logs" / "eval_Apertus-test_123_1.err").write_text("failed\n", encoding="utf-8")
    (result_dir / "20260525_010101_gqa_results.json").write_text(
        json.dumps({"results": {"gqa": {"accuracy,none": 0.62, "accuracy_stderr,none": 0.01}}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "python",
            str(SUMMARY_SCRIPT),
            "--run-root",
            str(run_root),
            "--task-list",
            str(run_root / "tasks.txt"),
            "--model-label",
            "Apertus-test",
            "--array-job-id",
            "123",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_dir = run_root / "summary"
    assert (summary_dir / "summary.md").is_file()
    assert (summary_dir / "summary.csv").is_file()
    assert (summary_dir / "summary.json").is_file()
    assert (summary_dir / "dashboard.html").is_file()

    summary = json.loads((summary_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed"] == 1
    assert summary["missing"] == 1
    assert summary["tasks"][0]["task"] == "gqa"
    assert summary["tasks"][0]["metric"] == "accuracy"
    assert summary["tasks"][0]["value"] == 0.62
    assert summary["tasks"][1]["task"] == "pope"
    assert summary["tasks"][1]["status"] == "missing"

    markdown = (summary_dir / "summary.md").read_text(encoding="utf-8")
    assert "Apertus Eval Summary" in markdown
    assert "| gqa | completed | accuracy | 0.6200 |" in markdown
    assert "| pope | missing |  |  |" in markdown

    html = (summary_dir / "dashboard.html").read_text(encoding="utf-8")
    assert "Apertus 1.5 Evaluation" in html
    assert '<section class="stats">' in html
    assert '<span class="badge completed">completed</span>' in html
    assert '<span class="badge missing">missing</span>' in html
    assert 'href="../results/gqa/hf-checkpoints__Apertus-test/20260525_010101_gqa_results.json"' in html
    assert 'data-status="missing"' in html
