#!/bin/bash
# Generate the medical benchmark results table from evaluation logs.
# Usage: bash scripts/generate_results_table.sh

LOG_DIR="/iopsstor/scratch/cscs/mikhaika/lmms-eval/logs"

# Model definitions: name, log_pattern, total_samples, n_ranks
# For models with multiple runs, we list all logs
cat << 'PYEOF' | python3 -
import re, os

LOG_DIR = "/iopsstor/scratch/cscs/mikhaika/lmms-eval/logs"

# Define models and their log files
MODELS = {
    "Apertus-8B":      {"logs": ["eval_apertus_1783312"], "samples": 7827, "ranks": 4, "release": "Aug 2025"},
    "Apertus-70B":     {"logs": ["eval_apertus_70b_1783313"], "samples": 7827, "ranks": 1, "release": "Aug 2025"},
    "Apertus-1.5":     {"logs": ["eval_apertus_emu3p5_1677203", "eval_apertus_emu3p5_1617212", "eval_apertus_emu3p5_1772858", "eval_apertus_emu3p5_1773638"], "samples": 25487, "ranks": 4, "release": "—"},
    "Llama-3.2":       {"logs": ["eval_llama_vision_1675635", "eval_llama_vision_1675655", "eval_llama_vision_1675601", "eval_llama_vision_1683998", "eval_llama_vision_1617171", "eval_llama_vision_1773141", "eval_mmlu_1777590"], "samples": None, "ranks": 4, "release": "Sep 2024"},
    "Qwen3-VL-8B":     {"logs": ["eval_qwen3_vl_1783308"], "samples": 50559, "ranks": 4, "release": "Jul 2025"},
    "Qwen3-VL-30B":    {"logs": ["eval_qwen3_vl_30b_1784098"], "samples": 50559, "ranks": 4, "release": "Jul 2025"},
    "Qwen3-VL-235B":   {"logs": ["eval_qwen3_vl_235b_1819158", "eval_qwen3_vl_235b_1826631"], "samples": 50559, "ranks": 1, "release": "Jul 2025"},
    "InternVL3-8B":    {"logs": ["eval_internvl3_1783309"], "samples": 50559, "ranks": 4, "release": "Apr 2025"},
    "InternVL3-78B":   {"logs": ["eval_internvl3_1784070"], "samples": 50559, "ranks": 1, "release": "Apr 2025"},
    "Gemma4-26B":      {"logs": ["eval_gemma4_26b_1840421"], "samples": 50559, "ranks": 4, "release": "Mar 2026"},
    "Gemma4-31B":      {"logs": ["eval_gemma4_31b_1849764"], "samples": 50559, "ranks": 4, "release": "Mar 2026"},
    "MedGemma-27B":    {"logs": ["eval_medgemma_1783311"], "samples": 50559, "ranks": 4, "release": "Jul 2025"},
    "No image":        {"logs": ["eval_internvl3_no_image_1783823"], "samples": None, "ranks": 4, "release": ""},
}

# Tasks to extract
TASKS = [
    ("medmcqa", "Text", "accuracy", "accuracy"),
    ("medqa", "Text", "accuracy", "accuracy"),
    ("mmlu (medical)", "Text", "exact_match", "exact_match"),
    ("pubmedqa", "Text", "accuracy", "accuracy"),
    ("pmc_vqa", "VQA", "accuracy", "accuracy"),
    ("slake", "VQA", "close_accuracy", "close_accuracy"),
    ("path_vqa", "VQA", "close_accuracy", "close_accuracy"),
    ("vqa_rad", "VQA", "close_accuracy", "close_accuracy"),
    ("path_mmu", "VQA", "accuracy", "accuracy"),
]

# Task name mapping for log parsing
TASK_LOG_NAMES = {
    "path_mmu": "path_mmu_test_tiny",
    "mmlu (medical)": "mmlu (medical)",
}

def extract_scores(log_path):
    """Extract task scores from a log file."""
    scores = {}
    try:
        with open(log_path) as f:
            for line in f:
                if not line.startswith("|"):
                    continue
                if "Stderr" in line or "---" in line or "strict-match" in line or " - mmlu_flan" in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 9:
                    continue
                task_name = parts[1]
                metric = parts[5]
                try:
                    value = float(parts[7])
                except:
                    continue
                if value == 0.0 and metric == "exact_match" and "strict" not in line:
                    continue
                scores[(task_name, metric)] = value
    except:
        pass
    return scores

def extract_throughput(log_paths, total_samples):
    """Extract inference time from Metric summary lines."""
    total_time = 0
    for log_path in log_paths:
        try:
            with open(log_path) as f:
                content = f.read()
            matches = re.findall(r"Total time: ([\d.]+)s", content)
            if matches:
                total_time += float(matches[-1])
        except:
            pass
    if total_samples and total_time > 0:
        return total_time / total_samples
    return None

# Collect all scores
all_scores = {}
all_throughput = {}
for model_name, info in MODELS.items():
    model_scores = {}
    log_paths = [os.path.join(LOG_DIR, f"{l}.log") for l in info["logs"]]
    for log_path in log_paths:
        model_scores.update(extract_scores(log_path))
    all_scores[model_name] = model_scores
    all_throughput[model_name] = extract_throughput(log_paths, info.get("samples"))

# Print table
model_names = list(MODELS.keys())
header = "Task             Type    Metric          " + "  ".join(f"{m:>13s}" for m in model_names)
print(header)

# Release dates
release_line = "Release date                             " + "  ".join(f"{MODELS[m]['release']:>13s}" for m in model_names)
print(release_line)

for task_display, task_type, metric_name, metric_display in TASKS:
    task_log = TASK_LOG_NAMES.get(task_display, task_display)
    values = []
    for model_name in model_names:
        scores = all_scores[model_name]
        key = (task_log, metric_name)
        # Try variations
        val = scores.get(key)
        if val is None:
            # Try with _no_image suffix
            if model_name == "No image":
                key2 = (task_log + "_no_image", metric_name)
                val = scores.get(key2)
        if val is not None:
            values.append(f"{val*100:.2f}%")
        else:
            values.append("—")
    line = f"{task_display:17s}{task_type:8s}{metric_display:16s}" + "  ".join(f"{v:>13s}" for v in values)
    print(line)

# Throughput
tp_values = []
for model_name in model_names:
    tp = all_throughput[model_name]
    if tp is not None:
        tp_values.append(f"{tp:.3f}")
    else:
        tp_values.append("—")
tp_line = "Throughput (s/sam)                        " + "  ".join(f"{v:>13s}" for v in tp_values)
print(tp_line)

print()
print("†mmlu underestimated — output format mismatch with FLAN filter.")
print("‡No image baseline: InternVL3-8B.")
print()
print("Logs:")
for model_name, info in MODELS.items():
    if model_name == "No image":
        continue
    logs_str = ", ".join(info["logs"])
    print(f"  {model_name:20s} {logs_str}")
PYEOF