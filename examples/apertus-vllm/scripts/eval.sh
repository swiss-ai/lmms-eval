#!/usr/bin/env bash
# eval.sh — Apertus VLM eval CLI (per-task SQLite cache, single user entry point).
#
# Usage:
#   bash eval.sh <model> [--tasks T | --suite full|smoke] [--mode fill|readonly] [--size 8b|70b] [--help]
#
# <model> forms:
#   /path/to/ckpt                       single path
#   /a,/b,/c                            comma-separated paths
#   @file.txt                           one path per line (comments # and blanks OK)
#
# --tasks   comma-separated, or @file.txt. If omitted, falls back to --suite.
# --suite   named curation: full (default, ~46 tasks) | smoke (3-task canary).
# --mode    fill (populate per-task cache) | readonly (default, eval against cache).
# --size    parallelism profile: 8b (default; TP=1, 4 data-parallel workers) |
#           70b (TP=4, 1 worker, model sharded across all 4 GPUs).
# --max-concurrent N  maximum active SLURM array elements per model.
# MAX_MODEL_LEN / MAX_NUM_BATCHED_TOKENS envs are forwarded to eval_job.slurm.
#
# Examples:
#   bash eval.sh /path/to/ckpt                          # full suite, readonly
#   bash eval.sh /path/to/ckpt --suite smoke            # 3-task sanity
#   bash eval.sh /path/to/ckpt --tasks mmmu_val,chartqa # specific tasks
#   bash eval.sh /a,/b,/c                               # multiple models
#   bash eval.sh @models.txt --tasks @custom.txt
#   bash eval.sh /path/to/ckpt --mode fill              # populate cache first
#
# Cache layout (per-task SQLite, bounded growth per file):
#   $CACHE_BASE/{task}/image_tokens/apertus_image_token_cache.sqlite3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_TEMPLATE="${SLURM_TEMPLATE:-${SCRIPT_DIR}/eval_job.slurm}"
SUMMARY_TEMPLATE="${SUMMARY_TEMPLATE:-${SCRIPT_DIR}/summarize_job.slurm}"
CACHE_BASE="${CACHE_BASE:-/capstor/store/cscs/swissai/infra01/vision-datasets/benchmark/image_token_cache}"

# ------------------------------------------------------------------
# Canonical task suites (curations — edit here, not in external files)
# ------------------------------------------------------------------
# Smoke: canary subset, ~5 min/model.
SUITE_SMOKE="gqa,mmstar,pope"

# Full: 57-task working set (includes the gqa/mmstar/pope canary tasks, which
# can also be run alone via --suite smoke). Excludes:
#   - spatial / multi-image: owned by VLMEvalKit/EASI (easi.lmms-lab.com), run
#     there for protocol parity, not here: embspatial, blink, cv_bench,
#     mmsi_bench, 3dsrbench_circular, site_bench_image, mindcube_tiny,
#     muirbench, erqa, omnispatial_test, sparbench
#   - broken upstream / known-fail: chartqapro, vending_bench2, amber_g, dude,
#     dynamath_reasoning, mmlongbench_doc, refspatial,
#     seedbench_2, vcr_wiki_en_easy, vcr_wiki_en_hard, viewspatial,
#     where2place, zerobench
#   - exceeds 131K context (32 images × 5K tokens): vsibench_multiimage, vsibench_debiased_multiimage, viewspatial
SUITE_FULL="gqa,realworldqa,seedbench,ocrbench,ocrbench_v2,textvqa_val,docvqa_val,vqav2_val,infovqa_val,chartqa,mme,ai2d,mmmu_val,mathvision_test,mathvision_testmini,VisualPuzzles_direct,countbench,pixmo_count,refcoco,refcoco+,refcocog,mmmu_pro,mathvision_reason_test,mathvision_reason_testmini,mmvp,vlmsareblind,vlms_are_biased,scienceqa,iconqa_val,omnidocbench,visulogic,vstar_bench,seedbench_2_plus,mmbench_en_dev,mmstar,pope,mathvista_testmini,refcoco_bbox_rec_test,refcoco_bbox_rec_testA,refcoco_bbox_rec_testB,refcoco_bbox_rec_val,refcoco+_bbox_rec_testA,refcoco+_bbox_rec_testB,refcoco+_bbox_rec_val,refcocog_bbox_rec_test,refcocog_bbox_rec_val,medqa,medmcqa,pubmedqa,path_vqa,vqa_rad,slake,pmc_vqa,path_mmu_test,medxpertqa_mm,medxpertqa_text,mmerealworld"

# ------------------------------------------------------------------
# CLI parsing
# ------------------------------------------------------------------
usage() {
  awk 'NR>1 && /^#/{sub(/^# ?/, ""); print; next} NR>1 && !/^#/{exit}' "${BASH_SOURCE[0]}"
}

if [[ $# -lt 1 ]]; then usage; exit 1; fi

MODELS_RAW=""
TASKS_RAW=""
SUITE=""
MODE="readonly"
SIZE="${SIZE:-8b}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)  usage; exit 0 ;;
    --tasks)    TASKS_RAW="$2"; shift 2 ;;
    --suite)    SUITE="$2"; shift 2 ;;
    --mode)     MODE="$2"; shift 2 ;;
    --size)     SIZE="$2"; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --*)        echo "unknown flag: $1" >&2; usage; exit 1 ;;
    *)
      # First positional = model(s)
      if [[ -z "$MODELS_RAW" ]]; then MODELS_RAW="$1"; shift
      else echo "unexpected positional: $1" >&2; usage; exit 1
      fi
      ;;
  esac
done

if [[ -z "$MODELS_RAW" ]]; then echo "missing <model> argument" >&2; usage; exit 1; fi

case "$MODE" in fill|readonly) ;; *) echo "--mode must be fill|readonly (got: $MODE)" >&2; exit 1 ;; esac
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-concurrent must be a positive integer (got: $MAX_CONCURRENT)" >&2
  exit 1
fi

# ------------------------------------------------------------------
# Resolve models: inline path, comma-separated, or @file
# ------------------------------------------------------------------
resolve_list() {
  # $1 = raw value ("/path", "/a,/b,/c", or "@file.txt")
  local raw="$1"
  if [[ "$raw" == @* ]]; then
    local f="${raw#@}"
    [[ -f "$f" ]] || { echo "file not found: $f" >&2; exit 1; }
    # Strip comments and blanks; keep one per line. grep exits 1 on an
    # all-blank list, which set -e would turn into a silent death — the
    # callers handle emptiness themselves.
    sed -E 's/[[:space:]]*#.*$//' "$f" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | grep -v '^$' || true
  else
    echo "$raw" | tr ',' '\n' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | grep -v '^$' || true
  fi
}

model_label_from_path() {
  local path="$1"
  local base parent

  base="$(basename "$path")"
  if [[ "$base" == "HF" ]]; then
    base="$(basename "$(dirname "$path")")"
  elif [[ "$base" == iter_*_hf ]]; then
    parent="$(dirname "$path")"
    base="$(basename "$(dirname "$parent")")"
  fi

  printf '%s\n' "${base// /_}"
}

MODELS=$(resolve_list "$MODELS_RAW")
[[ -z "$MODELS" ]] && { echo "no models resolved from: $MODELS_RAW" >&2; exit 1; }

# ------------------------------------------------------------------
# Resolve tasks: --tasks takes priority, else --suite, else default full
# ------------------------------------------------------------------
if [[ -n "$TASKS_RAW" ]]; then
  TASKS=$(resolve_list "$TASKS_RAW")
else
  case "${SUITE:-full}" in
    full)  TASKS=$(echo "$SUITE_FULL"  | tr ',' '\n') ;;
    smoke) TASKS=$(echo "$SUITE_SMOKE" | tr ',' '\n') ;;
    *)     echo "--suite must be full|smoke (got: $SUITE)" >&2; exit 1 ;;
  esac
fi
[[ -z "$TASKS" ]] && { echo "no tasks resolved" >&2; exit 1; }

# ------------------------------------------------------------------
# Container-environment fixes for sbatch from inside Pyxis container.
#
# This script is invoked from inside a container. Two things bite sbatch here:
#   1. The container inherits SLURM_SPANK_* env vars from the parent job which
#      conflict with the new submission's --environment flag. Strip them.
#   2. libjson-c.so.5 (needed by pyxis) is missing from default library path
#      inside the container; we keep a copy in the team wheelhouse.
# ------------------------------------------------------------------
unset $(env | awk -F= '/^SLURM_SPANK/{print $1}') 2>/dev/null || true
export LD_LIBRARY_PATH="/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}"

# ------------------------------------------------------------------
# Auto-load WANDB_API_KEY from ~/.netrc and HF_TOKEN from HF cache.
# /users isn't mounted inside the job container so token files there are
# unreachable from the inner slurm job; we re-export here.
# ------------------------------------------------------------------
if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.netrc" ]]; then
  WANDB_API_KEY=$(awk '/api.wandb.ai/{flag=1;next} flag && /password/{print $2; exit}' "$HOME/.netrc")
  export WANDB_API_KEY
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN=$(<"$HOME/.cache/huggingface/token")
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# ------------------------------------------------------------------
# Eval defaults (override via env if needed)
# ------------------------------------------------------------------
DEFAULT_TOKENIZER_PATH="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed"
TOKENIZER_PATH="${TOKENIZER_PATH:-${DEFAULT_TOKENIZER_PATH}}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
if [[ -z "${CHAT_TEMPLATE}" && "${TOKENIZER_PATH}" == "${DEFAULT_TOKENIZER_PATH}" ]]; then
  CHAT_TEMPLATE="${TOKENIZER_PATH}/chat_template.jinja"
fi
# 8192 covers ~85% of reasoning-task generations without truncation. Math/reasoning
# tasks at lower budgets show 30-50% mid-response truncation. MCQ hits EOS well
# before this, so no cost for short-answer tasks.
GEN_KWARGS="${GEN_KWARGS:-max_new_tokens=8192,temperature=0}"
# Extra model_args appended verbatim (e.g. EXTRA_MODEL_ARGS="min_image_pixels=28").
# Note: image_first only affects the simple backend; the chat backend used in
# production takes message ordering from each task's doc_to_messages.
# eval_job.slurm only appends when non-empty, so empty default is safe.
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"
# Empty default = do not pass enable_thinking at all, preserving the
# established deliberation-disabled protocol of all historical runs.
# Set ENABLE_THINKING=true explicitly to switch the chat template.
ENABLE_THINKING="${ENABLE_THINKING:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-49152}"
HF_OVERRIDES="${HF_OVERRIDES:-{\"max_position_embeddings\":${MAX_MODEL_LEN}}}"
# Parallelism profile by model size. 8b fits one GH200, so 4 data-parallel
# workers (1 per GPU), TP=1; 70b is sharded across all 4 GPUs (TP=4, one
# worker). Explicit env overrides any single knob for advanced use.
case "$SIZE" in
  8b)  _NUM_PROCESSES=4; _TENSOR_PARALLEL_SIZE=1; _GPU_MEMORY_UTILIZATION=0.6 ;;
  70b) _NUM_PROCESSES=1; _TENSOR_PARALLEL_SIZE=4; _GPU_MEMORY_UTILIZATION=0.85 ;;
  *)   echo "--size must be 8b|70b (got: $SIZE)" >&2; exit 1 ;;
esac
NUM_PROCESSES="${NUM_PROCESSES:-$_NUM_PROCESSES}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$_TENSOR_PARALLEL_SIZE}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$_GPU_MEMORY_UTILIZATION}"
# Request-chunk size handed to vLLM per LLM.chat call (continuous-batching
# window), not a memory knob; 1 disables continuous batching (~100x slower).
BATCH_SIZE="${BATCH_SIZE:-512}"

# HealthBench is scored by an OpenAI-compatible rubric grader after target-model
# generation. In Apertus runs, launch Qwen separately through the sibling
# model-launch checkout and point this script at a direct worker endpoint such
# as HEALTHBENCH_GRADER_ENDPOINT=nid006114:8080. Direct worker endpoints do not
# require an API key.
HEALTHBENCH_GRADER_USER="${USER:-$(id -un)}"
HEALTHBENCH_GRADER_MODEL="${HEALTHBENCH_GRADER_MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-${HEALTHBENCH_GRADER_USER}}"
HEALTHBENCH_GRADER_ENDPOINT="${HEALTHBENCH_GRADER_ENDPOINT:-}"
HEALTHBENCH_GRADER_BASE_URL="${HEALTHBENCH_GRADER_BASE_URL:-}"
if [[ -z "${HEALTHBENCH_GRADER_BASE_URL}" && -n "${HEALTHBENCH_GRADER_ENDPOINT}" ]]; then
  if [[ "${HEALTHBENCH_GRADER_ENDPOINT}" == http://* || "${HEALTHBENCH_GRADER_ENDPOINT}" == https://* ]]; then
    HEALTHBENCH_GRADER_BASE_URL="${HEALTHBENCH_GRADER_ENDPOINT%/}"
  else
    HEALTHBENCH_GRADER_BASE_URL="http://${HEALTHBENCH_GRADER_ENDPOINT%/}"
  fi
  [[ "${HEALTHBENCH_GRADER_BASE_URL}" == */v1 ]] || HEALTHBENCH_GRADER_BASE_URL="${HEALTHBENCH_GRADER_BASE_URL}/v1"
fi
HEALTHBENCH_GRADER_API_KEY="${HEALTHBENCH_GRADER_API_KEY:-${SML_CSCS_API_KEY:-${OPENAI_API_KEY:-EMPTY}}}"
HEALTHBENCH_GRADER_MAX_TOKENS="${HEALTHBENCH_GRADER_MAX_TOKENS:-2048}"
HEALTHBENCH_GRADER_CONCURRENCY="${HEALTHBENCH_GRADER_CONCURRENCY:-16}"
HEALTHBENCH_GRADER_MAX_RETRIES="${HEALTHBENCH_GRADER_MAX_RETRIES:-3}"
# Optional escape hatch for explicit local judge launchers. Empty means use the
# external OpenAI-compatible endpoint above.
HEALTHBENCH_LAUNCHER_ARGS="${HEALTHBENCH_LAUNCHER_ARGS:-}"

tasks_include_healthbench() {
  local t
  while IFS= read -r t; do
    case "$t" in
      healthbench|healthbench_hard|healthbench_consensus) return 0 ;;
    esac
  done <<< "$TASKS"
  return 1
}

if tasks_include_healthbench; then
  if [[ -z "${HEALTHBENCH_GRADER_BASE_URL}" ]]; then
    echo "ERROR: HealthBench needs a Qwen judge endpoint." >&2
    echo "       Launch Qwen with apertus/model-launch, then set HEALTHBENCH_GRADER_ENDPOINT=host:port" >&2
    echo "       or HEALTHBENCH_GRADER_BASE_URL=http://host:port/v1." >&2
    exit 1
  fi
  if [[ "${HEALTHBENCH_GRADER_BASE_URL}" == https://api.swissai.svc.cscs.ch* && "${HEALTHBENCH_GRADER_API_KEY}" == "EMPTY" ]]; then
    echo "ERROR: HealthBench via SwissAI serving requires HEALTHBENCH_GRADER_API_KEY or SML_CSCS_API_KEY." >&2
    echo "       For no-key scoring, use a direct endpoint: HEALTHBENCH_GRADER_ENDPOINT=host:port." >&2
    exit 1
  fi
fi

# WandB config. Default off: in-job streaming needs outbound HTTPS from
# compute nodes; the reliable path is the post-hoc push
# (scripts/push_results_to_wandb.py) against the gathered results.
# Set ENABLE_WANDB=true to opt into in-job streaming.
ENABLE_WANDB="${ENABLE_WANDB:-false}"
WANDB_ENTITY="${WANDB_ENTITY:-alvor}"
WANDB_PROJECT="${WANDB_PROJECT:-apertus-1p5-eval}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-}"
WANDB_LOG_SAMPLES="${WANDB_LOG_SAMPLES:-false}"

if [[ "$ENABLE_WANDB" == "true" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARNING: ENABLE_WANDB=true but WANDB_API_KEY is empty (not in ~/.netrc). Jobs will fail fast."
fi

# ------------------------------------------------------------------
# Logging — Anunay's inner default uses scripts/logs which is group-readable
# but NOT writable for us (signal-53). Use the repo logs dir instead.
# ------------------------------------------------------------------
LOG_DIR="${LOG_DIR:-/capstor/store/cscs/swissai/infra01/users/xyixuan/apertus-1p5-eval}"
mkdir -p "$LOG_DIR"
RUNS_ROOT="${RUNS_ROOT:-${LOG_DIR}/runs}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_$$}"

# ------------------------------------------------------------------
# Mode-specific cache flags
#   fill:     write to per-task SQLite (populate it)
#   readonly: read from per-task SQLite (no writes; a missing DB fails the
#             first image batch — strict mode in the cache code)
# Per-task caches are read directly from the shared filesystem (readonly
# opens use SQLite immutable mode, so no lock traffic).
# ------------------------------------------------------------------
case "$MODE" in
  fill)
    CACHE_READONLY=0
    CACHE_WRITE_MISSES=1
    ;;
  readonly)
    CACHE_READONLY=1
    CACHE_WRITE_MISSES=0
    ;;
esac

# ------------------------------------------------------------------
# Pretty header
# ------------------------------------------------------------------
echo "========================================"
echo "Apertus eval"
echo "  mode:       $MODE"
echo "  models:     $(echo "$MODELS" | tr '\n' ',' | sed 's/,$//')"
echo "  tasks:      $(echo "$TASKS"  | tr '\n' ',' | sed 's/,$//')"
echo "  tokenizer:  $TOKENIZER_PATH"
echo "  template:   ${CHAT_TEMPLATE:-<tokenizer default>}"
echo "  cache base: $CACHE_BASE"
echo "  slurm:      $SLURM_TEMPLATE"
echo "  finalizer:  $SUMMARY_TEMPLATE"
echo "  runs root:  $RUNS_ROOT"
echo "  run id:     $RUN_ID"
echo "  size:       $SIZE (procs=$NUM_PROCESSES tp=$TENSOR_PARALLEL_SIZE gpu_mem=$GPU_MEMORY_UTILIZATION)"
echo "  concurrency:$MAX_CONCURRENT"
echo "  max len:    $MAX_MODEL_LEN"
echo "  thinking:   $ENABLE_THINKING"
echo "  hf overrides:$HF_OVERRIDES"
if tasks_include_healthbench; then
  echo "  healthbench judge: $HEALTHBENCH_GRADER_MODEL @ $HEALTHBENCH_GRADER_BASE_URL"
fi
echo "  wandb:      $ENABLE_WANDB ($WANDB_ENTITY/$WANDB_PROJECT)"
echo "========================================"

# ------------------------------------------------------------------
# Submit as SLURM job array: one array element per task for each model.
# Benefits over individual sbatch calls:
#   - One submission per model (not 50+)
#   - No filename collision (each element has unique SLURM_ARRAY_TASK_ID)
#   - Easy cancel/monitor (one job ID for all tasks)
#   - Concurrency control via %N suffix
# ------------------------------------------------------------------
SUBMITTED_ARRAYS=()
SUBMITTED_SUMMARIES=()
SUBMITTED_RUN_ROOTS=()

while IFS= read -r MODEL_PATH; do
  [[ -z "$MODEL_PATH" ]] && continue
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "WARNING: model path not found, skipping: $MODEL_PATH" >&2
    continue
  fi

  MODEL_LABEL="${MODEL_LABEL_OVERRIDE:-$(model_label_from_path "$MODEL_PATH")}"
  WANDB_GROUP="${WANDB_GROUP_PREFIX}${MODEL_LABEL}"
  RUN_ROOT="${RUNS_ROOT}/${MODEL_LABEL}/${RUN_ID}"
  RUN_LOG_DIR="${RUN_ROOT}/logs"
  RUN_RESULTS_ROOT="${RUN_ROOT}/results"
  RUN_SUMMARY_DIR="${RUN_ROOT}/summary"
  mkdir -p "${RUN_LOG_DIR}" "${RUN_RESULTS_ROOT}" "${RUN_SUMMARY_DIR}"

  TASK_LIST="${RUN_ROOT}/tasks.txt"
  printf "%s\n" "$TASKS" > "$TASK_LIST"
  NUM_TASKS=$(printf "%s\n" "$TASKS" | grep -cve '^[[:space:]]*$')

  # Pre-create cache dirs only when filling; in readonly mode an absent dir
  # is the signal that a task was never filled.
  if [[ "$MODE" == "fill" ]]; then
    while IFS= read -r T; do
      [[ -n "$T" ]] && mkdir -p "$CACHE_BASE/$T"
    done <<< "$TASKS"
  fi

  echo "--- submit array: model=$MODEL_LABEL  tasks=$NUM_TASKS  max_concurrent=$MAX_CONCURRENT ---"
  echo "    run root:  $RUN_ROOT"
  echo "    task list: $TASK_LIST"

  ARRAY_JOB_ID=$(sbatch --parsable \
    --array="0-$((NUM_TASKS - 1))%${MAX_CONCURRENT}" \
    --output "${RUN_LOG_DIR}/eval_${MODEL_LABEL}_%A_%a.out" \
    --error  "${RUN_LOG_DIR}/eval_${MODEL_LABEL}_%A_%a.err" \
    "$SLURM_TEMPLATE" \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --chat-template "$CHAT_TEMPLATE" \
    --task-list "$TASK_LIST" \
    --run-output-root "$RUN_RESULTS_ROOT" \
    --num-processes "$NUM_PROCESSES" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --batch-size "$BATCH_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --hf-overrides "$HF_OVERRIDES" \
    --gen-kwargs "$GEN_KWARGS" \
    --extra-model-args "$EXTRA_MODEL_ARGS" \
    --enable-thinking "$ENABLE_THINKING" \
    --launcher-args "$HEALTHBENCH_LAUNCHER_ARGS" \
    --healthbench-grader-model "$HEALTHBENCH_GRADER_MODEL" \
    --healthbench-grader-base-url "$HEALTHBENCH_GRADER_BASE_URL" \
    --healthbench-grader-api-key "$HEALTHBENCH_GRADER_API_KEY" \
    --healthbench-grader-max-tokens "$HEALTHBENCH_GRADER_MAX_TOKENS" \
    --healthbench-grader-concurrency "$HEALTHBENCH_GRADER_CONCURRENCY" \
    --healthbench-grader-max-retries "$HEALTHBENCH_GRADER_MAX_RETRIES" \
    --enable-image-token-cache true \
    --image-token-cache-base "$CACHE_BASE" \
    --image-token-cache-readonly "$CACHE_READONLY" \
    --image-token-cache-write-misses "$CACHE_WRITE_MISSES" \
    --enable-wandb "$ENABLE_WANDB" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-group "$WANDB_GROUP" \
    --wandb-log-samples "$WANDB_LOG_SAMPLES" \
    --wandb-api-key "${WANDB_API_KEY:-}")

  SUMMARY_JOB_ID=$(sbatch --parsable \
    --dependency=afterany:${ARRAY_JOB_ID} \
    --output "${RUN_LOG_DIR}/summary_%j.out" \
    --error  "${RUN_LOG_DIR}/summary_%j.err" \
    "$SUMMARY_TEMPLATE" \
    --run-root "$RUN_ROOT" \
    --task-list "$TASK_LIST" \
    --model-label "$MODEL_LABEL" \
    --array-job-id "$ARRAY_JOB_ID" \
    --scripts-dir "$SCRIPT_DIR")

  SUBMITTED_ARRAYS+=("$ARRAY_JOB_ID")
  SUBMITTED_SUMMARIES+=("$SUMMARY_JOB_ID")
  SUBMITTED_RUN_ROOTS+=("$RUN_ROOT")

  # Record which code submitted this run: the jobs execute whatever tree the
  # container toml mounts, so commit + dirty-count are the only provenance.
  GIT_COMMIT_JSON="$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)" \
  GIT_DIRTY_COUNT_JSON="$(git -C "$SCRIPT_DIR" status --porcelain 2>/dev/null | wc -l | tr -d ' ')" \
  SUBMISSION_JSON="${RUN_ROOT}/submission.json" \
  MODEL_PATH_JSON="${MODEL_PATH}" \
  MODEL_LABEL_JSON="${MODEL_LABEL}" \
  RUN_ROOT_JSON="${RUN_ROOT}" \
  TASK_LIST_JSON="${TASK_LIST}" \
  RUN_RESULTS_ROOT_JSON="${RUN_RESULTS_ROOT}" \
  RUN_LOG_DIR_JSON="${RUN_LOG_DIR}" \
  ARRAY_JOB_ID_JSON="${ARRAY_JOB_ID}" \
  SUMMARY_JOB_ID_JSON="${SUMMARY_JOB_ID}" \
  MODE_JSON="${MODE}" \
  TOKENIZER_PATH_JSON="${TOKENIZER_PATH}" \
  CHAT_TEMPLATE_JSON="${CHAT_TEMPLATE}" \
  MAX_CONCURRENT_JSON="${MAX_CONCURRENT}" \
  MAX_MODEL_LEN_JSON="${MAX_MODEL_LEN}" \
  ENABLE_THINKING_JSON="${ENABLE_THINKING}" \
  MAX_NUM_BATCHED_TOKENS_JSON="${MAX_NUM_BATCHED_TOKENS}" \
  HF_OVERRIDES_JSON="${HF_OVERRIDES}" \
  GEN_KWARGS_JSON="${GEN_KWARGS}" \
  python3 - <<'PY'
import json
import os

with open(os.environ["TASK_LIST_JSON"], encoding="utf-8") as fh:
    tasks = [line.strip() for line in fh if line.strip()]

payload = {
    "model_path": os.environ["MODEL_PATH_JSON"],
    "model_label": os.environ["MODEL_LABEL_JSON"],
    "mode": os.environ["MODE_JSON"],
    "run_root": os.environ["RUN_ROOT_JSON"],
    "task_list": os.environ["TASK_LIST_JSON"],
    "results_root": os.environ["RUN_RESULTS_ROOT_JSON"],
    "log_dir": os.environ["RUN_LOG_DIR_JSON"],
    "array_job_id": os.environ["ARRAY_JOB_ID_JSON"],
    "summary_job_id": os.environ["SUMMARY_JOB_ID_JSON"],
    "max_concurrent": int(os.environ["MAX_CONCURRENT_JSON"]),
    "max_model_len": int(os.environ["MAX_MODEL_LEN_JSON"]),
    "enable_thinking": os.environ["ENABLE_THINKING_JSON"],
    "gen_kwargs": os.environ["GEN_KWARGS_JSON"],
    "max_num_batched_tokens": int(os.environ["MAX_NUM_BATCHED_TOKENS_JSON"]),
    "hf_overrides": os.environ["HF_OVERRIDES_JSON"],
    "tokenizer_path": os.environ["TOKENIZER_PATH_JSON"],
    "chat_template": os.environ["CHAT_TEMPLATE_JSON"],
    "git_commit": os.environ.get("GIT_COMMIT_JSON", "unknown"),
    "git_dirty_files": int(os.environ.get("GIT_DIRTY_COUNT_JSON", "0") or 0),
    "tasks": tasks,
}
with open(os.environ["SUBMISSION_JSON"], "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
    fh.write("\n")
PY

  echo "    Submitted array job:   $ARRAY_JOB_ID (${NUM_TASKS} tasks, max ${MAX_CONCURRENT} concurrent)"
  echo "    Submitted summary job: $SUMMARY_JOB_ID (afterany:${ARRAY_JOB_ID})"
done <<< "$MODELS"

if [[ "${#SUBMITTED_ARRAYS[@]}" -eq 0 ]]; then
  echo "ERROR: No array jobs submitted." >&2
  exit 1
fi

echo "========================================"
echo "All submissions complete."
for idx in "${!SUBMITTED_ARRAYS[@]}"; do
  echo "  Array:   ${SUBMITTED_ARRAYS[$idx]}"
  echo "  Summary: ${SUBMITTED_SUMMARIES[$idx]}"
  echo "  Monitor: squeue -j ${SUBMITTED_ARRAYS[$idx]},${SUBMITTED_SUMMARIES[$idx]}"
  echo "  Cancel:  scancel ${SUBMITTED_ARRAYS[$idx]} ${SUBMITTED_SUMMARIES[$idx]}"
  echo "  Run:     ${SUBMITTED_RUN_ROOTS[$idx]}"
  echo "  Report:  ${SUBMITTED_RUN_ROOTS[$idx]}/summary/summary.md"
  echo "  CSV:     ${SUBMITTED_RUN_ROOTS[$idx]}/summary/summary.csv"
  echo "  JSON:    ${SUBMITTED_RUN_ROOTS[$idx]}/summary/summary.json"
  echo "  HTML:    ${SUBMITTED_RUN_ROOTS[$idx]}/summary/dashboard.html"
done
echo "========================================"
