#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=apertus-eval
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/lmms-eval/examples/apertus-vllm/apertus-vllm-lmms-eval-iopsstor.toml
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/store/cscs/swissai/infra01/multimodal-eval/apertus-lmms-eval/lmms-eval/examples/apertus-vllm/logs/apertus_eval_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/infra01/multimodal-eval/apertus-lmms-eval/lmms-eval/examples/apertus-vllm/logs/apertus_eval_%j.err

# Canonical eval job template — called by eval.sh.
# Supports both fill and readonly modes via --image-token-cache-* flags
# passed by the orchestrator. Defaults are tuned for readonly (production eval).
#
# Direct invocation is supported for advanced use; the more common entrypoint
# is the orchestrator:
#   bash eval.sh <model> [--tasks ...] [--suite ...] [--mode fill|readonly]
#
# Direct example:
#   sbatch eval_job.slurm \
#     --model-path /path/to/ckpt \
#     --tokenizer-path /capstor/.../tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed \
#     --chat-template /capstor/.../tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed/chat_template.jinja \
#     --tasks mmmu_val \
#     --image-token-cache-dir /capstor/.../benchmark/image_token_cache/mmmu_val
set -euo pipefail

REPO_ROOT="/capstor/store/cscs/swissai/infra01/multimodal-eval/apertus-lmms-eval/lmms-eval/examples/apertus-vllm"
CACHE_ROOT="${REPO_ROOT}/cache"



########################################
# Required runtime configuration
########################################
DEFAULT_TOKENIZER_PATH="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed"
MODEL_PATH="/path/to/apertus/model-or-hf-id"
TOKENIZER_PATH="${DEFAULT_TOKENIZER_PATH}"
MODEL_BACKEND="${MODEL_BACKEND:-apertus_1p5_vllm}"
OUTPUT_PATH="${REPO_ROOT}/results"
LOG_DIR="${REPO_ROOT}/logs"
HF_HOME_PATH="${HF_HOME:-${CACHE_ROOT}/hf}"
NLTK_DATA_PATH="${NLTK_DATA:-${CACHE_ROOT}/nltk_data}"
IMAGE_TOKEN_CACHE_DIR="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR:-${CACHE_ROOT}}"
XDG_CACHE_HOME_PATH="${XDG_CACHE_HOME:-${CACHE_ROOT}/.cache}"
VLLM_CACHE_ROOT_PATH="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm_cache}"
LMMS_EVAL_MODELS_CACHE_PATH="${LMMS_EVAL_MODELS_CACHE:-${CACHE_ROOT}/models_cache}"

########################################
# Evaluation configuration
########################################
TASKS="pope,gqa,vqav2_val,mmmu_test_group_img,mme,ai2d,ocrbench_v2,chartqa,docvqa,infovqa"
# batch_size is the request-chunk size handed to vLLM's LLM.chat per call
# (the continuous-batching window), not a GPU memory knob — max_num_seqs /
# max_num_batched_tokens govern memory. 1 disables continuous batching
# (~100x slower).
BATCH_SIZE=512
NUM_PROCESSES=4
# Data-parallel workers (NUM_PROCESSES) shard requests; tensor-parallel
# (TENSOR_PARALLEL_SIZE) shards one model across GPUs. 8B fits one GPU
# (TP=1, 4 DP workers); a 70B needs TP=4 with NUM_PROCESSES=1.
TENSOR_PARALLEL_SIZE=1
MAIN_PROCESS_PORT=29541
GPU_MEMORY_UTILIZATION=0.6
MAX_MODEL_LEN=131072
MAX_NUM_BATCHED_TOKENS=49152
HF_OVERRIDES='{"max_position_embeddings":131072}'
TRUST_REMOTE_CODE=False
ENABLE_PREFIX_CACHING=true
ENFORCE_EAGER=false


OFFLINE_DATASETS=false
LIMIT=""
GEN_KWARGS=""
CHAT_TEMPLATE=""
EXTRA_MODEL_ARGS=""
# Empty default preserves the established eval protocol (deliberation
# disabled): no enable_thinking kwarg is passed at all, keeping the
# client.chat call identical to historical runs. Set true/false explicitly
# to override.
ENABLE_THINKING="${ENABLE_THINKING:-}"
LAUNCHER_ARGS=""
SEED=1234
EVAL_SEED="${SEED}"
ENABLE_IMAGE_TOKEN_CACHE=true
IMAGE_TOKEN_CACHE_READONLY="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY:-1}"
IMAGE_TOKEN_CACHE_WRITE_MISSES="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES:-0}"
IMAGE_TOKEN_MEMORY_CACHE_SIZE="${VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE:-131072}"
IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS="${VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS:-5000}"
IMAGE_TOKEN_SQLITE_MMAP_SIZE="${VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE:-1073741824}"
IMAGE_TOKEN_CACHE_DEBUG="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG:-0}"
PREFETCH_EMU35_VISION_TOKENIZER=true
ENABLE_WANDB=false
WANDB_PROJECT="${WANDB_PROJECT:-lmms-eval}"
WANDB_ENTITY="${WANDB_ENTITY:-alvor}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-eval}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_LOG_SAMPLES="${WANDB_LOG_SAMPLES:-false}"
WANDB_ARGS="${WANDB_ARGS:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
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

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# LOCAL_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# REPO_ROOT="${REPO_ROOT:-${LOCAL_REPO_ROOT}}"
# if [[ ! -d "${REPO_ROOT}/vllm" ]]; then
REPO_ROOT="/workspace"
# fi
VLLM_REPO="${VLLM_REPO:-${REPO_ROOT}/vllm}"
LMMS_EVAL_REPO="${LMMS_EVAL_REPO:-${REPO_ROOT}/lmms-eval}"

usage() {
  cat <<'EOF'
Usage:
  sbatch eval_job.slurm -- [options]

Options:
  --model-path <path_or_hf_id>           Required model path or HF id.
  --tokenizer-path <path>                Tokenizer path.
  --model-backend <name>                 lmms-eval model backend.
  --tasks <csv>                          lmms-eval task list.
  --task-list <path>                     Task list for SLURM array mode.
  --run-output-root <path>               Per-run output root; array tasks write under <path>/<task>.
  --output-path <path>                   Output directory.
  --log-dir <path>                       Log directory.
  --hf-home <path>                       HF_HOME path.
  --nltk-data <path>                     NLTK_DATA path.
  --xdg-cache-home <path>                XDG_CACHE_HOME path.
  --vllm-cache-root <path>               VLLM_CACHE_ROOT path.
  --models-cache <path>                  LMMS_EVAL_MODELS_CACHE path.
  --batch-size <int>                     Per-rank batch size.
  --num-processes <int>                  Accelerate process count (data-parallel workers).
  --tensor-parallel-size <int>           vLLM tensor_parallel_size (model sharding). Default: 1.
  --main-process-port <int>              Accelerate main process port.
  --gpu-memory-utilization <float>       vLLM gpu_memory_utilization.
  --max-model-len <int>                  Max model length (>8192).
  --max-num-batched-tokens <int>         vLLM max_num_batched_tokens.
  --trust-remote-code <True|False>       vLLM trust_remote_code.
  --enable-prefix-caching <true|false>   vLLM prefix caching.
  --enforce-eager <true|false>           vLLM enforce_eager.
  --hf-overrides <json>                  vLLM hf_overrides JSON string.
  --gen-kwargs <kv_csv>                  lmms-eval --gen_kwargs string.
  --launcher-args <kv_csv>               lmms-eval --launcher_args for local judge launchers.
  --seed <seed_spec>                     lmms-eval deterministic seed tuple
                                         (e.g. 0,1234,1234,1234 or single int).
  --limit <int_or_float>                 lmms-eval --limit.
  --chat-template <path_or_template>     chat template for vLLM.
  --extra-model-args <kv_csv>            Extra --model_args suffix.
  --enable-thinking <true|false>         Pass enable_thinking to chat templates that support it.
  --offline-datasets <true|false>        Default: false.
  --enable-image-token-cache <true|false>
                                         Enable Apertus image tokenization cache.
  --image-token-cache-dir <path>         Existing directory for SQLite cache.
  --image-token-cache-readonly <0|1|true|false>
                                         Open/use image-token cache as read-only during eval. Default: 1.
                                         Readonly implies strict mode: a missing prebuilt DB fails the
                                         first image batch (VLLM_APERTUS_IMAGE_TOKEN_CACHE_STRICT=0 to allow
                                         uncached fallback).
  --image-token-cache-write-misses <0|1|true|false>
                                         Persist misses to SQLite during eval. Default: 0.
  --image-token-memory-cache-size <int>  In-process LRU size.
  --image-token-sqlite-busy-timeout-ms <int>
                                         SQLite busy timeout in milliseconds.
  --image-token-sqlite-mmap-size <bytes> SQLite mmap_size hint in bytes.
  --image-token-cache-debug <0|1|true|false>
                                         Enable cache debug logs.
  --prefetch-emu35-vision-tokenizer <true|false>
                                         Prefetch BAAI/Emu3.5-VisionTokenizer once before launch.
  --enable-wandb <true|false>            Enable W&B metric logging from inside the job. Default: false.
                                         Note: compute nodes may need proxy setup for outbound HTTPS;
                                         the post-hoc push (push_results_to_wandb.py) is the safer path.
  --wandb-project <name>                 W&B project. Default: lmms-eval.
  --wandb-entity <name>                  W&B entity/team. Default: alvor.
  --wandb-job-type <name>                W&B job_type. Default: eval.
  --wandb-group <name>                   W&B run group.
  --wandb-run-name <name>                W&B run name. Default: model name.
  --wandb-run-id <id>                    W&B run id. Default: deterministic id from model name.
  --wandb-resume <mode>                  W&B resume mode (allow|must|never|auto). Default: allow.
  --wandb-log-samples <true|false>       Also upload eval samples to W&B. Default: false.
  --wandb-args <kv_csv>                  Extra lmms-eval --wandb_args (comma-separated key=value).
  --wandb-api-key <key>                  W&B API key. If omitted, uses WANDB_API_KEY env.
  --healthbench-grader-model <name>      OpenAI-compatible grader model for HealthBench.
  --healthbench-grader-base-url <url>    OpenAI-compatible grader API base URL, including /v1.
  --healthbench-grader-api-key <key>     API key sent to the HealthBench grader endpoint.
  --healthbench-grader-max-tokens <int>  Max tokens for one rubric grading request.
  --healthbench-grader-concurrency <int> Per-example rubric grading concurrency.
  --healthbench-grader-max-retries <int> Retries for malformed or failed grader responses.
  --debug-mode                           Enable debug mode (sets VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG=1).
  --help                                 Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --model-backend) MODEL_BACKEND="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --task-list) TASK_LIST_FILE="$2"; shift 2 ;;
    --run-output-root) RUN_OUTPUT_ROOT="$2"; shift 2 ;;
    --image-token-cache-base) IMAGE_TOKEN_CACHE_BASE="$2"; shift 2 ;;
    --output-path) OUTPUT_PATH="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --hf-home) HF_HOME_PATH="$2"; shift 2 ;;
    --nltk-data) NLTK_DATA_PATH="$2"; shift 2 ;;
    --xdg-cache-home) XDG_CACHE_HOME_PATH="$2"; shift 2 ;;
    --vllm-cache-root) VLLM_CACHE_ROOT_PATH="$2"; shift 2 ;;
    --models-cache) LMMS_EVAL_MODELS_CACHE_PATH="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-processes) NUM_PROCESSES="$2"; shift 2 ;;
    --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --main-process-port) MAIN_PROCESS_PORT="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --trust-remote-code) TRUST_REMOTE_CODE="$2"; shift 2 ;;
    --enable-prefix-caching) ENABLE_PREFIX_CACHING="$2"; shift 2 ;;
    --enforce-eager) ENFORCE_EAGER="$2"; shift 2 ;;
    --hf-overrides) HF_OVERRIDES="$2"; shift 2 ;;
    --gen-kwargs) GEN_KWARGS="$2"; shift 2 ;;
    --launcher-args) LAUNCHER_ARGS="$2"; shift 2 ;;
    --seed) EVAL_SEED="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --chat-template) CHAT_TEMPLATE="$2"; shift 2 ;;
    --extra-model-args) EXTRA_MODEL_ARGS="$2"; shift 2 ;;
    --enable-thinking) ENABLE_THINKING="$2"; shift 2 ;;
    --offline-datasets) OFFLINE_DATASETS="$2"; shift 2 ;;
    --enable-image-token-cache) ENABLE_IMAGE_TOKEN_CACHE="$2"; shift 2 ;;
    --image-token-cache-dir) IMAGE_TOKEN_CACHE_DIR="$2"; shift 2 ;;
    # Deprecated no-ops (collision guard, node-local copy, and preload were
    # removed from the cache implementation); accepted so mixed-version
    # orchestrators don't hard-fail.
    --image-token-cache-collision-guard|--image-token-cache-local-copy|--image-token-cache-local-base|--image-token-cache-preload)
      echo "Info: $1 is deprecated and ignored."
      if [[ $# -ge 2 ]]; then shift 2; else shift; fi ;;
    --image-token-cache-readonly) IMAGE_TOKEN_CACHE_READONLY="$2"; shift 2 ;;
    --image-token-cache-write-misses) IMAGE_TOKEN_CACHE_WRITE_MISSES="$2"; shift 2 ;;
    --image-token-memory-cache-size) IMAGE_TOKEN_MEMORY_CACHE_SIZE="$2"; shift 2 ;;
    --image-token-sqlite-busy-timeout-ms) IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS="$2"; shift 2 ;;
    --image-token-sqlite-mmap-size) IMAGE_TOKEN_SQLITE_MMAP_SIZE="$2"; shift 2 ;;
    --image-token-cache-debug) IMAGE_TOKEN_CACHE_DEBUG="$2"; shift 2 ;;
    --prefetch-emu35-vision-tokenizer) PREFETCH_EMU35_VISION_TOKENIZER="$2"; shift 2 ;;
    --enable-wandb) ENABLE_WANDB="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-job-type) WANDB_JOB_TYPE="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --wandb-run-name) WANDB_RUN_NAME="$2"; shift 2 ;;
    --wandb-run-id) WANDB_RUN_ID="$2"; shift 2 ;;
    --wandb-resume) WANDB_RESUME="$2"; shift 2 ;;
    --wandb-log-samples) WANDB_LOG_SAMPLES="$2"; shift 2 ;;
    --wandb-args) WANDB_ARGS="$2"; shift 2 ;;
    --wandb-api-key) WANDB_API_KEY="$2"; shift 2 ;;
    --healthbench-grader-model) HEALTHBENCH_GRADER_MODEL="$2"; shift 2 ;;
    --healthbench-grader-base-url) HEALTHBENCH_GRADER_BASE_URL="$2"; shift 2 ;;
    --healthbench-grader-api-key) HEALTHBENCH_GRADER_API_KEY="$2"; shift 2 ;;
    --healthbench-grader-max-tokens) HEALTHBENCH_GRADER_MAX_TOKENS="$2"; shift 2 ;;
    --healthbench-grader-concurrency) HEALTHBENCH_GRADER_CONCURRENCY="$2"; shift 2 ;;
    --healthbench-grader-max-retries) HEALTHBENCH_GRADER_MAX_RETRIES="$2"; shift 2 ;;
    --debug-mode) IMAGE_TOKEN_CACHE_DEBUG=1; shift ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

is_true() {
  local value
  value="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

tasks_include_healthbench() {
  local task_csv=",${TASKS},"
  case "${task_csv}" in
    *,healthbench,*|*,healthbench_hard,*|*,healthbench_consensus,*) return 0 ;;
    *) return 1 ;;
  esac
}

# ─── Array job support ───
# If --task-list was passed and SLURM_ARRAY_TASK_ID is set, resolve the task
# from the list file. This enables: sbatch --array=0-N eval_job.slurm --task-list file.txt
if [[ -n "${TASK_LIST_FILE:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if [[ ! -f "${TASK_LIST_FILE}" ]]; then
    echo "ERROR: task list not found: ${TASK_LIST_FILE}" >&2
    exit 1
  fi
  TASKS="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASK_LIST_FILE}")"
  if [[ -z "$TASKS" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for ${TASK_LIST_FILE}" >&2
    exit 1
  fi
  if [[ -n "${RUN_OUTPUT_ROOT:-}" ]]; then
    OUTPUT_PATH="${RUN_OUTPUT_ROOT}/${TASKS}"
    mkdir -p "${OUTPUT_PATH}"
  fi
  # Resolve per-task cache dir from base. Only fill mode creates the dir:
  # pre-creating it in readonly mode makes "dir exists" useless as a
  # filled-vs-never-filled signal.
  if [[ -n "${IMAGE_TOKEN_CACHE_BASE:-}" ]]; then
    IMAGE_TOKEN_CACHE_DIR="${IMAGE_TOKEN_CACHE_BASE}/${TASKS}"
    if ! is_true "${IMAGE_TOKEN_CACHE_READONLY}"; then
      mkdir -p "${IMAGE_TOKEN_CACHE_DIR}"
    fi
  fi
  echo "Array element ${SLURM_ARRAY_TASK_ID}: task=${TASKS}"
fi

if tasks_include_healthbench; then
  if [[ -z "${HEALTHBENCH_GRADER_BASE_URL}" ]]; then
    echo "ERROR: HealthBench needs a Qwen judge endpoint." >&2
    echo "       Launch Qwen with apertus/model-launch, then set HEALTHBENCH_GRADER_ENDPOINT=host:port" >&2
    echo "       or --healthbench-grader-base-url http://host:port/v1." >&2
    exit 1
  fi
  if [[ "${HEALTHBENCH_GRADER_BASE_URL}" == https://api.swissai.svc.cscs.ch* && "${HEALTHBENCH_GRADER_API_KEY}" == "EMPTY" ]]; then
    echo "ERROR: HealthBench via SwissAI serving requires --healthbench-grader-api-key, HEALTHBENCH_GRADER_API_KEY, or SML_CSCS_API_KEY." >&2
    echo "       For no-key scoring, use a direct endpoint: HEALTHBENCH_GRADER_ENDPOINT=host:port." >&2
    exit 1
  fi
fi

# Ensure OCR-safe default generation cap.
# If max_new_tokens is absent, append/set it to 32768.
# If caller already set max_new_tokens, keep caller value.
if [[ -z "${GEN_KWARGS}" ]]; then
  GEN_KWARGS="max_new_tokens=32768"
elif [[ ! "${GEN_KWARGS}" =~ (^|,)max_new_tokens= ]]; then
  GEN_KWARGS="${GEN_KWARGS},max_new_tokens=32768"
fi

########################################
# Runtime environment
########################################
export API_TYPE=dummy
export PATH="/opt/venv/bin:${PATH}"
# Optional dev-tree override: prepend a user-local lmms-eval checkout to PYTHONPATH
# so changes in dev are picked up without writing to the shared prod tree.
# Usage: LMMS_EVAL_DEV_PATH=/path/to/your/lmms-eval bash scripts/eval.sh ...
#        VLLM_DEV_PATH=/path/to/your/vllm bash scripts/eval.sh ...
LMMS_EVAL_DEV_PATH="${LMMS_EVAL_DEV_PATH:-}"
VLLM_DEV_PATH="${VLLM_DEV_PATH:-}"
if [[ -n "${VLLM_DEV_PATH}" ]]; then
  # Copy only changed .py files into the pip-installed vllm package.
  # Cannot use PYTHONPATH override because compiled C extensions (.so) live
  # in the installed package and won't be found from a source checkout.
  VLLM_INSTALLED="$(python3 -c 'import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent)' 2>/dev/null)"
  if [[ -n "${VLLM_INSTALLED}" && -d "${VLLM_INSTALLED}" ]]; then
    n=0
    while IFS= read -r src; do
      rel="${src#${VLLM_DEV_PATH}/vllm/}"
      dst="${VLLM_INSTALLED}/${rel}"
      [[ -f "${dst}" ]] && cp "${src}" "${dst}" && n=$((n+1))
    done < <(find "${VLLM_DEV_PATH}/vllm" -name "*.py" -newer "${VLLM_INSTALLED}/__init__.py" 2>/dev/null)
    echo "vLLM dev patch: copied ${n} file(s) from ${VLLM_DEV_PATH}"
  fi
fi
if [[ -n "${LMMS_EVAL_DEV_PATH}" ]]; then
  echo "PYTHONPATH lmms-eval dev-tree override: ${LMMS_EVAL_DEV_PATH}"
  export PYTHONPATH="${LMMS_EVAL_DEV_PATH}:${LMMS_EVAL_REPO}:${VLLM_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${LMMS_EVAL_REPO}:${VLLM_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME_PATH}"
export NLTK_DATA="${NLTK_DATA_PATH}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME_PATH}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT_PATH}"
export LMMS_EVAL_MODELS_CACHE="${LMMS_EVAL_MODELS_CACHE_PATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TOKENIZERS_PARALLELISM=false
if is_true "${ENABLE_WANDB}"; then
  if [[ -z "${WANDB_API_KEY}" && -t 0 ]]; then
    read -r -s -p "Enter WANDB_API_KEY: " WANDB_API_KEY
    echo
  fi
  if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "ERROR: W&B enabled but API key is missing. Pass --wandb-api-key or set WANDB_API_KEY."
    exit 1
  fi
  export WANDB_API_KEY
  export WANDB_ENTITY="${WANDB_ENTITY}"
  export WANDB_PROJECT="${WANDB_PROJECT}"
fi
# Keep Python hash() stable across processes/runs (affects some internal naming).
HASH_SEED="${EVAL_SEED%%,*}"
if ! [[ "${HASH_SEED}" =~ ^[0-9]+$ ]]; then
  HASH_SEED="${SEED}"
fi
export PYTHONHASHSEED="${PYTHONHASHSEED:-${HASH_SEED}}"
export VLLM_SEED="${VLLM_SEED:-${SEED}}"
# Pin the lmms-eval image transport to lossless PNG. The image token cache is
# keyed on decoded pixels; a lossy format (JPEG/WEBP) would change pixels and
# silently 100%-miss every prebuilt cache.
export LMMS_IMAGE_ENCODE_FORMAT=PNG
if tasks_include_healthbench; then
  if [[ "${HEALTHBENCH_GRADER_BASE_URL}" == http://* ]]; then
    HEALTHBENCH_GRADER_HOST="${HEALTHBENCH_GRADER_BASE_URL#http://}"
    HEALTHBENCH_GRADER_HOST="${HEALTHBENCH_GRADER_HOST%%/*}"
    HEALTHBENCH_GRADER_HOST="${HEALTHBENCH_GRADER_HOST%%:*}"
    export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${HEALTHBENCH_GRADER_HOST}"
    export no_proxy="${no_proxy:+${no_proxy},}${HEALTHBENCH_GRADER_HOST}"
  fi
  export HEALTHBENCH_GRADER_MODEL
  export HEALTHBENCH_GRADER_BASE_URL
  export HEALTHBENCH_GRADER_API_KEY
  export HEALTHBENCH_GRADER_MAX_TOKENS
  export HEALTHBENCH_GRADER_CONCURRENCY
  export HEALTHBENCH_GRADER_MAX_RETRIES
fi

if is_true "${OFFLINE_DATASETS}"; then
  export HF_DATASETS_OFFLINE=1
else
  unset HF_DATASETS_OFFLINE || true
fi

if is_true "${ENABLE_IMAGE_TOKEN_CACHE}"; then
  echo "Image-token cache dir: ${IMAGE_TOKEN_CACHE_DIR}"

  CACHE_DB="${IMAGE_TOKEN_CACHE_DIR}/image_tokens/apertus_image_token_cache.sqlite3"
  if is_true "${IMAGE_TOKEN_CACHE_READONLY}"; then
    # Submit-time visibility only: text-only tasks legitimately have no DB,
    # so the hard gate lives in the cache code (strict mode raises at the
    # first image batch when the DB is missing).
    if [[ ! -s "${CACHE_DB}" ]]; then
      echo "WARNING: readonly image-token cache requested but ${CACHE_DB} is missing or empty."
      echo "         Image-bearing tasks will FAIL at the first image batch (strict mode)."
      echo "         Fill it first: bash eval.sh <model> --tasks ${TASKS} --mode fill"
    fi
  else
    mkdir -p "${IMAGE_TOKEN_CACHE_DIR}"
  fi

  export VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR="${IMAGE_TOKEN_CACHE_DIR}"
  export VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY="${IMAGE_TOKEN_CACHE_READONLY}"
  export VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES="${IMAGE_TOKEN_CACHE_WRITE_MISSES}"
  export VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE="${IMAGE_TOKEN_MEMORY_CACHE_SIZE}"
  export VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS="${IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS}"
  export VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE="${IMAGE_TOKEN_SQLITE_MMAP_SIZE}"
  export VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG="${IMAGE_TOKEN_CACHE_DEBUG}"
else
  unset VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR || true
  unset VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY || true
  unset VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES || true
  unset VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE || true
  unset VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS || true
  unset VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE || true
  unset VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG || true
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH must be set."
  exit 1
fi

if [[ "${MAX_MODEL_LEN}" -le 8192 ]]; then
  echo "MAX_MODEL_LEN must be > 8192 (got ${MAX_MODEL_LEN})."
  exit 1
fi

if [[ -z "${CHAT_TEMPLATE}" && "${TOKENIZER_PATH}" == "${DEFAULT_TOKENIZER_PATH}" ]]; then
  CHAT_TEMPLATE="${TOKENIZER_PATH}/chat_template.jinja"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
else
  set +e
  NUM_GPUS_RAW="$(nvidia-smi -L 2>/dev/null | wc -l)"
  NVS_STATUS=$?
  set -e
  if [[ ${NVS_STATUS} -eq 0 && "${NUM_GPUS_RAW}" -gt 0 ]]; then
    NUM_GPUS="${NUM_GPUS_RAW}"
  else
    set +e
    TORCH_GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
    TORCH_STATUS=$?
    set -e
    if [[ ${TORCH_STATUS} -eq 0 && "${TORCH_GPU_COUNT}" -gt 0 ]]; then
      NUM_GPUS="${TORCH_GPU_COUNT}"
      echo "Warning: nvidia-smi probe failed (status=${NVS_STATUS}); using torch.cuda.device_count=${NUM_GPUS}."
    else
      NUM_GPUS=0
      echo "Warning: unable to detect GPUs via nvidia-smi or torch; assuming 0 visible GPUs."
    fi
  fi
fi

NEED_GPUS=$(( NUM_PROCESSES * TENSOR_PARALLEL_SIZE ))
if [[ "${NUM_GPUS}" -lt "${NEED_GPUS}" ]]; then
  echo "Need at least ${NEED_GPUS} visible GPUs (${NUM_PROCESSES} workers x TP=${TENSOR_PARALLEL_SIZE}), found ${NUM_GPUS}."
  exit 1
fi

mkdir -p "${OUTPUT_PATH}" "${NLTK_DATA}" "${LOG_DIR}" "${HF_HOME}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${LMMS_EVAL_MODELS_CACHE}"

# Ensure ANTLR runtime version expected by OmegaConf/Emu3.5 tokenizer path.
# This keeps jobs robust even if the base image has a newer incompatible ANTLR.
if [[ "${PIN_ANTLR_RUNTIME:-true}" == "true" ]]; then
  /opt/venv/bin/pip install --disable-pip-version-check --no-cache-dir --quiet --force-reinstall \
    "antlr4-python3-runtime==4.9.3"
fi

# Fail fast on known ANTLR/OmegaConf mismatch that otherwise appears late.
python - <<'PY'
import importlib.util
import os
import sys

try:
    import omegaconf  # noqa: F401
except Exception as exc:
    print("Preflight failed: cannot import omegaconf.", file=sys.stderr)
    print(f"Root cause: {exc}", file=sys.stderr)
    sys.exit(2)

emu35_root = os.getenv("VLLM_APERTUS_EMU35_CODEBASE", "/workspace/Emu3.5")
module_path = os.path.join(emu35_root, "src", "vision_tokenizer", "__init__.py")
if not os.path.isfile(module_path):
    print(f"Preflight failed: missing Emu3.5 vision tokenizer module: {module_path}", file=sys.stderr)
    sys.exit(2)

spec = importlib.util.spec_from_file_location(
    "emu35_preflight_module",
    module_path,
    submodule_search_locations=[os.path.dirname(module_path)],
)
if spec is None or spec.loader is None:
    print(f"Preflight failed: unable to build import spec for {module_path}", file=sys.stderr)
    sys.exit(2)

module = importlib.util.module_from_spec(spec)
sys.modules["emu35_preflight_module"] = module
try:
    spec.loader.exec_module(module)
except Exception as exc:
    print("Preflight failed: cannot import Emu3.5 vision tokenizer module.", file=sys.stderr)
    print(f"Root cause: {exc}", file=sys.stderr)
    sys.exit(2)
PY

if is_true "${PREFETCH_EMU35_VISION_TOKENIZER}"; then
  PREFETCH_DIR="${LMMS_EVAL_MODELS_CACHE}/BAAI/Emu3.5-VisionTokenizer"
  if [[ ! -f "${PREFETCH_DIR}/config.yaml" || ! -f "${PREFETCH_DIR}/model.ckpt" ]]; then
    echo "Prefetching BAAI/Emu3.5-VisionTokenizer into ${PREFETCH_DIR}"
    PREFETCH_DIR_ENV="${PREFETCH_DIR}" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="BAAI/Emu3.5-VisionTokenizer",
    local_dir=os.environ["PREFETCH_DIR_ENV"],
    allow_patterns=["config.yaml", "model.ckpt"],
)
PY
  fi
fi

MODEL_ARGS="model=${MODEL_PATH},tokenizer=${TOKENIZER_PATH},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},trust_remote_code=${TRUST_REMOTE_CODE},max_model_len=${MAX_MODEL_LEN},hf_overrides=${HF_OVERRIDES}"
MODEL_ARGS="${MODEL_ARGS},max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS},enable_prefix_caching=${ENABLE_PREFIX_CACHING},enforce_eager=${ENFORCE_EAGER}"
# Only pass enable_thinking when explicitly set; an absent kwarg keeps the
# chat-template call identical to historical (deliberation-disabled) runs.
if [[ -n "${ENABLE_THINKING}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},enable_thinking=${ENABLE_THINKING}"
fi

if [[ -n "${CHAT_TEMPLATE}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},chat_template=${CHAT_TEMPLATE}"
fi
if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
fi

MODEL_LABEL="$(basename "${MODEL_PATH}")"
if [[ "${MODEL_LABEL}" == "HF" ]]; then
  MODEL_LABEL="$(basename "$(dirname "${MODEL_PATH}")")"
fi
MODEL_LABEL="${MODEL_LABEL// /_}"
if [[ -z "${WANDB_RUN_NAME}" ]]; then
  WANDB_RUN_NAME="${MODEL_LABEL}"
fi
if [[ -z "${WANDB_RUN_ID}" ]]; then
  WANDB_MODEL_HASH="$(printf "%s" "${MODEL_LABEL}" | sha1sum | awk '{print substr($1,1,12)}')"
  WANDB_RUN_ID="model_${WANDB_MODEL_HASH}"
fi

append_wandb_arg() {
  local key="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    return
  fi
  if [[ -n "${FINAL_WANDB_ARGS}" ]]; then
    FINAL_WANDB_ARGS+=","
  fi
  FINAL_WANDB_ARGS+="${key}=${value}"
}

wandb_arg_exists() {
  local key="$1"
  [[ ",${FINAL_WANDB_ARGS}," == *",${key}="* ]]
}

FINAL_WANDB_ARGS="${WANDB_ARGS}"
if ! wandb_arg_exists "project"; then
  append_wandb_arg "project" "${WANDB_PROJECT}"
fi
if ! wandb_arg_exists "entity"; then
  append_wandb_arg "entity" "${WANDB_ENTITY}"
fi
if ! wandb_arg_exists "job_type"; then
  append_wandb_arg "job_type" "${WANDB_JOB_TYPE}"
fi
if ! wandb_arg_exists "group"; then
  append_wandb_arg "group" "${WANDB_GROUP}"
fi
if ! wandb_arg_exists "name"; then
  append_wandb_arg "name" "${WANDB_RUN_NAME}"
fi
if ! wandb_arg_exists "id"; then
  append_wandb_arg "id" "${WANDB_RUN_ID}"
fi
if ! wandb_arg_exists "resume"; then
  append_wandb_arg "resume" "${WANDB_RESUME}"
fi

CMD=(
  accelerate launch
  --num_processes "${NUM_PROCESSES}"
  --num_machines 1
  --main_process_port "${MAIN_PROCESS_PORT}"
  -m lmms_eval
  --model "${MODEL_BACKEND}"
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --seed "${EVAL_SEED}"
  --output_path "${OUTPUT_PATH}"
  --log_samples
  --log_samples_suffix "vllm_prod"
)

if [[ -n "${GEN_KWARGS}" ]]; then
  CMD+=(--gen_kwargs "${GEN_KWARGS}")
fi
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if tasks_include_healthbench && [[ -n "${LAUNCHER_ARGS}" ]]; then
  CMD+=(--launcher_args "${LAUNCHER_ARGS}")
fi
if is_true "${ENABLE_WANDB}" && [[ -n "${FINAL_WANDB_ARGS}" ]]; then
  CMD+=(--wandb_args "${FINAL_WANDB_ARGS}")
  if is_true "${WANDB_LOG_SAMPLES}"; then
    CMD+=(--wandb_log_samples)
  fi
fi

/opt/venv/bin/python -c "import nltk; nltk.download('wordnet')"

echo "========== lmms-eval prod launch =========="
echo "VLLM_REPO=${VLLM_REPO}"
echo "LMMS_EVAL_REPO=${LMMS_EVAL_REPO}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "CHAT_TEMPLATE=${CHAT_TEMPLATE:-<tokenizer default>}"
echo "MODEL_BACKEND=${MODEL_BACKEND}"
echo "TASKS=${TASKS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}"
echo "ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
echo "ENABLE_THINKING=${ENABLE_THINKING}"
echo "ENFORCE_EAGER=${ENFORCE_EAGER}"
echo "HF_OVERRIDES=${HF_OVERRIDES}"
echo "OFFLINE_DATASETS=${OFFLINE_DATASETS}"
echo "EVAL_SEED=${EVAL_SEED}"
echo "PYTHONHASHSEED=${PYTHONHASHSEED}"
echo "VLLM_SEED=${VLLM_SEED}"
echo "ENABLE_WANDB=${ENABLE_WANDB}"
if is_true "${ENABLE_WANDB}"; then
  echo "WANDB_ARGS=${FINAL_WANDB_ARGS}"
  echo "WANDB_LOG_SAMPLES=${WANDB_LOG_SAMPLES}"
fi
echo "ENABLE_IMAGE_TOKEN_CACHE=${ENABLE_IMAGE_TOKEN_CACHE}"
if is_true "${ENABLE_IMAGE_TOKEN_CACHE}"; then
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE=${VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS=${VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE=${VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES}"
fi
echo "HF_HOME=${HF_HOME}"
echo "NLTK_DATA=${NLTK_DATA}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
echo "LMMS_EVAL_MODELS_CACHE=${LMMS_EVAL_MODELS_CACHE}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "PREFETCH_EMU35_VISION_TOKENIZER=${PREFETCH_EMU35_VISION_TOKENIZER}"
if [[ -n "${GEN_KWARGS}" ]]; then
  echo "GEN_KWARGS=${GEN_KWARGS}"
fi
if tasks_include_healthbench; then
  echo "HEALTHBENCH_GRADER_MODEL=${HEALTHBENCH_GRADER_MODEL}"
  echo "HEALTHBENCH_GRADER_BASE_URL=${HEALTHBENCH_GRADER_BASE_URL}"
  echo "HEALTHBENCH_GRADER_CONCURRENCY=${HEALTHBENCH_GRADER_CONCURRENCY}"
  echo "HEALTHBENCH_LAUNCHER_ARGS=${LAUNCHER_ARGS:-<none>}"
fi
echo "==========================================="

"${CMD[@]}"
