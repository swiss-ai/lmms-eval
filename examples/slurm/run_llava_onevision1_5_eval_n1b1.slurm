#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=eval-llava-ov1_5
#SBATCH --environment=/users/rkreft/.edf/new_ngc.toml
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/lmms-eval/logs/eval_llava_onevision1_5_%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/lmms-eval/logs/eval_llava_onevision1_5_%j.err

# Unset SSL_CERT_FILE to avoid FileNotFoundError inside NGC container
# (host path /etc/ssl/ca-bundle.pem doesn't exist in container)
unset SSL_CERT_FILE

TASKS="ai2d"
BATCH_SIZE=3
MODEL="llava_onevision1_5_chat"
MODEL_PATH="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"        # Required: path to HF model or HF hub identifier (e.g. "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct")
ATTN_IMPLEMENTATION="sdpa"
RES_PATH="/capstor/scratch/cscs/$USER/PDM/results/lmms_eval/llava_onevision1_5_results"
EVAL_DIR=/capstor/scratch/cscs/$USER/lmms-eval
OFFLINE_DATASETS="true"

# Wandb logging (disabled by default, use --wandb to enable)
WANDB_ENABLED="false"
WANDB_PROJECT="lmms-eval"
WANDB_LOG_SAMPLES="false"

# Debug mode (disabled by default, use --debug [N] to enable)
DEBUG_SAMPLES="false"
NUM_DEBUG_SAMPLES=5

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --attn-implementation)
            ATTN_IMPLEMENTATION="$2"
            shift 2
            ;;
        --wandb)
            WANDB_ENABLED="true"
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-log-samples)
            WANDB_LOG_SAMPLES="true"
            shift
            ;;
        --debug)
            DEBUG_SAMPLES="true"
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                NUM_DEBUG_SAMPLES="$2"
                shift
            fi
            shift
            ;;
        --no-offline-datasets)
            OFFLINE_DATASETS="false"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: --model-path is required"
    echo "Usage: sbatch $0 --model-path /path/to/model [OPTIONS]"
    exit 1
fi

# Automatically get number of available GPUS to configure accelerate properly
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Number of GPUs available: $NUM_GPUS"
echo "GPUS: $CUDA_VISIBLE_DEVICES"

# Print configuration
echo "========================================"
echo "MLLM Evaluation Configuration"
echo "========================================"
echo "Model:                ${MODEL}"
echo "Model path:           ${MODEL_PATH}"
echo "Tasks:                ${TASKS}"
echo "Batch size(per GPU):  ${BATCH_SIZE}"
echo "Attn implementation:  ${ATTN_IMPLEMENTATION}"
echo "Offline datasets:     ${OFFLINE_DATASETS}"
echo "Wandb enabled:        ${WANDB_ENABLED}"
echo "Wandb project:        ${WANDB_PROJECT}"
echo "Wandb log samples:    ${WANDB_LOG_SAMPLES}"
echo "Debug samples:        ${DEBUG_SAMPLES} (${NUM_DEBUG_SAMPLES})"
echo "Num GPU:              ${NUM_GPUS}"
echo "Output-Path:          ${RES_PATH}"
echo "Eval-Repo-Location:   ${EVAL_DIR}"
echo "========================================"
echo ""

# Conditionally set/unset offline mode
if [ "$OFFLINE_DATASETS" = "true" ]; then
    export HF_DATASETS_OFFLINE=1
else
    unset HF_DATASETS_OFFLINE
fi

echo "Setting up dependencies..."
python "${EVAL_DIR}/examples/install_deps.py" \
    --model "${MODEL}" \
    --tasks "${TASKS}" \
    --eval-dir "${EVAL_DIR}"
echo ""

# Build model_args with required parameters
MODEL_ARGS="pretrained=${MODEL_PATH},attn_implementation=${ATTN_IMPLEMENTATION}"
if [ -n "$MAX_LENGTH" ]; then
    MODEL_ARGS="${MODEL_ARGS},max_length=${MAX_LENGTH}"
fi

LIMIT_ARG=""
if [ "$DEBUG_SAMPLES" = "true" ]; then
    # Ensure --limit is at least NUM_GPUS to avoid empty ranks in multi-GPU
    DEBUG_LIMIT=$(( NUM_DEBUG_SAMPLES > NUM_GPUS ? NUM_DEBUG_SAMPLES : NUM_GPUS ))
    LIMIT_ARG="--limit ${DEBUG_LIMIT}"
    MODEL_ARGS="${MODEL_ARGS},debug_samples=True,num_debug_samples=${NUM_DEBUG_SAMPLES}"
fi

WANDB_ARGS=""
WANDB_SAMPLES_ARG=""
if [ "$WANDB_ENABLED" = "true" ]; then
    WANDB_ARGS="--wandb_args project=${WANDB_PROJECT},job_type=eval,name=${MODEL}-eval"
    if [ "$WANDB_LOG_SAMPLES" = "true" ]; then
        WANDB_SAMPLES_ARG="--wandb_log_samples"
    fi
fi

echo "Pre-downloading model code and datasets..."
python -c "
from datasets import load_dataset
import os, glob, shutil

# Pre-download model code by attempting to load the model (will fail but caches code)
print('Pre-downloading model code for: ${MODEL_PATH}')
try:
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.from_pretrained(
        '${MODEL_PATH}', device_map='cpu', trust_remote_code=True
    )
except Exception as e:
    print(f'Model pre-download triggered (expected error): {type(e).__name__}')

# Fix flash_attn import in cached model code.
# The model HF code imports flash_attn_varlen_func from transformers, but
# that symbol does not exist in transformers >= 4.48. Import from flash_attn
# directly instead.
cache_dir = os.path.expanduser('~/.cache/huggingface/modules/transformers_modules')
patched = False
for f in glob.glob(os.path.join(cache_dir, '**', 'modeling_llavaonevision1_5.py'), recursive=True):
    with open(f, 'r') as fp:
        content = fp.read()
    old = 'from transformers.modeling_flash_attention_utils import _flash_attention_forward, flash_attn_varlen_func'
    if old in content:
        new = ('from transformers.modeling_flash_attention_utils import _flash_attention_forward\n'
               '    from flash_attn import flash_attn_varlen_func')
        content = content.replace(old, new)
        with open(f, 'w') as fp:
            fp.write(content)
        pycache = os.path.join(os.path.dirname(f), '__pycache__')
        if os.path.exists(pycache):
            shutil.rmtree(pycache)
        print(f'Patched flash_attn import in {f}')
        patched = True
    else:
        print(f'No patch needed for {f}')
if not patched:
    print('WARNING: No modeling file found to patch!')

# Pre-download datasets to avoid multi-rank race conditions
tasks = '${TASKS}'.split(',')
task_to_dataset = {
    'ai2d': ('lmms-lab/ai2d', None),
    'mmstar': ('Lin-Chen/MMStar', None),
    'mme': ('lmms-lab/MME', None),
    'mmmu': ('lmms-lab/MMMU', None),
}
for task in tasks:
    task = task.strip()
    if task in task_to_dataset:
        ds_path, ds_name = task_to_dataset[task]
        print(f'Pre-downloading dataset: {ds_path}')
        load_dataset(ds_path, ds_name, trust_remote_code=True)
    else:
        print(f'No pre-download mapping for task: {task}, relying on runtime download')
print('Pre-download complete.')
"

echo "Running LM evaluation..."
accelerate launch -m lmms_eval \
    --model "${MODEL}" \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${RES_PATH}" \
    ${LIMIT_ARG} \
    ${WANDB_ARGS} \
    ${WANDB_SAMPLES_ARG}

echo "End Reached"
