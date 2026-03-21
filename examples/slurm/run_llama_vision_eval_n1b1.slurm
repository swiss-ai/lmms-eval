#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=eval-llama-vision
#SBATCH --environment=emu3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/lmms-eval/logs/eval_llama_vision_%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/lmms-eval/logs/eval_llama_vision_%j.err

# Unset SSL_CERT_FILE to avoid FileNotFoundError inside NGC container
# (host path /etc/ssl/ca-bundle.pem doesn't exist in container)
unset SSL_CERT_FILE

TASKS="mmmu_val_group_img,vqav2_val"
BATCH_SIZE=1
MODEL="llama_vision"
MODEL_PATH=""        # Required: path to HF model or HF hub identifier (e.g. "meta-llama/Llama-3.2-11B-Vision-Instruct")
ATTN_IMPLEMENTATION="sdpa"
RES_PATH="/capstor/scratch/cscs/$USER/PDM/results/lmms_eval/llama_vision_results"
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

LIMIT_ARG=""
if [ "$DEBUG_SAMPLES" = "true" ]; then
    LIMIT_ARG="--limit ${NUM_DEBUG_SAMPLES}"
fi

WANDB_ARGS=""
WANDB_SAMPLES_ARG=""
if [ "$WANDB_ENABLED" = "true" ]; then
    WANDB_ARGS="--wandb_args project=${WANDB_PROJECT},job_type=eval,name=${MODEL}-eval"
    if [ "$WANDB_LOG_SAMPLES" = "true" ]; then
        WANDB_SAMPLES_ARG="--wandb_log_samples"
    fi
fi

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
