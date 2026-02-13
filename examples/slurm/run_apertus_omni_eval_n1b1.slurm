#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=eval-apertus-omni
#SBATCH --environment=emu3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_apertus_omni_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_apertus_omni_%j.err

TASKS="mme" #,pope,gqa,vqav2_val,mmmu_test_group_img,mme,ai2d,ocrbench_v2,chartqa,docvqa,infovqa"
BATCH_SIZE=1
MODEL="apertus_omni"
OFFLINE_DATASETS="true"

# Required runtime args for apertus_omni adapter:
# - model_descriptor=<HF model id or local path>
# Optional:
# - tokenizer_path=<HF tokenizer id or local path>
# - stage_configs_path=/path/to/vllm_omni/model_executor/stage_configs/apertus.yaml
MODEL_ARGS=""
export PYTHONPATH="/iopsstor/scratch/cscs/$USER/swissai/vllm-omni:${PYTHONPATH}"
export APERTUS_OMNI_STAGE_CONFIG="/iopsstor/scratch/cscs/$USER/swissai/vllm-omni/vllm_omni/model_executor/stage_configs/apertus.yaml"

RES_PATH="/iopsstor/scratch/cscs/$USER/PDM/results/lmms_eval/apertus_omni_results"
EVAL_DIR=/iopsstor/scratch/cscs/$USER/swissai/lmms-eval
export HF_HOME="/iopsstor/scratch/cscs/anunay/swissai/hf"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model-args)
            MODEL_ARGS="$2"
            shift 2
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

if [ -z "${MODEL_ARGS}" ]; then
    echo "MODEL_ARGS is empty. Provide at least:"
    echo '  --model-args "model_descriptor=<your_apertus_model>"'
    exit 1
fi

# Automatically get number of visible GPUs
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Number of GPUs available: $NUM_GPUS"
echo "GPUS: $CUDA_VISIBLE_DEVICES"

# Print configuration
echo "========================================"
echo "MLLM Evaluation Configuration"
echo "========================================"
echo "Model:                ${MODEL}"
echo "Model args:           ${MODEL_ARGS}"
echo "Tasks:                ${TASKS}"
echo "Batch size(per GPU):  ${BATCH_SIZE}"
echo "Offline datasets:     ${OFFLINE_DATASETS}"
echo "Num GPU:              ${NUM_GPUS}"
echo "Output-Path:          ${RES_PATH}"
echo "Eval-Repo-Location:   ${EVAL_DIR}"
echo "========================================"
echo ""

# Run evaluation command
echo "Running LM evaluation..."
echo "### Setup packages & environment ###"

echo "Setting up lmms-eval package with uv..."
cd "$EVAL_DIR" || exit
uv pip install -e ".[all]" --system

echo "Installing task-specific dependencies..."
source "${EVAL_DIR}/examples/install_task_deps.sh" "${TASKS}" "${EVAL_DIR}"

# vLLM-Omni should run as a single front-end process; inherited DDP env vars can
# make EngineCore try torch.distributed handshakes and hang on TCPStore.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE

python3 -m lmms_eval \
    --model "${MODEL}" \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${RES_PATH}"

echo "End Reached"
