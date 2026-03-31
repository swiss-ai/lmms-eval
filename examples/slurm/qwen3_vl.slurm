#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=eval-qwen3_vl
#SBATCH --environment=/users/mikhaika/.edf/lmms_eval_vllm.toml
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_qwen3_vl_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_qwen3_vl_%j.err

# Log base path
LOG_BASE=/iopsstor/scratch/cscs/$USER/lmms-eval/logs/eval_qwen3_vl_$SLURM_JOB_ID

# Create a combined log capturing stdout and stderr
exec > >(tee -a ${LOG_BASE}.log)
exec 2> >(tee -a ${LOG_BASE}.log >&2)

TASKS="path_mmu_test_tiny" #,pmc_vqa,slake,medqa,pubmedqa,mmlu_medical,vqa_rad" # medmcqa, path_vqa
BATCH_SIZE=64
MODEL="vllm"
MODEL_ARGS="model=Qwen/Qwen3-VL-8B-Instruct,tensor_parallel_size=4,gpu_memory_utilization=0.9,dtype=bfloat16,enforce_eager=True"
RES_PATH="/iopsstor/scratch/cscs/$USER/PDM/results/lmms_eval/${MODEL}_results"
EVAL_DIR=/iopsstor/scratch/cscs/$USER/lmms-eval
HF_DATASETS_OFFLINE=0

# Automatically get number of available GPUS to configure accelerate properöy
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Number of GPUs available: $NUM_GPUS"
echo "GPUS: $CUDA_VISIBLE_DEVICES"

# Print configuration
echo "========================================"
echo "MLLM Evaluation Configuration"
echo "========================================"
echo "Model:                ${MODEL}"
echo "Tasks:                ${TASKS}"
echo "Batch size(per GPU):  ${BATCH_SIZE}"
echo "Offline datasets:     ${HF_DATASETS_OFFLINE}"
echo "Num GPU:              ${NUM_GPUS}"
echo "Output-Path:          ${RES_PATH}"
echo "Eval-Repo-Location:   ${EVAL_DIR}"
echo "========================================"
echo ""

# Run evaluation command
echo "Running LM evaluation..."

echo "###Setup Packages & environment#####"
echo ""
echo "<<Current Packages>>"
pip list
echo ""

echo "📦 Setting up lmms-eval package..."
cd "$EVAL_DIR" || exit
pip uninstall jupyterlab -y  # Uninstall conflicting package
unset PIP_CONSTRAINT
pip install -e .

echo "📦 Installing task-specific dependencies..."
source "${EVAL_DIR}/examples/install_task_deps.sh" "${TASKS}" "${EVAL_DIR}"

echo "<<New Packages>>"
pip list
echo ""

# Construct model_args parameter if MODEL_ARGS is non-empty
MODEL_ARGS_PARAM=""
if [ -n "${MODEL_ARGS}" ]; then
    MODEL_ARGS_PARAM="--model_args ${MODEL_ARGS}"
fi

python3 -m lmms_eval \
    --model "${MODEL}" \
    ${MODEL_ARGS_PARAM} \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${RES_PATH}" \
    --gen_kwargs "temperature=0.7,top_p=0.8,top_k=20,max_new_tokens=4096,repetition_penalty=1.0"

echo "End Reached"
