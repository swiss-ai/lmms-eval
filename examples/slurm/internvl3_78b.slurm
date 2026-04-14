#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=eval-internvl3
#SBATCH --environment=/users/mikhaika/.edf/lmms_eval_vllm.toml
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_internvl3_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_internvl3_%j.err

# Log base path
LOG_BASE=/iopsstor/scratch/cscs/$USER/lmms-eval/logs/eval_internvl3_$SLURM_JOB_ID

# Create a combined log capturing stdout and stderr
exec > >(tee -a ${LOG_BASE}.log)
exec 2> >(tee -a ${LOG_BASE}.log >&2)

TASKS="pmc_vqa,slake,vqa_rad,medqa,medmcqa,pubmedqa,mmlu_medical,path_vqa,path_mmu_test_tiny"
BATCH_SIZE=16
MODEL="vllm"
MODEL_ARGS="model=OpenGVLab/InternVL3-78B,tensor_parallel_size=4,gpu_memory_utilization=0.95,dtype=bfloat16,max_model_len=16384"
RES_PATH="/iopsstor/scratch/cscs/$USER/PDM/results/lmms_eval/internvl3_78b_results"
EVAL_DIR=/iopsstor/scratch/cscs/$USER/lmms-eval
HF_DATASETS_OFFLINE=0

# Automatically get number of available GPUS to configure accelerate properly
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
    --output_path "${RES_PATH}"

echo "End Reached"
