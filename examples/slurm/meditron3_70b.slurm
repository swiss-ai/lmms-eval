#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=eval-meditron3
#SBATCH --environment=/users/mikhaika/.edf/lmms_eval_vllm.toml
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_meditron3_70b_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_meditron3_70b_%j.err
LOG_BASE=/iopsstor/scratch/cscs/$USER/lmms-eval/logs/eval_meditron3_70b_$SLURM_JOB_ID
exec > >(tee -a ${LOG_BASE}.log)
exec 2> >(tee -a ${LOG_BASE}.log >&2)

TASKS="medqa,medmcqa,pubmedqa,mmlu_medical"
BATCH_SIZE=16
MODEL="vllm"
MODEL_ARGS="model=OpenMeditron/Meditron3-70B,tensor_parallel_size=4,gpu_memory_utilization=0.95,dtype=bfloat16,max_model_len=8192"
RES_PATH="/iopsstor/scratch/cscs/$USER/PDM/results/lmms_eval/meditron3_70b_results"
EVAL_DIR=/iopsstor/scratch/cscs/$USER/lmms-eval

cd "$EVAL_DIR" || exit
pip uninstall jupyterlab -y
unset PIP_CONSTRAINT
pip install -e .

python3 -m lmms_eval \
    --model "${MODEL}" \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${RES_PATH}"

echo "End Reached"
