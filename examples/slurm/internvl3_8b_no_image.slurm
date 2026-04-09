#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=iv3-no-image
#SBATCH --environment=/users/mikhaika/.edf/lmms_eval_vllm.toml
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_internvl3_no_image_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_internvl3_no_image_%j.err
LOG_BASE=/iopsstor/scratch/cscs/$USER/lmms-eval/logs/eval_internvl3_no_image_$SLURM_JOB_ID
exec > >(tee -a ${LOG_BASE}.log)
exec 2> >(tee -a ${LOG_BASE}.log >&2)
cd /iopsstor/scratch/cscs/$USER/lmms-eval
pip uninstall jupyterlab -y
unset PIP_CONSTRAINT
pip install -e .
torchrun --nproc_per_node=4 -m lmms_eval \
    --model vllm \
    --model_args model=OpenGVLab/InternVL3-8B,tensor_parallel_size=1,gpu_memory_utilization=0.9,dtype=bfloat16,max_model_len=8192 \
    --tasks pmc_vqa_no_image,slake_no_image,path_vqa_no_image,vqa_rad_no_image,path_mmu_test_tiny_no_image \
    --batch_size 64 \
    --output_path /iopsstor/scratch/cscs/$USER/PDM/results/lmms_eval/internvl3_8b_no_image_results
