#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=build-container
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/lmms-eval/logs/build_container_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/lmms-eval/logs/build_container_%j.err

# Log base path
LOG_BASE=/iopsstor/scratch/cscs/$USER/lmms-eval/logs/build_container_$SLURM_JOB_ID

# Create a combined log capturing stdout and stderr
exec > >(tee -a ${LOG_BASE}.log)
exec 2> >(tee -a ${LOG_BASE}.log >&2)

set -ex

IMAGE_NAME="lmms-eval-vllm"
IMAGE_DIR="/iopsstor/scratch/cscs/$USER/images"
BASE_SQSH="/iopsstor/scratch/cscs/dealmeih/images/ngc-transformers+25.05.sqsh"

mkdir -p "$IMAGE_DIR"

# Create enroot container from the working base image
enroot remove -f build-env 2>/dev/null || true
enroot create --name build-env "$BASE_SQSH"

# Install vllm + upgrade transformers (keep torch/NCCL/CUDA intact)
enroot start --root --rw build-env bash <<'ENROOT_SCRIPT'
    set -ex
    # Remove NGC constraint file that pins incompatible package versions
    if [ -n "$PIP_CONSTRAINT" ]; then
        echo "Removing constraint file: $PIP_CONSTRAINT"
        rm -f "$PIP_CONSTRAINT" 2>/dev/null || true
        unset PIP_CONSTRAINT
    fi

    echo "=== Before ==="
    python3 -c "import torch; print(f'torch: {torch.__version__}')"
    python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"

    pip install --upgrade --force-reinstall --no-deps "transformers>=4.56,<5"
    pip install --upgrade "accelerate>=0.29.1" "numpy<2"

    # Install vllm v0.18.0 from source, preserving existing NGC torch
    git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm.git /tmp/vllm
    cd /tmp/vllm
    python3 use_existing_torch.py
    pip install -r requirements/build.txt
    # Limit parallel compile jobs to avoid OOM
    MAX_JOBS=50 pip install --no-build-isolation .
    cd /
    rm -rf /tmp/vllm

    # Downgrade numpy after vllm build (NGC modules compiled against numpy 1.x)
    pip install "numpy<2"

    echo "=== After ==="
    python3 -c "import torch; print(f'torch: {torch.__version__}')"
    python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"
    python3 -c "from accelerate import Accelerator; print('accelerate OK')"
    pip show vllm | grep Version
ENROOT_SCRIPT

# Export to squashfs
enroot export --output "$IMAGE_DIR/${IMAGE_NAME}.sqsh" build-env

# Cleanup
enroot remove -f build-env

echo "Container saved to: $IMAGE_DIR/${IMAGE_NAME}.sqsh"
echo "Done!"
