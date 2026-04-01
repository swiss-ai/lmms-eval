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

# Install updated packages inside the container
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

    # Install torch 2.10.0+cu129 from PyTorch index (prebuilt aarch64 wheel)
    pip install \
        "torch==2.10.0" "torchaudio==2.10.0" "torchvision==0.25.0" \
        --index-url https://download.pytorch.org/whl/cu129

    # Install vllm 0.18.1 (prebuilt aarch64 wheel, pins transformers>=4.56,<5)
    pip install "vllm==0.18.1"

    # Rebuild flash_attn against new torch to fix ABI mismatch
    MAX_JOBS=4 NVCC_THREADS=4 pip install --upgrade --no-build-isolation flash-attn

    # Upgrade transformers to latest compatible with vllm (4.57.x)
    pip install --upgrade "transformers>=4.56,<5"
    pip install --upgrade "accelerate>=0.29.1"

    # Downgrade numpy to <2 (NGC compiled modules expect numpy 1.x ABI)
    pip install "numpy<2"

    echo "=== After ==="
    python3 -c "import torch; print(f'torch: {torch.__version__}')"
    python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"
    python3 -c "import vllm; print(f'vllm: {vllm.__version__}')"
    python3 -c "from accelerate import Accelerator; print('accelerate OK')"
ENROOT_SCRIPT

# Export to squashfs
enroot export --output "$IMAGE_DIR/${IMAGE_NAME}.sqsh" build-env

# Cleanup
enroot remove -f build-env

echo "Container saved to: $IMAGE_DIR/${IMAGE_NAME}.sqsh"
echo "Done!"
