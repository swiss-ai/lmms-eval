#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=eval-apertus-omni-dp4
#SBATCH --environment=/iopsstor/scratch/cscs/anunay/swissai/lmms-vllm-omni-final/lmms-eval.toml
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --reservation=PA-2338-RL
#SBATCH --output=/iopsstor/scratch/cscs/%u/swissai/lmms-eval/logs/eval_apertus_omni_dp4_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/swissai/lmms-eval/logs/eval_apertus_omni_dp4_%j.err

set -euo pipefail
# export VLLM_LOGGING_LEVEL=DEBUG
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,COLL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ENSURE THESE VARIABLES ARE SET IN THE ENVIRONMENT OR PROVIDE DEFAULTS
ROOT_DIR="/iopsstor/scratch/cscs/$USER/swissai/"
VLLM_OMNI_DIR="${ROOT_DIR}/vllm-omni"
EVAL_DIR="${ROOT_DIR}/lmms-eval"
RES_PATH="${ROOT_DIR}/results/lmms_eval/apertus_omni_results/"

# TASKS="pope,gqa,vqav2_val,mmmu_test_group_img,mme,ai2d,ocrbench_v2,chartqa,docvqa,infovqa"
TASKS="vqa_rad,path_vqa" 
BATCH_SIZE=4096
MODEL="apertus_omni"
OFFLINE_DATASETS="true"
MODEL_ARGS=""
GEN_KWARGS="${GEN_KWARGS:-}"
LIMIT="${LIMIT:-}"
VISION_TOKENIZER_DEVICE="${VISION_TOKENIZER_DEVICE:-cuda}"
INIT_TIMEOUT="${INIT_TIMEOUT:-1800}"
STAGE_INIT_TIMEOUT="${STAGE_INIT_TIMEOUT:-1800}"

TOKENIZER_PATH="${TOKENIZER_PATH:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_instruct}"
STAGE_CONFIG_PATH="${STAGE_CONFIG_PATH:-${VLLM_OMNI_DIR}/vllm_omni/model_executor/stage_configs/apertus_dp4.yaml}"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/hf}"
export NLTK_DATA="${NLTK_DATA:-${ROOT_DIR}/nltk_data}"


export PYTHONPATH="${VLLM_OMNI_DIR}:${PYTHONPATH:-}"
export APERTUS_OMNI_STAGE_CONFIG="${STAGE_CONFIG_PATH}"
export VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_DIR="${ROOT_DIR}/cache/image_tokens/"
export VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_PRELOAD_TO_RAM=true
export VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_RAM_CACHE_KIB=$((5*1024*1024))
export VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_DISK_CACHE_KIB=$((5*1024*1024))
export VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_DISK_MMAP_BYTES=$((5*1024*1024*1024))
export VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_BUSY_TIMEOUT_MS=5000
export VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_DISK_SYNCHRONOUS=NORMAL
# export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
# export TOKENIZERS_PARALLELISM=false

mkdir -p "${VLLM_OMNI_APERTUS_IMAGE_TOKEN_CACHE_DIR}" "${NLTK_DATA}" "${RES_PATH}" "${EVAL_DIR}/logs"

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --gen-kwargs)
            GEN_KWARGS="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --vision-tokenizer-device)
            VISION_TOKENIZER_DEVICE="$2"
            shift 2
            ;;
        --init-timeout)
            INIT_TIMEOUT="$2"
            shift 2
            ;;
        --stage-init-timeout)
            STAGE_INIT_TIMEOUT="$2"
            shift 2
            ;;
        --stage-config)
            STAGE_CONFIG_PATH="$2"
            export APERTUS_OMNI_STAGE_CONFIG="${STAGE_CONFIG_PATH}"
            shift 2
            ;;
        --output-path)
            RES_PATH="$2"
            mkdir -p "${RES_PATH}"
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

if [[ "${MODEL_ARGS}" != *"tokenizer_path="* ]]; then
    MODEL_ARGS="${MODEL_ARGS},tokenizer_path=${TOKENIZER_PATH}"
fi
if [[ "${MODEL_ARGS}" != *"tokenizer="* ]]; then
    MODEL_ARGS="${MODEL_ARGS},tokenizer=${TOKENIZER_PATH}"
fi
if [[ "${MODEL_ARGS}" != *"vision_tokenizer_device="* ]]; then
    MODEL_ARGS="${MODEL_ARGS},vision_tokenizer_device=${VISION_TOKENIZER_DEVICE}"
fi
if [[ "${MODEL_ARGS}" != *"init_timeout="* ]]; then
    MODEL_ARGS="${MODEL_ARGS},init_timeout=${INIT_TIMEOUT}"
fi
if [[ "${MODEL_ARGS}" != *"stage_init_timeout="* ]]; then
    MODEL_ARGS="${MODEL_ARGS},stage_init_timeout=${STAGE_INIT_TIMEOUT}"
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
else
    NUM_GPUS=$(nvidia-smi -L | wc -l)
fi

if [ "${NUM_GPUS}" -lt 4 ]; then
    echo "Need at least 4 visible GPUs for apertus_dp4.yaml, found ${NUM_GPUS}."
    exit 1
fi

if [ "$OFFLINE_DATASETS" = "true" ]; then
    export HF_DATASETS_OFFLINE=1
else
    unset HF_DATASETS_OFFLINE
fi

echo "========================================"
echo "Apertus Omni DP4 Evaluation (1 Process)"
echo "========================================"
echo "Model:                ${MODEL}"
echo "Model args:           ${MODEL_ARGS}"
echo "Tasks:                ${TASKS}"
echo "Batch size:           ${BATCH_SIZE}"
echo "Gen kwargs:           ${GEN_KWARGS:-<task-defaults>}"
echo "Limit:                ${LIMIT:-<none>}"
echo "Vision tok device:    ${VISION_TOKENIZER_DEVICE}"
echo "Offline datasets:     ${OFFLINE_DATASETS}"
echo "Visible GPUs:         ${NUM_GPUS}"
echo "Init timeout (s):     ${INIT_TIMEOUT}"
echo "Stage init timeout(s):${STAGE_INIT_TIMEOUT}"
echo "Stage config:         ${APERTUS_OMNI_STAGE_CONFIG}"
echo "Output path:          ${RES_PATH}"
echo "========================================"

cd "${VLLM_OMNI_DIR}" || exit 1
python3 -m pip install -e . --no-build-isolation --no-deps

cd "${EVAL_DIR}" || exit 1
python3 -m pip install -e ".[all]"
source "${EVAL_DIR}/examples/install_task_deps.sh" "${TASKS}" "${EVAL_DIR}"
python3 -m pip install ipdb jieba distance apted lxml zss nltk datasets sacrebleu Polygon3
python3 -c "import nltk, os; nltk.download('wordnet', download_dir=os.environ['NLTK_DATA'], quiet=True)"

EXTRA_ARGS=()
if [ -n "${GEN_KWARGS}" ]; then
    EXTRA_ARGS+=(--gen_kwargs "${GEN_KWARGS}")
fi
if [ -n "${LIMIT}" ]; then
    EXTRA_ARGS+=(--limit "${LIMIT}")
fi

python3 -m lmms_eval \
    --model "${MODEL}" \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    "${EXTRA_ARGS[@]}" \
    --output_path "${RES_PATH}"

echo "End Reached"
