# Apertus Omni LMMS Eval Repro (SLURM, DP4)

This guide documents all steps needed to reproduce the Apertus Omni LMMS evaluation using:

- `lmms-eval/examples/slurm/run_apertus_omni_eval_dp4_n1b1.slurm`

## 1. Create a working directory and clone both repos

```bash
ROOT_DIR="/iopsstor/scratch/cscs/$USER/swissai/test"
mkdir -p "${ROOT_DIR}"
cd "${ROOT_DIR}"

git clone git@github.com:swiss-ai/lmms-eval.git
cd lmms-eval
git checkout apertus_adapter
cd "${ROOT_DIR}"

git clone git@github.com:swiss-ai/vllm-omni.git
cd vllm-omni
git checkout apertus_integration
git submodule update --init --recursive external/Emu3.5
cd "${ROOT_DIR}"
```

## 2. Copy the prepared environment bundle

Copy `/iopsstor/scratch/cscs/anunay/swissai/lmms-vllm-omni-final` into your working directory:

```bash
cp -r /iopsstor/scratch/cscs/anunay/swissai/lmms-vllm-omni-final "${ROOT_DIR}/"
```

After this, you should have:

- `${ROOT_DIR}/lmms-eval`
- `${ROOT_DIR}/vllm-omni`
- `${ROOT_DIR}/lmms-vllm-omni-final`

## 3. Fix paths in the LMMS eval TOML

Open the TOML that will be passed via `#SBATCH --environment` (for example: `${ROOT_DIR}/lmms-vllm-omni-final/lmms-eval.toml`) and ensure paths point to your current workdir.

At minimum, verify/update:

- workdir path: `/iopsstor/scratch/cscs/$USER/swissai/test`
- image path: the squash image under `${ROOT_DIR}/lmms-vllm-omni-final/`

If these are wrong, jobs will fail to start in the expected environment.

## 4. Verify the SLURM runner script settings

Script:

- `lmms-eval/examples/slurm/run_apertus_omni_eval_dp4_n1b1.slurm`

Check SBATCH fields in that script:

- `--environment` points to the TOML from step 3
- `--output` log path is valid
- `--error` log path is valid
- `--job-name` is set as desired

Also verify these variables are exactly set for this layout:

```bash
ROOT_DIR="/iopsstor/scratch/cscs/$USER/swissai/test"
VLLM_OMNI_DIR="${ROOT_DIR}/vllm-omni"
EVAL_DIR="${ROOT_DIR}/lmms-eval"
RES_PATH="${ROOT_DIR}/results/lmms_eval/apertus_omni_results/"
```

## 5. Submit the job

From `${ROOT_DIR}` run:

```bash
sbatch lmms-eval/examples/slurm/run_apertus_omni_eval_dp4_n1b1.slurm \
  --model-args "model_descriptor=/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF"
```

Full example with timeouts:

```bash
sbatch lmms-eval/examples/slurm/run_apertus_omni_eval_dp4_n1b1.slurm \
  --model-args "model_descriptor=/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF" \
  --init-timeout 3600 \
  --stage-init-timeout 3600
```
