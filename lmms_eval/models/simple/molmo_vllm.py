from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.vllm import VLLM

# Molmo-1 (*-0924) was trained multi-task with the source dataset prepended to
# every prompt, so a bare question leaves it guessing the task: without a prefix
# it answers MCQs at close to the chance rate. The prefixes and the MCQ/VQA
# fallback mirror VLMEvalKit's vlmeval/vlm/molmo.py, which is the reference
# implementation for this model family. Molmo2 does not need this — it resolves
# its own task style inside chat_template.jinja.
TASK_PREFIXES = {
    "ai2d": "ai2_diagram:",
    "chartqa": "chart_qa:",
    "docvqa": "doc_qa:",
    "infovqa": "info_qa:",
    "ocrvqa": "ocr_vqa:",
    "scienceqa": "science_qa:",
    "textvqa": "text_vqa:",
    "coco_cap": "coco_captioning:",
    "tablevqa": "tabwmp_da:",
}

MCQ_TASKS = {
    "ai2d",
    "blink",
    "cv_bench",
    "mmbench",
    "mmmu",
    "mmstar",
    "mmvp",
    "muirbench",
    "realworldqa",
    "scienceqa",
    "seedbench",
    "vlmsareblind",
    "vlms_are_biased",
    "mathvista",
    "mathvision",
    "mathverse",
    "visulogic",
    "mmsi_bench",
    "mindcube",
    "embspatial",
    "erqa",
    "medxpertqa",
    "medmcqa",
    "medqa",
    "pmc_vqa",
    "path_mmu",
}

MCQ_PREFIX = "a_okvqa_mc:"
VQA_PREFIX = "vqa2:"


@register_model("molmo_vllm")
class MolmoVLLM(VLLM):
    """Molmo-1 through the vLLM backend, with the task prefix it was trained on."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logged_prefixes = set()

    def _prefix_for(self, task):
        name = str(task).lower()
        for key, prefix in TASK_PREFIXES.items():
            if key in name:
                return prefix
        return MCQ_PREFIX if any(k in name for k in MCQ_TASKS) else VQA_PREFIX

    def _format_context(self, contexts, task):
        prefix = self._prefix_for(task)
        if task not in self._logged_prefixes:
            self._logged_prefixes.add(task)
            from lmms_eval.utils import eval_logger

            eval_logger.info(f"molmo_vllm: task {task!r} -> prefix {prefix!r}")
        return f"{prefix} {contexts}"
