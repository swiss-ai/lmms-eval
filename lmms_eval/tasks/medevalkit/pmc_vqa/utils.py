import io
from typing import Any, Dict, List

from huggingface_hub import hf_hub_download
from PIL import Image

from lmms_eval.tasks._task_utils.zip_reader import ThreadLocalZipReader
from lmms_eval.tasks.medevalkit.eval_utils import agg_mean  # noqa: F401
from lmms_eval.tasks.medevalkit.eval_utils import no_image_doc_to_visual  # noqa: F401
from lmms_eval.tasks.medevalkit.eval_utils import judge_multi_choice

# ---------------------------------------------------------------------------
# Dataset images: read directly from images_2.zip (no extraction, no inode
# churn, no extract-race). One ZipFile handle per worker thread.
# ---------------------------------------------------------------------------

CHOICE_LETTERS = ["A", "B", "C", "D"]


def _get_archive_path():
    return hf_hub_download(
        repo_id="RadGenome/PMC-VQA",
        filename="images_2.zip",
        repo_type="dataset",
    )


_IMAGE_ZIP = ThreadLocalZipReader(_get_archive_path)


# ---------------------------------------------------------------------------
# lmms-eval interface
# ---------------------------------------------------------------------------


def pmc_vqa_doc_to_visual(doc: Dict[str, Any]):
    image_bytes = _IMAGE_ZIP.read(f"figures/{doc['Figure_path']}")
    return [Image.open(io.BytesIO(image_bytes)).convert("RGB")]


def pmc_vqa_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any] = None,
) -> str:
    question = doc["Question"].strip()
    choices = []
    for letter in CHOICE_LETTERS:
        choice_text = doc[f"Choice {letter}"].strip()
        # Choices may already include the letter prefix like " A:..."
        if not choice_text.startswith(letter):
            choice_text = f"{letter}. {choice_text}"
        choices.append(choice_text)
    options = "\n".join(choices)
    return f"Question: {question}\n" f"Options:\n{options}\n" "Answer with the option's letter from the given choices directly."


def pmc_vqa_doc_to_target(doc: Dict[str, Any]) -> str:
    return doc["Answer"].strip().upper()


def pmc_vqa_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, float]:
    raw_response = result[0] if result else ""
    choices = [doc[f"Choice {letter}"].strip() for letter in CHOICE_LETTERS]
    answer = doc["Answer"].strip()
    correct = judge_multi_choice(choices, answer, raw_response)
    return {"accuracy": float(correct)}
