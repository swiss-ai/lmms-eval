import io
import os
import re
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image

from lmms_eval.tasks._task_utils.zip_reader import ThreadLocalZipReader

DATASET_REPO_ID = "VisuLogic/VisuLogic"
OPTION_LETTERS = {"A", "B", "C", "D"}

_IMAGES_DIR = os.environ.get("VISULOGIC_IMAGES_DIR", "")
_DATASET_DIR = Path(snapshot_download(repo_id=DATASET_REPO_ID, repo_type="dataset", local_dir_use_symlinks=False)) if not _IMAGES_DIR else None
_IMAGES_ARCHIVE_PATH = _DATASET_DIR / "images.zip" if _DATASET_DIR else None
_ANSWER_PATTERNS = [
    re.compile(r"<answer>\s*\(?([A-D])\)?\s*</answer>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\\boxed\{\s*([A-D])\s*\}", re.IGNORECASE),
    re.compile(r"answer\s*(?:is|:|-)\s*\(?([A-D])\)?\b", re.IGNORECASE),
    re.compile(r"option\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"\(([A-D])\)", re.IGNORECASE),
]


def _get_images_archive_path():
    global _DATASET_DIR, _IMAGES_ARCHIVE_PATH
    if _IMAGES_ARCHIVE_PATH is None:
        _DATASET_DIR = Path(snapshot_download(repo_id=DATASET_REPO_ID, repo_type="dataset", local_dir_use_symlinks=False))
        _IMAGES_ARCHIVE_PATH = _DATASET_DIR / "images.zip"
    return _IMAGES_ARCHIVE_PATH


_IMAGE_ZIP = ThreadLocalZipReader(_get_images_archive_path)


def _extract_option_letter(text: str) -> str:
    normalized = str(text).strip()
    if not normalized:
        return ""

    for pattern in _ANSWER_PATTERNS:
        matches = pattern.findall(normalized)
        if matches:
            return matches[-1].upper()

    if len(normalized) <= 3:
        first_char = normalized.upper()[0]
        if first_char in OPTION_LETTERS:
            return first_char

    return ""


def visulogic_doc_to_visual(doc):
    image_path = str(doc.get("image_path", "")).strip().lstrip("./")
    if not image_path:
        return []

    if _IMAGES_DIR:
        # Prefer pre-extracted images from VISULOGIC_IMAGES_DIR (avoids zip corruption issues)
        filename = Path(image_path).name
        full_path = Path(_IMAGES_DIR) / filename
        if full_path.exists():
            return [Image.open(full_path).convert("RGB")]

    try:
        image_bytes = _IMAGE_ZIP.read(image_path)
    except KeyError as error:
        raise FileNotFoundError(f"Image not found in {DATASET_REPO_ID} archive: {image_path}") from error

    return [Image.open(io.BytesIO(image_bytes)).convert("RGB")]


def visulogic_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")

    question = str(doc.get("question", "")).strip()
    return f"{pre_prompt}{question}{post_prompt}"


def visulogic_doc_to_target(doc):
    return str(doc.get("label", "")).strip().upper()[:1]


def visulogic_process_results(doc, results):
    prediction = str(results[0]).strip() if results else ""
    predicted_letter = _extract_option_letter(prediction)
    target = visulogic_doc_to_target(doc)
    score = 1.0 if predicted_letter == target else 0.0
    return {"visulogic_acc": score}


def visulogic_aggregate_acc(items):
    if not items:
        return 0.0
    return sum(float(item) for item in items) / len(items)
