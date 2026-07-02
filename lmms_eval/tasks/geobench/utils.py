import os
import re
from collections import defaultdict
from io import BytesIO

from loguru import logger as eval_logger
from PIL import Image
from pycocoevalcap.eval import Bleu, Cider, Meteor, Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO


def _base_dir():
    return os.environ.get("GEOBENCH_DIR", "")


def _open_image(path_or_dict):
    """Open a PIL image from either a file path or a HF-style {'bytes':..,'path':..} dict."""
    if isinstance(path_or_dict, dict):
        return Image.open(BytesIO(path_or_dict["bytes"])).convert("RGB")
    return Image.open(path_or_dict).convert("RGB")


# ---------------------------------------------------------------------------
# Single split — embedded images, MCQ A–E
# ---------------------------------------------------------------------------


def geobench_doc_to_visual(doc):
    img = doc["image"]
    if isinstance(img, dict) and img.get("bytes"):
        return [_open_image(img)]
    return [_open_image(os.path.join(_base_dir(), doc["image_path"]))]


def geobench_single_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    question = doc["prompts"][0]
    options = doc["options"]
    post = kwargs.get("post_prompt", "\nAnswer with the option letter only (A, B, C, D, or E).")
    return f"{question}\n{options}{post}"


def geobench_single_doc_to_target(doc):
    return doc["ground_truth_option"].strip().upper()


def _extract_option_letter(pred: str) -> str:
    match = re.search(r"\b([A-E])\b", pred.strip().upper())
    if match:
        return match.group(1)
    if pred and pred[0].upper() in "ABCDE":
        return pred[0].upper()
    return ""


def geobench_single_process_results(doc, results):
    pred = _extract_option_letter(results[0])
    gold = doc["ground_truth_option"].strip().upper()
    correct = int(pred == gold)
    task = doc.get("task", "unknown")
    return {
        "single_accuracy": correct,
        "per_task": {"task": task, "correct": correct},
    }


def geobench_single_aggregate(results):
    return sum(results) / len(results) if results else 0.0


def geobench_per_task_aggregate(results):
    """Log per-task breakdown; return macro-averaged accuracy across tasks."""
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r["correct"])
    per_task = {t: sum(v) / len(v) for t, v in sorted(by_task.items())}
    lines = "\n".join(f"  {t}: {acc:.3f} ({len(by_task[t])} samples)" for t, acc in per_task.items())
    eval_logger.info(f"GEOBench per-task accuracy:\n{lines}")
    return sum(per_task.values()) / len(per_task) if per_task else 0.0


# ---------------------------------------------------------------------------
# Temporal split — two images (before/after), MCQ A–E
# ---------------------------------------------------------------------------


def geobench_temporal_doc_to_visual(doc):
    base = _base_dir()
    paths = doc["image_path"]  # list of 2 relative paths
    return [_open_image(os.path.join(base, p)) for p in paths]


def geobench_temporal_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "You are shown two satellite images from different time periods (before and after).\n")
    post = kwargs.get("post_prompt", "\nAnswer with the option letter only (A, B, C, D, or E).")
    return f"{pre}{doc['prompts'][0]}\n{doc['options']}{post}"


def geobench_temporal_doc_to_target(doc):
    return doc["ground_truth_option"].strip().upper()


def geobench_temporal_process_results(doc, results):
    pred = _extract_option_letter(results[0])
    gold = doc["ground_truth_option"].strip().upper()
    correct = int(pred == gold)
    task = doc.get("task", "unknown")
    return {
        "temporal_accuracy": correct,
        "temporal_per_task": {"task": task, "correct": correct},
    }


def geobench_temporal_aggregate(results):
    return sum(results) / len(results) if results else 0.0


def geobench_temporal_per_task_aggregate(results):
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r["correct"])
    per_task = {t: sum(v) / len(v) for t, v in sorted(by_task.items())}
    lines = "\n".join(f"  {t}: {acc:.3f} ({len(by_task[t])} samples)" for t, acc in per_task.items())
    eval_logger.info(f"GEOBench Temporal per-task accuracy:\n{lines}")
    return sum(per_task.values()) / len(per_task) if per_task else 0.0


# ---------------------------------------------------------------------------
# Captioning split — free-text, CIDEr/BLEU/METEOR/ROUGE
# ---------------------------------------------------------------------------

_CAP_METRICS = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]


def geobench_cap_doc_to_visual(doc):
    return [_open_image(os.path.join(_base_dir(), doc["image_path"]))]


def geobench_cap_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    return kwargs.get("prompt", doc["prompts"][0])


def geobench_cap_doc_to_target(doc):
    return doc["ground_truth"].strip()


def geobench_cap_process_results(doc, results):
    data = {"pred": results[0], "gt": doc["ground_truth"].strip(), "id": int(doc["question_id"])}
    return {f"cap_{m}": data for m in _CAP_METRICS}


def _make_scorer(metric):
    if metric.startswith("Bleu"):
        return Bleu(4), metric
    if metric == "METEOR":
        return Meteor(), "METEOR"
    if metric == "ROUGE_L":
        return Rouge(), "ROUGE_L"
    if metric == "CIDEr":
        return Cider(), "CIDEr"
    raise ValueError(f"Unknown metric: {metric}")


def _cap_aggregate(results, metric):
    dataset = {"annotations": [], "images": []}
    stored = []
    for r in results:
        rid = r["id"]
        stored.append({"image_id": rid, "caption": r["pred"]})
        dataset["annotations"].append({"image_id": rid, "caption": r["gt"], "id": rid})
        dataset["images"].append({"id": rid})
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    coco_res = coco.loadRes(stored)
    img_ids = list({r["id"] for r in results})
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize({i: coco.imgToAnns[i] for i in img_ids})
    res = tokenizer.tokenize({i: coco_res.imgToAnns[i] for i in img_ids})
    scorer, name = _make_scorer(metric)
    score, _ = scorer.compute_score(gts, res)
    if isinstance(score, list):
        score = score[int(name.split("_")[-1]) - 1]
    return score


def geobench_cap_bleu1(r):
    return _cap_aggregate(r, "Bleu_1")


def geobench_cap_bleu2(r):
    return _cap_aggregate(r, "Bleu_2")


def geobench_cap_bleu3(r):
    return _cap_aggregate(r, "Bleu_3")


def geobench_cap_bleu4(r):
    return _cap_aggregate(r, "Bleu_4")


def geobench_cap_meteor(r):
    return _cap_aggregate(r, "METEOR")


def geobench_cap_rouge(r):
    return _cap_aggregate(r, "ROUGE_L")


def geobench_cap_cider(r):
    return _cap_aggregate(r, "CIDEr")


# ---------------------------------------------------------------------------
# Ref-Det split — referring expression detection, Acc@0.5 IoU
# ---------------------------------------------------------------------------


def geobench_ref_doc_to_visual(doc):
    return [_open_image(os.path.join(_base_dir(), doc["image_path"]))]


def geobench_ref_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    post = kwargs.get("post_prompt", "\nProvide the bounding box as [x1, y1, x2, y2] with coordinates normalized to [0, 1].")
    return f"{doc['prompts'][0]}{post}"


def geobench_ref_doc_to_target(doc):
    return str(doc["ground_truth"])


def _polygons_to_bbox_01(polygons, img_size):
    """Convert list-of-polygons to [x1,y1,x2,y2] normalized 0-1."""
    all_x, all_y = [], []
    for poly in polygons:
        for pt in poly:
            all_x.append(pt[0])
            all_y.append(pt[1])
    if not all_x:
        return None
    return [min(all_x) / img_size, min(all_y) / img_size, max(all_x) / img_size, max(all_y) / img_size]


def _parse_pred_bbox_01(s):
    nums = re.findall(r"\d+\.?\d*", s)
    if len(nums) < 4:
        return None
    vals = [float(n) for n in nums[:4]]
    # If values look like 0-100 scale, normalize to 0-1
    if max(vals) > 1.5:
        vals = [v / 100.0 for v in vals]
    return vals


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def geobench_ref_process_results(doc, results):
    img_size = doc.get("image_size", 1024)
    gt_box = _polygons_to_bbox_01(doc["ground_truth"], img_size)
    pred_box = _parse_pred_bbox_01(results[0])
    if gt_box is None or pred_box is None:
        acc = 0
    else:
        acc = int(_iou(gt_box, pred_box) >= 0.5)
    return {"ref_acc50": acc}


def geobench_ref_aggregate(results):
    return sum(results) / len(results) if results else 0.0
