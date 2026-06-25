import os
import re
from collections import defaultdict

from PIL import Image
from pycocoevalcap.eval import Bleu, Cider, Meteor, Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

VRSBENCH_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]


def _img_dir():
    explicit = os.environ.get("VRSBENCH_IMG_DIR", "")
    if explicit:
        return explicit
    base = os.environ.get("VRSBENCH_DIR", "")
    if base:
        return os.path.join(base, "Images_val")
    raise RuntimeError("Set VRSBENCH_DIR (VRSBench base dir) or VRSBENCH_IMG_DIR (Images_val dir)")


# ---------------------------------------------------------------------------
# Shared: load image
# ---------------------------------------------------------------------------

def vrsbench_doc_to_visual(doc):
    img_path = os.path.join(_img_dir(), doc["image_id"])
    return [Image.open(img_path).convert("RGB")]


# ---------------------------------------------------------------------------
# VQA task
# ---------------------------------------------------------------------------

def vrsbench_vqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre  = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "\nAnswer the question using a single word or phrase.")
    return f"{pre}{doc['question']}{post}"


def vrsbench_vqa_doc_to_target(doc):
    return doc["ground_truth"]


def _normalize_answer(s):
    return re.sub(r"[^\w\s]", "", s).strip().lower()


def vrsbench_vqa_process_results(doc, results):
    pred  = _normalize_answer(results[0])
    gold  = _normalize_answer(doc["ground_truth"])
    correct = int(pred == gold)
    qtype = doc.get("type", "unknown")
    return {"vqa_accuracy": correct, f"vqa_{qtype.replace(' ', '_')}": correct}


def vrsbench_vqa_aggregate(results):
    return sum(results) / len(results) if results else 0.0


# ---------------------------------------------------------------------------
# Captioning task
# ---------------------------------------------------------------------------

def vrsbench_cap_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    return kwargs.get("prompt", "Describe the image in detail.")


def vrsbench_cap_doc_to_target(doc):
    return doc["ground_truth"]


def vrsbench_cap_process_results(doc, results):
    pred = results[0]
    qid  = int(doc["question_id"])
    data = {"pred": pred, "gt": doc["ground_truth"], "question_id": qid}
    return {f"cap_{m}": data for m in VRSBENCH_METRICS}


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
    stored  = []
    for r in results:
        qid = r["question_id"]
        stored.append({"image_id": qid, "caption": r["pred"]})
        dataset["annotations"].append({"image_id": qid, "caption": r["gt"], "id": qid})
        dataset["images"].append({"id": qid})

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    coco_res  = coco.loadRes(stored)
    imgIds    = list({r["question_id"] for r in results})

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize({i: coco.imgToAnns[i] for i in imgIds})
    res = tokenizer.tokenize({i: coco_res.imgToAnns[i] for i in imgIds})

    scorer, name = _make_scorer(metric)
    score, _ = scorer.compute_score(gts, res)
    if isinstance(score, list):
        n = int(name.split("_")[-1])
        score = score[n - 1]
    return score


def vrsbench_cap_bleu1(results):  return _cap_aggregate(results, "Bleu_1")
def vrsbench_cap_bleu2(results):  return _cap_aggregate(results, "Bleu_2")
def vrsbench_cap_bleu3(results):  return _cap_aggregate(results, "Bleu_3")
def vrsbench_cap_bleu4(results):  return _cap_aggregate(results, "Bleu_4")
def vrsbench_cap_meteor(results): return _cap_aggregate(results, "METEOR")
def vrsbench_cap_rouge(results):  return _cap_aggregate(results, "ROUGE_L")
def vrsbench_cap_cider(results):  return _cap_aggregate(results, "CIDEr")


# ---------------------------------------------------------------------------
# Referring expression task
# ---------------------------------------------------------------------------

def vrsbench_ref_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre  = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "\nProvide the bounding box of the described object as {<x1><y1><x2><y2>}, where coordinates are integers from 0 to 100.")
    return f"{pre}{doc['question']}{post}"


def vrsbench_ref_doc_to_target(doc):
    return doc["ground_truth"]


def _parse_bbox(s):
    """Extract up to 4 numbers from any bbox-like string."""
    nums = re.findall(r"\d+\.?\d*", s)
    if len(nums) >= 4:
        return [float(n) for n in nums[:4]]
    return None


def _iou(a, b):
    """IoU for two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def vrsbench_ref_process_results(doc, results):
    pred_box = _parse_bbox(results[0])
    gt_box   = _parse_bbox(doc["ground_truth"])
    if pred_box is None or gt_box is None:
        acc = 0
    else:
        acc = int(_iou(pred_box, gt_box) >= 0.5)
    return {"ref_acc50": acc}


def vrsbench_ref_aggregate(results):
    return sum(results) / len(results) if results else 0.0


