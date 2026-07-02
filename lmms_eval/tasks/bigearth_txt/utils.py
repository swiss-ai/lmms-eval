"""
BigEarthNet.txt task utilities.

Question/answer data is loaded from HF Hub (BIFOLD-BigEarthNetv2-0/BigEarthNet.txt).
Image loading supports two backends (checked in order):

1. LMDB (preferred): Set BIGEARTH_LMDB_DIR to a rico-hdl-encoded LMDB directory.
   Build with: rico-hdl bigearthnet --bigearthnet-s2-dir <S2_DIR> --target-dir <LMDB_DIR>

2. Raw S2 TIF fallback: Set BIGEARTH_S2_DIR to the BigEarthNet-S2 root directory.
   Expected layout: <S2_DIR>/<acquisition>/<patch_id>/<patch_id>_B0{2,3,4}.tif
"""

import os
import re

import numpy as np
from PIL import Image
from pycocoevalcap.eval import Bleu, Cider, Meteor, Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

# ---------------------------------------------------------------------------
# Paths (override with env vars)
# ---------------------------------------------------------------------------

_DEFAULT_LMDB = ""  # set via BIGEARTH_LMDB_DIR env var

# Band statistics from BigEarthNet v2 train split (after 120x120 resize)
_MEANS = {"B04": 588.4096, "B03": 614.0557, "B02": 438.3721}
_STDS = {"B04": 684.5688, "B03": 603.2968, "B02": 607.0269}

# Shared LMDB environment (opened once per process)
_lmdb_env = None


def _normalize_band(arr: np.ndarray, band: str) -> np.ndarray:
    """Clip to ±2 std and scale to uint8."""
    arr = arr.astype(np.float32)
    lo = _MEANS[band] - 2 * _STDS[band]
    hi = _MEANS[band] + 2 * _STDS[band]
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _get_lmdb_env():
    global _lmdb_env
    if _lmdb_env is None:
        import lmdb

        lmdb_dir = os.environ.get("BIGEARTH_LMDB_DIR", _DEFAULT_LMDB)
        _lmdb_env = lmdb.open(
            str(lmdb_dir),
            readonly=True,
            lock=False,
            meminit=False,
            readahead=False,
            map_size=8 * 1024**3,
        )
    return _lmdb_env


def _load_rgb_from_lmdb(patch_id: str) -> Image.Image:
    """Load Sentinel-2 B04/B03/B02 from LMDB and return a PIL RGB image."""
    import torch
    from safetensors.numpy import load as safetensor_load

    env = _get_lmdb_env()
    with env.begin(write=False, buffers=True) as txn:
        byte_data = txn.get(patch_id.encode())
    if byte_data is None:
        raise KeyError(f"patch_id {patch_id!r} not found in LMDB")

    data = safetensor_load(bytes(byte_data))

    bands = []
    for band in ("B04", "B03", "B02"):
        arr = np.array(data[band], dtype=np.float32)
        if arr.shape != (120, 120):
            arr = (
                torch.nn.functional.interpolate(
                    torch.tensor(arr).unsqueeze(0).unsqueeze(0),
                    (120, 120),
                    mode="nearest",
                )
                .squeeze()
                .numpy()
            )
        bands.append(_normalize_band(arr, band))

    return Image.fromarray(np.stack(bands, axis=-1), mode="RGB")


def _load_rgb_from_tif(patch_id: str) -> Image.Image:
    """Load Sentinel-2 B04/B03/B02 from raw GeoTIF files (fallback path)."""
    s2_dir = os.environ.get("BIGEARTH_S2_DIR", "")
    if not s2_dir:
        raise RuntimeError("Neither BIGEARTH_LMDB_DIR nor BIGEARTH_S2_DIR is set")
    # acquisition = patch_id minus the trailing _row_col suffix
    acquisition = "_".join(patch_id.split("_")[:-2])
    patch_dir = os.path.join(s2_dir, acquisition, patch_id)
    bands = []
    for band in ("B04", "B03", "B02"):
        tif_path = os.path.join(patch_dir, f"{patch_id}_{band}.tif")
        arr = np.array(Image.open(tif_path))
        bands.append(_normalize_band(arr, band))
    return Image.fromarray(np.stack(bands, axis=-1), mode="RGB")


def _load_rgb(patch_id: str) -> Image.Image:
    if os.environ.get("BIGEARTH_LMDB_DIR"):
        return _load_rgb_from_lmdb(patch_id)
    if os.environ.get("BIGEARTH_S2_DIR"):
        return _load_rgb_from_tif(patch_id)
    raise RuntimeError("Set BIGEARTH_LMDB_DIR (LMDB dir) or BIGEARTH_S2_DIR (raw TIF dir)")


# ---------------------------------------------------------------------------
# Shared: load image
# ---------------------------------------------------------------------------


def bigearth_doc_to_visual(doc):
    img = _load_rgb(doc["patch_id"])
    return [img]


# ---------------------------------------------------------------------------
# Binary (yes/no)
# ---------------------------------------------------------------------------


def bigearth_binary_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    post = kwargs.get("post_prompt", "\nAnswer with yes or no.")
    return f"{doc['input']}{post}"


def bigearth_binary_doc_to_target(doc):
    return doc["output"].strip().lower()


def bigearth_binary_process_results(doc, results):
    pred = results[0].strip().lower()
    # Accept any response that starts with "yes" or "no"
    if pred.startswith("yes"):
        pred = "yes"
    elif pred.startswith("no"):
        pred = "no"
    gold = doc["output"].strip().lower()
    correct = int(pred == gold)
    cat = doc.get("category", "unknown").replace(" ", "_")
    return {"binary_accuracy": correct, f"binary_{cat}": correct}


def bigearth_binary_aggregate(results):
    return sum(results) / len(results) if results else 0.0


# ---------------------------------------------------------------------------
# Multiple choice
# ---------------------------------------------------------------------------


def bigearth_mcq_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    post = kwargs.get("post_prompt", "\nAnswer with the option letter only (e.g. a, b, c, ...).")
    return f"{doc['input']}{post}"


def bigearth_mcq_doc_to_target(doc):
    return doc["output"].strip().lower()


def bigearth_mcq_process_results(doc, results):
    pred = results[0].strip().lower()
    # Extract first option letter from response
    match = re.search(r"\b([a-z])\b", pred)
    pred = match.group(1) if match else pred[:1]
    gold = doc["output"].strip().lower()
    correct = int(pred == gold)
    cat = doc.get("category", "unknown").replace(" ", "_")
    return {"mcq_accuracy": correct, f"mcq_{cat}": correct}


def bigearth_mcq_aggregate(results):
    return sum(results) / len(results) if results else 0.0


# ---------------------------------------------------------------------------
# Bounding box (Acc@0.5 IoU)
# ---------------------------------------------------------------------------


def bigearth_bbox_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    post = kwargs.get("post_prompt", "\nOutput the bounding box as [x1 y1, x2 y2] with coordinates normalized to [0, 1].")
    return f"{doc['input']}{post}"


def bigearth_bbox_doc_to_target(doc):
    return doc["output"].strip()


def _parse_bbox_01(s):
    """Parse [x1 y1, x2 y2] or similar into [x1,y1,x2,y2] float 0-1."""
    nums = re.findall(r"\d+\.?\d*", s)
    if len(nums) >= 4:
        return [float(n) for n in nums[:4]]
    return None


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


_SCALE_CANDIDATES = (1.0, 0.01, 0.001, 1.0 / 120.0)


def _best_iou(pred_box, gt_box, scales=_SCALE_CANDIDATES):
    """Best IoU across the coordinate conventions models emit (0-1, 0-100, 0-1000, 120px)."""
    return max(_iou([v * s for v in pred_box], gt_box) for s in scales)


def bigearth_bbox_process_results(doc, results):
    pred_box = _parse_bbox_01(results[0])
    gt_box = _parse_bbox_01(doc["output"])
    if pred_box is None or gt_box is None:
        return {"bbox_acc50": 0, "strict_bbox_acc50": 0}
    return {
        "bbox_acc50": int(_best_iou(pred_box, gt_box) >= 0.5),
        "strict_bbox_acc50": int(_iou(pred_box, gt_box) >= 0.5),
    }


def bigearth_bbox_aggregate(results):
    return sum(results) / len(results) if results else 0.0


# ---------------------------------------------------------------------------
# Captioning (CIDEr / BLEU / METEOR / ROUGE-L)
# ---------------------------------------------------------------------------

CAP_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]


def bigearth_cap_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["input"]


def bigearth_cap_doc_to_target(doc):
    return doc["output"].strip()


def bigearth_cap_process_results(doc, results):
    data = {"pred": results[0], "gt": doc["output"].strip(), "id": int(doc["ID"])}
    return {f"cap_{m}": data for m in CAP_METRICS}


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


def bigearth_cap_bleu1(r):
    return _cap_aggregate(r, "Bleu_1")


def bigearth_cap_bleu2(r):
    return _cap_aggregate(r, "Bleu_2")


def bigearth_cap_bleu3(r):
    return _cap_aggregate(r, "Bleu_3")


def bigearth_cap_bleu4(r):
    return _cap_aggregate(r, "Bleu_4")


def bigearth_cap_meteor(r):
    return _cap_aggregate(r, "METEOR")


def bigearth_cap_rouge(r):
    return _cap_aggregate(r, "ROUGE_L")


def bigearth_cap_cider(r):
    return _cap_aggregate(r, "CIDEr")


# ---------------------------------------------------------------------------
# Dataset filtering (called via process_docs in each sub-task YAML)
# ---------------------------------------------------------------------------
# The HF Hub dataset (BigEarthNet.txt.parquet) contains all tasks and splits in
# one file. The `type` column selects the task; `split == "bench"` selects the
# evaluation subset (~7k rows out of 9.5M total).


def bigearth_binary_filter_docs(docs):
    return docs.filter(lambda x: x["type"] == "binary" and x["split"] == "bench")


def bigearth_mcq_filter_docs(docs):
    return docs.filter(lambda x: x["type"] == "mcq" and x["split"] == "bench")


def bigearth_bbox_filter_docs(docs):
    return docs.filter(lambda x: x["type"] == "bounding box" and x["split"] == "bench")


def bigearth_cap_filter_docs(docs):
    return docs.filter(lambda x: x["type"] == "captioning" and x["split"] == "bench")
