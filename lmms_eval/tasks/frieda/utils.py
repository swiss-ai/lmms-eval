import os
import re
from collections import defaultdict

from loguru import logger as eval_logger
from PIL import Image

def _img_dir():
    explicit = os.environ.get("FRIEDA_IMG_DIR", "")
    if explicit:
        return explicit
    base = os.environ.get("FRIEDA_DIR", "")
    if base:
        return os.path.join(base, "images")
    raise RuntimeError("Set FRIEDA_DIR (FRIEDA base dir) or FRIEDA_IMG_DIR (images/ dir)")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def frieda_doc_to_visual(doc):
    base = _img_dir()
    return [Image.open(os.path.join(base, url)).convert("RGB") for url in doc["image_urls"]]


# ---------------------------------------------------------------------------
# Text / target
# ---------------------------------------------------------------------------

def frieda_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    n = len(doc["image_urls"])
    if n == 1:
        pre = kwargs.get("pre_prompt_single", "Look at this map and answer the following question.\n")
    else:
        pre = kwargs.get("pre_prompt_multi", f"You are shown {n} maps. Answer the following question based on them.\n")
    post = kwargs.get("post_prompt", "\nProvide a concise answer.")
    return f"{pre}{doc['question_text']}{post}"


def frieda_doc_to_target(doc):
    return doc["expected_answer"].strip()


# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Lowercase, collapse whitespace, strip leading/trailing punctuation."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".,;:\"'")
    return s


def _f1(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return 0.0
    prec = len(common) / len(pred_toks)
    rec  = len(common) / len(gold_toks)
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# Process results
# ---------------------------------------------------------------------------

def frieda_process_results(doc, results):
    pred  = results[0].strip()
    gold  = doc["expected_answer"].strip()
    em    = int(_normalize(pred) == _normalize(gold))
    f1    = _f1(pred, gold)
    domain = doc.get("domain", "unknown")
    atype  = doc.get("answer_type", "unknown").replace(" ", "_")
    return {
        "exact_match": em,
        "f1": f1,
        "per_domain": {"domain": domain, "em": em, "f1": f1},
        "per_type":   {"atype": atype,  "em": em, "f1": f1},
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def frieda_exact_match(results):
    return sum(results) / len(results) if results else 0.0


def frieda_f1(results):
    return sum(results) / len(results) if results else 0.0


def _breakdown_aggregate(results, key, score_key):
    by_group = defaultdict(list)
    for r in results:
        by_group[r[key]].append(r[score_key])
    per_group = {g: sum(v) / len(v) for g, v in sorted(by_group.items())}
    lines = "\n".join(f"  {g}: {s:.3f} ({len(by_group[g])} samples)" for g, s in per_group.items())
    eval_logger.info(f"FRIEDA breakdown by {key} ({score_key}):\n{lines}")
    return sum(per_group.values()) / len(per_group) if per_group else 0.0


def frieda_per_domain_em(results): return _breakdown_aggregate(results, "domain", "em")
def frieda_per_type_em(results):   return _breakdown_aggregate(results, "atype",  "em")
