import re
from typing import Any, Dict, List, Optional

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{...}."""
    m = re.search(r"\\boxed\{(.+?)\}", text)
    return m.group(1).strip() if m else None


def _extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _clean_response(response: str) -> str:
    boxed = _extract_boxed(response)
    if boxed is not None:
        return boxed
    tagged = _extract_tag(response, "answer")
    if tagged is not None:
        return tagged
    return response.strip()


# ---------------------------------------------------------------------------
# Judgement helpers
# ---------------------------------------------------------------------------


def _judge_yesno(answer: str, response: str) -> bool:
    """Return True if response matches the yes/no ground truth."""
    ans = answer.lower().strip()
    resp = response.lower().strip()
    if ans == "yes" and re.search(r"\byes\b", resp):
        return True
    if ans == "no" and re.search(r"\bno\b", resp) and not re.search(r"\byes\b", resp):
        return True
    return False


def _tokenize(text: str) -> List[str]:
    return text.lower().replace(".", " .").split()


def _bleu(pred: str, target: str, n: int) -> float:
    weights = tuple(1.0 / n for _ in range(n))
    smooth = SmoothingFunction().method1
    return sentence_bleu([_tokenize(target)], _tokenize(pred), weights=weights, smoothing_function=smooth)


def _rouge_scores(pred: str, target: str) -> Dict[str, float]:
    if not pred.strip() or not target.strip():
        return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    scorer = Rouge()
    try:
        scores = scorer.get_scores(pred.lower(), target.lower())[0]
    except Exception:
        return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    return scores


def _judge_open(answer: str, response: str) -> Dict[str, float]:
    em = float(response.strip().lower() == answer.strip().lower())
    b1 = _bleu(response, answer, 1)
    b2 = _bleu(response, answer, 2)
    b3 = _bleu(response, answer, 3)
    b4 = _bleu(response, answer, 4)
    rouge = _rouge_scores(response, answer)
    # token-level F1
    pred_toks = set(_tokenize(response))
    gt_toks = set(_tokenize(answer))
    common = pred_toks & gt_toks
    if common:
        precision = len(common) / len(pred_toks)
        recall = len(common) / len(gt_toks)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision = recall = f1 = 0.0
    return {
        "em": em,
        "bleu1": b1,
        "bleu2": b2,
        "bleu3": b3,
        "bleu4": b4,
        "rouge1": rouge["rouge-1"]["f"],
        "rouge2": rouge["rouge-2"]["f"],
        "rougel": rouge["rouge-l"]["f"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# lmms-eval interface
# ---------------------------------------------------------------------------


def vqa_rad_doc_to_visual(doc: Dict[str, Any]):
    return [doc["image"].convert("RGB")]


def vqa_rad_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] = None):
    question = doc["question"].strip()
    answer = doc["answer"].lower().strip()
    if answer in ("yes", "no"):
        return question + "\nPlease answer 'yes' or 'no' (no extra output)."
    else:
        return question + "\nPlease answer the question concisely."


def vqa_rad_doc_to_target(doc: Dict[str, Any]) -> str:
    return doc["answer"].lower().strip()


def vqa_rad_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, Any]:
    raw_response = result[0] if result else ""
    response = _clean_response(raw_response).lower().strip()
    answer = doc["answer"].lower().strip()
    is_close = answer in ("yes", "no")

    if is_close:
        correct = float(_judge_yesno(answer, response))
        return {
            "close_accuracy": correct,
            "open_em": None,
            "bleu1": None,
            "bleu2": None,
            "bleu3": None,
            "bleu4": None,
            "rouge1": None,
            "rouge2": None,
            "rougel": None,
            "f1": None,
        }
    else:
        m = _judge_open(answer, response)
        return {
            "close_accuracy": None,
            "open_em": m["em"],
            "bleu1": m["bleu1"],
            "bleu2": m["bleu2"],
            "bleu3": m["bleu3"],
            "bleu4": m["bleu4"],
            "rouge1": m["rouge1"],
            "rouge2": m["rouge2"],
            "rougel": m["rougel"],
            "f1": m["f1"],
        }


# ---------------------------------------------------------------------------
# Aggregation helpers (filter None values from the other question type)
# ---------------------------------------------------------------------------


def _mean(items):
    valid = [x for x in items if x is not None]
    return sum(valid) / len(valid) if valid else 0.0


def agg_close_accuracy(items):
    return _mean(items)


def agg_open_em(items):
    return _mean(items)


def agg_bleu1(items):
    return _mean(items)


def agg_bleu2(items):
    return _mean(items)


def agg_bleu3(items):
    return _mean(items)


def agg_bleu4(items):
    return _mean(items)


def agg_rouge1(items):
    return _mean(items)


def agg_rouge2(items):
    return _mean(items)


def agg_rougel(items):
    return _mean(items)


def agg_f1(items):
    return _mean(items)
