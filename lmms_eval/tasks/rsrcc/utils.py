import os
import re

from loguru import logger as eval_logger


def _parse_text(text):
    """Returns (question_with_options, answer, is_mcq); the text field escapes newlines as literal backslash-n."""
    parts = re.split(r"(?:\\n|\s)*\*\*Answer:", text, maxsplit=1)
    question_block = re.sub(r"^\*\*Question:\*\*\s*", "", parts[0])
    question_block = question_block.replace("\\n", "\n").strip()

    answer = ""
    if len(parts) > 1:
        match = re.search(r"[A-Za-z0-9]+", parts[1])
        if match:
            answer = match.group(0)

    is_mcq = bool(re.search(r"\*\*[A-D]\)", question_block))
    return question_block, answer, is_mcq


def rsrcc_filter_docs(dataset):
    kept = dataset.filter(lambda doc: _parse_text(doc["text"])[1] != "")
    dropped = len(dataset) - len(kept)
    if dropped:
        eval_logger.warning(f"rsrcc: dropped {dropped}/{len(dataset)} docs with an unparseable gold answer")
    return kept


def rsrcc_doc_to_visual(doc):
    return [doc["before"].convert("RGB"), doc["after"].convert("RGB")]


def rsrcc_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question_block, _, is_mcq = _parse_text(doc["text"])
    if "**Answer:" in question_block:
        raise ValueError("rsrcc: answer marker survived prompt construction; refusing to leak the label into the prompt")

    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "You are shown two satellite images: the first image is the BEFORE image and the second image is the AFTER image.\n\n")
    if is_mcq:
        post_prompt = kwargs.get("post_prompt_mcq", "\nAnswer with the option letter from the given choices directly (A, B, C, or D).")
    else:
        post_prompt = kwargs.get("post_prompt_yesno", "\nAnswer with Yes or No.")

    return f"{pre_prompt}**Question:** {question_block}{post_prompt}"


def rsrcc_doc_to_target(doc):
    _, answer, _ = _parse_text(doc["text"])
    return answer


def _extract_pred(pred, is_mcq):
    pred = pred.strip()
    if is_mcq:
        match = re.search(r"\b([A-D])\b", pred)
        return match.group(1) if match else (pred[0].upper() if pred else "")
    else:
        lower = pred.lower()
        if lower.startswith("yes"):
            return "Yes"
        if lower.startswith("no"):
            return "No"
        match = re.search(r"\b(yes|no)\b", lower)
        return match.group(1).capitalize() if match else pred


def rsrcc_process_results(doc, results):
    pred_raw = results[0]
    _, gold, is_mcq = _parse_text(doc["text"])
    pred = _extract_pred(pred_raw, is_mcq)
    correct = int(pred.upper() == gold.upper())

    result = {"accuracy": correct}
    if is_mcq:
        result["mcq_accuracy"] = correct
    else:
        result["yesno_accuracy"] = correct
    return result


def rsrcc_aggregate_accuracy(results):
    return sum(results) / len(results) if results else 0.0


def rsrcc_aggregate_mcq(results):
    return sum(results) / len(results) if results else 0.0


def rsrcc_aggregate_yesno(results):
    return sum(results) / len(results) if results else 0.0
