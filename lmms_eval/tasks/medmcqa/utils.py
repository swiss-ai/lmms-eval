import random
from typing import Any, Dict, List

import numpy as np

MEDMCQA_PROMPT = (
    "Answer the following multiple choice medical question. " "There is only one correct answer. " "The last line of your response should be in the format " "'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, or D."
)

ANSWER_LETTERS = ["A", "B", "C", "D"]


def medmcqa_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any],
) -> str:
    """Format MedMCQA sample into a prompt with question and options."""
    question = doc["question"]
    options_block = "\n".join(f"{letter}. {doc[key]}" for letter, key in zip(ANSWER_LETTERS, ["opa", "opb", "opc", "opd"]))
    return f"{MEDMCQA_PROMPT}\nQuestion: {question}\n{options_block}\n"


def medmcqa_doc_to_target(doc: Dict[str, Any]) -> str:
    """Return the ground-truth answer letter."""
    return ANSWER_LETTERS[doc["cop"]]


def medmcqa_doc_to_choice(doc: Dict[str, Any]) -> List[str]:
    """Return the list of choice letters."""
    return ANSWER_LETTERS


def medmcqa_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, float]:
    """Parse model output and compute accuracy against the gold letter."""
    response = result[0].strip()
    all_choices = medmcqa_doc_to_choice(doc)
    pred = _parse_multi_choice_response(response, all_choices)
    gt_ans = medmcqa_doc_to_target(doc)
    score = 1.0 if pred == gt_ans else 0.0
    return {"accuracy": score}


def _parse_multi_choice_response(response: str, all_choices: List[str]) -> str:
    """Extract a single letter answer from the model response."""
    for ch in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(ch)
    response = " " + response + " "

    candidates: List[str] = []

    # (A) style
    for c in all_choices:
        if f"({c})" in response:
            candidates.append(c)

    # plain letter surrounded by spaces
    if len(candidates) == 0:
        for c in all_choices:
            if f" {c} " in response:
                candidates.append(c)

    # A., B., etc.
    if len(candidates) == 0:
        for c in all_choices:
            if f"{c}." in response:
                candidates.append(c)

    if len(candidates) == 0:
        return random.choice(all_choices)
    if len(candidates) > 1:
        start_indexes = [response.rfind(f" {can} ") for can in candidates]
        return candidates[int(np.argmax(start_indexes))]
    return candidates[0]
