"""Shared POLOS metric aggregation for captioning tasks."""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger as eval_logger

from lmms_eval.metrics.polos import PolosScorer

_scorer: PolosScorer | None = None


def _get_scorer() -> PolosScorer:
    global _scorer
    if _scorer is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        eval_logger.info(f"Loading POLOS scorer on {device}...")
        _scorer = PolosScorer(device=device)
    return _scorer


def polos_aggregation(
    results: list[dict[str, Any]],
    args: Any = None,
) -> float:
    """Aggregate POLOS scores over a list of result dicts.

    Each dict must contain 'image' (PIL Image or None), 'pred' (str),
    and 'answer' (list[str]).
    """
    images = [r["image"] for r in results]
    if any(img is None for img in images):
        raise ValueError("POLOS metric requires images but got None. " "Pass --process_with_media to include images in evaluation.")
    candidates = [r["pred"] for r in results]
    references_list = [r["answer"] for r in results]

    scorer = _get_scorer()
    mean_score, _ = scorer.score(
        images=images,
        candidates=candidates,
        references_list=references_list,
    )
    return mean_score
