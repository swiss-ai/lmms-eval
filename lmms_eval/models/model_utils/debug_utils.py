"""Shared debug sample logging for model evaluation."""

from __future__ import annotations

from typing import Optional

import torch
from loguru import logger as eval_logger


def log_debug_sample(
    sample_num: int,
    total_samples: int,
    prompt_clean: str,
    prompt_with_tokens: str,
    answer_clean: str,
    answer_with_tokens: str,
    attention_mask: Optional[torch.Tensor] = None,
) -> None:
    """Log a single debug sample with standardized formatting.

    Args:
        sample_num: Current sample number (1-indexed).
        total_samples: Total number of debug samples to print.
        prompt_clean: The prompt text without special tokens.
        prompt_with_tokens: The prompt text with special tokens.
        answer_clean: The model answer without special tokens.
        answer_with_tokens: The model answer with special tokens.
        attention_mask: If provided, logs attention mask statistics
            (sequence length, padding count, head/tail values).
    """
    eval_logger.info("=" * 80)
    eval_logger.info(f"DEBUG SAMPLE {sample_num}/{total_samples}")
    eval_logger.info("=" * 80)
    eval_logger.info(f"PROMPT (clean): {prompt_clean}")
    eval_logger.info(f"PROMPT (with tokens): {prompt_with_tokens}")
    if attention_mask is not None:
        seq_len = attention_mask.shape[0]
        n_pad = (attention_mask == 0).sum().item()
        n_real = (attention_mask == 1).sum().item()
        head = attention_mask[:8].tolist()
        tail = attention_mask[-8:].tolist()
        eval_logger.info(
            f"ATTENTION MASK: len={seq_len} pad={n_pad} "
            f"real={n_real} head={head} tail={tail}"
        )
    eval_logger.info(f"ANSWER (clean): {answer_clean}")
    eval_logger.info(f"ANSWER (with tokens): {answer_with_tokens}")
    eval_logger.info("=" * 80)
