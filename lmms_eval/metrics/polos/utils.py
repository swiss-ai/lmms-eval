"""Reimplementation of torchnlp utilities used by POLOS.

Replaces:
- torchnlp.utils.collate_tensors
- torchnlp.utils.lengths_to_mask
- torchnlp.encoders.text.stack_and_pad_tensors
"""

from __future__ import annotations

from typing import Any

import torch


def collate_tensors(
    batch: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """Transpose a list of dicts into a dict of lists.

    Equivalent to torchnlp.utils.collate_tensors.
    """
    keys = batch[0].keys()
    return {key: [sample[key] for sample in batch] for key in keys}


def lengths_to_mask(
    lengths: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a boolean mask from sequence lengths.

    Returns a [batch_size x max_len] tensor where True means valid
    (non-padding) position.
    """
    max_len = int(lengths.max().item())
    arange = torch.arange(max_len, device=device)
    return arange.unsqueeze(0) < lengths.unsqueeze(1)


def stack_and_pad_tensors(
    tensors: list[torch.Tensor],
    padding_index: int = 0,
    dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length tensors and stack them.

    Returns (padded_tensor, lengths).
    """
    lengths = torch.tensor([t.size(dim) for t in tensors], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = []
    for t in tensors:
        pad_size = max_len - t.size(dim)
        if pad_size > 0:
            pad_tensor = torch.full((pad_size,), padding_index, dtype=t.dtype)
            t = torch.cat([t, pad_tensor], dim=dim)
        padded.append(t)
    return torch.stack(padded, dim=0), lengths
