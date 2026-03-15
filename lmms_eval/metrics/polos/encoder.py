"""BERT encoder for POLOS (simplified from polos/models/encoders/).

Strips training-only code (freeze_embeddings, layerwise_lr) and
replaces torchnlp dependencies with local reimplementations.
"""

from __future__ import annotations

import warnings
from argparse import Namespace
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .utils import lengths_to_mask, stack_and_pad_tensors


class HFTextEncoder:
    """Minimal HuggingFace tokenizer wrapper.

    Replaces polos/tokenizers_/hf_tokenizer.py +
    tokenizer_base.py without torchnlp dependency.
    """

    def __init__(self, model: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self._pad_index: int = self.tokenizer.pad_token_id  # type: ignore[assignment]

    @property
    def padding_index(self) -> int:
        return self._pad_index

    def encode(self, sequence: str) -> torch.Tensor:
        ids = self.tokenizer(sequence, truncation=False)["input_ids"]
        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(self, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and pad a batch of strings."""
        encoded = [self.encode(s) for s in sequences]
        return stack_and_pad_tensors(encoded, padding_index=self.padding_index)


class BERTEncoder(nn.Module):
    """BERT/RoBERTa encoder for POLOS inference."""

    def __init__(
        self,
        tokenizer: HFTextEncoder,
        hparams: Namespace,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained(hparams.pretrained_model)
        self.model.encoder.output_hidden_states = True
        self._output_units: int = self.model.config.hidden_size
        self._n_layers: int = self.model.config.num_hidden_layers + 1
        self._max_pos: int = self.model.config.max_position_embeddings

    @classmethod
    def from_pretrained(cls, hparams: Namespace) -> BERTEncoder:
        tokenizer = HFTextEncoder(model=hparams.pretrained_model)
        return cls(tokenizer=tokenizer, hparams=hparams)

    @property
    def output_units(self) -> int:
        return self._output_units

    @property
    def max_positions(self) -> int:
        return self._max_pos

    @property
    def num_layers(self) -> int:
        return self._n_layers

    def prepare_sample(self, sample: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a list of strings, returning tokens and lengths."""
        tokens, lengths = self.tokenizer.batch_encode(sample)
        # Truncate to max positions if needed
        if lengths.max() > self.max_positions:
            warnings.warn(
                f"Encoder max length exceeded " f"({lengths.max()} > {self.max_positions}).",
                category=RuntimeWarning,
            )
            lengths = lengths.clamp(max=self.max_positions)
            tokens = tokens[:, : self.max_positions]
        return {"tokens": tokens, "lengths": lengths}

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, Any]:
        mask = lengths_to_mask(lengths, device=tokens.device)
        last_hidden_states, pooler_output, all_layers = self.model(
            tokens,
            mask,
            output_hidden_states=True,
            return_dict=False,
        )
        return {
            "sentemb": pooler_output,
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "mask": mask,
        }
