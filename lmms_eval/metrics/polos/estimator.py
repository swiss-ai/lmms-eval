"""POLOS estimator for inference (stripped of PyTorch Lightning).

Vendored from polos/models/estimators/polos_estimator.py and
estimator_base.py with all training code removed.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from . import clip_model as clip
from .encoder import BERTEncoder
from .modules import FeedForward, ScalarMixWithDropout
from .utils import collate_tensors


def _apply_to_sample(f: Any, sample: Any) -> Any:
    if torch.is_tensor(sample):
        return f(sample)
    if isinstance(sample, dict):
        return {k: _apply_to_sample(f, v) for k, v in sample.items()}
    if isinstance(sample, list):
        return [_apply_to_sample(f, x) for x in sample]
    return sample


def _move_to_cuda(sample: Any) -> Any:
    return _apply_to_sample(lambda t: t.cuda(), sample)


def _move_to_cpu(sample: Any) -> Any:
    return _apply_to_sample(lambda t: t.cpu(), sample)


def _average_pooling(
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    padding_index: int,
) -> torch.Tensor:
    """Average pooling over non-padding positions."""
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    wordemb = embeddings.float().masked_fill_(padding_mask, 0.0).type_as(embeddings)
    sentemb = torch.sum(wordemb, 1)
    sum_mask = mask.unsqueeze(-1).expand(embeddings.size()).float().sum(1)
    return sentemb / sum_mask


class PolosEstimator(nn.Module):
    """POLOS multimodal caption scoring model (inference only).

    Combines BERT/RoBERTa text features with CLIP visual/text
    features to predict caption quality scores.
    """

    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self.hparams = hparams

        # Build text encoder
        self.encoder = BERTEncoder.from_pretrained(hparams)

        # Layer selection / mixing
        self.layer: int | str = int(hparams.layer) if hparams.layer != "mix" else hparams.layer

        self.scalar_mix: ScalarMixWithDropout | None = None
        if self.layer == "mix" and hparams.pool != "default":
            self.scalar_mix = ScalarMixWithDropout(
                mixture_size=self.encoder.num_layers,
                dropout=hparams.scalar_mix_dropout,
                do_layer_norm=True,
            )

        # Feed-forward estimator
        input_emb_sz = self.encoder.output_units * 4 + 512 * 6

        final_activation = getattr(hparams, "final_activation", "Sigmoid")
        self.ff = nn.Sequential(
            FeedForward(
                in_dim=input_emb_sz,
                hidden_sizes=hparams.hidden_sizes,
                activations=hparams.activations,
                dropout=hparams.dropout,
                final_activation=final_activation,
            ),
            nn.Sigmoid(),
        )

        # CLIP model (loaded lazily by load_polos_model)
        self.clip: clip.CLIP | None = None
        self.clip_preprocess: Any = None

    def _init_clip(self, device: str = "cpu") -> None:
        """Load CLIP model to given device."""
        self.clip, self.clip_preprocess = clip.load("ViT-B/32", device=device)

    def get_sentence_embedding(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Extract sentence embeddings using the configured pooling."""
        encoder_out = self.encoder(tokens, lengths)

        if self.scalar_mix:
            embeddings = self.scalar_mix(encoder_out["all_layers"], encoder_out["mask"])
        elif isinstance(self.layer, int) and 0 <= self.layer < self.encoder.num_layers:
            embeddings = encoder_out["all_layers"][self.layer]
        else:
            raise ValueError(f"Invalid layer: {self.layer}")

        padding_index = self.encoder.tokenizer.padding_index

        if self.hparams.pool == "avg":
            sentemb = _average_pooling(
                tokens,
                embeddings,
                encoder_out["mask"],
                padding_index,
            )
        elif self.hparams.pool == "cls":
            sentemb = embeddings[:, 0, :]
        elif self.hparams.pool == "default":
            sentemb = encoder_out["sentemb"]
        else:
            raise ValueError(f"Invalid pooling: {self.hparams.pool}")

        return (
            sentemb,
            embeddings,
            encoder_out["mask"],
            padding_index,
        )

    def prepare_sample(
        self,
        sample: list[dict[str, Any]],
        inference: bool = False,
    ) -> dict[str, Any]:
        """Prepare a batch for the model."""
        sample_dict = collate_tensors(sample)
        mt_inputs = self.encoder.prepare_sample(sample_dict["mt"])

        # Transpose refs from (batch, ref_count) to (ref_count, batch)
        # so each ref_inputs[k] contains the k-th reference for all
        # batch items, matching the expected shape in forward().
        transposed_refs = list(zip(*sample_dict["refs"]))
        ref_inputs = [self.encoder.prepare_sample(list(ref)) for ref in transposed_refs]
        return {
            "mt_inputs": mt_inputs,
            "ref_inputs": ref_inputs,
            "refs": [list(r) for r in transposed_refs],
            "mt": sample_dict["mt"],
            "imgs": sample_dict["img"],
        }

    def forward(
        self,
        refs: list[list[str]],
        mt: list[str],
        ref_inputs: list[dict[str, torch.Tensor]],
        mt_inputs: dict[str, torch.Tensor],
        imgs: list[Any],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Score candidate captions against references and images."""
        if self.clip is None:
            raise RuntimeError("CLIP not initialized")
        device = next(self.parameters()).device

        mt_tokens = mt_inputs["tokens"]
        mt_lengths = mt_inputs["lengths"]
        mt_sentemb, _, _, _ = self.get_sentence_embedding(mt_tokens, mt_lengths)

        ref_sentemb_list = []
        for ref in ref_inputs:
            ref_sentemb, _, _, _ = self.get_sentence_embedding(ref["tokens"], ref["lengths"])
            ref_sentemb_list.append(ref_sentemb)

        # CLIP features
        refs_clip = []
        for ref_list in refs:
            subset = [clip.tokenize("A photo depicts " + r, truncate=True).to(device) for r in ref_list]
            subset_tensor = torch.cat(subset, dim=0)
            refs_clip.append(self.clip.encode_text(subset_tensor))

        mts_clip = clip.tokenize(["A photo depicts " + x for x in mt], truncate=True).to(device)
        imgs_clip = torch.cat(
            [self.clip_preprocess(img).unsqueeze(0) for img in imgs],
            dim=0,
        ).to(device)

        imgs_clip = self.clip.encode_image(imgs_clip)
        mts_clip = self.clip.encode_text(mts_clip)

        scores = []
        for ref_sentemb, ref_clip in zip(ref_sentemb_list, refs_clip):
            diff = torch.abs(mt_sentemb - ref_sentemb)
            mul = mt_sentemb * ref_sentemb
            diff_clip = torch.abs(imgs_clip - mts_clip)
            mul_clip = imgs_clip * mts_clip
            diff_clip_txt = torch.abs(ref_clip - mts_clip)
            mul_clip_txt = ref_clip * mts_clip

            x = torch.cat(
                (
                    ref_sentemb,
                    mt_sentemb,
                    diff,
                    mul,
                    imgs_clip,
                    mts_clip,
                    diff_clip,
                    mul_clip,
                    diff_clip_txt,
                    mul_clip_txt,
                ),
                dim=1,
            )
            scores.append(self.ff(x))

        score = torch.max(torch.stack(scores), dim=0).values
        return {"score": score}

    def predict(
        self,
        samples: list[dict[str, Any]],
        cuda: bool = False,
        show_progress: bool = True,
        batch_size: int = 32,
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """Run batched inference.

        Returns (samples_with_scores, list_of_scores).
        """
        if self.training:
            self.eval()

        if cuda and torch.cuda.is_available():
            self.to("cuda")

        with torch.no_grad():
            batches = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]
            model_inputs = []
            if show_progress:
                pbar = tqdm(
                    total=len(batches),
                    desc="Preparing batches...",
                    dynamic_ncols=True,
                    leave=None,
                )
            for batch in batches:
                model_inputs.append(self.prepare_sample(batch, inference=True))
                if show_progress:
                    pbar.update(1)
            if show_progress:
                pbar.close()

            scores: list[float] = []
            if show_progress:
                pbar = tqdm(
                    total=len(batches),
                    desc="Scoring hypothesis...",
                    dynamic_ncols=True,
                    leave=None,
                )
            for model_input in model_inputs:
                if cuda and torch.cuda.is_available():
                    model_input = _move_to_cuda(model_input)
                    model_out = self.forward(**model_input)
                    model_out = _move_to_cpu(model_out)
                else:
                    model_out = self.forward(**model_input)

                batch_scores = model_out["score"].numpy().tolist()
                for s in batch_scores:
                    scores.append(s[0])
                if show_progress:
                    pbar.update(1)
            if show_progress:
                pbar.close()

        for i, s in enumerate(scores):
            samples[i]["predicted_score"] = s
        return samples, scores
