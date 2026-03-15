"""POLOS: learned image captioning metric (CVPR 2024).

Vendored inference-only implementation that removes dependencies on
pytorch-lightning, torchnlp, fairseq, and pandas.

Usage::

    from lmms_eval.metrics.polos import PolosScorer

    scorer = PolosScorer(device="cpu")
    mean_score, scores = scorer.score(
        images=[img1, img2],
        candidates=["a cat on a mat", "a dog running"],
        references_list=[
            ["a cat sitting on a mat"],
            ["a dog is running outdoors"],
        ],
    )
"""

from __future__ import annotations

import logging
import os
import urllib.request
import zipfile
from argparse import Namespace

import torch
import yaml
from PIL import Image
from tqdm import tqdm

from .estimator import PolosEstimator

logger = logging.getLogger(__name__)

_CHECKPOINT_URL = "https://polos-polaris.s3.ap-northeast-1.amazonaws.com/reprod.zip"
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "yuigawada")


def _download_checkpoint(cache_dir: str = _CACHE_DIR) -> str:
    """Download and extract the POLOS checkpoint if needed.

    Returns path to the checkpoint directory.
    """
    ckpt_dir = os.path.join(cache_dir, "reprod")
    hparams_path = os.path.join(ckpt_dir, "hparams.yaml")
    if os.path.isfile(hparams_path):
        return ckpt_dir

    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "reprod.zip")

    if not os.path.isfile(zip_path):
        logger.info("Downloading POLOS checkpoint...")
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc="POLOS checkpoint",
        ) as pbar:

            def _reporthook(
                block_num: int,
                block_size: int,
                total_size: int,
            ) -> None:
                if pbar.total is None and total_size > 0:
                    pbar.total = total_size
                pbar.update(block_size)

            urllib.request.urlretrieve(_CHECKPOINT_URL, zip_path, reporthook=_reporthook)

    logger.info("Extracting POLOS checkpoint...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    if not os.path.isfile(hparams_path):
        raise RuntimeError(f"Expected hparams.yaml at {hparams_path} after extraction")
    return ckpt_dir


def _find_checkpoint_file(ckpt_dir: str) -> str:
    """Find the .ckpt file in the checkpoint directory."""
    for fname in os.listdir(ckpt_dir):
        if fname.endswith(".ckpt"):
            return os.path.join(ckpt_dir, fname)
    raise FileNotFoundError(f"No .ckpt file found in {ckpt_dir}")


def load_polos_model(
    device: str = "cpu",
) -> PolosEstimator:
    """Load the POLOS model from the official checkpoint.

    Downloads the checkpoint on first use.
    """
    ckpt_dir = _download_checkpoint()
    hparams_path = os.path.join(ckpt_dir, "hparams.yaml")
    ckpt_path = _find_checkpoint_file(ckpt_dir)

    with open(hparams_path) as f:
        hparams_dict = yaml.safe_load(f)
    hparams = Namespace(**hparams_dict)

    model = PolosEstimator(hparams)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Remove keys that belong to CLIP (loaded separately)
    clip_keys = [k for k in state_dict if k.startswith("clip.")]
    for k in clip_keys:
        del state_dict[k]

    # Remove keys not present in the model
    # (e.g. metrics, loss from training)
    model_keys = set(model.state_dict().keys())
    extra_keys = [k for k in state_dict if k not in model_keys]
    for k in extra_keys:
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model._init_clip(device=device)
    model.eval()
    model.to(device)
    return model


class PolosScorer:
    """Lazy-loading POLOS scorer for image captioning evaluation."""

    def __init__(self, device: str = "cpu") -> None:
        self._model: PolosEstimator | None = None
        self._device = device

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = load_polos_model(self._device)

    def score(
        self,
        images: list[Image.Image],
        candidates: list[str],
        references_list: list[list[str]],
        batch_size: int = 32,
    ) -> tuple[float, list[float]]:
        """Score candidate captions against references.

        Args:
            images: List of PIL images.
            candidates: List of candidate caption strings.
            references_list: List of reference caption lists
                (one list per image).
            batch_size: Batch size for inference.

        Returns:
            (mean_score, per_sample_scores)
        """
        self._ensure_model()
        if self._model is None:
            raise RuntimeError("Failed to load POLOS model")

        data = [{"img": img, "mt": cand, "refs": refs} for img, cand, refs in zip(images, candidates, references_list)]
        _, scores = self._model.predict(
            data,
            batch_size=batch_size,
            cuda=(self._device != "cpu"),
        )
        mean = sum(scores) / len(scores)
        return mean, scores
