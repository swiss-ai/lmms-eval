import os

import av
import numpy as np
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks.vsibench.utils import (
    base_cache_dir,
    cache_name,
)


def vsibench_doc_to_visual_as_images(doc, lmms_eval_specific_kwargs=None):
    """
    Return video frames as a list of PIL Images instead of video path.
    This allows the model to process the video as multi-image input.
    """
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)

    if not os.path.exists(video_path):
        raise FileExistsError(f"video path:{video_path} does not exist.")

    num_frames = 32
    if lmms_eval_specific_kwargs:
        num_frames = lmms_eval_specific_kwargs.get("num_frames", 32)

    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames or sum(1 for _ in container.decode(video=0))

    if total_frames == 0:
        container.close()
        raise ValueError(f"Video has 0 frames: {video_path}")

    num_frames = min(num_frames, total_frames)
    indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist())

    container.seek(0)
    pil_images = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            pil_images.append(frame.to_image())
        if len(pil_images) >= num_frames:
            break
    container.close()

    eval_logger.info(f"Loaded {len(pil_images)} frames from video as images (total_frames={total_frames})")
    return pil_images
