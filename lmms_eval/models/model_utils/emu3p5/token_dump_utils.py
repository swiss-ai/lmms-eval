"""Utilities for dumping tokenized EMU3.5 samples for pipeline comparison."""

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from loguru import logger as eval_logger

TOKENIZED_SAMPLES_DUMP_ENV_VAR = "LMMS_EVAL_EMU3P5_TOKENIZED_SAMPLES_DUMP_PATH"


def resolve_tokenized_samples_dump_path(configured_path: str | None = None) -> Path | None:
    """Resolve output path for tokenized-sample JSONL dumps."""
    path_value = configured_path or os.getenv(TOKENIZED_SAMPLES_DUMP_ENV_VAR)
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    return Path(os.path.expandvars(path_value.strip())).expanduser()


def dump_tokenized_samples_jsonl(
    dump_path: Path | str | None,
    samples: Sequence[Mapping[str, Any]],
) -> None:
    """Append tokenized sample payloads to a JSONL file."""
    if dump_path is None or not samples:
        return

    path = Path(dump_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as dump_file:
            for sample in samples:
                dump_file.write(json.dumps(sample, ensure_ascii=False))
                dump_file.write("\n")
    except OSError as exc:
        eval_logger.warning(f"Failed writing EMU3.5 tokenized-sample dump to {path}: {exc}")
