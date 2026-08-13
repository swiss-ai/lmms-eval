# Fork delta vs upstream (EvolvingLMMs-Lab/lmms-eval)

This branch is maintained as a **thematic patch series** on top of `upstream/main`:
`git log upstream/main..HEAD` is the authoritative delta. Re-cut the series onto
new upstream instead of merging; keep every commit thematic. Audit this file at
each sync: every invasive entry must still justify itself or be dropped.

## Patch series (rebase units)

| Patch | Contents | Conflict risk at sync |
|---|---|---|
| models (additive) | `apertus_1p5_vllm`, `molmo_vllm`, Emu3.5 integration (gemma4_hf retired 2026-08-13: could not complete at fleet batch size — OOM'd ranks hung nodes — while the generic vllm backend serves Gemma4 in 9 min) | none |
| models (invasive) | registry entries in `models/__init__.py`; `chat/vllm.py`, `chat/huggingface.py`, `simple/vllm.py`, `media_encoder.py` | dict-union + small hunks |
| security | HF-token masking in `__main__.py`, `loggers/evaluation_tracker.py`, `loggers/wandb_logger.py` | low — **upstream-PR candidate** |
| core | `api/task.py`, `evaluator.py`, `protocol.py` (visual routing, encode settings) | medium — upstream refactors here |
| judge | `llm_judge/providers/openai.py`, `dummy.py` | low |
| tasks (additive) | medevalkit, geobench, vrsbench and other suite benchmarks | none |
| tasks (invasive) | scoring/loader fixes across existing benchmarks | low, mostly disjoint |
| infra | pyproject, gitignore, CI, slurm examples; deletes upstream `tools/batch_watchdog.py` | low |

## Notes from the 2026-08-11 sync

- `chat/vllm.py` per-request sampling: upstream fixed the same bug (9c7a55ae);
  kept our variant. Next sync, try dropping our hunk in favour of upstream.
- `api/task.py`: upstream refactored auto-messages into `_auto_doc_to_messages` /
  `_visual_to_content`; our dict-visual metadata and audio routing survived
  inside the new structure. Our old inline copy was dropped.
- `protocol.py`: kept our `_encode_settings`/`_image_url_entry` helpers plus
  upstream's `pass_video_url` parameter.
- covost2 yamls: took upstream's new dataset orgs (`lmms-lab-audio/...`).
- `screenspot/utils_rec.py`: union — our permissive regex + upstream's
  0-1000 coordinate normalization.

## Upstream-PR queue (each merged PR permanently shrinks this delta)

1. HF-token masking (security patch) — ready as-is.
2. vlmsareblind zero-correct subtask mean fix.
3. Generic task scoring fixes from the tasks-invasive patch.
