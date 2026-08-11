# Remote Sensing Benchmarks

Five remote sensing benchmarks added to lmms-eval.

## Benchmarks

| Benchmark | HF dataset | Task name(s) | Metrics |
|---|---|---|---|
| [RSRCC](https://huggingface.co/datasets/google/RSRCC) | `google/RSRCC` | `rsrcc_test`, `rsrcc_val` | accuracy, mcq_accuracy, yesno_accuracy |
| [VRSBench](https://huggingface.co/datasets/xiang709/VRSBench) | `xiang709/VRSBench` | `vrsbench_vqa`, `vrsbench_cap`, `vrsbench_ref` | vqa_accuracy, CIDEr/BLEU/METEOR/ROUGE-L, ref_acc50 |
| [BigEarthNet.txt](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt) | `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` | `bigearth_binary`, `bigearth_mcq`, `bigearth_bbox`, `bigearth_cap` | binary/mcq accuracy, bbox_acc50, CIDEr/BLEU/METEOR/ROUGE-L |
| [GEOBench-VLM](https://huggingface.co/datasets/aialliance/GEOBench-VLM) | `aialliance/GEOBench-VLM` | `geobench_single`, `geobench_temporal`, `geobench_cap`, `geobench_ref` | single/temporal accuracy, CIDEr/BLEU/METEOR/ROUGE-L, ref_acc50 |
| [FRIEDA](https://huggingface.co/datasets/knowledge-computing/FRIEDA) | `knowledge-computing/FRIEDA` | `frieda` | exact_match, f1 |

## What loads automatically vs. what you need locally

QA data for all benchmarks loads from HF Hub automatically. Some benchmarks require local images:

| Benchmark | Images |
|---|---|
| RSRCC | Embedded in HF dataset — nothing to download |
| GEOBench Single | Embedded in HF dataset — nothing to download |
| VRSBench | Local — set `VRSBENCH_DIR` to the directory containing `Images_val/` |
| BigEarthNet.txt | Local — set `BIGEARTH_LMDB_DIR` (LMDB) or `BIGEARTH_S2_DIR` (raw TIF) |
| GEOBench Temporal/Cap/Ref | Local — set `GEOBENCH_DIR` to the root with `Temporal/`, `Captioning/`, `Ref-Det/` extracted |
| FRIEDA | Local — set `FRIEDA_DIR` to the directory containing `images/` |

## Running

```bash
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks rsrcc_test \
    --batch_size 1 \
    --output_path ./results
```

Replace `rsrcc_test` with any task name from the table above, or use the group names `rsrcc`, `vrsbench`, `bigearth_txt`, `geobench` to run all sub-tasks at once.

**METEOR scoring requires Java:**
```bash
conda install -y -c conda-forge openjdk --no-deps
```

## Notes

- **Multi-image tasks**: RSRCC and GEOBench Temporal pass two images per sample. FRIEDA passes 1–2 map images. The model must support multi-image input.
- **BigEarthNet image backends**: LMDB (via `rico-hdl`) is faster; raw TIF files work as a fallback. Only Sentinel-2 RGB bands (B04/B03/B02) are used.
- **RSRCC is gated**: run `huggingface-cli login` before first use.
