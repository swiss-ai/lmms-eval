#!/usr/bin/env python3
"""Per-task truncation rate for a run: the fraction of generations that hit the
token cap (no clean stop). Truncation matters most for thinking runs, where it
signals non-termination rather than a budget shortfall.

Authoritative source is the logged ``output_tokens`` (>= max_new_tokens means
the cap was hit); runs predating the output-token capture fall back to a
char-length heuristic on the response text.

Usage:
  python3 truncation_report.py --run-root /path/to/run
"""

import argparse
import json
import re
from pathlib import Path


def _max_new_tokens(run_root: Path) -> int | None:
    submission = run_root / "submission.json"
    if not submission.is_file():
        return None
    gen = json.loads(submission.read_text()).get("gen_kwargs") or ""
    if isinstance(gen, dict):
        gen = ",".join(f"{k}={v}" for k, v in gen.items())
    m = re.search(r"max_new_tokens=(\d+)", str(gen))
    return int(m.group(1)) if m else None


def _looks_truncated(text: str) -> bool:
    tail = text.strip().lower()[-15:]
    clean_end = any(tail.endswith(t) for t in (".", "?", "!", '"', "yes", "no", ")"))
    return len(text) > 14000 and not clean_end and "answer" not in tail


def _truncation_rate(samples_path: Path, max_new_tokens: int | None) -> tuple[int, float | None, str]:
    n = hit = authoritative = 0
    for line in samples_path.open():
        record = json.loads(line)
        tcs = record.get("token_counts") or []
        out = next((tc.get("output_tokens") for tc in tcs if tc and tc.get("output_tokens") is not None), None)
        n += 1
        if out is not None and max_new_tokens is not None:
            hit += out >= max_new_tokens
            authoritative += 1
        else:
            resp = record.get("filtered_resps", [""])
            resp = resp[0] if isinstance(resp, list) else str(resp)
            hit += _looks_truncated(resp)
    if n == 0:
        return 0, None, "n/a"
    if authoritative == n:
        source = "output_tokens"
    elif authoritative == 0:
        source = "char-heuristic"
    else:
        source = f"mixed({authoritative}/{n} authoritative)"
    return n, hit / n, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True, type=Path, help="run directory containing results/ and submission.json")
    parser.add_argument("--max-new-tokens", type=int, help="override the cap (else read from submission.json gen_kwargs)")
    args = parser.parse_args()

    cap = args.max_new_tokens or _max_new_tokens(args.run_root)
    print(f"max_new_tokens={cap}")
    for samples in sorted(args.run_root.rglob("*_samples_*.jsonl")):
        task = re.sub(r".*_samples_", "", samples.name).replace(".jsonl", "")
        n, rate, source = _truncation_rate(samples, cap)
        if rate is not None:
            print(f"  {task:32} {100 * rate:5.1f}%  (n={n}, via {source})")


if __name__ == "__main__":
    main()
