#!/usr/bin/env python
"""Report distribution stats on a built manifest: duration bands, tier counts,
confidence histogram, total hours. Run after build_dataset.py.

Usage:
    python scripts/stats.py build/dataset/train.jsonl
"""
import json
import sys
from collections import Counter


def main(path: str):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if not rows:
        print("empty manifest")
        return

    durs = [r.get("duration", 0.0) for r in rows]
    total_h = sum(durs) / 3600
    bands = Counter()
    for d in durs:
        if d < 3:      bands["<3s"] += 1
        elif d < 8:    bands["3-8s"] += 1
        elif d < 18:   bands["8-18s"] += 1
        elif d <= 28:  bands["18-28s"] += 1
        else:          bands[">28s"] += 1

    tiers = Counter(r.get("tier", "?") for r in rows)
    sources = Counter(r.get("source", "silver") for r in rows)

    confs = [r["confidence"] for r in rows if r.get("confidence") is not None]
    chist = Counter()
    for c in confs:
        chist[f"{int(c*10)/10:.1f}"] += 1

    print(f"records:        {len(rows)}")
    print(f"total hours:    {total_h:.1f}")
    print(f"mean duration:  {sum(durs)/len(durs):.1f}s")
    print("\nduration bands:")
    for k in ["<3s", "3-8s", "8-18s", "18-28s", ">28s"]:
        n = bands[k]
        print(f"  {k:>7}: {n:6d}  ({100*n/len(rows):4.1f}%)")
    print("\ntiers:", dict(tiers))
    print("sources:", dict(sources))
    print("\nconfidence histogram (bucketed):")
    for k in sorted(chist):
        print(f"  {k}: {'#' * (chist[k] * 40 // max(chist.values()))} {chist[k]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/dataset/train.jsonl")
