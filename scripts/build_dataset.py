#!/usr/bin/env python
"""Build the Whisper-ready dataset from a corpus directory.

Expects each call as a (.json, .wav) pair sharing a basename, e.g.
    data/calls/CALL123.json
    data/calls/CALL123.wav

Usage:
    python scripts/build_dataset.py \
        --corpus data/calls \
        --out build/dataset \
        --config config/default.yaml \
        --gold data/gold/train.jsonl       # optional, your clean 2k set
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_dataset.config import load_config
from asr_dataset.pipeline import process_corpus


def find_pairs(corpus: Path) -> list[tuple[str, str]]:
    pairs = []
    for jp in sorted(corpus.rglob("*.json")):
        wp = jp.with_suffix(".wav")
        if wp.exists():
            pairs.append((str(jp), str(wp)))
        else:
            print(f"  [skip] no wav for {jp.name}", file=sys.stderr)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="dir of paired .json/.wav")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--config", default=None)
    ap.add_argument("--gold", default=None, help="optional gold train.jsonl to mix + oversample")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--no-second-asr", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_align:
        cfg.align.enabled = False
    if args.no_second_asr:
        cfg.second_asr.enabled = False

    pairs = find_pairs(Path(args.corpus))
    print(f"found {len(pairs)} call pairs")

    gold = None
    if args.gold:
        gold = [json.loads(l) for l in open(args.gold, encoding="utf-8")]
        print(f"loaded {len(gold)} gold records (oversample x{cfg.split.gold_oversample})")

    stats = process_corpus(pairs, cfg, args.out, gold_records=gold)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    Path(args.out, "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
