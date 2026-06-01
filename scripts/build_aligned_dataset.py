#!/usr/bin/env python
"""Build a Whisper-ready dataset using FORCED ALIGNMENT (recommended).

Ignores Gemini's coarse timestamps. For each call it splits the stereo wav into
one channel per speaker, concatenates that speaker's text, forced-aligns them to
get accurate word timings, then cuts chunks at real pauses.

    python scripts/build_aligned_dataset.py \
        --corpus data/calls \
        --out build/dataset \
        --config config/default.yaml \
        --gold data/gold/train.jsonl     # optional

Requires: pip install ctc-forced-aligner torch torchaudio
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_dataset.config import load_config
from asr_dataset import manifest
from asr_dataset.forced_align import ForcedAligner, process_call_aligned


def find_pairs(corpus: Path):
    pairs = []
    for jp in sorted(corpus.rglob("*.json")):
        wp = jp.with_suffix(".wav")
        if wp.exists():
            pairs.append((str(jp), str(wp)))
            continue
        siblings = list(jp.parent.glob("*.wav"))
        if len(siblings) == 1:
            pairs.append((str(jp), str(siblings[0])))
        elif len(siblings) > 1:
            print(f"  [skip] multiple wavs in {jp.parent}, ambiguous", file=sys.stderr)
        else:
            print(f"  [skip] no wav for {jp.name}", file=sys.stderr)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--gold", default=None, help="optional gold train.jsonl to mix + oversample")
    ap.add_argument("--keep-prep", action="store_true",
                    help="also keep the per-channel wav+txt artifacts under <out>/prepared")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N calls")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    audio_dir = out_dir / "audio"
    prep_dir = (out_dir / "prepared") if args.keep_prep else None

    pairs = find_pairs(Path(args.corpus))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"found {len(pairs)} call pairs")

    aligner = ForcedAligner(cfg)

    all_chunks = []
    tier_counts: dict[str, int] = {}
    dropped_short = 0
    for n, (jp, wp) in enumerate(pairs, 1):
        try:
            chunks = process_call_aligned(jp, wp, cfg, aligner, audio_dir, prep_dir)
        except Exception as e:
            print(f"  [error] {Path(jp).name}: {e}", file=sys.stderr)
            continue
        for c in chunks:
            if c.effective_duration < cfg.chunk.min_s:
                dropped_short += 1
                continue
            all_chunks.append(c)
            tier_counts[c.tier or "?"] = tier_counts.get(c.tier or "?", 0) + 1
        if n % 25 == 0 or n == len(pairs):
            print(f"  {n}/{len(pairs)} calls -> {len(all_chunks)} chunks")

    split = manifest.split_by_call(all_chunks, cfg)
    gold = None
    if args.gold:
        gold = [json.loads(l) for l in open(args.gold, encoding="utf-8")]
        print(f"loaded {len(gold)} gold records (oversample x{cfg.split.gold_oversample})")

    train_records = manifest.merge_gold(split["train"], gold or [], cfg)
    test_records = [c.to_record() for c in split["test"]]
    review = [c.to_record() for c in all_chunks if c.tier == "review"]

    n_train = manifest.write_jsonl(train_records, out_dir / "train.jsonl")
    n_test = manifest.write_jsonl(test_records, out_dir / "test.jsonl")
    n_review = manifest.write_jsonl(review, out_dir / "review_queue.jsonl")

    # quick listen-check sidecar: audio_path <tab> tier(conf) <tab> text
    lines = [f"{c.audio_path}\t{c.tier}({(c.confidence or 0):.2f})\t{c.text}"
             for c in all_chunks if c.audio_path]
    (out_dir / "clips.tsv").write_text("\n".join(lines), encoding="utf-8")

    stats = {
        "calls": len(pairs),
        "chunks": len(all_chunks),
        "dropped_short": dropped_short,
        "train": n_train,
        "test": n_test,
        "review_queue": n_review,
        "tiers": tier_counts,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(out_dir, "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
