#!/usr/bin/env python
"""Convert the built JSONL manifest(s) into a HuggingFace Arrow dataset.

Keeps only real speakers (drops the "Silence" pseudo-speaker) and emits exactly
three columns:

    audio       -> {"path": str, "array": np.ndarray, "sampling_rate": int}
    text        -> str
    confidence  -> float        (the per-chunk score, i.e. the `review(...)` value)

The Audio feature embeds the decoded waveform when saved, so the resulting Arrow
dataset is self-contained (no dependency on the original wav paths).

Usage:
    # whole build dir -> a DatasetDict with train/test splits
    python scripts/build_arrow_dataset.py \
        --manifest build/dataset \
        --out build/arrow_dataset

    # a single manifest file -> a single Dataset
    python scripts/build_arrow_dataset.py \
        --manifest build/dataset/train.jsonl \
        --out build/arrow_train
"""
import argparse
import json
import sys
from pathlib import Path


def resolve_audio_path(raw: str, manifest_dir: Path) -> Path | None:
    """audio_path may be absolute, relative-to-cwd, or relative-to-build-dir.
    Return the first one that exists, else None."""
    p = Path(raw)
    candidates = [p]
    if not p.is_absolute():
        candidates += [Path.cwd() / p, manifest_dir / p, manifest_dir.parent / p]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_rows(jsonl_path: Path, keep_silence: bool) -> tuple[list[dict], dict]:
    """Read one manifest, filter, and shape into {audio, text, confidence} rows."""
    manifest_dir = jsonl_path.parent
    rows: list[dict] = []
    stats = {"total": 0, "dropped_silence": 0, "missing_audio": 0,
             "missing_text": 0, "kept": 0}

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            stats["total"] += 1

            speaker = (r.get("speaker") or "").strip().lower()
            if not keep_silence and speaker == "silence":
                stats["dropped_silence"] += 1
                continue

            text = (r.get("text") or "").strip()
            if not text:
                stats["missing_text"] += 1
                continue

            ap = r.get("audio_path") or r.get("audio")
            resolved = resolve_audio_path(ap, manifest_dir) if ap else None
            if resolved is None:
                stats["missing_audio"] += 1
                continue

            conf = r.get("confidence")
            rows.append({
                "audio": str(resolved),
                "text": text,
                "confidence": float(conf) if conf is not None else None,
            })
            stats["kept"] += 1
    return rows, stats


def make_dataset(rows: list[dict], sr: int):
    from datasets import Dataset, Audio, Features, Value
    features = Features({
        "audio": Audio(sampling_rate=sr),
        "text": Value("string"),
        "confidence": Value("float32"),
    })
    return Dataset.from_list(rows, features=features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="a .jsonl manifest, or a build dir containing the split jsonls")
    ap.add_argument("--out", required=True, help="output dir for the Arrow dataset")
    ap.add_argument("--splits", default="train,test",
                    help="when --manifest is a dir, which <name>.jsonl to load (comma-sep)")
    ap.add_argument("--sr", type=int, default=16000, help="audio sampling rate")
    ap.add_argument("--keep-silence", action="store_true",
                    help="keep the 'Silence' pseudo-speaker (dropped by default)")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    out = Path(args.out)

    # Figure out which jsonl files to load and under what split name.
    if manifest.is_dir():
        jsonls = {}
        for name in (s.strip() for s in args.splits.split(",") if s.strip()):
            p = manifest / f"{name}.jsonl"
            if p.exists():
                jsonls[name] = p
            else:
                print(f"  [skip] no {p}", file=sys.stderr)
        if not jsonls:
            sys.exit(f"no manifest jsonls found in {manifest}")
    else:
        jsonls = {manifest.stem: manifest}

    splits = {}
    for name, path in jsonls.items():
        rows, stats = load_rows(path, keep_silence=args.keep_silence)
        print(f"{name}: kept {stats['kept']}/{stats['total']} "
              f"(dropped silence={stats['dropped_silence']}, "
              f"missing audio={stats['missing_audio']}, "
              f"missing text={stats['missing_text']})")
        if rows:
            splits[name] = make_dataset(rows, args.sr)

    if not splits:
        sys.exit("nothing to write (all rows filtered out)")

    out.mkdir(parents=True, exist_ok=True)
    if len(splits) == 1 and manifest.is_file():
        ds = next(iter(splits.values()))
        ds.save_to_disk(str(out))
        print(f"\nwrote Dataset with {len(ds)} rows -> {out}")
    else:
        from datasets import DatasetDict
        dd = DatasetDict(splits)
        dd.save_to_disk(str(out))
        for name, ds in dd.items():
            print(f"  {name}: {len(ds)} rows")
        print(f"\nwrote DatasetDict -> {out}")

    print("\nload it back with:")
    print("  from datasets import load_from_disk")
    print(f"  ds = load_from_disk({str(out)!r})")


if __name__ == "__main__":
    main()
