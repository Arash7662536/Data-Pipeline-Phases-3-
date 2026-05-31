"""Final assembly: enforce the overlap cap, split by call (no leakage),
oversample the gold set, and write a JSONL manifest (+ optional HF dataset).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Optional

from .config import Config
from .types import Chunk


def enforce_overlap_cap(chunks: list[Chunk], cfg: Config, seed: int = 0) -> list[Chunk]:
    """Keep overlap chunks to <= max_fraction_of_dataset. We keep the highest
    confidence overlap chunks and drop the rest."""
    if not cfg.overlap.keep:
        return [c for c in chunks if not c.has_overlap]

    overlap = [c for c in chunks if c.has_overlap]
    clean = [c for c in chunks if not c.has_overlap]
    if not overlap:
        return chunks

    total = len(chunks)
    cap = int(cfg.overlap.max_fraction_of_dataset * total)
    if len(overlap) <= cap:
        return chunks

    overlap.sort(key=lambda c: (c.confidence or 0.0), reverse=True)
    kept = overlap[:cap]
    return clean + kept


def split_by_call(chunks: list[Chunk], cfg: Config) -> dict[str, list[Chunk]]:
    """Train/test split at the CALL level so no call appears in both."""
    rng = random.Random(cfg.split.seed)
    call_ids = sorted({c.call_id for c in chunks})
    rng.shuffle(call_ids)
    n_test = max(1, int(len(call_ids) * cfg.split.test_fraction)) if call_ids else 0
    test_calls = set(call_ids[:n_test])

    train = [c for c in chunks if c.call_id not in test_calls]
    test = [c for c in chunks if c.call_id in test_calls]
    return {"train": train, "test": test}


def merge_gold(train: list[Chunk], gold_records: list[dict], cfg: Config) -> list[dict]:
    """Mix the clean gold set into train, oversampled, so silver data doesn't
    swamp the clean anchor. Gold records are passed through as-is (already
    chunked/verified) and tagged tier='gold'."""
    out = [c.to_record() for c in train]
    for g in gold_records:
        g = {**g, "tier": "gold", "loss_weight": float(g.get("loss_weight", 1.0)),
             "source": "gold"}
        out.extend([dict(g) for _ in range(max(1, cfg.split.gold_oversample))])
    return out


def write_jsonl(records: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def to_hf_dataset(jsonl_path: str | Path):
    """Optional: load the manifest as a HF Dataset with an audio column ready
    for Whisper feature extraction. Requires `datasets`."""
    from datasets import Dataset, Audio
    rows = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
    ds = Dataset.from_list(rows)
    if rows and rows[0].get("audio_path"):
        ds = ds.rename_column("audio_path", "audio").cast_column("audio", Audio(sampling_rate=16_000))
    return ds
