"""End-to-end orchestration. `process_call` runs one (json, wav) pair through
every stage; `process_corpus` walks a directory and assembles the manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import io, chunking, filters, confidence, manifest
from .alignment import Aligner
from .second_asr import SecondASR
from .agreement import agreement_score, word_disagreements
from .config import Config
from .types import Chunk


@dataclass
class CallResult:
    call_id: str
    kept: list[Chunk] = field(default_factory=list)
    rejected: list[tuple[Chunk, str]] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)


def process_call(
    json_path: str | Path,
    wav_path: str | Path,
    cfg: Config,
    aligner: Optional[Aligner] = None,
    second_asr: Optional[SecondASR] = None,
    out_audio_dir: Optional[str | Path] = None,
) -> CallResult:
    call = io.load_call_json(json_path)
    segments = io.extract_segments(call)
    call_id = io.call_id_of(call, fallback=Path(json_path).stem)
    audio_quality = (call.get("raw_response", {}) or {}).get("audio_quality")

    res = CallResult(call_id=call_id)
    if not segments:
        return res

    channels, sr = io.load_stereo(wav_path, target_sr=cfg.audio.target_sr)
    spk_ch = io.map_channels_to_speakers(channels, sr, segments)

    chunks = chunking.build_chunks(call_id, segments, channels, sr, spk_ch, cfg)

    # render audio, score, and filter each chunk
    safe_id = "".join(c if c.isalnum() else "_" for c in call_id)[:80]
    for k, ch in enumerate(chunks):
        clip = io.slice_channel(channels, ch.channel, sr, ch.start, ch.end,
                                pad_s=cfg.audio.edge_pad_s)
        if cfg.chunk.trim_silence_edges:
            clip = io.trim_silence_edges(clip, sr, margin_s=cfg.chunk.trim_margin_s)
        chunking.fill_clip_metrics(ch, clip, sr)

        # cheap filters first (skip scoring on chunks we'll throw away)
        reason = filters.check(ch, cfg, audio_quality=audio_quality)
        if reason:
            res.rejected.append((ch, reason))
            continue

        if aligner is not None:
            ch.alignment_score = aligner.score(ch.text, clip, sr)
        if second_asr is not None and second_asr.enabled:
            ch.second_asr_text = second_asr.transcribe(clip, sr)
            ch.agreement_score = agreement_score(ch.text, ch.second_asr_text)
            res.disagreements.extend(word_disagreements(ch.text, ch.second_asr_text))

        ch.confidence = confidence.compute_confidence(ch, cfg)
        confidence.assign_tier(ch, cfg)

        if out_audio_dir is not None:
            ap = Path(out_audio_dir) / f"{safe_id}_ch{ch.channel}_{k:03d}.wav"
            io.write_wav(ap, clip, sr)
            ch.audio_path = str(ap)

        res.kept.append(ch)

    return res


def process_corpus(
    pairs: list[tuple[str, str]],
    cfg: Config,
    out_dir: str | Path,
    gold_records: Optional[list[dict]] = None,
) -> dict:
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"

    aligner = Aligner(cfg) if cfg.align.enabled else None
    second = SecondASR(cfg) if cfg.second_asr.enabled else None

    all_kept: list[Chunk] = []
    rejected_counts: dict[str, int] = {}
    for json_path, wav_path in pairs:
        r = process_call(json_path, wav_path, cfg, aligner, second, audio_dir)
        all_kept.extend(r.kept)
        for _, reason in r.rejected:
            rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

    capped = manifest.enforce_overlap_cap(all_kept, cfg)
    split = manifest.split_by_call(capped, cfg)

    train_records = manifest.merge_gold(split["train"], gold_records or [], cfg)
    test_records = [c.to_record() for c in split["test"]]

    n_train = manifest.write_jsonl(train_records, out_dir / "train.jsonl")
    n_test = manifest.write_jsonl(test_records, out_dir / "test.jsonl")
    # review queue = everything that landed in the review tier (kept but weight 0)
    review = [c.to_record() for c in capped if c.tier == "review"]
    n_review = manifest.write_jsonl(review, out_dir / "review_queue.jsonl")

    return {
        "calls": len(pairs),
        "chunks_kept": len(all_kept),
        "after_overlap_cap": len(capped),
        "train": n_train,
        "test": n_test,
        "review_queue": n_review,
        "rejected": rejected_counts,
        "tiers": _tier_counts(capped),
    }


def _tier_counts(chunks: list[Chunk]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in chunks:
        out[c.tier or "?"] = out.get(c.tier or "?", 0) + 1
    return out
