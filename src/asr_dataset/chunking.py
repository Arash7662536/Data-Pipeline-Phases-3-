"""Chunking: turn Gemini's diarization turns into Whisper-ready training clips.

The transcript timestamps are CONTIGUOUS, coarse (rounded to ~1s) turn
boundaries — they partition the timeline by speaker, they are not tight speech
bounds. But the channels are cleanly separated (one speaker per channel), so a
window on the speaker's own channel contains only that speaker.

Strategy (per channel, per speaker):
  1. Take that speaker's turns in time order.
  2. Greedily merge adjacent turns into a window while the merged duration stays
     <= max_s, the inter-turn gap is small, and we're under ~target_mean_s.
  3. Slice the speaker's channel over [window_start, window_end] AS-IS.

We deliberately do NOT pad into neighbouring turns or drop "internal silence":
both corrupt the audio<->text correspondence (extra / missing words). The only
cleanup is conservative leading/trailing silence trimming, done later in the
pipeline. Genuinely mislabeled turns (coarse timestamps that drift off the
audio) are caught downstream by the alignment + second-ASR confidence tiers.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .types import Segment, Chunk
from . import vad


def build_chunks(
    call_id: str,
    segments: list[Segment],
    channels: np.ndarray,
    sr: int,
    speaker_channel: dict[str, int],
    cfg: Config,
) -> list[Chunk]:
    by_speaker: dict[str, list[tuple[int, Segment]]] = {}
    for i, seg in enumerate(segments):
        if not seg.text or not seg.speaker:
            continue
        by_speaker.setdefault(seg.speaker, []).append((i, seg))

    chunks: list[Chunk] = []
    for speaker, items in by_speaker.items():
        ch = speaker_channel.get(speaker, 0)
        items.sort(key=lambda x: x[1].start)
        for group in _group_speaker_turns(items, cfg):
            chunks.append(Chunk(
                call_id=call_id, channel=ch, speaker=speaker,
                start=group[0][1].start, end=group[-1][1].end,
                text=" ".join(s.text for _, s in group).strip(),
                segment_indices=[i for i, _ in group],
            ))

    # annotate overlap; clip_duration / silence_ratio / chars_per_sec are filled
    # in the pipeline once the clip is rendered (and edge-trimmed).
    for c in chunks:
        c.has_overlap = any(segments[i].overlap for i in c.segment_indices)
        c.overlap_fraction = _overlap_fraction(c, segments)
    return chunks


def _group_speaker_turns(items, cfg: Config) -> list[list[tuple[int, Segment]]]:
    """Greedily merge adjacent same-speaker turns into groups (text grouping).
    No min-duration drop here: short-but-real turns are kept and filtered later
    on their ACTUAL speech length."""
    groups: list[list[tuple[int, Segment]]] = []
    cur: list[tuple[int, Segment]] = []

    for i, seg in items:
        if not cur:
            cur.append((i, seg))
            continue
        prev_end = cur[-1][1].end
        gap = seg.start - prev_end
        merged_dur = seg.end - cur[0][1].start

        too_long = merged_dur > cfg.chunk.max_s
        gap_too_big = gap > cfg.chunk.merge_gap_max_s
        past_target = (prev_end - cur[0][1].start) >= cfg.chunk.target_mean_s

        if too_long or gap_too_big or past_target:
            groups.append(cur)
            cur = [(i, seg)]
        else:
            cur.append((i, seg))
    if cur:
        groups.append(cur)
    return groups


def _overlap_fraction(chunk: Chunk, segments: list[Segment]) -> float:
    if chunk.duration <= 0:
        return 0.0
    ov = sum(segments[i].duration for i in chunk.segment_indices if segments[i].overlap)
    return float(min(1.0, ov / chunk.duration))


def fill_clip_metrics(chunk: Chunk, audio: np.ndarray, sr: int) -> None:
    """Set clip_duration, silence_ratio and chars_per_sec from the rendered clip."""
    chunk.clip_duration = len(audio) / sr if sr else 0.0
    chunk.silence_ratio = vad.silence_ratio(audio, sr)
    if chunk.clip_duration and chunk.clip_duration > 0:
        chunk.chars_per_sec = len(chunk.text.replace(" ", "")) / chunk.clip_duration


# Back-compat alias (older callers / tests).
def fill_silence_ratio(chunk: Chunk, audio: np.ndarray, sr: int) -> None:
    fill_clip_metrics(chunk, audio, sr)


def length_buckets(chunks: list[Chunk], cfg: Config) -> dict[str, int]:
    """For the stats report: how the chunk durations fall across target bands."""
    b = {"short": 0, "mid": 0, "long": 0, "under_min": 0, "over_max": 0}
    for c in chunks:
        d = c.effective_duration
        if d < cfg.chunk.min_s:
            b["under_min"] += 1
        elif d > cfg.chunk.max_s:
            b["over_max"] += 1
        elif cfg.chunk.short_band[0] <= d < cfg.chunk.short_band[1]:
            b["short"] += 1
        elif cfg.chunk.mid_band[0] <= d < cfg.chunk.mid_band[1]:
            b["mid"] += 1
        else:
            b["long"] += 1
    return b
