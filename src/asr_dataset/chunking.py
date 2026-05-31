"""Chunking: turn Gemini's short diarization turns into Whisper-ready training
clips. Strategy (per channel, per speaker):

  1. Take that speaker's turns in time order.
  2. Greedily merge adjacent turns into a growing chunk while:
       - the merged duration stays <= max_s, and
       - the inter-turn gap is small (merge_gap_max_s).
  3. Close a chunk when it reaches ~target_mean_s or merging would exceed max_s.
  4. Drop sub-min_s leftovers (or attach them to a neighbour).

This produces multi-turn, continuous-speech windows (8-28s) instead of clipped
one-liners, which is what keeps large-v3's long-form recall intact.
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
    chunks: list[Chunk] = []
    by_speaker: dict[str, list[tuple[int, Segment]]] = {}
    for i, seg in enumerate(segments):
        if not seg.text or not seg.speaker:
            continue
        by_speaker.setdefault(seg.speaker, []).append((i, seg))

    for speaker, items in by_speaker.items():
        ch = speaker_channel.get(speaker, 0)
        items.sort(key=lambda x: x[1].start)
        chunks.extend(_merge_speaker_turns(call_id, speaker, ch, items, cfg))

    # annotate overlap / chars-per-sec; silence ratio is filled later once audio is rendered
    for c in chunks:
        c.has_overlap = any(segments[i].overlap for i in c.segment_indices)
        c.overlap_fraction = _overlap_fraction(c, segments)
        if c.duration > 0:
            c.chars_per_sec = len(c.text.replace(" ", "")) / c.duration
    return chunks


def _merge_speaker_turns(call_id, speaker, ch, items, cfg: Config) -> list[Chunk]:
    out: list[Chunk] = []
    cur: list[tuple[int, Segment]] = []

    def flush():
        if not cur:
            return
        start = cur[0][1].start
        end = cur[-1][1].end
        text = " ".join(s.text for _, s in cur).strip()
        idxs = [i for i, _ in cur]
        dur = end - start
        if dur >= cfg.chunk.min_s:
            out.append(Chunk(
                call_id=call_id, channel=ch, speaker=speaker,
                start=start, end=end, text=text, segment_indices=idxs,
            ))
        # sub-min leftovers are dropped here; see README on attaching instead
        cur.clear()

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
            flush()
            cur.append((i, seg))
        else:
            cur.append((i, seg))
    flush()
    return out


def _overlap_fraction(chunk: Chunk, segments: list[Segment]) -> float:
    if chunk.duration <= 0:
        return 0.0
    ov = sum(segments[i].duration for i in chunk.segment_indices if segments[i].overlap)
    return float(min(1.0, ov / chunk.duration))


def fill_silence_ratio(chunk: Chunk, audio: np.ndarray, sr: int) -> None:
    chunk.silence_ratio = vad.silence_ratio(audio, sr)


def length_buckets(chunks: list[Chunk], cfg: Config) -> dict[str, int]:
    """For the stats report: how the chunk durations fall across target bands."""
    b = {"short": 0, "mid": 0, "long": 0, "under_min": 0, "over_max": 0}
    for c in chunks:
        d = c.duration
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
