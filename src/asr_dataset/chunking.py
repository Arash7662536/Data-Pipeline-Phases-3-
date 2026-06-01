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

    # Speech regions per used channel (once). Used to split a turn window at long
    # internal silences and trim it to where the speaker is actually talking.
    channel_speech: dict[int, list[tuple[float, float]]] = {}
    if cfg.chunk.split_long_silence:
        for ch in {speaker_channel.get(sp, 0) for sp in by_speaker}:
            if 0 <= ch < channels.shape[0]:
                channel_speech[ch] = vad.speech_timestamps(channels[ch], sr)

    chunks: list[Chunk] = []
    for speaker, items in by_speaker.items():
        ch = speaker_channel.get(speaker, 0)
        items.sort(key=lambda x: x[1].start)
        for group in _group_speaker_turns(items, cfg):
            for text, idxs, start, end in _split_group(group, channel_speech.get(ch), cfg):
                chunks.append(Chunk(
                    call_id=call_id, channel=ch, speaker=speaker,
                    start=start, end=end, text=text, segment_indices=idxs,
                ))

    # annotate overlap; clip_duration / silence_ratio / chars_per_sec are filled
    # in the pipeline once the clip is rendered (and edge-trimmed).
    for c in chunks:
        c.has_overlap = any(segments[i].overlap for i in c.segment_indices)
        c.overlap_fraction = _overlap_fraction(c, segments)
    return chunks


def _split_group(group, regions, cfg: Config):
    """Yield (text, seg_idxs, start, end) pieces for one merged group.

    Coarse turn windows can span long internal silence (the other speaker
    talking, or a hold) that no text covers. We cut the window at silences
    >= split_silence_s on the speaker's own channel, then attach each transcript
    turn to the speech cluster it overlaps most. A turn whose audio drifted out
    of its labeled window lands in a tiny/empty cluster and is caught downstream
    by the duration / chars-per-sec / silence filters."""
    g0, g1 = group[0][1].start, group[-1][1].end
    whole = (" ".join(s.text for _, s in group).strip(), [i for i, _ in group], g0, g1)
    if not regions or not cfg.chunk.split_long_silence:
        return [whole]

    inwin = [(max(s, g0), min(e, g1)) for s, e in regions
             if min(e, g1) > max(s, g0)]
    if not inwin:
        return [whole]

    # cluster speech regions separated by < split_silence_s
    clusters = [[inwin[0]]]
    for s, e in inwin[1:]:
        if s - clusters[-1][-1][1] >= cfg.chunk.split_silence_s:
            clusters.append([(s, e)])
        else:
            clusters[-1].append((s, e))

    bounds = [(cl[0][0], cl[-1][1]) for cl in clusters]
    if len(bounds) == 1:                       # tight: just trim to speech extent
        return [(whole[0], whole[1], bounds[0][0], bounds[0][1])]

    # assign each turn to the cluster it overlaps most
    assigned: dict[int, list[tuple[int, Segment]]] = {ci: [] for ci in range(len(bounds))}
    for i, seg in group:
        ov = [_overlap(seg.start, seg.end, b0, b1) for b0, b1 in bounds]
        best = max(range(len(bounds)), key=lambda c: ov[c])
        if ov[best] > 0:
            assigned[best].append((i, seg))

    out = []
    for ci, (b0, b1) in enumerate(bounds):
        members = assigned[ci]
        if not members:
            continue
        out.append((" ".join(s.text for _, s in members).strip(),
                    [i for i, _ in members], b0, b1))
    return out or [whole]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


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
