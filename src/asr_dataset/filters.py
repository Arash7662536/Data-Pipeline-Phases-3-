"""Quality gates. Each chunk is checked against cheap filters; rejected chunks
carry a reason for the stats report. These run BEFORE the overlap cap and
confidence tiering."""
from __future__ import annotations

from typing import Optional

from .config import Config
from .types import Chunk
from .agreement import persian_char_ratio


def check(chunk: Chunk, cfg: Config, audio_quality: Optional[str] = None) -> Optional[str]:
    """Return None if the chunk passes, else a short rejection reason."""
    f = cfg.filt

    if len(chunk.text) < f.min_chars:
        return "too_short_text"

    dur = chunk.effective_duration
    if dur < cfg.chunk.min_s:
        return "under_min_duration"
    if dur > cfg.chunk.max_s:
        return "over_max_duration"

    if f.require_audio_quality_good and audio_quality and audio_quality != "good":
        return f"audio_quality:{audio_quality}"

    if chunk.silence_ratio is not None and chunk.silence_ratio > f.max_silence_ratio:
        return "too_silent"

    if chunk.chars_per_sec is not None:
        if chunk.chars_per_sec < f.cps_min:
            return "cps_too_low"
        if chunk.chars_per_sec > f.cps_max:
            return "cps_too_high"

    if persian_char_ratio(chunk.text) < f.min_persian_char_ratio:
        return "not_persian"

    if chunk.overlap_fraction > cfg.overlap.drop_if_overlap_fraction_above:
        return "excess_overlap"

    return None
