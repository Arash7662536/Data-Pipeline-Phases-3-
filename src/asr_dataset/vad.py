"""Voice-activity detection, used to find clean silence boundaries to cut on
and to estimate silence ratio per chunk. Prefers Silero; falls back to a
simple energy gate so the module is usable without torch.hub access."""
from __future__ import annotations

import numpy as np


def speech_timestamps(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Return list of (start_s, end_s) speech regions on a single channel."""
    try:
        return _silero(audio, sr)
    except Exception:
        return _energy_vad(audio, sr)


def silence_ratio(audio: np.ndarray, sr: int) -> float:
    """Fraction of the clip that is non-speech."""
    if audio.size == 0:
        return 1.0
    speech = speech_timestamps(audio, sr)
    total = len(audio) / sr
    spoken = sum(e - s for s, e in speech)
    return float(max(0.0, 1.0 - spoken / total)) if total > 0 else 1.0


def silence_gaps(audio: np.ndarray, sr: int, min_gap_s: float) -> list[tuple[float, float]]:
    """Return silence gaps (start_s, end_s) at least min_gap_s long.
    These are the only places chunking is allowed to cut."""
    speech = speech_timestamps(audio, sr)
    total = len(audio) / sr
    gaps: list[tuple[float, float]] = []
    prev = 0.0
    for s, e in speech:
        if s - prev >= min_gap_s:
            gaps.append((prev, s))
        prev = e
    if total - prev >= min_gap_s:
        gaps.append((prev, total))
    return gaps


# ---- backends -------------------------------------------------------------

_SILERO = None


def _silero(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    global _SILERO
    import torch
    if _SILERO is None:
        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        _SILERO = (model, utils)
    model, utils = _SILERO
    get_ts = utils[0]
    wav = torch.from_numpy(audio.astype(np.float32))
    ts = get_ts(wav, model, sampling_rate=sr)
    return [(t["start"] / sr, t["end"] / sr) for t in ts]


def _energy_vad(audio: np.ndarray, sr: int,
                frame_ms: int = 30, thresh_db: float = -35.0) -> list[tuple[float, float]]:
    frame = max(1, int(sr * frame_ms / 1000))
    n = len(audio) // frame
    if n == 0:
        return []
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
    db = 20 * np.log10(rms + 1e-10)
    active = db > thresh_db

    regions: list[tuple[float, float]] = []
    start = None
    for i, a in enumerate(active):
        t = i * frame / sr
        if a and start is None:
            start = t
        elif not a and start is not None:
            regions.append((start, t))
            start = None
    if start is not None:
        regions.append((start, n * frame / sr))
    return regions
