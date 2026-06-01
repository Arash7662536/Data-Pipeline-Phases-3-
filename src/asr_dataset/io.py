"""Loading the Gemini JSON and the dual-channel WAV, plus channel<->speaker
mapping (the key step that gives us free speaker separation)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .types import Segment


def load_call_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_segments(call: dict) -> list[Segment]:
    """Pull the `transcript` array out of the JSON (handles the
    root.analysis.transcript / root.raw_response.transcript shapes seen in the data)."""
    transcript = None
    for getter in (
        lambda c: c["analysis"]["transcript"],
        lambda c: c["raw_response"]["transcript"],
        lambda c: c["transcript"],
    ):
        try:
            transcript = getter(call)
            if transcript:
                break
        except (KeyError, TypeError):
            continue
    if not transcript:
        return []
    return [Segment.from_json(d) for d in transcript]


def call_id_of(call: dict, fallback: str = "unknown") -> str:
    af = call.get("audio_file", {}) or {}
    return (
        af.get("ccplatform_url")
        or af.get("id")
        or call.get("request_id")
        or fallback
    )


def load_stereo(path: str | Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Return (channels, sr) where channels has shape (n_channels, n_samples).
    Resamples to target_sr (telephony is usually 8k narrowband -> upsample to 16k).

    Falls back to ffmpeg when soundfile can't read the format (μ-law, GSM, etc.).
    """
    audio, sr = _sf_read(path)
    if audio is None:
        audio, sr = _ffmpeg_read(path)

    audio = audio.T.astype(np.float32)   # (n_channels, n_samples)
    if sr != target_sr:
        audio = _resample(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


def _sf_read(path: str | Path):
    """Try soundfile; return (audio_2d, sr) or (None, None) on failure."""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), always_2d=True)
        return audio, sr
    except Exception:
        return None, None


def _ffmpeg_read(path: str | Path):
    """Decode via ffmpeg subprocess — handles μ-law, GSM, MP3, etc."""
    import subprocess
    import tempfile, os

    path = str(path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", path,
             "-ar", "16000", "-ac", "2",    # always ask for stereo; mono file → dup channel
             "-sample_fmt", "s16", tmp_path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {path}:\n{result.stderr.decode(errors='replace')}"
            )
        import soundfile as sf
        audio, sr = sf.read(tmp_path, always_2d=True)
        return audio, sr
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    try:
        import librosa
        return np.stack([librosa.resample(ch, orig_sr=sr, target_sr=target_sr)
                         for ch in audio])
    except ImportError:
        # linear fallback if librosa unavailable
        n_new = int(round(audio.shape[1] * target_sr / sr))
        x_old = np.linspace(0, 1, audio.shape[1])
        x_new = np.linspace(0, 1, n_new)
        return np.stack([np.interp(x_new, x_old, ch) for ch in audio])


def map_channels_to_speakers(
    channels: np.ndarray,
    sr: int,
    segments: list[Segment],
) -> dict[str, int]:
    """Decide which physical channel belongs to which speaker using energy.

    For each speaker, sum frame energy in every channel during *that speaker's*
    turns. The channel where the speaker is loudest is theirs. Robust to L/R
    being swapped between recordings.
    """
    n_ch = channels.shape[0]
    if n_ch == 1:
        # mono fallback: everyone on channel 0 (you lose the overlap advantage)
        return {sp: 0 for sp in {s.speaker for s in segments}}

    speakers = sorted({s.speaker for s in segments if s.speaker})
    energy = {sp: np.zeros(n_ch) for sp in speakers}

    for seg in segments:
        if seg.speaker not in energy:
            continue
        a = int(seg.start * sr)
        b = int(seg.end * sr)
        if b <= a:
            continue
        for ch in range(n_ch):
            energy[seg.speaker][ch] += float(np.sum(channels[ch, a:b] ** 2))

    mapping: dict[str, int] = {}
    used: set[int] = set()
    # assign greedily by strongest preference so two speakers don't grab the same channel
    order = sorted(speakers, key=lambda sp: -float(energy[sp].max()))
    for sp in order:
        ranked = np.argsort(-energy[sp])
        choice = next((int(c) for c in ranked if int(c) not in used), int(ranked[0]))
        mapping[sp] = choice
        used.add(choice)
    return mapping


def slice_channel(channels: np.ndarray, channel: int, sr: int,
                  start: float, end: float, pad_s: float = 0.0) -> np.ndarray:
    a = max(0, int((start - pad_s) * sr))
    b = min(channels.shape[1], int((end + pad_s) * sr))
    return channels[channel, a:b].copy()


def trim_silence_edges(audio: np.ndarray, sr: int, margin_s: float = 0.2) -> np.ndarray:
    """Trim leading/trailing silence only, keeping a margin and ALL internal
    audio. Coarse timestamps pad turns with silence (and sometimes shift the
    speech to one end); this tightens the clip without ever dropping a word or
    reaching into a neighbouring turn. No-ops if VAD finds no speech.
    """
    from . import vad
    if audio.size == 0:
        return audio
    regions = vad.speech_timestamps(audio, sr)
    if not regions:
        return audio
    total = len(audio) / sr
    s = max(0.0, regions[0][0] - margin_s)
    e = min(total, regions[-1][1] + margin_s)
    a, b = int(s * sr), int(e * sr)
    return audio[a:b].copy() if b > a else audio


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
