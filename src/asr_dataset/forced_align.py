"""Forced-alignment chunking — the robust path.

Gemini's per-turn timestamps are coarse (~1s rounding) and often drift off the
audio, which is what broke the timestamp-trusting chunkers. This module ignores
them. Because the call is dual-channel with clean speaker separation, we:

  1. Split the stereo wav into one mono channel per speaker.
  2. Concatenate that speaker's transcript text (time order) into one reference.
  3. Forced-align text <-> channel audio with a CTC aligner
     (MahmoudAshraf/mms-300m-1130-forced-aligner) to get WORD-level timestamps
     computed from the audio itself.
  4. Cut the word stream into Whisper-ready chunks at real pauses.

Each word carries an alignment score, so chunks where Gemini's text doesn't
match the audio (hallucinated / wrong) score low and are routed to review.

Requires: pip install ctc-forced-aligner torch torchaudio
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Config
from .types import Chunk
from . import io, confidence


@dataclass
class Word:
    text: str
    start: float
    end: float
    score: float


# --------------------------------------------------------------------------- #
# Aligner wrapper (loads the model once)
# --------------------------------------------------------------------------- #
class ForcedAligner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._tok = None
        self._device = "cpu"

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from ctc_forced_aligner import load_alignment_model

        fa = self.cfg.forced_align
        device = fa.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[forced_align] CUDA not available -> using CPU")
            device = "cpu"
        dtype = torch.float16 if (device == "cuda" and fa.dtype == "float16") else torch.float32
        self._device = device
        self._model, self._tok = load_alignment_model(
            device, model_path=fa.model_id, dtype=dtype)

    def align(self, audio: np.ndarray, sr: int, text: str) -> list[Word]:
        """Align `text` against a single-channel `audio` array. Returns
        word-level timestamps (in seconds, relative to the clip)."""
        text = (text or "").strip()
        if not text or audio.size == 0:
            return []
        self._ensure_loaded()

        import torch
        from ctc_forced_aligner import (
            generate_emissions, preprocess_text, get_alignments,
            get_spans, postprocess_results,
        )
        fa = self.cfg.forced_align

        wav = torch.as_tensor(np.ascontiguousarray(audio), dtype=self._model.dtype,
                              device=self._model.device)
        emissions, stride = generate_emissions(self._model, wav, batch_size=fa.batch_size)
        tokens_starred, text_starred = preprocess_text(
            text, romanize=fa.romanize, language=fa.language)
        segments, scores, blank = get_alignments(emissions, tokens_starred, self._tok)
        spans = get_spans(tokens_starred, segments, blank)
        word_ts = postprocess_results(text_starred, spans, stride, scores)

        words = [Word(w["text"], float(w["start"]), float(w["end"]),
                      float(w.get("score", 1.0))) for w in word_ts]
        floor = fa.min_word_score
        if floor > 0:
            # only strip low-confidence words at the very edges (keep interior)
            while words and words[0].score < floor:
                words.pop(0)
            while words and words[-1].score < floor:
                words.pop()
        return words


# --------------------------------------------------------------------------- #
# Word-stream -> chunks
# --------------------------------------------------------------------------- #
def chunk_words(words: list[Word], cfg: Config) -> list[list[Word]]:
    """Greedily pack words into chunks <= max_s, preferring to cut at the
    largest word gap once past target_mean_s."""
    c = cfg.chunk
    out: list[list[Word]] = []
    i, n = 0, len(words)
    while i < n:
        start = words[i].start
        # furthest j that still fits under max_s
        j = i + 1
        while j < n and (words[j].end - start) <= c.max_s:
            j += 1
        # within [i+1, j), once past target, cut at the biggest pause
        cut, best_gap = j, -1.0
        for k in range(i + 1, j):
            if (words[k - 1].end - start) < c.target_mean_s:
                continue
            gap = words[k].start - words[k - 1].end
            if gap >= c.pause_cut_s and gap > best_gap:
                best_gap, cut = gap, k
        out.append(words[i:cut])
        i = cut
    return out


# --------------------------------------------------------------------------- #
# Per-call orchestration
# --------------------------------------------------------------------------- #
def prepare_channels(json_path, wav_path, cfg: Config, prep_dir: Optional[Path] = None):
    """Split a call into per-speaker (channel audio, reference text). Optionally
    writes the 2 channel wavs + 2 txt files as artifacts."""
    call = io.load_call_json(json_path)
    segments = io.extract_segments(call)
    # Fall back to the WAV stem (a unique ULID), not the JSON stem: in some
    # corpora every file is named report.json, so the JSON stem collides.
    call_id = io.call_id_of(call, fallback=Path(wav_path).stem)
    if not segments:
        return call_id, [], None, 0

    channels, sr = io.load_stereo(wav_path, target_sr=cfg.audio.target_sr)
    spk_ch = io.map_channels_to_speakers(channels, sr, segments)

    by_spk: dict[str, list] = {}
    for s in segments:
        if s.text and s.speaker:
            by_spk.setdefault(s.speaker, []).append(s)

    safe = "".join(c if c.isalnum() else "_" for c in call_id)[:80]
    out = []
    for speaker, segs in by_spk.items():
        ch = spk_ch.get(speaker, 0)
        segs.sort(key=lambda x: x.start)
        text = " ".join(s.text for s in segs).strip()
        if prep_dir is not None:
            prep_dir = Path(prep_dir)
            wp = prep_dir / f"{safe}_{speaker}.wav"
            io.write_wav(wp, channels[ch], sr)
            wp.with_suffix(".txt").write_text(text, encoding="utf-8")
        out.append((speaker, ch, text))
    return call_id, out, (channels, sr), len(segments)


def process_call_aligned(json_path, wav_path, cfg: Config, aligner: ForcedAligner,
                         out_audio_dir, prep_dir: Optional[Path] = None) -> list[Chunk]:
    call_id, per_spk, loaded, _ = prepare_channels(json_path, wav_path, cfg, prep_dir)
    if not loaded:
        return []
    channels, sr = loaded
    safe = "".join(c if c.isalnum() else "_" for c in call_id)[:80]

    chunks: list[Chunk] = []
    for speaker, ch, text in per_spk:
        words = aligner.align(channels[ch], sr, text)
        for gi, group in enumerate(chunk_words(words, cfg)):
            if not group:
                continue
            start, end = group[0].start, group[-1].end
            mean_score = float(np.mean([w.score for w in group]))
            chunk = Chunk(
                call_id=call_id, channel=ch, speaker=speaker,
                start=start, end=end,
                text=" ".join(w.text for w in group).strip(),
            )
            chunk.confidence = mean_score
            confidence.assign_tier(chunk, cfg)

            clip = io.slice_channel(channels, ch, sr, start, end,
                                    pad_s=cfg.audio.edge_pad_s)
            chunk.clip_duration = len(clip) / sr if sr else 0.0
            if chunk.clip_duration:
                chunk.chars_per_sec = len(chunk.text.replace(" ", "")) / chunk.clip_duration
            if out_audio_dir is not None:
                ap = Path(out_audio_dir) / f"{safe}_{speaker}_{gi:03d}.wav"
                io.write_wav(ap, clip, sr)
                chunk.audio_path = str(ap)
            chunks.append(chunk)
    return chunks
