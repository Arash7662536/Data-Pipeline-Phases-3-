"""Forced alignment: align the Gemini text to the chunk audio and read off a
per-word acoustic posterior. Low posterior = the audio does not support that
word (likely wrong or misaligned). This is the *acoustic* half of confidence.

Backends:
  - whisperx: wav2vec2-based aligner, returns per-word scores.
  - torchaudio: CTC forced_align primitive.
  - none: returns None -> confidence falls back to agreement only.

Heavy deps are imported lazily so the rest of the repo runs without them.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import Config


class Aligner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._meta = None

    def score(self, text: str, audio: np.ndarray, sr: int) -> Optional[float]:
        """Return mean per-word alignment posterior in [0,1], or None if disabled
        / backend unavailable."""
        if not self.cfg.align.enabled or self.cfg.align.backend == "none":
            return None
        if not text.strip():
            return None
        try:
            if self.cfg.align.backend == "whisperx":
                return self._whisperx(text, audio, sr)
            if self.cfg.align.backend == "torchaudio":
                return self._torchaudio(text, audio, sr)
        except Exception:
            return None
        return None

    def _whisperx(self, text, audio, sr) -> Optional[float]:
        import whisperx
        if self._model is None:
            self._model, self._meta = whisperx.load_align_model(
                language_code=self.cfg.align.language, device=self.cfg.align.device
            )
        # one segment spanning the whole clip
        seg = [{"start": 0.0, "end": len(audio) / sr, "text": text}]
        result = whisperx.align(
            seg, self._model, self._meta, audio, self.cfg.align.device,
            return_char_alignments=False,
        )
        words = [w for s in result.get("segments", []) for w in s.get("words", [])]
        scores = [w["score"] for w in words if w.get("score") is not None]
        return float(np.mean(scores)) if scores else None

    def _torchaudio(self, text, audio, sr) -> Optional[float]:
        import torch
        import torchaudio
        from torchaudio.pipelines import MMS_FA as bundle  # multilingual CTC aligner

        if self._model is None:
            self._model = bundle.get_model().to(self.cfg.align.device).eval()
            self._tokenizer = bundle.get_tokenizer()
            self._aligner = bundle.get_aligner()

        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(self.cfg.align.device)
        with torch.inference_mode():
            emission, _ = self._model(wav)
            tokens = self._tokenizer(text.split())
            spans = self._aligner(emission[0], tokens)
        scores = [float(s.score) for word in spans for s in word]
        return float(np.mean(scores)) if scores else None
