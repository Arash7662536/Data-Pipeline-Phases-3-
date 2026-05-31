"""Second ASR labeler. Point `model_id` at YOUR fine-tuned Farsi checkpoint and
transcribe each chunk independently. Gemini + this model = two independent weak
labelers; their agreement is a strong, cheap confidence signal, and their
*disagreement* is your active-learning queue.

Lazy transformers import. Batches chunks for throughput.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import Config


class SecondASR:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._pipe = None

    @property
    def enabled(self) -> bool:
        return self.cfg.second_asr.enabled

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline)

        sc = self.cfg.second_asr
        dtype = torch.float16 if sc.device.startswith("cuda") else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            sc.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(sc.device)
        processor = AutoProcessor.from_pretrained(sc.model_id)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model, tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype, device=sc.device,
            chunk_length_s=30, batch_size=sc.batch_size,
            generate_kwargs={"language": sc.language, "task": "transcribe"},
        )

    def transcribe(self, audio: np.ndarray, sr: int) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            self._ensure()
            out = self._pipe({"array": audio.astype(np.float32), "sampling_rate": sr})
            return (out.get("text") or "").strip()
        except Exception:
            return None

    def transcribe_batch(self, clips: list[np.ndarray], sr: int) -> list[Optional[str]]:
        if not self.enabled:
            return [None] * len(clips)
        try:
            self._ensure()
            inputs = [{"array": c.astype(np.float32), "sampling_rate": sr} for c in clips]
            outs = self._pipe(inputs)
            return [(o.get("text") or "").strip() for o in outs]
        except Exception:
            return [None] * len(clips)
