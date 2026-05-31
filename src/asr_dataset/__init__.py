"""farsi-asr-dataset: build Whisper-ready training chunks from dual-channel
call-center recordings + Gemini transcripts.

Pipeline overview (see README):
    JSON + stereo WAV
      -> split channels (one speaker per channel)
      -> merge same-speaker turns into length-budgeted chunks (cut on VAD silence)
      -> forced-align text to audio (per-word acoustic posterior)
      -> run second ASR (your Farsi checkpoint) as an independent labeler
      -> word-level agreement + composite confidence
      -> quality filters + overlap cap
      -> tiered manifest, split by call, gold oversampled
"""

from .config import Config, load_config
from .types import Segment, Chunk

__version__ = "0.1.0"

__all__ = ["Config", "load_config", "Segment", "Chunk", "__version__"]
