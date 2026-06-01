"""Configuration. Defaults encode the choices we settled on for
Whisper large-v3 + ~300h + an existing Farsi checkpoint as second labeler."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class AudioConfig:
    target_sr: int = 16_000          # Whisper input; telephony is usually 8k -> upsample
    mono_per_channel: bool = True    # split stereo, never downmix to mono
    n_mels: int = 128                # large-v3 uses 128 mel bins (v2 used 80)
    edge_pad_s: float = 0.15         # padding each cut edge so onsets aren't clipped


@dataclass
class ChunkConfig:
    min_s: float = 1.0               # floor on the (edge-trimmed) clip duration
    max_s: float = 28.0              # ceiling: ~2s headroom under Whisper's 30s window
    target_mean_s: float = 12.0
    silence_cut_min_s: float = 0.30  # only cut at silences >= this
    merge_gap_max_s: float = 1.5     # merge same-speaker turns if gap below this
    # Coarse timestamps pad turns with silence; trim ONLY the leading/trailing
    # silence (never internal audio, never into neighbours) so words aren't lost.
    trim_silence_edges: bool = True
    trim_margin_s: float = 0.2       # keep this much silence around the speech
    # target length distribution (for the stats report, not a hard constraint)
    long_band: tuple[float, float] = (18.0, 28.0)   # keep ~30% here for long-form recall
    mid_band: tuple[float, float] = (8.0, 18.0)     # ~60%
    short_band: tuple[float, float] = (3.0, 8.0)    # ~10%


@dataclass
class OverlapConfig:
    keep: bool = True                # keep, because channel split makes target speaker dominant
    max_fraction_of_dataset: float = 0.12  # cap crosstalk chunks at ~10-15%
    # an overlap chunk is only kept on the channel where its speaker is dominant
    drop_if_overlap_fraction_above: float = 0.6  # too much crosstalk -> unusable


@dataclass
class ConfidenceConfig:
    use_forced_alignment: bool = True
    use_second_asr: bool = True
    # composite = (w_align*align + w_agree*agree) * penalty, then clamped
    w_align: float = 0.5
    w_agree: float = 0.5
    # tier thresholds
    accept_at: float = 0.85
    review_below: float = 0.60
    downweight_loss: float = 0.5     # loss weight for the middle tier


@dataclass
class FilterConfig:
    require_audio_quality_good: bool = False   # JSON has audio_quality; gate optionally
    max_silence_ratio: float = 0.75            # after edge-trim; internal pauses are fine
    cps_min: float = 3.0             # chars/sec lower bound (Persian; tune on your data)
    cps_max: float = 28.0            # upper bound catches hallucination / misalignment
    min_persian_char_ratio: float = 0.55       # language sanity check
    min_chars: int = 2


@dataclass
class SplitConfig:
    test_fraction: float = 0.05      # held out BY CALL to avoid leakage
    gold_oversample: int = 3         # oversample your clean 2k gold set 2-4x
    seed: int = 1337


@dataclass
class SecondASRConfig:
    enabled: bool = True
    # point this at YOUR fine-tuned Farsi checkpoint
    model_id: str = "openai/whisper-large-v3"
    device: str = "cuda"
    language: str = "fa"
    batch_size: int = 8


@dataclass
class AlignConfig:
    enabled: bool = True
    backend: str = "whisperx"        # "whisperx" | "torchaudio" | "none"
    device: str = "cuda"
    language: str = "fa"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    overlap: OverlapConfig = field(default_factory=OverlapConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    filt: FilterConfig = field(default_factory=FilterConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    second_asr: SecondASRConfig = field(default_factory=SecondASRConfig)
    align: AlignConfig = field(default_factory=AlignConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(dc, d: dict):
    """Shallow merge of a dict onto a dataclass instance (one level of nesting)."""
    for k, v in (d or {}).items():
        if not hasattr(dc, k):
            continue
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, v)
    return dc


def load_config(path: Optional[str | Path] = None) -> Config:
    cfg = Config()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _merge(cfg, data)
    return cfg
