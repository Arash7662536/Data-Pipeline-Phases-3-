"""Core data structures passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Segment:
    """One transcript turn as produced by Gemini (a single row in the JSON
    `transcript` array)."""
    start: float
    end: float
    text: str
    speaker: str                     # "Customer" | "Expert"
    overlap: bool = False
    sentiment: Optional[str] = None
    tone_description: Optional[str] = None
    src_confidence: Optional[float] = None   # Gemini's own field (unreliable, kept for reference)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @classmethod
    def from_json(cls, d: dict) -> "Segment":
        return cls(
            start=float(d["start"]),
            end=float(d["end"]),
            text=(d.get("text") or "").strip(),
            speaker=str(d.get("speaker", "")).strip(),
            overlap=bool(d.get("overlap", False)),
            sentiment=d.get("sentiment"),
            tone_description=d.get("tone_description"),
            src_confidence=d.get("confidence"),
        )


@dataclass
class Chunk:
    """A training candidate: a contiguous span on ONE channel, one speaker,
    formed by merging adjacent same-speaker segments."""
    call_id: str
    channel: int                     # 0 or 1 (the channel the speaker is dominant on)
    speaker: str
    start: float
    end: float
    text: str
    segment_indices: list[int] = field(default_factory=list)

    # populated by later stages
    has_overlap: bool = False
    overlap_fraction: float = 0.0
    silence_ratio: Optional[float] = None
    chars_per_sec: Optional[float] = None

    # scoring
    alignment_score: Optional[float] = None     # mean per-word forced-align posterior
    agreement_score: Optional[float] = None     # word agreement vs second ASR
    confidence: Optional[float] = None          # composite 0..1
    second_asr_text: Optional[str] = None

    # routing
    tier: Optional[str] = None                  # "accept" | "downweight" | "review"
    loss_weight: float = 1.0
    audio_path: Optional[str] = None            # path to the rendered chunk wav

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_record(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        return d
