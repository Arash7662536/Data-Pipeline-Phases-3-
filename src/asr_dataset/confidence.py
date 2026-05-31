"""Composite confidence and tiering.

confidence = (w_align * alignment + w_agree * agreement) * penalty

where penalty downweights heavy crosstalk and out-of-range speech rate. When a
signal is missing (e.g. no aligner), weights are renormalized over what's
available. The score is intentionally simple and *uncalibrated*; calibrate it
on a small hand-labeled gold set (see scripts/calibrate.py stub in README)
before trusting the absolute thresholds.
"""
from __future__ import annotations

from typing import Optional

from .config import Config
from .types import Chunk


def compute_confidence(chunk: Chunk, cfg: Config) -> float:
    cc = cfg.confidence
    parts: list[tuple[float, float]] = []  # (weight, value)
    if chunk.alignment_score is not None:
        parts.append((cc.w_align, _clamp(chunk.alignment_score)))
    if chunk.agreement_score is not None:
        parts.append((cc.w_agree, _clamp(chunk.agreement_score)))

    if not parts:
        base = 0.5  # no acoustic evidence at all -> neutral, will land in review tier
    else:
        wsum = sum(w for w, _ in parts)
        base = sum(w * v for w, v in parts) / wsum

    penalty = _penalty(chunk, cfg)
    return _clamp(base * penalty)


def _penalty(chunk: Chunk, cfg: Config) -> float:
    p = 1.0
    # crosstalk penalty scales with overlap fraction
    if chunk.overlap_fraction:
        p *= max(0.4, 1.0 - chunk.overlap_fraction)
    # speech-rate sanity (outliers => hallucination / misalignment)
    cps = chunk.chars_per_sec
    if cps is not None:
        if cps < cfg.filt.cps_min or cps > cfg.filt.cps_max:
            p *= 0.6
    return p


def assign_tier(chunk: Chunk, cfg: Config) -> None:
    c = chunk.confidence if chunk.confidence is not None else 0.0
    cc = cfg.confidence
    if c >= cc.accept_at:
        chunk.tier, chunk.loss_weight = "accept", 1.0
    elif c >= cc.review_below:
        chunk.tier, chunk.loss_weight = "downweight", cc.downweight_loss
    else:
        chunk.tier, chunk.loss_weight = "review", 0.0


def _clamp(x: float) -> float:
    return float(max(0.0, min(1.0, x)))
