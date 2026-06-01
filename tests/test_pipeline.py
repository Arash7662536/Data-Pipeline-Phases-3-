"""Lightweight tests that run without torch/whisperx/transformers.
    pytest tests/  (or python -m pytest)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_dataset.config import Config
from asr_dataset.types import Segment
from asr_dataset import chunking, io
from asr_dataset.agreement import normalize_fa, agreement_score, persian_char_ratio


def _segs():
    # alternating speakers, short turns that should merge within a channel
    return [
        Segment(0.0, 0.8, "سلام بفرمایید", "Customer"),
        Segment(0.8, 2.0, "سلام وقت بخیر", "Expert"),
        Segment(2.0, 7.8, "وقت بخیر از دیجی‌کالا تماس می‌گیرم", "Expert"),
        Segment(7.8, 9.8, "جانم عزیزم بفرمایید", "Customer"),
        Segment(9.8, 20.0, "درباره سفارش شما تماس گرفتم", "Expert"),
        Segment(20.0, 31.0, "بسیار خب پیگیری می‌کنم", "Expert"),
    ]


def test_merge_respects_max_length():
    cfg = Config()
    segs = _segs()
    # one-channel-each: Customer->0, Expert->1
    spk_ch = {"Customer": 0, "Expert": 1}
    channels = np.zeros((2, int(40 * 16000)), dtype=np.float32)
    chunks = chunking.build_chunks("c1", segs, channels, 16000, spk_ch, cfg)
    assert chunks, "should produce chunks"
    for c in chunks:
        assert c.duration <= cfg.chunk.max_s + 1e-6
        assert c.channel == (0 if c.speaker == "Customer" else 1)


def test_expert_turns_merge_into_multiturn_window():
    cfg = Config()
    chunks = chunking.build_chunks("c1", _segs(),
                                   np.zeros((2, int(40 * 16000)), np.float32),
                                   16000, {"Customer": 0, "Expert": 1}, cfg)
    expert = [c for c in chunks if c.speaker == "Expert"]
    # the two short expert turns at the start should merge, not stay isolated
    assert any(len(c.segment_indices) >= 2 for c in expert)


def test_normalize_fa_canonicalizes_yeh_kaf():
    # Arabic yeh/kaf should map to Persian forms -> identical after norm
    a = normalize_fa("كتاب علي")     # arabic kaf + arabic yeh
    b = normalize_fa("کتاب علی")     # persian kaf + persian yeh
    assert a == b


def test_agreement_perfect_and_partial():
    assert agreement_score("سلام دنیا", "سلام دنیا") == 1.0
    partial = agreement_score("سلام دنیا خوب", "سلام دنیا")
    assert 0.0 < partial < 1.0
    assert agreement_score("سلام", None) is None


def test_persian_char_ratio():
    assert persian_char_ratio("سلام دنیا") > 0.9
    assert persian_char_ratio("hello world") < 0.1


def test_trim_silence_edges_keeps_internal_audio():
    sr = 16000
    # 10s clip: silence | speech(2-3s) | silence | speech(6-7s) | silence
    x = np.zeros(sr * 10, dtype=np.float32)
    x[2 * sr:3 * sr] = 0.5
    x[6 * sr:7 * sr] = 0.5
    out = io.trim_silence_edges(x, sr, margin_s=0.2)
    dur = len(out) / sr
    # leading 0-1.8s and trailing 7.2-10s trimmed; internal silence (3-6s) KEPT
    assert 4.8 < dur < 5.6, dur


def test_trim_silence_edges_noop_when_all_silent():
    sr = 16000
    x = np.zeros(sr * 3, dtype=np.float32)
    out = io.trim_silence_edges(x, sr)
    assert len(out) == len(x)   # never nuke a clip we can't analyse


def test_no_min_duration_drop_keeps_short_turns():
    # a lone 1s turn must still produce a chunk (filtered later on real speech)
    cfg = Config()
    segs = [Segment(0.0, 1.0, "بله", "Expert")]
    chunks = chunking.build_chunks("c1", segs, np.zeros((2, sr_const()), np.float32),
                                   16000, {"Expert": 1}, cfg)
    assert len(chunks) == 1 and chunks[0].text == "بله"


def sr_const():
    return int(40 * 16000)


def test_channel_mapping_by_energy():
    sr = 16000
    chans = np.zeros((2, sr * 4), dtype=np.float32)
    # Customer loud on ch0 during 0-1s; Expert loud on ch1 during 1-2s
    chans[0, 0:sr] = 0.5
    chans[1, sr:2 * sr] = 0.5
    segs = [Segment(0, 1, "x", "Customer"), Segment(1, 2, "y", "Expert")]
    m = io.map_channels_to_speakers(chans, sr, segs)
    assert m["Customer"] == 0 and m["Expert"] == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
