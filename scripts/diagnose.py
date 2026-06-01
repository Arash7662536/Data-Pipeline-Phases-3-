#!/usr/bin/env python
"""Diagnose why a single call produces few/bad chunks.

Usage:
    python scripts/diagnose.py --json /path/report.json --wav /path/xxx.wav
or point it at a folder containing one report.json + one .wav:
    python scripts/diagnose.py --dir /workspace/MEGA/.../test
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_dataset.config import load_config
from asr_dataset import io, chunking, filters, vad


def db(x: np.ndarray) -> float:
    if x.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--wav")
    ap.add_argument("--dir")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    if args.dir:
        d = Path(args.dir)
        jp = next(iter(d.glob("*.json")))
        wp = next(iter(d.glob("*.wav")))
    else:
        jp, wp = Path(args.json), Path(args.wav)

    cfg = load_config(args.config)
    print(f"json: {jp}")
    print(f"wav : {wp}\n")

    call = io.load_call_json(jp)
    segments = io.extract_segments(call)
    print(f"=== TRANSCRIPT: {len(segments)} segments ===")
    spk_dur = {}
    for s in segments:
        spk_dur[s.speaker] = spk_dur.get(s.speaker, 0.0) + s.duration
    for sp, dur in spk_dur.items():
        n = sum(1 for s in segments if s.speaker == sp)
        print(f"  {sp:12s}: {n:3d} turns, {dur:6.1f}s total speech")
    durs = [s.duration for s in segments]
    print(f"  per-turn duration: min={min(durs):.2f} median={np.median(durs):.2f} "
          f"max={max(durs):.2f}  (chunk.min_s={cfg.chunk.min_s})")
    n_short = sum(1 for s in segments if s.duration < cfg.chunk.min_s)
    print(f"  turns shorter than min_s: {n_short}/{len(segments)}\n")

    channels, sr = io.load_stereo(wp, target_sr=cfg.audio.target_sr)
    n_ch, n_samp = channels.shape
    print(f"=== AUDIO ===")
    print(f"  channels={n_ch}  sr={sr}  duration={n_samp/sr:.1f}s")
    for c in range(n_ch):
        print(f"  channel {c}: overall RMS = {db(channels[c]):6.1f} dBFS")
    if n_ch == 2:
        # how correlated are the two channels? high corr => effectively mono / heavy bleed
        a, b = channels[0], channels[1]
        m = min(len(a), len(b))
        corr = float(np.corrcoef(a[:m], b[:m])[0, 1]) if m > 1 else float("nan")
        print(f"  L/R correlation = {corr:.3f}  "
              f"(>0.9 = effectively mono / heavy crosstalk, channel split won't help)")
    print()

    spk_ch = io.map_channels_to_speakers(channels, sr, segments)
    print(f"=== CHANNEL <-> SPEAKER MAPPING ===")
    print(f"  {spk_ch}")
    # show energy table so we can see separation quality
    speakers = sorted({s.speaker for s in segments if s.speaker})
    print("  per-speaker energy share by channel (during that speaker's turns):")
    for sp in speakers:
        e = np.zeros(n_ch)
        for s in segments:
            if s.speaker != sp:
                continue
            a, b = int(s.start * sr), int(s.end * sr)
            for c in range(n_ch):
                e[c] += float(np.sum(channels[c, a:b] ** 2))
        tot = e.sum() + 1e-12
        share = ", ".join(f"ch{c}={e[c]/tot:5.1%}" for c in range(n_ch))
        print(f"    {sp:12s}: {share}  -> assigned ch{spk_ch.get(sp)}")
    print()

    chunks = chunking.build_chunks(str(jp.stem), segments, channels, sr, spk_ch, cfg)
    print(f"=== CHUNKS BUILT: {len(chunks)} (from {len(segments)} segments) ===")
    audio_quality = (call.get("raw_response", {}) or {}).get("audio_quality")
    kept = 0
    for k, ch in enumerate(chunks):
        if ch.speech_spans:
            clip = io.slice_spans(channels, ch.channel, sr, ch.speech_spans,
                                  edge_pad_s=cfg.audio.edge_pad_s,
                                  max_gap_s=cfg.chunk.max_internal_silence_s)
        else:
            clip = io.slice_channel(channels, ch.channel, sr, ch.start, ch.end,
                                    pad_s=cfg.audio.edge_pad_s)
        chunking.fill_clip_metrics(ch, clip, sr)
        reason = filters.check(ch, cfg, audio_quality=audio_quality)
        status = "KEEP" if reason is None else f"DROP[{reason}]"
        if reason is None:
            kept += 1
        preview = ch.text[:45].replace("\n", " ")
        print(f"  #{k:02d} {status:22s} {ch.speaker:8s} ch{ch.channel} "
              f"extent {ch.start:5.1f}-{ch.end:5.1f}s -> clip {ch.effective_duration:4.1f}s "
              f"sil={ch.silence_ratio:.0%} dB={db(clip):5.1f} "
              f"cps={ch.chars_per_sec or 0:4.1f} segs={len(ch.segment_indices)} | {preview}")
    print(f"\n  => {kept}/{len(chunks)} chunks would be kept")
    print(f"\n  config knobs: min_s={cfg.chunk.min_s} target_mean_s={cfg.chunk.target_mean_s} "
          f"merge_gap_max_s={cfg.chunk.merge_gap_max_s} use_vad={cfg.chunk.use_vad_boundaries} "
          f"vad_pad_s={cfg.chunk.vad_pad_s} max_internal_silence_s={cfg.chunk.max_internal_silence_s}")


if __name__ == "__main__":
    main()
