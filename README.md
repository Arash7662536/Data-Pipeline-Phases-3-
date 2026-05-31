# farsi-asr-dataset

Turn dual-channel call-center recordings + Gemini transcripts into a
**Whisper large-v3 fine-tuning dataset** for Farsi, with per-chunk confidence
and quality tiering.

Built for the specific setup: ~300h of 2-channel (customer / expert) telephony
WAVs, JSON transcripts from Gemini, and an **existing Farsi checkpoint** you can
reuse as a second, independent labeler.

---

## Why it's built this way

**Split the stereo channels — never downmix to mono.** Telephony puts one
speaker per channel, so you get hardware-level speaker separation for free. The
unit of the dataset is *(one channel, one speaker, that speaker's text)*.
`io.map_channels_to_speakers` decides the channel↔speaker mapping per call by
energy, so it's robust to L/R being swapped between recordings.

**Merge turns into 8–28s windows, cut only on silence.** Gemini's turns are
short diarization spans (often <1s). Training Whisper on clipped one-liners
wrecks its long-form recall — a documented failure mode where models fine-tuned
on ~8s clips transcribe one-minute audio poorly. `chunking.py` greedily merges
adjacent same-speaker turns up to `max_s` (28s, leaving headroom under Whisper's
hard 30s window), targeting a ~15s mean, cutting at VAD silences ≥300ms.

**Keep overlap — but only the dominant channel, and capped.** On the customer's
channel during crosstalk, the customer *is* the foreground voice; the expert is
faint bleed. Pairing that channel with the customer's text teaches robustness to
crosstalk, not deletion. The rule enforced everywhere: *the label always
corresponds to the speaker dominant on that channel.* Overlap chunks are capped
at ~12% of the set (`overlap.max_fraction_of_dataset`).

**Confidence = acoustic alignment × cross-model agreement.** Gemini's own
`confidence` field is an LLM self-report and is not calibrated to the audio, so
it's ignored for scoring. Instead:
- *Forced alignment* (`alignment.py`) gives a per-word acoustic posterior — does
  the audio actually support this word?
- *Second ASR* (`second_asr.py`) — your Farsi checkpoint — transcribes each
  chunk independently; `agreement.py` scores word agreement (Persian-normalized,
  so yeh/kaf/ZWNJ differences don't count as errors).
- `confidence.py` blends them and penalizes crosstalk and out-of-range speech
  rate, then tiers each chunk.

**Tiers drive training:** `≥0.85` accept (full weight) · `0.60–0.85` downweight
(loss ×0.5) · `<0.60` to the review queue (not trained raw). The
*disagreement* set (your model confident but differs from Gemini) is your
active-learning goldmine — usually a Gemini hallucination to drop, or new
vocabulary worth labeling.

**Continuing from a good checkpoint changes the math.** 300h of silver labels
can *regress* an already-strong model, so: filter hard (150h clean > 300h
noisy), mix in your clean 2k gold set oversampled 2–4× (`split.gold_oversample`),
split test **by call** to avoid leakage, and fine-tune at a low LR (~1e-5).
Evaluate every checkpoint on *both* your old gold set (catches regression) and a
new human-verified test set from this data (measures domain gain).

---

## Install

```bash
pip install -e ".[align,asr,hf,audio]"   # or: pip install -r requirements.txt
```

Core stages (chunking, filtering, agreement, manifests) run on just
numpy/soundfile/pyyaml. `torch` + `whisperx`/`transformers` are only needed for
the alignment and second-ASR stages.

## Layout your data

Each call is a `.json` + `.wav` pair sharing a basename:

```
data/calls/01KPNM....json
data/calls/01KPNM....wav
```

## Run

```bash
python scripts/build_dataset.py \
  --corpus data/calls \
  --out build/dataset \
  --config config/default.yaml \
  --gold data/gold/train.jsonl        # your clean 2k set (optional)

python scripts/stats.py build/dataset/train.jsonl
```

**Point the second ASR at your checkpoint** in `config/default.yaml`:

```yaml
second_asr:
  model_id: "/path/to/your-farsi-whisper-large-v3"
```

Outputs in `build/dataset/`: `train.jsonl`, `test.jsonl`,
`review_queue.jsonl`, `audio/*.wav`, `build_stats.json`. Each manifest row has
`audio_path`, `text`, `speaker`, `confidence`, `tier`, `loss_weight`,
`overlap_fraction`, etc. Load straight into a HF pipeline via
`manifest.to_hf_dataset(...)`.

## Pipeline stages

| stage | module | output |
|---|---|---|
| load json + split stereo | `io.py` | channels, segments, speaker→channel map |
| merge turns → chunks | `chunking.py` | length-budgeted `Chunk`s |
| forced alignment | `alignment.py` | per-word acoustic posterior |
| second ASR | `second_asr.py` | independent transcript |
| agreement | `agreement.py` | word-agreement score (Persian-normalized) |
| composite confidence | `confidence.py` | 0–1 score + tier + loss weight |
| quality gates | `filters.py` | keep / reject + reason |
| overlap cap, split, gold | `manifest.py` | `train/test/review` JSONL |
| orchestration | `pipeline.py` | `process_call` / `process_corpus` |

## Calibrate before trusting thresholds

The composite confidence is intentionally simple and **uncalibrated**. Hand-label
a few hundred chunks (correct / not), then fit a small calibrator (logistic
regression on `[alignment_score, agreement_score, overlap_fraction, chars_per_sec]`)
and check expected calibration error. Adjust `accept_at` / `review_below` so the
tiers mean what you want.

## Test

```bash
pytest tests/            # runs without torch/whisperx
```

## NeMo later

The manifest is plain JSONL with `audio_path` + `text`, trivially convertible to
NeMo's manifest format (`audio_filepath`, `duration`, `text`). The chunk-length
budget is the main thing to revisit — CTC/Conformer models prefer ~4–20s.
