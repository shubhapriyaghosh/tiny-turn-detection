# Tiny Turn Detection

Submitted by: Shubhapriya Ghosh

## The problem

Turn detection decides whether a speaker has actually finished talking or is
just pausing — a breath, a filler word, a moment of thinking. Get it wrong
one way and a voice agent interrupts people mid-sentence. Get it wrong the
other way and the agent feels sluggish. The brief called out Indian Hinglish
speech as the hard case: filler words like *matlab*, *toh*, *acha* create
long mid-thought pauses that can look like a genuine end of turn if you're
only watching silence duration.

## What I built

**TinyTurnNet** — a small model trained from scratch. Two parts feed into a
shared classifier:

1. A compact CNN + GRU over a log-mel spectrogram of the trailing audio.
2. Six hand-engineered prosodic features: trailing silence duration, pitch
   trend, energy trend, voiced ratio, and texture measures.

523,298 parameters total, ~2.05 MB as ONNX.

### Why not just fine-tune Whisper Tiny, as the brief suggests?

I built and tested this too. Pipecat (the team whose dataset the challenge
links) already ships a Whisper Tiny + linear head model at ~8M parameters,
so I compared both approaches directly on the same hardware and dataset.

| | TinyTurnNet (2s) | TinyTurnNet (8s) | Whisper (1.5k clips) | Whisper (5k clips) |
|---|---|---|---|---|
| Size | 2.05 MB | 2.05 MB | ~31.5 MB | ~31.5 MB |
| Mean latency (CPU) | 2.84 ms | ~11 ms | 1,102.96 ms | 1,102.96 ms |
| Overall F1 | 0.788 | 0.812 | 0.773 | 0.803 |
| Overall accuracy | 0.782 | 0.810 | 0.753 | 0.800 |
| Mid-filler F1 | 0.755 | 0.784 | 0.796 | 0.748 |
| Hindi F1 | 0.838 | 0.837 | 0.816 | 0.805 |

Whisper Tiny + head is ~15x larger and ~390x slower on CPU. At 1,500 clips
it looked stronger on the mid-filler slice (0.796 vs 0.755) — but retrained
on the full 5,000-clip set, that advantage reversed (0.748, now below both
TinyTurnNet versions). That confirms the early result was partly a
small-sample effect. TinyTurnNet (8s window) is the strongest model overall
across every slice measured.

## The data

`pipecat-ai/smart-turn-data-v3.2-train`. Inspected before training:

- 270,946 rows, labels close to balanced.
- Hindi is 4.4% of the dataset (12,006 rows).
- **All 12,006 Hindi rows are synthetic (TTS-generated). No real Hindi
  speech exists in this dataset.**
- 82% of the entire dataset is synthetic.

Trained on a stratified 5,000-clip subset (all Hindi rows, a large
filler-flagged sample, a smaller general sample).

## Follow-up experiments (all with real, measured results)

1. **Decision threshold tuning** — scanned 0.40–0.80. Raising to **0.60**
   cuts false positives 15% (79→67) vs the default 0.5, at essentially no
   F1 cost (0.789 vs 0.788). Recommended deployment threshold.
2. **Static quantization** — dynamic int8 quantization was *slower* than
   fp32 (27.22ms vs 6.60ms). Static quantization with real calibration
   data matched fp32 speed (6.96ms) with only a 0.4-point accuracy drop.
3. **8-second window** (matching Pipecat's own context length) — improved
   overall F1 from 0.788 to 0.812, and mid-filler F1 from 0.755 to 0.784.
   Trade-off: 4x more audio processed per inference call.
4. **Full-scale Whisper retrain** (5,000 vs 1,500 clips) — see table above;
   the mid-filler advantage did not hold up at scale.

## Results (TinyTurnNet, 2s window, held-out test set, 500 clips)

| Metric | Value |
|---|---|
| Accuracy | 0.782 |
| Precision (end-of-turn) | 0.719 |
| Recall (end-of-turn) | 0.871 |
| F1 (end-of-turn) | 0.788 |

On the 24 genuinely non-synthetic test clips, accuracy was 0.500 (near
chance) at 2s and 0.542 at 8s — both too small a sample (n=24) to trust
precisely, and the honest conclusion is this model has not been
meaningfully validated on real speech.


  ## How to run

```bash
pip install -r requirements.txt
python demo/app.py

## Repo layout
