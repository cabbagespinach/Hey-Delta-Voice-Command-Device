# "Hey Delta" baseline model: BC-ResNet (2026-09-29/30)

Owner-approved plan: BC-ResNet on the existing 40 × 147 log-mel front end; the per-category threshold policy and a
2-of-3 streaming rule; mild random microphone colouring; hard-negative mining; ONNX for a Raspberry Pi 5 (4 GB,
offline).

| File | What it is |
|---|---|
| `bc_resnet.py` | BC-ResNet (Kim et al., Interspeech 2021) with a 1-logit head. `BCResNet1/3/6` = widths τ 1/3/6: 8.9k / 53k / 186k parameters. τ = 1 with the paper's 12-class head has 9,232 parameters (paper: 9.2k). |
| `train.py`, `train_config.json` | Training recipe: 60 epochs, AdamW, cosine schedule, SpecAugment, label smoothing, best-epoch selection on the frozen validation set, optional hard-negative round (`--mine-from`) |
| `run_round1.sh`, `run_round2.sh` | Train and evaluate one or more widths on ONE GPU (`GPU=<idx>`). The owner's rule: at most 2 GPUs at once. |
| `export_onnx.py` | One self-contained ONNX file (front end + network + sigmoid), front-end and probability parity checks, int8 gate |
| `evaluate_test.py` | One-time held-out test evaluation of the chosen model (refuses to run twice) |
| `make_laptop_kit.py`, `laptop/` → `laptop_kit/`, `laptop_kit.zip` | Test kit for any laptop: all models live on the microphone, detection + efficiency reports |
| `make_pi_bundle.py`, `pi/` → `pi_kit/`, `pi_kit.zip` | Raspberry Pi test kit: the same live script plus `heydelta_pi.py check / bench` |
| `deploy/` | Deployment module for the chosen model: `HeyDeltaListener` (wakeword → command capture → your callback) and a recording demo |
| `ENHANCEMENTS.md` | Deferred improvements (post-project) |
| `runs/` | Every run's checkpoints, `train_log.json` and full evaluation (`eval/`, `eval/diagnostics/`) |

## Result: the chosen model

**BC-ResNet-6 (round 1c)**: `runs/bcresnet6/best.pt`, threshold **0.7335**, fires when 2 of the last 3 windows
(100 ms apart) reach it, then 1 s refractory. Owner's choice (2026-09-30), confirmed by the owner's live test on the
Pi: no firing on ordinary dialogue; "Hey Delta" fires in any tone.

**Held-out test split, evaluated once, after the choice** (`runs/bcresnet6/test/test_report.md`):

| | Validation | Test |
|---|---:|---:|
| Real RPI positives detected | 37/37 | **33/33** |
| Synthetic positives (unseen voices) | 176/194 (91%) | **171/199 (86%)**, 50 voices |
| Worst negative category (FPR) | 0.16% | **0.16%** (confusables, 2/1,281); every other category 0 |
| ROC AUC / average precision | 0.987 / 0.953 | 0.991 / 0.952 |

**Other checks (validation / evaluation-only sets):**
- Qualcomm "Hey/Hi + word" false accepts: 0.07% at device level, 0.05% as recorded.
- Confound (synthetic positives moved to device level, mean score change): −0.02.
- Streaming (synthetic): 6.2 false accepts/hour, 22% of wakewords detected.
- Cost: 4.8 ms and ~5% of one core per window on one server core (~10% expected on the Pi 5); 2.0 MB fp32 ONNX.
  int8 failed the export gate, so the Pi uses fp32.

## How we got here

| Round | Change | BC-ResNet-3 | BC-ResNet-6 |
|---|---|---|---|
| 1 (validation v5) | recipe as planned | RPI 36/37, synthetic 15/27 (one voice), 47 FA/h | RPI 37/37, synthetic 17/27, 27 FA/h |
| 1b (v6) | + 405 new TTS speakers (`generate_tts_voices_v2.py`) | RPI 37/37, synthetic 181/194 | RPI 37/37, synthetic 179/194 |
| **1c (v6)** | + augmentation on the whole framed window | RPI 37/37, synthetic 176/194, 14.5 FA/h | **RPI 37/37, synthetic 176/194, 6.2 FA/h, Qualcomm 0.07%** |
| 2 (v6) | + hard-negative mining (957 windows, 10% of draws) | – | RPI 37/37, synthetic 182/194, 17.3 FA/h, Qualcomm 0.94% |

BC-ResNet-1 (round 1c): RPI 35/37, synthetic 139/194. Too small.

**What mattered most was data, not architecture:**
1. **Voice diversity.** Train had 96 synthetic positive windows from 5 TTS voices; round-1 models memorised them
   (validation synthetic 55–63% on one unseen voice, falling during training). 405 Whisper-verified Piper speakers
   fixed it.
2. **Padding flanks.** Short clips are padded to 1.5 s, and augmentation ran before framing, so positives had quiet
   flanks and "speech with quiet around it" became a cue. With background over the whole window, synthetic positives
   dropped from 89% to 37% detected, while real RPI (little padding) stayed at 100%. After the fix, Qualcomm
   "Hey + word" false accepts fell from 3.7% to 0.16–0.07%. The `source_confound.py` flank check now guards against
   it.
3. **Hard-negative mining did not help.** The mined set was mostly synthetic (318 confusables, 305 noise clips, 20
   real). The model got better on validation's synthetic near-misses, so its threshold fell to 0.45, which let more
   mid-score sounds through on real-like audio. At a matched 10 FA/h it also detected fewer streaming wakewords
   (19% vs 23%).

**Selection caveat:** the streaming sets and the Qualcomm corpus informed the choice between trained models; only
the test split is fully independent.

## Known limitations (see `ENHANCEMENTS.md`)

- **Near-misses in the owner's calling tone** ("Hey + word" said with the melody used for "Hey Delta") can fire. Fix:
  record such near-misses as negatives.
- **A wakeword spoken over other people's speech** is mostly missed in the synthetic streams (22% detected overall;
  speech-background median score ~0.13). The owner did not hit this in live use. Fix: speech/music background mixing
  in training.
- **Only real voice is the owner's**, on one device and one room set; the RPI microphone response is not measured
  (random colouring is an interim measure).
- **Streaming evaluation is synthetic.** The owner's live Pi sessions are the real-world check.
