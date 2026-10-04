# Ablations: what made the difference

One page for every comparison we ran, with each number copied from the full report it links to. Nothing here was
re-run. For each experiment, only the named change differs from the row before; everything else (recipe, data
splits, cutoff rules) stays the same.

- **Wakeword** (BC-ResNet, "Hey Delta"): [W1 voice diversity](#w1-voice-diversity-5-tts-voices--405-tts-voices),
  [W2 noise over the whole window](#w2-noise-over-the-whole-window),
  [W3 hard-negative mining](#w3-hard-negative-mining), [W4 model size](#w4-model-size-bc-resnet-1--3--6)
- **Command classifier** (31 commands + `unknown`): [C1 BC-ResNet-6 vs DS-CNN](#c1-bc-resnet-6-vs-ds-cnn-same-size-same-data),
  [C2 class Hugging Face data only vs with our data](#c2-class-hugging-face-data-only-vs-with-our-data),
  [C3 the same two models live on the Pi](#c3-the-same-two-models-live-on-the-raspberry-pi-class-benchmark)

**In short:** the data mattered more than the architecture. For the wakeword, more voices (W1) and removing a hidden
shortcut (W2) helped most; hard-negative mining did not (W3). For commands, BC-ResNet-6 clearly beat a DS-CNN of the
same size (C1), and adding our data to the class dataset helped on every group of speakers (C2, C3).

Terms: **RPI** = real "Hey Delta" recordings made on the Raspberry Pi; **synthetic** = TTS voices that were never in
training; **FA/h** = false accepts per hour on streaming audio; **Qualcomm** = "Hey/Hi + word" near-misses from the
Qualcomm Keyword Speech Dataset (evaluation only).

## Wakeword

Source for W1–W4: [`data/deliverables/model/README.md`](../data/deliverables/model/README.md) ("How we got here"),
with each run's full evaluation in `data/deliverables/model/runs/<run>/eval/metrics_report.md`. All W numbers are on
**validation** sets. The chosen model (BC-ResNet-6, round 1c) was then evaluated once on the held-out test split:
real Pi positives **33/33**, unseen synthetic voices **171/199 (86%)**, worst negative category **0.16%** false
positives ([`runs/bcresnet6/test/test_report.md`](../data/deliverables/model/runs/bcresnet6/test/test_report.md)).

### W1. Voice diversity: 5 TTS voices → 405 TTS voices

| Round | Training data | BC-ResNet-3 | BC-ResNet-6 | Runs |
|---|---|---|---|---|
| 1 (validation v5) | synthetic positives from **5 TTS voices** (96 windows) | RPI 36/37, synthetic 15/27, 47 FA/h | RPI 37/37, synthetic 17/27, 27 FA/h | `runs/round1_v5/` |
| 1b (validation v6) | **+ 405 new Piper TTS speakers**, each checked with Whisper (`scripts/generate_tts_voices_v2.py`) | RPI 37/37, synthetic 181/194 | RPI 37/37, synthetic 179/194 | `runs/round1b_v6/` |

With 5 voices the models memorised them. On new voices they detected only 55–63%, and this got worse as training
went on. 405 voices fixed it (about 92–93% of unseen synthetic positives). Real Pi recordings were fine in both
rounds. The validation set grew with the new voices (v5: 27 synthetic positives from one unseen voice; v6: 194 from
many), so the two rows are on different validation sets.

### W2. Noise over the whole window

Short clips are padded to 1.5 s, and augmentation (background noise) ran *before* framing. So positives had quiet
edges, and the model learned "speech with silence around it" as a cue. The diagnostic: with background over the
whole window, synthetic positives dropped from **89% to 37%** detected, while real Pi recordings (little padding)
stayed at 100%.

| Round | Change | BC-ResNet-3 | BC-ResNet-6 |
|---|---|---|---|
| 1b (v6) | augmentation before framing (quiet edges) | RPI 37/37, synthetic 181/194 | RPI 37/37, synthetic 179/194 |
| **1c (v6)** | **augmentation on the whole framed window** | RPI 37/37, synthetic 176/194, 14.5 FA/h | **RPI 37/37, synthetic 176/194, 6.2 FA/h, Qualcomm 0.07%** |

After the fix, Qualcomm "Hey + word" false accepts fell from **3.7% to 0.07–0.16%**. `source_confound.py` now checks
for this every evaluation (`runs/<run>/eval/diagnostics/source_confound_report.md`). Round 1c BC-ResNet-6 is the
chosen model.

### W3. Hard-negative mining

| Round | Change | BC-ResNet-6 | Run |
|---|---|---|---|
| 1c (v6) | none (chosen model) | RPI 37/37, synthetic 176/194, **6.2 FA/h**, Qualcomm **0.07%** | `runs/bcresnet6/` |
| 2 (v6) | + hard-negative mining (957 windows, 10% of draws) | RPI 37/37, synthetic 182/194, 17.3 FA/h, Qualcomm 0.94% | `runs/bcresnet6_hn/` |

It did not help, so it was not used. The mined windows were mostly synthetic (318 confusables, 305 noise clips, 20
real). The model got better on validation's synthetic near-misses, so its threshold fell to 0.45, which let more
mid-score sounds through on real-like audio. At a matched 10 FA/h it also detected fewer streaming wakewords (19% vs
23%).

### W4. Model size: BC-ResNet-1 / 3 / 6

Round 1c, same data and recipe. Parameters: 8.9k / 53k / 186k (`data/deliverables/model/bc_resnet.py`).

| Model | RPI | Synthetic | FA/h | Run |
|---|---|---|---|---|
| BC-ResNet-1 (8.9k) | 35/37 | 139/194 | – | `runs/bcresnet1/` |
| BC-ResNet-3 (53k) | 37/37 | 176/194 | 14.5 | `runs/bcresnet3/` |
| **BC-ResNet-6 (186k)** | **37/37** | **176/194** | **6.2** | `runs/bcresnet6/` |

BC-ResNet-1 is too small. BC-ResNet-6 has the same detection as BC-ResNet-3 with fewer than half the false accepts,
and still costs only about 4.8 ms per window on one server core (2.0 MB ONNX).

**Selection caveat (all W):** the streaming sets and the Qualcomm corpus informed the choice between trained models;
only the test split is fully independent.

## Command classifier

### C1. BC-ResNet-6 vs DS-CNN (same size, same data)

Both trained on the same schema-B data with the same recipe; cutoffs chosen on validation; numbers from the
**held-out test split** ([`results_schema_b/test_summary.md`](../data/deliverables/command_classifier/evaluation/results_schema_b/test_summary.md)).
DS-CNN = the depthwise-separable CNN of "Hello Edge" (Zhang et al. 2017), sized to match.

| | BC-ResNet-6 | DS-CNN |
|---|---|---|
| parameters | 191,672 | 194,888 |
| best epoch | 51 | 35 |
| cutoffs (cautious / balanced) | 0.990 / 0.945 | 0.920 / 0.720 |

**Best guess (argmax), correct / wrong command** (macro average over commands):

| Test group | Clips | BC-ResNet-6 | DS-CNN |
|---|---:|---|---|
| owner | 59 | **98.4% / 0.0%** | 72.1% / 8.7% |
| speaker2 (second speaker in the owner's recordings) | 48 | **100.0% / 0.0%** | 88.1% / 2.4% |
| public recordings (web) | 1,339 | **87.0% / 4.3%** | 61.9% / 7.0% |
| unseen synthetic voices | 834 | **98.5% / 0.0%** | 93.8% / 1.9% |
| classmates | 1,793 | **98.0% / 1.0%** | 92.0% / 4.5% |

**Balanced cutoff, correct / wrong command:**

| Test group | BC-ResNet-6 | DS-CNN |
|---|---|---|
| owner | **90.5% / 0.0%** | 41.7% / 1.6% |
| speaker2 | **86.5% / 0.0%** | 75.0% / 0.0% |
| web | **63.3% / 0.0%** | 45.8% / 2.1% |
| synthetic | **89.7% / 0.0%** | 84.5% / 0.4% |
| classmates | **83.5% / 0.0%** | 79.7% / 1.3% |

**Non-commands that trigger an action** (mean over groups): argmax 11.6% vs 12.4%; cautious 0.1% vs 0.4%; balanced
**0.2% vs 2.6%**.

BC-ResNet-6 is better on every group and every rule, with fewer wrong commands. Results on speakers never seen in
training: [`results_schema_b/unseen_speakers.md`](../data/deliverables/command_classifier/evaluation/results_schema_b/unseen_speakers.md).

### C2. Class Hugging Face data only vs with our data

Both BC-ResNet-6, same recipe and validation (our filtered validation clips, never HF). Test = the class Hugging Face
dataset's test split (4,418 clips; no test speaker in either model's training)
([`results_hf_compare.md`](../data/deliverables/command_classifier/evaluation/results_hf_compare.md)).

Correct / wrong command:

| Rule | Group | HF data only | HF + our data |
|---|---|---|---|
| argmax | classmates (3,557 clips) | 90.3% / 7.7% | **95.7% / 1.2%** |
| argmax | web (814 clips) | 52.9% / 16.7% | **63.9% / 9.4%** |
| balanced | classmates | 80.1% / 1.8% | **95.6% / 1.1%** |
| balanced | web | 37.7% / 2.0% | **63.6%** / 8.4% |
| cautious | classmates | 63.7% / 0.2% | **92.9%** / 0.5% |
| cautious | web | 25.4% / 0.3% | **57.8%** / 3.7% |

Non-commands that trigger an action: argmax 27.7% vs **19.1%**; balanced **2.1%** vs 17.0%; cautious **0.0%** vs
8.5%.

Our data makes the model right far more often. The trade-off: at the same rule, HF + our data acts on more
non-commands, because its cutoffs were chosen on our validation set, which has few HF-like non-commands. Per-command
accuracy is in the same report (largest gains: LIGHT_ON +33 points, COLOR_RED +23, CALL +21).

### C3. The same two models live on the Raspberry Pi (class benchmark)

The class benchmark ([airimonda/vcm-benchmark](https://github.com/airimonda/vcm-benchmark)) played the class holdout
set from a laptop speaker about 1 m from the Pi; on Oct 3 both models answered the **same captures**
([`data/deliverables/evaluation/benchmark_comparison.md`](../data/deliverables/evaluation/benchmark_comparison.md)).

| | hf_plus balanced | hf_plus cautious | hf_only balanced | hf_only cautious |
|---|---|---|---|---|
| intent accuracy | **84.7%** | 75.2% | 50.5% | 35.6% |
| false reject (command ignored) | **14.0%** | 25.3% | 52.7% | 69.9% |
| false accept (out of scope fired) | 6.2% (1/16) | 0.0% (0/16) | 6.2% (1/16) | 0.0% (0/16) |
| false wake (no wake word, fired) | 0/16 | 0/16 | 0/16 | 0/16 |

Live on the Pi, HF + our data is far more accurate with either rule. Only 16 out-of-scope trials were played, so the
false-accept rates are rough (95% intervals in the report).
