# Evaluation report

Generated 2026-09-29T16:54:24+00:00 by `run_evaluation.py`.
Model: **bc_resnet:BCResNet6 from /home/arvir.jane.redondo/sandbox/AI231/ME2 v2/data/deliverables/model/runs/bcresnet6_hn/best.pt**.

## Data and independence

| Set | Audio | Real / synthetic | Used for |
|---|---|---|---|
| Isolated validation | 4117 fixed windows (`isolated_set/isolated_validation_manifest.csv`) | positives: real_device_rpi 37, real_other_device 1, synthetic 194 | metrics + choosing the threshold |
| Streaming Set A (phase-1 holdout) | 120 streams, 0.86 h scored exposure | synthetic | metrics only |
| Streaming Set B (composed for evaluation) | 60 streams, 0.93 h scored exposure | synthetic (new Piper TTS + procedural audio) | metrics only |

- Streaming set verified: 180 files match their frozen sha256; no stream id, file or source group appears in train, validation, test or the isolated set (`streaming_eval.verify_set`).
- Set B text is eval-only (no sentence, confusable or carrier phrase equals a training transcription), and wakewords are new stochastic syntheses off the training speed/pitch grid; see `streaming_set/build_log.json` for the near-duplicate check against existing positives.
- **Streaming evaluation contains no real device audio.** Real RPI performance is measured only by isolated validation (subset `real_device_rpi`).

## Operating point

Threshold **0.4530** — chosen on isolated validation: lowest threshold with FPR <= 0.01 in every negative category of deployment-like audio (excluding real_external). Streaming data never influenced it. Detection: fire when 2 of the last 3 window scores reach it, 1-window smoothing, 1.0 s refractory, hit zone [wake start − 0.0 s, wake end + 1.5 s]; sliding 1.5 s windows every 100 ms via `StreamingPreprocessor`.

Per-category thresholds (deployment-like validation negatives; the operating threshold is the largest): negative_confusable 0.4530, negative_real_background 0.0935, negative_general_speech 0.0795, negative_media 0.0523, negative_silence_noise 0.0483, negative_partial_wakeword 0.0444, other (pooled small categories) 0.0414.

## Isolated validation

ROC AUC **0.9917**, average precision **0.9726**. At the operating threshold: detection rate **94.8%** [91.1, 97.3]% (220/232), false-positive rate **0.08%** (3/3885).

**Positives by source:**

| Subset | n | Detected | Rate (95% CI) |
|---|---:|---:|---|
| real_device_rpi | 37 | 37 | 100.0% [90.5, 100.0]% |
| real_other_device | 1 | 1 | 100.0% [2.5, 100.0]% |
| synthetic | 194 | 182 | 93.8% [89.4, 96.8]% |

Owner-verified low-signal RPI positives in this split: Manual-Fil-RPI19 score 0.951 (detected).

**Negatives by category:**

| Category | n | False positives | FPR (95% CI) | Score p99 | Score max |
|---|---:|---:|---|---:|---:|
| negative_confusable | 1278 | 2 | 0.16% [0.02, 0.56]% | 0.085 | 0.931 |
| negative_general_speech | 1075 | 1 | 0.09% [0.00, 0.52]% | 0.063 | 0.778 |
| negative_media | 266 | 0 | 0.00% [0.00, 1.38]% | 0.088 | 0.146 |
| negative_non_speech_from_positive_recording | 7 | 0 | 0.00% [0.00, 40.96]% | 0.041 | 0.041 |
| negative_partial_wakeword | 227 | 0 | 0.00% [0.00, 1.61]% | 0.108 | 0.135 |
| negative_real_background | 439 | 0 | 0.00% [0.00, 0.84]% | 0.088 | 0.155 |
| negative_silence_noise | 593 | 0 | 0.00% [0.00, 0.62]% | 0.107 | 0.229 |

**Negatives by source:**

| Subset | n | False positives | FPR (95% CI) |
|---|---:|---:|---|
| real_device_rpi | 479 | 1 | 0.21% [0.01, 1.16]% |
| real_external | 2532 | 1 | 0.04% [0.00, 0.22]% |
| real_other_device | 2 | 0 | 0.00% [0.00, 84.19]% |
| synthetic | 872 | 1 | 0.11% [0.00, 0.64]% |

`real_external` = public-corpus negatives (MSWC, FLEURS, MUSAN, Hey Snips): reported, never used to choose the threshold. Per category and source: `negatives_by_category_and_subset` in `results.json`.

## Streaming (always listening)

Overall: **17.34 false accepts/hour** [11.78, 24.62] (31 in 1.79 h of wakeword-free exposure); detection rate **23.4%** [17.9, 29.8]% (49/209, 160 misses); median latency **0.07 s** (p90 0.17 s) after the end of the wakeword; 0 duplicate triggers.

With the plain rule (fire on any single window at the same threshold): 20.70 false accepts/hour, detection 26.3%.

- **FA/hour** = false accepts ÷ hours during which a false accept was possible (stream time after the first full window, minus every wakeword hit zone). Intervals are exact Poisson 95% CIs.
- **Latency** = trigger time (end of the triggering window) − wakeword end; negative means it fired before the word ended.

| Set / condition | Hours | FA | FA/h (95% CI) | Wakewords | Detected | Detection rate (95% CI) | Median latency |
|---|---:|---:|---|---:|---:|---|---:|
| set: A_phase1_holdout | 0.86 | 0 | 0.00 [0.0, 4.3] | 128 | 33 | 25.8% [18.5, 34.3]% | 0.05 s |
| set: B_composed | 0.93 | 31 | 33.51 [22.8, 47.6] | 81 | 16 | 19.8% [11.7, 30.1]% | 0.10 s |
| condition: continuous_speech | 0.15 | 3 | 19.57 [4.0, 57.2] | 15 | 0 | 0.0% [0.0, 21.8]% | – |
| condition: environmental_noise | 0.15 | 3 | 19.82 [4.1, 57.9] | 18 | 6 | 33.3% [13.3, 59.0]% | 0.08 s |
| condition: loud_continuous_speech | 0.86 | 0 | 0.00 [0.0, 4.3] | 128 | 33 | 25.8% [18.5, 34.3]% | 0.05 s |
| condition: media | 0.15 | 4 | 26.02 [7.1, 66.6] | 14 | 3 | 21.4% [4.7, 50.8]% | 0.10 s |
| condition: mixed | 0.16 | 5 | 32.13 [10.4, 75.0] | 12 | 2 | 16.7% [2.1, 48.4]% | 0.07 s |
| condition: noisy_speech | 0.16 | 5 | 31.99 [10.4, 74.7] | 10 | 0 | 0.0% [0.0, 30.8]% | – |
| condition: quiet | 0.15 | 11 | 71.00 [35.4, 127.0] | 12 | 5 | 41.7% [15.2, 72.3]% | 0.11 s |

### False accepts by category

Category = the non-wakeword event overlapping the triggering 1.5 s window the most; otherwise the stream's background condition. Full table: `fa_by_category.csv`; every false accept: `streaming_false_accepts.csv`.

| Category | False accepts | Share | Events of this type | Events that caused an FA (95% CI) | FA/h in that background |
|---|---:|---:|---:|---|---|
| confusable | 12 | 38.7% | 176 | 6.8% [3.6, 11.6]% | – |
| partial_tail | 6 | 19.4% | 67 | 9.0% [3.4, 18.5]% | – |
| partial_head | 5 | 16.1% | 60 | 8.3% [2.8, 18.4]% | – |
| background:noisy_speech | 3 | 9.7% | – | – | 19.20 [4.0, 56.1] |
| background:continuous_speech | 2 | 6.5% | – | – | 13.05 [1.6, 47.1] |
| background:media | 2 | 6.5% | – | – | 13.01 [1.6, 47.0] |
| background:mixed | 1 | 3.2% | – | – | 6.43 [0.2, 35.8] |
| speech_snippet | 0 | 0.0% | 43 | 0.0% [0.0, 8.2]% | – |

### Detection by wakeword type, level and voice

| Breakdown | Group | n | Detection rate (95% CI) |
|---|---|---:|---|
| type | wakeword | 186 | 24.2% [18.2, 31.0]% |
| type | wakeword_embedded | 23 | 17.4% [5.0, 38.8]% |
| level (peak over floor) | high (>=18 dB) | 42 | 16.7% [7.0, 31.4]% |
| level (peak over floor) | low (<10 dB) | 10 | 30.0% [6.7, 65.2]% |
| level (peak over floor) | mid (10-18 dB) | 29 | 20.7% [8.0, 39.7]% |
| level (peak over floor) | unmeasured (Set A) | 128 | 25.8% [18.5, 34.3]% |
| voice | en_GB-alan-medium | 26 | 7.7% [0.9, 25.1]% |
| voice | en_GB-alba-medium | 31 | 0.0% [0.0, 11.2]% |
| voice | en_US-amy-medium | 24 | 33.3% [15.6, 55.3]% |
| voice | en_US-lessac-medium | 34 | 44.1% [27.2, 62.1]% |
| voice | en_US-ryan-medium | 28 | 53.6% [33.9, 72.5]% |
| voice | fil-PH-AngeloNeural | 10 | 0.0% [0.0, 30.8]% |
| voice | fil-PH-BlessicaNeural | 15 | 40.0% [16.3, 67.7]% |
| voice | id_ID-news_tts-medium | 20 | 10.0% [1.2, 31.7]% |
| voice | vi_VN-vivos-x_low | 21 | 4.8% [0.1, 23.8]% |

### Wakeword positions

Relative start positions span 0.01–0.98 of the stream (quartiles 0.33, 0.48, 0.65). By set: A_phase1_holdout 0.21–0.77; B_composed 0.01–0.98. Set B places events uniformly; Set A (phase-1) restricted wakewords to the middle 20–80 %. With a 100 ms hop the wakeword also lands at every offset inside the sliding windows.

### Threshold sweep

`streaming_sweep.csv` (FA/h vs miss rate per set) and the `sweep` list in `results.json` (isolated detection rate vs FPR) give the full trade-off, independent of the chosen operating point.

## Caveats

- **Set A ground truth includes TTS silence padding** (phase-1 insertion metadata, used verbatim; several spans are 1.87 s long). Its wakeword 'end' is late, so Set A latencies are biased negative and not comparable with Set B, whose spans are energy-trimmed. Set A detection and false-accept counts are unaffected: the hit zone still starts at the wakeword start.
- **Levels differ between sets.** Set A is peak-normalised (loud, like phase-1 synthetic data); Set B is at the measured device level (floor −58 to −51 dB, events 5–26 dB above the floor). The louder `environmental_noise`, `noisy_speech` and `mixed` floors (−50 to −38 dB) are an assumption, not a measurement.
- **Small isolated positive counts** (validation: real_device_rpi 37, real_other_device 1, synthetic 194) give wide intervals; they are reported with every rate.
- **Latency** is measured from the end of the triggering 1.5 s window, i.e. the earliest moment a streaming detector could fire, excluding model compute time.

