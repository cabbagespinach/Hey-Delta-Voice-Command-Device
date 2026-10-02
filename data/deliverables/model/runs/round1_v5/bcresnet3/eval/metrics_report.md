# Evaluation report

Generated 2026-09-29T13:19:31+00:00 by `run_evaluation.py`.
Model: **bc_resnet:BCResNet3 from /home/arvir.jane.redondo/sandbox/AI231/ME2 v2/data/deliverables/model/runs/bcresnet3/best.pt**.

## Data and independence

| Set | Audio | Real / synthetic | Used for |
|---|---|---|---|
| Isolated validation | 3848 fixed windows (`isolated_set/isolated_validation_manifest.csv`) | positives: real_device_rpi 37, real_other_device 1, synthetic 27 | metrics + choosing the threshold |
| Streaming Set A (phase-1 holdout) | 120 streams, 0.86 h scored exposure | synthetic | metrics only |
| Streaming Set B (composed for evaluation) | 60 streams, 0.93 h scored exposure | synthetic (new Piper TTS + procedural audio) | metrics only |

- Streaming set verified: 180 files match their frozen sha256; no stream id, file or source group appears in train, validation, test or the isolated set (`streaming_eval.verify_set`).
- Set B text is eval-only (no sentence, confusable or carrier phrase equals a training transcription), and wakewords are new stochastic syntheses off the training speed/pitch grid; see `streaming_set/build_log.json` for the near-duplicate check against existing positives.
- **Streaming evaluation contains no real device audio.** Real RPI performance is measured only by isolated validation (subset `real_device_rpi`).

## Operating point

Threshold **0.3517** — chosen on isolated validation: lowest threshold with FPR <= 0.01 in every negative category of deployment-like audio (excluding real_external). Streaming data never influenced it. Detection: fire when 2 of the last 3 window scores reach it, 1-window smoothing, 1.0 s refractory, hit zone [wake start − 0.0 s, wake end + 1.5 s]; sliding 1.5 s windows every 100 ms via `StreamingPreprocessor`.

Per-category thresholds (deployment-like validation negatives; the operating threshold is the largest): negative_real_background 0.3517, negative_confusable 0.3504, negative_general_speech 0.1586, negative_partial_wakeword 0.1549, other (pooled small categories) 0.0743, negative_silence_noise 0.0519, negative_media 0.0506.

## Isolated validation

ROC AUC **0.9363**, average precision **0.8475**. At the operating threshold: detection rate **80.0%** [68.2, 88.9]% (52/65), false-positive rate **0.16%** (6/3783).

**Positives by source:**

| Subset | n | Detected | Rate (95% CI) |
|---|---:|---:|---|
| real_device_rpi | 37 | 36 | 97.3% [85.8, 99.9]% |
| real_other_device | 1 | 1 | 100.0% [2.5, 100.0]% |
| synthetic | 27 | 15 | 55.6% [35.3, 74.5]% |

Owner-verified low-signal RPI positives in this split: Manual-Fil-RPI19 score 0.107 (missed).

**Negatives by category:**

| Category | n | False positives | FPR (95% CI) | Score p99 | Score max |
|---|---:|---:|---|---:|---:|
| negative_real_background | 439 | 4 | 0.91% [0.25, 2.32]% | 0.350 | 0.699 |
| negative_silence_noise | 593 | 1 | 0.17% [0.00, 0.94]% | 0.220 | 0.736 |
| negative_confusable | 1176 | 1 | 0.09% [0.00, 0.47]% | 0.103 | 0.972 |
| negative_general_speech | 1075 | 0 | 0.00% [0.00, 0.34]% | 0.125 | 0.320 |
| negative_media | 266 | 0 | 0.00% [0.00, 1.38]% | 0.127 | 0.280 |
| negative_non_speech_from_positive_recording | 7 | 0 | 0.00% [0.00, 40.96]% | 0.073 | 0.074 |
| negative_partial_wakeword | 227 | 0 | 0.00% [0.00, 1.61]% | 0.091 | 0.177 |

**Negatives by source:**

| Subset | n | False positives | FPR (95% CI) |
|---|---:|---:|---|
| real_device_rpi | 479 | 5 | 1.04% [0.34, 2.42]% |
| real_external | 2532 | 1 | 0.04% [0.00, 0.22]% |
| real_other_device | 2 | 0 | 0.00% [0.00, 84.19]% |
| synthetic | 770 | 0 | 0.00% [0.00, 0.48]% |

`real_external` = public-corpus negatives (MSWC, FLEURS, MUSAN, Hey Snips): reported, never used to choose the threshold. Per category and source: `negatives_by_category_and_subset` in `results.json`.

## Streaming (always listening)

Overall: **46.99 false accepts/hour** [37.48, 58.18] (84 in 1.79 h of wakeword-free exposure); detection rate **22.5%** [17.0, 28.8]% (47/209, 162 misses); median latency **0.12 s** (p90 0.48 s) after the end of the wakeword; 0 duplicate triggers.

With the plain rule (fire on any single window at the same threshold): 65.46 false accepts/hour, detection 24.9%.

- **FA/hour** = false accepts ÷ hours during which a false accept was possible (stream time after the first full window, minus every wakeword hit zone). Intervals are exact Poisson 95% CIs.
- **Latency** = trigger time (end of the triggering window) − wakeword end; negative means it fired before the word ended.

| Set / condition | Hours | FA | FA/h (95% CI) | Wakewords | Detected | Detection rate (95% CI) | Median latency |
|---|---:|---:|---|---:|---:|---|---:|
| set: A_phase1_holdout | 0.86 | 15 | 17.40 [9.7, 28.7] | 128 | 38 | 29.7% [21.9, 38.4]% | 0.04 s |
| set: B_composed | 0.93 | 69 | 74.58 [58.0, 94.4] | 81 | 9 | 11.1% [5.2, 20.0]% | 0.48 s |
| condition: continuous_speech | 0.15 | 7 | 45.67 [18.4, 94.1] | 15 | 0 | 0.0% [0.0, 21.8]% | – |
| condition: environmental_noise | 0.15 | 6 | 39.65 [14.6, 86.3] | 18 | 4 | 22.2% [6.4, 47.6]% | 0.37 s |
| condition: loud_continuous_speech | 0.86 | 15 | 17.40 [9.7, 28.7] | 128 | 38 | 29.7% [21.9, 38.4]% | 0.04 s |
| condition: media | 0.15 | 27 | 175.63 [115.7, 255.5] | 14 | 2 | 14.3% [1.8, 42.8]% | 1.08 s |
| condition: mixed | 0.16 | 14 | 89.96 [49.2, 150.9] | 12 | 3 | 25.0% [5.5, 57.2]% | 0.48 s |
| condition: noisy_speech | 0.16 | 8 | 51.19 [22.1, 100.9] | 10 | 0 | 0.0% [0.0, 30.8]% | – |
| condition: quiet | 0.15 | 7 | 45.18 [18.2, 93.1] | 12 | 0 | 0.0% [0.0, 26.5]% | – |

### False accepts by category

Category = the non-wakeword event overlapping the triggering 1.5 s window the most; otherwise the stream's background condition. Full table: `fa_by_category.csv`; every false accept: `streaming_false_accepts.csv`.

| Category | False accepts | Share | Events of this type | Events that caused an FA (95% CI) | FA/h in that background |
|---|---:|---:|---:|---|---|
| background:media | 19 | 22.6% | – | – | 123.59 [74.4, 193.0] |
| confusable | 18 | 21.4% | 176 | 10.2% [6.2, 15.7]% | – |
| background:loud_continuous_speech | 15 | 17.9% | – | – | 17.40 [9.7, 28.7] |
| background:mixed | 11 | 13.1% | – | – | 70.68 [35.3, 126.5] |
| partial_tail | 8 | 9.5% | 67 | 11.9% [5.3, 22.2]% | – |
| background:noisy_speech | 5 | 6.0% | – | – | 31.99 [10.4, 74.7] |
| background:continuous_speech | 4 | 4.8% | – | – | 26.10 [7.1, 66.8] |
| speech_snippet | 4 | 4.8% | 43 | 9.3% [2.6, 22.1]% | – |
| partial_head | 0 | 0.0% | 60 | 0.0% [0.0, 6.0]% | – |

### Detection by wakeword type, level and voice

| Breakdown | Group | n | Detection rate (95% CI) |
|---|---|---:|---|
| type | wakeword | 186 | 24.2% [18.2, 31.0]% |
| type | wakeword_embedded | 23 | 8.7% [1.1, 28.0]% |
| level (peak over floor) | high (>=18 dB) | 42 | 7.1% [1.5, 19.5]% |
| level (peak over floor) | low (<10 dB) | 10 | 10.0% [0.3, 44.5]% |
| level (peak over floor) | mid (10-18 dB) | 29 | 17.2% [5.8, 35.8]% |
| level (peak over floor) | unmeasured (Set A) | 128 | 29.7% [21.9, 38.4]% |
| voice | en_GB-alan-medium | 26 | 3.8% [0.1, 19.6]% |
| voice | en_GB-alba-medium | 31 | 0.0% [0.0, 11.2]% |
| voice | en_US-amy-medium | 24 | 29.2% [12.6, 51.1]% |
| voice | en_US-lessac-medium | 34 | 17.6% [6.8, 34.5]% |
| voice | en_US-ryan-medium | 28 | 17.9% [6.1, 36.9]% |
| voice | fil-PH-AngeloNeural | 10 | 40.0% [12.2, 73.8]% |
| voice | fil-PH-BlessicaNeural | 15 | 60.0% [32.3, 83.7]% |
| voice | id_ID-news_tts-medium | 20 | 60.0% [36.1, 80.9]% |
| voice | vi_VN-vivos-x_low | 21 | 14.3% [3.0, 36.3]% |

### Wakeword positions

Relative start positions span 0.01–0.98 of the stream (quartiles 0.33, 0.48, 0.65). By set: A_phase1_holdout 0.21–0.77; B_composed 0.01–0.98. Set B places events uniformly; Set A (phase-1) restricted wakewords to the middle 20–80 %. With a 100 ms hop the wakeword also lands at every offset inside the sliding windows.

### Threshold sweep

`streaming_sweep.csv` (FA/h vs miss rate per set) and the `sweep` list in `results.json` (isolated detection rate vs FPR) give the full trade-off, independent of the chosen operating point.

## Caveats

- **Set A ground truth includes TTS silence padding** (phase-1 insertion metadata, used verbatim; several spans are 1.87 s long). Its wakeword 'end' is late, so Set A latencies are biased negative and not comparable with Set B, whose spans are energy-trimmed. Set A detection and false-accept counts are unaffected: the hit zone still starts at the wakeword start.
- **Levels differ between sets.** Set A is peak-normalised (loud, like phase-1 synthetic data); Set B is at the measured device level (floor −58 to −51 dB, events 5–26 dB above the floor). The louder `environmental_noise`, `noisy_speech` and `mixed` floors (−50 to −38 dB) are an assumption, not a measurement.
- **Small isolated positive counts** (validation: real_device_rpi 37, real_other_device 1, synthetic 27) give wide intervals; they are reported with every rate.
- **Latency** is measured from the end of the triggering 1.5 s window, i.e. the earliest moment a streaming detector could fire, excluding model compute time.

