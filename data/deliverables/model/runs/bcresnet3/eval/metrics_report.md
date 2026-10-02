# Evaluation report

Generated 2026-09-29T16:16:12+00:00 by `run_evaluation.py`.
Model: **bc_resnet:BCResNet3 from /home/arvir.jane.redondo/sandbox/AI231/ME2 v2/data/deliverables/model/runs/bcresnet3/best.pt**.

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

Threshold **0.8153** — chosen on isolated validation: lowest threshold with FPR <= 0.01 in every negative category of deployment-like audio (excluding real_external). Streaming data never influenced it. Detection: fire when 2 of the last 3 window scores reach it, 1-window smoothing, 1.0 s refractory, hit zone [wake start − 0.0 s, wake end + 1.5 s]; sliding 1.5 s windows every 100 ms via `StreamingPreprocessor`.

Per-category thresholds (deployment-like validation negatives; the operating threshold is the largest): negative_confusable 0.8153, negative_real_background 0.1998, negative_silence_noise 0.1048, negative_partial_wakeword 0.0999, negative_media 0.0945, negative_general_speech 0.0877, other (pooled small categories) 0.0315.

## Isolated validation

ROC AUC **0.9912**, average precision **0.9536**. At the operating threshold: detection rate **92.2%** [88.0, 95.3]% (214/232), false-positive rate **0.05%** (2/3885).

**Positives by source:**

| Subset | n | Detected | Rate (95% CI) |
|---|---:|---:|---|
| real_device_rpi | 37 | 37 | 100.0% [90.5, 100.0]% |
| real_other_device | 1 | 1 | 100.0% [2.5, 100.0]% |
| synthetic | 194 | 176 | 90.7% [85.7, 94.4]% |

Owner-verified low-signal RPI positives in this split: Manual-Fil-RPI19 score 0.956 (detected).

**Negatives by category:**

| Category | n | False positives | FPR (95% CI) | Score p99 | Score max |
|---|---:|---:|---|---:|---:|
| negative_confusable | 1278 | 2 | 0.16% [0.02, 0.56]% | 0.424 | 0.950 |
| negative_general_speech | 1075 | 0 | 0.00% [0.00, 0.34]% | 0.099 | 0.257 |
| negative_media | 266 | 0 | 0.00% [0.00, 1.38]% | 0.192 | 0.396 |
| negative_non_speech_from_positive_recording | 7 | 0 | 0.00% [0.00, 40.96]% | 0.031 | 0.032 |
| negative_partial_wakeword | 227 | 0 | 0.00% [0.00, 1.61]% | 0.201 | 0.464 |
| negative_real_background | 439 | 0 | 0.00% [0.00, 0.84]% | 0.190 | 0.654 |
| negative_silence_noise | 593 | 0 | 0.00% [0.00, 0.62]% | 0.183 | 0.283 |

**Negatives by source:**

| Subset | n | False positives | FPR (95% CI) |
|---|---:|---:|---|
| real_device_rpi | 479 | 1 | 0.21% [0.01, 1.16]% |
| real_external | 2532 | 0 | 0.00% [0.00, 0.15]% |
| real_other_device | 2 | 0 | 0.00% [0.00, 84.19]% |
| synthetic | 872 | 1 | 0.11% [0.00, 0.64]% |

`real_external` = public-corpus negatives (MSWC, FLEURS, MUSAN, Hey Snips): reported, never used to choose the threshold. Per category and source: `negatives_by_category_and_subset` in `results.json`.

## Streaming (always listening)

Overall: **14.55 false accepts/hour** [9.50, 21.31] (26 in 1.79 h of wakeword-free exposure); detection rate **23.0%** [17.4, 29.3]% (48/209, 161 misses); median latency **0.10 s** (p90 0.28 s) after the end of the wakeword; 0 duplicate triggers.

With the plain rule (fire on any single window at the same threshold): 18.46 false accepts/hour, detection 23.0%.

- **FA/hour** = false accepts ÷ hours during which a false accept was possible (stream time after the first full window, minus every wakeword hit zone). Intervals are exact Poisson 95% CIs.
- **Latency** = trigger time (end of the triggering window) − wakeword end; negative means it fired before the word ended.

| Set / condition | Hours | FA | FA/h (95% CI) | Wakewords | Detected | Detection rate (95% CI) | Median latency |
|---|---:|---:|---|---:|---:|---|---:|
| set: A_phase1_holdout | 0.86 | 0 | 0.00 [0.0, 4.3] | 128 | 26 | 20.3% [13.7, 28.3]% | 0.06 s |
| set: B_composed | 0.93 | 26 | 28.10 [18.4, 41.2] | 81 | 22 | 27.2% [17.9, 38.2]% | 0.17 s |
| condition: continuous_speech | 0.15 | 2 | 13.05 [1.6, 47.1] | 15 | 0 | 0.0% [0.0, 21.8]% | – |
| condition: environmental_noise | 0.15 | 6 | 39.65 [14.6, 86.3] | 18 | 8 | 44.4% [21.5, 69.2]% | 0.20 s |
| condition: loud_continuous_speech | 0.86 | 0 | 0.00 [0.0, 4.3] | 128 | 26 | 20.3% [13.7, 28.3]% | 0.06 s |
| condition: media | 0.15 | 2 | 13.01 [1.6, 47.0] | 14 | 4 | 28.6% [8.4, 58.1]% | 0.20 s |
| condition: mixed | 0.16 | 4 | 25.70 [7.0, 65.8] | 12 | 2 | 16.7% [2.1, 48.4]% | 0.12 s |
| condition: noisy_speech | 0.16 | 4 | 25.59 [7.0, 65.5] | 10 | 1 | 10.0% [0.3, 44.5]% | 0.46 s |
| condition: quiet | 0.15 | 8 | 51.64 [22.3, 101.7] | 12 | 7 | 58.3% [27.7, 84.8]% | 0.09 s |

### False accepts by category

Category = the non-wakeword event overlapping the triggering 1.5 s window the most; otherwise the stream's background condition. Full table: `fa_by_category.csv`; every false accept: `streaming_false_accepts.csv`.

| Category | False accepts | Share | Events of this type | Events that caused an FA (95% CI) | FA/h in that background |
|---|---:|---:|---:|---|---|
| confusable | 15 | 57.7% | 176 | 8.5% [4.8, 13.7]% | – |
| partial_head | 5 | 19.2% | 60 | 8.3% [2.8, 18.4]% | – |
| background:noisy_speech | 3 | 11.5% | – | – | 19.20 [4.0, 56.1] |
| partial_tail | 2 | 7.7% | 67 | 3.0% [0.4, 10.4]% | – |
| background:mixed | 1 | 3.8% | – | – | 6.43 [0.2, 35.8] |
| speech_snippet | 0 | 0.0% | 43 | 0.0% [0.0, 8.2]% | – |

### Detection by wakeword type, level and voice

| Breakdown | Group | n | Detection rate (95% CI) |
|---|---|---:|---|
| type | wakeword | 186 | 22.6% [16.8, 29.3]% |
| type | wakeword_embedded | 23 | 26.1% [10.2, 48.4]% |
| level (peak over floor) | high (>=18 dB) | 42 | 26.2% [13.9, 42.0]% |
| level (peak over floor) | low (<10 dB) | 10 | 40.0% [12.2, 73.8]% |
| level (peak over floor) | mid (10-18 dB) | 29 | 24.1% [10.3, 43.5]% |
| level (peak over floor) | unmeasured (Set A) | 128 | 20.3% [13.7, 28.3]% |
| voice | en_GB-alan-medium | 26 | 19.2% [6.6, 39.4]% |
| voice | en_GB-alba-medium | 31 | 3.2% [0.1, 16.7]% |
| voice | en_US-amy-medium | 24 | 41.7% [22.1, 63.4]% |
| voice | en_US-lessac-medium | 34 | 38.2% [22.2, 56.4]% |
| voice | en_US-ryan-medium | 28 | 46.4% [27.5, 66.1]% |
| voice | fil-PH-AngeloNeural | 10 | 0.0% [0.0, 30.8]% |
| voice | fil-PH-BlessicaNeural | 15 | 40.0% [16.3, 67.7]% |
| voice | id_ID-news_tts-medium | 20 | 0.0% [0.0, 16.8]% |
| voice | vi_VN-vivos-x_low | 21 | 0.0% [0.0, 16.1]% |

### Wakeword positions

Relative start positions span 0.01–0.98 of the stream (quartiles 0.33, 0.48, 0.65). By set: A_phase1_holdout 0.21–0.77; B_composed 0.01–0.98. Set B places events uniformly; Set A (phase-1) restricted wakewords to the middle 20–80 %. With a 100 ms hop the wakeword also lands at every offset inside the sliding windows.

### Threshold sweep

`streaming_sweep.csv` (FA/h vs miss rate per set) and the `sweep` list in `results.json` (isolated detection rate vs FPR) give the full trade-off, independent of the chosen operating point.

## Caveats

- **Set A ground truth includes TTS silence padding** (phase-1 insertion metadata, used verbatim; several spans are 1.87 s long). Its wakeword 'end' is late, so Set A latencies are biased negative and not comparable with Set B, whose spans are energy-trimmed. Set A detection and false-accept counts are unaffected: the hit zone still starts at the wakeword start.
- **Levels differ between sets.** Set A is peak-normalised (loud, like phase-1 synthetic data); Set B is at the measured device level (floor −58 to −51 dB, events 5–26 dB above the floor). The louder `environmental_noise`, `noisy_speech` and `mixed` floors (−50 to −38 dB) are an assumption, not a measurement.
- **Small isolated positive counts** (validation: real_device_rpi 37, real_other_device 1, synthetic 194) give wide intervals; they are reported with every rate.
- **Latency** is measured from the end of the triggering 1.5 s window, i.e. the earliest moment a streaming detector could fire, excluding model compute time.

