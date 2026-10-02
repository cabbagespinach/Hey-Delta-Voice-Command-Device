# Evaluation report

Generated 2026-09-29T15:18:28+00:00 by `run_evaluation.py`.
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

Threshold **0.5808** — chosen on isolated validation: lowest threshold with FPR <= 0.01 in every negative category of deployment-like audio (excluding real_external). Streaming data never influenced it. Detection: fire when 2 of the last 3 window scores reach it, 1-window smoothing, 1.0 s refractory, hit zone [wake start − 0.0 s, wake end + 1.5 s]; sliding 1.5 s windows every 100 ms via `StreamingPreprocessor`.

Per-category thresholds (deployment-like validation negatives; the operating threshold is the largest): negative_confusable 0.5808, negative_real_background 0.2580, negative_general_speech 0.1629, negative_partial_wakeword 0.1187, negative_media 0.0597, other (pooled small categories) 0.0375, negative_silence_noise 0.0338.

## Isolated validation

ROC AUC **0.9899**, average precision **0.9789**. At the operating threshold: detection rate **94.4%** [90.6, 97.0]% (219/232), false-positive rate **0.08%** (3/3885).

**Positives by source:**

| Subset | n | Detected | Rate (95% CI) |
|---|---:|---:|---|
| real_device_rpi | 37 | 37 | 100.0% [90.5, 100.0]% |
| real_other_device | 1 | 1 | 100.0% [2.5, 100.0]% |
| synthetic | 194 | 181 | 93.3% [88.8, 96.4]% |

Owner-verified low-signal RPI positives in this split: Manual-Fil-RPI19 score 0.981 (detected).

**Negatives by category:**

| Category | n | False positives | FPR (95% CI) | Score p99 | Score max |
|---|---:|---:|---|---:|---:|
| negative_real_background | 439 | 1 | 0.23% [0.01, 1.26]% | 0.241 | 0.864 |
| negative_confusable | 1278 | 2 | 0.16% [0.02, 0.56]% | 0.400 | 0.971 |
| negative_general_speech | 1075 | 0 | 0.00% [0.00, 0.34]% | 0.099 | 0.534 |
| negative_media | 266 | 0 | 0.00% [0.00, 1.38]% | 0.210 | 0.431 |
| negative_non_speech_from_positive_recording | 7 | 0 | 0.00% [0.00, 40.96]% | 0.037 | 0.038 |
| negative_partial_wakeword | 227 | 0 | 0.00% [0.00, 1.61]% | 0.224 | 0.507 |
| negative_silence_noise | 593 | 0 | 0.00% [0.00, 0.62]% | 0.139 | 0.237 |

**Negatives by source:**

| Subset | n | False positives | FPR (95% CI) |
|---|---:|---:|---|
| real_device_rpi | 479 | 2 | 0.42% [0.05, 1.50]% |
| real_external | 2532 | 0 | 0.00% [0.00, 0.15]% |
| real_other_device | 2 | 0 | 0.00% [0.00, 84.19]% |
| synthetic | 872 | 1 | 0.11% [0.00, 0.64]% |

`real_external` = public-corpus negatives (MSWC, FLEURS, MUSAN, Hey Snips): reported, never used to choose the threshold. Per category and source: `negatives_by_category_and_subset` in `results.json`.

## Streaming (always listening)

Overall: **13.99 false accepts/hour** [9.05, 20.65] (25 in 1.79 h of wakeword-free exposure); detection rate **18.2%** [13.2, 24.1]% (38/209, 171 misses); median latency **0.12 s** (p90 0.21 s) after the end of the wakeword; 0 duplicate triggers.

With the plain rule (fire on any single window at the same threshold): 17.90 false accepts/hour, detection 20.1%.

- **FA/hour** = false accepts ÷ hours during which a false accept was possible (stream time after the first full window, minus every wakeword hit zone). Intervals are exact Poisson 95% CIs.
- **Latency** = trigger time (end of the triggering window) − wakeword end; negative means it fired before the word ended.

| Set / condition | Hours | FA | FA/h (95% CI) | Wakewords | Detected | Detection rate (95% CI) | Median latency |
|---|---:|---:|---|---:|---:|---|---:|
| set: A_phase1_holdout | 0.86 | 0 | 0.00 [0.0, 4.3] | 128 | 24 | 18.8% [12.4, 26.6]% | 0.12 s |
| set: B_composed | 0.93 | 25 | 27.02 [17.5, 39.9] | 81 | 14 | 17.3% [9.8, 27.3]% | 0.16 s |
| condition: continuous_speech | 0.15 | 2 | 13.05 [1.6, 47.1] | 15 | 1 | 6.7% [0.2, 31.9]% | 0.01 s |
| condition: environmental_noise | 0.15 | 3 | 19.82 [4.1, 57.9] | 18 | 4 | 22.2% [6.4, 47.6]% | 0.16 s |
| condition: loud_continuous_speech | 0.86 | 0 | 0.00 [0.0, 4.3] | 128 | 24 | 18.8% [12.4, 26.6]% | 0.12 s |
| condition: media | 0.15 | 3 | 19.51 [4.0, 57.0] | 14 | 2 | 14.3% [1.8, 42.8]% | 0.22 s |
| condition: mixed | 0.16 | 6 | 38.55 [14.1, 83.9] | 12 | 1 | 8.3% [0.2, 38.5]% | 0.08 s |
| condition: noisy_speech | 0.16 | 2 | 12.80 [1.5, 46.2] | 10 | 0 | 0.0% [0.0, 30.8]% | – |
| condition: quiet | 0.15 | 9 | 58.09 [26.6, 110.3] | 12 | 6 | 50.0% [21.1, 78.9]% | 0.13 s |

### False accepts by category

Category = the non-wakeword event overlapping the triggering 1.5 s window the most; otherwise the stream's background condition. Full table: `fa_by_category.csv`; every false accept: `streaming_false_accepts.csv`.

| Category | False accepts | Share | Events of this type | Events that caused an FA (95% CI) | FA/h in that background |
|---|---:|---:|---:|---|---|
| partial_tail | 10 | 40.0% | 67 | 14.9% [7.4, 25.7]% | – |
| confusable | 8 | 32.0% | 176 | 4.5% [2.0, 8.8]% | – |
| partial_head | 3 | 12.0% | 60 | 5.0% [1.0, 13.9]% | – |
| background:media | 2 | 8.0% | – | – | 13.01 [1.6, 47.0] |
| background:noisy_speech | 1 | 4.0% | – | – | 6.40 [0.2, 35.7] |
| speech_snippet | 1 | 4.0% | 43 | 2.3% [0.1, 12.3]% | – |

### Detection by wakeword type, level and voice

| Breakdown | Group | n | Detection rate (95% CI) |
|---|---|---:|---|
| type | wakeword | 186 | 18.8% [13.5, 25.2]% |
| type | wakeword_embedded | 23 | 13.0% [2.8, 33.6]% |
| level (peak over floor) | high (>=18 dB) | 42 | 11.9% [4.0, 25.6]% |
| level (peak over floor) | low (<10 dB) | 10 | 30.0% [6.7, 65.2]% |
| level (peak over floor) | mid (10-18 dB) | 29 | 20.7% [8.0, 39.7]% |
| level (peak over floor) | unmeasured (Set A) | 128 | 18.8% [12.4, 26.6]% |
| voice | en_GB-alan-medium | 26 | 23.1% [9.0, 43.6]% |
| voice | en_GB-alba-medium | 31 | 0.0% [0.0, 11.2]% |
| voice | en_US-amy-medium | 24 | 29.2% [12.6, 51.1]% |
| voice | en_US-lessac-medium | 34 | 26.5% [12.9, 44.4]% |
| voice | en_US-ryan-medium | 28 | 35.7% [18.6, 55.9]% |
| voice | fil-PH-AngeloNeural | 10 | 20.0% [2.5, 55.6]% |
| voice | fil-PH-BlessicaNeural | 15 | 20.0% [4.3, 48.1]% |
| voice | id_ID-news_tts-medium | 20 | 5.0% [0.1, 24.9]% |
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

