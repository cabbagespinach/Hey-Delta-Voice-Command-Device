# Evaluation report

Generated 2026-09-29T13:23:36+00:00 by `run_evaluation.py`.
Model: **bc_resnet:BCResNet6 from ../model/runs/bcresnet6/best.pt**.

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

Threshold **0.2090** — chosen on isolated validation: lowest threshold with FPR <= 0.01 in every negative category of deployment-like audio (excluding real_external). Streaming data never influenced it. Detection: fire when 2 of the last 3 window scores reach it, 1-window smoothing, 1.0 s refractory, hit zone [wake start − 0.0 s, wake end + 1.5 s]; sliding 1.5 s windows every 100 ms via `StreamingPreprocessor`.

Per-category thresholds (deployment-like validation negatives; the operating threshold is the largest): negative_real_background 0.2090, negative_confusable 0.2039, negative_general_speech 0.1120, negative_partial_wakeword 0.1038, negative_silence_noise 0.0778, other (pooled small categories) 0.0583, negative_media 0.0434.

## Isolated validation

ROC AUC **0.8914**, average precision **0.8373**. At the operating threshold: detection rate **84.6%** [73.5, 92.4]% (55/65), false-positive rate **0.37%** (14/3783).

**Positives by source:**

| Subset | n | Detected | Rate (95% CI) |
|---|---:|---:|---|
| real_device_rpi | 37 | 37 | 100.0% [90.5, 100.0]% |
| real_other_device | 1 | 1 | 100.0% [2.5, 100.0]% |
| synthetic | 27 | 17 | 63.0% [42.4, 80.6]% |

Owner-verified low-signal RPI positives in this split: Manual-Fil-RPI19 score 0.328 (detected).

**Negatives by category:**

| Category | n | False positives | FPR (95% CI) | Score p99 | Score max |
|---|---:|---:|---|---:|---:|
| negative_silence_noise | 593 | 7 | 1.18% [0.48, 2.42]% | 0.224 | 0.583 |
| negative_real_background | 439 | 4 | 0.91% [0.25, 2.32]% | 0.196 | 0.616 |
| negative_media | 266 | 2 | 0.75% [0.09, 2.69]% | 0.138 | 0.445 |
| negative_confusable | 1176 | 1 | 0.09% [0.00, 0.47]% | 0.078 | 0.819 |
| negative_general_speech | 1075 | 0 | 0.00% [0.00, 0.34]% | 0.076 | 0.162 |
| negative_non_speech_from_positive_recording | 7 | 0 | 0.00% [0.00, 40.96]% | 0.057 | 0.058 |
| negative_partial_wakeword | 227 | 0 | 0.00% [0.00, 1.61]% | 0.053 | 0.127 |

**Negatives by source:**

| Subset | n | False positives | FPR (95% CI) |
|---|---:|---:|---|
| real_device_rpi | 479 | 5 | 1.04% [0.34, 2.42]% |
| real_external | 2532 | 9 | 0.36% [0.16, 0.67]% |
| real_other_device | 2 | 0 | 0.00% [0.00, 84.19]% |
| synthetic | 770 | 0 | 0.00% [0.00, 0.48]% |

`real_external` = public-corpus negatives (MSWC, FLEURS, MUSAN, Hey Snips): reported, never used to choose the threshold. Per category and source: `negatives_by_category_and_subset` in `results.json`.

## Streaming (always listening)

Overall: **26.85 false accepts/hour** [19.80, 35.60] (48 in 1.79 h of wakeword-free exposure); detection rate **19.1%** [14.0, 25.1]% (40/209, 169 misses); median latency **0.04 s** (p90 0.34 s) after the end of the wakeword; 0 duplicate triggers.

With the plain rule (fire on any single window at the same threshold): 41.40 false accepts/hour, detection 20.1%.

- **FA/hour** = false accepts ÷ hours during which a false accept was possible (stream time after the first full window, minus every wakeword hit zone). Intervals are exact Poisson 95% CIs.
- **Latency** = trigger time (end of the triggering window) − wakeword end; negative means it fired before the word ended.

| Set / condition | Hours | FA | FA/h (95% CI) | Wakewords | Detected | Detection rate (95% CI) | Median latency |
|---|---:|---:|---|---:|---:|---|---:|
| set: A_phase1_holdout | 0.86 | 5 | 5.80 [1.9, 13.5] | 128 | 32 | 25.0% [17.8, 33.4]% | 0.00 s |
| set: B_composed | 0.93 | 43 | 46.48 [33.6, 62.6] | 81 | 8 | 9.9% [4.4, 18.5]% | 0.27 s |
| condition: continuous_speech | 0.15 | 7 | 45.67 [18.4, 94.1] | 15 | 0 | 0.0% [0.0, 21.8]% | – |
| condition: environmental_noise | 0.15 | 2 | 13.22 [1.6, 47.7] | 18 | 1 | 5.6% [0.1, 27.3]% | 0.74 s |
| condition: loud_continuous_speech | 0.86 | 5 | 5.80 [1.9, 13.5] | 128 | 32 | 25.0% [17.8, 33.4]% | 0.00 s |
| condition: media | 0.15 | 13 | 84.56 [45.0, 144.6] | 14 | 1 | 7.1% [0.2, 33.9]% | 1.27 s |
| condition: mixed | 0.16 | 7 | 44.98 [18.1, 92.7] | 12 | 2 | 16.7% [2.1, 48.4]% | 0.27 s |
| condition: noisy_speech | 0.16 | 7 | 44.79 [18.0, 92.3] | 10 | 0 | 0.0% [0.0, 30.8]% | – |
| condition: quiet | 0.15 | 7 | 45.18 [18.2, 93.1] | 12 | 4 | 33.3% [9.9, 65.1]% | 0.24 s |

### False accepts by category

Category = the non-wakeword event overlapping the triggering 1.5 s window the most; otherwise the stream's background condition. Full table: `fa_by_category.csv`; every false accept: `streaming_false_accepts.csv`.

| Category | False accepts | Share | Events of this type | Events that caused an FA (95% CI) | FA/h in that background |
|---|---:|---:|---:|---|---|
| background:media | 12 | 25.0% | – | – | 78.06 [40.3, 136.4] |
| confusable | 9 | 18.8% | 176 | 5.1% [2.4, 9.5]% | – |
| background:noisy_speech | 6 | 12.5% | – | – | 38.39 [14.1, 83.6] |
| background:continuous_speech | 5 | 10.4% | – | – | 32.62 [10.6, 76.1] |
| background:loud_continuous_speech | 5 | 10.4% | – | – | 5.80 [1.9, 13.5] |
| partial_tail | 4 | 8.3% | 67 | 6.0% [1.7, 14.6]% | – |
| background:mixed | 3 | 6.2% | – | – | 19.28 [4.0, 56.3] |
| speech_snippet | 3 | 6.2% | 43 | 7.0% [1.5, 19.1]% | – |
| partial_head | 1 | 2.1% | 60 | 1.7% [0.0, 8.9]% | – |

### Detection by wakeword type, level and voice

| Breakdown | Group | n | Detection rate (95% CI) |
|---|---|---:|---|
| type | wakeword | 186 | 19.9% [14.4, 26.4]% |
| type | wakeword_embedded | 23 | 13.0% [2.8, 33.6]% |
| level (peak over floor) | high (>=18 dB) | 42 | 7.1% [1.5, 19.5]% |
| level (peak over floor) | low (<10 dB) | 10 | 10.0% [0.3, 44.5]% |
| level (peak over floor) | mid (10-18 dB) | 29 | 13.8% [3.9, 31.7]% |
| level (peak over floor) | unmeasured (Set A) | 128 | 25.0% [17.8, 33.4]% |
| voice | en_GB-alan-medium | 26 | 3.8% [0.1, 19.6]% |
| voice | en_GB-alba-medium | 31 | 0.0% [0.0, 11.2]% |
| voice | en_US-amy-medium | 24 | 8.3% [1.0, 27.0]% |
| voice | en_US-lessac-medium | 34 | 23.5% [10.7, 41.2]% |
| voice | en_US-ryan-medium | 28 | 28.6% [13.2, 48.7]% |
| voice | fil-PH-AngeloNeural | 10 | 30.0% [6.7, 65.2]% |
| voice | fil-PH-BlessicaNeural | 15 | 60.0% [32.3, 83.7]% |
| voice | id_ID-news_tts-medium | 20 | 35.0% [15.4, 59.2]% |
| voice | vi_VN-vivos-x_low | 21 | 9.5% [1.2, 30.4]% |

### Wakeword positions

Relative start positions span 0.01–0.98 of the stream (quartiles 0.33, 0.48, 0.65). By set: A_phase1_holdout 0.21–0.77; B_composed 0.01–0.98. Set B places events uniformly; Set A (phase-1) restricted wakewords to the middle 20–80 %. With a 100 ms hop the wakeword also lands at every offset inside the sliding windows.

### Threshold sweep

`streaming_sweep.csv` (FA/h vs miss rate per set) and the `sweep` list in `results.json` (isolated detection rate vs FPR) give the full trade-off, independent of the chosen operating point.

## Caveats

- **Set A ground truth includes TTS silence padding** (phase-1 insertion metadata, used verbatim; several spans are 1.87 s long). Its wakeword 'end' is late, so Set A latencies are biased negative and not comparable with Set B, whose spans are energy-trimmed. Set A detection and false-accept counts are unaffected: the hit zone still starts at the wakeword start.
- **Levels differ between sets.** Set A is peak-normalised (loud, like phase-1 synthetic data); Set B is at the measured device level (floor −58 to −51 dB, events 5–26 dB above the floor). The louder `environmental_noise`, `noisy_speech` and `mixed` floors (−50 to −38 dB) are an assumption, not a measurement.
- **Small isolated positive counts** (validation: real_device_rpi 37, real_other_device 1, synthetic 27) give wide intervals; they are reported with every rate.
- **Latency** is measured from the end of the triggering 1.5 s window, i.e. the earliest moment a streaming detector could fire, excluding model compute time.

