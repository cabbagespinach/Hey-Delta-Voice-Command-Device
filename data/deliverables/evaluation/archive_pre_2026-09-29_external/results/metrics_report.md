# Evaluation report

Generated 2026-09-29T08:07:40+00:00 by `run_evaluation.py`.
Model: **pipeline-check baseline CNN (`baseline_model.py`, train split only) — not the project model**.

## Data and independence

| Set | Audio | Real / synthetic | Used for |
|---|---|---|---|
| Isolated validation | 1316 fixed windows (`isolated_set/isolated_validation_manifest.csv`) | positives: real_device_rpi 37, real_other_device 1, synthetic 27 | metrics + choosing the threshold |
| Streaming Set A (phase-1 holdout) | 120 streams, 0.86 h scored exposure | synthetic | metrics only |
| Streaming Set B (composed for evaluation) | 60 streams, 0.93 h scored exposure | synthetic (new Piper TTS + procedural audio) | metrics only |

- Streaming set verified: 180 files match their frozen sha256; no stream id, file or source group appears in train, validation, test or the isolated set (`streaming_eval.verify_set`).
- Set B text is eval-only (no sentence, confusable or carrier phrase equals a training transcription), and wakewords are new stochastic syntheses off the training speed/pitch grid; see `streaming_set/build_log.json` for the near-duplicate check against existing positives.
- **Streaming evaluation contains no real device audio.** Real RPI performance is measured only by isolated validation (subset `real_device_rpi`).

## Operating point

Threshold **0.2799** — chosen on isolated validation: lowest threshold with FPR <= 0.01. Streaming data never influenced it. Detection: 1-window smoothing, 1.0 s refractory, hit zone [wake start − 0.0 s, wake end + 1.5 s]; sliding 1.5 s windows every 100 ms via `StreamingPreprocessor`.

## Isolated validation

ROC AUC **0.9427**, average precision **0.6718**. At the operating threshold: detection rate **55.4%** [42.5, 67.7]% (36/65), false-positive rate **0.96%** (12/1251).

**Positives by source:**

| Subset | n | Detected | Rate (95% CI) |
|---|---:|---:|---|
| real_device_rpi | 37 | 29 | 78.4% [61.8, 90.2]% |
| real_other_device | 1 | 1 | 100.0% [2.5, 100.0]% |
| synthetic | 27 | 6 | 22.2% [8.6, 42.3]% |

Owner-verified low-signal RPI positives in this split: Manual-Fil-RPI19 score 0.209 (missed).

**Negatives by category:**

| Category | n | False positives | FPR (95% CI) | Score p99 | Score max |
|---|---:|---:|---|---:|---:|
| negative_real_background | 439 | 9 | 2.05% [0.94, 3.86]% | 0.450 | 0.951 |
| negative_partial_wakeword | 172 | 2 | 1.16% [0.14, 4.14]% | 0.330 | 0.654 |
| negative_confusable | 125 | 1 | 0.80% [0.02, 4.38]% | 0.278 | 0.999 |
| negative_general_speech | 276 | 0 | 0.00% [0.00, 1.33]% | 0.005 | 0.009 |
| negative_media | 126 | 0 | 0.00% [0.00, 2.89]% | 0.027 | 0.057 |
| negative_non_speech_from_positive_recording | 7 | 0 | 0.00% [0.00, 40.96]% | 0.015 | 0.015 |
| negative_silence_noise | 106 | 0 | 0.00% [0.00, 3.42]% | 0.050 | 0.060 |

Real vs synthetic negatives: see `negatives_by_category_and_subset` in `results.json`.

## Streaming (always listening)

Overall: **54.27 false accepts/hour** [44.01, 66.20] (97 in 1.79 h of wakeword-free exposure); detection rate **13.4%** [9.1, 18.8]% (28/209, 181 misses); median latency **0.01 s** (p90 0.29 s) after the end of the wakeword; 0 duplicate triggers.

- **FA/hour** = false accepts ÷ hours during which a false accept was possible (stream time after the first full window, minus every wakeword hit zone). Intervals are exact Poisson 95% CIs.
- **Latency** = trigger time (end of the triggering window) − wakeword end; negative means it fired before the word ended.

| Set / condition | Hours | FA | FA/h (95% CI) | Wakewords | Detected | Detection rate (95% CI) | Median latency |
|---|---:|---:|---|---:|---:|---|---:|
| set: A_phase1_holdout | 0.86 | 3 | 3.48 [0.7, 10.2] | 128 | 13 | 10.2% [5.5, 16.7]% | -0.33 s |
| set: B_composed | 0.93 | 94 | 101.60 [82.1, 124.3] | 81 | 15 | 18.5% [10.8, 28.7]% | 0.03 s |
| condition: continuous_speech | 0.15 | 10 | 65.25 [31.3, 120.0] | 15 | 1 | 6.7% [0.2, 31.9]% | 0.42 s |
| condition: environmental_noise | 0.15 | 12 | 79.30 [41.0, 138.5] | 18 | 8 | 44.4% [21.5, 69.2]% | 0.02 s |
| condition: loud_continuous_speech | 0.86 | 3 | 3.48 [0.7, 10.2] | 128 | 13 | 10.2% [5.5, 16.7]% | -0.33 s |
| condition: media | 0.15 | 20 | 130.10 [79.5, 200.9] | 14 | 2 | 14.3% [1.8, 42.8]% | 0.41 s |
| condition: mixed | 0.16 | 24 | 154.22 [98.8, 229.5] | 12 | 1 | 8.3% [0.2, 38.5]% | -0.22 s |
| condition: noisy_speech | 0.16 | 14 | 89.58 [49.0, 150.3] | 10 | 1 | 10.0% [0.3, 44.5]% | 0.21 s |
| condition: quiet | 0.15 | 14 | 90.36 [49.4, 151.6] | 12 | 2 | 16.7% [2.1, 48.4]% | -0.02 s |

### False accepts by category

Category = the non-wakeword event overlapping the triggering 1.5 s window the most; otherwise the stream's background condition. Full table: `fa_by_category.csv`; every false accept: `streaming_false_accepts.csv`.

| Category | False accepts | Share | Events of this type | Events that caused an FA (95% CI) | FA/h in that background |
|---|---:|---:|---:|---|---|
| confusable | 30 | 30.9% | 176 | 16.5% [11.3, 22.8]% | – |
| background:media | 15 | 15.5% | – | – | 97.57 [54.6, 160.9] |
| background:mixed | 14 | 14.4% | – | – | 89.96 [49.2, 150.9] |
| background:noisy_speech | 9 | 9.3% | – | – | 57.59 [26.3, 109.3] |
| background:continuous_speech | 8 | 8.2% | – | – | 52.20 [22.5, 102.8] |
| speech_snippet | 7 | 7.2% | 43 | 14.0% [5.3, 27.9]% | – |
| partial_tail | 5 | 5.2% | 67 | 7.5% [2.5, 16.6]% | – |
| partial_head | 4 | 4.1% | 60 | 6.7% [1.8, 16.2]% | – |
| background:loud_continuous_speech | 3 | 3.1% | – | – | 3.48 [0.7, 10.2] |
| background:environmental_noise | 2 | 2.1% | – | – | 13.22 [1.6, 47.7] |

### Detection by wakeword type, level and voice

| Breakdown | Group | n | Detection rate (95% CI) |
|---|---|---:|---|
| type | wakeword | 186 | 11.8% [7.6, 17.4]% |
| type | wakeword_embedded | 23 | 26.1% [10.2, 48.4]% |
| level (peak over floor) | high (>=18 dB) | 42 | 16.7% [7.0, 31.4]% |
| level (peak over floor) | low (<10 dB) | 10 | 10.0% [0.3, 44.5]% |
| level (peak over floor) | mid (10-18 dB) | 29 | 24.1% [10.3, 43.5]% |
| level (peak over floor) | unmeasured (Set A) | 128 | 10.2% [5.5, 16.7]% |
| voice | en_GB-alan-medium | 26 | 7.7% [0.9, 25.1]% |
| voice | en_GB-alba-medium | 31 | 0.0% [0.0, 11.2]% |
| voice | en_US-amy-medium | 24 | 8.3% [1.0, 27.0]% |
| voice | en_US-lessac-medium | 34 | 20.6% [8.7, 37.9]% |
| voice | en_US-ryan-medium | 28 | 28.6% [13.2, 48.7]% |
| voice | fil-PH-AngeloNeural | 10 | 30.0% [6.7, 65.2]% |
| voice | fil-PH-BlessicaNeural | 15 | 20.0% [4.3, 48.1]% |
| voice | id_ID-news_tts-medium | 20 | 5.0% [0.1, 24.9]% |
| voice | vi_VN-vivos-x_low | 21 | 9.5% [1.2, 30.4]% |

### Wakeword positions

Relative start positions span 0.01–0.98 of the stream (quartiles 0.33, 0.48, 0.65). By set: A_phase1_holdout 0.21–0.77; B_composed 0.01–0.98. Set B places events uniformly; Set A (phase-1) restricted wakewords to the middle 20–80 %. With a 100 ms hop the wakeword also lands at every offset inside the sliding windows.

### Threshold sweep

`streaming_sweep.csv` (FA/h vs miss rate per set) and the `sweep` list in `results.json` (isolated detection rate vs FPR) give the full trade-off, independent of the chosen operating point.

## Caveats

- **Set A ground truth includes TTS silence padding** (phase-1 insertion metadata, used verbatim; several spans are 1.87 s long). Its wakeword 'end' is late, so Set A latencies are biased negative and not comparable with Set B, whose spans are energy-trimmed. Set A detection and false-accept counts are unaffected: the hit zone still starts at the wakeword start.
- **Levels differ between sets.** Set A is peak-normalised (loud, like phase-1 synthetic data); Set B is at the measured device level (floor −58 to −51 dB, events 5–26 dB above the floor). The louder `environmental_noise`, `noisy_speech` and `mixed` floors (−50 to −38 dB) are an assumption, not a measurement.
- **Small isolated positive counts** (validation: 15 real RPI, 27 synthetic) give wide intervals; they are reported with every rate.
- **Latency** is measured from the end of the triggering 1.5 s window, i.e. the earliest moment a streaming detector could fire, excluding model compute time.

