# Evaluation report

Generated 2026-09-29T11:38:01+00:00 by `run_evaluation.py`.
Model: **pipeline-check baseline CNN (`baseline_model.py`, train split only) — not the project model**.

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

Threshold **0.0663** — chosen on isolated validation: lowest threshold with FPR <= 0.01. Streaming data never influenced it. Detection: 1-window smoothing, 1.0 s refractory, hit zone [wake start − 0.0 s, wake end + 1.5 s]; sliding 1.5 s windows every 100 ms via `StreamingPreprocessor`.

## Isolated validation

ROC AUC **0.9669**, average precision **0.7390**. At the operating threshold: detection rate **76.9%** [64.8, 86.5]% (50/65), false-positive rate **0.98%** (37/3783).

**Positives by source:**

| Subset | n | Detected | Rate (95% CI) |
|---|---:|---:|---|
| real_device_rpi | 37 | 34 | 91.9% [78.1, 98.3]% |
| real_other_device | 1 | 1 | 100.0% [2.5, 100.0]% |
| synthetic | 27 | 15 | 55.6% [35.3, 74.5]% |

Owner-verified low-signal RPI positives in this split: Manual-Fil-RPI19 score 0.006 (missed).

**Negatives by category:**

| Category | n | False positives | FPR (95% CI) | Score p99 | Score max |
|---|---:|---:|---|---:|---:|
| negative_real_background | 439 | 12 | 2.73% [1.42, 4.73]% | 0.279 | 0.723 |
| negative_silence_noise | 593 | 13 | 2.19% [1.17, 3.72]% | 0.119 | 0.863 |
| negative_partial_wakeword | 227 | 4 | 1.76% [0.48, 4.45]% | 0.227 | 0.407 |
| negative_media | 266 | 4 | 1.50% [0.41, 3.81]% | 0.093 | 0.329 |
| negative_confusable | 1176 | 4 | 0.34% [0.09, 0.87]% | 0.014 | 0.997 |
| negative_general_speech | 1075 | 0 | 0.00% [0.00, 0.34]% | 0.003 | 0.044 |
| negative_non_speech_from_positive_recording | 7 | 0 | 0.00% [0.00, 40.96]% | 0.004 | 0.004 |

Real vs synthetic negatives: see `negatives_by_category_and_subset` in `results.json`.

## Streaming (always listening)

Overall: **78.32 false accepts/hour** [65.89, 92.43] (140 in 1.79 h of wakeword-free exposure); detection rate **28.2%** [22.2, 34.9]% (59/209, 150 misses); median latency **-0.03 s** (p90 0.51 s) after the end of the wakeword; 3 duplicate triggers.

- **FA/hour** = false accepts ÷ hours during which a false accept was possible (stream time after the first full window, minus every wakeword hit zone). Intervals are exact Poisson 95% CIs.
- **Latency** = trigger time (end of the triggering window) − wakeword end; negative means it fired before the word ended.

| Set / condition | Hours | FA | FA/h (95% CI) | Wakewords | Detected | Detection rate (95% CI) | Median latency |
|---|---:|---:|---|---:|---:|---|---:|
| set: A_phase1_holdout | 0.86 | 9 | 10.44 [4.8, 19.8] | 128 | 35 | 27.3% [19.8, 35.9]% | -0.11 s |
| set: B_composed | 0.93 | 131 | 141.60 [118.4, 168.0] | 81 | 24 | 29.6% [20.0, 40.8]% | 0.02 s |
| condition: continuous_speech | 0.15 | 15 | 97.87 [54.8, 161.4] | 15 | 3 | 20.0% [4.3, 48.1]% | 0.41 s |
| condition: environmental_noise | 0.15 | 14 | 92.51 [50.6, 155.2] | 18 | 8 | 44.4% [21.5, 69.2]% | -0.12 s |
| condition: loud_continuous_speech | 0.86 | 9 | 10.44 [4.8, 19.8] | 128 | 35 | 27.3% [19.8, 35.9]% | -0.11 s |
| condition: media | 0.15 | 21 | 136.60 [84.6, 208.8] | 14 | 3 | 21.4% [4.7, 50.8]% | 0.51 s |
| condition: mixed | 0.16 | 25 | 160.64 [104.0, 237.1] | 12 | 2 | 16.7% [2.1, 48.4]% | -0.03 s |
| condition: noisy_speech | 0.16 | 35 | 223.95 [156.0, 311.5] | 10 | 3 | 30.0% [6.7, 65.2]% | 0.03 s |
| condition: quiet | 0.15 | 21 | 135.55 [83.9, 207.2] | 12 | 5 | 41.7% [15.2, 72.3]% | 0.01 s |

### False accepts by category

Category = the non-wakeword event overlapping the triggering 1.5 s window the most; otherwise the stream's background condition. Full table: `fa_by_category.csv`; every false accept: `streaming_false_accepts.csv`.

| Category | False accepts | Share | Events of this type | Events that caused an FA (95% CI) | FA/h in that background |
|---|---:|---:|---:|---|---|
| confusable | 31 | 22.1% | 176 | 16.5% [11.3, 22.8]% | – |
| background:noisy_speech | 26 | 18.6% | – | – | 166.36 [108.7, 243.8] |
| background:media | 16 | 11.4% | – | – | 104.08 [59.5, 169.0] |
| background:mixed | 16 | 11.4% | – | – | 102.81 [58.8, 167.0] |
| background:continuous_speech | 10 | 7.1% | – | – | 65.25 [31.3, 120.0] |
| partial_tail | 10 | 7.1% | 67 | 13.4% [6.3, 24.0]% | – |
| background:loud_continuous_speech | 9 | 6.4% | – | – | 10.44 [4.8, 19.8] |
| partial_head | 8 | 5.7% | 60 | 13.3% [5.9, 24.6]% | – |
| speech_snippet | 8 | 5.7% | 43 | 11.6% [3.9, 25.1]% | – |
| background:environmental_noise | 5 | 3.6% | – | – | 33.04 [10.7, 77.1] |
| background:quiet | 1 | 0.7% | – | – | 6.45 [0.2, 36.0] |

### Detection by wakeword type, level and voice

| Breakdown | Group | n | Detection rate (95% CI) |
|---|---|---:|---|
| type | wakeword | 186 | 28.5% [22.1, 35.6]% |
| type | wakeword_embedded | 23 | 26.1% [10.2, 48.4]% |
| level (peak over floor) | high (>=18 dB) | 42 | 26.2% [13.9, 42.0]% |
| level (peak over floor) | low (<10 dB) | 10 | 30.0% [6.7, 65.2]% |
| level (peak over floor) | mid (10-18 dB) | 29 | 34.5% [17.9, 54.3]% |
| level (peak over floor) | unmeasured (Set A) | 128 | 27.3% [19.8, 35.9]% |
| voice | en_GB-alan-medium | 26 | 23.1% [9.0, 43.6]% |
| voice | en_GB-alba-medium | 31 | 0.0% [0.0, 11.2]% |
| voice | en_US-amy-medium | 24 | 29.2% [12.6, 51.1]% |
| voice | en_US-lessac-medium | 34 | 35.3% [19.7, 53.5]% |
| voice | en_US-ryan-medium | 28 | 39.3% [21.5, 59.4]% |
| voice | fil-PH-AngeloNeural | 10 | 50.0% [18.7, 81.3]% |
| voice | fil-PH-BlessicaNeural | 15 | 53.3% [26.6, 78.7]% |
| voice | id_ID-news_tts-medium | 20 | 40.0% [19.1, 63.9]% |
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

