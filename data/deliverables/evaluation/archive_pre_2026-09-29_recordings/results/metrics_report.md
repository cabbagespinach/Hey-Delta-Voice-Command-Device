# Evaluation report

Generated 2026-09-28T14:53:52+00:00 by `run_evaluation.py`.
Model: **pipeline-check baseline CNN (`baseline_model.py`, train split only) — not the project model**.

## Data and independence

| Set | Audio | Real / synthetic | Used for |
|---|---|---|---|
| Isolated validation | 1013 fixed windows (`isolated_set/isolated_validation_manifest.csv`) | positives: real_device_rpi 15, synthetic 27 | metrics + choosing the threshold |
| Streaming Set A (phase-1 holdout) | 120 streams, 0.86 h scored exposure | synthetic | metrics only |
| Streaming Set B (composed for evaluation) | 60 streams, 0.93 h scored exposure | synthetic (new Piper TTS + procedural audio) | metrics only |

- Streaming set verified: 180 files match their frozen sha256; no stream id, file or source group appears in train, validation, test or the isolated set (`streaming_eval.verify_set`).
- Set B text is eval-only (no sentence, confusable or carrier phrase equals a training transcription), and wakewords are new stochastic syntheses off the training speed/pitch grid; see `streaming_set/build_log.json` for the near-duplicate check against existing positives.
- **Streaming evaluation contains no real device audio.** Real RPI performance is measured only by isolated validation (subset `real_device_rpi`).

## Operating point

Threshold **0.3343** — chosen on isolated validation: lowest threshold with FPR <= 0.01. Streaming data never influenced it. Detection: 1-window smoothing, 1.0 s refractory, hit zone [wake start − 0.0 s, wake end + 1.5 s]; sliding 1.5 s windows every 100 ms via `StreamingPreprocessor`.

## Isolated validation

ROC AUC **0.9681**, average precision **0.6642**. At the operating threshold: detection rate **57.1%** [41.0, 72.3]% (24/42), false-positive rate **0.93%** (9/971).

**Positives by source:**

| Subset | n | Detected | Rate (95% CI) |
|---|---:|---:|---|
| real_device_rpi | 15 | 11 | 73.3% [44.9, 92.2]% |
| synthetic | 27 | 13 | 48.1% [28.7, 68.1]% |

Owner-verified low-signal RPI positives in this split: Manual-Fil-RPI19 score 0.188 (missed).

**Negatives by category:**

| Category | n | False positives | FPR (95% CI) | Score p99 | Score max |
|---|---:|---:|---|---:|---:|
| negative_confusable | 125 | 5 | 4.00% [1.31, 9.09]% | 0.811 | 1.000 |
| negative_partial_wakeword | 147 | 2 | 1.36% [0.17, 4.83]% | 0.277 | 0.713 |
| negative_real_background | 191 | 2 | 1.05% [0.13, 3.73]% | 0.220 | 0.735 |
| negative_general_speech | 276 | 0 | 0.00% [0.00, 1.33]% | 0.021 | 0.071 |
| negative_media | 126 | 0 | 0.00% [0.00, 2.89]% | 0.039 | 0.121 |
| negative_silence_noise | 106 | 0 | 0.00% [0.00, 3.42]% | 0.070 | 0.075 |

Real vs synthetic negatives: see `negatives_by_category_and_subset` in `results.json`.

## Streaming (always listening)

Overall: **119.16 false accepts/hour** [103.70, 136.29] (213 in 1.79 h of wakeword-free exposure); detection rate **20.6%** [15.3, 26.7]% (43/209, 166 misses); median latency **-0.10 s** (p90 0.66 s) after the end of the wakeword; 2 duplicate triggers.

- **FA/hour** = false accepts ÷ hours during which a false accept was possible (stream time after the first full window, minus every wakeword hit zone). Intervals are exact Poisson 95% CIs.
- **Latency** = trigger time (end of the triggering window) − wakeword end; negative means it fired before the word ended.

| Set / condition | Hours | FA | FA/h (95% CI) | Wakewords | Detected | Detection rate (95% CI) | Median latency |
|---|---:|---:|---|---:|---:|---|---:|
| set: A_phase1_holdout | 0.86 | 15 | 17.40 [9.7, 28.7] | 128 | 25 | 19.5% [13.1, 27.5]% | -0.13 s |
| set: B_composed | 0.93 | 198 | 214.02 [185.2, 246.0] | 81 | 18 | 22.2% [13.7, 32.8]% | 0.03 s |
| condition: continuous_speech | 0.15 | 57 | 371.90 [281.7, 481.8] | 15 | 4 | 26.7% [7.8, 55.1]% | 0.36 s |
| condition: environmental_noise | 0.15 | 14 | 92.51 [50.6, 155.2] | 18 | 5 | 27.8% [9.7, 53.5]% | -0.10 s |
| condition: loud_continuous_speech | 0.86 | 15 | 17.40 [9.7, 28.7] | 128 | 25 | 19.5% [13.1, 27.5]% | -0.13 s |
| condition: media | 0.15 | 43 | 279.71 [202.4, 376.8] | 14 | 3 | 21.4% [4.7, 50.8]% | -0.72 s |
| condition: mixed | 0.16 | 38 | 244.17 [172.8, 335.1] | 12 | 2 | 16.7% [2.1, 48.4]% | -0.09 s |
| condition: noisy_speech | 0.16 | 24 | 153.56 [98.4, 228.5] | 10 | 1 | 10.0% [0.3, 44.5]% | 1.41 s |
| condition: quiet | 0.15 | 22 | 142.00 [89.0, 215.0] | 12 | 3 | 25.0% [5.5, 57.2]% | 0.09 s |

### False accepts by category

Category = the non-wakeword event overlapping the triggering 1.5 s window the most; otherwise the stream's background condition. Full table: `fa_by_category.csv`; every false accept: `streaming_false_accepts.csv`.

| Category | False accepts | Share | Events of this type | Events that caused an FA (95% CI) | FA/h in that background |
|---|---:|---:|---:|---|---|
| background:continuous_speech | 43 | 20.2% | – | – | 280.56 [203.0, 377.9] |
| confusable | 36 | 16.9% | 176 | 19.3% [13.8, 25.9]% | – |
| background:media | 32 | 15.0% | – | – | 208.16 [142.4, 293.9] |
| background:mixed | 30 | 14.1% | – | – | 192.77 [130.1, 275.2] |
| background:noisy_speech | 19 | 8.9% | – | – | 121.57 [73.2, 189.8] |
| speech_snippet | 17 | 8.0% | 43 | 30.2% [17.2, 46.1]% | – |
| background:loud_continuous_speech | 15 | 7.0% | – | – | 17.40 [9.7, 28.7] |
| partial_head | 10 | 4.7% | 60 | 13.3% [5.9, 24.6]% | – |
| partial_tail | 9 | 4.2% | 67 | 11.9% [5.3, 22.2]% | – |
| background:environmental_noise | 2 | 0.9% | – | – | 13.22 [1.6, 47.7] |

### Detection by wakeword type, level and voice

| Breakdown | Group | n | Detection rate (95% CI) |
|---|---|---:|---|
| type | wakeword | 186 | 20.4% [14.9, 26.9]% |
| type | wakeword_embedded | 23 | 21.7% [7.5, 43.7]% |
| level (peak over floor) | high (>=18 dB) | 42 | 23.8% [12.1, 39.5]% |
| level (peak over floor) | low (<10 dB) | 10 | 0.0% [0.0, 30.8]% |
| level (peak over floor) | mid (10-18 dB) | 29 | 27.6% [12.7, 47.2]% |
| level (peak over floor) | unmeasured (Set A) | 128 | 19.5% [13.1, 27.5]% |
| voice | en_GB-alan-medium | 26 | 3.8% [0.1, 19.6]% |
| voice | en_GB-alba-medium | 31 | 3.2% [0.1, 16.7]% |
| voice | en_US-amy-medium | 24 | 8.3% [1.0, 27.0]% |
| voice | en_US-lessac-medium | 34 | 26.5% [12.9, 44.4]% |
| voice | en_US-ryan-medium | 28 | 39.3% [21.5, 59.4]% |
| voice | fil-PH-AngeloNeural | 10 | 40.0% [12.2, 73.8]% |
| voice | fil-PH-BlessicaNeural | 15 | 53.3% [26.6, 78.7]% |
| voice | id_ID-news_tts-medium | 20 | 20.0% [5.7, 43.7]% |
| voice | vi_VN-vivos-x_low | 21 | 14.3% [3.0, 36.3]% |

### Wakeword positions

Relative start positions span 0.01–0.98 of the stream (quartiles 0.33, 0.48, 0.65). By set: A_phase1_holdout 0.21–0.77; B_composed 0.01–0.98. Set B places events uniformly; Set A (phase-1) restricted wakewords to the middle 20–80 %. With a 100 ms hop the wakeword also lands at every offset inside the sliding windows.

### Threshold sweep

`streaming_sweep.csv` (FA/h vs miss rate per set) and the `sweep` list in `results.json` (isolated detection rate vs FPR) give the full trade-off, independent of the chosen operating point.

## Caveats

- **Set A ground truth includes TTS silence padding** (phase-1 insertion metadata, used verbatim; several spans are 1.87 s long). Its wakeword 'end' is late, so Set A latencies are biased negative and not comparable with Set B, whose spans are energy-trimmed. Set A detection and false-accept counts are unaffected: the hit zone still starts at the wakeword start.
- **Levels differ between sets.** Set A is peak-normalised (loud, like phase-1 synthetic data); Set B is at the measured device level (floor −58 to −51 dB, events 5–26 dB above the floor). The louder `environmental_noise`, `noisy_speech` and `mixed` floors (−50 to −38 dB) are an assumption, not a measurement.
- **Small isolated positive counts** (validation: 15 real RPI, 27 synthetic) give wide intervals; they are reported with every rate.
- **Latency** is measured from the end of the triggering 1.5 s window, i.e. the earliest moment a streaming detector could fire, excluding model compute time.

