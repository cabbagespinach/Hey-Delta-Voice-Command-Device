# Evaluation: isolated clips and always-listening streams

| File | Deliverable |
|---|---|
| `isolated_eval.py` | Isolated validation evaluator (fixed manifest in `isolated_set/`) |
| `build_streaming_set.py` → `streaming_set/{streams,events}.csv`, `data/streaming_eval_composed/` | Streaming evaluation dataset and manifest |
| `streaming_eval.py`, `detection.py` | Streaming evaluator: sliding-window inference, triggers, matching, metrics |
| `run_evaluation.py` → `results/metrics_report.md`, `results/results.json`, CSVs | Metrics report and machine-readable results |
| `results/fa_by_category.csv`, `results/streaming_false_accepts.csv`, `results/isolated_false_positives.csv` | Category-level false-accept analysis |
| `eval_config.json` | Every evaluation setting, with reasons |
| `test_evaluation.py` | Acceptance checks and metric-logic unit tests (`python test_evaluation.py`) |
| `baseline_model.py` → `baseline/` | Pipeline-check baseline (not the project model) |
| `diagnostics/source_confound.py` → `diagnostics/source_confound_report.md` / `.json` | Checks whether a model scores the recording condition (real device vs synthetic, level, noise floor) instead of the wakeword. Validation only; rerun it for every trained model |

```
python build_streaming_set.py        # once; the set is then frozen (refuses to rebuild without --force)
python baseline_model.py             # or bring your own model
python run_evaluation.py [--checkpoint m.pt --model-class module:Class]
python test_evaluation.py
```

Any model mapping features `[B, 1, 40, 147]` to logits `[B]` can be evaluated.

## Evaluation sets

| Set | Contents | Real / synthetic | Role |
|---|---|---|---|
| **Isolated validation** | All 847 validation windows (42 positive), fixed preprocessing, natural class mix | 15 real RPI + 27 synthetic positives; real and synthetic negatives | Metrics, and the **only** data used to choose the threshold |
| Isolated test | The test split, same evaluator (`isolated_eval.evaluate(scorer, "test", threshold)`) | real + synthetic | Final held-out check (not read by streaming) |
| **Streaming Set A** | 120 × 30 s phase-1 holdout streams: loud synthetic continuous speech + pink noise, 128 wakewords (middle 20–80 % of each stream) | synthetic | Metrics only |
| **Streaming Set B** | 60 × 60 s streams composed for evaluation (below), 81 wakewords | synthetic (new Piper TTS + procedural audio) | Metrics only |

- **The isolated set is frozen.** `isolated_set/isolated_validation_manifest.csv` lists every window, and the `.json`
  next to it stores fingerprints (manifest hash, preprocessing feature hash, normalization-statistics hash). Every run
  verifies them and refuses to continue if anything changed.
- **Set B** was built with new Piper syntheses, as chosen, so the test split stays untouched. It has:
  - **Six conditions**, 10 streams each: quiet, environmental noise (white, pink, hum, fan, clicks, rain, traffic
    rumble), continuous speech, media (podcast speech over music), noisy speech, and mixed (alternating speech and
    music).
  - **Events in every stream:** 2–4 confusables, 1–3 partial wakewords (head or tail cut of a new "Hey Delta"),
    speech snippets in the quiet and noise conditions, and 0–3 complete "Hey Delta". 30% of wakewords are embedded in
    a sentence ("Okay, … Hey Delta … turn on the lights").
  - **Device-level loudness:** the floor is U(−58.2, −51.4) dB and events sit U(5.4, 25.7) dB above it (the measured
    RPI range). The louder noise floors (−50 to −38 dB) are an assumption.
  - **Random positions:** every event is placed uniformly at random, with only spacing rules. Wakeword starts span
    1–98% of the stream (tested against a uniform distribution).
- **Independence, enforced in code** (`streaming_eval.verify_set`, run on every evaluation):
  - Every stream file matches its frozen sha256.
  - No stream id is a train/validation/test source group, and no stream file is a window source.
  - The streaming set and the isolated set share no file or source group.
  - Set B text (40 sentences, 30 confusables, 16 carrier phrases) is checked against every training transcription;
    the build refuses any overlap, and any negative text containing the wakeword. That check caught "Hey Delta-ray",
    which was replaced.
  - Set B wakewords use speeds and pitches off the training grid. Their highest normalized cross-correlation with any
    existing positive clip is 0.59 (median 0.31), so no clip reproduces training audio.
- **Piper is stochastic**, so the set cannot be regenerated bit for bit. It is built once and frozen by its checksums.
- **The streaming sets contain no real device audio.** Real RPI behaviour is measured by isolated evaluation (subset
  `real_device_rpi`), and every report keeps real and synthetic results separate.

## Streaming inference and metrics

- **Inference is exactly the deployment front end:** `preprocessing.StreamingPreprocessor` (1.5 s windows every
  100 ms, the same features and frozen normalization as training), fed 0.5 s audio chunks as a capture loop would.
  Results are the same for any chunk size (tested).
- **Triggers (2026-09-29):** a trigger fires when at least 2 of the last 3 window scores (100 ms apart) reach the
  threshold (`detection.k_of_n`), followed by a 1 s refractory period. The rule was fixed in advance, not tuned on
  streaming data. The report also gives the plain rule (any single window) at the same threshold.
- **Matching:**
  - A trigger at time `t` (the end of its window) detects a wakeword if `wake_start ≤ t ≤ wake_end + 1.5 s`.
  - The first such trigger is the detection, with latency `t − wake_end`. Later ones are duplicates, not false
    accepts.
  - Any other trigger is a false accept.
- **False accepts per hour** = false accepts ÷ time during which a false accept was possible. That is stream time
  after the first full window, minus the hit zones. Intervals are exact Poisson 95% CIs.
- **False-accept category:** the non-wakeword event (confusable, partial head/tail, speech snippet) that most overlaps
  the triggering window, otherwise `background:<condition>`. For event categories the report also gives the share of
  events of that type that caused a false accept.
- **Also reported:** detection rate and misses (Clopper-Pearson CIs), latency (median, p90, range), duplicates, and
  breakdowns by set, acoustic condition, wakeword type (isolated / embedded), level bucket (peak over floor) and
  voice.
- **Threshold sweeps:** a false-accepts-per-hour vs miss-rate trade-off per set (`streaming_sweep.csv`) and an
  isolated detection-rate vs FPR trade-off (`results.json`).
- **Operating threshold (owner-approved, 2026-09-29):** the lowest threshold at which **every** negative category
  of deployment-like validation audio has FPR ≤ 1%. Deployment-like means our device recordings plus synthetic
  audio; public-corpus `real_external` negatives are excluded.
  - Categories with fewer than 100 windows are pooled into one group.
  - `real_external` negatives are reported separately at that threshold.
  - Streaming results never influence it.
  - Before, the rule was a pooled 1% over all validation negatives. Once about 2,500 easy public-corpus negatives
    joined validation, the pooled rate hid 2.7% on real device background.
- **Result folders:** `run_evaluation.py --out <dir>` (and `--results/--out` on both diagnostics) keep each model's
  evaluation separate. The BC-ResNet runs live in `../model/runs/*/eval`.

## Determinism

- Isolated features and scores are bit-identical across runs and worker counts.
- Streaming scores are bit-identical across runs, and chunk-size invariant.
- Two complete `run_evaluation.py` runs produced identical `results.json` content.
- Scoring runs on CPU with one thread per worker (`reproducibility.single_thread`).

## Current results (pipeline-check baseline)

`baseline_model.py` trains a small CNN (8 epochs, about 30 s on GPU, train split only, through the dataloading layer).
It exists to exercise every metric and **is not the project model**.

**Augmentation v3 (2026-09-28).** `diagnostics/source_confound.py` found the v2-trained baseline scoring the device
recording condition instead of the wakeword. Augmentation v3 and source-balanced sampling followed (see
`../deployment_driven_augmentation_strategy/augmentation_strategy.md`). Same architecture, epochs and seed:

| | v2 baseline | v3 baseline (current) |
|---|---:|---:|
| Isolated AUC / average precision | 0.958 / 0.613 | **0.978 / 0.726** |
| Isolated detection: real RPI / synthetic (at 1% FPR) | 13/15 / 6/27 | 11/15 / **13/27** |
| Confound: device-conditioned synthetic negatives newly above threshold | +40 | **+18** |
| Confound: real RPI positives made loud, mean score change | −0.26 | **−0.04** |
| Level-matched source AUC (real vs synthetic negatives) | 0.76 | **0.35** |
| Streaming false accepts per hour (A / B / all) | 9.3 / 244 / 131 | **5.8 / 96 / 53** |
| Streaming detections (A / B) at the chosen threshold | 9/128, 18/81 | 3/128, 12/81 |

The v3 threshold is higher (0.34 vs 0.28) because validation negatives score lower, so streaming detections at the
chosen threshold fall. At a matched 100 false accepts per hour, Set B detection is 15% (v3) vs 0% (v2). Neither
baseline reaches a usable operating point; the next step is a properly trained model. The v2 checkpoint is kept as
`baseline/baseline_cnn_aug_v2.pt`.

**Retrain after the 2026-09-29 recordings.** Manual-BG-RPI20..36 and Manual-Fil-RPI65..82 were integrated.
Train now has 9,838 windows (400 positive, up from 268; most of the new positives are real RPI). The normalization
statistics were recomputed, and the isolated set was re-frozen as v3 (1,313 windows, 64 positive: 37 real, 27
synthetic). v2 is in `isolated_set/archive/v2_2026-09-28/`. The previous checkpoint, results and confound report are in
`archive_pre_2026-09-29_recordings/`. Same architecture, epochs and seed. v2 and v3 isolated numbers are not directly comparable.
Later the same day, bounds marked by ear for Manual-Fil-Phonemic1/2 added the first phone positive to validation. The
isolated set was re-frozen as v4 (1,316 windows, 65 positive). Train was unchanged, so the model was not retrained.
Phonemic2 scores 0.68 and is detected. AUC 0.943, and every other figure below is unchanged.

| | before (isolated v2) | after (isolated v3) |
|---|---:|---:|
| Isolated AUC / average precision | 0.968 / 0.664 | 0.942 / 0.667 |
| Detection at 1% FPR: real RPI / synthetic | 11/15 / 13/27 | 29/37 / **6/27** |
| Real-background false positives | 2/191 | 9/439 (6 from BG-RPI29, 2 on "Hey Martha" in BG-RPI32, 1 BG-RPI8) |
| Synthetic positives given device level + floor, mean score change | +0.13 | **+0.24** (6 → 19 above threshold) |
| Synthetic speech-like negatives given device level + floor, above threshold | 4 → 35 | 0 → 14 |
| Real RPI positives made loud, mean score change | −0.00 | +0.05 |
| Real RPI negatives made loud, above threshold | 5 → 11 | 12 → 44 |
| Level-matched source AUC (real vs synthetic negatives) | 0.54 | 0.43 |
| Streaming false accepts per hour (A / B / all) | 17 / 214 / 119 | 3.5 / 102 / 54 |
| Streaming detections (A / B) | 25/128, 18/81 | 13/128, 15/81 |

The negative-side confound improved, but the positive side got worse: the model now relies more on the device
recording condition to call a positive. Clean synthetic positives are mostly missed and become detectable once they
are given device level and noise floor. BG-RPI29 has no speech that ASR or VAD could find, yet it contributes 6 false
positives. Those windows hold faint events 4–9 dB above the room floor, about the same margin as the low-signal RPI
positives.

**Retrain after the public-corpus negatives (2026-09-29, evening).** MSWC near-miss words, FLEURS, MUSAN and Sonos
"Hey Snips" were added as negatives (source `ext_*`, eval subset `real_external`), along with the owner's near-miss
recordings Manual-BG-Macmic2 and Manual-BG-Phonemic1, and MIT IR Survey reverberation. The isolated set was
re-frozen as v5: 3,848 windows, 65 positive, 2,532 real_external negatives. v4 is in `isolated_set/archive/v4_2026-09-29/`.
The previous checkpoint and reports are in `archive_pre_2026-09-29_external/`.

**Threshold caveat:** the 1% FPR rule now runs over many easy external negatives, so the operating threshold fell
from 0.280 to 0.066. The "like-for-like" column instead uses the threshold that gives 1% FPR on the non-external
negatives only (0.166).

| | before (v4) | after, official thr 0.066 | after, like-for-like thr 0.166 |
|---|---:|---:|---:|
| Isolated AUC / AP | 0.943 / 0.667 | 0.967 / 0.739 | – |
| Real RPI positives detected | 29/37 | 34/37 | 30/37 |
| Synthetic positives detected | 6/27 | 15/27 | **12/27** |
| Synthetic positives given device level + floor, mean score change | +0.24 | **+0.10** | – |
| Level-matched source AUC (real vs synthetic negatives) | 0.43 | 0.38 | – |
| Real RPI negatives made loud, mean score change | +0.056 | +0.036 | – |
| Qualcomm "Hey/Hi + word", as recorded / at device level (`diagnostics/hey_phrase_negatives_report.md`) | – | 0.56% / 9.5% | 0.12% / 4.0% |
| Streaming false accepts per hour / detection | 54 / 13% | 78 / 28% | – |

The positive-side confound roughly halved and synthetic detection doubled at a matched threshold. Two problems
remain:
1. Real "Hey + word" phrases at device level: "Hi Galaxy" is accepted on 13% of clips even at 0.166.
2. Loud real speech still raises scores.

17 of the external false positives are MUSAN clips, mostly free-sound noises 0349 and 0834.

The v2 results that follow are kept for the record.

- **Isolated validation:** AUC 0.958. At the threshold (0.280) the false-positive rate is 0.99%. Detection is 86.7% of
  real RPI positives but only 22.2% of synthetic positives. False positives come from confusables (5.6%) and real
  device background (1 of 25).
- **Streaming:** 130.9 false accepts per hour, and 12.9% of wakewords detected.
  - Nearly all false accepts come from Set B, which is at device level (244 per hour, vs 9.3 per hour in the loud
    Set A).
  - The largest false-accept sources are speech and media backgrounds, confusables (22% of confusable events trigger)
    and speech snippets.
  - I checked the matching logic directly against the raw scores: no wakeword whose in-zone score cleared the
    threshold was counted as missed. The weak streaming result reflects the baseline's poor synthetic-wakeword recall
    and its false accepts on quiet speech.

## Caveats

- **Set A latencies are not comparable with Set B.** Set A's ground-truth spans include TTS silence padding, so its
  latencies are biased negative. Detection and false-accept counts are unaffected.
- **Small positive counts.** Isolated validation has 42 positives and streaming has 209 wakewords; every rate is
  reported with its interval.
