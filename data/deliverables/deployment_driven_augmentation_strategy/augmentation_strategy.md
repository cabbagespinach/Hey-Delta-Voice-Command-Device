# Deployment-Driven Augmentation Strategy — Hey Delta (v4)

## Scope

This is a **strategy-only** deliverable. The transforms are implemented in `../dataloading/augmentation.py`.

## What changed in v4 (2026-09-29)

- **New transform, `random_mic_coloring` (p = 0.5, every category, applied first).** It adds a mild random spectral
  colour: shelves of ±6 dB and one ±4 dB peak.
  - *Why:* after v3, a baseline CNN still accepted real "Hey/Hi + word" phrases (Qualcomm corpus) once they were
    given the device condition. Every real positive carries the owner's own microphones' colour.
  - *Owner-approved as an interim measure.* The RPI's true response stays DEFERRED (section 6) until a simultaneous
    RPI + reference recording exists. The transform makes colour uninformative about the label; it does not claim
    to reproduce the RPI microphone.
- **`rir_reverberation` now has its curated bank.** 132 real impulse responses from the MIT IR Survey (CC-BY 4.0),
  with RT60 in the configured 0.15–0.6 s range (`build_rir_bank.py` → `data/rir/rir_bank.csv`).
- `check_augmentation_config.py` checks that the colouring is label-agnostic, mild and first in order (24/24 pass).

## What changed in v3 (2026-09-28)

v2 is archived in `archive/v2_2026-09-28/`.

**Finding.** `../evaluation/diagnostics/source_confound.py` (validation only) showed that a model trained under v2
scored the **device recording condition** instead of the wakeword. In that condition, speech sits quietly over a noise
floor:
- Giving synthetic negatives that condition raised the number above threshold from 6 to 46 (confusables: 6 to 28).
  Changing only the level, or only the floor, had little effect.
- Raising real RPI positives to synthetic loudness lost detections (13 to 9 of 15).

**Cause.** Every real RPI positive carries the device condition, but under v2 only about 30% of augmented synthetic
examples did (gain p 0.5 × mixing p 0.6), and real audio was 64% of positives against 6% of negatives. So the
condition predicted the label.

**Principle.** Deployment audio is always device-conditioned, so training audio of **both** labels should be too.

| Item | v2 | v3 |
|---|---|---|
| `device_level_gain` probability | 0.5 | **0.9** |
| `device_level_gain` target | −49.7 to −27.5 dB (p10–p90) | **−49.7 to −18.8 dB** (p10 to measured max). Quiet real clips can now be raised, not only synthetic ones lowered. |
| `background_noise_mixing` probability | 0.6 | **0.9**, and noise is added **only if the example is cleaner than the target**. Real device audio is not pushed below the device range. |
| Sampler (`dataloading/dataloader_config.json`) | one positive bucket | **Positive quota split evenly between real and synthetic**; real device background 8% → 14%. Real share of draws: positives 64% → 50%, negatives 6% → 26%. |

**Effect on the same pipeline-check baseline** (same architecture, epochs and seed; validation only):
- Device-conditioned synthetic negatives above threshold: +40 → **+18**; confusables +22 → **+12**.
- Real RPI positives raised to synthetic loudness: −0.26 → **−0.04** mean score (no detections lost).
- Level-matched source AUC: 0.76 → **0.35**.
- Isolated AUC 0.958 → 0.978; streaming false accepts per hour 131 → 53.
- The residual confound is mostly synthetic positives (one validation voice) scoring higher when device-conditioned.
  The durable fix is more real RPI negatives (section 9).

The sections below describe v2. Where v3 changes a parameter, the table above takes precedence.

**v2 (2026-09-25)** replaces v1, which is archived in `archive/v1_2026-09-25/`. v1 was written from manifests only, with no
waveforms and no real recordings, so it treated level, signal-over-background, clipping and device as **unknown**
and deferred every transform that needed them. v2 measures those properties from the audio itself
(`measure_deployment_levels.py`), using the 83 real recordings, including the deployment device (RPI).
Every number cited below is re-derived from those measurements by `check_augmentation_config.py`.

## What changed from v1

| Item | v1 | v2 | Why |
|---|---|---|---|
| Gain | Reject for now | **Enable** (`device_level_gain`) | RPI speech is 29 dB quieter than the synthetic positives |
| Noise mixing | SNR 5–25 dB, p = 0.35, synthetic noise | **Target 5.4–25.7 dB peak-over-floor, never < 4.5 dB, p = 0.6, 50 % real RPI noise** | Ranges now measured on the deployment device |
| Digital silence | Not considered | **New:** `device_noise_floor` | Exact-zero audio exists only in synthetic data |
| Microphone simulation | Reject for now | **Defer**, now measurable | Three devices exist, but device is confounded with session |
| Clipping | Reject | Reject, **confirmed by measurement** | No real recording clips |
| Sample rate | "48 kHz for all rows" | **Resample everything to 16 kHz first** | 1,249 rows are 16/22.05/24 kHz |
| Evaluation | Not specified | **Report real-device results separately** | Synthetic positives dominate validation/test |

## 1. Dataset evidence

- `manifest.csv`: **7,044** rows. v1 used 6,961; the difference is the **83 manual recordings**.
- Real recordings after review: **67 positive, 16 negative**, from 3 devices (Macmic, Phonemic, **RPI = deployment device**).
  Manual-Fil-RPI13 was relabelled to `negatives_confusable` by its recorder, because it contains only "Delta".
  Manual-Fil-RPI40 was relabelled to `negatives_confusable` on 2026-09-28, because it is not a complete "Hey Delta".
- Synthetic material: 50 TTS positives (7 voices, pitch/speed grid), 6,911 synthetic negatives.
- Header sample rates: **16, 22.05, 24 and 48 kHz**. The manifest lists 48 kHz for all rows, which is wrong for 1,249 of them.
- Speaker metadata is absent for the manual recordings, so the number of real speakers is undocumented.

## 2. Measured deployment conditions

All metrics use a 150–4000 Hz band-pass and 20 ms frames, in dB re full scale:
- **peak** = 98th-percentile frame level;
- **floor** = 20th-percentile frame level;
- **peak-over-floor** = peak − floor.

Peak-over-floor is a robust stand-in for signal-over-background. It can be computed even where the wakeword was not
located, which makes every group comparable. It is **not** a conventional RMS SNR, and the config uses the same
definition so targets and measurements stay comparable.
Source: `deployment_level_measurements.csv` (883 files) and `deployment_level_summary.json`.

| Group | n | Peak (median) | Floor (median) | Peak-over-floor (median, p10–p90) |
|---|---:|---:|---:|---|
| **RPI positives (deployment device)** | 62 | **−39.8 dB** (p10 −49.7, p90 −27.5) | −54.2 dB | **14.8 dB** (5.4–25.7) |
| — located by ASR | 22 | −32.7 | −54.7 | 22.0 |
| — not locatable by ASR, bounds marked by the owner by ear | 32 | −46.1 | −54.6 | 9.1 |
| — owner-confirmed, whole-clip span only (clips ≤ 1.4 s) | 8 | −36.2 | −51.1 | 17.7 |
| Macmic / Phonemic positives | 2 / 3 | −22.4 / −22.6 | −54.8 / −56.9 | 32.4 / 33.1 |
| **Synthetic TTS positives** | 50 | **−10.9 dB** | −47.4 (17 of 50 contain digital silence) | 39.8 |
| Synthetic confusables | 150 | −11.2 | −45.4 | 35.2 |
| Synthetic general speech / media | 150 / 150 | −13.7 / −13.6 | −35.8 / −32.5 | 22.2 / 18.1 |
| Synthetic partial-wakeword negatives | 150 | −14.7 | **digital silence in 150 of 150** | — |

What the measurements show:
1. **Level gap.** The wakeword reaches the deployment device about **29 dB quieter** than the synthetic positives it is trained on.
2. **Signal over background.** On the RPI, the wakeword peaks only 5–26 dB above the room background. The quietest clip
   the recording owner listened to and confirmed as a complete "Hey Delta" (Manual-Fil-RPI20) sits at **4.42 dB**.
3. **Digital silence.** Exact-zero audio occurs in all sampled partial-wakeword negatives, a third of TTS positives, a fifth of
   confusables, and every zero-padded window, but in **none** of the 62 real RPI positives. A model can learn
   "digital silence around speech means negative", a cue that never appears on the device.
4. **No clipping** in any of the 83 real recordings. It appears only in synthetic files, at most 0.05 % of samples.
5. **Two background recordings contain only zeros** (Manual-BG-RPI1, validation; Manual-BG-RPI2, train): the device captured nothing.
   At the recording owner's request they are kept and relabelled `negatives_silence`, with splits unchanged.
6. **Device start click** louder than 10 dB above the median occurs in 8 of 62 RPI positives and 2 of 15 RPI negatives. It is present
   in both classes and is a minor artefact, noted rather than augmented.

**Is the RPI level representative?** These measurements assume the recorded capture level is how the device will run in deployment.
If the deployed RPI will use a different microphone gain or distance, fix that setting first and re-measure. Augmentation
should not be used to compensate for a misconfigured device.

## 3. Acoustic coverage matrix

| Dimension | Status | Evidence |
|---|---|---|
| Speaker diversity | underrepresented | 7 TTS voices plus 68 real positive recordings; number of real speakers undocumented |
| Pitch / speaking rate | partially represented | TTS grids plus natural variation in the real recordings |
| **Level / loudness** | **measured: device level underrepresented** | RPI peak median −39.8 dB vs TTS −10.9 dB |
| **Signal over background** | **measured: underrepresented** | RPI 5.4–25.7 dB; synthetic 18–40 dB or digital silence |
| Microphone / device | partially represented | 3 devices; device response not yet isolable |
| Room acoustics / reverberation | missing | No room or RIR metadata |
| Background noise | partially represented | 1,000 synthetic clips plus 7 usable train-split real recordings (278 windows) |
| **Recording quality / channel** | **measured** | Mixed sample rates; digital silence only in synthetic data; no real clipping |
| Near/far field | unknown | No distance metadata |
| Hard negatives | synthetic: well represented; real: sparse | 1,199 confusables and 800 partials; real: RPI13 "Delta", RPI40, one "Hey, Michelle." |
| Real acoustic recordings | partially represented | 83 recordings; 32 RPI positives still lack wakeword timing |

Full wording: `acoustic_coverage_matrix.csv`.

## 4. Augmentation decisions

| Transform | Decision | Parameters |
|---|---|---|
| Sample-rate harmonisation | **Required preprocessing** | 16 kHz mono, before any transform |
| Gain (`device_level_gain`) | **ENABLE** | p = 0.5; scale so peak ~ U(−49.7, −27.5) dB |
| Background-noise mixing | **ENABLE** (re-parameterised) | p = 0.6; peak-over-floor ~ U(5.4, 25.7) dB, hard minimum 4.5 dB; 50 % real RPI noise |
| Device noise floor | **ENABLE** (new) | p = 1.0 when ≥ 20 ms of exact zeros; real RPI noise at floor ~ U(−58.2, −51.4) dB |
| RIR / reverberation | ENABLE (curated bank) | p = 0.2; RT60 0.15–0.60 s; DRR 6–18 dB |
| Microphone simulation | **DEFER** | Simultaneous RPI + reference-mic recording needed |
| Pitch shift / time stretch | REJECT | Unchanged from v1 |
| Clipping / distortion | REJECT (measured) | No real clipping |

Full rationale, constraints and split rules: `augmentation_decision_table.csv`. Machine-readable form: `augmentation_config.json`.

## 5. Enabled strategy

Composition order: **resample → device_level_gain → RIR → background_noise_mixing → device_noise_floor**.
Gain comes first, so every later level is set relative to the final speech level.

### A. Device-level gain — ENABLED (new in v2)
- **Gap:** device-level speech is almost absent from training (29 dB level gap).
- **Policy (v3):** p = 0.9. Scale the example so its peak equals a target drawn uniformly from −49.7 to −18.8 dB (RPI positives: p10 to maximum). In v2 this was p = 0.5 with a range of −49.7 to −27.5 dB.
- **Level must never predict the label.** The same probability and range apply to **every category, positives and negatives alike**, including
  silence/noise negatives. Otherwise the model learns "quiet means wakeword" and fires on quiet background.
- Never scale above full scale.

### B. Background-noise mixing — ENABLED (re-parameterised)
- **Gap:** the RPI hears the wakeword 5–26 dB above the room background; synthetic speech is 18–40 dB above its floor or over digital silence.
- **Policy (v3):** p = 0.9, and only when the example is cleaner than the target. Scale the noise so the mixture's peak-over-floor equals a target drawn uniformly from 5.4 to 25.7 dB. In v2 this was p = 0.6, unconditional. The noise level is corrected in closed loop (up to 4 steps) and the 4.5 dB hard minimum is enforced on the achieved mixture. Measured on 1,118 training mixes: stationary noise lands within ±1 dB; non-stationary noise (clicks, real background) within about 3 dB below the target, or cleaner when its own peaks dominate. None fell below 5.6 dB.
- **Hard minimum 4.5 dB**, just above the quietest owner-verified audible positive. Below that, an augmented
  "positive" may no longer contain an audible wakeword, and nothing quieter has been verified by listening.
- **Noise sources:** 50 % real device background from `real_noise_bank.csv`, 50 % the synthetic bank (white, pink, hum, fan, clicks).
- **Real noise bank:** 278 windows from train-split background recordings. They are vetted by the segmentation
  pipeline: no wakeword-like speech, with a 0.5 s guard. Validation/test recordings are excluded. The two all-zero recordings are not
  background audio and are never used as noise.
- Not applied to silence/noise negatives. One noise source per example. No lowest-level draws combined with extreme reverberation.

### C. Device noise floor — ENABLED (new in v2)
- **Gap:** digital silence exists only in synthetic data and zero-padded windows.
- **Policy:** whenever an example contains ≥ 20 ms of exact zeros (window padding included), add real RPI background over the **whole** example.
  The level is drawn from −58.2 to −51.4 dB, the p10–p90 of the measured RPI floor. The rule is identical for every label.
- Adding noise over the whole clip, not just the zeros, avoids creating splice boundaries.

### D. RIR / reverberation — ENABLED (bank supplied 2026-09-29: 132 MIT IR Survey responses)
- p = 0.2; RT60 0.15–0.60 s; DRR 6–18 dB; one RIR per example.
- **New constraint:** do not reverberate real RPI recordings; they already contain the deployment room.

## 6. Deferred and rejected transforms

- **Microphone / frequency response — DEFER.** A preliminary speech-spectrum comparison (octave bands, in
  `deployment_level_summary.json`) shows the RPI with relatively more 125 Hz and less 1 kHz energy than the Macmic.
  But there are only 2 Macmic and 3 Phonemic recordings, made in different sessions, so speaker, distance and room are confounded with the device.
  **Next measurement:** record the same utterances simultaneously on the RPI and a reference microphone at the deployment
  distance. The ratio of their spectra is the device response.
- **Pitch shift / time stretch — REJECT** (unchanged). The gap is real speakers, which these transforms cannot create.
- **Clipping / distortion — REJECT, now measured:** 0 of 83 real recordings clip.

## 7. Hard-negative policy

Hard negatives still take priority over easy-negative expansion. They receive the **same** gain, noise and noise-floor
transforms as positives, which matters most for the 800 partial-wakeword negatives: every sampled one contains digital silence.
Real device hard negatives are the scarcest data: Manual-Fil-RPI13 ("Delta"), Manual-Fil-RPI40 (not a complete "Hey Delta") and one "Hey, Michelle." so far.

## 8. Data-quality actions (before training)

1. **Manual-BG-RPI1 and Manual-BG-RPI2 contain only zeros.** Resolved: relabelled `negatives_silence` and kept, with splits unchanged.
   In training, `device_noise_floor` applies to Manual-BG-RPI2 (it is all digital silence), turning it into realistic device
   silence. Manual-BG-RPI1 is in validation, which is never augmented, so it stays pure zeros there. It is an easy negative that
   the device will never produce, and it should not be read as evidence of robustness to device silence.
2. **32 low-signal RPI positives (27 in train) still need wakeword timing.** They are the main source of real device-level
   training positives, which is exactly the condition this strategy targets.
3. Correct `manifest.csv` sample rates for the 1,249 affected rows.
4. Confirm the RPI capture configuration is the deployment configuration (see §2).

## 9. Required new recordings

1. Real wakeword recordings from additional speakers, with speaker metadata.
2. More RPI recordings at the deployment distance and gain, including far-field positions.
3. Simultaneous RPI + reference-microphone recordings (for the deferred device-response measurement).
4. Multiple rooms/environments with documented conditions.
5. More real RPI background environments; only 6 usable train-split RPI recordings exist.
6. Real RPI hard negatives: near-miss phrases and partial wakewords.

## 10. Split and inheritance rules

- The split unit is `source_group`. The 83 manual recordings take their split from `segmentation_windowing/outputs/split_extension.csv`,
  and recording relabels never change splits.
- **Only training source groups produce augmented audio.** Validation, test and `streaming_eval_holdout` are never augmented
  and never used as augmentation inputs.
- The real noise bank is built from train-split recordings only.
- External noise/RIR assets may be shared across splits only if they contain no source-group speech.

## 11. Evaluation reporting

Validation and test each hold 27 synthetic positive windows but only 8 from real recordings, so a pooled score mostly
reflects synthetic speech. Report metrics **separately** for: all data; real-device (RPI) data; synthetic data. Also list the
owner-verified low-signal positives individually: RPI19 (validation), RPI20 and RPI21 (test).

## 12. Acceptance checks

`check_augmentation_config.py` → `augmentation_checks.md`. **21 of 21 checks pass.** They verify that:
- every measured value cited in the config matches the measurements;
- every parameter range lies inside the measured RPI range;
- the noise hard minimum is not below the quietest owner-verified audible positive;
- gain and noise floor are label-independent;
- the noise bank is train-only, vetted, and has no all-zero files;
- validation, test and streaming evaluation are never augmented;
- the preprocessing sample rate matches the window export;
- the clipping rejection is supported by the measurements.

v1 criteria that still hold: every enabled transform maps to a documented gap; nothing is justified by sample count alone;
the existing pitch/rate diversity is preserved; hard negatives take priority; unknowns are not presented as measured.

## 13. Reproducing

```bash
cd data/deliverables/deployment_driven_augmentation_strategy
python measure_deployment_levels.py     # needs ../segmentation_windowing/outputs
python check_augmentation_config.py
python build_docx.py                    # regenerates augmentation_strategy.docx (needs python-docx)
```

## 14. Limitations

- The measurements come from ~60 short RPI clips, likely few speakers and one room, so they are a first estimate
  of deployment conditions, not a population estimate.
- Peak-over-floor is not a conventional RMS SNR; augmentation code must use the same definition as the measurements.
- 8 of the 62 RPI positives have no located wakeword (whole-clip span only), so their peak may include non-wakeword sound.
  The 32 spans marked by ear are the quietest RPI positives (median peak-over-floor 9.1 dB vs 22.0 dB for ASR-located ones).
- Room acoustics, distance and the device frequency response remain unmeasured.
