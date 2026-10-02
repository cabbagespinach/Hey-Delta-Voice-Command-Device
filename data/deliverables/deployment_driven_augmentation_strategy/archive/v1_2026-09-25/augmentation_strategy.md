# Deployment-Driven Augmentation Strategy — Hey Delta

## Scope

This is a **strategy-only** deliverable. No augmented audio is generated.

The analysis is based on the supplied `manifestfile.csv` and `dataset_split_deliverables.zip`. The supplied split package contains manifests/reports/configuration, not the underlying WAV corpus. Therefore acoustic properties that require waveform inspection—loudness, SNR, clipping, codec artifacts, actual reverberation, microphone frequency response, and source-to-microphone distance—are explicitly treated as **unknown**, not inferred.

## 1. Dataset evidence

- Total rows: **6,961**
- Positives: **50**
- Negatives: **6,911**
- Recording IDs: **2,650**
- Non-null speaker IDs: **74**
- Positive speaker IDs: **7**
- Positive recording IDs: **7**
- Parent/child rows: **800**
- Sample rate: **48000 Hz for all rows**
- All identified sources are TTS/synthetic or positive-derived; no real microphone-recorded source is identified.

### Category inventory

|                      |   rows |
|:---------------------|-------:|
| negatives_general    |   2679 |
| negatives_media      |   1233 |
| negatives_confusable |   1199 |
| negatives_silence    |   1000 |
| negatives_partial    |    800 |
| positives            |     50 |

### Important positive-data limitation

The 50 positives are generated from only 7 speaker/voice IDs and explicitly vary TTS speed ([0.85, 1.0, 1.15]) and pitch ([-2.0, 0.0, 2.0]). This is useful controlled variation, but it is not equivalent to natural speaker, microphone, room, distance, or level diversity.

## 2. Acoustic coverage matrix

| dimension                         | status                | evidence                                                                                                                                                                                     | interpretation                                                                                                                     |
|:----------------------------------|:----------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------|
| Speaker diversity                 | underrepresented      | 7 atomic/source voices contribute the 50 positives; all material is synthetic/TTS or derived. The split report identifies 9 atomic voice IDs overall and speaker overlap across every split. | Real human speaker diversity is not present; TTS voice count is not equivalent to speaker diversity.                               |
| Pitch                             | partially represented | Positive TTS metadata explicitly covers pitch offsets [-2.0, 0.0, 2.0] semitone-like units, but only for synthetic voices and only 50 positive clips.                                        | Coverage is a designed TTS grid, not natural pitch variation.                                                                      |
| Speaking rate                     | partially represented | Positive TTS metadata explicitly covers speed factors [0.85, 1.0, 1.15]; confusable negatives also have measured speed values.                                                               | Natural rate variability is absent because the dataset is synthetic/derived.                                                       |
| Level / loudness                  | unknown               | No level, RMS, LUFS, peak, SNR, or gain metadata exists in the manifest.                                                                                                                     | Cannot establish measured level coverage without waveform analysis.                                                                |
| Microphone / device               | missing               | No microphone, device, codec, channel, or frequency-response metadata exists; all clips are 48 kHz.                                                                                          | Device diversity cannot be established from the manifest.                                                                          |
| Room acoustics                    | missing               | No room, enclosure, acoustic treatment, or room-ID metadata exists; no RIR source is represented in the manifest.                                                                            | No evidence of real-room acoustic diversity.                                                                                       |
| Reverberation                     | missing               | No reverberation/RIR metadata or RIR source is represented.                                                                                                                                  | RIR augmentation can cover some synthetic room variation, but cannot replace real recordings.                                      |
| Background noise                  | partially represented | There are 1000 standalone synthetic noise clips across white, pink, hum, fan, and click sources, but no manifest category represents speech+noise mixtures.                                  | Noise type diversity exists, but deployment SNR/mixture diversity is absent.                                                       |
| Environmental sounds              | partially represented | Fan and clicks are represented; white/pink noise and hum are also present.                                                                                                                   | Real household/office/vehicle/outdoor soundscapes are not represented.                                                             |
| Recording quality / channel chain | unknown               | The manifest has no clipping, codec, bit-depth, channel-count, SNR, or quality metrics; all files are reported at 48 kHz.                                                                    | Waveform inspection is required before enabling quality-specific transforms.                                                       |
| Near/far field                    | unknown               | No distance, source-to-microphone distance, SPL, geometry, or near/far label exists.                                                                                                         | Cannot infer distance from duration, speaker, or filename.                                                                         |
| Hard negatives                    | well represented      | 1199 phonetic/lexical confusables plus 800 positive-derived partial-wakeword negatives are present.                                                                                          | These are the most deployment-relevant negative sources currently represented, but they are still predominantly synthetic/derived. |
| Real acoustic recordings          | missing               | The manifest sources are TTS, synthetic noise, or positive-derived truncations; no real microphone-recorded source is identified.                                                            | This is the principal limitation for deployment realism.                                                                           |

## 3. Augmentation decision table

| transform                                | decision                       | measured_coverage_rationale                                                                                                                                                                                                           | parameters_or_next_measurement                                                                                                                                                                     | applicable_categories                                                                                      | constraints                                                                                                                                                                            | split_rule                                                                                         |
|:-----------------------------------------|:-------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|
| gain/amplitude                           | REJECT FOR NOW                 | Level coverage is unknown: no RMS/LUFS/peak/SPL metadata is present. Enabling gain now would be a generic count/robustness transform rather than a measured deployment correction.                                                    | Revisit after waveform-level loudness audit and deployment level measurements.                                                                                                                     |                                                                                                            |                                                                                                                                                                                        |                                                                                                    |
| background-noise mixing                  | ENABLE                         | The dataset contains 1,000 standalone synthetic noise clips (white/pink/hum/fan/clicks) but no speech+noise mixture category. This is a direct, observable coverage gap.                                                              | SNR 5–25 dB, sampled continuously; default 12–20 dB; probability 0.35.                                                                                                                             | positive_wakeword; negative_general_speech; negative_media; negative_confusable; negative_partial_wakeword | Do not mix noise-only negatives; do not create SNRs below 5 dB; preserve wakeword audibility; do not stack multiple synthetic noise sources in one transform.                          | Enabled only for training split source groups.                                                     |
| RIR/reverberation                        | ENABLE (with curated RIR bank) | No RIR/room representation exists in the manifest. Room/reverberation is therefore a documented missing condition that can be approximated with measured/curated RIRs.                                                                | p=0.20; use RIRs spanning approximately 0.15–0.60 s RT60, with direct-to-reverberant ratio constrained to approximately 6–18 dB. Prefer short/small-room RIRs until deployment measurements exist. | positive_wakeword; negative_general_speech; negative_media; negative_confusable; negative_partial_wakeword | Use one RIR per example; do not combine extreme RT60 with extreme low SNR; reject RIRs whose direct path is absent unless deployment specifically requires diffuse/rear-field capture. | Enabled only for training split source groups; RIR assets must be external/curated and documented. |
| mild pitch shift                         | REJECT                         | Positive TTS already explicitly covers -2, 0, +2 pitch settings and negative confusables span roughly -1.49 to +1.50. The measured metadata therefore does not show a missing pitch-factor dimension that requires another transform. | None                                                                                                                                                                                               | None                                                                                                       | Do not use augmentation merely to expand the synthetic grid.                                                                                                                           | Keep existing source diversity; obtain real speakers instead.                                      |
| mild time stretch                        | REJECT                         | Positive TTS already covers speed factors 0.85, 1.0, and 1.15; confusable negatives have measured speed variation. The remaining gap is natural speaking-rate diversity, which time stretching cannot authentically create.           | None                                                                                                                                                                                               | None                                                                                                       | Do not use time stretch as a substitute for real-speaker recordings.                                                                                                                   | Keep existing source diversity; obtain real speakers instead.                                      |
| microphone/frequency-response simulation | REJECT FOR NOW                 | Microphone/device identity and frequency response are completely absent, so there is no measured target response to parameterize safely.                                                                                              | None                                                                                                                                                                                               | None                                                                                                       | Do not invent device EQ curves or codec chains.                                                                                                                                        | Requires deployment-device measurements or real multi-device recordings first.                     |
| clipping/distortion                      | REJECT                         | No clipping/distortion metrics or deployment evidence are present.                                                                                                                                                                    | None                                                                                                                                                                                               | None                                                                                                       | Do not introduce clipping as a generic robustness transform.                                                                                                                           | Only enable after measured deployment captures demonstrate this failure mode.                      |

## 4. Enabled augmentation strategy

### A. Background-noise mixing — ENABLED

**Measured gap:** the dataset contains 1,000 standalone synthetic noise clips across white noise, pink noise, hum, fan, and clicks, but no category/source represents actual speech-plus-noise mixtures. This is a concrete source-coverage gap.

**Training policy**
- Probability: **0.35**
- SNR: **5–25 dB**, with most draws concentrated in **12–20 dB**
- One noise source per example
- Applicable to wakeword and speech/hard-negative categories, not noise-only negatives
- Do not create extreme SNR conditions merely for sample-count expansion

The transform should be used to convert the existing standalone-noise inventory into realistic mixture conditions, not to manufacture more clean positives.

### B. RIR/reverberation — ENABLED, conditional on a curated RIR bank

**Measured gap:** no room/RIR representation exists in the manifest.

**Training policy**
- Probability: **0.20**
- RT60 target range: approximately **0.15–0.60 s**
- Direct-to-reverberant ratio: approximately **6–18 dB**
- One RIR per example
- Prefer measured/curated RIRs that are plausibly relevant to deployment
- Avoid combining extreme reverberation with very low SNR

This is an approximation of missing room acoustics; it does **not** substitute for real recordings.

## 5. Rejected/deferred transforms

- **Gain/amplitude:** deferred. Level is not measured in the manifest and the underlying WAVs were not supplied in the split package, so a numeric gain range would currently be arbitrary.
- **Pitch shift:** rejected. Positive TTS already explicitly covers -2, 0, and +2 pitch settings; more synthetic pitch perturbation is not a measured deployment gap.
- **Time stretch:** rejected. Positive TTS already covers 0.85, 1.0, and 1.15 speed factors; stretching cannot create real-speaker rate/prosody diversity.
- **Microphone/frequency-response simulation:** deferred. No device identity or target frequency-response measurements exist.
- **Clipping/distortion:** rejected. No deployment evidence or waveform metrics establish clipping/distortion as a relevant failure mode.

## 6. Hard-negative policy

Hard-negative source coverage should be prioritized over repeatedly transforming easy negatives.

Existing hard-negative material:
- **1,199** `negative_confusable` examples.
- **800** `negative_partial` examples derived from positive recordings.
- Confusable transcriptions include lexical/phonetic near-misses such as variants around “Hey Delta”.

Augmentation should therefore be allocated first to these categories and to real deployment-condition recordings of the same hard-negative classes. Large-scale transformation of silence/noise-only negatives is not a substitute for hard-negative source diversity.

## 7. Required new recordings

Augmentation cannot safely manufacture the following missing factors:

1. **Real human speaker diversity**, including natural accents, prosody, articulation, hesitations, and pronunciation variability.
2. **Real microphone/device diversity**, especially the actual deployment capture chain.
3. **Near-field and far-field distance variation**, with documented source-to-microphone distance.
4. **Real rooms and environments**, including room geometry and acoustic treatment differences.
5. **Measured level/SPL variation** across realistic distances and operating conditions.
6. **Real background environments and speech+noise mixtures.**
7. **Real hard negatives** captured under the same devices/rooms/distances as positives.
8. **Natural recordings of the complete wakeword** from multiple speakers; these are the highest-value replacement for synthetic-only positive diversity.

A useful collection design is to cross speakers, devices, rooms, and distances rather than recording many repeated clips from one synthetic voice. Exact quotas should be determined from the intended deployment population and hardware.

## 8. Split and inheritance rules

The supplied split uses `source_group` as the leakage-control unit:
- If `parent_filepath` exists, the parent's `recording_id` defines `source_group`.
- Otherwise the row's `recording_id` defines `source_group`.
- Source groups do not cross train/validation/test.
- Parent/child rows inherit the same split.
- The supplied report shows **0 source-group overlap**, **0 recording-ID overlap**, and **0 parent/child split mismatches**.
- Speaker-level disjointness is not achieved because the nine atomic voice IDs form one connected component across composite speaker IDs.

**Augmentation inheritance**
- Generate augmented audio **only from training source groups**.
- Do not augment validation or test audio.
- Do not use validation/test source audio as augmentation inputs.
- Independent external noise/RIR assets may be shared, provided they contain no source-group-specific speech content.
- Validation/test remain clean evaluation sets unless a separate, explicitly defined robustness benchmark is created.

## 9. Acceptance checks

| Check | Result |
|---|---|
| Every enabled transform maps to a documented gap | **PASS** |
| Every enabled transform has explicit probability and parameters | **PASS** |
| No transform is justified solely by sample-count expansion | **PASS** |
| Existing pitch/rate diversity is preserved rather than unnecessarily transformed | **PASS** |
| Hard negatives receive priority over easy-negative expansion | **PASS** |
| Train/validation/test inheritance is explicit | **PASS** |
| Waveform-dependent unknowns are not falsely labeled as measured | **PASS** |

## 10. Machine-readable configuration

See `augmentation_config.json`.

The configuration intentionally keeps gain and microphone simulation out of the enabled set until the required measurements exist. This prevents a plausible-looking but unsupported augmentation policy from being mistaken for deployment-derived calibration.
