# Dataset / DataLoader distribution report

Generated 2026-09-28T14:37:01+00:00 by `distribution_report.py` (seed 20260928, source_group_exponent 0.5, epoch size 6991). Machine-readable: `distribution_report.json`.

## 1. Natural distribution per split (no sampling)

| Split | Windows | Positive | Hard negative | Easy negative | Eval-only | Source groups | Positives: RPI / other real / synthetic |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 6,991 | 268 (3.8%) | 1547 (22.1%) | 5176 (74.0%) | 0 | 2,154 | 105 / 67 / 96 |
| validation | 1,013 | 42 (4.1%) | 272 (26.9%) | 699 (69.0%) | 0 | 265 | 15 / 0 / 27 |
| test | 991 | 41 (4.1%) | 272 (27.4%) | 678 (68.4%) | 0 | 260 | 14 / 0 / 27 |
| streaming_eval_holdout | 4,642 | 288 (6.2%) | 236 (5.1%) | 0 (0.0%) | 4118 | 120 | 0 / 0 / 288 |

Validation and test are loaded unsampled and unaugmented, so metrics see this natural mix.
Streaming-eval positives/negatives come from continuous synthetic streams (`negative_stream_background`).

## 2. Training: before vs after sampling

| | Before (natural) | After (quota) |
|---|---:|---:|
| Positive | 3.8% | 25.0% |
| Hard negative | 22.1% | 34.0% |
| Easy negative | 74.0% | 41.0% |
| Hard : easy negative ratio | 0.30 | 0.83 |
| Real-recording share of positives | 64.2% | 50.0% |
| Real-recording share of negatives | 19.4% | 33.9% |

The real-recording share per class shows how strongly *source* could predict the label (see `evaluation/diagnostics/source_confound_report.md`); ideally it is equal for both classes.

**Negative categories as a share of all negatives:**

| Bucket | Role | Before | After |
|---|---|---:|---:|
| device_background | easy_negative | 18.7% | 26.7% |
| general_speech | easy_negative | 31.7% | 16.0% |
| media | easy_negative | 14.6% | 6.7% |
| silence_noise | easy_negative | 11.9% | 5.3% |
| confusable | hard_negative | 14.6% | 26.7% |
| partial_wakeword | hard_negative | 8.4% | 18.7% |

**Per bucket:**

| Bucket | Windows | Before | After (quota) | After (empirical, 5 epochs) | Draws/epoch | Repeat factor | Windows seen/epoch | Source groups | Effective groups before → after | Largest group share before → after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| device_background | 1,260 | 18.0% | 20.0% | 20.0% | 1,398 | 1.11× | 58.2% | 12 | 3.0 → 6.2 | 49.9% → 28.3% (`Manual-BG-RPI19`) |
| general_speech | 2,132 | 30.5% | 12.0% | 12.0% | 839 | 0.39× | 32.6% | 189 | 179.9 → 186.5 | 0.8% → 0.6% (`syn_general_rec_0000`) |
| media | 981 | 14.0% | 5.0% | 5.0% | 349 | 0.36× | 30.1% | 146 | 141.7 → 144.9 | 0.9% → 0.8% (`syn_media_rec_0013`) |
| silence_noise | 803 | 11.5% | 4.0% | 4.0% | 280 | 0.35× | 29.3% | 801 | 797.0 → 800.5 | 0.4% → 0.2% (`Manual-BG-RPI2`) |
| confusable | 980 | 14.0% | 20.0% | 20.0% | 1,398 | 1.43× | 76.2% | 960 | 941.6 → 956.7 | 0.2% → 0.1% (`syn_conf_00194`) |
| partial_wakeword | 567 | 8.1% | 14.0% | 14.0% | 979 | 1.73× | 68.4% | 32 | 5.0 → 11.8 | 27.0% → 15.2% (`fil-PH-BlessicaNeural`) |
| positive_real | 172 | 2.5% | 12.5% | 12.5% | 874 | 5.08× | 94.3% | 42 | 13.8 → 32.5 | 16.9% → 7.2% (`Manual-Fil-Macmic1`) |
| positive_synthetic | 96 | 1.4% | 12.5% | 12.5% | 874 | 9.10× | 100.0% | 5 | 4.4 → 4.8 | 28.1% → 24.3% (`fil-PH-BlessicaNeural`) |

- **Repeat factor** = draws per epoch / windows in the bucket. Above 1, windows are drawn several times per epoch; each draw gets its own augmentation seed, so repeats are different augmented examples.
- **Effective groups** = 1 / Σ p²ᵍ over source groups, i.e. how many equally weighted recordings the bucket is worth. Square-root source-group weighting raises it by limiting long recordings.

## 3. Augmentation applied on 1,000 sampled training draws

| Bucket | Gain | Reverb | Noise mixing | Device noise floor | Any |
|---|---:|---:|---:|---:|---:|
| confusable | 89.5% | 0.0% | 82.8% | 2.4% | 98.6% |
| device_background | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| general_speech | 86.3% | 0.0% | 76.6% | 8.9% | 98.4% |
| media | 96.6% | 0.0% | 58.6% | 0.0% | 100.0% |
| partial_wakeword | 87.1% | 0.0% | 60.0% | 20.0% | 100.0% |
| positive_real | 86.4% | 0.0% | 41.5% | 0.0% | 98.3% |
| positive_synthetic | 88.2% | 0.0% | 85.0% | 17.3% | 100.0% |
| silence_noise | 90.5% | 0.0% | 0.0% | 0.0% | 90.5% |
| ALL | 71.8% | 0.0% | 55.2% | 6.9% | 80.3% |

Configured probabilities (augmentation config): gain 0.9, reverb 0.2 (disabled: no RIR bank), noise mixing 0.9 (only when the example is cleaner than the target; noisier real recordings are skipped), device noise floor 1.0 when the excerpt contains ≥ 20 ms of exact zeros. Each transform applies only to the categories listed in the augmentation config (e.g. no gain or mixing for real device background; no mixing for silence/noise clips), so rates differ by bucket.
