# Held-out test result: BC-ResNet-6 (one-time run)

Generated 2026-09-30T01:58:22+00:00. Threshold 0.7335, chosen on validation (not on test). 4162 test windows, 234 positive.

ROC AUC **0.9912**, average precision **0.9518**.

| Positives | n | Detected | Rate (95% CI) |
|---|---:|---:|---|
| real_device_rpi | 33 | 33 | 100.0% [89.4%, 100.0%] |
| real_other_device | 2 | 2 | 100.0% [15.8%, 100.0%] |
| synthetic | 199 | 171 | 85.9% [80.3%, 90.4%] |

| Negatives by category | n | False positives | FPR (95% CI) |
|---|---:|---:|---|
| negative_confusable | 1281 | 2 | 0.16% [0.02, 0.56]% |
| negative_general_speech | 1071 | 0 | 0.00% [0.00, 0.34]% |
| negative_media | 266 | 0 | 0.00% [0.00, 1.38]% |
| negative_non_speech_from_positive_recording | 6 | 0 | 0.00% [0.00, 45.93]% |
| negative_partial_wakeword | 223 | 0 | 0.00% [0.00, 1.64]% |
| negative_real_background | 495 | 0 | 0.00% [0.00, 0.74]% |
| negative_silence_noise | 586 | 0 | 0.00% [0.00, 0.63]% |

| Negatives by source | n | False positives | FPR |
|---|---:|---:|---:|
| real_device_rpi | 523 | 0 | 0.00% |
| real_external | 2537 | 0 | 0.00% |
| real_other_device | 2 | 0 | 0.00% |
| synthetic | 866 | 2 | 0.23% |
