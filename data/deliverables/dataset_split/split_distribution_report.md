# Split distribution report

- Input rows: **6,961**
- Source groups: **2,600**
- Seed: **20260925**
- Target ratio: train 80%, validation 10%, test 10%

## Split totals

| Split | Rows | Fraction | Source groups | Positive |
|---|---:|---:|---:|---:|
| train | 5,416 | 77.80% | 2,099 | 32 |
| validation | 775 | 11.13% | 251 | 9 |
| test | 770 | 11.06% | 250 | 9 |

## Category counts

| Category | Train | Validation | Test |
|---|---:|---:|---:|
| negative_confusable | 959 | 120 | 120 |
| negative_general_speech | 2,132 | 276 | 271 |
| negative_media | 981 | 126 | 126 |
| negative_partial_wakeword | 512 | 144 | 144 |
| negative_silence_noise | 800 | 100 | 100 |
| positive_wakeword | 32 | 9 | 9 |

## Speaker overlap

Speaker IDs are treated as provided metadata. Composite `speaker_id` values are expanded only for this report.
- `train_vs_validation`: 9 overlapping atomic speaker/voice IDs
- `train_vs_test`: 9 overlapping atomic speaker/voice IDs
- `validation_vs_test`: 9 overlapping atomic speaker/voice IDs

The manifest's composite speaker IDs connect all 9 atomic voice IDs into one connected component. Therefore a fully speaker-disjoint split cannot be achieved without either discarding source material or breaking the source/recording leakage constraint. Speaker overlap is reported rather than hidden.

## Leakage checks

- `source_group` overlap across splits: **0**
- `recording_id` overlap across splits: **0**
- parent/child split mismatches: **0**
- source groups assigned to multiple splits: **0**