# Canonical taxonomy

## Positive
- `positive_wakeword`: a complete **Hey Delta** occurrence.
  - Operational rule for this manifest: the existing positive label is retained only for rows whose normalized transcription is exactly `hey delta`.
  - All 50 existing positive rows satisfy this rule.
  - No negative row with a transcription normalizes to exactly `hey delta`.
  - Missing transcription is never treated as positive.

## Negative
- `negative_general_speech`: continuous/general speech that is not a complete wakeword.
- `negative_media`: media/background-content negatives.
- `negative_confusable`: phonetic/lexical near-misses or confusable phrases.
- `negative_partial_wakeword`: derived/truncated descendants of positive clips that do not contain the complete wakeword.
- `negative_silence_noise`: silence/noise-only negatives (white, pink, hum, fan, clicks).

## Provenance
- `synthetic_tts`: TTS-generated material.
- `synthetic_noise`: generated noise material.
- `derived_from_positive`: samples explicitly linked to a positive parent through `parent_filepath`.

## Metadata normalization
- `source_group` is the leakage-control unit.
- If `parent_filepath` exists, its parent's `recording_id` becomes the `source_group`.
- Otherwise `source_group = recording_id`.
- `recording_id` is preserved from the input; it is not fabricated.
- `session_id = NA` because no session field exists.
- `speaker_id` is preserved/whitespace-normalized. Composite IDs separated by `|` are preserved.
- No speaker or recording identity is inferred from filenames beyond the explicit parent relationship.

## Split policy
- Split occurs only at `source_group` level.
- All descendants of a source recording inherit the parent's split.
- Target ratios: train 80%, validation 10%, test 10%.
- The two largest positive source groups are assigned to test and validation; remaining positive groups go to train.
- Remaining source groups are assigned deterministically using seed `20260925` and a category-aware greedy objective.

## Speaker-separation limitation
The manifest contains 74 distinct `speaker_id` strings, but only 9 atomic voice IDs after expanding `|`-separated composite values. The composite IDs connect all 9 atomic voices into one connected component. Consequently, strict speaker-disjoint train/validation/test splitting is incompatible with retaining all source recordings. Speaker overlap is therefore reported rather than fabricated away.
