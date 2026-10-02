# Wake Word Dataset — Phase 1 Generation Summary

Generated: 2026-09-24T15:37:04.634912+00:00
Wake word: "Hey Delta"
Sample rate: 48000 Hz
Random seed: 42

## Clip counts by category
category
negatives_confusable    1199
negatives_general       2679
negatives_media         1233
negatives_partial        800
negatives_silence       1000
positives                 50

## Target counts
{
  "positives": 2000,
  "negatives_general": 1800,
  "negatives_confusable": 1200,
  "negatives_partial": 800,
  "negatives_media": 1000,
  "negatives_silence": 1000,
  "streaming_eval": 120,
  "rir": 300,
  "noise": 500
}

## Manifest
See `manifest.csv` in this directory (columns: filepath, label, category, source, source_id, speaker_id, recording_id, duration_sec, sample_rate, voice, speed, pitch, transcription, edit_distance, parent_filepath, created_at).

Full config snapshot: `logs/phase1_summary.json`.
This dataset feeds the downstream source-level split + augmentation pipeline. The separate `streaming_eval` directory/manifest is reserved for sliding-window false-accept evaluation and must not be used for training.
