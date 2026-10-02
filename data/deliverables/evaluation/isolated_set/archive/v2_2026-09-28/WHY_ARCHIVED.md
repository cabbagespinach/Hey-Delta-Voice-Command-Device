# Isolated validation set v2 (archived 2026-09-29)

1,013 validation windows (42 positive). Archived deliberately, not because of an error:
the 2026-09-29 recordings were integrated (Manual-BG-RPI20..36 except BG-RPI35, and
Manual-Fil-RPI65..82 except RPI72, both left out as duplicates). The append-only split
extension placed Manual-Fil-RPI74, 76, 78 and some BG recordings in validation, so the
validation split now has 1,313 windows (64 positive). The normalization statistics were
recomputed for the new training windows. v3 is frozen on the next evaluation run.
Metrics computed on v2 and v3 are not directly comparable. The v2-era checkpoint,
results and confound report are kept in ../../../archive_pre_2026-09-29_recordings/.
