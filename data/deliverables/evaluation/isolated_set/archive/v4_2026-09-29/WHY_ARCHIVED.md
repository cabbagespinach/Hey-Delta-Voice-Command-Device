# Isolated validation set v4 (archived 2026-09-29)

1,316 validation windows (65 positive). Archived deliberately, not because of an error: public-corpus
negatives were integrated (MSWC near-miss words, FLEURS, MUSAN, Sonos "Hey Snips"; sources ext_*,
reported as eval subset real_external), plus Manual-BG-Macmic2 and Manual-BG-Phonemic1 (train). The
validation split now has 3,848 windows (65 positive). The normalization statistics were recomputed and
the baseline retrained; the v4-era checkpoint, results and confound report are in
../../../archive_pre_2026-09-29_external/. v5 is frozen on the next evaluation run.
