# Isolated validation set v3 (archived 2026-09-29)

1,313 validation windows (64 positive). Archived deliberately, not because of an error: the owner
marked Hey Delta bounds by ear for Manual-Fil-Phonemic2 (validation) and Manual-Fil-Phonemic1 (test),
whose ASR spans were too long for a window. Phonemic2 adds 1 positive and 2 partial-wakeword
negatives, the first phone audio in validation. The train split and normalization statistics are
unchanged, so the baseline checkpoint was not retrained. v4 is frozen on the next evaluation run.
