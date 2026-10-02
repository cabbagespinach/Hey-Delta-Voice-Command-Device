# Isolated validation set v5 (archived 2026-09-29)

3,848 validation windows (65 positive; the 27 synthetic positives were all one voice, en_US-amy-medium).
Archived deliberately, not because of an error: generate_tts_voices_v2.py added 625 Whisper-verified
"Hey Delta" clips and 809 near-miss clips from 405 new Piper speakers (each speaker one source group),
because BC-ResNet round-1 models memorised the 5 training TTS voices (synthetic validation detection fell to
10-40% while real RPI reached 100%). v6 is frozen on the next evaluation run. Round-1 BC-ResNet results
(model/runs/bcresnet3, bcresnet6) were computed on v5.
