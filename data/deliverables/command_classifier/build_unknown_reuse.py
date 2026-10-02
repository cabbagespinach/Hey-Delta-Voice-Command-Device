#!/usr/bin/env python3
"""
`unknown` clips for the command classifier, reused from the wakeword project's negatives (owner-approved 2026-09-30).

    CUDA_VISIBLE_DEVICES=<one gpu> python build_unknown_reuse.py

| label (folder)   | source                                                        | licence                    | clips                        |
|------------------|---------------------------------------------------------------|----------------------------|------------------------------|
| unknown_room     | owner's Manual-BG recordings (RPI / Mac / phone mic; ~1 h)     | own data                   | 2-5 s windows, <= 60 per file |
| unknown_noise    | MUSAN noise (data/negatives_silence/Ext-MUSAN-noise-*)        | CC-BY 4.0 / public domain  | one 1.5-5 s window per file  |
| unknown_music    | MUSAN music (data/negatives_media/Ext-MUSAN-music-*)          | CC-BY 4.0 / public domain  | two 1.5-5 s windows per file |
| unknown_speech   | FLEURS fil_ph + en_us, MUSAN speech (data/negatives_general)  | CC-BY 4.0 / public domain  | one 1.5-4 s window per file  |
| unknown_word     | MSWC near-miss single words (data/negatives_confusable)       | CC-BY 4.0                  | <= 15 per word               |

Rules:
- Any window that contains speech is transcribed with Whisper large-v3 (word timestamps). A window is dropped if a word
  in it (or within 0.3 s of it) is a command word or "hey"/"delta" (COMMAND_WORDS), so no command or wakeword ends up
  in `unknown`. Owner recordings are transcribed whole, then windowed around the words.
- Silent (all-zero) recordings are skipped.
- Clips are 16 kHz mono, as-is (no "Delta" tail), like data/commands_real_web; format alignment is left to training.
- `source_split` is the wakeword project's split of the source recording (recording_inventory.csv), so a recording
  stays in one split across both models.
Hey Snips is not reused (academic-only terms; SLURP already covers everyday requests). Qualcomm is not licensed for it.

Output: data/commands_unknown_reuse/<label>/<source file>__<n>.wav + manifest.csv
(file, label, class, source, speaker, source_split, transcript, license, origin_file, offset_sec).
"""
from pathlib import Path
import re, sys, wave

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "data/commands_unknown_reuse"
INV = HERE.parent / "segmentation_windowing/outputs/recording_inventory.csv"
sys.path.insert(0, str(HERE))
from generate_synthetic_commands import to16k                        # noqa: E402

SR, SEED = 16000, 20260930
COMMAND_WORDS = {"hey", "delta", "stop", "next", "skip", "pause", "play", "party", "louder", "volume", "light", "lights",
                 "timer", "alarm", "remind", "reminder", "reminders", "temperature", "dim", "weather", "call", "text",
                 "music", "jane", "time"}
LICENSE = {"owner": "own data", "musan": "CC-BY 4.0 / US public domain (MUSAN)", "fleurs": "CC-BY 4.0 (FLEURS)",
           "mswc": "CC-BY 4.0 (MSWC)"}
MSWC_PER_WORD = 15
ROOM_PER_FILE = 60


def write(path, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())


def load(rel):
    y, sr = sf.read(str(ROOT / rel), dtype="float32", always_2d=True)
    return to16k(y.mean(1), sr)


def clean(w):
    return re.sub(r"[^a-z]", "", w.lower())


def main():
    rng = np.random.default_rng(SEED)
    inv = pd.read_csv(INV, low_memory=False)
    inv = inv[inv.label == "negative"]
    split = dict(zip(inv.filepath, inv.split))
    name = lambda p: Path(p).name
    rows, jobs = [], []            # jobs: (label, source, rel, y, offset, needs_asr)

    # 1. owner's room recordings: all Manual-* negatives except the all-zero ones
    own = inv[(inv.source == "manual_recording") & ~inv.category.isin(["negatives_confusable"])]
    owner_audio = {}
    for rel in own.filepath:
        y = load(rel)
        if np.abs(y).max() == 0:
            continue
        owner_audio[rel] = y

    # 2. public sources: one or two random windows per file
    def windows(rel, label, src, n, lo, hi, asr):
        y = load(rel)
        for _ in range(n):
            L = int(rng.uniform(lo, hi) * SR)
            if len(y) <= L:
                jobs.append((label, src, rel, y, 0.0, asr))
            else:
                a = int(rng.integers(0, len(y) - L))
                jobs.append((label, src, rel, y[a:a + L], a / SR, asr))
    pub = inv[inv.source.astype(str).str.startswith("ext_")]
    for rel, s in zip(pub.filepath, pub.source):
        n = name(rel)
        if n.startswith("Ext-MUSAN-noise"):
            windows(rel, "unknown_noise", "musan", 1, 1.5, 5.0, False)
        elif n.startswith("Ext-MUSAN-music"):
            windows(rel, "unknown_music", "musan", 2, 1.5, 5.0, False)
        elif n.startswith("Ext-MUSAN-speech"):
            windows(rel, "unknown_speech", "musan", 1, 1.5, 4.0, True)
        elif n.startswith("Ext-FLEURS"):
            windows(rel, "unknown_speech", "fleurs", 1, 1.5, 4.0, True)
    mswc = pub[pub.filepath.map(name).str.startswith("Ext-MSWC") & (pub.category == "negatives_confusable")].copy()
    mswc["word"] = mswc.filepath.map(lambda p: name(p).split("-")[2])
    mswc = mswc[~mswc.word.isin(COMMAND_WORDS)]
    for _, g in mswc.groupby("word"):
        for rel in g.sample(min(MSWC_PER_WORD, len(g)), random_state=SEED).filepath:
            jobs.append(("unknown_word", "mswc", rel, load(rel), 0.0, False))
    print(f"owner recordings: {len(owner_audio)} ({sum(len(y) for y in owner_audio.values()) / SR / 60:.1f} min); "
          f"public windows: {len(jobs)}", flush=True)

    # 3. Whisper: owner recordings whole, public speech windows one by one
    from faster_whisper import WhisperModel
    wm = WhisperModel("large-v3", device="cuda", device_index=0, compute_type="float16")

    def words_of(y):
        seg, _ = wm.transcribe(y, language=None, beam_size=5, word_timestamps=True, vad_filter=True)
        return [(w.start, w.end, w.word.strip()) for s in seg for w in (s.words or [])]

    dropped = 0
    for label, src, rel, y, off, asr in jobs:
        ws = words_of(y) if asr else []
        if any(clean(w) in COMMAND_WORDS for _, _, w in ws):
            dropped += 1
            continue
        k = sum(r["origin_file"] == rel for r in rows)
        f = f"{label}/{Path(rel).stem}__{k}.wav"
        write(OUT / f, y)
        spk = Path(rel).stem.rsplit("_", 1)[-1] if src == "mswc" else Path(rel).stem
        rows.append(dict(file=f, label=label, **{"class": "unknown"}, source=src, speaker=f"{src}-{spk}",
                         source_split=split.get(rel, ""), transcript=" ".join(w for _, _, w in ws),
                         license=LICENSE[src], origin_file=rel, offset_sec=round(off, 3)))
    print(f"public: kept {len(rows)}, dropped {dropped} windows with command words", flush=True)

    for rel, y in owner_audio.items():
        ws = words_of(y)
        bad = [(s - 0.3, e + 0.3) for s, e, w in ws if clean(w) in COMMAND_WORDS]
        t, n, dur = 0.0, 0, len(y) / SR
        while n < ROOM_PER_FILE:
            L = rng.uniform(2.0, 5.0)
            if t + L > dur:
                if n == 0 and dur >= 1.0 and not bad:          # short file: keep it whole
                    L = dur
                else:
                    break
            if any(s < t + L and e > t for s, e in bad):      # overlaps a command word: skip past it
                t = min(e for s, e in bad if s < t + L and e > t) + 0.05
                continue
            f = f"unknown_room/{Path(rel).stem}__{n}.wav"
            write(OUT / f, y[int(t * SR):int((t + L) * SR)])
            rows.append(dict(file=f, label="unknown_room", **{"class": "unknown"}, source="owner", speaker="owner-room",
                             source_split=split.get(rel, ""),
                             transcript=" ".join(w for s, e, w in ws if s < t + L and e > t),
                             license=LICENSE["owner"], origin_file=rel, offset_sec=round(t, 3)))
            n += 1
            t += L
        print(f"  {Path(rel).name}: {n} windows ({len(bad)} command/wake words avoided)", flush=True)

    man = pd.DataFrame(rows)
    man.to_csv(OUT / "manifest.csv", index=False)
    print(f"manifest: {len(man)} clips -> {OUT / 'manifest.csv'}")
    print(man.groupby(["label", "source_split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
