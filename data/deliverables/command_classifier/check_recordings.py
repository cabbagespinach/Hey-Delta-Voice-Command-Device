#!/usr/bin/env python3
"""
Check the owner's command recordings (demo_record.py output) with Whisper large-v3 (ONE GPU).

For every clip: transcript with word timings, whether the prompted command is complete, cut off, or something else,
and the speech-band frame levels needed to calibrate HeyDeltaListener's end-of-speech thresholds.

    CUDA_VISIBLE_DEVICES=<gpu> python check_recordings.py <recordings dir>

Writes <recordings dir>/_check/check.csv (one row per clip) and levels.npz (per clip: frame levels, speech mask).
Verdicts:
  complete       all prompted words heard (normalised edit ratio <= 0.25)
  cut_off        the heard words are a strict beginning of the prompt (the end is missing), or speech runs to
                 within 0.15 s of the clip end
  empty          no words (expected only for unknown_silent)
  other          words heard but not the prompt (e.g. said something else); listen to it
"""
from pathlib import Path
import argparse, json, re, sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_synthetic_commands import norm, edit_ratio        # noqa: E402

SR, FRAME = 16000, 320
SOS = butter(4, [150, 4000], "bandpass", fs=SR, output="sos")      # speech band, as the project's level metric


def band_db(y):
    yb = sosfilt(SOS, y)
    k = len(yb) // FRAME
    return 20 * np.log10(np.sqrt((yb[:k * FRAME].reshape(k, FRAME) ** 2).mean(1)) + 1e-9)


def expected_texts(label, prompts):
    t = prompts.get(label, "")
    alts = {t}
    m = re.match(r"dim_lights_(\d+)", label)
    if m:
        alts.add(f"Dim lights to {m.group(1)} percent")
    return [a for a in alts if a and not a.startswith("(")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings")
    a = ap.parse_args()
    rec = Path(a.recordings)
    prompts = {}
    for line in (HERE.parent / "model/deploy/commands.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            lab, _, txt = (x.strip() for x in line.partition("|"))
            prompts[lab] = txt
    man = pd.read_csv(rec / "manifest.csv")
    man = man[[(rec / f).exists() for f in man.file]].reset_index(drop=True)

    from faster_whisper import WhisperModel
    wm = WhisperModel("large-v3", device="cuda", device_index=0, compute_type="float16")
    rows, levels = [], {}
    for r in man.itertuples(index=False):
        y, sr = sf.read(str(rec / r.file), dtype="float32")
        assert sr == SR, (r.file, sr)
        seg, _ = wm.transcribe(y, language="en", beam_size=5, word_timestamps=True, vad_filter=False)
        words = [w for s in seg for w in (s.words or [])]
        heard = "".join(w.word for w in words).strip()
        # the clip starts with the tail of "Delta": drop a leading "delta"/"hey delta" from the comparison
        heard_cmd = re.sub(r"^(hey\s+)?(delta|dealt|del)\b[\s,.!?]*", "", heard, flags=re.I)
        exp = expected_texts(r.label, prompts)
        er = min((edit_ratio(norm(heard_cmd), norm(e)) for e in exp), default=1.0)
        nh = norm(heard_cmd)
        prefix = any(nh and norm(e).startswith(nh) and len(nh) < len(norm(e)) for e in exp)
        cmd_words = [w for w in words if not re.fullmatch(r"\s*(hey|delta|dealt)[,.!?]*\s*", w.word, flags=re.I)]
        last_end = cmd_words[-1].end if cmd_words else None
        dur = len(y) / SR
        if not cmd_words:
            verdict = "empty"
        elif er <= 0.25 and not (last_end is not None and dur - last_end < 0.15 and r.end_reason != "end_of_speech"):
            verdict = "complete"
        elif prefix or (last_end is not None and dur - last_end < 0.15):
            verdict = "cut_off"
        else:
            verdict = "other"
        db = band_db(y)
        mask = np.zeros(len(db), bool)
        for w in cmd_words:
            mask[int(w.start * SR / FRAME):int(np.ceil(w.end * SR / FRAME))] = True
        levels[r.file] = np.stack([db, mask.astype(float)])
        gaps = [round(b.start - a_.end, 3) for a_, b in zip(cmd_words, cmd_words[1:])]
        rows.append(dict(file=r.file, label=r.label, end_reason=r.end_reason, duration_sec=round(dur, 2),
                         heard=heard, expected=" / ".join(exp), edit_ratio=round(er, 3), verdict=verdict,
                         cmd_start=cmd_words[0].start if cmd_words else None, cmd_end=last_end,
                         max_word_gap=max(gaps) if gaps else 0.0))
    out = rec / "_check"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "check.csv", index=False)
    np.savez_compressed(out / "levels.npz", **{k.replace("/", "__"): v for k, v in levels.items()})
    d = pd.DataFrame(rows)
    print(d.groupby(["verdict", "end_reason"]).size().to_string())


if __name__ == "__main__":
    main()
