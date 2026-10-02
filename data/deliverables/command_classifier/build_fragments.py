#!/usr/bin/env python3
"""
Cut-off commands for the `unknown` class (owner-approved 2026-09-30): a capture that ends before the word that tells
commands apart must not trigger an action.

    CUDA_VISIBLE_DEVICES=<one gpu> python build_fragments.py

For each label below, a clip is cut just before its DECIDING word (the number, name, target or direction); only the
ambiguous beginning is kept ("Set an alarm at", "Remind me to", "Dim lights to", "Call", "What's the", ...).
Commands whose beginning already identifies one command (play_music -> "Play" is itself a command; party; pause, stop,
next, skip, louder; list_reminders "What are my" is unique) get no fragments.

Sources (the split of each fragment is the split of its source; the combined list decides it):
- owner recordings Whisper confirmed as complete (both speakers; the combined list keeps val/test sources out of training)
- synthetic clips (ph / clone / piper tiers), up to --per-label-synthetic per label and tier
- public real clips (Timers and Such, SLURP) of the labels above
Word times come from Whisper large-v3 (word timestamps) on each source clip. The cut is 30 ms before the deciding
word starts. Half of the fragments then get the source clip's own last 0.6-1.0 s appended (quiet room / noise floor),
like a capture that ended on a pause; the other half end abruptly, like a capture cut at its time limit.
A fragment is kept only if at least one command word is left before the cut and it is >= 0.4 s long.

Output: data/commands_fragments/unknown_partial/<source stem>__frag.wav + manifest.csv
(file, label=unknown_partial, class=unknown, source_file, source_dataset, source_label, kept_text, cut_sec, tail).
"""
from pathlib import Path
import re, sys, wave

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "data/commands_fragments"
REC = HERE.parent / "model/deploy/recordings"
sys.path.insert(0, str(HERE))
from generate_synthetic_commands import to16k, num_words                  # noqa: E402

SR, SEED = 16000, 20260930
NUM = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "eighteen",
       "twenty", "thirty", "fifty", "eighty", "hundred", "percent", "degrees", "minute", "minutes"}
# label pattern -> words that decide the command (the cut goes before the first of them that follows a command word)
DECIDE = {
    r"set_alarm_": NUM, r"set_timer_": NUM, r"set_temperature_": NUM, r"dim_lights_": NUM,
    r"remind_": {"take", "study"}, r"call_jane": {"jane", "jean", "jay", "gene"},
    r"text_jane": {"jane", "jean", "jay", "gene"}, r"volume_": {"up", "down"}, r"lights_o": {"on", "off"},
    r"^time$": {"time"}, r"^weather$": {"weather"},
}
FILLER = {"hey", "delta", "dealt", "uh", "um", "ah", "oh", "and", "so", "okay"}


def words_norm(w):
    w = w.lower().strip()
    w = re.sub(r"(\d+)", lambda m: " " + num_words(m.group(1)) + " ", w)
    return [x for x in re.sub(r"[^a-z ]", " ", w).split() if x]


def write(path, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())


def rule(label):
    for pat, dec in DECIDE.items():
        if re.search(pat, label):
            return dec
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label-synthetic", type=int, default=15, help="per label and synthetic tier")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    src = []            # (path, dataset, label)
    chk = pd.read_csv(REC / "_check/check.csv")
    for r in chk[chk.verdict == "complete"].itertuples(index=False):
        if rule(r.label):
            src.append((REC / r.file, "owner_recordings", r.label))
    syn = pd.read_csv(ROOT / "data/commands_synthetic/manifest.csv")
    syn = syn[syn.label.map(lambda l: rule(l) is not None)]
    for (tier, lab), g in syn.groupby(["tier", "label"]):
        for f in g.sample(min(a.per_label_synthetic, len(g)), random_state=SEED).file:
            src.append((ROOT / "data/commands_synthetic" / f, f"synthetic_{tier}", lab))
    web = pd.read_csv(ROOT / "data/commands_real_web/manifest.csv")
    web = web[web.source.isin(["tas", "slurp"]) & web.label.map(lambda l: rule(l) is not None)]
    for r in web.itertuples(index=False):
        src.append((ROOT / "data/commands_real_web" / r.file, f"web_{r.source}", r.label))
    print(f"{len(src)} source clips", flush=True)

    from faster_whisper import WhisperModel
    wm = WhisperModel("large-v3", device="cuda", device_index=0, compute_type="float16")
    rows, skipped = [], 0
    for path, ds, lab in src:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        y = to16k(y.mean(1), sr)
        seg, _ = wm.transcribe(y, language="en", beam_size=5, word_timestamps=True)
        ws = [(w.start, w.end, t) for s in seg for w in (s.words or []) for t in words_norm(w.word)[:1]]
        dec, cut, kept = rule(lab), None, []
        for st, en, t in ws:
            if t in FILLER and not kept:
                continue
            if t in dec and kept:
                cut = st - 0.03
                break
            kept.append(t)
        if cut is None or cut < 0.4:
            skipped += 1
            continue
        frag = y[:int(cut * SR)]
        tail = rng.random() < 0.5
        if tail:
            L = int(rng.uniform(0.6, 1.0) * SR)
            frag = np.concatenate([frag, y[-L:]])
        name = f"unknown_partial/{Path(path).stem}__frag.wav"
        write(OUT / name, frag)
        rows.append(dict(file=name, label="unknown_partial", **{"class": "unknown"},
                         source_file=str(Path(path).relative_to(ROOT)), source_dataset=ds, source_label=lab,
                         kept_text=" ".join(kept), cut_sec=round(cut, 3), tail=tail))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "manifest.csv", index=False)
    print(f"wrote {len(d)} fragments ({skipped} sources skipped: deciding word not found or too early)")
    print(d.groupby(["source_dataset"]).size().to_string())
    print(d.kept_text.value_counts().head(25).to_string())


if __name__ == "__main__":
    main()
