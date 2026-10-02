#!/usr/bin/env python3
"""
Real (non-synthetic) command recordings from public datasets, mapped onto the command classes (label_map.csv).

    python prepare_real_commands.py [--sources tas fsc slurp]

| source | dataset (licence)                                   | speakers | how it is mapped |
|--------|-----------------------------------------------------|----------|------------------|
| tas    | Timers and Such v1.0, real part (CC0)                | 95       | from its semantics: SetTimer 1/5/10 min -> set_timer_*; SetAlarm 6:00 AM / 7:00 AM / 9:00 PM -> set_alarm_*; every other timer, alarm, maths or unit-conversion request -> unknown (hard negatives for the numbers) |
| fsc    | Fluent Speech Commands (academic/non-commercial ONLY; no sharing of the audio or anything derived) | 97 | lights on/off without a location, volume up/down, play/pause/stop music -> their classes; heat/temperature, language, "bring ..." -> unknown; ambiguous ones (lamp, lights in a named room, "turn off the music", "resume") dropped |
| slurp  | SLURP real part (CC BY-NC 4.0)                        | many     | transcripts that normalise to one of the class phrasings -> that class; scenarios unrelated to any command (news, qa, cooking, transport, lists, email, social, recommendation, takeaway, general) -> unknown; everything else (paraphrases of supported commands) dropped |

Unknown is capped per source (--max-unknown) so no source floods that class. Audio is trimmed to the speech and saved
as-is (no "Delta" tail): format alignment with HeyDeltaListener captures is left to training.
Output: data/commands_real_web/<label>/<source>__<id>.wav (16 kHz mono 16-bit) + manifest.csv
(file, label, class, source, speaker, source_split, transcript, license).
"""
from pathlib import Path
import argparse, ast, json, re, sys, wave

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / "external_raw/commands"
OUT = ROOT / "data/commands_real_web"
sys.path.insert(0, str(HERE))
from generate_synthetic_commands import norm, to16k, trim            # noqa: E402

LM = pd.read_csv(HERE.parent / "model/deploy/label_map.csv")
CLASS = dict(zip(LM.label, LM["class"]))
SEED = 20260930
LICENSE = {"tas": "CC0 1.0", "fsc": "Fluent Speech Commands Public License (non-commercial/academic, no sharing)",
           "slurp": "CC BY-NC 4.0"}

# phrasings accepted for each label (normalised with norm())
PHRASES = {
    "play_music": ["play music", "play some music", "play the music", "start the music", "put on the music",
                   "put on some music", "start music"],
    "weather": ["what is the weather", "what is the weather like", "what is the weather today", "how is the weather"],
    "time": ["what time is it", "what is the time", "what is the time now", "what time is it now"],
    "lights_on": ["turn on lights", "turn on the lights", "turn the lights on", "switch on the lights",
                  "switch the lights on", "lights on"],
    "lights_off": ["turn off lights", "turn off the lights", "turn the lights off", "switch off the lights",
                   "switch the lights off", "lights off"],
    "play": ["play"],                                   # added 2026-09-30 (FSC "Play" = activate music)
    "party": ["party party", "party"],                  # added 2026-09-30 (not expected in these corpora)
    "pause": ["pause", "pause music", "pause the music"],
    "stop": ["stop", "stop music", "stop the music"],
    "next": ["next", "next song", "next track", "play the next song"],
    "skip": ["skip", "skip song", "skip this song", "skip the song"],
    "volume_up": ["volume up", "turn the volume up", "turn up the volume", "increase volume", "increase the volume",
                  "turn volume up"],
    "louder": ["louder", "louder please", "make it louder", "make the music louder"],
    "volume_down": ["volume down", "turn the volume down", "turn down the volume", "decrease volume",
                    "decrease the volume", "lower the volume", "turn volume down", "quieter", "make it quieter"],
    "list_reminders": ["what are my reminders", "what reminders do i have"],
    "set_alarm_6am": ["set an alarm at six a m", "set an alarm for six a m", "set alarm for six a m"],
    "set_alarm_7am": ["set an alarm at seven a m", "set an alarm for seven a m", "set alarm for seven a m"],
    "set_alarm_9pm": ["set an alarm at nine p m", "set an alarm for nine p m", "set alarm for nine p m"],
}
PHRASE2LABEL = {norm(p): lab for lab, ps in PHRASES.items() for p in ps}
SLURP_UNKNOWN_SCENARIOS = {"news", "qa", "cooking", "transport", "lists", "email", "social", "recommendation",
                           "takeaway", "general"}


def write(path, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())


def rows_tas():
    base = RAW / "timers_and_such"
    d = pd.concat([pd.read_csv(base / f"{s}-real.csv").assign(split=s) for s in ("train", "dev", "test")])
    out = []
    for r in d.itertuples(index=False):
        m = ast.literal_eval(r.semantics)
        i, sl = m["intent"], m["slots"]
        lab = "unknown"
        if i == "SetTimer" and sl.get("hours", 0) == 0 and sl.get("seconds", 0) == 0 and sl.get("minutes") in (1, 5, 10):
            lab = f"set_timer_{sl['minutes']}"
        elif i == "SetAlarm":
            lab = {("AM", 6, 0): "set_alarm_6am", ("AM", 7, 0): "set_alarm_7am",
                   ("PM", 9, 0): "set_alarm_9pm"}.get((sl["am_or_pm"], sl["alarm_hour"], sl["alarm_minute"]), "unknown")
        out.append(dict(src=base / r.path, label=lab, speaker=f"tas-{r.speakerId}", source_split=r.split,
                        transcript=r.transcription, id=Path(r.path).stem))
    return out


def rows_fsc():
    base = RAW / "fsc/fluent_speech_commands_dataset"
    d = pd.concat([pd.read_csv(base / f"data/{s}_data.csv").assign(split=s) for s in ("train", "valid", "test")])
    out = []
    for r in d.itertuples(index=False):
        t = norm(r.transcription)
        lab = PHRASE2LABEL.get(t)
        if lab is None:
            if r.object in ("heat", "language") or r.action == "bring" or r.action == "change language":
                lab = "unknown"
            else:
                continue                                   # lamp, located lights, "turn off the music", resume...
        out.append(dict(src=base / r.path, label=lab, speaker=f"fsc-{r.speakerId}", source_split=r.split,
                        transcript=r.transcription, id=Path(r.path).stem))
    return out


def rows_slurp():
    base = RAW / "slurp"
    audio = base / "slurp_real"
    if not audio.exists():
        print("  slurp: audio not extracted yet (slurp_real/), skipped")
        return []
    out = []
    for split in ("train", "devel", "test"):
        for line in open(base / f"{split}.jsonl"):
            r = json.loads(line)
            lab = PHRASE2LABEL.get(norm(r["sentence"]))
            if lab is None:
                if r["scenario"] not in SLURP_UNKNOWN_SCENARIOS:
                    continue
                lab = "unknown"
            for rec in r["recordings"]:
                f = audio / rec["file"]
                if not f.exists():
                    continue
                session = re.sub(r"-headset$", "", Path(rec["file"]).stem)   # headset + far-field = same session
                out.append(dict(src=f, label=lab, speaker=f"slurp-{session}", source_split=split,
                                transcript=r["sentence"], id=Path(rec["file"]).stem))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["tas", "fsc", "slurp"])
    ap.add_argument("--max-unknown", type=int, default=1500, help="cap on unknown clips per source")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    man_path = OUT / "manifest.csv"
    old = pd.read_csv(man_path) if man_path.exists() else pd.DataFrame()
    new = []
    for src in a.sources:
        rows = {"tas": rows_tas, "fsc": rows_fsc, "slurp": rows_slurp}[src]()
        if not rows:
            continue
        d = pd.DataFrame(rows)
        unk = d[d.label == "unknown"]
        if len(unk) > a.max_unknown:          # cap unknown, one clip per transcript first for variety
            unk = unk.sample(frac=1, random_state=SEED).drop_duplicates("transcript")
            unk = unk.head(a.max_unknown) if len(unk) >= a.max_unknown else \
                pd.concat([unk, d[(d.label == "unknown") & ~d.id.isin(unk.id)].sample(a.max_unknown - len(unk), random_state=SEED)])
        d = pd.concat([d[d.label != "unknown"], unk])
        for r in d.itertuples(index=False):
            y, sr = sf.read(str(r.src), dtype="float32")
            y = trim(to16k(y, sr))
            if len(y) < 1600:
                continue
            rel = f"{r.label}/{src}__{r.id}.wav"
            write(OUT / rel, y)
            new.append(dict(file=rel, label=r.label, **{"class": CLASS.get(r.label, "unknown")}, source=src,
                            speaker=r.speaker, source_split=r.source_split, transcript=r.transcript, license=LICENSE[src]))
        print(f"  {src}: {len(d)} clips; " + ", ".join(f"{k} {v}" for k, v in d.label.value_counts().items()))
    new = pd.DataFrame(new)
    if len(old):
        new = pd.concat([old[~old.source.isin(a.sources)], new], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    new.to_csv(man_path, index=False)
    print(f"manifest: {len(new)} clips -> {man_path}")
    print(new.groupby("class").size().to_string())


if __name__ == "__main__":
    main()
