#!/usr/bin/env python3
"""
Recordings by other speakers (owner upload 2026-10-01): model/deploy/recordings_otherspeakers/<folder>/ -> command clips.

Folders: Classmate1-recordings (prompt-coded names, e.g. ALARM_V1_1_t1.wav, no manifest), Classmate2-recordings
(prompt-coded + manifest.csv of another project's prompts), Classmate3 (.mp4, no labels), Classmate4 (.wav, no labels).

    python prepare_other_speakers.py transcribe   # 16 kHz mono copies + Whisper large-v3 transcript (ONE GPU)
    python prepare_other_speakers.py trim         # "Hey Delta" + command clips: cut the wake word off (ONE GPU)
    python prepare_other_speakers.py label        # transcript -> our class (or unknown / review) -> manifest.csv

Output: data/commands_other_speakers/<speaker>/<original stem>.wav, transcripts.csv, manifest.csv
(file, label, class, speaker, source_file, prompt_id, prompt_text, heard, matched_text, edit_ratio, how, keep).
A clip is labelled with one of our command classes only when Whisper heard one of our wordings of it
(commands.txt, generate_phrasing_variants TRAIN / HELD_OUT); a clear request for something the device cannot do
(colours, a call without a name, ...) is `unknown`; anything else is `review` and left out until listened to.
"""
from pathlib import Path
import argparse, re, subprocess, sys

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from generate_synthetic_commands import norm, edit_ratio        # noqa: E402

SRC = HERE.parent / "model/deploy/recordings_otherspeakers"
OUT = ROOT / "data/commands_other_speakers"
SPEAKER = {"Classmate1-recordings": "classmate1", "Classmate2-recordings": "classmate2", "Classmate3": "classmate3",
           "Classmate4": "classmate4"}
SR = 16000


def sources():
    for folder, spk in SPEAKER.items():
        for f in sorted((SRC / folder).iterdir()):
            if f.suffix.lower() in (".wav", ".mp4", ".m4a", ".mp3"):
                yield spk, f


def transcribe(a):
    from faster_whisper import WhisperModel
    wm = WhisperModel("large-v3", device="cuda", device_index=0, compute_type="float16", cpu_threads=2)
    rows = []
    for spk, f in sources():
        dst = OUT / spk / (f.stem.replace(" ", "_").replace("(", "").replace(")", "") + ".wav")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-y", "-i", str(f), "-ac", "1", "-ar", str(SR),
                            "-sample_fmt", "s16", str(dst)], check=True)
        y, sr = sf.read(str(dst), dtype="float32")
        seg, _ = wm.transcribe(y, language="en", beam_size=5, word_timestamps=True, vad_filter=False)
        words = [w for s in seg for w in (s.words or [])]
        rows.append(dict(file=str(dst.relative_to(ROOT)), speaker=spk, source_file=str(f.relative_to(ROOT)),
                         duration_sec=round(len(y) / SR, 2), peak_db=round(20 * np.log10(np.abs(y).max() + 1e-9), 1),
                         heard="".join(w.word for w in words).strip(),
                         speech_start=round(words[0].start, 2) if words else None,
                         speech_end=round(words[-1].end, 2) if words else None))
        print(f"{spk:11s} {f.name:48s} {rows[-1]['heard']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "transcripts.csv", index=False)
    print(f"{len(rows)} clips -> {OUT / 'transcripts.csv'}")


def trim(a):
    """Clips that start with "Hey Delta" + a command: cut the wake word off (the device's command window starts after
    it). Cut halfway between the end of "Delta" and the next word; <stem>_nowake.wav replaces the clip in
    transcripts.csv (original kept on disk)."""
    from faster_whisper import WhisperModel
    t = pd.read_csv(OUT / "transcripts.csv")
    todo = [i for i, h in t.heard.items() if isinstance(h, str) and re.match(r"^hey delta \S", norm(h))
            and not t.file[i].endswith("_nowake.wav")]
    if not todo:
        print("nothing to trim"); return
    wm = WhisperModel("large-v3", device="cuda", device_index=0, compute_type="float16", cpu_threads=2)
    for i in todo:
        y, _ = sf.read(str(ROOT / t.file[i]), dtype="float32")
        seg, _ = wm.transcribe(y, language="en", beam_size=5, word_timestamps=True, vad_filter=False)
        words = [w for s in seg for w in (s.words or [])]
        k = next(j for j, w in enumerate(words) if re.search(r"delta", w.word, re.I))
        cut = (words[k].end + words[k + 1].start) / 2
        z = y[int(cut * SR):]
        dst = ROOT / t.file[i].replace(".wav", "_nowake.wav")
        sf.write(str(dst), z, SR, subtype="PCM_16")
        seg, _ = wm.transcribe(z, language="en", beam_size=5, word_timestamps=True, vad_filter=False)
        w2 = [w for s in seg for w in (s.words or [])]
        heard = "".join(w.word for w in w2).strip()
        print(f"{t.file[i]}: cut at {cut:.2f} s ('{words[k].word.strip()}' ends {words[k].end:.2f}, "
              f"'{words[k + 1].word.strip()}' starts {words[k + 1].start:.2f}) -> {dst.name}: {heard}")
        t.loc[i, ["file", "duration_sec", "heard", "speech_start", "speech_end"]] = [
            str(dst.relative_to(ROOT)), round(len(z) / SR, 2), heard,
            round(w2[0].start, 2) if w2 else None, round(w2[-1].end, 2) if w2 else None]
    t.to_csv(OUT / "transcripts.csv", index=False)


def our_wordings():
    import generate_phrasing_variants as gpv
    lab_cls = dict(pd.read_csv(HERE.parent / "model/deploy/label_map.csv")[["label", "class"]].values)
    w = []
    for line in (HERE.parent / "model/deploy/commands.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            lab, _, txt = (x.strip() for x in line.partition("|"))
            if not txt.startswith("("):
                w.append((lab, txt, "trained"))
    w += [(lab, t, "variant") for lab, ts in gpv.TRAIN.items() for t in ts]
    w += [(lab, t, "held_out_wording") for lab, t in gpv.HELD_OUT.items()]
    w += [("unknown_other", t, "near_miss") for t in gpv.NEAR_MISS]
    return [(lab, lab_cls[lab], t, kind) for lab, t, kind in w]


# requests the device has no class for (said by these speakers) -> unknown; matched on the normalised transcript
UNKNOWN_PATTERNS = [r"\b(colou?r|red|blue|green)\b", r"^(make a (phone )?)?call$"]


NUMBER_WORDS = set("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
                   "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred".split())


def numbers(t):
    return [w for w in t.split() if w in NUMBER_WORDS] + re.findall(r"\b([ap]) m\b", t)


# Classmate1/2 read another project's prompts (prompt id in the file name; texts in Classmate2's manifest, Classmate1's
# read off the transcripts): the PROMPT decides the label. Whisper agrees on every command prompt; its slips
# ("Paul", "Make a hole", "Cholera", "quality", "Alarm ATM") are all on prompts that are `unknown` anyway.
PROMPT_LABEL = [
    (r"ALARM_V\d_1", "set_alarm_6am", "prompt: alarm at 6 AM"),
    (r"ALARM_V\d_2", "unknown_other", "prompt: alarm at 8 AM/PM (no such alarm on the device)"),
    (r"ALARM_V\d_3", "set_alarm_9pm", "prompt: alarm at 9 PM"),
    (r"BRIGHTNESS_V\d_1", "dim_lights_20", "prompt: brightness 20% (= dim lights to 20%)"),
    (r"BRIGHTNESS_V\d_[23]", "unknown_other", "prompt: brightness 60% / 100% (no such level on the device)"),
    (r"CALL_V\d", "unknown_other", "prompt: call without a name (like the near-miss 'Call the doctor')"),
    (r"COLOR_V\d_\d", "unknown_other", "prompt: light colour (no colour command)"),
    (r"NEXT_V\d", "next", "prompt: skip / next song"),
    (r"PAUSE_V\d", "pause", "prompt: pause"),
]
# Classmate3 and Classmate4 read OUR prompt list; decisions on the clips the automatic match could not settle
# (levels measured: the Classmate4 "Thank you" clips are 4-6 dB above the floor = silence, a Whisper hallucination)
MANUAL = {
    "classmate3/7f4b2400-73b4-4922-a6e3-567c745fd1cd_1": ("drop", "byte-identical duplicate upload"),
    "classmate3/1d091d14-e7eb-4e98-b542-496f5842ec67": ("unknown_other", "'Nevermind' (not a command)"),
    "classmate3/aa637010-d351-46ac-b10e-d777bb3cd372": ("unknown_other", "'Hello there' (not a command)"),
    "classmate3/04d3b88b-9c5b-468c-ae62-3587c904f746": ("volume_up", "owner listened: 'Volume up' (Whisper 'We follow you up')"),
    "classmate3/5785427b-f141-43d3-8935-16e87d982513": ("pause", "owner listened: 'Pause' (Whisper 'Boss')"),
    "classmate3/1879902e-a3ef-453e-8589-8e7335631235": ("set_alarm_9pm", "owner listened: alarm 9 PM (Whisper 'Set an alarm at 19')"),
    "classmate4/recording_2026-10-01_13-09-22": ("drop", "owner listened: wake word only (wakeword model is done)"),
    "classmate4/recording_2026-10-01_13-09-32": ("unknown_other", "'Okay, there we go' (not a command)"),
    "classmate4/recording_2026-10-01_15-58-18": ("drop", "owner listened: wake word only (wakeword model is done)"),
    "classmate4/recording_2026-10-01_15-59-12": ("unknown_silent", "silence (4 dB above floor; Whisper invented 'Thank you')"),
    "classmate4/recording_2026-10-01_16-00-14": ("unknown_silent", "silence (5 dB above floor; Whisper invented 'Thank you')"),
    "classmate4/recording_2026-10-01_15-59-24": ("drop", "0.06 s long"),
    "classmate4/recording_2026-10-01_15-59-17": ("unknown_silent", "owner listened: nothing said (0.5 s)"),
    "classmate4/recording_2026-10-01_15-59-21": ("unknown_silent", "owner listened: nothing said (0.6 s)"),
    "classmate4/recording_2026-10-01_15-59-29": ("louder", "owner listened: 'Louder' (Whisper 'Mooder')"),
    "classmate4/recording_2026-10-01_16-00-38": ("pause", "owner listened: 'Pause' (Whisper 'Buzz off')"),
    "classmate4/recording_2026-10-01_16-00-44": ("list_reminders", "owner listened: 'What are my reminders' (wake word cut off)"),
    "classmate4/recording_2026-10-01_16-00-49": ("remind_trash", "'Remind me to take out the trash tomorrow'"),
    "classmate4/recording_2026-10-01_16-00-58": ("call_jane", "owner listened: 'Call Jane' (Whisper 'Call it a day')"),
    "classmate4/recording_2026-10-01_16-01-06": ("play_music", "owner listened: play music (Whisper 'Place on')"),
    "classmate4/recording_2026-10-01_16-01-07": ("play_music", "owner listened: play music (Whisper 'Play song')"),
    "classmate4/recording_2026-10-01_16-01-12": ("unknown_other", "'Never mind' (not a command)"),
}


def label(a):
    t = pd.read_csv(OUT / "transcripts.csv")
    prompts = pd.read_csv(SRC / "Classmate2-recordings/manifest.csv")
    ptext = dict(zip(prompts.prompt_id, prompts.text))
    W = our_wordings()
    lab_cls = dict(pd.read_csv(HERE.parent / "model/deploy/label_map.csv")[["label", "class"]].values)
    rows = []
    for r in t.itertuples(index=False):
        stem = Path(r.file).stem.removesuffix("_nowake")
        m = re.match(r"([A-Z_]+_V\d(?:_\d)?)_t\d+$", stem)
        pid = m.group(1) if m else ""
        heard = r.heard if isinstance(r.heard, str) else ""
        nh = re.sub(r"^(hey )?(delta|dealt|del) ", "", norm(heard)).strip()
        # a different number or a m / p m is a different command ("eight a m" is not "six a m"), whatever the edit ratio
        best = min(((edit_ratio(nh, norm(txt)) if numbers(nh) == numbers(norm(txt)) else 9.0, lab, cls, txt, kind)
                    for lab, cls, txt, kind in W), default=None)
        er, lab, cls, txt, kind = best
        if not nh:
            lab, cls, how = "unknown_silent", "unknown", "nothing heard"
        elif er <= 0.25:
            how = f"heard our wording ({kind})"
        elif any(re.search(p, nh) for p in UNKNOWN_PATTERNS):
            lab, cls, txt, how = "unknown_other", "unknown", "", "no such command on the device"
        else:
            lab, cls, how = "review", "review", f"closest: {txt} ({er:.2f})"
        rule = next(((l, h) for p, l, h in PROMPT_LABEL if pid and re.fullmatch(p, pid)), None)
        key = f"{r.speaker}/{stem}"
        if rule or key in MANUAL:
            lab, how = rule or MANUAL[key]
            cls = lab_cls.get(lab, lab)
            txt = ptext.get(pid, "") if rule else ""
            how = ("manual: " if key in MANUAL else "") + how
        # one of the never-trained wordings of evaluation/phrasing_variants_test.py: keep it out of training
        # (its closest wording of all is a held-out one)
        heldout = bool(nh) and best[4] == "held_out_wording" and best[0] <= 0.25
        rows.append(dict(file=r.file, label=lab, **{"class": cls}, speaker=r.speaker, source_file=r.source_file,
                         prompt_id=pid, prompt_text=ptext.get(pid, ""), heard=heard, matched_text=txt,
                         edit_ratio=round(er, 3), how=how, held_out_wording=heldout, keep=cls not in ("review", "drop"), duration_sec=r.duration_sec,
                         peak_db=r.peak_db))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "manifest.csv", index=False)
    print(d.groupby(["speaker", "class"]).size().unstack(fill_value=0).to_string())
    print(f"\nreview ({(~d.keep).sum()}):")
    print(d[~d.keep][["file", "prompt_id", "heard", "how"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["transcribe", "trim", "label"])
    a = ap.parse_args()
    {"transcribe": transcribe, "trim": trim, "label": label}[a.cmd](a)


if __name__ == "__main__":
    main()
