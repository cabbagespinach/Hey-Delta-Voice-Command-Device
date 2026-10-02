#!/usr/bin/env python3
"""
Record a command dataset through the real wake-up path: say "Hey Delta", then the command; each captured command is
saved as a WAV (exactly the audio a command classifier would receive on the device).

    python demo_record.py --prompts commands.txt --speaker me --note "RPI, PulseAudio AEC, living room"
    python demo_record.py                          # no prompts: every capture is saved as "unlabeled"

commands.txt: one prompt per line, "label | words to say" (e.g. "set_timer_5 | Set a timer for 5 minutes"), or
just a label. Lines starting with # are comments. The demo shows what to say next, cycling through the list
(--order random to shuffle each round). The label becomes the folder name. Type then Enter:
    d   delete the last saved recording (e.g. you said the wrong thing)
    s   skip the current prompt
    q   quit

Output (in --out, default recordings/):
    <label>/<label>_<speaker>_<time>.wav        16 kHz mono 16-bit, the captured command (starts 0.3 s before the wake-up
                                               fired, ends after 1.0 s of quiet, or 6 s after speech started;
                                               if no speech is detected within 3 s it is still saved, with
                                               end_reason "no_speech_detected")
    _wake/<label>_<speaker>_<time>.wav         the 1.5 s window that woke the device (--save-wake; real "Hey Delta" data)
    _no_speech.csv                             wake-ups with nothing captured (only with --no-always-capture)
    manifest.csv                               one row per saved command: file, label, speaker, duration, how it ended,
                                               wakeword probability, time, note
"""
from pathlib import Path
import argparse, csv, datetime, random, sys, threading, wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heydelta_listener import HeyDeltaListener, Settings, SR      # noqa: E402

FIELDS = ["file", "label", "speaker", "duration_sec", "end_reason", "wake_prob", "recorded_at", "wake_file", "note"]


def write_wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", help="text file, one command label per line")
    ap.add_argument("--order", choices=["cycle", "random"], default="cycle")
    ap.add_argument("--speaker", default="owner", help="who is speaking (goes in file names and the manifest)")
    ap.add_argument("--out", default="recordings")
    ap.add_argument("--note", default="", help="saved in every manifest row (device, room, AEC, ...)")
    ap.add_argument("--device", type=int, default=None, help="microphone number (see: python -m sounddevice)")
    ap.add_argument("--save-wake", action="store_true", help="also save the 1.5 s 'Hey Delta' window")
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--max-command-sec", type=float, default=Settings.max_command_sec)
    ap.add_argument("--end-silence-sec", type=float, default=Settings.end_silence_sec,
                    help="pause length that ends a command (default 1.0 s)")
    ap.add_argument("--speech-margin-db", type=float, default=Settings.speech_margin_db,
                    help="dB above the room background needed to start a command")
    ap.add_argument("--continue-margin-db", type=float, default=Settings.continue_margin_db,
                    help="dB above the room background that still counts as speech once a command has started")
    ap.add_argument("--max-wait-sec", type=float, default=Settings.max_wait_sec,
                    help="how long to wait for speech after the wake-up before ending the capture")
    ap.add_argument("--no-chimes", action="store_true", help="don't play the wake-up / done chimes")
    ap.add_argument("--chime-device", default=None, help="output device for chimes (default: system default output)")
    ap.add_argument("--no-always-capture", action="store_true",
                    help="discard the capture when no speech is detected (old behaviour)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # prompt file lines: "label" or "label | words to say"; blank lines and lines starting with # are ignored
    say = {}
    labels = []
    for line in (Path(a.prompts).read_text().splitlines() if a.prompts else []):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lab, _, text = (x.strip() for x in line.partition("|"))
        labels.append(lab)
        say[lab] = text or lab
    labels = labels or ["unlabeled"]
    order = []
    state = dict(label=None, last=None)

    def next_label():
        if not order:
            order.extend(random.sample(labels, len(labels)) if a.order == "random" else labels)
        state["label"] = order.pop(0)
        if a.prompts:
            print(f"\n>>> say: \"Hey Delta, {say[state['label']]}\"      [{state['label']}]")

    man = out / "manifest.csv"
    new = not man.exists()
    mf = open(man, "a", newline="")
    mw = csv.DictWriter(mf, fieldnames=FIELDS)
    if new:
        mw.writeheader()

    def on_command(cap):
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        if cap.audio is None:
            with open(out / "_no_speech.csv", "a") as f:
                f.write(f"{stamp},{state['label']},{cap.wake_prob:.4f}\n")
            print("    (no command heard - say it again)")
            return
        name = f"{state['label']}_{a.speaker}_{stamp}.wav"
        path = out / state["label"] / name
        write_wav(path, cap.audio)
        wake_path = ""
        if a.save_wake:
            write_wav(out / "_wake" / name, cap.wake_window)
            wake_path = f"_wake/{name}"
        row = dict(file=f"{state['label']}/{name}", label=state["label"], speaker=a.speaker,
                   duration_sec=round(cap.duration_sec, 3), end_reason=cap.reason, wake_prob=round(cap.wake_prob, 4),
                   recorded_at=stamp, wake_file=wake_path, note=a.note)
        mw.writerow(row)
        mf.flush()
        state["last"] = (path, row, wake_path)
        flag = "  <- no speech detected; press d + Enter if you want to redo it" if cap.reason == "no_speech_detected" else ""
        print(f"    saved {row['file']} ({row['duration_sec']} s, {cap.reason}){flag}")
        next_label()

    listener = HeyDeltaListener(on_command=on_command, device=a.device,
                                settings=Settings(max_command_sec=a.max_command_sec, end_silence_sec=a.end_silence_sec,
                                                  max_wait_sec=a.max_wait_sec, always_capture=not a.no_always_capture,
                                                  speech_margin_db=a.speech_margin_db,
                                                  continue_margin_db=a.continue_margin_db,
                                                  chimes=not a.no_chimes,
                                                  chime_output_device=int(a.chime_device) if str(a.chime_device).isdigit()
                                                  else a.chime_device))

    def keys():
        for line in sys.stdin:
            k = line.strip().lower()
            if k == "q":
                listener.stop()
                return
            if k == "s":
                print("    skipped")
                next_label()
            elif k == "d" and state["last"]:
                path, row, wake_path = state["last"]
                path.unlink(missing_ok=True)
                if wake_path:
                    (out / wake_path).unlink(missing_ok=True)
                rows = [r for r in csv.DictReader(open(man)) if r["file"] != row["file"]]
                mf.seek(0), mf.truncate()
                mw.writeheader(), mw.writerows(rows), mf.flush()
                state["last"] = None
                print(f"    deleted {row['file']} - record it again")
                order.insert(0, state["label"])          # the prompt that was next comes after the redo
                state["label"] = row["label"]
                if a.prompts:
                    print(f"\n>>> say: \"Hey Delta, {say[state['label']]}\"      [{state['label']}]")

    threading.Thread(target=keys, daemon=True).start()
    if not a.prompts:
        print("No --prompts file given: every capture is saved as 'unlabeled' and no command is shown.\n"
              "For prompts, run: python demo_record.py --prompts commands.txt   (one command label per line)\n")
    next_label()
    listener.run(minutes=a.minutes)
    mf.close()
    counts = {}
    for r in csv.DictReader(open(man)):
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print("\nrecordings so far:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
