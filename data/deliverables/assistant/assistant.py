#!/usr/bin/env python3
"""
"Hey Delta" voice assistant for the Raspberry Pi 5: wake word -> command model -> action -> spoken reply.

    python3 assistant.py                      # listen on the default microphone (Ctrl+C to quit)
    python3 assistant.py --model hf_only --rule balanced --music ~/Music
    python3 assistant.py --type               # no microphone: type command names (TIMER_30S, LIGHT_ON, ...)
    python3 assistant.py --selftest           # no microphone/speaker: reference clips -> model -> actions
    python3 assistant.py --init-config        # create ~/.heydelta/config.json (private: location, Tapo, contact)
    python3 assistant.py --do LIGHT_ON        # carry out one command (no microphone, no model) and exit
    python3 assistant.py --list               # the command names

The home's state is shown live in a browser: http://<pi-ip>:8080/, printed at start with instructions (dashboard.py; --dashboard-port 0 = off).

Pieces: listener/heydelta_listener.py (wake word + capture), command_pi.py (command model, ONNX), actions.py (the
smart home), devices.py (Open-Meteo weather, Tapo bulb, phone calls/texts over Bluetooth), audio_out.py (one speaker output: music, replies, rings), tts.py (offline Piper voice).
State (lights, thermostat, volume, alarms, reminders) is saved in ~/.heydelta/state.json and survives restarts.
While the assistant talks, a wake-up is ignored (its own voice); music is turned down while it listens and talks.
"""
from pathlib import Path
import argparse, sys, threading, time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "listener"))
from actions import Home                                                     # noqa: E402
from audio_out import AudioOut                                               # noqa: E402
from tts import Speaker                                                      # noqa: E402
import devices                                                               # noqa: E402
import dashboard                                                             # noqa: E402

MODELS = {"hf_plus": "bcresnet6_hf_plus", "hf_only": "bcresnet6_hf_only"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="hf_plus")
    ap.add_argument("--rule", choices=["argmax", "cautious", "balanced"], help="default: the model's (balanced)")
    ap.add_argument("--music", default=str(Path.home() / "Music/Party Music"), help="folder with songs (.wav .flac .ogg .mp3)")
    ap.add_argument("--config", default=str(devices.CONFIG), help="private settings (location, Tapo, contact)")
    ap.add_argument("--init-config", action="store_true", help="create the private settings file and exit")
    ap.add_argument("--voice", default=str(HERE / "voices/en_US-amy-medium.onnx"))
    ap.add_argument("--state", default=str(Path.home() / ".heydelta/state.json"))
    ap.add_argument("--mic", help="input device (number or name)")
    ap.add_argument("--speaker", help="output device (number or name)")
    ap.add_argument("--no-voice", action="store_true", help="print replies instead of speaking them")
    ap.add_argument("--type", action="store_true", help="type command names instead of speaking")
    ap.add_argument("--selftest", action="store_true", help="classify the reference clips, run the actions, no audio")
    ap.add_argument("--do", nargs="+", metavar="COMMAND", help="carry out these commands (no microphone/model), then exit")
    ap.add_argument("--list", action="store_true", help="print the command names and exit")
    ap.add_argument("--dashboard-port", type=int, default=8080, help="browser dashboard port (0 = off)")
    a = ap.parse_args()
    if a.list:
        import csv
        names = [r["class"] for r in csv.DictReader(open(HERE / "label_map_schema_b.csv"))]
        print("\n".join(names + ["unknown"]))
        return
    if a.init_config:
        p = devices.init_config(Path(a.config).expanduser())
        print(f"settings file: {p} (only you can read it). Fill in location, tapo, contact, phone.bt_address.")
        return
    dev = lambda x: int(x) if x and x.isdigit() else x
    quiet = a.selftest
    audio = AudioOut(Path(a.music).expanduser(), device=dev(a.speaker), enabled=not quiet)
    speaker = Speaker(audio, Path(a.voice).expanduser(), enabled=not (quiet or a.no_voice))
    devs = devices.Devices({} if quiet else devices.load_config(Path(a.config).expanduser()))
    home = Home(audio, speaker.say, state_path=Path(a.state).expanduser() if not quiet else None,
                weather_path=HERE / "weather.json", devices=devs)
    print(devs.summary())
    print(home.status())
    srv = None
    if a.dashboard_port and not quiet:
        try:
            srv = dashboard.serve(home, a.dashboard_port)
            dashboard.announce(a.dashboard_port)        # number-style address + how to open it; again if it changes
        except OSError as e:
            print(f"(dashboard not started: {e})")

    def act(command, extra="", confidence=None):
        reply = home.handle(command, confidence)
        print(f"-> {command}{extra}")
        speaker.say(reply)
        print(home.status(), flush=True)

    try:
        if a.do:
            for c in a.do:
                act(c.upper())
            home.wait_background()                     # e.g. --do MESSAGE: let the text finish sending
            speaker.wait()
            home.save()
            import time as _t
            _t.sleep(0.5)
            if audio.is_playing():                     # e.g. --do PLAY_MUSIC: keep playing until Ctrl+C
                print(f"music playing: {audio.track}  (Ctrl+C to stop)", flush=True)
                while audio.is_playing() or audio.is_paused():
                    _t.sleep(0.5)
            return
        if a.type:
            print("type a command name (e.g. TIMER_10S, PLAY_MUSIC, LIST_REMINDERS), empty line to quit")
            for line in sys.stdin:
                if not line.strip():
                    break
                act(line.strip().upper())
            speaker.wait()
            return

        from command_pi import CommandClassifier
        clf = CommandClassifier(HERE / "models" / MODELS[a.model] / "command_config.json", rule=a.rule)
        print(f"command model {MODELS[a.model]}, rule {clf.rule} (cutoff {clf.cutoff})")

        if a.selftest:
            ref = HERE / "models" / MODELS[a.model] / "reference_clips.npz"
            if not ref.exists():
                print("no reference clips in this kit (audio is not in the public repository): selftest skipped")
                return
            r = np.load(ref)
            for x, want in zip(r["audio"], r["classes"]):
                label, p = clf(x)
                act(label, f"  ({p:.2f}; reference clip: {want})", p)
            return

        from heydelta_listener import HeyDeltaListener
        flags = {"ignore": False}

        def on_wake(p, t):
            if speaker.speaking:                       # its own reply said something like "Delta": not a wake-up
                flags["ignore"] = True
                return
            home.listening = True
            audio.set_duck(True)

        def restore_music():
            speaker.wait()
            audio.set_duck(False)

        def on_command(cap):
            if flags["ignore"]:                       # the wake-up came from its own voice
                flags["ignore"] = False
                return
            home.listening = False
            if cap.audio is None or cap.reason in ("no_speech", "no_speech_detected"):
                audio.set_duck(False)
                return
            t0 = time.perf_counter()
            label, p = clf(cap.audio)
            act(label, f"  ({p:.2f}, {cap.duration_sec:.1f} s, {(time.perf_counter() - t0) * 1000:.0f} ms)", p)
            threading.Thread(target=restore_music, daemon=True).start()

        HeyDeltaListener(on_command=on_command, on_wake=on_wake, device=dev(a.mic),
                         config_path=HERE / "listener/deploy_config.json").run()
    except KeyboardInterrupt:
        pass
    finally:
        if srv is not None:
            srv.shutdown()
        home.close()
        audio.close()


if __name__ == "__main__":
    main()
