#!/usr/bin/env python3
"""
Assemble the assistant kit for the Raspberry Pi (AI231ME2RedondoAssistantDeploy/ + AI231ME2RedondoAssistantDeploy.zip, 2026-10-02):
assistant code, command models (official cutoffs), wake-word listener, Piper voice, empty music folder.

    python data/deliverables/assistant/make_assistant_kit.py

Works in a fresh clone of the public repository: the Piper voice (not redistributed here) is downloaded once from
rhasspy/piper-voices (pinned v1.0.0) into voices/piper_raw/ when it is not there yet, and the 16 reference clips for
`assistant.py --selftest` (audio, so not in the repository) are added only if the ONNX export step made them.
"""
from pathlib import Path
import json, shutil, urllib.request

HERE = Path(__file__).resolve().parent
DELIV = HERE.parent
ROOT = DELIV.parents[1]
CMD = DELIV / "command_classifier/model"
DEPLOY = DELIV / "model/deploy"
VOICE = ROOT / "voices/piper_raw/en/en_US/amy/medium/en_US-amy-medium.onnx"
VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/"
RUNS = ["bcresnet6_hf_plus", "bcresnet6_hf_only"]


def fetch_voice():
    for f in (VOICE, VOICE.with_suffix(".onnx.json")):
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {f.name} from rhasspy/piper-voices (v1.0.0) ...")
            tmp = f.with_name(f.name + ".part")
            urllib.request.urlretrieve(VOICE_URL + f.name, tmp)
            tmp.rename(f)


def main():
    fetch_voice()
    out = HERE / "AI231ME2RedondoAssistantDeploy"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    for f in ("assistant.py", "actions.py", "devices.py", "map_send.py", "dashboard.py", "audio_out.py", "tts.py", "test_assistant.py", "start_assistant.sh", "install_autostart.sh", "install.sh",
              "README_ASSISTANT.md"):
        shutil.copy(HERE / f, out / f)
    shutil.copy(DEPLOY / "label_map_schema_b.csv", out / "label_map_schema_b.csv")
    shutil.copy(CMD / "export" / RUNS[0] / "command_pi.py", out / "command_pi.py")
    for run in RUNS:
        ex, d = CMD / "export" / run, out / "models" / run
        d.mkdir(parents=True)
        cfg = json.loads((ex / "command_config.json").read_text())
        for f in (cfg["model_file"], "command_config.json"):
            shutil.copy(ex / f, d / f)
        if (ex / "reference_clips.npz").exists():
            shutil.copy(ex / "reference_clips.npz", d / "reference_clips.npz")
        else:
            print(f"{run}: no reference_clips.npz (audio, not in the repository; made by the ONNX export step): "
                  "--selftest will be skipped")
        print(f"{run}: cutoffs {cfg['cutoffs']}, default {cfg['default_rule']}")
    (out / "listener").mkdir()
    dcfg = json.loads((DEPLOY / "deploy_config.json").read_text())
    for f in ("heydelta_listener.py", "deploy_config.json", dcfg["model_file"]):
        shutil.copy(DEPLOY / f, out / "listener" / f)
    (out / "voices").mkdir()
    for f in (VOICE, VOICE.with_suffix(".onnx.json")):
        shutil.copy(f, out / "voices" / f.name)
    (out / "music").mkdir()
    (out / "music/PUT_SONGS_HERE.txt").write_text("Put .wav .flac .ogg or .mp3 files in this folder.\n")
    z = shutil.make_archive(str(HERE / "AI231ME2RedondoAssistantDeploy"), "zip", root_dir=HERE, base_dir="AI231ME2RedondoAssistantDeploy")
    print(f"{z}: {Path(z).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
