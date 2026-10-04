#!/usr/bin/env python3
"""
Model-only deployment kit for the Raspberry Pi 5 (owner, 2026-10-04): "Hey Delta" wakeword -> command model ->
the recognised command printed on screen. No actions, no voice replies (for those: the assistant kit,
data/deliverables/assistant/make_assistant_kit.py).

    python3 data/deliverables/model/deploy/make_model_kit.py      # -> data/deliverables/model/deploy/model_kit.zip

Works in a fresh clone of the public repository: it only uses the committed models. Reference clips for
`command_pi.py check` (audio, not in the repository) are added only when present.
"""
from pathlib import Path
import json, shutil

HERE = Path(__file__).resolve().parent                       # data/deliverables/model/deploy
EXPORT = HERE.parents[1] / "command_classifier/model/export"
RUNS = ["bcresnet6_hf_plus", "bcresnet6_hf_only"]
README = """# "Hey Delta" model kit: wakeword + command classifier on the Raspberry Pi 5

Say "Hey Delta", wait for the rising chime, say a command: the Pi prints what it recognised. Nothing is carried out
(for lights, music, calls, spoken replies...: the assistant kit, `data/deliverables/assistant/` in the repository).

## Install and run: one line (Pi 5, Raspberry Pi OS 64-bit)

```bash
unzip -o model_kit.zip && bash model_kit/install.sh
```

`install.sh` installs the PortAudio library (asks for your password once if missing) and a Python environment in
`~/heydelta` (numpy, onnxruntime, sounddevice, scipy), checks that every model loads and that there is a microphone,
then starts listening. Run `bash model_kit/install.sh` again to start it later (nothing is installed twice);
`--no-run` only installs and checks, `hf_only` picks the other model, further options go to `command_pi.py`
(e.g. `bash model_kit/install.sh hf_plus --rule balanced`).

By hand instead: `sudo apt install libportaudio2`, `python3 -m venv ~/heydelta && source ~/heydelta/bin/activate`,
`pip install numpy onnxruntime sounddevice scipy`, then the commands below.

## Run (with `source ~/heydelta/bin/activate` first)

```bash
python3 models/bcresnet6_hf_plus/command_pi.py live --deploy listener     # Ctrl+C to quit
python3 models/bcresnet6_hf_only/command_pi.py live --deploy listener     # the other model
python3 models/bcresnet6_hf_plus/command_pi.py live --deploy listener --rule balanced   # argmax | cautious | balanced
python3 models/bcresnet6_hf_plus/command_pi.py wav my_command.wav         # classify 16 kHz mono WAV files
```

Each command prints e.g. `-> TIMER_30S  (0.93, 2.1 s capture, 58 ms)`: the label (`unknown` = not a command or not
sure), its probability, the capture length and the inference time. The microphone is the system default; the
listener's settings (threshold 0.7335, 2 of 3 windows, chimes, end-of-speech rules) are in
`listener/deploy_config.json` and `listener/README_DEPLOY.md`.

| Folder | What |
|---|---|
| `listener/` | wakeword model `heydelta_bcresnet6.onnx` (BC-ResNet-6), `deploy_config.json`, `heydelta_listener.py`, `README_DEPLOY.md` |
| `models/bcresnet6_hf_plus/` | command model trained on the class Hugging Face data + our data (default rule: cautious) |
| `models/bcresnet6_hf_only/` | the same model trained on the class data only (for comparison) |

Speed of every model on the Pi: run `python3 data/deliverables/evaluation/pi_latency.py` from a clone of the
repository (results in `docs/COMPUTE.md`).
"""


def main():
    out = HERE / "model_kit"
    if out.exists():
        shutil.rmtree(out)
    (out / "listener").mkdir(parents=True)
    dcfg = json.loads((HERE / "deploy_config.json").read_text())
    for f in ("heydelta_listener.py", "deploy_config.json", dcfg["model_file"], "README_DEPLOY.md"):
        shutil.copy(HERE / f, out / "listener" / f)
    for run in RUNS:
        ex, d = EXPORT / run, out / "models" / run
        d.mkdir(parents=True)
        cfg = json.loads((ex / "command_config.json").read_text())
        for f in (cfg["model_file"], "command_config.json", "command_pi.py", "reference_clips.npz"):
            if (ex / f).exists():
                shutil.copy(ex / f, d / f)
        print(f"{run}: cutoffs {cfg['cutoffs']}, default rule {cfg['default_rule']}")
    (out / "README_MODEL_KIT.md").write_text(README)
    shutil.copy(HERE / "install_model_kit.sh", out / "install.sh")
    z = shutil.make_archive(str(HERE / "model_kit"), "zip", root_dir=HERE, base_dir="model_kit")
    print(f"{z}: {Path(z).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
