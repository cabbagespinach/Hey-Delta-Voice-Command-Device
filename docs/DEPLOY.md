# Deploying on a Raspberry Pi 5

Two setups, both built from a clone of this repository with one command, using only the committed models (no
training, no GPU, no project audio needed). Copy the zip to the Pi, unzip, install, run.

| | 1. Model deployment | 2. Model + assistant deployment |
|---|---|---|
| What it does | "Hey Delta" → command → **prints** the recognised command | "Hey Delta" → command → **carries it out** (Tapo bulb, phone calls/texts, weather, music, timers, alarms, reminders, simulated thermostat), **spoken reply**, live dashboard in the laptop's browser |
| Build (any computer, Python 3) | `python3 data/deliverables/model/deploy/make_model_kit.py` | `python3 data/deliverables/assistant/make_assistant_kit.py` |
| Result | `data/deliverables/model/deploy/model_kit.zip` (~11 MB) | `data/deliverables/assistant/assistant_kit.zip` (~63 MB; downloads the Piper voice once) |
| On the Pi | `pip install numpy onnxruntime sounddevice scipy` | + `piper-tts python-kasa` and the device setup |
| Run | `python3 models/bcresnet6_hf_plus/command_pi.py live --deploy listener` | `python3 assistant.py` (or at every boot: `bash install_autostart.sh`) |
| Full instructions | `README_MODEL_KIT.md` inside the kit | `README_ASSISTANT.md` inside the kit / [`data/deliverables/assistant/README_ASSISTANT.md`](../data/deliverables/assistant/README_ASSISTANT.md) |
| Personal settings | none | one private file on the Pi, `~/.heydelta/config.json` (anything left empty stays simulated) |

Copy a kit to the Pi with its number-style address (`hostname -I` on the Pi):

```bash
scp <kit>.zip <user>@<pi-ip>:~
```

## Models in both kits

| Part | Model | File in the repository |
|---|---|---|
| Wakeword | BC-ResNet-6, threshold 0.7335, fires on 2 of 3 windows (100 ms apart) | `data/deliverables/model/deploy/heydelta_bcresnet6.onnx` + `deploy_config.json` |
| Commands (default) | BC-ResNet-6 on the class Hugging Face data + our data, cautious rule | `data/deliverables/command_classifier/model/export/bcresnet6_hf_plus/` |
| Commands (comparison) | the same model on the class data only | `.../export/bcresnet6_hf_only/` |

Each ONNX file holds the front end and the network: raw 16 kHz audio in, probabilities out. Licence of the
weights: CC BY-NC 4.0 (`MODEL_LICENSE.md`).

## Checking the deployment

- **Latency of every model on the Pi:** `python3 data/deliverables/evaluation/pi_latency.py` from a clone on the Pi
  (`numpy` + `onnxruntime` only); results in [`COMPUTE.md`](COMPUTE.md).
- **Class benchmark on the Pi** (the laptop plays the class holdout, the Pi answers):
  `python3 data/deliverables/command_classifier/model/make_bench_kit.py`, then
  `data/deliverables/command_classifier/model/bench_kit_src/README_BENCH.md`.
- **Assistant without a Pi or devices:** inside the assistant kit, `python3 test_assistant.py` (offline tests with a
  fake bulb and phone) and `python3 assistant.py --type` (type command names instead of speaking).

Reference clips for the kits' self-checks (`command_pi.py check`, `assistant.py --selftest`) are audio and so are
not in the repository: a kit built from a fresh clone skips those checks; everything else works.
