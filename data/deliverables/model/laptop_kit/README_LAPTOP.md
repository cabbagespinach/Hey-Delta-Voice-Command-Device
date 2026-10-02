# "Hey Delta" on your laptop: streaming test, all model sizes side by side

Works on Windows, macOS and Linux. The models already include the sound processing; PyTorch isn't needed.

| Model | Size | Notes |
|---|---:|---|
| `bcresnet1` | 9k parameters | smallest; clearly weaker on validation |
| `bcresnet3` | 53k | fallback if the Pi is too slow |
| `bcresnet6` | 186k | current best overall |
| `bcresnet6_hn` | 186k | same size, retrained with mined hard negatives: better on validation, worse on the streaming and "Hey + word" checks |

Each model has its own threshold, chosen on validation data (`models.json`).

## 1. Set up (once)

Install Python 3.9 or newer, open a terminal in this folder, then:

```bash
python -m venv venv
# Windows:        venv\Scripts\activate
# macOS / Linux:  source venv/bin/activate
pip install numpy onnxruntime sounddevice scipy
```

## 2. Test with your microphone

```bash
python heydelta_laptop.py devices                        # optional: list microphones
python heydelta_laptop.py live --minutes 10 --note "laptop mic, quiet room"
```

- **All four models listen to the same audio at once.** The bottom line shows the microphone level and each model's
  current score (0 to 1). When a model wakes, it prints `WAKE: <model>`.
- **Press Enter right after each "Hey Delta" you say.** At the end, the script counts for each model how many it
  caught, how many it missed, and how many wake-ups were false.
- **Things to try:** normal speech, TV or music in the background, near-miss phrases ("Hey Delia", "Hey Della",
  "Delta"), and different distances from the laptop.
- **Choosing a microphone:** use `--device <number>` (from `devices`).
- **Saving the audio:** add `--save-audio` to keep the recording and send it to me.

## 3. Test a recording

```bash
python heydelta_laptop.py wav myrecording.wav --marks 3.2,8.9,15.0
```

`--marks` are the times, in seconds, right after each "Hey Delta" in the file. Leave it out if the file has none; then
every wake-up counts as false.

## Important: a laptop isn't the Raspberry Pi

The models learned the loudness and sound of the RPI microphone, plus some phone and laptop recordings.
- **Your laptop's microphone is different.** Results here can be better or worse than on the Pi. The Pi test kit
  (`pi_bundle/`) is the real deployment check.
- **Check the level meter.** If speech from about 1 m away barely moves it, or constantly hits the top, adjust the
  input volume in your system's sound settings.

Results are saved in `results/`. Send that folder back if you want me to look at them.
