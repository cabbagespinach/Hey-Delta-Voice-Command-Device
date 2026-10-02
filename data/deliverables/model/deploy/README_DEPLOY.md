# "Hey Delta" deployment module (BC-ResNet-6)

| File | What it is |
|---|---|
| `heydelta_bcresnet6.onnx` | The chosen model, front end included (1.5 s of 16 kHz audio in, probability out) |
| `deploy_config.json` | Its threshold (0.7335) and trigger rule (2 of the last 3 windows) |
| `heydelta_listener.py` | `HeyDeltaListener`: wakeword → command capture from the same mic stream → your callback |
| `demo_record.py` | Records a command dataset through the real wake-up path |

Setup, as for the Pi kit:

```bash
sudo apt install libportaudio2      # skip if `python3 -c "import sounddevice"` already works
python3 -m venv wakeword_test && source wakeword_test/bin/activate
pip install numpy onnxruntime sounddevice scipy
```

## Using it in your program

```python
from heydelta_listener import HeyDeltaListener

def on_command(cap):
    label = classify(cap.audio)            # your command classifier: 16 kHz mono float32 numpy array
                                           # (cap.reason == "no_speech_detected": probably "unknown")

HeyDeltaListener(on_command=on_command, device=<mic number>).run()
```

- **`on_wake(prob, time)`** (optional) is called the moment it fires, e.g. to play a beep or light an LED.
- **`on_command(cap)`** receives a `Capture` with these fields:
  - `audio`: the command, 16 kHz mono.
  - `reason`: `end_of_speech`, `max_length` or `no_speech_detected`. The last means nothing was heard within 3 s;
    the ~3.3 s of audio is still handed over (`always_capture=True`, the default). With `always_capture=False` you
    get `audio=None` and reason `no_speech` instead.
  - `wake_prob`, `wake_time`.
  - `wake_window`: the 1.5 s that fired.
- **Timings are settings:**
  `Settings(pre_roll_sec=0.3, max_wait_sec=3, end_silence_sec=1.0, max_command_sec=6, speech_margin_db=6, continue_margin_db=4, smooth_sec=0.1, ignore_after_wake_sec=0.25, min_speech_sec=0.15, cooldown_sec=1)`.
- **Supplying audio yourself:** if you already read the microphone elsewhere, call `listener.feed(audio16k)` with
  your chunks instead of `run()`.
- **Chimes (on by default).** A short rising tone plays when it wakes up (speak after it) and a falling tone when
  the command has been captured. They play through the default output. **Make that your PulseAudio echo-cancel
  sink** (or `PULSE_SINK=<echo-cancel sink name>` for one run; `pactl list short sinks` shows the names), so AEC
  removes the chime from the microphone. While the start chime plays, the capture ignores the microphone. Tested
  with the chime fed straight back into the microphone at 50% level (no AEC): it never counted as a command. Options:
  `Settings(chimes=False)`, `chime_volume` (default 0.25), `chime_output_device`. Record your dataset with the same
  chime setting the device will use.
- **Captured clips start with the end of "Delta".** The model usually fires 0.1–0.3 s before "Delta" ends, and the
  capture starts 0.3 s before that, so a command said without a pause is never clipped. Train the command classifier
  on clips captured this way (`demo_record.py` does exactly that), and it will see the same thing on the device.
- **Capture limits:** it waits up to 3 s for you to start speaking. The command then gets up to 6 s from its start.
- **End of speech (changed 2026-09-30):**
  - Pauses of up to **1.0 s** inside a sentence are allowed.
  - Speech must be 6 dB above the room to *start* a command, but only 4 dB to *continue*, so a dropping voice or
    soft sounds don't end it.
  - These values were calibrated on 230 owner recordings from the Pi (commands confirmed by Whisper). Compared
    with the old 10 dB / 7 dB bars, commands whose start was missed fell from 73 to 25, with none cut short.
  - Levels are averaged over 100 ms, so room clicks don't keep it open.
  - In a room with steady intermittent sounds (a ticking fan, a TV), captures can run on for a second or more after
    you finish. That's the limit of a loudness rule; a VAD model fixes it later.
  - Tune with `--end-silence-sec`, `--speech-margin-db` and `--continue-margin-db`.
- **End of speech has no VAD yet.** It is decided by loudness relative to the room (above). Replace `_is_speech()`
  when you have a VAD.
- **Microphone:** use the PulseAudio device (`python3 -m sounddevice` lists devices) with your echo-cancel source as
  input, so audio matches the training recordings. Either make it PulseAudio's default source, or set it for one run
  only: `PULSE_SOURCE=<echo-cancel source name> python3 demo_record.py --device <pulse number> ...`
  (`pactl list short sources` shows the names). Don't use a raw `hw:` device; it bypasses echo cancellation.

## Recording the command dataset

`commands.txt` is ready: 31 prompts, one per line as `label | words to say`. The labels are the 21 phrases, with 3
values for each number command, plus two kinds of `unknown`. `label_map.csv` maps each label to its class. There are
28 classes, because "Skip" → `next` and "Louder" → `volume_up`. Then:

```bash
python3 demo_record.py --prompts commands.txt --speaker me --note "RPI, AEC, living room" --save-wake
```

- **It shows exactly what to say,** e.g. `>>> say: "Hey Delta, Set a timer for 5 minutes"   [set_timer_5]`. Say it
  the way you normally would. For `unknown_silent` say nothing after "Hey Delta"; for `unknown_other` say anything
  that isn't one of the commands.
- **Each capture is saved as `recordings/<label>/<label>_<speaker>_<time>.wav`** and added to
  `recordings/manifest.csv`.
- **While recording:** type `d` + Enter to delete the last clip, `s` to skip a prompt, `q` to quit.
- **`--save-wake`** also keeps the "Hey Delta" part (`recordings/_wake/`). That's extra real wakeword data.
- **Chimes are on:** wait for the rising tone, then say the command; the falling tone means it's saved.
  `--no-chimes` turns them off, and `--chime-device <number>` picks the output.
- **`--order random`** shuffles the prompts each round, so the same command isn't always recorded in the same
  state (fresh vs. tired, and so on).
- **Every wake-up is saved,** even when no speech was detected within 3 s (`end_reason` `no_speech_detected` in the
  manifest; the demo flags it so you can press `d` to redo). `--max-wait-sec` changes the 3 s, and
  `--no-always-capture` restores discarding.
