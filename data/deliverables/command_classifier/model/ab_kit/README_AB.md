# Command models side by side on the Raspberry Pi 5

Every time you say "Hey Delta" and a command, the **same capture** goes to both models, and the kit shows and saves
both answers.

| Model | What it is | Cutoffs (cautious / balanced) |
|---|---|---|
| `bcresnet6` | the current command model | 0.885 / 0.345 |
| `bcresnet6_ownernoise` | retrained 2026-10-01 with extra room noise added to your recordings | 0.945 / 0.575 |

**Rules** (all three are saved for every capture; `--rule` only picks which one is printed live):
- `argmax`: always takes the most likely answer.
- `cautious`: answers "didn't catch that" unless very sure.
- `balanced`: in between (the default shown).

The kit includes the "Hey Delta" listener and wakeword model, so it doesn't need your deploy folder.

## 1. Set up (once)

Copy the whole `ab_kit` folder to the Pi, then in a terminal inside it:

```bash
sudo apt install libportaudio2                 # skip if already installed for the wakeword kit
python3 -m venv ab_test
source ab_test/bin/activate
pip install numpy onnxruntime sounddevice scipy
```

(You can reuse the wakeword kit's virtual environment instead: `source <path>/wakeword_test/bin/activate`.)

## 2. Quick check (under a minute)

```bash
python3 command_ab.py check        # both models should say OK (same scores as the server)
```

## 3. Live test

Use the **same microphone and settings as your recordings** (PulseAudio echo-cancel source; see the deploy README).
`python3 command_ab.py devices` lists microphones; choose one with `--device <number>`.

```bash
# the commands as trained (one round = 33 prompts, shown in random order)
python3 command_ab.py live --prompts commands.txt --note "living room, quiet"

# other ways of saying each command (122 prompts)
python3 command_ab.py live --prompts phrasings.txt --note "living room, quiet"
```

- It shows what to say, e.g. `>>> say: "Hey Delta, Turn on lights"`. Say it.
- After each capture you see both answers:
  ```
      ok  bcresnet6              lights_on            0.98  correct
      ?   bcresnet6_ownernoise   unknown              0.52  asks again   (argmax: lights_on)
  ```
  `ok` = correct, `?` = "didn't catch that" (asks again), `XX` = **wrong command** (the harmful mistake), or a
  non-command that made it do something.
- Type then Enter: **`d`** delete the last capture (you misspoke; it asks again), **`s`** skip the prompt,
  **`q`** quit.
- The `unknown_other` prompt means: say anything that is *not* a command. It checks that neither model acts on it.
- **Test in the conditions you care about.** Run it once in a quiet room, then with the TV, a fan or music on, and
  put that in `--note`. The new model was trained to handle room noise better.

## 4. Results

When you quit, it prints a report and saves everything in `ab_results/<date-time>/`:
- `report.txt`: per model and rule, the share of commands **correct / WRONG / asks again**, and how often a
  non-command caused an action; the captures where the two models disagreed; per phrasing; time per capture.
- `results.csv`: one row per capture with both models' answers.
- `<label>/*.wav`: the captures themselves.

Send Claude the `ab_results` folder (or just `report.txt` and `results.csv`) to compare the sessions.

**Re-score saved audio** (no microphone needed), e.g. after a model update or on your older recordings:

```bash
python3 command_ab.py replay ab_results/<date-time>
python3 command_ab.py replay <path>/recordings       # a demo_record.py folder (uses its manifest.csv)
```

Note: most of your older recordings were used to train both models, so they will look better than new captures.
New live captures are the fair test.
