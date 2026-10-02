# Command classifier data (started 2026-09-30)

## Classes

30 classes: 17 plain commands, 12 number commands and `unknown`. They come from 33 recording prompts
(`../model/deploy/commands.txt`; `label_map.csv` maps prompt → class): "Skip" → `next`, "Louder" → `volume_up`, and
`unknown_silent` / `unknown_other` → `unknown`. Each number command has 3 fixed values: dim 20/50/80% (100% until 2026-09-30), timer
1/5/10 min, alarm 6 AM / 7 AM / 9 PM (7:30 AM until 2026-09-30; changed to match the owner's recordings), temperature 18/22/26°.

## Data sources

| Folder | What | Made by | Licence / terms |
|---|---|---|---|
| `recordings/` (on the Pi) | The owner's real captures through the wake-up path | `../model/deploy/demo_record.py` | own data |
| `data/commands_synthetic/` tier `ph` | Microsoft neural voices with a Philippine accent (en-PH James, Rosa; fil-PH Angelo, Blessica), 3 speeds × 3 pitches | `generate_synthetic_commands.py --tier ph` | generated with edge-tts (Microsoft online service) |
| `data/commands_synthetic/` tier `clone` | Up to 60 Filipino-accented voices cloned from FLEURS fil_ph speakers (picked for distinct voice timbre) | `clone_fleurs_voices.py` (Chatterbox, MIT; separate env `envs/chatterbox`), then `--tier clone` | FLEURS CC-BY 4.0 references; owner-approved use of the voices for training data. Chatterbox output carries an inaudible Perth watermark (every cloned clip, every class) |
| `data/commands_real_web/` | Real recordings from public datasets | `prepare_real_commands.py` | see below |
| `data/commands_unknown_reuse/` | `unknown` only, reused from the wakeword project: the owner's room recordings (`unknown_room`, 2–5 s windows), MUSAN noise and music, FLEURS Filipino/English + MUSAN speech (`unknown_speech`), MSWC single near-miss words (`unknown_word`). Speech windows with a command word or "hey"/"delta" are dropped (Whisper large-v3); `source_split` keeps the wakeword split of each source recording | `build_unknown_reuse.py` | own data; MUSAN CC-BY 4.0 / public domain; FLEURS and MSWC CC-BY 4.0 |

**Real public datasets** (`prepare_real_commands.py` has the mapping rules):

| Source | Speakers | Licence | Terms that matter |
|---|---:|---|---|
| Timers and Such v1.0, real part | 95 | CC0 | none |
| Fluent Speech Commands | 97 | Fluent Speech Commands Public License | **Non-commercial and academic use only. The audio and anything derived from it may NOT be shared or redistributed.** Keep it on this machine. |
| SLURP, real part | many | CC BY-NC 4.0 | non-commercial; credit the authors |

Raw downloads and licences are in `external_raw/commands/`. From Timers and Such, only the real-speech part was
fetched (`remote_zip.py` reads selected files of the 13 GB archive).

## Synthetic clips look like captures

A capture starts 0.3 s before the wakeword fires, which is ~0.1–0.3 s before "Delta" ends. So every synthetic clip is:
the last 0.4–0.6 s of "Hey Delta" in the same voice, a 0.3–1.0 s pause (waiting for the chime), the command, then
0.6–0.7 s of quiet. A −65 dBFS noise floor keeps any part from being exact digital silence. `unknown_silent` clips are
the "Delta" tail plus ~3 s of quiet.

**Checks.** Whisper `small`, then `large-v3` for what `small` rejects, must hear the command: normalised character
edit ratio ≤ 0.2.
- If the command is present with extra sounds around it, only the matching words are kept (`[cropped]` in the
  manifest).
- Short commands from the cloning model get 3 takes; the first confirmed take is used.
- Rejects are in `data/commands_synthetic/rejected.csv`. Many Philippine-accent rejects are plausibly correct accented
  speech: "Pause" heard as "boss", "Text Jane" as "Text Jean". They were dropped to keep labels safe.

Real web clips are trimmed to the speech but have no "Delta" tail. Aligning them with the capture format is left to
training.
