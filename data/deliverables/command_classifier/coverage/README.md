# Phase 1: acoustic coverage analysis and augmentation design (command classifier)

This mirrors the wakeword project's `deployment_driven_augmentation_strategy/`. It covers the conditions the command
classifier will meet on the Raspberry Pi, what the data covers, the gaps, and how the training augmentation closes
them. The exact counts are generated from `data/commands_all.csv` by `coverage_analysis.py`
(`coverage_matrix.csv`, `coverage_tables.md`).

## 1. Deployment conditions

| Condition | On the device | Source of the fact |
|---|---|---|
| Input | One capture per wake-up: 0.3 s pre-roll (tail of "Delta"), chime gap, pause, command, then 1.0 s of quiet, or cut at 6 s of speech / 3 s without speech | `model/deploy/heydelta_listener.py` |
| Microphone and path | Pi 5 microphone through PulseAudio echo cancellation, 16 kHz | Owner's recordings |
| Levels | Speech peak about −37 dB (p10–p90 −46 to −30), background about −50 dB, speech 15 dB above background (p10–p90 6–24 dB) | 300 owner captures, `augment_commands.py` level check |
| Rooms | The owner's home rooms; sometimes a TV or video playing (talk show, vlog) | Owner recordings 13:02–13:06 and 16:43–17:27 |
| Speakers | The owner (higher voice) and one other person (`speaker2`, lower voice); Philippine English; Filipino/Taglish chatter nearby | `recordings/_check/speakers.csv` |
| Non-commands | Silence after a false wake-up, room noise, TV/music, other speech (English, Filipino), near-miss requests, commands cut off by the end-of-speech rule | Owner recordings and captures |

## 2. What the data covers

Six sources (see `../README.md` for licences):

| Source | Real? | Voices | Condition |
|---|---|---|---|
| Owner recordings | Yes | 2 (owner, speaker2) | The actual device, path and rooms |
| Public datasets (FSC, SLURP, Timers and Such) | Yes | ~1,700 | Other microphones and rooms; trimmed speech, no "Delta" tail |
| Reused unknowns (owner room, MUSAN, FLEURS, MSWC) | Yes | Many | Owner's room (Pi, Mac, phone), studio and web audio |
| Philippine-accent TTS (edge-tts) | No | 4 voices × 9 speed/pitch variants | Studio-clean, capture format |
| Cloned voices (Chatterbox, FLEURS Filipino speakers) | No | 150 | Studio-clean, capture format |
| Piper English voices | No | 150 | Studio-clean, capture format |
| Voice-converted owner recordings | No | 10 target voices | Owner's timing and pauses, cleaner than the device |
| Cut-off fragments (`unknown_partial`) | Derived | As the source clip | Cut before the deciding word |

## 3. Gaps and how each is handled

| Gap | Evidence | Mitigation | Remaining risk |
|---|---|---|---|
| **12 commands have no real speaker except the owner and speaker2** (call/text Jane, both reminders, dim 20/50/80, temperatures, timer 10, "Party party") | `coverage_matrix.csv`, `real_voices_train` | 150 cloned + 150 Piper + 36 TTS voices; voice conversion of the owner's multi-word commands (10 voices, strict Whisper check) | Synthetic voices are cleaner and more regular than people; measured by the small-scale test (`../model/runs/probes/`) and by speaker2 in the test |
| **Synthetic and public clips are far cleaner and louder than captures** (synthetic: 52 dB speech over background; captures: 15 dB) | Level check, `coverage_tables.md` | Training augmentation (section 4) | Real device colour is still unmeasured (a random colouring is used instead) |
| **Public clips and reused unknowns have no "Delta" tail**, while every capture starts with one | Dataset formats | Format alignment in the loader: a real prefix from the training bank is prepended to every such clip, in every split | None known |
| **The unknown class had no noise, music or Filipino speech** | Web unknowns were all English requests | Reused owner room recordings (861 windows), MUSAN noise/music/speech, FLEURS Filipino/English, MSWC single words | TV speech that sounds like a command |
| **Commands cut off by the end-of-speech rule** | 9 cut-off owner captures on 2026-09-30 | `unknown_partial` fragments cut before the deciding word | Cut-offs after the deciding word keep their label, by design |
| **Unknown is 60% of all clips** | Clip counts | The loader draws `unknown` for 20% of examples; command classes drawn uniformly | — |
| **One recording day** | All owner clips are 2026-09-30 | Session-based split; test on a separate session and on speaker2 | A new room or season is untested |

## 4. Augmentation design (training only)

The wakeword project's measured Pi augmentation (version 4) is reused through `../augment_commands.py`. The order is:
random microphone colouring → level set to the measured Pi speech range → room echo (MIT IR Survey, 132 rooms,
RT60 0.15–0.6 s) → background from the owner's own room recordings (training split, command words removed) at the
measured speech-over-background range → device noise floor.

- **The same for every class, with the same probabilities.** No condition (level, noise, echo, colour) can hint at
  the answer. This follows the lesson of the wakeword confound analysis.
- **The owner's recordings are augmented too, but never get room echo,** because they already have a real room.
  Background mixing skips them when they are already as noisy as the target.
- **Measured effect (median):** synthetic speech over background goes from 52 dB to 17 dB, public clips from 27 dB to
  16 dB, against 15 dB for the owner's captures. Levels land within the Pi range.
- **SpecAugment** on the features, with time masks of at most 150 ms. That is shorter than any deciding word, so a
  mask cannot turn one command into another.
- **Not used:** clipping (no measured capture is clipped) and a measured microphone response (needs a simultaneous
  Pi and reference recording; deferred, see `../../model/ENHANCEMENTS.md`).
