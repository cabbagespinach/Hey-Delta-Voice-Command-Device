# Potential enhancements (after the project is finished)

## 1. Near-miss recordings in the owner's "calling" tone (owner's live Pi test, 2026-09-30)

**Observation:** on the Pi, BC-ResNet-6 and 6_hn behave about the same.
- They don't fire on ordinary dialogue.
- They sometimes fire on deliberate "Hey + word" near-misses, but only when said in the melody the owner uses to call
  "Hey Delta". Said in a different tone, the same phrases don't fire.
- "Hey Delta" fires in any tone.

**Likely cause:** nearly every real positive is the owner calling "Hey Delta" in that tone, while real negatives with the
same voice + melody are rare (only the near-misses in Manual-BG-Macmic2 / Manual-BG-Phonemic1). The models treat
"owner's voice + calling melody + Hey + something" as strong evidence of the wakeword. That's a data gap, so a
bigger model or hard-negative mining can't fix it.

**Plan:** record 150–250 negative clips on the RPI (same mic, PulseAudio + AEC, same distance range):
- **Near-misses, in exactly the calling tone:** "Hey Delia", "Hey Della", "Hey Dell", "Hey Denta", "Hey Belta",
  "Hey Melda", "Hey Martha", "Hey Walter", "Hey Stella", "Hey Siri", "Hey Google", "Hey Alexa", and anything that
  fired during the live tests.
- **Half-wakewords in that tone:** "Hey…" alone, "Delta" alone.
- About 10–15 of each, across 2–3 sessions (different days or rooms), with small distance changes. Add other people's
  voices if possible.
- File them like the background recordings (e.g. `Manual-BG-RPI37.wav` onward) and integrate them as negatives. The
  usual steps apply: duplicate check, ASR check, owner confirms there is no "Hey Delta", then retrain.
- **Useful first:** live-test sessions saved with `--save-audio` and a `--note` listing the phrases said, to measure
  how close each near-miss comes to each model's cutoff.

## 2. A speech model instead of the loudness rule for end of speech (discussed 2026-09-30)

**Now:** the listener decides "speech" by loudness relative to the room: 6 dB above it to start, 4 dB to continue,
1.0 s pause allowance. These values were calibrated on 230 owner Pi recordings where Whisper confirmed the command.
The owner chose to keep this rule for the project.

**Known limits:**
- The start of about 1 command in 9 is still missed (25 of 230). The owner's voice is often only ~12 dB above the room.
- The rule can't tell speech from noise. TV or other people's speech keeps captures open, sometimes to the 6 s maximum
  (see the talk-show clips recorded 13:02–13:06 on 2026-09-30).

**Options, simplest first:**
- **Silero VAD v6** (MIT, 1.2 MB ONNX, already bundled with faster-whisper): a small CNN + LSTM that gives a speech
  probability per 32 ms. On the same 230 recordings it missed 17 starts instead of 25, with no command cut short, at
  about 0.15 ms per 32 ms chunk on one server core (not yet timed on the Pi). Only `_is_speech()` changes. It still
  counts TV speech as speech.
- **Speaker verification on the wake window** (Siri-style, text-dependent): enrol the owner's "Hey Delta", then accept
  a wake-up only if the 1.5 s window the listener already keeps matches the owner's voiceprint. Candidate models:
  ECAPA-TDNN (SpeechBrain), WeSpeaker ResNet34, CAM++. Test data already exists: ~400 owner Pi clips versus the 60
  cloned FLEURS voices and the talk-show clips. Short one-word commands are harder to verify.
- **Personal VAD** (speech from the owner vs. from anyone else, per chunk): the only option that would stop TV speech
  keeping a capture open. Mostly research work with no ready-made open model, so it would need training.
- **Endpointing from the ASR:** use the on-device ASR's partial transcript to judge whether the sentence is finished
  ("Set a timer for…" vs. "…five minutes"), as commercial assistants do.

## Other open items, noted earlier

- **Speech backgrounds in training:** a "Hey Delta" inside someone else's talking is mostly missed (22% of streaming
  wakewords caught). Add MUSAN / FLEURS speech and music (train split) as background-mixing sources, then retrain.
- **Real RPI microphone response:** record the same sentences on the RPI and a reference mic at the same time.
  Replace the interim random colouring with the measured response.
- **int8 model:** failed the quality gate (the SubSpectralNorm layers can't be folded). The Pi uses fp32; revisit
  only if CPU becomes tight.
- **`augmentation_strategy.docx`** is one version behind the markdown: `pip install python-docx`, then
  `python build_docx.py` in `deployment_driven_augmentation_strategy/`.
